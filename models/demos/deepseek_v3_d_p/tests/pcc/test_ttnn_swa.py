# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""
PCC tests for the DeepSeek-V4 sliding-only attention layer (prefill), against the reference
modeling_deepseek_v4.py.

Every test drives a public forward -- no TtSWA private method is called directly:
  - TtSWA.forward, single-shot     4 prompt lengths
  - TtSWA.forward, chunked         TtSWAState across chunks, 4 scenarios + two 56K runs

Both V4 variants run, flash and pro. Pro has no sliding layer in the real checkpoint, but its
dimensions are what stress the reductions, so it is kept as a width test.

Prompt lengths are multiples of the 128-token window: the next chunk's first query needs the 128 keys
before it, so a chunk that ended mid-window would leave the carry unable to name them.

Device-perf for the same layer lives in tests/perf/test_ttnn_swa_perf.py.
"""

import pytest
import torch
from loguru import logger
from tracy import signpost

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.deepseek_v3_d_p.reference.deepseek_v4.configuration_deepseek_v4 import DeepseekV4Config
from models.demos.deepseek_v3_d_p.reference.deepseek_v4.modeling_deepseek_v4 import (
    DeepseekV4Attention,
    DeepseekV4HCACache,
    DeepseekV4RotaryEmbedding,
)
from models.demos.deepseek_v3_d_p.reference.deepseek_v4_flash_config import DeepSeekV4FlashConfig
from models.demos.deepseek_v3_d_p.reference.deepseek_v4_pro_config import DeepSeekV4ProConfig
from models.demos.deepseek_v3_d_p.tests.fabric_profiles import fabric2d_device_params, torus_xy_device_params
from models.demos.deepseek_v3_d_p.tt.mla.sliding_window_attention import TtSWA
from tests.ttnn.utils_for_testing import assert_with_pcc

_SEED = 42
# One shape per behaviour: 128 is the shortest legal prompt, 1024 the pad granularity (window * sp),
# 2048 needs no padding, 5120 is the chunk width.
_SHAPES = [128, 1024, 2048, 5120]
# One program must serve every chunk; a sliding layer has no cache write, so nothing may compile after
# the first chunk at all.
_WRITE_COMPILES_PER_CHUNK = 0


def _config(model_config, num_hidden_layers=4):
    """Reference config from one variant's dimension constants, with layer 0 forced to sliding.

    ``compress_ratios=[0, ...]`` is the legacy per-layer form the config folds into ``layer_types``;
    0 maps to "sliding_attention", which is what makes ``DeepseekV4Attention.compressor`` None.
    ``q_lora_rank`` / ``o_groups`` are explicit for the reason the HCA test gives: their defaults are
    Flash's values, so a Pro config that omitted them would silently build a hybrid."""
    m = model_config
    cfg = DeepseekV4Config(
        hidden_size=m.EMB_SIZE,
        head_dim=m.HEAD_DIM,
        num_attention_heads=m.NUM_ATTENTION_HEADS,
        q_lora_rank=m.Q_LORA_RANK,
        o_groups=m.O_GROUPS,
        num_hidden_layers=num_hidden_layers,
        compress_ratios=[0] * num_hidden_layers,
        compress_rates=dict(m.COMPRESS_RATES),
        compress_rope_theta=m.COMPRESS_ROPE_THETA,
        rms_norm_eps=m.RMS_NORM_EPS,
    )
    cfg._attn_implementation = "eager"  # V4 is eager-only: the sdpa interface silently drops the sinks
    return cfg


# (id, dimension constants, per-chunk floor, long-run floor, single-shot floor).
#
# All three floors are measured worst case minus 1e-3, not guessed. Unlike HCA, PCC does not decay with
# depth here: the only state crossing a chunk boundary is 128 rows, so nothing accumulates and the
# softmax never widens. Over 56K the worst chunk lands where the shortest scenario does -- flash
# 0.99913, pro 0.99900, every one of the 20 runs inside 3.4e-4 -- which is why the long floor is the
# chunked one and not looser.
_VARIANTS = [
    ("flash", DeepSeekV4FlashConfig, 0.998, 0.998, 0.998),
    ("pro", DeepSeekV4ProConfig, 0.998, 0.998, 0.998),
]
_MODEL_CONFIGS_CHUNKED = [pytest.param(cfg, chunked, id=name) for name, cfg, chunked, _, _ in _VARIANTS]
_MODEL_CONFIGS_LONG = [pytest.param(cfg, long, id=name) for name, cfg, _, long, _ in _VARIANTS]
_MODEL_CONFIGS_FORWARD = [pytest.param(cfg, fwd, id=name) for name, cfg, _, _, fwd in _VARIANTS]

# Blackhole runs a mesh config only when it uses every chip, so one shape per box class.
_MESH_CONFIGS = [
    pytest.param(
        (2, 2),
        fabric2d_device_params(),
        ttnn.Topology.Linear,
        marks=pytest.mark.requires_mesh_topology(mesh_shape=(2, 2), topology="mesh-2x2"),
        id="fabric2d-mesh-2x2",
    ),
    pytest.param(
        (4, 2),
        fabric2d_device_params(),
        ttnn.Topology.Linear,
        marks=pytest.mark.requires_mesh_topology(mesh_shape=(4, 2), topology="mesh-4x2"),
        id="fabric2d-mesh-4x2",
    ),
    pytest.param(
        (8, 4),
        torus_xy_device_params(),
        ttnn.Topology.Ring,
        marks=pytest.mark.requires_mesh_topology(mesh_shape=(8, 4), topology="mesh-8x4"),
        id="torus-xy-8x4",
    ),
]

_CHUNKED_SCENARIOS = [
    ("2chunk-ragged", 4096, [4096, 3072]),  # the other chunk width, and a short final chunk
    ("chunk5120-full", 5120, [5120, 5120]),  # what the perf gate measures
    ("chunk5120-ragged", 5120, [5120, 5120, 3072]),  # three chunks, last one short
    # Non-final chunks below chunk_size. Pins the place real_len and the padded slab width must not be
    # confused: the carry, which the next chunk's first chip reads.
    ("chunk5120-varying", 5120, [1024, 256, 5120]),
]

# The two 56,320-token scenarios are the same length by construction, so the mixed one differs only in
# how the tokens are split -- every chunk a multiple of the window, none wider than chunk_size.
_LONG_SCENARIOS = [
    pytest.param("11x5120", 5120, [5120] * 11, id="11x5120"),
    pytest.param(
        "mixed5120",
        5120,
        [1024, 256, 5120, 2048, 5120, 5120, 3072, 5120, 640, 5120, 5120, 4096, 5120, 5120, 4224],
        id="mixed5120",
    ),
]


def _build_reference(config):
    """A randomised sliding-only reference layer. Weights are perturbed the way the HCA test does, so a
    PCC pass cannot come from near-identity norms."""
    ref = DeepseekV4Attention(config, layer_idx=0).eval()
    assert ref.compressor is None, "layer_idx=0 must be a sliding_attention layer for TtSWA"
    with torch.no_grad():
        ref.q_a_norm.weight.uniform_(0.5, 1.5)
        ref.kv_norm.weight.uniform_(0.5, 1.5)
        ref.sinks.normal_(0.0, 1.0)
    # The layer itself carries none: HCA reads its compressor's, a sliding layer's comes from the model.
    ref.rotary_emb = DeepseekV4RotaryEmbedding(config)
    return ref


def _reference_forward(ref, config, hidden, total, batch):
    """One unchunked pass over the whole prompt: the ground truth every chunk is compared against.

    Deliberately not chunked, so the chunked path has to reproduce plain attention rather than agree
    with a reference that shares its assumptions."""
    sw = config.sliding_window
    position_ids = torch.arange(total).unsqueeze(0).expand(batch, -1)
    with torch.no_grad():
        cos, sin = ref.rotary_emb(hidden, position_ids=position_ids, layer_type="main")
        i = torch.arange(total).view(total, 1)
        j = torch.arange(total).view(1, total)
        attn_mask = torch.zeros(total, total).masked_fill(~((j <= i) & (i - j < sw)), float("-inf"))
        attn_mask = attn_mask.view(1, 1, total, total).expand(batch, 1, total, total)
        out_ref, _ = ref(hidden, {"main": (cos, sin)}, position_ids, attn_mask, past_key_values=None)
    return out_ref


class _RefCache:
    """Minimal ``past_key_values`` for driving the reference chunk by chunk: the attention calls
    ``.update(k, v, layer_idx)`` on the container.

    ``DeepseekV4HCACache`` is used for its update body, which is the plain shared-KV sliding-window
    one every V4 layer shares; its compressor dicts stay untouched on a sliding-only layer."""

    def __init__(self, layer):
        self.layers = [layer]

    def update(self, key_states, value_states, layer_idx, *args, **kwargs):
        return self.layers[layer_idx].update(key_states, value_states)


def _sliding_mask(q_pos, k_pos, sliding_window):
    i, j = q_pos.view(-1, 1), k_pos.view(1, -1)
    allowed = (j <= i) & (i - j < sliding_window)
    return torch.zeros(i.shape[0], j.shape[1]).masked_fill(~allowed, float("-inf"))


def _prepare_input(hidden, sp_factor, window):
    """Pad the prompt up to a whole number of windows per chip, the granularity alloc_state asserts."""
    align = window * sp_factor
    real = hidden.shape[1]
    padded = -(-real // align) * align
    if padded == real:
        return hidden, real
    out = torch.zeros(hidden.shape[0], padded, hidden.shape[2], dtype=hidden.dtype)
    out[:, :real] = hidden
    return out, real


def _to_device(chunk, mesh_device):
    return ttnn.from_torch(
        chunk.unsqueeze(1),
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ttnn.ShardTensor2dMesh(  # seq @ SP, hidden @ TP
            mesh_device, mesh_shape=tuple(mesh_device.shape), dims=(2, 3)
        ),
    )


def _from_device(out_tt, mesh_device):
    return ttnn.to_torch(
        out_tt,
        mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=(2, 3)),
    ).squeeze(1)


def _report_chunk_pccs(pccs, floor):
    """Log every chunk's PCC, then judge them together: the per-chunk numbers describe how error grows
    with depth, and a single failing chunk is more useful reported alongside its neighbours."""
    for it, kv_actual, valid, pcc in pccs:
        logger.debug(f"  chunk {it}: kv_actual={kv_actual} valid={valid} PCC={pcc}")
    worst = min(pccs, key=lambda row: row[3])
    logger.info(f"worst chunk: iter={worst[0]} kv_actual={worst[1]} PCC={worst[3]} (floor {floor})")
    assert worst[3] >= floor, f"chunk {worst[0]} (kv_actual={worst[1]}) PCC {worst[3]} below floor {floor}"


def _run_chunked(mesh_device, topology, chunk_size, iters_valid, model_config, floor, label):
    """Shared body for the chunked and the long scenarios: they differ only in the chunk list."""
    torch.manual_seed(_SEED)
    batch = 1
    config = _config(model_config)
    total = sum(iters_valid)

    ref = _build_reference(config)
    hidden = torch.randn(batch, total, config.hidden_size)
    out_ref = _reference_forward(ref, config, hidden, total, batch)

    tt_model = TtSWA.from_reference(mesh_device, ref, config, sp_axis=0, tp_axis=1, topology=topology)
    state = tt_model.alloc_state(total, chunk_tokens=chunk_size)
    logger.debug(f"mesh={tuple(mesh_device.shape)} {label} chunk_size={chunk_size} iters={iters_valid}")

    signpost("SWA_START")
    kv_actual = 0
    pccs = []
    programs = mesh_device.num_program_cache_entries()
    for it, valid in enumerate(iters_valid):
        # Fixed device width every chunk; a short final chunk is padded up to it.
        chunk = torch.zeros(batch, chunk_size, config.hidden_size)
        chunk[:, :valid] = hidden[:, kv_actual : kv_actual + valid]

        out_tt = tt_model(_to_device(chunk, mesh_device), seq_len_actual=valid, state=state)
        out = _from_device(out_tt, mesh_device)[:, :valid]

        expected = out_ref[:, kv_actual : kv_actual + valid]
        _, pcc = comp_pcc(expected.to(torch.float32), out.to(torch.float32))
        pccs.append((it, kv_actual, valid, pcc))

        # Every chunk drives identical device shapes, so one compiled program must serve them all. A new
        # entry means some op's attributes moved with the chunk -- exactly what breaks trace later.
        now = mesh_device.num_program_cache_entries()
        logger.debug(f"  iter {it}: program cache {programs} -> {now} (+{now - programs})")
        if it > 0:
            assert now - programs == _WRITE_COMPILES_PER_CHUNK, (
                f"chunk {it} compiled {now - programs} new program(s) ({programs} -> {now}), expected "
                f"{_WRITE_COMPILES_PER_CHUNK}; an op attribute or a shape changed between chunks"
            )
        programs = now
        kv_actual += valid
    signpost("SWA_END")

    _report_chunk_pccs(pccs, floor)
    assert state.kv_actual == total


@pytest.mark.parametrize("seq_len", _SHAPES, ids=[f"seq{s}" for s in _SHAPES])
@pytest.mark.parametrize(
    "mesh_device, device_params, topology",
    _MESH_CONFIGS,
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.parametrize("model_config, forward_pcc", _MODEL_CONFIGS_FORWARD)
def test_swa_forward_mesh(mesh_device, device_params, topology, seq_len, model_config, forward_pcc):
    """Single-shot TtSWA.forward, SP+TP sharded. Proves the halo and the first-chunk mask variant: chip
    0's history does not exist yet, and the 128 zero rows standing in for it must not reach the
    softmax."""
    torch.manual_seed(_SEED)

    batch = 1
    config = _config(model_config)
    sp_factor = mesh_device.shape[0]

    ref = _build_reference(config)
    hidden = torch.randn(batch, seq_len, config.hidden_size)
    out_ref = _reference_forward(ref, config, hidden, seq_len, batch)

    tt_model = TtSWA.from_reference(mesh_device, ref, config, sp_axis=0, tp_axis=1, topology=topology)
    hidden_padded, seq_len_actual = _prepare_input(hidden, sp_factor, config.sliding_window)
    logger.debug(f"mesh={tuple(mesh_device.shape)} S_real={seq_len_actual} S_pad={hidden_padded.shape[1]}")

    state = tt_model.alloc_state(hidden_padded.shape[1])  # a one-chunk prefill still owns its state
    signpost("SWA_START")
    out_tt = tt_model(_to_device(hidden_padded, mesh_device), seq_len_actual=seq_len_actual, state=state)
    signpost("SWA_END")
    out = _from_device(out_tt, mesh_device)[:, :seq_len_actual]

    assert out.shape == out_ref.shape, f"shape mismatch: tt {tuple(out.shape)} vs ref {tuple(out_ref.shape)}"
    passed, message = assert_with_pcc(out_ref.to(torch.float32), out.to(torch.float32), pcc=forward_pcc)
    logger.debug(f"mesh SWA layer PCC: {message}")
    assert passed, f"SWA mesh layer PCC test failed: {message}"


@pytest.mark.parametrize("name, chunk_size, iters_valid", _CHUNKED_SCENARIOS, ids=[n for n, _, _ in _CHUNKED_SCENARIOS])
@pytest.mark.parametrize(
    "mesh_device, device_params, topology",
    _MESH_CONFIGS,
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.parametrize("model_config, chunked_pcc", _MODEL_CONFIGS_CHUNKED)
def test_swa_chunked_prefill_mesh(
    mesh_device, device_params, topology, name, chunk_size, iters_valid, model_config, chunked_pcc
):
    """Chunked prefill with TtSWAState carried across chunks."""
    _run_chunked(mesh_device, topology, chunk_size, iters_valid, model_config, chunked_pcc, f"scenario={name}")


@pytest.mark.parametrize("name, chunk_size, iters_valid", _LONG_SCENARIOS)
@pytest.mark.parametrize(
    "mesh_device, device_params, topology",
    _MESH_CONFIGS,
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.parametrize("model_config, long_pcc", _MODEL_CONFIGS_LONG)
@pytest.mark.timeout(0)
def test_swa_long_prefill_mesh(
    mesh_device, device_params, topology, name, chunk_size, iters_valid, model_config, long_pcc
):
    """The demo context, 56,320 tokens, split two ways. This is where the carry crosses fifteen chunk
    boundaries and where a ragged non-final chunk actually appears.

    The reference is chunked here, unlike the short scenarios. An unchunked one is not an option at
    this length: eager attention (forced, since sinks make V4 eager-only) materializes an [S, S] mask,
    12.7 GB at 56K, and the scores on top of it. The short scenarios are what pin chunked == plain
    attention; this test only shows it stays true that many chunks deep."""
    torch.manual_seed(_SEED)

    batch = 1
    config = _config(model_config)
    sw = config.sliding_window
    total = sum(iters_valid)

    ref = _build_reference(config)
    hidden = torch.randn(batch, total, config.hidden_size)
    position_ids = torch.arange(total).unsqueeze(0).expand(batch, -1)

    tt_model = TtSWA.from_reference(mesh_device, ref, config, sp_axis=0, tp_axis=1, topology=topology)
    state = tt_model.alloc_state(total, chunk_tokens=chunk_size)
    ref_cache = _RefCache(DeepseekV4HCACache(config))
    logger.debug(f"mesh={tuple(mesh_device.shape)} long={name} chunks={len(iters_valid)} total={total}")

    signpost("SWA_START")
    kv_actual = 0
    pccs = []
    programs = mesh_device.num_program_cache_entries()
    for it, valid in enumerate(iters_valid):
        real = hidden[:, kv_actual : kv_actual + valid]
        chunk_pos = position_ids[:, kv_actual : kv_actual + valid]

        # The reference cache returns [carry | chunk] from update(), so the mask must cover exactly
        # those keys -- sw - 1 of history, which is what its slice keeps.
        carry = min(sw - 1, kv_actual)
        k_pos = torch.cat([torch.arange(kv_actual - carry, kv_actual), chunk_pos[0]])
        ref_mask = _sliding_mask(chunk_pos[0], k_pos, sw).view(1, 1, valid, -1).expand(batch, 1, -1, -1)
        with torch.no_grad():
            cos, sin = ref.rotary_emb(real, position_ids=chunk_pos, layer_type="main")
            expected, _ = ref(real, {"main": (cos, sin)}, chunk_pos, ref_mask, past_key_values=ref_cache)

        # Fixed device width every chunk; a short chunk is padded up to it.
        chunk = torch.zeros(batch, chunk_size, config.hidden_size)
        chunk[:, :valid] = real
        out_tt = tt_model(_to_device(chunk, mesh_device), seq_len_actual=valid, state=state)
        out = _from_device(out_tt, mesh_device)[:, :valid]

        _, pcc = comp_pcc(expected.to(torch.float32), out.to(torch.float32))
        pccs.append((it, kv_actual, valid, pcc))

        now = mesh_device.num_program_cache_entries()
        logger.debug(f"  iter {it}: program cache {programs} -> {now} (+{now - programs})")
        if it > 0:
            assert now - programs == _WRITE_COMPILES_PER_CHUNK, (
                f"chunk {it} compiled {now - programs} new program(s) ({programs} -> {now}), expected "
                f"{_WRITE_COMPILES_PER_CHUNK}; an op attribute or a shape changed between chunks"
            )
        programs = now
        kv_actual += valid
    signpost("SWA_END")

    _report_chunk_pccs(pccs, long_pcc)
    assert state.kv_actual == total
