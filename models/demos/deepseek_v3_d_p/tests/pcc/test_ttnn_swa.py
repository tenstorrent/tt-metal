# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""PCC tests for the DeepSeek-V4 sliding-only attention layer (prefill), against the reference.

A non-final chunk must end on a tile row; a final one may end anywhere. Device-perf lives in
tests/perf/test_ttnn_swa_perf.py.
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
from models.demos.deepseek_v3_d_p.reference.deepseek_v4_flash_config import (
    DeepSeekV4FlashConfig,
    deepseek_v4_flash_hf_config,
)
from models.demos.deepseek_v3_d_p.tests.fabric_profiles import fabric2d_device_params, torus_xy_device_params
from models.demos.deepseek_v3_d_p.tt.mla.sliding_window_attention import TtSWA
from models.demos.deepseek_v3_d_p.tt.moe.init_helpers import create_fabric_router_config, get_max_payload_size
from models.demos.deepseek_v3_d_p.tt.v4.attention.swa import TtSWA as V4FlashTtSWA
from tests.ttnn.utils_for_testing import assert_with_pcc

_SEED = 42
# Shortest legal prompt, the pad granularity, one needing no padding, the chunk width, and two lengths
# that end mid-tile: a single-shot prompt is a final chunk, which may end anywhere.
_SHAPES = [128, 130, 1024, 2048, 4095, 5120]
# A sliding layer has no cache write, so nothing may compile after the first chunk at all.
_WRITE_COMPILES_PER_CHUNK = 0


def _ci_unsupported_param_combos_swa(**params):
    """The Blackhole e2e pipeline collects this whole folder, so every leg it should not run has to
    opt out here rather than be filtered by name in the pipeline's -k expression."""
    return params["is_ci_env"] or params["is_ci_v2_env"]


def _ci_unsupported_param_combos_swa_long(**params):
    if not (params["is_ci_env"] or params["is_ci_v2_env"]):
        return False
    return params["name"] != "mixed5120"


def _config(model_config, num_hidden_layers=4):
    """Reference config from one variant's dimension constants, with layer 0 forced to sliding.

    ``compress_ratios=[0, ...]`` folds into ``layer_types``, and 0 means "sliding_attention".
    ``q_lora_rank`` / ``o_groups`` are explicit because their defaults are Flash's."""
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


# (id, dimension constants, per-chunk floor, long-run floor, single-shot floor). All three are equal
# because the only state crossing a chunk boundary is one window, so error does not accumulate with
# depth and the softmax never widens.
_VARIANTS = [
    ("flash", DeepSeekV4FlashConfig, 0.998, 0.998, 0.998),
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
    ("2chunk-ragged", 4096, [4096, 3000]),  # the other chunk width, and a final chunk ending mid-tile
    ("chunk5120-full", 5120, [5120, 5120]),  # what the perf gate measures
    ("chunk5120-ragged", 5120, [5120, 5120, 3000]),  # three chunks, last one ragged
    # Non-final chunks below chunk_size: real_len and the padded slab width must not be confused.
    ("chunk5120-varying", 5120, [1024, 256, 5120]),
    ("chunk1024-ragged", 1024, [1024, 1024, 1000]),  # a final chunk 8 tokens past a tile row
    ("chunk1280", 1280, [1280, 1280, 1000]),  # whole tiles per chip, but not a whole number of windows
    # Non-final chunks below the window, which the carry can serve only by reaching back into itself.
    ("sub-window", 5120, [1024, 32, 96, 128, 5120, 1000]),
]

# Same length by construction, so the mixed one differs only in how the tokens are split.
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
    """A randomised sliding-only reference layer, so a PCC pass cannot come from near-identity norms."""
    ref = DeepseekV4Attention(config, layer_idx=0).eval()
    assert ref.compressor is None, "layer_idx=0 must be a sliding_attention layer for TtSWA"
    with torch.no_grad():
        ref.q_a_norm.weight.uniform_(0.5, 1.5)
        ref.kv_norm.weight.uniform_(0.5, 1.5)
        ref.sinks.normal_(0.0, 1.0)
    # A sliding layer carries no rotary_emb of its own; the config determines it.
    ref.rotary_emb = DeepseekV4RotaryEmbedding(config)
    return ref


def _reference_forward(ref, config, hidden, total, batch):
    """One unchunked pass over the whole prompt: plain attention, which the chunked path must reproduce."""
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
    """Minimal ``past_key_values`` for driving the reference chunk by chunk. ``DeepseekV4HCACache``
    supplies the shared-KV sliding-window update body every V4 layer uses."""

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
    """Log every chunk's PCC, then judge them together: a failing chunk reads better beside its neighbours."""
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

        # Identical device shapes every chunk, so a new entry means an op attribute moved with the chunk.
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


@pytest.mark.uncollect_if(pred=_ci_unsupported_param_combos_swa)
@pytest.mark.parametrize("seq_len", _SHAPES, ids=[f"seq{s}" for s in _SHAPES])
@pytest.mark.parametrize(
    "mesh_device, device_params, topology",
    _MESH_CONFIGS,
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.parametrize("model_config, forward_pcc", _MODEL_CONFIGS_FORWARD)
def test_swa_forward_mesh(mesh_device, device_params, topology, seq_len, model_config, forward_pcc):
    """Single-shot TtSWA.forward, SP+TP sharded. Proves the halo and the first-chunk mask variant:
    chip 0 has no history yet, and the zero rows standing in for it must not reach the softmax."""
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


@pytest.mark.uncollect_if(pred=_ci_unsupported_param_combos_swa)
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


@pytest.mark.uncollect_if(pred=_ci_unsupported_param_combos_swa_long)
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
    """The demo context, split two ways.

    The reference runs chunked: an unchunked one would materialize an [S, S] mask, which does not fit
    at this length. What this test shows is that the carry stays correct that many chunks deep."""
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

        # The reference cache returns [carry | chunk] from update(), so the mask covers sw - 1 of
        # history plus the chunk.
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


# =====================================================================================================================
# DeepSeek-V4-Flash production SWA (tt/v4/attention/swa.py, a compressor-less TtHCA) -- from the prefill branch.
#
# TtSWA (V4-Flash sliding-window attention, layers 0-1) vs the reference DeepseekV4Attention with random weights.
#
# Mirrors tests/pcc/test_ttnn_hca.py: single-shot for arbitrary prompt lengths (padding + trim + mask), and chunked
# prefill with the window carry across chunks against an UNCHUNKED reference. Floors: 0.998 single-shot, 0.997 per
# chunk (the HCA floors; SWA has no compressor, so it should meet them). Sub-galaxy configs run via
# TT_VISIBLE_DEVICES (+ TT_MESH_GRAPH_DESC_PATH for [2,4]) and carry no requires_mesh_topology mark.
#
# Its module-level helpers carry a _v4f / _V4F prefix and its classes a V4Flash alias so they cannot shadow the
# tt/mla tests' helpers above (tests/perf/test_ttnn_swa_perf.py imports _MESH_CONFIGS, _SEED, _build_reference and
# _config from this module, and CI selects the *_mesh tests by node id). Test function names are unchanged.
# =====================================================================================================================


_V4F_SEED = 42
_V4F_SINGLE_PCC = 0.998
_V4F_CHUNK_PCC = 0.997
_V4F_SHAPES = [128, 130, 1024, 4095]
# (name, chunk width, real lengths): non-final chunks >= 128 and tile-aligned; the last may be ragged.
_V4F_CHUNKED = [
    ("chunk1024-ragged", 1024, [1024, 1024, 300]),
    ("chunk5120-varying", 5120, [1024, 256, 5120]),
    ("chunk5120-full", 5120, [5120, 5120]),
]

_V4F_MESH_CONFIGS = [
    pytest.param((1, 1), {}, 0, 1, id="single-1x1"),
    pytest.param((2, 1), {"fabric_config": ttnn.FabricConfig.FABRIC_1D}, 0, 1, id="sp2-2x1"),  # SP 2, TP 1
    pytest.param(
        (2, 4),
        {
            "fabric_config": ttnn.FabricConfig.FABRIC_2D,
            "fabric_router_config": create_fabric_router_config(max_payload_size=get_max_payload_size()),
            "reliability_mode": ttnn.FabricReliabilityMode.RELAXED_INIT,
        },
        0,
        1,
        id="fabric2d-mesh-2x4",
    ),
    pytest.param(
        (8, 4),
        {
            "fabric_config": ttnn.FabricConfig.FABRIC_2D,
            "fabric_router_config": create_fabric_router_config(max_payload_size=get_max_payload_size()),
            "reliability_mode": ttnn.FabricReliabilityMode.RELAXED_INIT,
        },
        0,
        1,
        marks=pytest.mark.requires_mesh_topology(mesh_shape=(8, 4), topology="mesh-8x4"),
        id="fabric2d-mesh-8x4",
    ),
]


def _v4f_ref_layer(cfg):
    ref = DeepseekV4Attention(cfg, layer_idx=0).eval()
    assert ref.compressor is None, "layer 0 of V4-Flash is a sliding_attention layer"
    with torch.no_grad():
        ref.q_a_norm.weight.uniform_(0.5, 1.5)
        ref.kv_norm.weight.uniform_(0.5, 1.5)
        ref.sinks.normal_(0.0, 1.0)
    return ref


def _v4f_ref_forward(ref, rot, hidden, sw):
    total = hidden.shape[1]
    position_ids = torch.arange(total).unsqueeze(0)
    with torch.no_grad():
        cos, sin = rot(hidden, position_ids=position_ids, layer_type="main")
        i = torch.arange(total).view(total, 1)
        j = torch.arange(total).view(1, total)
        mask = torch.zeros(total, total).masked_fill(~((j <= i) & (i - j < sw)), float("-inf")).view(1, 1, total, total)
        out, _ = ref(hidden, {"main": (cos, sin)}, position_ids, mask, past_key_values=None)
    return out


def _v4f_dims(mesh_device, sp_axis, tp_axis):
    dims = [None, None]
    if mesh_device.shape[sp_axis] > 1:
        dims[sp_axis] = 2
    if mesh_device.shape[tp_axis] > 1:
        dims[tp_axis] = 3
    return dims


def _v4f_to_device(mesh_device, x, sp_axis, tp_axis):
    dims = _v4f_dims(mesh_device, sp_axis, tp_axis)
    mapper = (
        ttnn.ReplicateTensorToMesh(mesh_device)
        if dims == [None, None]
        else ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=dims)
    )
    return ttnn.from_torch(x, device=mesh_device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, mesh_mapper=mapper)


def _v4f_to_host(mesh_device, t, sp_axis, tp_axis):
    dims = [0, 0]
    dims[sp_axis], dims[tp_axis] = 2, 3
    return ttnn.to_torch(
        t, mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=tuple(dims))
    )


@pytest.mark.parametrize("seq_len", _V4F_SHAPES, ids=[f"seq{s}" for s in _V4F_SHAPES])
@pytest.mark.parametrize(
    "mesh_device, device_params, sp_axis, tp_axis", _V4F_MESH_CONFIGS, indirect=["mesh_device", "device_params"]
)
def test_swa_forward(mesh_device, device_params, sp_axis, tp_axis, seq_len):
    torch.manual_seed(_V4F_SEED)
    cfg = deepseek_v4_flash_hf_config(num_hidden_layers=4)
    ref, rot = _v4f_ref_layer(cfg), DeepseekV4RotaryEmbedding(cfg)
    sp = mesh_device.shape[sp_axis]
    hidden = torch.randn(1, seq_len, cfg.hidden_size)
    out_ref = _v4f_ref_forward(ref, rot, hidden, cfg.sliding_window)

    tt = V4FlashTtSWA.from_reference(mesh_device, ref, cfg, rotary_emb=rot, sp_axis=sp_axis, tp_axis=tp_axis)
    padded, real = V4FlashTtSWA.prepare_input(hidden, sp)
    state = tt.alloc_state(padded.shape[1])
    out = tt(_v4f_to_device(mesh_device, padded.unsqueeze(1), sp_axis, tp_axis), seq_len_actual=real, state=state)
    out = _v4f_to_host(mesh_device, out, sp_axis, tp_axis).squeeze(1)[:, :real]
    assert out.shape == out_ref.shape
    ok, msg = assert_with_pcc(out_ref.float(), out.float(), pcc=_V4F_SINGLE_PCC)
    logger.info(f"SWA single-shot seq {seq_len} mesh {tuple(mesh_device.shape)}: {msg}")
    assert ok, msg


@pytest.mark.parametrize("name, chunk_size, iters_valid", _V4F_CHUNKED, ids=[n for n, _, _ in _V4F_CHUNKED])
@pytest.mark.parametrize(
    "mesh_device, device_params, sp_axis, tp_axis", _V4F_MESH_CONFIGS, indirect=["mesh_device", "device_params"]
)
def test_swa_chunked_prefill(mesh_device, device_params, sp_axis, tp_axis, name, chunk_size, iters_valid):
    """The reference is NOT chunked: one pass over the whole prompt, each chunk compared to its slice, so the
    carry has to reproduce plain sliding-window attention."""
    torch.manual_seed(_V4F_SEED)
    cfg = deepseek_v4_flash_hf_config(num_hidden_layers=4)
    ref, rot = _v4f_ref_layer(cfg), DeepseekV4RotaryEmbedding(cfg)
    total = sum(iters_valid)
    hidden = torch.randn(1, total, cfg.hidden_size)
    out_ref = _v4f_ref_forward(ref, rot, hidden, cfg.sliding_window)

    tt = V4FlashTtSWA.from_reference(mesh_device, ref, cfg, rotary_emb=rot, sp_axis=sp_axis, tp_axis=tp_axis)
    state = tt.alloc_state(total, chunk_tokens=chunk_size)
    kv_actual, pccs = 0, []
    programs = mesh_device.num_program_cache_entries()
    for it, valid in enumerate(iters_valid):
        chunk = torch.zeros(1, chunk_size, cfg.hidden_size)
        chunk[:, :valid] = hidden[:, kv_actual : kv_actual + valid]
        out = tt(_v4f_to_device(mesh_device, chunk.unsqueeze(1), sp_axis, tp_axis), seq_len_actual=valid, state=state)
        out = _v4f_to_host(mesh_device, out, sp_axis, tp_axis).squeeze(1)[:, :valid]
        _, pcc = comp_pcc(out_ref[:, kv_actual : kv_actual + valid].float(), out.float())
        pccs.append((it, kv_actual, valid, pcc))
        now = mesh_device.num_program_cache_entries()
        logger.info(
            f"  {name} iter {it} (kv_actual={kv_actual} valid={valid}): PCC {pcc:.6f}, programs +{now - programs}"
        )
        if it > 0:
            assert now == programs, f"chunk {it} compiled {now - programs} new program(s); a shape or attribute moved"
        programs = now
        kv_actual += valid
    assert state.kv_actual == total and state.compressed_kv is None
    worst = min(pccs, key=lambda r: r[3])
    assert worst[3] >= _V4F_CHUNK_PCC, f"worst chunk PCC {worst[3]:.6f} (iter {worst[0]}) < {_V4F_CHUNK_PCC}"
