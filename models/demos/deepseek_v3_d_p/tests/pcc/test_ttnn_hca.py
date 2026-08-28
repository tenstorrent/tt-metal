# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""
PCC tests for the DeepSeek-V4 Heavily Compressed Attention (HCA) block (prefill), against the
reference modeling_deepseek_v4.py.

Every test drives a public forward -- no TtHCA private method is called directly:
  - TtHCACompressor.forward        compressed KV entries + block bias
  - TtHCA.forward, single-shot     5 prompt lengths
  - TtHCA.forward, chunked         TtHCAState across chunks, 4 scenarios + two 56K runs

Each of them runs on both V4 variants, flash and pro.

Device-perf for the same block lives in tests/perf/test_ttnn_hca_perf.py.
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
    DeepseekV4HCACompressor,
)
from models.demos.deepseek_v3_d_p.reference.deepseek_v4_flash_config import DeepSeekV4FlashConfig
from models.demos.deepseek_v3_d_p.reference.deepseek_v4_pro_config import DeepSeekV4ProConfig
from models.demos.deepseek_v3_d_p.tests.fabric_profiles import fabric2d_device_params, torus_xy_device_params
from models.demos.deepseek_v3_d_p.tt.mla.heavily_compressed_attention import TtHCA, TtHCACompressor
from tests.ttnn.utils_for_testing import assert_with_pcc

# Real (pre-pad) prompt lengths. prepare_input pads each up to compress_rate*sp, so the ragged ones
# exercise the pad + trim + mask path.
_SEED = 42
# One shape per behaviour, not one per number: everything under 1024 pads to the same slab. 128 is the
# shortest legal prompt, 130 and 4095 are the padding cases, 1024 needs none, 5120 is the chunk width.
_SHAPES = [128, 130, 1024, 4095, 5120]
_COMPRESSOR_SHAPES = [900, 2048, 5120]
# Single-shot floors are higher than the chunked ones: one pass accumulates nothing. The block's is per
# variant and lives in _VARIANTS; the compressor's holds for both.
_COMPRESSOR_PCC = 0.999
# The stored entries, checked at the end of a chunked run. They are written once and never recomputed, so
# this holds ~0.9997 no matter how deep the run goes.
_CACHE_PCC = 0.998


def _config(model_config, num_hidden_layers=4):
    """Reference config from one variant's dimension constants.

    ``q_lora_rank`` and ``o_groups`` are passed explicitly because DeepseekV4Config's defaults happen to
    be Flash's values -- a Pro config that left them out would build Pro widths with Flash's latent and
    grouping, and the reference would agree with it, so PCC would pass on the wrong model."""
    m = model_config
    cfg = DeepseekV4Config(
        hidden_size=m.EMB_SIZE,
        head_dim=m.HEAD_DIM,
        num_attention_heads=m.NUM_ATTENTION_HEADS,
        q_lora_rank=m.Q_LORA_RANK,
        o_groups=m.O_GROUPS,
        num_hidden_layers=num_hidden_layers,
        compress_rates=dict(m.COMPRESS_RATES),
        compress_rope_theta=m.COMPRESS_ROPE_THETA,
        rms_norm_eps=m.RMS_NORM_EPS,
    )
    cfg._attn_implementation = "eager"  # V4 is eager-only: the sdpa interface silently drops the sinks
    return cfg


# (id, dimension constants, per-chunk floor, long-run floor, single-shot floor).
#
# PCC decays with depth on both variants: every chunk inherits the previous one's error through the cache
# and the carry, and the softmax widens as the cache fills. Pro decays ~2.6x faster, since 128 heads and a
# 7168-wide hidden make every bf16 reduction longer. Over 56K flash goes 0.9989 -> 0.9925 and pro
# 0.9984 -> 0.9816, the per-chunk drop halving on both while the cache entries hold 0.9997. The two long
# scenarios split the same 56K differently and land within 1.2e-4, so the floors track depth, not chunk
# boundaries.
#
# TODO: move to a V4 prefill adapter once V4 has a block and a runtime, so build_runtime and
# allocate_kv_cache become writable. The adapter would own the config, these floors and the golden-trace
# dir, and serve both the `variant` fixture and whatever runtime is built.
_VARIANTS = [
    ("flash", DeepSeekV4FlashConfig, 0.997, 0.99, 0.998),
    ("pro", DeepSeekV4ProConfig, 0.994, 0.98, 0.997),
]
_MODEL_CONFIGS = [pytest.param(cfg, id=name) for name, cfg, *_ in _VARIANTS]
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


def _report_chunk_pccs(pccs, floor):
    """Log every chunk's PCC, then let the worst one decide. Asserting inside the loop stops at the first
    chunk under the floor, and PCC can dip and recover -- reporting first means one run tells the whole
    story instead of one chunk per run."""
    for it, kv_actual, valid, pcc in pccs:
        log = logger.warning if pcc < floor else logger.info
        log(f"  iter {it} (kv_actual={kv_actual} valid={valid}): PCC {pcc:.6f}")
    worst_it, _, _, worst = min(pccs, key=lambda row: row[3])
    assert worst >= floor, f"worst chunk PCC {worst:.6f} (iter {worst_it}) is below the floor {floor}"


# The compressed-cache write must compile nothing per chunk: fill_cache keeps update_idx out of its
# program hash, and the shift carries its offset as matrix data.
_WRITE_COMPILES_PER_CHUNK = 0
# (chunk_size, real lengths). chunk_size only has to carry whole compression windows on every SP shard, so
# 5120 is as legal as 4096. It is what the prefill runtime defaults to, and the width where the append
# offset does not land tile-aligned, which is why it belongs here.
_CHUNKED_SCENARIOS = [
    ("2chunk-ragged", 4096, [4096, 3000]),  # the other chunk width, where the append offset is tile-aligned
    ("chunk5120-full", 5120, [5120, 5120]),  # what the perf gate measures
    ("chunk5120-ragged", 5120, [5120, 5120, 3000]),  # three appends, last one ragged
    # Non-final chunks below chunk_size. Pins the two places real_len and the padded slab width must not
    # be confused: the carry, and the tail the compressed-cache write carries across chunks.
    ("chunk5120-varying", 5120, [1024, 256, 5120]),
]


class _RefHCACache:
    """Minimal ``past_key_values`` for driving the reference chunk by chunk: the attention calls
    ``.update(k, v, layer_idx)`` on the container, the compressor reaches into ``.layers[layer_idx]``."""

    def __init__(self, layer):
        self.layers = [layer]

    def update(self, key_states, value_states, layer_idx, *args, **kwargs):
        return self.layers[layer_idx].update(key_states, value_states)


def _sliding_mask(q_pos, k_pos, sliding_window):
    i, j = q_pos.view(-1, 1), k_pos.view(1, -1)
    allowed = (j <= i) & (i - j < sliding_window)
    return torch.zeros(i.shape[0], j.shape[1]).masked_fill(~allowed, float("-inf"))


# Demo-sized prompts. The two 56,320-token scenarios are the same length by construction
# (11 * 5120 = 55 * 1024), so the mixed one differs from the uniform one only in how the tokens are
# split -- every non-final chunk a multiple of compress_rate, none wider than chunk_size.
_LONG_SCENARIOS = [
    pytest.param("11x5120", 5120, [5120] * 11, id="11x5120"),
    pytest.param(
        "mixed5120",
        5120,
        [1024, 256, 5120, 2048, 5120, 5120, 3072, 5120, 640, 5120, 5120, 4096, 5120, 5120, 4224],
        id="mixed5120",
    ),
]


@pytest.mark.parametrize("seq_len", _COMPRESSOR_SHAPES, ids=[f"seq{s}" for s in _COMPRESSOR_SHAPES])
@pytest.mark.parametrize(
    "mesh_device, device_params, topology",
    _MESH_CONFIGS,
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.parametrize("model_config", _MODEL_CONFIGS)
def test_hca_compressor_mesh(mesh_device, device_params, topology, seq_len, model_config):
    """TtHCACompressor against the unpadded reference. seq_len is the REAL pre-pad length, so the
    unaligned value exercises the pad + trim path."""
    torch.manual_seed(_SEED)

    batch = 1
    config = _config(model_config)
    sp_factor = mesh_device.shape[0]
    compress_rate = config.compress_rates["heavily_compressed_attention"]
    logger.debug(f"mesh={tuple(mesh_device.shape)} seq_len={seq_len} sp={sp_factor}")

    ref = DeepseekV4HCACompressor(config).eval()
    with torch.no_grad():
        ref.position_bias.normal_(0.0, 0.02)
        ref.kv_norm.weight.uniform_(0.5, 1.5)

    hidden = torch.randn(batch, seq_len, config.hidden_size)
    q_residual = torch.zeros(batch, seq_len, config.q_lora_rank)  # unused by HCA
    position_ids = torch.arange(seq_len).unsqueeze(0).expand(batch, -1)

    with torch.no_grad():
        compressed_kv_ref, block_bias_ref = ref(hidden, q_residual, position_ids, past_key_values=None, layer_idx=0)

    tt_model = TtHCACompressor.from_reference(mesh_device, ref, config, sp_axis=0, tp_axis=1, topology=topology)

    hidden_padded, seq_len_actual = TtHCACompressor.prepare_input(hidden, sp_factor, compress_rate)
    tt_input = ttnn.from_torch(
        hidden_padded.unsqueeze(1),
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=(2, 3)),
    )

    # One chunk on its own, so the mask spans exactly this call's padded entries.
    tt_model.alloc_tables(hidden_padded.shape[1], hidden_padded.shape[1], hidden_padded.shape[1] // compress_rate)
    signpost("HCA_START")
    compressed_kv_tt, mask_block_tt = tt_model(tt_input, seq_len_actual=seq_len_actual)
    signpost("HCA_END")
    # Fully replicated (SP-gathered + TP-replicated): take a single replica.
    compressed_kv_out = ttnn.to_torch(
        compressed_kv_tt,
        mesh_composer=ttnn.create_mesh_composer(mesh_device, ttnn.MeshComposerConfig([0, 1], ttnn.MeshShape(1, 1))),
    )
    # One entry per PADDED window; only the leading t_real are real, which is what the reference emits.
    t_real = seq_len_actual // compress_rate
    assert compressed_kv_out.shape[2] == hidden_padded.shape[1] // compress_rate, "expected untrimmed block"
    compressed_kv_out = compressed_kv_out[:, :, :t_real]
    logger.debug(f"TTNN compressed_kv shape: {tuple(compressed_kv_out.shape)} (T_real={t_real})")

    assert (
        compressed_kv_out.shape == compressed_kv_ref.shape
    ), f"shape mismatch: tt {tuple(compressed_kv_out.shape)} vs ref {tuple(compressed_kv_ref.shape)}"

    pcc_passed, pcc_message = assert_with_pcc(
        compressed_kv_ref.to(torch.float32), compressed_kv_out.to(torch.float32), pcc=_COMPRESSOR_PCC
    )
    logger.debug(f"mesh compressor PCC: {pcc_message}")
    assert pcc_passed, f"HCA mesh compressor PCC test failed: {pcc_message}"

    # The compressor emits the mask's compressed columns straight to device, so this checks what production
    # consumes. The block spans every entry the cache can hold and every PADDED query row, while the
    # reference covers only real rows x this call's entries, so slice it down to that. Values are only 0 and
    # -inf, both exact in bfloat16, so the check stays exact.
    mask_block = ttnn.to_torch(
        mask_block_tt,
        mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=(2, 3)),
    )
    width = mask_block.shape[-1] // mesh_device.shape[1]  # TP-replicated; keep one replica
    got = mask_block[:, :, :seq_len_actual, :width][..., :t_real].to(torch.float32)
    torch.testing.assert_close(got, block_bias_ref.to(torch.float32), rtol=0, atol=0)
    logger.debug("PCC test passed!")


@pytest.mark.parametrize("seq_len", _SHAPES, ids=[f"seq{s}" for s in _SHAPES])
@pytest.mark.parametrize(
    "mesh_device, device_params, topology",
    _MESH_CONFIGS,
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.parametrize("model_config, forward_pcc", _MODEL_CONFIGS_FORWARD)
def test_hca_forward_mesh(mesh_device, device_params, topology, seq_len, model_config, forward_pcc):
    """Single-shot TtHCA.forward, SP+TP sharded, for an ARBITRARY prompt length. This is where padding
    awareness is proven: pad-derived compressed entries get trimmed and pad rows masked out."""
    torch.manual_seed(_SEED)

    batch = 1
    config = _config(model_config)
    sw = config.sliding_window
    sp_factor = mesh_device.shape[0]
    compress_rate = config.compress_rates["heavily_compressed_attention"]

    ref = DeepseekV4Attention(config, layer_idx=0).eval()
    assert ref.compressor is not None, "layer_idx=0 must be a heavily_compressed_attention layer"
    with torch.no_grad():
        ref.q_a_norm.weight.uniform_(0.5, 1.5)
        ref.kv_norm.weight.uniform_(0.5, 1.5)
        ref.sinks.normal_(0.0, 1.0)
        ref.compressor.position_bias.normal_(0.0, 0.02)
        ref.compressor.kv_norm.weight.uniform_(0.5, 1.5)

    hidden = torch.randn(batch, seq_len, config.hidden_size)
    position_ids = torch.arange(seq_len).unsqueeze(0).expand(batch, -1)

    with torch.no_grad():
        cos, sin = ref.compressor.rotary_emb(hidden, position_ids=position_ids, layer_type="compress")
        i = torch.arange(seq_len).view(seq_len, 1)
        j = torch.arange(seq_len).view(1, seq_len)
        attn_mask = torch.zeros(seq_len, seq_len).masked_fill(~((j <= i) & (i - j < sw)), float("-inf"))
        attn_mask = attn_mask.view(1, 1, seq_len, seq_len).expand(batch, 1, seq_len, seq_len)
        out_ref, _ = ref(hidden, {"compress": (cos, sin)}, position_ids, attn_mask, past_key_values=None)

    tt_model = TtHCA.from_reference(mesh_device, ref, config, sp_axis=0, tp_axis=1, topology=topology)

    hidden_padded, seq_len_actual = TtHCA.prepare_input(hidden, sp_factor, compress_rate)
    logger.debug(f"mesh={tuple(mesh_device.shape)} S_real={seq_len_actual} S_pad={hidden_padded.shape[1]}")
    ms = tuple(mesh_device.shape)
    tt_input = ttnn.from_torch(
        hidden_padded.unsqueeze(1),
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=ms, dims=(2, 3)),  # seq @ SP, hidden @ TP
    )

    state = tt_model.alloc_state(hidden_padded.shape[1])  # a one-chunk prefill still owns its state
    signpost("HCA_START")
    out_tt = tt_model(tt_input, seq_len_actual=seq_len_actual, state=state)
    signpost("HCA_END")
    out = ttnn.to_torch(
        out_tt, mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, mesh_shape=ms, dims=(2, 3))
    ).squeeze(
        1
    )  # sp -> seq (dim2), tp -> hidden (dim3)
    out = out[:, :seq_len_actual]  # drop the padded tail
    logger.debug(f"TTNN output shape: {tuple(out.shape)}")

    assert out.shape == out_ref.shape, f"shape mismatch: tt {tuple(out.shape)} vs ref {tuple(out_ref.shape)}"

    pcc_passed, pcc_message = assert_with_pcc(out_ref.to(torch.float32), out.to(torch.float32), pcc=forward_pcc)
    logger.debug(f"mesh HCA block PCC: {pcc_message}")
    assert pcc_passed, f"HCA mesh block PCC test failed: {pcc_message}"

    logger.debug("PCC test passed!")


@pytest.mark.parametrize("name, chunk_size, iters_valid", _CHUNKED_SCENARIOS, ids=[n for n, _, _ in _CHUNKED_SCENARIOS])
@pytest.mark.parametrize(
    "mesh_device, device_params, topology",
    _MESH_CONFIGS,
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.parametrize("model_config, chunked_pcc", _MODEL_CONFIGS_CHUNKED)
def test_hca_chunked_prefill_mesh(
    mesh_device, device_params, topology, name, chunk_size, iters_valid, model_config, chunked_pcc
):
    """Chunked prefill with TtHCAState carried across chunks.

    The reference is deliberately NOT chunked: it runs once over the whole prompt and each chunk is
    compared against the matching slice, so the chunked path has to reproduce plain attention rather
    than agree with a reference that shares its assumptions."""
    torch.manual_seed(_SEED)

    batch = 1
    config = _config(model_config)
    sw = config.sliding_window
    compress_rate = config.compress_rates["heavily_compressed_attention"]
    total = sum(iters_valid)

    ref = DeepseekV4Attention(config, layer_idx=0).eval()
    assert ref.compressor is not None, "layer_idx=0 must be a heavily_compressed_attention layer"
    with torch.no_grad():
        ref.q_a_norm.weight.uniform_(0.5, 1.5)
        ref.kv_norm.weight.uniform_(0.5, 1.5)
        ref.sinks.normal_(0.0, 1.0)
        ref.compressor.position_bias.normal_(0.0, 0.02)
        ref.compressor.kv_norm.weight.uniform_(0.5, 1.5)

    hidden = torch.randn(batch, total, config.hidden_size)
    position_ids = torch.arange(total).unsqueeze(0).expand(batch, -1)

    # Ground truth: one unchunked pass over the whole prompt.
    with torch.no_grad():
        cos, sin = ref.compressor.rotary_emb(hidden, position_ids=position_ids, layer_type="compress")
        i = torch.arange(total).view(total, 1)
        j = torch.arange(total).view(1, total)
        attn_mask = torch.zeros(total, total).masked_fill(~((j <= i) & (i - j < sw)), float("-inf"))
        attn_mask = attn_mask.view(1, 1, total, total).expand(batch, 1, total, total)
        out_ref, _ = ref(hidden, {"compress": (cos, sin)}, position_ids, attn_mask, past_key_values=None)

    tt_model = TtHCA.from_reference(mesh_device, ref, config, sp_axis=0, tp_axis=1, topology=topology)
    state = tt_model.alloc_state(total, chunk_tokens=chunk_size)
    ms = tuple(mesh_device.shape)
    logger.debug(f"mesh={ms} scenario={name} chunk_size={chunk_size} iters={iters_valid} total={total}")

    signpost("HCA_START")
    kv_actual = 0
    pccs = []  # (iter, kv_actual, valid, pcc); _report_chunk_pccs judges them after the run
    programs = mesh_device.num_program_cache_entries()
    for it, valid in enumerate(iters_valid):
        # Fixed device width every chunk; a short final chunk is padded up to it.
        chunk = torch.zeros(batch, chunk_size, config.hidden_size)
        chunk[:, :valid] = hidden[:, kv_actual : kv_actual + valid]

        tt_in = ttnn.from_torch(
            chunk.unsqueeze(1),
            device=mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=ms, dims=(2, 3)),
        )
        out_tt = tt_model(tt_in, seq_len_actual=valid, state=state)
        out = ttnn.to_torch(
            out_tt, mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, mesh_shape=ms, dims=(2, 3))
        ).squeeze(1)[:, :valid]

        expected = out_ref[:, kv_actual : kv_actual + valid]
        _, pcc = comp_pcc(expected.to(torch.float32), out.to(torch.float32))
        pccs.append((it, kv_actual, valid, pcc))

        # Every chunk drives identical device shapes, so one compiled program must serve them all. A
        # new entry means some op's attributes moved with the chunk -- exactly what breaks trace later.
        now = mesh_device.num_program_cache_entries()
        logger.debug(f"  iter {it}: program cache {programs} -> {now} (+{now - programs})")
        if it > 0:
            assert now - programs == _WRITE_COMPILES_PER_CHUNK, (
                f"chunk {it} compiled {now - programs} new program(s) ({programs} -> {now}), expected "
                f"{_WRITE_COMPILES_PER_CHUNK}; an op attribute or a shape changed between chunks"
            )
        programs = now

        kv_actual += valid
    signpost("HCA_END")

    _report_chunk_pccs(pccs, chunked_pcc)
    assert state.kv_actual == total
    assert state.entry_count == sum(v // compress_rate for v in iters_valid)

    with torch.no_grad():
        ref_entries, _ = ref.compressor(
            hidden, torch.zeros(batch, total, config.q_lora_rank), position_ids, past_key_values=None, layer_idx=0
        )
    cache = ttnn.to_torch(
        state.compressed_kv,
        mesh_composer=ttnn.create_mesh_composer(mesh_device, ttnn.MeshComposerConfig([0, 1], ttnn.MeshShape(1, 1))),
    )[:, :, : state.entry_count]
    assert cache.shape == ref_entries.shape, f"cache {tuple(cache.shape)} vs ref {tuple(ref_entries.shape)}"
    cache_passed, cache_msg = assert_with_pcc(ref_entries.to(torch.float32), cache.to(torch.float32), pcc=_CACHE_PCC)
    logger.debug(f"  compressed cache PCC: {cache_msg}")
    assert cache_passed, f"compressed cache mismatch: {cache_msg}"

    logger.debug(f"PCC test passed! entries={state.entry_count}")


@pytest.mark.timeout(0)
@pytest.mark.parametrize("name, chunk_size, iters_valid", _LONG_SCENARIOS)
@pytest.mark.parametrize(
    "mesh_device, device_params, topology",
    _MESH_CONFIGS,
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.parametrize("model_config, long_pcc", _MODEL_CONFIGS_LONG)
def test_hca_long_chunked_prefill_mesh(
    mesh_device, device_params, topology, name, chunk_size, iters_valid, model_config, long_pcc
):
    """Demo-sized prompts, 56,320 tokens, 440 entries deep.

    The reference is chunked here, unlike the short scenarios. An unchunked one is not an option at
    this length: eager attention (forced, since sinks make V4 eager-only) materializes
    [1, num_heads, S, S] scores = 726 GB at this length. The short scenarios are what pin chunked == plain
    attention; this test only shows it stays true that many chunks deep."""
    torch.manual_seed(_SEED)

    batch = 1
    config = _config(model_config)
    sw = config.sliding_window
    compress_rate = config.compress_rates["heavily_compressed_attention"]
    total = sum(iters_valid)

    ref = DeepseekV4Attention(config, layer_idx=0).eval()
    with torch.no_grad():
        ref.q_a_norm.weight.uniform_(0.5, 1.5)
        ref.kv_norm.weight.uniform_(0.5, 1.5)
        ref.sinks.normal_(0.0, 1.0)
        ref.compressor.position_bias.normal_(0.0, 0.02)
        ref.compressor.kv_norm.weight.uniform_(0.5, 1.5)

    hidden = torch.randn(batch, total, config.hidden_size)
    position_ids = torch.arange(total).unsqueeze(0).expand(batch, -1)

    tt_model = TtHCA.from_reference(mesh_device, ref, config, sp_axis=0, tp_axis=1, topology=topology)
    state = tt_model.alloc_state(total, chunk_tokens=chunk_size)
    ref_cache = _RefHCACache(DeepseekV4HCACache(config))
    ms = tuple(mesh_device.shape)
    logger.debug(f"mesh={ms} scenario={name} chunk_size={chunk_size} chunks={len(iters_valid)} total={total}")

    signpost("HCA_START")
    kv_actual = 0
    pccs = []  # (iter, kv_actual, valid, pcc); _report_chunk_pccs judges them after the run
    programs = mesh_device.num_program_cache_entries()
    for it, valid in enumerate(iters_valid):
        real = hidden[:, kv_actual : kv_actual + valid]
        chunk_pos = position_ids[:, kv_actual : kv_actual + valid]

        # The reference cache returns [carry | chunk] from update(), so the mask must cover exactly
        # those keys; the reference cats its own block_bias on for the compressed slots.
        carry = min(sw - 1, kv_actual)
        k_pos = torch.cat([torch.arange(kv_actual - carry, kv_actual), chunk_pos[0]])
        ref_mask = _sliding_mask(chunk_pos[0], k_pos, sw).view(1, 1, valid, -1).expand(batch, 1, -1, -1)
        with torch.no_grad():
            cos, sin = ref.compressor.rotary_emb(real, position_ids=chunk_pos, layer_type="compress")
            expected, _ = ref(real, {"compress": (cos, sin)}, chunk_pos, ref_mask, past_key_values=ref_cache)

        # Fixed device width every chunk; a short chunk is padded up to it.
        chunk = torch.zeros(batch, chunk_size, config.hidden_size)
        chunk[:, :valid] = real
        tt_in = ttnn.from_torch(
            chunk.unsqueeze(1),
            device=mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=ms, dims=(2, 3)),
        )
        out_tt = tt_model(tt_in, seq_len_actual=valid, state=state)
        out = ttnn.to_torch(
            out_tt, mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, mesh_shape=ms, dims=(2, 3))
        ).squeeze(1)[:, :valid]

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
    signpost("HCA_END")

    _report_chunk_pccs(pccs, long_pcc)
    assert state.kv_actual == total
    assert state.entry_count == sum(v // compress_rate for v in iters_valid)

    with torch.no_grad():
        ref_entries, _ = ref.compressor(
            hidden, torch.zeros(batch, total, config.q_lora_rank), position_ids, past_key_values=None, layer_idx=0
        )
    cache = ttnn.to_torch(
        state.compressed_kv,
        mesh_composer=ttnn.create_mesh_composer(mesh_device, ttnn.MeshComposerConfig([0, 1], ttnn.MeshShape(1, 1))),
    )[:, :, : state.entry_count]
    assert cache.shape == ref_entries.shape, f"cache {tuple(cache.shape)} vs ref {tuple(ref_entries.shape)}"
    cache_passed, cache_msg = assert_with_pcc(ref_entries.to(torch.float32), cache.to(torch.float32), pcc=_CACHE_PCC)
    logger.debug(f"  compressed cache PCC: {cache_msg}")
    assert cache_passed, f"compressed cache mismatch: {cache_msg}"

    logger.debug(f"{total} tokens OK: entries={state.entry_count}")
