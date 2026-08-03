# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""
PCC tests for the DeepSeek-V4 Heavily Compressed Attention (HCA) block (prefill), against the
reference modeling_deepseek_v4.py (paper §2.3.2).

Every test drives a public forward -- no TtHCA private method is called directly:
  - TtHCACompressor.forward        compressed KV entries + block bias
  - TtHCA.forward, single-shot     single device (all 64 heads) and mesh, 12 prompt lengths
  - TtHCA.forward, chunked         TtHCAState across chunks, 3 scenarios + a 14-chunk 57K run

Device-perf for the same block lives in tests/perf/test_ttnn_hca_perf.py.
"""

import pytest
import torch
from loguru import logger
from tracy import signpost

import ttnn
from models.demos.deepseek_v3_d_p.reference.deepseek_v4.configuration_deepseek_v4 import DeepseekV4Config
from models.demos.deepseek_v3_d_p.reference.deepseek_v4.modeling_deepseek_v4 import (
    DeepseekV4Attention,
    DeepseekV4HCACache,
    DeepseekV4HCACompressor,
)
from models.demos.deepseek_v3_d_p.reference.deepseek_v4_flash_config import DeepSeekV4FlashConfig
from models.demos.deepseek_v3_d_p.tt.mla.heavily_compressed_attention import TtHCA, TtHCACompressor
from models.demos.deepseek_v3_d_p.tt.moe.init_helpers import create_fabric_router_config, get_max_payload_size
from tests.ttnn.utils_for_testing import assert_with_pcc

# Real (pre-pad) prompt lengths. Used by the single-device tests directly and by the full-block mesh
# test through prepare_input, which pads each up to compress_rate*sp -- so ragged values are fine on
# both and exercise the pad + trim + mask path. Mesh tests that shard a tensor DIRECTLY (the stems,
# attention, o_proj) cannot use these: an SP shard needs seq divisible by 32*sp.
_SHAPES = [128, 130, 256, 260, 300, 512, 513, 1024, 2048, 4095, 4096, 5120]


def _flash_config(num_hidden_layers=4):
    return DeepseekV4Config(
        hidden_size=DeepSeekV4FlashConfig.EMB_SIZE,
        head_dim=DeepSeekV4FlashConfig.HEAD_DIM,
        num_attention_heads=DeepSeekV4FlashConfig.NUM_ATTENTION_HEADS,
        num_hidden_layers=num_hidden_layers,
        compress_rates=dict(DeepSeekV4FlashConfig.COMPRESS_RATES),
        compress_rope_theta=DeepSeekV4FlashConfig.COMPRESS_ROPE_THETA,
        rms_norm_eps=DeepSeekV4FlashConfig.RMS_NORM_EPS,
    )


# Mesh configs: seq_len restricted to multiples of 32*sp (SP seq shard must stay tile-aligned).
# No fabric for 1x1 (sp=tp=1 skips every collective); FABRIC_1D for 4x2; FABRIC_2D
# (+router/RELAXED_INIT) for 8x4.
_MESH_CONFIGS = [
    pytest.param(
        (1, 1),
        {},
        ttnn.Topology.Linear,
        marks=pytest.mark.requires_mesh_topology(mesh_shape=(1, 1), topology="linear"),
        id="single-1x1",
    ),
    pytest.param(
        (4, 2),
        {"fabric_config": ttnn.FabricConfig.FABRIC_1D},
        ttnn.Topology.Linear,
        marks=pytest.mark.requires_mesh_topology(mesh_shape=(4, 2), topology="mesh-4x2"),
        id="mesh-4x2",
    ),
    pytest.param(
        (8, 4),
        {
            "fabric_config": ttnn.FabricConfig.FABRIC_2D,
            "fabric_router_config": create_fabric_router_config(max_payload_size=get_max_payload_size()),
            "reliability_mode": ttnn.FabricReliabilityMode.RELAXED_INIT,
        },
        ttnn.Topology.Linear,
        marks=pytest.mark.requires_mesh_topology(mesh_shape=(8, 4), topology="mesh-8x4"),
        id="fabric2d-mesh-8x4",
    ),
]


@pytest.mark.parametrize("seq_len", [900, 2048, 5120], ids=["seq900-unaligned", "seq2k", "seq5120"])
@pytest.mark.parametrize(
    "mesh_device, device_params, topology",
    _MESH_CONFIGS,
    indirect=["mesh_device", "device_params"],
)
def test_hca_compressor_mesh(mesh_device, device_params, topology, seq_len):
    """Mesh/TP/SP PCC for TtHCACompressor: wkv/wgate contraction-parallel + TP all-reduce, SP-parallel
    window pooling (input host-padded to compress_rate*sp), compressed KV SP-gathered + trimmed to the
    real length. seq_len is the REAL (pre-pad) length; unaligned values exercise the pad + trim path.
    Compared against the full unpadded DeepseekV4HCACompressor."""
    torch.manual_seed(42)

    batch = 1
    config = _flash_config()
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

    # Host-pad the seq to compress_rate*sp (pre-shard) so each SP shard holds whole compression windows.
    hidden_padded, seq_len_actual = TtHCACompressor.prepare_input(hidden, sp_factor, compress_rate)
    tt_input = ttnn.from_torch(
        hidden_padded.unsqueeze(1),
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=(2, 3)),
    )

    signpost("HCA_START")
    compressed_kv_tt, block_bias_tt = tt_model(tt_input, position_ids, seq_len_actual=seq_len_actual)
    signpost("HCA_END")
    # compressed_kv is fully replicated (SP-gathered + TP-replicated): take a single replica.
    compressed_kv_out = ttnn.to_torch(
        compressed_kv_tt,
        mesh_composer=ttnn.create_mesh_composer(mesh_device, ttnn.MeshComposerConfig([0, 1], ttnn.MeshShape(1, 1))),
    )
    # The block is one entry per PADDED window (a fixed width, so a short chunk adds no program); only
    # the leading seq_len_actual/compress_rate are real, which is what the reference emits.
    t_real = seq_len_actual // compress_rate
    assert compressed_kv_out.shape[2] == hidden_padded.shape[1] // compress_rate, "expected untrimmed block"
    compressed_kv_out = compressed_kv_out[:, :, :t_real]
    logger.debug(f"TTNN compressed_kv shape: {tuple(compressed_kv_out.shape)} (T_real={t_real})")

    assert (
        compressed_kv_out.shape == compressed_kv_ref.shape
    ), f"shape mismatch: tt {tuple(compressed_kv_out.shape)} vs ref {tuple(compressed_kv_ref.shape)}"

    pcc_passed, pcc_message = assert_with_pcc(
        compressed_kv_ref.to(torch.float32), compressed_kv_out.to(torch.float32), pcc=0.999
    )
    logger.debug(f"mesh compressor PCC: {pcc_message}")
    assert pcc_passed, f"HCA mesh compressor PCC test failed: {pcc_message}"

    torch.testing.assert_close(block_bias_tt, block_bias_ref.to(torch.float32), rtol=0, atol=0)
    logger.debug("PCC test passed!")


@pytest.mark.parametrize("seq_len", _SHAPES, ids=[f"seq{s}" for s in _SHAPES])
def test_hca_forward(device, seq_len):
    """
    Full TtHCA block (prefill, single-shot) PCC against DeepseekV4Attention.forward:
    hidden -> query/kv stems + compressor + attention core + grouped output projection
    -> [B, S, hidden].

    Single device, so this is the only test that runs _o_proj with all 64 heads on one chip --
    the shape whose nlp_concat_heads L1 footprint is largest. Keep it: the mesh tests shard the
    heads down to 16 and would not catch a head-count-dependent circular-buffer blowup.
    """
    torch.manual_seed(42)
    batch = 1
    config = _flash_config()
    config._attn_implementation = "eager"  # V4 is eager-only (sinks); force it for the reference
    nh, hd, sw = config.num_attention_heads, config.head_dim, config.sliding_window
    logger.debug(f"batch={batch}, seq_len={seq_len}, heads={nh}, head_dim={hd}, sw={sw}")

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

    logger.debug("Running torch reference DeepseekV4Attention.forward")
    with torch.no_grad():
        cos, sin = ref.compressor.rotary_emb(hidden, position_ids=position_ids, layer_type="compress")
        i = torch.arange(seq_len).view(seq_len, 1)
        j = torch.arange(seq_len).view(1, seq_len)
        attn_mask = torch.zeros(seq_len, seq_len).masked_fill(~((j <= i) & (i - j < sw)), float("-inf"))
        attn_mask = attn_mask.view(1, 1, seq_len, seq_len).expand(batch, 1, seq_len, seq_len)
        out_ref, _ = ref(hidden, {"compress": (cos, sin)}, position_ids, attn_mask, past_key_values=None)
    logger.debug(f"Reference output shape: {tuple(out_ref.shape)}")

    tt_model = TtHCA.from_reference(device, ref, config)
    tt_input = ttnn.from_torch(
        hidden.unsqueeze(1),
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
    )

    logger.debug("Running ttnn TtHCA forward")
    signpost("HCA_START")
    out_tt = tt_model(tt_input, position_ids)
    signpost("HCA_END")
    out = ttnn.to_torch(out_tt).squeeze(1)  # [B, 1, S, hidden] -> [B, S, hidden]
    logger.debug(f"TTNN output shape: {tuple(out.shape)}")

    assert out.shape == out_ref.shape, f"shape mismatch: tt {tuple(out.shape)} vs ref {tuple(out_ref.shape)}"

    pcc_passed, pcc_message = assert_with_pcc(out_ref.to(torch.float32), out.to(torch.float32), pcc=0.998)
    logger.debug(f"HCA block PCC: {pcc_message}")
    assert pcc_passed, f"HCA block PCC test failed: {pcc_message}"

    logger.debug("PCC test passed!")


@pytest.mark.parametrize("seq_len", _SHAPES, ids=[f"seq{s}" for s in _SHAPES])
@pytest.mark.parametrize(
    "mesh_device, device_params, topology",
    _MESH_CONFIGS,
    indirect=["mesh_device", "device_params"],
)
def test_hca_forward_mesh(mesh_device, device_params, topology, seq_len):
    """Full TtHCA block on the mesh for an ARBITRARY prompt length: prepare_input host-pads the seq to
    compress_rate*sp, the block runs SP+TP sharded end to end, and the first seq_len output rows are
    compared against the unpadded DeepseekV4Attention.forward. This is where padding-awareness is proven:
    pad-derived compressed entries are trimmed and pad key/query rows are masked out."""
    torch.manual_seed(42)

    batch = 1
    config = _flash_config()
    config._attn_implementation = "eager"  # V4 is eager-only (sinks); force it for the reference
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

    signpost("HCA_START")
    out_tt = tt_model(tt_input, position_ids, seq_len_actual=seq_len_actual)
    signpost("HCA_END")
    out = ttnn.to_torch(
        out_tt, mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, mesh_shape=ms, dims=(2, 3))
    ).squeeze(
        1
    )  # sp -> seq (dim2), tp -> hidden (dim3)
    out = out[:, :seq_len_actual]  # drop the padded tail
    logger.debug(f"TTNN output shape: {tuple(out.shape)}")

    assert out.shape == out_ref.shape, f"shape mismatch: tt {tuple(out.shape)} vs ref {tuple(out_ref.shape)}"

    pcc_passed, pcc_message = assert_with_pcc(out_ref.to(torch.float32), out.to(torch.float32), pcc=0.998)
    logger.debug(f"mesh HCA block PCC: {pcc_message}")
    assert pcc_passed, f"HCA mesh block PCC test failed: {pcc_message}"

    logger.debug("PCC test passed!")


# Chunked-prefill scenarios as per-iteration VALID token counts. The device tensor is always
# _CHUNK_SIZE wide; only how much of it is real varies -- that is what keeps one compiled program for the
# whole prefill. Non-final chunks must be full: the compressed-cache write copies whole tiles, so its
# offset (entry_count) has to stay a multiple of TILE_SIZE, i.e. chunk_valid % (compress_rate*32) == 0.
# A partial FINAL chunk is fine, nothing follows it.
_CHUNK_SIZE = 4096
_CHUNKED_SCENARIOS = [
    ("2x4k", [4096, 4096]),
    ("3x4k", [4096, 4096, 4000]),
    ("ragged-tail", [4096, 4096, 3000]),
]


@pytest.mark.parametrize("name, iters_valid", _CHUNKED_SCENARIOS, ids=[n for n, _ in _CHUNKED_SCENARIOS])
@pytest.mark.parametrize(
    "mesh_device, device_params, topology",
    _MESH_CONFIGS,
    indirect=["mesh_device", "device_params"],
)
def test_hca_chunked_prefill_mesh(mesh_device, device_params, topology, name, iters_valid):
    """Chunked prefill on the mesh: the prompt is consumed in fixed-width chunks while TtHCAState carries
    the compressed-KV cache and the raw look-back across them.

    The reference is NOT chunked -- it runs once over the whole prompt and each chunk's output is compared
    against the matching slice (the pattern test_mla uses). Comparing against a chunked reference could
    hide a shared misconception; this way the chunked implementation has to reproduce plain attention.

    Also asserts no program is added after the first chunk: every chunk must present identical shapes.
    """
    torch.manual_seed(42)

    batch = 1
    config = _flash_config()
    config._attn_implementation = "eager"  # V4 is eager-only (sinks); force it for the reference
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
    state = tt_model.alloc_state(total)
    ms = tuple(mesh_device.shape)
    logger.debug(f"mesh={ms} scenario={name} iters={iters_valid} total={total}")

    signpost("HCA_START")
    kv_actual, entries_after_first = 0, None
    for it, valid in enumerate(iters_valid):
        # Fixed device width every chunk; a short final chunk is padded up to it.
        chunk = torch.zeros(batch, _CHUNK_SIZE, config.hidden_size)
        chunk[:, :valid] = hidden[:, kv_actual : kv_actual + valid]
        chunk_pos = torch.arange(kv_actual, kv_actual + valid).unsqueeze(0).expand(batch, -1)

        tt_in = ttnn.from_torch(
            chunk.unsqueeze(1),
            device=mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=ms, dims=(2, 3)),
        )
        out_tt = tt_model(tt_in, chunk_pos, seq_len_actual=valid, state=state)
        out = ttnn.to_torch(
            out_tt, mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, mesh_shape=ms, dims=(2, 3))
        ).squeeze(1)[:, :valid]

        # Looser than the single-shot block (0.998): a chunk attends compressed entries emitted by
        # earlier chunks, each already carrying its own bf16 error, so accuracy drifts ~0.0006 per chunk
        # against the fp32 reference. Still far stricter than the 0.98 test_mla holds chunked prefill to.
        expected = out_ref[:, kv_actual : kv_actual + valid]
        pcc_passed, pcc_message = assert_with_pcc(expected.to(torch.float32), out.to(torch.float32), pcc=0.997)
        logger.debug(f"  iter {it} (kv_actual={kv_actual} valid={valid}): PCC {pcc_message}")
        assert pcc_passed, f"chunk {it} PCC failed: {pcc_message}"

        # Shapes are identical every chunk, so nothing may be compiled after the first one.
        cache_entries = mesh_device.num_program_cache_entries()
        if it == 0:
            entries_after_first = cache_entries
        else:
            assert cache_entries == entries_after_first, (
                f"chunk {it} added programs ({entries_after_first} -> {cache_entries}); a shape must have "
                f"changed between chunks"
            )
        kv_actual += valid
    signpost("HCA_END")

    assert state.kv_actual == total
    assert state.entry_count == sum(v // compress_rate for v in iters_valid)

    # The accumulated cache must hold what an unchunked compressor would emit for the whole prompt: the
    # chunk boundaries fall on window boundaries, so both compress the same token windows. This is the
    # only check on the stateful plumbing the compressor test cannot reach -- it never writes at an
    # offset and always runs with first_window_position=0, so a mis-rotated or misplaced chunk would
    # only show up here (its share of the attention mass is too small to move the output PCC much).
    with torch.no_grad():
        ref_entries, _ = ref.compressor(
            hidden, torch.zeros(batch, total, config.q_lora_rank), position_ids, past_key_values=None, layer_idx=0
        )
    cache = ttnn.to_torch(
        state.compressed_kv,
        mesh_composer=ttnn.create_mesh_composer(mesh_device, ttnn.MeshComposerConfig([0, 1], ttnn.MeshShape(1, 1))),
    )[:, :, : state.entry_count]
    assert cache.shape == ref_entries.shape, f"cache {tuple(cache.shape)} vs ref {tuple(ref_entries.shape)}"
    cache_passed, cache_msg = assert_with_pcc(ref_entries.to(torch.float32), cache.to(torch.float32), pcc=0.998)
    logger.debug(f"  compressed cache PCC: {cache_msg}")
    assert cache_passed, f"compressed cache mismatch: {cache_msg}"

    logger.debug(f"PCC test passed! entries={state.entry_count} program cache stable at {entries_after_first}")


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


@pytest.mark.timeout(0)
@pytest.mark.parametrize(
    "mesh_device, device_params, topology",
    _MESH_CONFIGS,
    indirect=["mesh_device", "device_params"],
)
def test_hca_long_chunked_prefill_mesh(mesh_device, device_params, topology):
    """Long chunked prefill: 14 full chunks of 4096 = 57,344 tokens, the demo-sized prompt.

    Unlike the short scenarios this drives the torch reference chunk by chunk too, through
    ``DeepseekV4HCACache``. An unchunked reference is not an option at this length: eager attention
    (forced, since sinks make V4 eager-only) materializes [1, num_heads, S, S] scores, which is 726 GB
    at 57K. The short scenarios keep the unchunked reference and are what pins chunked == plain
    attention; this test only has to show it stays true 14 chunks and 448 compressed entries deep.
    """
    torch.manual_seed(42)

    batch = 1
    chunks = 14  # 14 * 4096 = 57,344 tokens
    config = _flash_config()
    config._attn_implementation = "eager"  # sinks: the sdpa interface silently drops s_aux
    sw = config.sliding_window
    compress_rate = config.compress_rates["heavily_compressed_attention"]
    total = _CHUNK_SIZE * chunks

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
    state = tt_model.alloc_state(total)
    ref_cache = _RefHCACache(DeepseekV4HCACache(config))
    ms = tuple(mesh_device.shape)
    logger.debug(f"mesh={ms} chunks={chunks}x{_CHUNK_SIZE} total={total}")

    signpost("HCA_START")
    kv_actual, entries_after_first = 0, None
    for it in range(chunks):
        chunk = hidden[:, kv_actual : kv_actual + _CHUNK_SIZE]
        chunk_pos = position_ids[:, kv_actual : kv_actual + _CHUNK_SIZE]

        # Reference: same chunk loop. The cache returns [carry | chunk] from update(), so the mask it
        # needs covers exactly those keys; the compressor's block_bias is cat'd on for the compressed
        # slots by the reference itself.
        carry = min(sw - 1, kv_actual)
        k_pos = torch.cat([torch.arange(kv_actual - carry, kv_actual), chunk_pos[0]])
        ref_mask = _sliding_mask(chunk_pos[0], k_pos, sw).view(1, 1, _CHUNK_SIZE, -1).expand(batch, 1, -1, -1)
        with torch.no_grad():
            cos, sin = ref.compressor.rotary_emb(chunk, position_ids=chunk_pos, layer_type="compress")
            expected, _ = ref(chunk, {"compress": (cos, sin)}, chunk_pos, ref_mask, past_key_values=ref_cache)

        tt_in = ttnn.from_torch(
            chunk.unsqueeze(1),
            device=mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=ms, dims=(2, 3)),
        )
        out_tt = tt_model(tt_in, chunk_pos, seq_len_actual=_CHUNK_SIZE, state=state)
        out = ttnn.to_torch(
            out_tt, mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, mesh_shape=ms, dims=(2, 3))
        ).squeeze(1)

        # Drift is ~0.00044 per chunk against the fp32 reference (same bf16 accumulation the short
        # chunked test sees at 0.997 over 3 chunks); 0.99 leaves headroom to roughly chunk 21.
        pcc_passed, pcc_message = assert_with_pcc(expected.to(torch.float32), out.to(torch.float32), pcc=0.99)
        logger.debug(f"  iter {it} (kv_actual={kv_actual} entries={state.entry_count}): PCC {pcc_message}")
        assert pcc_passed, f"chunk {it} PCC failed: {pcc_message}"

        cache_entries = mesh_device.num_program_cache_entries()
        if it == 0:
            entries_after_first = cache_entries
        else:
            assert cache_entries == entries_after_first, (
                f"chunk {it} added programs ({entries_after_first} -> {cache_entries}); a shape must have "
                f"changed between chunks"
            )
        kv_actual += _CHUNK_SIZE
    signpost("HCA_END")

    assert state.kv_actual == total
    assert state.entry_count == total // compress_rate

    # Independent check on the stateful plumbing: 448 entries written at 14 different offsets must match
    # what a single unchunked compressor pass emits (cheap -- the compressor has no [S, S] anywhere).
    with torch.no_grad():
        ref_entries, _ = ref.compressor(
            hidden, torch.zeros(batch, total, config.q_lora_rank), position_ids, past_key_values=None, layer_idx=0
        )
    cache = ttnn.to_torch(
        state.compressed_kv,
        mesh_composer=ttnn.create_mesh_composer(mesh_device, ttnn.MeshComposerConfig([0, 1], ttnn.MeshShape(1, 1))),
    )[:, :, : state.entry_count]
    assert cache.shape == ref_entries.shape, f"cache {tuple(cache.shape)} vs ref {tuple(ref_entries.shape)}"
    cache_passed, cache_msg = assert_with_pcc(ref_entries.to(torch.float32), cache.to(torch.float32), pcc=0.998)
    logger.debug(f"  compressed cache PCC: {cache_msg}")
    assert cache_passed, f"compressed cache mismatch: {cache_msg}"

    logger.debug(f"{total} tokens OK: entries={state.entry_count} program cache stable at {entries_after_first}")
