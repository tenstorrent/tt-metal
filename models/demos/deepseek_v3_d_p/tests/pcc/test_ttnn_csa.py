# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""
PCC tests for the DeepSeek-V4 Compressed Sparse Attention (CSA) block (prefill), against the
reference modeling_deepseek_v4.py.

Every test drives a public forward -- no TtCSA private method is called directly:
  - TtCSA.forward, single-shot     1 per-chip prompt length
  - TtCSA.forward, chunked         TtCSAState across chunks, 3 scenarios

Both run on both V4 variants, flash and pro.

Every test here runs in the regime where the indexer can reach EVERY entry the cache holds, so it
drops nothing and the block mask reduces to plain causality. That is deliberate: outside it the two
indexers rank near-tied bf16 scores differently, they select different entries, and the block's output
legitimately differs from the reference's by more than numerics -- there is nothing for PCC to say. What
the top-k itself does is covered by tests/pcc/test_ttnn_csa_indexer.py, on its own overlap metric.

The mesh list, the chunk-PCC reporter and the reference sliding mask come from test_ttnn_hca.py and
mesh_configs.py rather than being restated here: CSA and HCA run the same attention body
(v4_attention_base.py) and the same chunked-vs-unchunked comparison, so those three are identical
between them by construction, not by coincidence.
"""

import pytest
import torch
from loguru import logger
from tracy import signpost

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.deepseek_v3_d_p.reference.deepseek_v4.configuration_deepseek_v4 import DeepseekV4Config
from models.demos.deepseek_v3_d_p.reference.deepseek_v4.modeling_deepseek_v4 import DeepseekV4Attention
from models.demos.deepseek_v3_d_p.reference.deepseek_v4_flash_config import DeepSeekV4FlashConfig
from models.demos.deepseek_v3_d_p.reference.deepseek_v4_pro_config import DeepSeekV4ProConfig
from models.demos.deepseek_v3_d_p.tests.pcc.mesh_configs import V4_MESH_CONFIGS
from models.demos.deepseek_v3_d_p.tests.pcc.test_ttnn_hca import _report_chunk_pccs, _sliding_mask
from models.demos.deepseek_v3_d_p.tt.mla.compressed_sparse_attention import TtCSA
from models.demos.deepseek_v3_d_p.tt.mla.compressor import TtCSACompressor
from tests.ttnn.utils_for_testing import assert_with_pcc

_SEED = 42
# PER-CHIP prompt length, not global: the test multiplies it by the mesh's SP factor, so every box runs
# the same local shape and the two variants stay comparable across meshes. 640 is five slabs on every
# mesh here -- csa_slab_align is compress_rate * TILE_SIZE * sp_factor throughout V4_MESH_CONFIGS, so
# 640 * sp_factor is a whole number of slabs and prepare_input pads nothing.
_LOCAL_SHAPES = [640]
# The stored entries, checked at the end of a chunked run. They are written once and never recomputed,
# so this holds no matter how deep the run goes.
_CACHE_PCC = 0.998


def _config(model_config, num_hidden_layers=1, min_index_topk=0):
    """Reference config from one variant's dimension constants, with layer 0 forced to CSA.

    Everything that differs between the variants is passed explicitly, including the indexer fields:
    DeepseekV4Config's defaults happen to be Flash's values, so a Pro config that left them out would
    build Pro widths with Flash's indexer, and the reference would agree with it -- PCC would pass on
    the wrong model.

    ``min_index_topk`` raises the top-k capacity to whatever keeps a run inside the reach-every-entry
    regime. It is a floor rather than an override, so a variant's real value still stands wherever it
    already reaches every entry -- which on the smaller meshes, where the run writes fewer entries than
    the stock capacity, it does."""
    m = model_config
    cfg = DeepseekV4Config(
        hidden_size=m.EMB_SIZE,
        head_dim=m.HEAD_DIM,
        num_attention_heads=m.NUM_ATTENTION_HEADS,
        q_lora_rank=m.Q_LORA_RANK,
        o_groups=m.O_GROUPS,
        num_hidden_layers=num_hidden_layers,
        layer_types=["compressed_sparse_attention"] * num_hidden_layers,
        mlp_layer_types=["moe"] * num_hidden_layers,
        compress_rates=dict(m.COMPRESS_RATES),
        compress_rope_theta=m.COMPRESS_ROPE_THETA,
        rms_norm_eps=m.RMS_NORM_EPS,
        index_n_heads=m.INDEX_N_HEADS,
        index_head_dim=m.INDEX_HEAD_DIM,
        index_topk=max(m.INDEX_TOPK, -(-min_index_topk // 16) * 16),
        sliding_window=m.SLIDING_WINDOW,
    )
    cfg._attn_implementation = "eager"  # V4 is eager-only: the sdpa interface silently drops the sinks
    return cfg


def _reference(config):
    """A CSA attention layer with every uninitialized parameter given a value. ``sinks`` and both
    ``position_bias`` tensors are ``torch.empty``, so a default-constructed layer scores garbage."""
    ref = DeepseekV4Attention(config, layer_idx=0).eval()
    assert ref.compressor is not None, "layer_idx=0 must be a compressed_sparse_attention layer"
    indexer = ref.compressor.indexer
    with torch.no_grad():
        ref.q_a_norm.weight.uniform_(0.5, 1.5)
        ref.kv_norm.weight.uniform_(0.5, 1.5)
        ref.sinks.normal_(0.0, 1.0)
        ref.compressor.position_bias.normal_(0.0, 0.02)
        ref.compressor.kv_norm.weight.uniform_(0.5, 1.5)
        indexer.position_bias.normal_(0.0, 0.02)
        indexer.kv_norm.weight.uniform_(0.5, 1.5)
    return ref


# (id, dimension constants, per-chunk floor, single-shot floor).
#
# PCC decays with depth: every chunk inherits the previous one's error through the compressed cache, the
# overlap state, the indexer's own key cache and the carry, and the softmax widens as the cache fills.
# Pro decays faster, since 128 heads and a 7168-wide hidden make every bf16 reduction longer -- the same
# split HCA shows.
_VARIANTS = [
    ("flash", DeepSeekV4FlashConfig, 0.997, 0.998),
    ("pro", DeepSeekV4ProConfig, 0.994, 0.997),
]
_MODEL_CONFIGS_CHUNKED = [pytest.param(cfg, chunked, id=name) for name, cfg, chunked, _ in _VARIANTS]
_MODEL_CONFIGS_FORWARD = [pytest.param(cfg, fwd, id=name) for name, cfg, _, fwd in _VARIANTS]


# (chunk_size, real lengths). 1024 is slab-aligned on every mesh in V4_MESH_CONFIGS, including the 8x4
# one, whose indexer wants a whole tile of the slab per chip.
_CHUNKED_SCENARIOS = [
    ("2chunk-full", 1024, [1024, 1024]),
    ("2chunk-ragged", 1024, [1024, 600]),  # a ragged FINAL chunk, the only place one is allowed
    # A non-final chunk below chunk_size. Pins the two places real_len and the padded slab width must
    # not be confused: the carry, and the compressed append offset.
    ("3chunk-varying", 1024, [512, 1024, 1024]),
]


def _golden(ref, hidden, config):
    """One unchunked reference pass over the whole prompt."""
    batch, total = hidden.shape[0], hidden.shape[1]
    position_ids = torch.arange(total).unsqueeze(0).expand(batch, -1)
    with torch.no_grad():
        cos, sin = ref.compressor.rotary_emb(hidden, position_ids=position_ids, layer_type="compress")
        mask = _sliding_mask(position_ids[0], position_ids[0], config.sliding_window)
        mask = mask.view(1, 1, total, total).expand(batch, 1, total, total)
        out, _ = ref(hidden, {"compress": (cos, sin)}, position_ids, mask, past_key_values=None)
    return out


def _assert_reaches_every_entry(tt_model, entries_written):
    """The precondition every test here rests on: the indexer's fixed top-k capacity covers every entry
    the run will put in the cache, so it drops none of them and the mask is plain causality.

    Counted on PADDED slabs, not real tokens: the indexer scores whole slabs, so a ragged chunk still
    contributes entries it could pick."""
    capacity = tt_model.indexer.index_topk_capacity
    assert capacity >= entries_written, (
        f"the indexer can only pick {capacity} of the {entries_written} entries this run writes, so the "
        f"mask depends on how near-tied scores rank and the reference is not a golden; shorten the run "
        f"or raise index_topk"
    )


def _upload(mesh_device, chunk):
    return ttnn.from_torch(
        chunk.unsqueeze(1),
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ttnn.ShardTensor2dMesh(  # seq @ SP, hidden @ TP
            mesh_device, mesh_shape=tuple(mesh_device.shape), dims=(2, 3)
        ),
    )


def _download(mesh_device, tensor):
    return ttnn.to_torch(
        tensor,
        mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=(2, 3)),
    ).squeeze(
        1
    )  # sp -> seq (dim2), tp -> hidden (dim3)


@pytest.mark.parametrize("local_seq_len", _LOCAL_SHAPES, ids=[f"local{s}" for s in _LOCAL_SHAPES])
@pytest.mark.parametrize(
    "mesh_device, device_params, topology",
    V4_MESH_CONFIGS,
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.parametrize("model_config, forward_pcc", _MODEL_CONFIGS_FORWARD)
def test_csa_forward_mesh(mesh_device, device_params, topology, local_seq_len, model_config, forward_pcc, tmp_path):
    """Single-shot TtCSA.forward, SP+TP sharded, at a fixed PER-CHIP prompt length: the global length is
    scaled by the mesh's SP factor so every mesh runs the same local shape."""
    torch.manual_seed(_SEED)

    batch = 1
    sp_factor, tp_factor = mesh_device.shape[0], mesh_device.shape[1]
    compress_rate = model_config.COMPRESS_RATES["compressed_sparse_attention"]
    seq_len = local_seq_len * sp_factor
    # Scaling the prompt with the mesh outruns a variant's stock index_topk on the wider ones, so ask
    # for a capacity that covers every entry this run writes.
    config = _config(model_config, min_index_topk=seq_len // compress_rate)

    ref = _reference(config)
    hidden = torch.randn(batch, seq_len, config.hidden_size)
    out_ref = _golden(ref, hidden, config)

    tt_model = TtCSA.from_reference(
        mesh_device, ref, config, sp_axis=0, tp_axis=1, topology=topology, weight_cache_path=tmp_path
    )
    hidden_padded, seq_len_actual = TtCSACompressor.prepare_input(hidden, sp_factor, compress_rate, tp_factor)
    logger.debug(f"mesh={tuple(mesh_device.shape)} S_real={seq_len_actual} S_pad={hidden_padded.shape[1]}")

    state = tt_model.alloc_state(hidden_padded.shape[1])  # a one-chunk prefill still owns its state
    _assert_reaches_every_entry(tt_model, hidden_padded.shape[1] // compress_rate)

    signpost("CSA_START")
    out_tt = tt_model(_upload(mesh_device, hidden_padded), seq_len_actual=seq_len_actual, state=state)
    signpost("CSA_END")
    out = _download(mesh_device, out_tt)[:, :seq_len_actual]  # drop the padded tail

    assert out.shape == out_ref.shape, f"shape mismatch: tt {tuple(out.shape)} vs ref {tuple(out_ref.shape)}"
    pcc_passed, pcc_message = assert_with_pcc(out_ref.to(torch.float32), out.to(torch.float32), pcc=forward_pcc)
    logger.debug(f"mesh CSA block PCC: {pcc_message}")
    assert pcc_passed, f"CSA mesh block PCC test failed: {pcc_message}"


@pytest.mark.parametrize("name, chunk_size, iters_valid", _CHUNKED_SCENARIOS, ids=[n for n, _, _ in _CHUNKED_SCENARIOS])
@pytest.mark.parametrize(
    "mesh_device, device_params, topology",
    V4_MESH_CONFIGS,
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.parametrize("model_config, chunked_pcc", _MODEL_CONFIGS_CHUNKED)
def test_csa_chunked_prefill_mesh(
    mesh_device, device_params, topology, name, chunk_size, iters_valid, model_config, chunked_pcc, tmp_path
):
    """Chunked prefill with TtCSAState carried across chunks.

    The reference is deliberately NOT chunked: it runs once over the whole prompt and each chunk is
    compared against the matching slice, so the chunked path has to reproduce plain attention rather
    than agree with a reference that shares its assumptions."""
    torch.manual_seed(_SEED)

    batch = 1
    compress_rate = model_config.COMPRESS_RATES["compressed_sparse_attention"]
    # The indexer scores whole slabs, so what it has to be able to reach is every entry the padded
    # chunks produce, not just the real ones.
    config = _config(model_config, min_index_topk=len(iters_valid) * chunk_size // compress_rate)
    total = sum(iters_valid)

    ref = _reference(config)
    hidden = torch.randn(batch, total, config.hidden_size)
    out_ref = _golden(ref, hidden, config)

    tt_model = TtCSA.from_reference(
        mesh_device, ref, config, sp_axis=0, tp_axis=1, topology=topology, weight_cache_path=tmp_path
    )
    state = tt_model.alloc_state(total, chunk_tokens=chunk_size)
    _assert_reaches_every_entry(tt_model, len(iters_valid) * chunk_size // compress_rate)
    logger.debug(f"mesh={tuple(mesh_device.shape)} scenario={name} chunk={chunk_size} iters={iters_valid}")

    signpost("CSA_START")
    kv_actual = 0
    pccs = []  # (iter, kv_actual, valid, pcc); _report_chunk_pccs judges them after the run
    for it, valid in enumerate(iters_valid):
        # Fixed device width every chunk; a short final chunk is padded up to it.
        chunk = torch.zeros(batch, chunk_size, config.hidden_size)
        chunk[:, :valid] = hidden[:, kv_actual : kv_actual + valid]

        out_tt = tt_model(_upload(mesh_device, chunk), seq_len_actual=valid, state=state)
        out = _download(mesh_device, out_tt)[:, :valid]

        expected = out_ref[:, kv_actual : kv_actual + valid]
        _, pcc = comp_pcc(expected.to(torch.float32), out.to(torch.float32))
        pccs.append((it, kv_actual, valid, pcc))
        kv_actual += valid
    signpost("CSA_END")

    _report_chunk_pccs(pccs, chunked_pcc)
    assert state.kv_actual == total
    assert state.entry_count == sum(v // compress_rate for v in iters_valid)

    # The compressed entries the run stored, against the reference's own. Checked separately from the
    # block output because the cache is what a later chunk and decode inherit.
    position_ids = torch.arange(total).unsqueeze(0).expand(batch, -1)
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
