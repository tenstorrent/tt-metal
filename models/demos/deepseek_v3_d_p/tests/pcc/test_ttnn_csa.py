# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""
PCC tests for the DeepSeek-V4 Compressed Sparse Attention (CSA) block (prefill), against the
reference modeling_deepseek_v4.py.

Every test drives a public forward -- no TtCSA private method is called directly:
  - TtCSA.forward, single-shot     1 per-chip prompt length
  - TtCSA.forward, chunked         3 compact scenarios plus one production 8x4 anchor

Both run on both V4 variants, flash and pro.

Every variant keeps its production ``index_topk``. The index list is checked with the same set-recall
principle as the prefill MoE gate: BF16 can swap near-tied candidates at the top-k boundary without
materially changing attention, so exact index equality is the wrong contract. Geometry remains exact:
the test separately rejects duplicates, out-of-range or future compressed IDs, missing sliding keys,
and non-contiguous sentinel tails.

The golden attention gathers only the selected compressed and sliding keys. This is mathematically the
same attention as the reference's dense block-bias path, but avoids materializing an
``[heads, query, full_kv]`` score tensor for the production-length case.

The mesh list, chunk-PCC reporter and cache-PCC floor come from mesh_configs.py and v4_test_utils.py
rather than being restated here: CSA and HCA run the same attention body (v4_attention_base.py) and the same
chunked-vs-unchunked comparison, so those pieces are identical by construction, not by coincidence.
"""

import os

import pytest
import torch
from loguru import logger
from tracy import signpost

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.deepseek_v3_d_p.reference.deepseek_v4.configuration_deepseek_v4 import DeepseekV4Config
from models.demos.deepseek_v3_d_p.reference.deepseek_v4.modeling_deepseek_v4 import (
    DeepseekV4Attention,
    apply_rotary_pos_emb,
)
from models.demos.deepseek_v3_d_p.reference.deepseek_v4_flash_config import (
    DeepSeekV4FlashConfig,
    deepseek_v4_flash_hf_config,
)
from models.demos.deepseek_v3_d_p.reference.deepseek_v4_pro_config import DeepSeekV4ProConfig
from models.demos.deepseek_v3_d_p.tests.pcc.mesh_configs import V4_MESH_CONFIGS
from models.demos.deepseek_v3_d_p.tests.pcc.v4_test_utils import V4_CACHE_PCC, report_chunk_pccs
from models.demos.deepseek_v3_d_p.tt.mla.compressed_sparse_attention import TtCSA
from models.demos.deepseek_v3_d_p.tt.mla.compressor import TtCSACompressor
from models.demos.deepseek_v3_d_p.tt.moe.init_helpers import create_fabric_router_config, get_max_payload_size
from models.demos.deepseek_v3_d_p.tt.v4.attention.csa import TtCSA as V4FlashTtCSA
from models.demos.deepseek_v3_d_p.tt.v4.attention.csa import TtCSACompressor as V4FlashTtCSACompressor
from tests.ttnn.utils_for_testing import assert_with_pcc

_SEED = 42
# PER-CHIP prompt length, not global: the test multiplies it by the mesh's SP factor, so every box runs
# the same local shape and the two variants stay comparable across meshes. 640 is five slabs on every
# mesh here -- csa_slab_align is compress_rate * TILE_SIZE * sp_factor throughout V4_MESH_CONFIGS, so
# 640 * sp_factor is a whole number of slabs and prepare_input pads nothing.
_LOCAL_SHAPES = [640]
# Same recall floor as the BF16 on-device prefill MoE gate.
_INDEX_RECALL = 0.95
# Bound both the reference indexer's [query, heads, compressed] temporary and attention's selected-KV
# contraction independently of total prompt length.
_GOLDEN_QUERY_CHUNK = 64
# sparse_sdpa's "no key here" id, as the index list carries it.
_SENTINEL = 0xFFFFFFFF


def _config(model_config, num_hidden_layers=1):
    """Reference config from one variant's dimension constants, with layer 0 forced to CSA.

    Everything that differs between the variants is passed explicitly, including the indexer fields:
    DeepseekV4Config's defaults happen to be Flash's values, so a Pro config that left them out would
    build Pro widths with Flash's indexer, and the reference would agree with it -- PCC would pass on
    the wrong model.

    ``index_topk`` is the production value. Raising it to make the index list deterministic would turn
    CSA back into dense causal attention and leave the sparse selection path untested."""
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
        index_topk=m.INDEX_TOPK,
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
# PCC decays with depth: every chunk inherits the previous one's error through the compressed cache,
# overlap state, indexer's key cache and carry. With production top-k, BF16 can also swap near-tied
# candidates at the selection boundary; recall judges that selection directly, while the chunked output
# floor allows its downstream effect.
_VARIANTS = [
    ("flash", DeepSeekV4FlashConfig, 0.98, 0.998),
    ("pro", DeepSeekV4ProConfig, 0.98, 0.997),
]
_MODEL_CONFIGS_CHUNKED = [pytest.param(cfg, chunked, id=name) for name, cfg, chunked, _ in _VARIANTS]
_MODEL_CONFIGS_FORWARD = [pytest.param(cfg, fwd, id=name) for name, cfg, _, fwd in _VARIANTS]


# Compact scenarios remain available on every mesh for local regression runs. The production anchor is
# 8x4-only and is the sole chunked scenario retained in CI: this avoids multiplying its 13k-token CPU
# golden over the smaller BH E2E meshes while still exercising the real 5120-token serving width.
_PRODUCTION_SCENARIO = "3chunk-production"
_CHUNKED_SCENARIOS = [
    ("2chunk-full", 1024, [1024, 1024]),
    ("2chunk-ragged", 1024, [1024, 600]),  # a ragged FINAL chunk, the only place one is allowed
    # Three appends, so chunk 1 is a MIDDLE chunk: neither the first (whose index list is the compacted
    # one) nor the last. It is the only place a full carry, the identity permutation and a mid-sequence
    # compressed append all run at once.
    ("3chunk-ragged", 1024, [1024, 1024, 600]),
    (_PRODUCTION_SCENARIO, 5120, [5120, 5120, 3000]),
]


def _ci_unsupported_param_combos_csa(**params):
    production = params["name"] == _PRODUCTION_SCENARIO
    if production and tuple(params["mesh_device"]) != (8, 4):
        return True
    if params["is_ci_env"] or params["is_ci_v2_env"]:
        return not production
    return False


def _ci_unsupported_param_combos_csa_forward(**params):
    return params["is_ci_env"] or params["is_ci_v2_env"]


def _golden(ref, hidden, config):
    """One unchunked reference pass, with index scoring and attention evaluated in bounded chunks."""
    batch, total = hidden.shape[0], hidden.shape[1]
    position_ids = torch.arange(total).unsqueeze(0).expand(batch, -1)
    indexer = ref.compressor.indexer
    indexer_forward = indexer.forward

    with torch.no_grad():
        cos, sin = ref.compressor.rotary_emb(hidden, position_ids=position_ids, layer_type="compress")
        q_residual = ref.q_a_norm(ref.q_a_proj(hidden))
        kv = ref.kv_norm(ref.kv_proj(hidden)).view(batch, total, 1, config.head_dim).transpose(1, 2)
        kv = apply_rotary_pos_emb(kv, cos, sin)
        reference_topk = _golden_topk(indexer, hidden, q_residual, position_ids)

        # The outer compressor's entries do not depend on the indexer's result. Inject the already
        # computed IDs so it builds its block bias without scoring the full prompt a second time.
        indexer.forward = lambda *args, **kwargs: reference_topk
        try:
            compressed_kv, _ = ref.compressor(hidden, q_residual, position_ids, past_key_values=None, layer_idx=0)
        finally:
            indexer.forward = indexer_forward

        raw = kv[:, 0]
        compressed = compressed_kv[:, 0]
        output_chunks = []
        offsets = torch.arange(1 - config.sliding_window, 1)

        for start in range(0, total, _GOLDEN_QUERY_CHUNK):
            stop = min(start + _GOLDEN_QUERY_CHUNK, total)
            query_positions = position_ids[:, start:stop]
            query_len = stop - start

            q = ref.q_b_proj(q_residual[:, start:stop])
            q = q.view(batch, query_len, config.num_attention_heads, config.head_dim).transpose(1, 2)
            q = ref.q_b_norm(q)
            q = apply_rotary_pos_emb(q, cos[:, start:stop], sin[:, start:stop])

            sliding_ids = query_positions.unsqueeze(-1) + offsets
            sliding_valid = sliding_ids >= 0
            sliding_ids = sliding_ids.clamp(min=0)
            batch_ids = torch.arange(batch).view(batch, 1, 1)
            sliding_kv = raw[batch_ids, sliding_ids]

            compressed_ids = reference_topk[:, start:stop]
            compressed_valid = compressed_ids >= 0
            compressed_kv_selected = compressed[batch_ids, compressed_ids.clamp(min=0)]

            selected_kv = torch.cat([sliding_kv, compressed_kv_selected], dim=2)
            selected_valid = torch.cat([sliding_valid, compressed_valid], dim=2)
            scores = torch.einsum("bhtd,btkd->bhtk", q, selected_kv) * ref.scaling
            scores = scores.masked_fill(~selected_valid.unsqueeze(1), float("-inf"))
            sinks = ref.sinks.view(1, -1, 1, 1).expand(batch, -1, query_len, 1)
            combined = torch.cat([scores, sinks], dim=-1)
            combined = combined - combined.max(dim=-1, keepdim=True).values
            probabilities = torch.softmax(combined, dim=-1)[..., :-1]
            attn = torch.einsum("bhtk,btkd->bhtd", probabilities, selected_kv)
            attn = apply_rotary_pos_emb(attn, cos[:, start:stop], -sin[:, start:stop]).transpose(1, 2)
            grouped = attn.reshape(batch, query_len, config.o_groups, -1)
            output_chunks.append(ref.o_b_proj(ref.o_a_proj(grouped).flatten(2)))

    return torch.cat(output_chunks, dim=1), reference_topk, compressed_kv


def _golden_topk(indexer, hidden, q_residual, position_ids):
    """Reference indexer with query scoring chunked to bound its ``[S, heads, S/4]`` intermediate."""
    batch, total, _ = hidden.shape
    ratio = indexer.compress_rate
    n_windows = total // ratio
    kv = indexer.kv_proj(hidden[:, : n_windows * ratio]).view(batch, n_windows, ratio, -1)
    gate = indexer.gate_proj(hidden[:, : n_windows * ratio]).view(batch, n_windows, ratio, -1)
    gate = gate + indexer.position_bias

    overlap_kv = kv.new_zeros((batch, n_windows, 2 * ratio, indexer.head_dim))
    overlap_gate = gate.new_full((batch, n_windows, 2 * ratio, indexer.head_dim), float("-inf"))
    overlap_kv[:, :, ratio:] = kv[..., indexer.head_dim :]
    overlap_gate[:, :, ratio:] = gate[..., indexer.head_dim :]
    if n_windows > 1:
        overlap_kv[:, 1:, :ratio] = kv[:, :-1, :, : indexer.head_dim]
        overlap_gate[:, 1:, :ratio] = gate[:, :-1, :, : indexer.head_dim]
    compressed = indexer.kv_norm(
        (overlap_kv * overlap_gate.softmax(dim=2, dtype=torch.float32).to(overlap_kv.dtype)).sum(dim=2)
    )
    key_positions = torch.arange(n_windows).unsqueeze(0).expand(batch, -1) * ratio
    key_cos, key_sin = indexer.rotary_emb(compressed, position_ids=key_positions, layer_type=indexer.rope_layer_type)
    compressed = apply_rotary_pos_emb(compressed.unsqueeze(1), key_cos, key_sin).squeeze(1)

    top_k = min(indexer.index_topk, n_windows)
    entry_ids = torch.arange(n_windows)
    chunks = []
    for start in range(0, total, _GOLDEN_QUERY_CHUNK):
        stop = min(start + _GOLDEN_QUERY_CHUNK, total)
        chunk_positions = position_ids[:, start:stop]
        query_cos, query_sin = indexer.rotary_emb(
            hidden[:, start:stop], position_ids=chunk_positions, layer_type=indexer.rope_layer_type
        )
        q = indexer.q_b_proj(q_residual[:, start:stop])
        q = q.view(batch, stop - start, indexer.num_heads, indexer.head_dim).transpose(1, 2)
        q = apply_rotary_pos_emb(q, query_cos, query_sin).transpose(1, 2)
        scores = indexer.scorer(q, compressed, hidden[:, start:stop])
        causal_threshold = (chunk_positions + 1) // ratio
        scores = scores.masked_fill(entry_ids.view(1, 1, -1) >= causal_threshold.unsqueeze(-1), float("-inf"))
        indices = scores.topk(top_k, dim=-1).indices
        chunks.append(torch.where(indices < causal_threshold.unsqueeze(-1), indices, -1))
    return torch.cat(chunks, dim=1)


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


def _validate_index_list(
    mesh_device,
    tt_model,
    reference_topk,
    *,
    first_chunk,
    kv_actual,
    valid,
    capacity,
    sliding,
    compress_rate,
):
    """Validate sparse-SDPA geometry exactly and indexer membership by average recall."""
    index = tt_model.index_list(first_chunk=first_chunk)
    ids = ttnn.to_torch(  # sp -> rows (dim2), tp -> dim1 and replicated, so one replica is the answer
        index,
        mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=(2, 1)),
    )
    ttnn.deallocate(index)
    ids = ids[0, 0, :valid].to(torch.int64) & 0xFFFFFFFF  # a -1 readback is the uint32 sentinel
    real = ids != _SENTINEL
    truncated = ((~real[:, :-1]) & real[:, 1:]).any(-1)  # a real id after a sentinel
    got = ids.sort(-1).values
    duplicated = ((got[:, :-1] == got[:, 1:]) & (got[:, :-1] != _SENTINEL)).any(-1)
    assert not truncated.any(), f"{int(truncated.sum())} index rows have a sentinel before a real ID"
    assert not duplicated.any(), f"{int(duplicated.sum())} index rows repeat a key"

    recalls = []
    reference_topk = reference_topk[:, kv_actual : kv_actual + valid][0]
    for row_idx, (row_ids, row_reference) in enumerate(zip(ids, reference_topk)):
        global_position = kv_actual + row_idx
        compressed_set = set(row_ids[row_ids < capacity].tolist())
        reference_set = set(row_reference[row_reference >= 0].tolist())
        causal_entries = (global_position + 1) // compress_rate
        assert len(compressed_set) == len(reference_set)
        assert all(entry < causal_entries for entry in compressed_set)

        window = min(global_position + 1, sliding)
        expected_sliding = set(range(capacity + sliding + row_idx - window + 1, capacity + sliding + row_idx + 1))
        actual_sliding = set(row_ids[(row_ids >= capacity) & (row_ids != _SENTINEL)].tolist())
        assert actual_sliding == expected_sliding
        if reference_set:
            recalls.append(len(compressed_set & reference_set) / len(reference_set))

    recall = sum(recalls) / len(recalls) if recalls else 1.0
    logger.info(f"  iter at kv_actual={kv_actual}: indexer top-k recall {recall:.6f}")
    assert recall >= _INDEX_RECALL, f"indexer top-k recall {recall:.6f} is below {_INDEX_RECALL}"


def _read_entries(mesh_device, state):
    """The compressed entries written so far: the leading ``entry_count`` rows of the joint table.

    The rows after them are the carry and this chunk's raw keys, which belong to the sliding path and
    have no counterpart in ``ref.compressor``. The table is replicated, so one replica is the answer."""
    table = ttnn.to_torch(
        state.joint_kv,
        mesh_composer=ttnn.create_mesh_composer(mesh_device, ttnn.MeshComposerConfig([0, 1], ttnn.MeshShape(1, 1))),
    )
    return table[:, :, : state.entry_count]


@pytest.mark.uncollect_if(pred=_ci_unsupported_param_combos_csa_forward)
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
    config = _config(model_config)

    ref = _reference(config)
    hidden = torch.randn(batch, seq_len, config.hidden_size)
    out_ref, reference_topk, _ = _golden(ref, hidden, config)

    tt_model = TtCSA.from_reference(
        mesh_device, ref, config, sp_axis=0, tp_axis=1, topology=topology, weight_cache_path=tmp_path
    )
    hidden_padded, seq_len_actual = TtCSACompressor.prepare_input(hidden, sp_factor, compress_rate, tp_factor)
    logger.debug(f"mesh={tuple(mesh_device.shape)} S_real={seq_len_actual} S_pad={hidden_padded.shape[1]}")

    state = tt_model.alloc_state(hidden_padded.shape[1])  # a one-chunk prefill still owns its state

    signpost("CSA_START")
    out_tt = tt_model(_upload(mesh_device, hidden_padded), seq_len_actual=seq_len_actual, state=state)
    signpost("CSA_END")
    out = _download(mesh_device, out_tt)[:, :seq_len_actual]  # drop the padded tail
    capacity = state.joint_kv.shape[2] - config.sliding_window - hidden_padded.shape[1]
    _validate_index_list(
        mesh_device,
        tt_model,
        reference_topk,
        first_chunk=True,
        kv_actual=0,
        valid=seq_len_actual,
        capacity=capacity,
        sliding=config.sliding_window,
        compress_rate=compress_rate,
    )

    assert out.shape == out_ref.shape, f"shape mismatch: tt {tuple(out.shape)} vs ref {tuple(out_ref.shape)}"
    pcc_passed, pcc_message = assert_with_pcc(out_ref.to(torch.float32), out.to(torch.float32), pcc=forward_pcc)
    logger.debug(f"mesh CSA block PCC: {pcc_message}")
    assert pcc_passed, f"CSA mesh block PCC test failed: {pcc_message}"


@pytest.mark.uncollect_if(pred=_ci_unsupported_param_combos_csa)
@pytest.mark.parametrize("name, chunk_size, iters_valid", _CHUNKED_SCENARIOS, ids=[n for n, _, _ in _CHUNKED_SCENARIOS])
@pytest.mark.parametrize(
    "mesh_device, device_params, topology",
    V4_MESH_CONFIGS,
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.parametrize("model_config, chunked_pcc", _MODEL_CONFIGS_CHUNKED)
def test_csa_chunked_prefill_mesh(
    mesh_device,
    device_params,
    topology,
    name,
    chunk_size,
    iters_valid,
    model_config,
    chunked_pcc,
    tmp_path,
    expect_error,
):
    """Chunked prefill with TtCSAState carried across chunks.

    The reference is deliberately NOT chunked: it runs once over the whole prompt and each chunk is
    compared against the matching slice, so the chunked path has to reproduce plain attention rather
    than agree with a reference that shares its assumptions."""
    torch.manual_seed(_SEED)

    batch = 1
    compress_rate = model_config.COMPRESS_RATES["compressed_sparse_attention"]
    config = _config(model_config)
    total = sum(iters_valid)

    ref = _reference(config)
    hidden = torch.randn(batch, total, config.hidden_size)
    out_ref, reference_topk, ref_entries = _golden(ref, hidden, config)

    tt_model = TtCSA.from_reference(
        mesh_device, ref, config, sp_axis=0, tp_axis=1, topology=topology, weight_cache_path=tmp_path
    )
    state = tt_model.alloc_state(total, chunk_tokens=chunk_size)
    logger.debug(f"mesh={tuple(mesh_device.shape)} scenario={name} chunk={chunk_size} iters={iters_valid}")

    # The joint table is [compressed | carry | this chunk], so what the other two regions leave is the
    # compressed capacity -- which is also the row the sliding ids are based at.
    capacity = state.joint_kv.shape[2] - config.sliding_window - chunk_size

    signpost("CSA_START")
    kv_actual = 0
    pccs = []  # (iter, kv_actual, valid, pcc); report_chunk_pccs judges them after the run
    cache_pccs = []  # the same, for the compressed entries the chunk appended
    for it, valid in enumerate(iters_valid):
        # Fixed device width every chunk; a short final chunk is padded up to it.
        chunk = torch.zeros(batch, chunk_size, config.hidden_size)
        chunk[:, :valid] = hidden[:, kv_actual : kv_actual + valid]

        cache_entries = mesh_device.num_program_cache_entries()
        out_tt = tt_model(_upload(mesh_device, chunk), seq_len_actual=valid, state=state)
        assert it == 0 or mesh_device.num_program_cache_entries() == cache_entries, f"chunk {it} recompiled"
        out = _download(mesh_device, out_tt)[:, :valid]

        expected = out_ref[:, kv_actual : kv_actual + valid]
        _, pcc = comp_pcc(expected.to(torch.float32), out.to(torch.float32))
        pccs.append((it, kv_actual, valid, pcc))

        # Checked EVERY chunk, and logged as we go rather than after the loop. A chunk's output depends
        # on the entries it attends over, so when the output regresses the first question is whether
        # those entries were right -- and judging the output first would abort before answering it.
        cache = _read_entries(mesh_device, state)
        _, cache_pcc = comp_pcc(ref_entries[..., : state.entry_count, :].to(torch.float32), cache.to(torch.float32))
        cache_log = logger.warning if cache_pcc < V4_CACHE_PCC else logger.info
        cache_log(f"  iter {it} (entries={state.entry_count}): compressed cache PCC {cache_pcc:.6f}")
        cache_pccs.append((it, kv_actual, state.entry_count, cache_pcc))

        _validate_index_list(
            mesh_device,
            tt_model,
            reference_topk,
            first_chunk=it == 0,
            kv_actual=kv_actual,
            valid=valid,
            capacity=capacity,
            sliding=config.sliding_window,
            compress_rate=compress_rate,
        )

        kv_actual += valid
    signpost("CSA_END")

    report_chunk_pccs(pccs, chunked_pcc)
    assert state.kv_actual == total
    assert state.entry_count == sum(v // compress_rate for v in iters_valid)

    worst_it, _, _, worst_cache = min(cache_pccs, key=lambda row: row[3])
    assert (
        worst_cache >= V4_CACHE_PCC
    ), f"worst compressed cache PCC {worst_cache:.6f} (iter {worst_it}) is below the floor {V4_CACHE_PCC}"

    # The contract the scenarios above are built around: a ragged chunk ends the prefill, because the
    # indexer's block-cyclic key cache rotates off a mid-slab offset. Driven by moving the state's own
    # counter rather than by a fourth chunk, so it costs no device work -- the assert is the first thing
    # forward does. Pinned here because loosening it silently is exactly what a PCC would not catch: the
    # picks stay plausible and only their positions are wrong.
    state.kv_actual = chunk_size // 2  # small enough that the max_seq_len check is not what fires
    with expect_error(AssertionError, "only the final chunk may be ragged"):
        tt_model(_upload(mesh_device, chunk), seq_len_actual=chunk_size, state=state)

    logger.debug(f"PCC test passed! entries={state.entry_count}")


# =====================================================================================================================
# DeepSeek-V4-Flash production CSA (tt/v4/attention/csa.py: path A / path B, trace islands) -- from the prefill branch.
#
# TtCSA (V4-Flash compressed sparse attention, path B) vs the reference DeepseekV4Attention of layer 2 with random
# weights: the two-series compressor's entries (PCC >= 0.999), the indexer's top-k selection (Jaccard >= 0.9 vs the
# reference block bias), and the block output single-shot / chunked (PCC >= 0.99) against an UNCHUNKED reference.
#
# Its module-level helpers carry a _v4f / _V4F prefix and its classes a V4Flash alias so they cannot shadow the
# tt/mla tests' helpers above (tests/perf/test_ttnn_swa_perf.py imports _MESH_CONFIGS, _SEED, _build_reference and
# _config from this module, and CI selects the *_mesh tests by node id). Test function names are unchanged.
# =====================================================================================================================


_V4F_SEED = 42
_V4F_COMPRESSOR_PCC = 0.999
_V4F_JACCARD = 0.9
_V4F_BLOCK_PCC = 0.99
_V4F_SINGLE_SHAPES = [1024, 4095]
_V4F_CHUNKED = [("chunk1024-ragged", 1024, [1024, 1024, 300]), ("chunk5120-varying", 5120, [1024, 256, 5120])]
_V4F_MESH_CONFIGS = [
    pytest.param((1, 1), {}, 0, 1, id="single-1x1"),
    pytest.param(
        (2, 1),
        {"fabric_config": ttnn.FabricConfig.FABRIC_1D},
        0,
        1,
        id="sp2-2x1",
        # skip unless exactly 2 devices are visible (TT_VISIBLE_DEVICES=0,1): opening a [2,1] mesh out of 8 visible chips
        # fails at fabric router sync (5 setup ERRORs in the 2x4 regression run of 2026-09-25)
        marks=pytest.mark.requires_mesh_topology(mesh_shape=(2, 1), topology="mesh-2x1"),
    ),
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
]


def _v4f_ref_layer(cfg):
    ref = DeepseekV4Attention(cfg, layer_idx=2).eval()
    assert ref.compressor is not None and hasattr(ref.compressor, "indexer"), "layer 2 must be a CSA layer"
    with torch.no_grad():
        ref.q_a_norm.weight.uniform_(0.5, 1.5)
        ref.kv_norm.weight.uniform_(0.5, 1.5)
        ref.sinks.normal_(0.0, 1.0)
        ref.compressor.position_bias.normal_(0.0, 0.5)
        ref.compressor.kv_norm.weight.uniform_(0.5, 1.5)
        ref.compressor.indexer.position_bias.normal_(0.0, 0.5)
        ref.compressor.indexer.kv_norm.weight.uniform_(0.5, 1.5)
    return ref


def _v4f_ref_forward(ref, hidden, sw):
    total = hidden.shape[1]
    pos = torch.arange(total).unsqueeze(0)
    with torch.no_grad():
        cos, sin = ref.compressor.rotary_emb(hidden, position_ids=pos, layer_type="compress")
        i, j = torch.arange(total).view(total, 1), torch.arange(total).view(1, total)
        mask = torch.zeros(total, total).masked_fill(~((j <= i) & (i - j < sw)), float("-inf")).view(1, 1, total, total)
        out, _ = ref(hidden, {"compress": (cos, sin)}, pos, mask, past_key_values=None)
    return out


def _v4f_mapper(mesh_device, sp_axis, tp_axis):
    dims = [None, None]
    if mesh_device.shape[sp_axis] > 1:
        dims[sp_axis] = 2
    if mesh_device.shape[tp_axis] > 1:
        dims[tp_axis] = 3
    if dims == [None, None]:
        return ttnn.ReplicateTensorToMesh(mesh_device)
    return ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=dims)


def _v4f_to_device(mesh_device, x, sp_axis, tp_axis):
    return ttnn.from_torch(
        x,
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=_v4f_mapper(mesh_device, sp_axis, tp_axis),
    )


def _v4f_to_host(mesh_device, t, sp_axis, tp_axis):
    dims = [0, 0]
    dims[sp_axis], dims[tp_axis] = 2, 3
    return ttnn.to_torch(
        t, mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=tuple(dims))
    )


def _v4f_one_chip(mesh_device, t):
    return ttnn.to_torch(
        t, mesh_composer=ttnn.create_mesh_composer(mesh_device, ttnn.MeshComposerConfig([0, 1], ttnn.MeshShape(1, 1)))
    )


@pytest.mark.parametrize("spike", [False, True], ids=["smooth", "spike"])
@pytest.mark.parametrize(
    "mesh_device, device_params, sp_axis, tp_axis", _V4F_MESH_CONFIGS, indirect=["mesh_device", "device_params"]
)
def test_csa_compressor_entries(mesh_device, device_params, sp_axis, tp_axis, spike):
    """Two chunks through the compressor alone: the second chunk's window 0 takes the first chunk's last Ca rows.

    ``spike``: three tokens scaled x100 so their gate scores sit far above the rest of the chunk (DS4F-0272: the
    real layer 40 does this on ordinary text). The pooling softmax must be stabilised PER GROUP -- one per-channel max
    over the whole chunk underflows every other group's exp to 0 in fp32 and the entries become 0/0."""
    torch.manual_seed(_V4F_SEED)
    cfg = deepseek_v4_flash_hf_config(num_hidden_layers=4)
    ref = _v4f_ref_layer(cfg)
    sp = mesh_device.shape[sp_axis]
    S1, S2 = 1024, 1024
    hidden = torch.randn(1, S1 + S2, cfg.hidden_size)
    if spike:
        hidden[0, [37, 1201, 1700]] *= 100.0
    q_res = torch.randn(1, S1 + S2, cfg.q_lora_rank)
    from models.demos.deepseek_v3_d_p.reference.deepseek_v4.modeling_deepseek_v4 import DeepseekV4CSACache

    class _Cache:
        def __init__(self, layer):
            self.layers = [None, None, layer]

    cache = _Cache(DeepseekV4CSACache(cfg))
    with torch.no_grad():
        ref.compressor(hidden[:, :S1], q_res[:, :S1], torch.arange(S1).unsqueeze(0), cache, 2)
        all_entries, _ = ref.compressor(hidden[:, S1:], q_res[:, S1:], torch.arange(S1, S1 + S2).unsqueeze(0), cache, 2)
    all_entries = all_entries[0, 0]  # [(S1+S2)/4, 512]

    tt = V4FlashTtCSACompressor.from_reference(mesh_device, ref.compressor, cfg, sp_axis=sp_axis, tp_axis=tp_axis)
    cap = (S1 + S2) // 4 + 32
    tt.alloc_tables(S1 + S2, S1, cap)
    prior = tt.empty_prior()
    got = []
    for c, (lo, hi) in enumerate(((0, S1), (S1, S1 + S2))):
        x = _v4f_to_device(mesh_device, hidden[:, lo:hi].unsqueeze(1), sp_axis, tp_axis)
        entries, mask_block, prior = tt(x, hi - lo, lo // 4 * 4, prior)
        got.append(_v4f_one_chip(mesh_device, entries)[0, 0].float())
        assert tuple(mask_block.shape) == (1, 1, S1 // sp, cap)
    got = torch.cat(got, 0)
    assert torch.isfinite(got).all(), f"non-finite entries: {(~torch.isfinite(got)).sum().item()} values"
    ok, pcc = comp_pcc(all_entries.float(), got, _V4F_COMPRESSOR_PCC)
    logger.info(f"CSA compressor entries PCC {pcc} (two chunks of {S1}, spike={spike})")
    assert ok, pcc


@pytest.mark.parametrize("seq_len", _V4F_SINGLE_SHAPES, ids=[f"seq{s}" for s in _V4F_SINGLE_SHAPES])
@pytest.mark.parametrize(
    "mesh_device, device_params, sp_axis, tp_axis", _V4F_MESH_CONFIGS, indirect=["mesh_device", "device_params"]
)
def test_csa_forward(mesh_device, device_params, sp_axis, tp_axis, seq_len):
    torch.manual_seed(_V4F_SEED)
    cfg = deepseek_v4_flash_hf_config(num_hidden_layers=4)
    ref = _v4f_ref_layer(cfg)
    sp = mesh_device.shape[sp_axis]
    hidden = torch.randn(1, seq_len, cfg.hidden_size)
    out_ref = _v4f_ref_forward(ref, hidden, cfg.sliding_window)

    tt = V4FlashTtCSA.from_reference(mesh_device, ref, cfg, sp_axis=sp_axis, tp_axis=tp_axis)
    padded, real = V4FlashTtCSA.prepare_input(hidden, sp)
    state = tt.alloc_state(padded.shape[1])
    out = tt(_v4f_to_device(mesh_device, padded.unsqueeze(1), sp_axis, tp_axis), seq_len_actual=real, state=state)
    out = _v4f_to_host(mesh_device, out, sp_axis, tp_axis).squeeze(1)[:, :real]
    ok, pcc = comp_pcc(out_ref.float(), out.float(), _V4F_BLOCK_PCC)
    logger.info(f"CSA single-shot seq {seq_len} mesh {tuple(mesh_device.shape)}: PCC {pcc}")

    # indexer selection vs the reference block bias (sets of selected entries per query row)
    with torch.no_grad():
        q_res = ref.q_a_norm(ref.q_a_proj(hidden))
        _, block_bias = ref.compressor(hidden, q_res, torch.arange(seq_len).unsqueeze(0), None, 2)
    ref_sel = block_bias[0, 0] == 0  # [S, T]
    sel = _v4f_to_host(mesh_device, tt.debug_last_selection, sp_axis, tp_axis)[0, 0]  # [S_pad, cap]
    sel = sel[:real, : ref_sel.shape[1]] == 0
    inter = (sel & ref_sel).sum(-1).float()
    union = (sel | ref_sel).sum(-1).float().clamp(min=1)
    jaccard = (inter / union).mean().item()
    logger.info(f"CSA indexer selection Jaccard {jaccard:.4f} (rows {real}, entries {ref_sel.shape[1]})")
    assert jaccard >= _V4F_JACCARD, jaccard
    assert ok, pcc


@pytest.mark.parametrize(
    "mesh_device, device_params, sp_axis, tp_axis", _V4F_MESH_CONFIGS, indirect=["mesh_device", "device_params"]
)
def test_csa_sparse_path_matches_dense(mesh_device, device_params, sp_axis, tp_axis):
    """PATH A (sparse_sdpa over [window | top-k], sink) vs PATH B (dense masked SDPA) on the same chunked prefill:
    both vs the reference (>= _V4F_BLOCK_PCC) and A vs B on the chunks that take path A (kv_actual >= topk*rate = 2048).
    """
    chunk_size, iters_valid = 1024, [1024, 1024, 1024, 1024]
    torch.manual_seed(_V4F_SEED)
    cfg = deepseek_v4_flash_hf_config(num_hidden_layers=4)
    ref = _v4f_ref_layer(cfg)
    total = sum(iters_valid)
    hidden = torch.randn(1, total, cfg.hidden_size)
    out_ref = _v4f_ref_forward(ref, hidden, cfg.sliding_window)
    outs, paths = {}, {}
    for sparse in (False, True):
        tt = V4FlashTtCSA.from_reference(mesh_device, ref, cfg, sp_axis=sp_axis, tp_axis=tp_axis, sparse_path=sparse)
        state = tt.alloc_state(total, chunk_tokens=chunk_size)
        kv_actual = 0
        outs[sparse], paths[sparse] = [], []
        for valid in iters_valid:
            chunk = hidden[:, kv_actual : kv_actual + valid]
            out = tt(
                _v4f_to_device(mesh_device, chunk.unsqueeze(1), sp_axis, tp_axis), seq_len_actual=valid, state=state
            )
            outs[sparse].append(_v4f_to_host(mesh_device, out, sp_axis, tp_axis).squeeze(1)[:, :valid].float())
            paths[sparse].append(tt.last_path)
            kv_actual += valid
    assert paths[False] == ["B"] * 4 and paths[True] == ["B", "B", "A", "A"], paths
    worst = {"A": 1.0, "B": 1.0, "A vs B": 1.0}
    kv_actual = 0
    for it, valid in enumerate(iters_valid):
        ref_slice = out_ref[:, kv_actual : kv_actual + valid].float()
        _, pcc_b = comp_pcc(ref_slice, outs[False][it])
        _, pcc_a = comp_pcc(ref_slice, outs[True][it])
        _, pcc_ab = comp_pcc(outs[False][it], outs[True][it])
        logger.info(f"  chunk {it} ({paths[True][it]}): dense {pcc_b:.6f}  sparse {pcc_a:.6f}  A vs B {pcc_ab:.6f}")
        worst["B"] = min(worst["B"], pcc_b)
        if paths[True][it] == "A":
            worst["A"] = min(worst["A"], pcc_a)
            worst["A vs B"] = min(worst["A vs B"], pcc_ab)
        kv_actual += valid
    assert worst["B"] >= _V4F_BLOCK_PCC and worst["A"] >= _V4F_BLOCK_PCC, worst
    # A vs B: 0.999+ when both paths rank the same indexer scores (materialised scorer); the fused scorer
    # (indexer_score_dsa, bf16 head-sum in a different order) breaks near-ties differently, so its selected sets --
    # and the outputs -- may differ at the 1e-3 level while both stay equally close to the reference.
    fused = os.environ.get("PREFILL_CSA_FUSED_INDEXER", "1") == "1"
    assert worst["A vs B"] >= (0.995 if fused else 0.999), worst
    assert worst["A"] >= worst["B"] - 0.002, f"path A drifted from the reference more than path B: {worst}"


@pytest.mark.parametrize("name, chunk_size, iters_valid", _V4F_CHUNKED, ids=[n for n, _, _ in _V4F_CHUNKED])
@pytest.mark.parametrize(
    "mesh_device, device_params, sp_axis, tp_axis", _V4F_MESH_CONFIGS, indirect=["mesh_device", "device_params"]
)
def test_csa_chunked_prefill(mesh_device, device_params, sp_axis, tp_axis, name, chunk_size, iters_valid):
    torch.manual_seed(_V4F_SEED)
    cfg = deepseek_v4_flash_hf_config(num_hidden_layers=4)
    ref = _v4f_ref_layer(cfg)
    total = sum(iters_valid)
    hidden = torch.randn(1, total, cfg.hidden_size)
    out_ref = _v4f_ref_forward(ref, hidden, cfg.sliding_window)
    tt = V4FlashTtCSA.from_reference(mesh_device, ref, cfg, sp_axis=sp_axis, tp_axis=tp_axis)
    state = tt.alloc_state(total, chunk_tokens=chunk_size)
    kv_actual, pccs = 0, []
    programs = mesh_device.num_program_cache_entries()
    for it, valid in enumerate(iters_valid):
        chunk = torch.zeros(1, chunk_size, cfg.hidden_size)
        chunk[:, :valid] = hidden[:, kv_actual : kv_actual + valid]
        out = tt(_v4f_to_device(mesh_device, chunk.unsqueeze(1), sp_axis, tp_axis), seq_len_actual=valid, state=state)
        out = _v4f_to_host(mesh_device, out, sp_axis, tp_axis).squeeze(1)[:, :valid]
        _, pcc = comp_pcc(out_ref[:, kv_actual : kv_actual + valid].float(), out.float())
        now = mesh_device.num_program_cache_entries()
        logger.info(
            f"  {name} iter {it} (kv_actual={kv_actual} valid={valid}): PCC {pcc:.6f}, programs +{now - programs}"
        )
        pccs.append(pcc)
        programs = now
        kv_actual += valid
    assert state.kv_actual == total and state.entry_count == sum(v // 4 for v in iters_valid)
    assert min(pccs) >= _V4F_BLOCK_PCC, f"worst chunk PCC {min(pccs):.6f} < {_V4F_BLOCK_PCC}"


_V4F_TRACE_PARAMS = {
    "fabric_config": ttnn.FabricConfig.FABRIC_2D,
    "fabric_router_config": create_fabric_router_config(max_payload_size=get_max_payload_size()),
    "reliability_mode": ttnn.FabricReliabilityMode.RELAXED_INIT,
    "trace_region_size": 256 * 1024 * 1024,
}
_V4F_MESH_ISLANDS = [
    pytest.param((2, 4), _V4F_TRACE_PARAMS, 0, 1, id="fabric2d-mesh-2x4-trace"),
    pytest.param((8, 4), _V4F_TRACE_PARAMS, 0, 1, id="fabric2d-mesh-8x4-trace"),
]


def _v4f_idx_sets(mesh_device, idx, sp_axis, tp_axis, topk):
    """[S, topk] host indices of a TP-replicated, SP-sharded uint32 [1,1,S_l,topk] tensor -> list of Python sets."""
    dims = [0, 0]
    dims[sp_axis], dims[tp_axis] = 2, 3
    t = ttnn.to_torch(
        idx, mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=tuple(dims))
    )[0, 0, :, :topk]
    return [set(r.tolist()) for r in t]


@pytest.mark.timeout(0)
@pytest.mark.parametrize(
    "mesh_device, device_params, sp_axis, tp_axis", _V4F_MESH_ISLANDS, indirect=["mesh_device", "device_params"]
)
def test_csa_islands_pinpoint(mesh_device, device_params, sp_axis, tp_axis):
    """DS4F-0262 pinpoint at the first TRACED chunk (chunk 1), same weights and inputs, two module instances:
    (1) the static scorer (traced form) vs the eager fused scorer on the SAME state -> selected-set Jaccard per query row;
    (2) the phase decomposition run EAGERLY (prepare/forward_pre/glue/forward_attn) vs the eager forward -> y PCC;
    (3) forward_attn captured as a TraceIsland and replayed -> y PCC vs the eager forward. Whichever stage first drops
    below 0.999 (Jaccard 0.9) carries the defect."""
    from models.demos.deepseek_v3_d_p.tt.v4.trace_island import TraceIsland

    sp = mesh_device.shape[sp_axis]
    chunk = 5120 if sp == 8 else 2048  # chunk 1 must take the sparse/traced path: kv_actual >= topk * rate = 2048
    torch.manual_seed(_V4F_SEED)
    cfg = deepseek_v4_flash_hf_config(num_hidden_layers=4)
    ref = _v4f_ref_layer(cfg)
    hidden = torch.randn(1, 2 * chunk, cfg.hidden_size)
    h0 = _v4f_to_device(mesh_device, hidden[:, :chunk].unsqueeze(1), sp_axis, tp_axis)
    h1 = _v4f_to_device(mesh_device, hidden[:, chunk:].unsqueeze(1), sp_axis, tp_axis)
    # eager reference: two chunks through forward
    tt_e = V4FlashTtCSA.from_reference(mesh_device, ref, cfg, sp_axis=sp_axis, tp_axis=tp_axis, sparse_path=True)
    st_e = tt_e.alloc_state(2 * chunk, chunk_tokens=chunk)
    y0 = tt_e(h0, seq_len_actual=chunk, state=st_e)
    ttnn.deallocate(y0)
    y1_e = tt_e(h1, seq_len_actual=chunk, state=st_e)
    assert tt_e.last_path == "A", tt_e.last_path
    y1_e_host = _v4f_to_host(mesh_device, y1_e, sp_axis, tp_axis).float()
    # islands instance: chunk 0 eager, chunk 1 through the phases
    tt_i = V4FlashTtCSA.from_reference(mesh_device, ref, cfg, sp_axis=sp_axis, tp_axis=tp_axis, sparse_path=True)
    assert tt_i.traceable(), "needs PREFILL_CSA_SPARSE=1 and the fused indexer"
    st_i = tt_i.alloc_state(2 * chunk, chunk_tokens=chunk)
    y0 = tt_i(h0, seq_len_actual=chunk, state=st_i)
    ttnn.deallocate(y0)
    assert tt_i.trace_ready(st_i)
    tt_i.prepare_chunk(st_i, chunk)
    outs = tt_i.forward_pre(h1, st_i, chunk)
    tt_i.glue_chunk(st_i, outs, chunk, None)
    q, q_latent, sliding_kv_g, _nc, entries, keys, cos, sin = outs[:8]
    rate = tt_i.compressor.compress_rate
    n_new = chunk // rate
    idx_static = tt_i.indexer.select_indices_static(q_latent, h1, cos, sin, st_i.index_k, st_i.score_mask)
    idx_fused = tt_i.indexer.select_indices_fused(q_latent, h1, cos, sin, st_i.index_k, st_i.entry_count, n_new, n_new)
    topk = tt_i.indexer.topk
    ss, sf = _v4f_idx_sets(mesh_device, idx_static, sp_axis, tp_axis, topk), _v4f_idx_sets(
        mesh_device, idx_fused, sp_axis, tp_axis, topk
    )
    jac = [len(a & b) / max(1, len(a | b)) for a, b in zip(ss, sf)]
    jac_mean, jac_min = sum(jac) / len(jac), min(jac)
    worst_rows = sorted(range(len(jac)), key=lambda i: jac[i])[:5]
    logger.info(
        f"[pinpoint] scorer static vs fused (chunk 1, {tuple(mesh_device.shape)}, chunk {chunk}): Jaccard mean {jac_mean:.4f} "
        f"min {jac_min:.4f}; worst rows {worst_rows}; static max idx {max(max(r) for r in ss)} fused max idx {max(max(r) for r in sf)}"
    )
    a2 = tt_i.forward_attn(h1, outs, st_i, chunk)
    y1_p_host = _v4f_to_host(mesh_device, a2[0], sp_axis, tp_axis).float()
    _, pcc_phase = comp_pcc(y1_e_host, y1_p_host)
    logger.info(f"[pinpoint] phases (eager) vs forward: y PCC {pcc_phase:.6f}")

    def _where(a, b, tag):
        # localise a deviation: per SP chip (rows r*S_l..) and the worst 128-row blocks
        S_l = chunk // sp
        per_chip = [comp_pcc(a[0, 0, r * S_l : (r + 1) * S_l], b[0, 0, r * S_l : (r + 1) * S_l])[1] for r in range(sp)]
        blocks = [(i, comp_pcc(a[0, 0, i : i + 128], b[0, 0, i : i + 128])[1]) for i in range(0, chunk, 128)]
        worst = sorted(blocks, key=lambda t: t[1])[:6]
        logger.info(
            f"[pinpoint] {tag}: per-SP-chip PCC {[round(x, 5) for x in per_chip]}; worst 128-row blocks (row, pcc) "
            f"{[(i, round(v, 4)) for i, v in worst]}"
        )

    _where(y1_e_host, y1_p_host, "phases vs forward")
    for t in a2:
        ttnn.deallocate(t)
    A2 = TraceIsland(mesh_device, lambda: tt_i.forward_attn(h1, outs, st_i, chunk), [], name="pinpoint.A2")
    cap = A2.capture()
    y1_c_host = _v4f_to_host(mesh_device, cap[0], sp_axis, tp_axis).float()
    _, pcc_cap = comp_pcc(y1_e_host, y1_c_host)
    A2.replay()
    ttnn.synchronize_device(mesh_device)
    y1_r_host = _v4f_to_host(mesh_device, cap[0], sp_axis, tp_axis).float()
    _, pcc_rep = comp_pcc(y1_e_host, y1_r_host)
    logger.info(f"[pinpoint] A2 captured vs forward: y PCC {pcc_cap:.6f}; replayed: {pcc_rep:.6f}")
    _where(y1_e_host, y1_r_host, "A2 replayed vs forward")
    _where(y1_p_host, y1_r_host, "A2 replayed vs phases-eager")
    A2.release()
    for t in outs:
        ttnn.deallocate(t)

    # (4) island A1 captured + replayed vs the eager forward_pre, on a THIRD instance at the same chunk-1 state
    tt_a = V4FlashTtCSA.from_reference(mesh_device, ref, cfg, sp_axis=sp_axis, tp_axis=tp_axis, sparse_path=True)
    st_a = tt_a.alloc_state(2 * chunk, chunk_tokens=chunk)
    y0 = tt_a(h0, seq_len_actual=chunk, state=st_a)
    ttnn.deallocate(y0)
    prior_snap = [ttnn.clone(t) for t in (st_a.prior_c + st_a.prior_i)]

    def restore_priors():
        for src, dst in zip(prior_snap, st_a.prior_c + st_a.prior_i):
            ttnn.copy(src, dst)

    names = [
        "q",
        "q_latent",
        "sliding_kv",
        "next_carry",
        "entries",
        "keys",
        "cos",
        "sin",
        "kv_rm",
        "entries_rm",
        "carry_rm",
    ]
    tt_a.prepare_chunk(st_a, chunk)
    outs_e = tt_a.forward_pre(h1, st_a, chunk)
    host_e = [
        ttnn.to_torch(
            t, mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=(2, 3))
        ).float()
        for t in outs_e
    ]
    prior_e = [
        ttnn.to_torch(
            t, mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=(2, 3))
        ).float()
        for t in st_a.prior_c
    ]
    for t in outs_e:
        ttnn.deallocate(t)
    restore_priors()
    tt_a.prepare_chunk(st_a, chunk)
    A1 = TraceIsland(mesh_device, lambda hh: tt_a.forward_pre(hh, st_a, chunk), [h1], name="pinpoint.A1")
    outs_c = A1.capture()
    ttnn.synchronize_device(mesh_device)
    host_c = [
        ttnn.to_torch(
            t, mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=(2, 3))
        ).float()
        for t in outs_c
    ]
    prior_c = [
        ttnn.to_torch(
            t, mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=(2, 3))
        ).float()
        for t in st_a.prior_c
    ]
    restore_priors()
    A1.replay()
    ttnn.synchronize_device(mesh_device)
    host_r = [
        ttnn.to_torch(
            t, mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=(2, 3))
        ).float()
        for t in outs_c
    ]
    prior_r = [
        ttnn.to_torch(
            t, mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=(2, 3))
        ).float()
        for t in st_a.prior_c
    ]
    rep_cap = {n: comp_pcc(e, c)[1] for n, e, c in zip(names, host_e, host_c)}
    rep_rep = {n: comp_pcc(e, r)[1] for n, e, r in zip(names, host_e, host_r)}
    logger.info(
        f"[pinpoint] A1 captured vs eager forward_pre: " + ", ".join(f"{n} {v:.6f}" for n, v in rep_cap.items())
    )
    logger.info(
        f"[pinpoint] A1 replayed vs eager forward_pre: " + ", ".join(f"{n} {v:.6f}" for n, v in rep_rep.items())
    )
    logger.info(
        f"[pinpoint] priors after A1 (kv, gate): captured vs eager {comp_pcc(prior_e[0], prior_c[0])[1]:.6f} / "
        f"{comp_pcc(prior_e[1], prior_c[1])[1]:.6f}; replayed vs eager {comp_pcc(prior_e[0], prior_r[0])[1]:.6f} / {comp_pcc(prior_e[1], prior_r[1])[1]:.6f}"
    )
    A1.release()
    a1_worst = min(rep_rep.values())
    assert jac_mean >= 0.9, (jac_mean, jac_min)
    # NOTE: a trace capture records without executing, so the output read right after capture() is whatever the buffer
    # held (0.0008 on 2x4, coincidentally the earlier eager result on 8x4) -- only the REPLAY comparisons are meaningful.
    assert pcc_phase >= 0.999 and pcc_rep >= 0.999, (pcc_phase, pcc_cap, pcc_rep)
    assert a1_worst >= 0.999, rep_rep
