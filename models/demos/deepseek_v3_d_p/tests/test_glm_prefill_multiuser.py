# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Multi-user chunked prefill for GLM-5.3 on synthetic weights: the oracle for packed (batched) forwards.

A packed forward holds several chunks, possibly from different users at different depths. Attention runs
once per chunk against the chunk's own KV slot; the FFN half (ffn_norm, MoE) runs once over every chunk's
rows. Whatever the packing, each chunk's result must equal what that chunk gets alone. These tests pin
that down on device, device against device, so they need no checkpoint, ttnn cache or golden trace; all
but the pipeline test also run on a LoudBox.

* test_glm_multiuser_interleaved_matches_sequential: one chunk per forward. N users prefilled chunk by
  chunk in round-robin order into one set of cache slots, and each user start to finish into another,
  must agree. Checks per-user state on the one-chunk path: KVPE and indexer-K slots, and the top-k that
  GLM-5.3 "shared" layers reuse from the last "full" layer.
* test_glm_packed_forward_matches_one_chunk_forwards: four chunks per forward (PrefillChunk, as in
  MiniMax-M3's packed forwards). A schedule with partial chunks, a partial chunk followed by a real one,
  one user's consecutive chunks in the same forward, users at different depths and empty tail chunks must
  match the same chunks run each alone in its own forward, and those must match a model built for one chunk
  per forward.
* test_glm_packed_traced_matches_eager: the same schedule through TtPrefillRuntime.prefill_forward with
  use_trace (one capture, replayed per forward; per-chunk metadata and the MoE padding config read on
  device) must fill the same KV as the eager packed forward.
* test_glm_packed_pipeline_matches_one_rank (8x4 only): the model split over two pipeline ranks on one mesh,
  packed forwards handed rank to rank, must match one rank, eager and traced.

Mesh and chunk: each mesh runs the chunk that gives the production per-chip shape, 640 rows per SP row
(2x4 -> 1,280-token chunks, 8x4 -> 5,120). On 2x4 every chip sees what a galaxy chip sees in attention
(640 rows, 16 heads, 160 sparse_sdpa tokens) but holds 32 experts instead of 8.

Depth: 4 layers with every GLM-5.3 layer kind (_TestGLM53Config): 0-1 dense with full indexers, 2 MoE with a
full indexer, 3 MoE reusing layer 2's top-k. The real model's first MoE layer with a full indexer is 6; the
weights are synthetic, so the test moves it to 2. Both MoE layers share one set of experts, about 19 GB of
host RAM.
"""

import copy
import gc

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import is_blackhole
from models.demos.common.prefill.chunk_layout import pack_forward_tokens, rotated_chunk_positions
from models.demos.common.prefill.prefill_chunk import PrefillChunk, validate_chunks
from models.demos.deepseek_v3_d_p.reference.glm_5_3_config import GLM53Config
from models.demos.deepseek_v3_d_p.tests.fabric_profiles import (
    fabric2d_device_params,
    moe_fabric_payload,
    torus_xy_device_params,
)
from models.demos.deepseek_v3_d_p.tests.glm_synthetic_weights import glm_random_transformer_state_dict
from models.demos.deepseek_v3_d_p.tt.mla.indexer import full_indexer_rank
from models.demos.deepseek_v3_d_p.tt.moe import tt_moe_gate_prefill
from models.demos.deepseek_v3_d_p.tt.moe.tt_moe_gate_prefill import GateComputeMode, assert_gate_mode_matches_adapter
from models.demos.deepseek_v3_d_p.tt.runners.adapters.glm_5_3 import GLM53Adapter
from models.demos.deepseek_v3_d_p.tt.runners.kv_caches import MlaKvCaches
from models.demos.deepseek_v3_d_p.tt.tt_ccl import per_axis_topology
from models.demos.deepseek_v3_d_p.tt.tt_prefill_runtime import TtPrefillRuntime, TtPrefillRuntimeConfig
from models.demos.deepseek_v3_d_p.tt.tt_prefill_transformer import TtPrefillTransformer
from models.demos.deepseek_v3_d_p.utils.chunk_config import PREFILL_CHUNK_TOKENS_PER_CHIP
from models.demos.deepseek_v3_d_p.utils.kv_cache_utils import MlaKvCacheFormat, init_kvpe_cache, init_mla_kv_cache
from models.demos.deepseek_v3_d_p.utils.test_utils import gather_cache_natural
from tests.ttnn.utils_for_testing import comp_pcc

NUM_LAYERS = 4


class _TestGLM53Config(GLM53Config):
    """GLM-5.3 with two dense layers instead of three, so 4 layers hold every layer kind (module docstring)."""

    NUM_DENSE_LAYERS = 2


_TEST_INDEXER_TYPES = ["full", "full", "full", "shared"]


def _test_config(config):
    """A copy of ``config`` with the test's layer kinds; config_only is shared process-wide, so never mutate it."""
    config = copy.deepcopy(config)
    config.first_k_dense_replace = _TestGLM53Config.NUM_DENSE_LAYERS
    config.indexer_types = list(_TEST_INDEXER_TYPES)
    return config


NUM_USERS = 3
# Three chunks, so the last chunk's queries see more than INDEX_TOPK (2,048) keys on 2x4 and the
# indexer really selects, and every chunk after the first gathers a non-empty KV prefix.
N_CHUNKS = 3
GATE_MODE = GateComputeMode.DEVICE_FP32

# Chunks per packed forward. 4 x 640 rows = 2,560 MoE rows per chip.
FORWARD_CHUNKS = 4

# Interleaving changes only the order of whole forward() calls, so it should be bit-exact; the bar
# leaves room for nothing but a reduction-order tie.
INTERLEAVED_PCC = 0.9999
# Packing changes nothing per chunk except which rows share the MoE call. Everything computed before the first MoE
# output (the dense layers' KV and indexer-K) must therefore be bit-exact. After it, a token's MoE output depends on
# the tokens sharing the call (#63555, rounding-sized), so the KV and the last layer get bars with margin over the
# measured 0.9994 (KV) and 0.9970 (output), tight enough to catch a real packing error.
PACKED_KV_PCC = 0.998
PACKED_OUTPUT_PCC = 0.995
# The packed model vs a model built for one chunk per forward (forward_chunks=1): the same chunk, but every matmul
# runs on FORWARD_CHUNKS chunks' rows in one and one chunk's in the other, so the tiling and its rounding differ from
# layer 0's output on (layer 1's KV: 0.99982), and the synthetic router's near-ties amplify that through the MoE
# layers (measured on galaxy 8x4: KV >= 0.9931, last layer >= 0.9832). Layer 0's KV is computed per row from the
# embedding, before any of that, so it must agree to rounding; a misplaced or mis-padded row would break it outright.
SINGLE_MODEL_FIRST_KV_PCC = 0.99999
SINGLE_MODEL_KV_PCC = 0.99
SINGLE_MODEL_OUTPUT_PCC = 0.975
# Two different users must NOT agree; this catches a comparison that reads the same slot twice.
DISTINCT_USERS_MAX_PCC = 0.99
# Traced and eager run the same programs (the traced ones read their scalars on device), so the caches
# should agree exactly.
TRACED_PCC = 0.9999
GLM_TRACE_REGION_SIZE = 512 * 1024 * 1024

MESH_PARAMS = [
    pytest.param(
        (2, 4),
        fabric2d_device_params(
            fabric_payload_size=moe_fabric_payload(GLM53Config), l1_small_size=GLM53Adapter.l1_small_size
        ),
        1,
        marks=pytest.mark.requires_mesh_topology(mesh_shape=(2, 4), topology="mesh-2x4"),
        id="fabric2d-2x4",
    ),
    pytest.param(
        (8, 4),
        torus_xy_device_params(
            fabric_payload_size=moe_fabric_payload(GLM53Config), l1_small_size=GLM53Adapter.l1_small_size
        ),
        2,
        marks=pytest.mark.requires_mesh_topology(mesh_shape=(8, 4), topology="mesh-8x4"),
        id="torus-xy-8x4",
    ),
]


def _build(mesh_device, config, device_params, num_links, chunk, num_slots, forward_chunks=1):
    """Random-weight chunked transformer plus KVPE and indexer-K caches with ``num_slots`` user slots."""
    sp_axis, tp_axis = 0, 1
    mesh_shape = list(mesh_device.shape)
    seq_cache = N_CHUNKS * chunk
    # _resolve_config_only is lru_cached and shared process-wide; don't leak max_seq_len into it.
    config = _test_config(config)
    config.max_seq_len = seq_cache

    state_dict = glm_random_transformer_state_dict(config, _TestGLM53Config, NUM_LAYERS, share_experts=True)
    transformer = TtPrefillTransformer(
        mesh_device=mesh_device,
        config=config,
        model_cfg=_TestGLM53Config,
        state_dict=state_dict,
        num_layers=NUM_LAYERS,
        seq_len=chunk,
        dispatch_buffer_capacity_factor=8,
        num_links=num_links,
        topology=per_axis_topology(device_params["fabric_config"]),
        sp_axis=sp_axis,
        tp_axis=tp_axis,
        gate_fallback_mode=GATE_MODE,
        is_chunked=True,
        slot_num=num_slots,
        max_seq_len=seq_cache,
        routing_use_l1_small_for_semaphores=GLM53Adapter.routing_use_l1_small_for_semaphores,
        tp_shard_kv=True,
        sparse_kv_cache_format=MlaKvCacheFormat.BF16_RM,
        forward_chunks=forward_chunks,
    )
    del state_dict

    kvpe_cache = init_mla_kv_cache(
        cache_format=MlaKvCacheFormat.BF16_RM,
        hf_config=config,
        mesh_device=mesh_device,
        seq_len=seq_cache,
        mesh_shape=mesh_shape,
        sp_axis=sp_axis,
        tp_axis=tp_axis,
        num_kvpe_cache_layers=NUM_LAYERS,
        num_users=num_slots,
    )
    # Strided by the full-indexer layers among those built (3 of 4), matching the indexer's slot formula.
    index_kv_cache = init_kvpe_cache(
        kvpe_cache_head_dim=config.index_head_dim,
        mesh_device=mesh_device,
        seq_len=seq_cache,
        mesh_shape=mesh_shape,
        sp_axis=sp_axis,
        tp_axis=tp_axis,
        num_kvpe_cache_layers=full_indexer_rank(config, NUM_LAYERS),
        num_users=num_slots,
        dtype=ttnn.bfloat8_b,
    )
    return config, transformer, kvpe_cache, index_kv_cache


def _token_ids(config, num_users, length):
    """Distinct deterministic in-vocab ids per user, [length] each."""
    return [
        torch.randint(0, config.vocab_size, (length,), generator=torch.Generator().manual_seed(1000 + u))
        for u in range(num_users)
    ]


def _forward_input(mesh_device, chunks, ids_by_slot, chunk):
    """A forward's tokens in the packed per-chip layout (pack_forward_tokens), SP-sharded on device. A
    partial chunk's tail and an empty chunk are padded with token 0."""
    sp = mesh_device.shape[0]
    chunk_tokens = []
    for c in chunks:
        real = [] if c.is_empty else ids_by_slot[c.slot_id][c.actual_start : c.actual_end].tolist()
        chunk_tokens.append(real + [0] * (chunk - len(real)))
    packed = pack_forward_tokens(chunk_tokens, [c.actual_start for c in chunks], sp)
    return ttnn.from_torch(
        torch.tensor(packed, dtype=torch.int64).reshape(sp, 1, len(chunks) * chunk // sp),
        device=mesh_device,
        dtype=ttnn.uint32,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=(0, None)),
    )


def _run_forward(transformer, kvpe_cache, index_kv_cache, mesh_device, chunks, ids_by_slot, chunk):
    """One forward of ``chunks``; returns the last layer's output on host, [1, len(chunks) * chunk, emb],
    in the packed per-chip row order. A one-chunk forward goes through the regular (non-packed) path."""
    tt_tokens = _forward_input(mesh_device, chunks, ids_by_slot, chunk)
    if len(chunks) == 1 and transformer.forward_chunks == 1:
        c = chunks[0]
        out = transformer.forward(
            tt_tokens,
            kvpe_cache,
            actual_isl=chunk,
            actual_start=c.actual_start,
            actual_end=c.actual_end,
            cache_user_id=c.slot_id,
            return_intermediates=True,
            index_kv_cache=index_kv_cache,
        )
    else:
        out = transformer.forward(
            tt_tokens,
            kvpe_cache,
            actual_isl=chunk,
            chunks=chunks,
            return_intermediates=True,
            index_kv_cache=index_kv_cache,
        )
    ttnn.synchronize_device(mesh_device)
    ttnn.deallocate(tt_tokens)
    return out[f"layer_{NUM_LAYERS - 1}"].to(torch.float32)


def _chunk_real_rows(hidden, k, num_chunks, chunk_obj, sp, chunk_local):
    """Chunk k's real rows out of a forward's host output (SP-chip-major: chip c holds every chunk's
    chunk_local rows back to back), as [n_real, emb]."""
    rows = []
    positions = rotated_chunk_positions(chunk_obj.actual_start, sp, chunk_local)
    for c in range(sp):
        base = (c * num_chunks + k) * chunk_local
        for r in range(chunk_local):
            if positions[c][r] < chunk_obj.actual_end:
                rows.append(hidden[0, base + r])
    return torch.stack(rows)


def _read_user_slots(cache, mesh_device, layers_per_user, num_slots):
    """Per-slot cache rows [layers_per_user, seq_cache, D], in device (shard-major) order. Both sides of
    every comparison are read the same way, so there is nothing to un-rotate."""
    rows, _ = gather_cache_natural(cache, mesh_device, tp_shard_kv=True)
    assert rows.shape[0] == num_slots * layers_per_user, (rows.shape, num_slots, layers_per_user)
    return [rows[s * layers_per_user : (s + 1) * layers_per_user] for s in range(num_slots)]


def _pcc(a, b):
    _, pcc = comp_pcc(a, b)
    return float(pcc)


def _compare_slots(label, kv_slots, idx_slots, pairs, threshold):
    """Each (reference slot, slot under test) pair must hold the same KVPE and indexer-K rows in every layer."""
    failures = []
    for ref_slot, got_slot in pairs:
        for name, slots in (("kvpe", kv_slots), ("indexer_k", idx_slots)):
            ref, got = slots[ref_slot], slots[got_slot]
            for layer in range(ref.shape[0]):
                pcc = _pcc(ref[layer], got[layer])
                logger.info(f"[{label}] slot {got_slot} vs {ref_slot} {name}[{layer}] PCC {pcc:.6f}")
                if pcc < threshold:
                    failures.append(f"slot {got_slot} vs {ref_slot} {name} layer {layer} PCC {pcc:.6f} < {threshold}")
    return failures


@pytest.mark.parametrize(
    "mesh_device, device_params, num_links", MESH_PARAMS, indirect=["mesh_device", "device_params"]
)
@pytest.mark.parametrize("variant", ["glm_5_3"], indirect=True, ids=["glm53"])
@pytest.mark.skipif(not is_blackhole(), reason="GLM DSA ops (indexer / sparse SDPA) are Blackhole-only")
@pytest.mark.timeout(0)
def test_glm_multiuser_interleaved_matches_sequential(variant, config_only, mesh_device, device_params, num_links):
    """Round-robin over users (slots N..2N-1) must reproduce each user run alone (slots 0..N-1)."""
    assert_gate_mode_matches_adapter(variant, GATE_MODE)
    chunk = PREFILL_CHUNK_TOKENS_PER_CHIP * mesh_device.shape[0]
    num_slots = 2 * NUM_USERS
    mesh_device.enable_program_cache()
    config, transformer, kvpe_cache, index_kv_cache = _build(
        mesh_device, config_only, device_params, num_links, chunk, num_slots
    )
    try:
        ids = _token_ids(config, NUM_USERS, N_CHUNKS * chunk)
        # Slots 0..N-1 and N..2N-1 hold the same users.
        ids_by_slot = {s: ids[s % NUM_USERS] for s in range(num_slots)}
        logger.info(f"mesh={list(mesh_device.shape)} chunk={chunk} users={NUM_USERS} chunks={N_CHUNKS}")

        def run(slot, c):
            chunks = [PrefillChunk(slot, c * chunk, (c + 1) * chunk)]
            return _run_forward(transformer, kvpe_cache, index_kv_cache, mesh_device, chunks, ids_by_slot, chunk)

        sequential = {(u, c): run(u, c) for u in range(NUM_USERS) for c in range(N_CHUNKS)}
        interleaved = {}
        for c in range(N_CHUNKS):
            for u in range(NUM_USERS):
                interleaved[(u, c)] = run(NUM_USERS + u, c)

        failures = []
        for key, ref in sequential.items():
            pcc = _pcc(ref, interleaved[key])
            logger.info(f"[interleaved] user {key[0]} chunk {key[1]} last-layer PCC {pcc:.6f}")
            if pcc < INTERLEAVED_PCC:
                failures.append(f"user {key[0]} chunk {key[1]} hidden PCC {pcc:.6f} < {INTERLEAVED_PCC}")
        kv_slots = _read_user_slots(kvpe_cache, mesh_device, NUM_LAYERS, num_slots)
        idx_slots = _read_user_slots(index_kv_cache, mesh_device, full_indexer_rank(config, NUM_LAYERS), num_slots)
        failures += _compare_slots(
            "interleaved", kv_slots, idx_slots, [(u, NUM_USERS + u) for u in range(NUM_USERS)], INTERLEAVED_PCC
        )

        # The users must actually differ, or a slot mix-up would pass unnoticed.
        cross = _pcc(sequential[(0, N_CHUNKS - 1)], sequential[(1, N_CHUNKS - 1)])
        assert cross < DISTINCT_USERS_MAX_PCC, f"users 0 and 1 produced the same output (PCC {cross:.6f})"
        assert not failures, f"[interleaved] {len(failures)} mismatch(es):\n" + "\n".join(failures)
    finally:
        transformer.release_sub_device_managers()


def _packed_schedule(chunk):
    """(user lengths, forwards) for four users. Every forward holds FORWARD_CHUNKS chunks; user u lives in
    slot u. Covers a partial chunk followed by a real one (that chip routes its pad rows too), one user's
    consecutive chunks in the same forward (the second reads the first's KV), users at different depths
    in one forward, and empty tail chunks."""
    C = chunk
    lengths = [3 * C, C + 704, 896, 2 * C]  # A: 3 full, B: full + partial, C: one partial, D: 2 full
    E = PrefillChunk(0, 0, 0)
    forwards = [
        [PrefillChunk(0, 0, C), PrefillChunk(2, 0, 896), PrefillChunk(1, 0, C), E],
        [PrefillChunk(0, C, 2 * C), PrefillChunk(1, C, C + 704), PrefillChunk(3, 0, C), PrefillChunk(3, C, 2 * C)],
        [PrefillChunk(0, 2 * C, 3 * C), E, E, E],
    ]
    return lengths, forwards


@pytest.mark.parametrize(
    "mesh_device, device_params, num_links", MESH_PARAMS, indirect=["mesh_device", "device_params"]
)
@pytest.mark.parametrize("variant", ["glm_5_3"], indirect=True, ids=["glm53"])
@pytest.mark.skipif(not is_blackhole(), reason="GLM DSA ops (indexer / sparse SDPA) are Blackhole-only")
@pytest.mark.timeout(0)
def test_glm_packed_forward_matches_one_chunk_forwards(
    variant, config_only, mesh_device, device_params, num_links, monkeypatch
):
    """Packed forwards (slots 0..3) must reproduce every chunk run alone in its own forward (slots 4..7).

    Both sides use the same packed model, so the reference forward is the chunk followed by empty chunks:
    the comparison isolates what sharing a forward with other chunks does to a chunk. A second model built for
    one chunk per forward (forward_chunks=1, the regular path) then runs the same chunks, and the packed model's
    one-chunk forwards must match it: that catches an error common to every packed forward (its row layout,
    the MoE sized for FORWARD_CHUNKS chunks, padding), which the first comparison cannot see.

    The packed padding-config cache is cut to one entry, so its LRU eviction runs between ragged forwards.
    """
    assert_gate_mode_matches_adapter(variant, GATE_MODE)
    monkeypatch.setattr(tt_moe_gate_prefill, "ROWS_PADDING_CONFIG_CACHE_SIZE", 1)
    sp = mesh_device.shape[0]
    chunk = PREFILL_CHUNK_TOKENS_PER_CHIP * sp
    chunk_local = chunk // sp
    lengths, forwards = _packed_schedule(chunk)
    num_users = len(lengths)
    num_slots = 2 * num_users
    mesh_device.enable_program_cache()
    config, transformer, kvpe_cache, index_kv_cache = _build(
        mesh_device, config_only, device_params, num_links, chunk, num_slots, forward_chunks=FORWARD_CHUNKS
    )
    try:
        for chunks in forwards:
            validate_chunks(
                chunks,
                chunk_size=chunk,
                forward_chunks=FORWARD_CHUNKS,
                num_users=num_slots,
                max_seq_len=N_CHUNKS * chunk,
            )
        ids = _token_ids(config, num_users, max(lengths))
        ids_by_slot = {s: ids[s % num_users] for s in range(num_slots)}
        empty = PrefillChunk(0, 0, 0)

        packed = {}  # (slot, actual_start) -> real rows of the last layer's output
        for f, chunks in enumerate(forwards):
            logger.info(f"[packed] forward {f}: {chunks}")
            hidden = _run_forward(transformer, kvpe_cache, index_kv_cache, mesh_device, chunks, ids_by_slot, chunk)
            for k, c in enumerate(chunks):
                if not c.is_empty:
                    packed[(c.slot_id, c.actual_start)] = _chunk_real_rows(
                        hidden, k, FORWARD_CHUNKS, c, sp, chunk_local
                    )

        # Reference: the same chunks, each alone in its own forward, in order within each user.
        failures = []
        alone = {}  # (slot, actual_start) -> real rows of the chunk alone in a packed forward
        for chunks in forwards:
            for c in chunks:
                if c.is_empty:
                    continue
                ref_chunk = PrefillChunk(c.slot_id + num_users, c.actual_start, c.actual_end)
                hidden = _run_forward(
                    transformer,
                    kvpe_cache,
                    index_kv_cache,
                    mesh_device,
                    [ref_chunk] + [empty] * (FORWARD_CHUNKS - 1),
                    ids_by_slot,
                    chunk,
                )
                ref = _chunk_real_rows(hidden, 0, FORWARD_CHUNKS, ref_chunk, sp, chunk_local)
                alone[(c.slot_id, c.actual_start)] = ref
                got = packed[(c.slot_id, c.actual_start)]
                pcc = _pcc(ref, got)
                logger.info(f"[packed] {c} last-layer PCC {pcc:.6f} max|diff| {(ref - got).abs().max().item():.3e}")
                if pcc < PACKED_OUTPUT_PCC:
                    failures.append(f"{c} hidden PCC {pcc:.6f} < {PACKED_OUTPUT_PCC}")

        kv_slots = _read_user_slots(kvpe_cache, mesh_device, NUM_LAYERS, num_slots)
        idx_slots = _read_user_slots(index_kv_cache, mesh_device, full_indexer_rank(config, NUM_LAYERS), num_slots)
        # KV of layer L comes from layer L-1's output, so it is exact while every earlier layer is dense.
        exact_through = _TestGLM53Config.NUM_DENSE_LAYERS
        full_layers = [l for l, t in enumerate(config.indexer_types[:NUM_LAYERS]) if t == "full"]
        for u in range(num_users):
            ref_slot, got_slot = u + num_users, u
            for name, slots, layers in (("kvpe", kv_slots, range(NUM_LAYERS)), ("indexer_k", idx_slots, full_layers)):
                for rank, layer in enumerate(layers):
                    ref, got = slots[ref_slot][rank], slots[got_slot][rank]
                    if layer <= exact_through:
                        diff = (ref - got).abs().max().item()
                        logger.info(f"[packed] slot {got_slot} vs {ref_slot} {name}[{layer}] max|diff| {diff:.3e}")
                        if diff != 0:
                            failures.append(
                                f"slot {got_slot} {name} layer {layer} not bit-exact (max|diff| {diff:.3e})"
                            )
                    else:
                        pcc = _pcc(ref, got)
                        logger.info(f"[packed] slot {got_slot} vs {ref_slot} {name}[{layer}] PCC {pcc:.6f}")
                        if pcc < PACKED_KV_PCC:
                            failures.append(f"slot {got_slot} {name} layer {layer} PCC {pcc:.6f} < {PACKED_KV_PCC}")

        cross = _pcc(packed[(0, 0)], packed[(3, 0)])
        assert cross < DISTINCT_USERS_MAX_PCC, f"users A and D produced the same output (PCC {cross:.6f})"
    finally:
        transformer.release_sub_device_managers()
    del transformer, kvpe_cache, index_kv_cache
    gc.collect()  # free the packed model's device buffers before the second model allocates

    # The packed model's one-chunk forwards vs a model built for one chunk per forward (same weights).
    _, single, single_kvpe, single_index = _build(
        mesh_device, config_only, device_params, num_links, chunk, num_users, forward_chunks=1
    )
    try:
        for chunks in forwards:
            for c in chunks:
                if c.is_empty:
                    continue
                hidden = _run_forward(single, single_kvpe, single_index, mesh_device, [c], ids_by_slot, chunk)
                ref = _chunk_real_rows(hidden, 0, 1, c, sp, chunk_local)
                got = alone[(c.slot_id, c.actual_start)]
                pcc = _pcc(ref, got)
                logger.info(f"[single] {c} last-layer PCC {pcc:.6f} max|diff| {(ref - got).abs().max().item():.3e}")
                if pcc < SINGLE_MODEL_OUTPUT_PCC:
                    failures.append(f"{c} hidden PCC {pcc:.6f} vs forward_chunks=1 < {SINGLE_MODEL_OUTPUT_PCC}")
        single_kv = _read_user_slots(single_kvpe, mesh_device, NUM_LAYERS, num_users)
        single_idx = _read_user_slots(single_index, mesh_device, full_indexer_rank(config, NUM_LAYERS), num_users)
        for u in range(num_users):
            for name, ours, theirs, layers in (
                ("kvpe", kv_slots, single_kv, range(NUM_LAYERS)),
                ("indexer_k", idx_slots, single_idx, full_layers),
            ):
                for rank, layer in enumerate(layers):
                    pcc = _pcc(theirs[u][rank], ours[u + num_users][rank])
                    logger.info(f"[single] slot {u} {name}[{layer}] PCC {pcc:.6f}")
                    bar = SINGLE_MODEL_FIRST_KV_PCC if layer == 0 else SINGLE_MODEL_KV_PCC
                    if pcc < bar:
                        failures.append(f"slot {u} {name} layer {layer} PCC {pcc:.6f} vs forward_chunks=1 < {bar}")
        assert not failures, f"[packed] {len(failures)} mismatch(es):\n" + "\n".join(failures)
    finally:
        single.release_sub_device_managers()


@pytest.mark.parametrize(
    "mesh_device, device_params, num_links",
    [
        pytest.param(
            (2, 4),
            fabric2d_device_params(
                fabric_payload_size=moe_fabric_payload(GLM53Config),
                l1_small_size=GLM53Adapter.l1_small_size,
                trace_region_size=GLM_TRACE_REGION_SIZE,
            ),
            1,
            marks=pytest.mark.requires_mesh_topology(mesh_shape=(2, 4), topology="mesh-2x4"),
            id="fabric2d-2x4",
        ),
        pytest.param(
            (8, 4),
            torus_xy_device_params(
                fabric_payload_size=moe_fabric_payload(GLM53Config),
                l1_small_size=GLM53Adapter.l1_small_size,
                trace_region_size=GLM_TRACE_REGION_SIZE,
            ),
            2,
            marks=pytest.mark.requires_mesh_topology(mesh_shape=(8, 4), topology="mesh-8x4"),
            id="torus-xy-8x4",
        ),
    ],
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.parametrize("variant", ["glm_5_3"], indirect=True, ids=["glm53"])
@pytest.mark.skipif(not is_blackhole(), reason="GLM DSA ops (indexer / sparse SDPA) are Blackhole-only")
@pytest.mark.timeout(0)
def test_glm_packed_traced_matches_eager(variant, config_only, mesh_device, device_params, num_links):
    """Traced packed forwards (slots 0..3) must fill the same KVPE and indexer-K as the eager packed
    forward of the same chunks on the same model (slots 4..7)."""
    assert_gate_mode_matches_adapter(variant, GATE_MODE)
    sp = mesh_device.shape[0]
    chunk = PREFILL_CHUNK_TOKENS_PER_CHIP * sp
    lengths, forwards = _packed_schedule(chunk)
    num_users = len(lengths)
    seq_cache = N_CHUNKS * chunk
    config = _test_config(config_only)
    config.max_seq_len = seq_cache
    rt_config = TtPrefillRuntimeConfig(
        num_layers=NUM_LAYERS,
        max_seq_len=seq_cache,
        mesh_shape=tuple(mesh_device.shape),
        chunk_size=chunk,
        forward_tokens_budget=FORWARD_CHUNKS * chunk,
        num_users=2 * num_users,
        num_links=num_links,
        topology=per_axis_topology(device_params["fabric_config"]),
        capacity_factor=8,
        gate_fallback_mode=GATE_MODE,
        model_cfg=_TestGLM53Config,
        routing_use_l1_small_for_semaphores=GLM53Adapter.routing_use_l1_small_for_semaphores,
        sparse_kv_cache_format=MlaKvCacheFormat.BF16_RM,
        use_trace=True,
        tp_shard_kv=True,
    )
    mesh_device.enable_program_cache()
    state_dict = glm_random_transformer_state_dict(config, _TestGLM53Config, NUM_LAYERS, share_experts=True)
    runtime = TtPrefillRuntime(mesh_device, config, state_dict, rt_config)
    del state_dict
    cache_slots = rt_config.cache_num_users  # the users plus the traced forward's scratch slot
    mesh_shape = list(mesh_device.shape)
    caches = MlaKvCaches(
        kvpe=init_mla_kv_cache(
            cache_format=MlaKvCacheFormat.BF16_RM,
            hf_config=config,
            mesh_device=mesh_device,
            seq_len=seq_cache,
            mesh_shape=mesh_shape,
            sp_axis=0,
            tp_axis=1,
            num_kvpe_cache_layers=NUM_LAYERS,
            num_users=cache_slots,
        ),
        index=init_kvpe_cache(
            kvpe_cache_head_dim=config.index_head_dim,
            mesh_device=mesh_device,
            seq_len=seq_cache,
            mesh_shape=mesh_shape,
            sp_axis=0,
            tp_axis=1,
            num_kvpe_cache_layers=full_indexer_rank(config, NUM_LAYERS),
            num_users=cache_slots,
            dtype=ttnn.bfloat8_b,
        ),
    )
    try:
        runtime.compile(caches)
        runtime.capture_trace(caches)
        ids = _token_ids(config, num_users, max(lengths))

        def forward_input(chunks, user_of):
            tokens = []
            for c in chunks:
                real = [] if c.is_empty else ids[user_of(c.slot_id)][c.actual_start : c.actual_end].tolist()
                tokens.append(real + [0] * (chunk - len(real)))
            return runtime.make_forward_input(tokens, [c.actual_start for c in chunks])

        for f, chunks in enumerate(forwards):
            logger.info(f"[traced] forward {f}: {chunks}")
            runtime.prefill_forward(forward_input(chunks, lambda s: s), caches, chunks)
        ttnn.synchronize_device(mesh_device)

        # Eager reference on the same model: the packed forward called directly, so it runs the host-scalar
        # programs rather than the captured ones.
        for chunks in forwards:
            ref = [
                c if c.is_empty else PrefillChunk(c.slot_id + num_users, c.actual_start, c.actual_end) for c in chunks
            ]
            inp = forward_input(ref, lambda s: s - num_users)
            runtime.model.forward(inp, caches.kvpe, actual_isl=chunk, chunks=ref, index_kv_cache=caches.index)
            ttnn.deallocate(inp)
        ttnn.synchronize_device(mesh_device)

        kv_slots = _read_user_slots(caches.kvpe, mesh_device, NUM_LAYERS, cache_slots)
        idx_slots = _read_user_slots(caches.index, mesh_device, full_indexer_rank(config, NUM_LAYERS), cache_slots)
        # Every user, including C, whose only chunk is partial from position 0 (#63408, fixed by #63538).
        failures = _compare_slots(
            "traced", kv_slots, idx_slots, [(u + num_users, u) for u in range(num_users)], TRACED_PCC
        )
        cross = _pcc(kv_slots[0][NUM_LAYERS - 1], kv_slots[3][NUM_LAYERS - 1])
        assert cross < DISTINCT_USERS_MAX_PCC, f"users A and D hold the same KV (PCC {cross:.6f})"
        assert not failures, f"[traced] {len(failures)} mismatch(es):\n" + "\n".join(failures)
    finally:
        runtime.release_trace()


# The pipeline test's layer kinds: rank 0 (layers 0-1, dense, both "full") hands its last top-k to rank 1, whose first
# layer is "shared" (MoE, reusing it) and whose second computes its own ("full", MoE). The boundary carries every chunk's
# indices packed, and the downstream rank also owns an indexer-K cache layer.
_PIPELINE_INDEXER_TYPES = ["full", "full", "shared", "full"]
PIPELINE_SPLIT = 2  # rank 0 owns layers [0, 2), rank 1 [2, 4)
D2H_READ_TIMEOUT_S = 120


def _pipeline_caches(config, mesh_device, first_layer_idx, num_layers, num_slots, seq_cache):
    mesh_shape = list(mesh_device.shape)
    first_full = full_indexer_rank(config, first_layer_idx)
    return MlaKvCaches(
        kvpe=init_mla_kv_cache(
            cache_format=MlaKvCacheFormat.BF16_RM,
            hf_config=config,
            mesh_device=mesh_device,
            seq_len=seq_cache,
            mesh_shape=mesh_shape,
            sp_axis=0,
            tp_axis=1,
            num_kvpe_cache_layers=num_layers,
            num_users=num_slots,
        ),
        index=init_kvpe_cache(
            kvpe_cache_head_dim=config.index_head_dim,
            mesh_device=mesh_device,
            seq_len=seq_cache,
            mesh_shape=mesh_shape,
            sp_axis=0,
            tp_axis=1,
            num_kvpe_cache_layers=full_indexer_rank(config, first_layer_idx + num_layers) - first_full,
            num_users=num_slots,
            dtype=ttnn.bfloat8_b,
        ),
    )


def _read_d2h_records(service, count):
    """``count`` 12 B (slot, start, end) records off the D2H FIFO, failing (not hanging) if fewer arrive."""
    import struct
    import threading

    records, errors = [], []

    def read():
        try:
            for _ in range(count):
                records.append(struct.unpack("<3I", service.read_metadata()[:12]))
        except Exception as e:  # surfaced below
            errors.append(e)

    t = threading.Thread(target=read, daemon=True)
    t.start()
    t.join(D2H_READ_TIMEOUT_S)
    assert not errors, errors
    assert not t.is_alive(), f"only {len(records)} of {count} D2H ack records arrived in {D2H_READ_TIMEOUT_S}s"
    return records


@pytest.mark.parametrize(
    "mesh_device, device_params, num_links",
    [
        pytest.param(
            (8, 4),
            torus_xy_device_params(
                fabric_payload_size=moe_fabric_payload(GLM53Config),
                l1_small_size=GLM53Adapter.l1_small_size,
                trace_region_size=GLM_TRACE_REGION_SIZE,
            ),
            2,
            marks=pytest.mark.requires_mesh_topology(mesh_shape=(8, 4), topology="mesh-8x4"),
            id="torus-xy-8x4",
        ),
    ],
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.parametrize("traced", [False, True], ids=["eager", "traced"])
@pytest.mark.parametrize("variant", ["glm_5_3"], indirect=True, ids=["glm53"])
@pytest.mark.skipif(not is_blackhole(), reason="GLM DSA ops (indexer / sparse SDPA) are Blackhole-only")
@pytest.mark.timeout(0)
def test_glm_packed_pipeline_matches_one_rank(variant, config_only, mesh_device, device_params, num_links, traced):
    """Packed forwards split over two pipeline ranks must fill the same KVPE and indexer-K as one rank holding every
    layer, with the activation and every chunk's GLM-5.3 indexer indices handed from rank 0 to rank 1 the way the
    runner's D2D sockets hand them over (the same per-chip tensors).

    Rank 0 acks through a real D2H service: one 12 B (slot, start, end) record per non-empty chunk per layer, each
    chunk's records contiguous and in pack order (#63438's packed contract), all after the forward's KV writes.

    Eager: the split must be bit-exact against one rank holding every layer (same programs on the same inputs).
    Traced: the ranks replay their captures (per-chunk metadata, MoE padding config, persistent input and indices
    staged before each replay) into slots 0..3, against the same two ranks' models run eagerly into slots 4..7, at
    TRACED_PCC -- only the traced models on the mesh, as in production, where each rank is its own process: an eager
    MoE model sharing the mesh's TT_CCL state and sub-device managers cannot coexist with two captures."""
    assert_gate_mode_matches_adapter(variant, GATE_MODE)
    sp = mesh_device.shape[0]
    chunk = PREFILL_CHUNK_TOKENS_PER_CHIP * sp
    lengths, forwards = _packed_schedule(chunk)
    num_users = len(lengths)
    seq_cache = N_CHUNKS * chunk
    config = _test_config(config_only)
    config.indexer_types = list(_PIPELINE_INDEXER_TYPES)
    config.max_seq_len = seq_cache
    mesh_device.enable_program_cache()

    num_slots = num_users * (2 if traced else 1)

    def rt_config(first_layer_idx, num_layers, use_trace):
        return TtPrefillRuntimeConfig(
            num_layers=num_layers,
            max_seq_len=seq_cache,
            mesh_shape=tuple(mesh_device.shape),
            chunk_size=chunk,
            forward_tokens_budget=FORWARD_CHUNKS * chunk,
            num_users=num_slots,
            num_links=num_links,
            topology=per_axis_topology(device_params["fabric_config"]),
            capacity_factor=8,
            gate_fallback_mode=GATE_MODE,
            model_cfg=_TestGLM53Config,
            routing_use_l1_small_for_semaphores=GLM53Adapter.routing_use_l1_small_for_semaphores,
            sparse_kv_cache_format=MlaKvCacheFormat.BF16_RM,
            use_trace=use_trace,
            tp_shard_kv=True,
            first_layer_idx=first_layer_idx,
            is_first_rank=first_layer_idx == 0,
            is_last_rank=first_layer_idx + num_layers == NUM_LAYERS,
        )

    state_dict = glm_random_transformer_state_dict(config, _TestGLM53Config, NUM_LAYERS, share_experts=True)

    def layers(a, b):  # shallow copies: a build may pop keys from its layer dicts
        return {"embed_weight": state_dict["embed_weight"], "layers": [dict(l) for l in state_dict["layers"][a:b]]}

    split = [(0, PIPELINE_SPLIT), (PIPELINE_SPLIT, NUM_LAYERS - PIPELINE_SPLIT)]
    ref = (
        None
        if traced
        else TtPrefillRuntime(mesh_device, config, layers(0, NUM_LAYERS), rt_config(0, NUM_LAYERS, False))
    )
    ranks = [
        TtPrefillRuntime(mesh_device, config, layers(first, first + n), rt_config(first, n, traced))
        for first, n in split
    ]
    del state_dict
    assert ranks[0].forwards_indexer_indices and ranks[1].receives_indexer_indices
    assert (
        ranks[1].indexer_indices_shape()[2] == FORWARD_CHUNKS * ranks[1].model.layers[0].mla.indexer_indices_shape()[2]
    )

    ref_caches = (
        None if traced else _pipeline_caches(config, mesh_device, 0, NUM_LAYERS, ref.config.cache_num_users, seq_cache)
    )
    rank_caches = [
        _pipeline_caches(config, mesh_device, first, n, r.config.cache_num_users, seq_cache)
        for (first, n), r in zip(split, ranks)
    ]
    d2h = ttnn.D2HStreamService(
        mesh_device,
        global_spec=None,
        fifo_size_bytes=16 * 1024,
        worker_cores=ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0)),
        metadata_size_bytes=12,
        metadata_ring=True,
    )
    acks_per_chunk = ranks[0].config.num_layers
    try:
        ids = _token_ids(config, num_users, max(lengths))

        def tokens(runtime, chunks):
            toks = []
            for c in chunks:
                real = [] if c.is_empty else ids[c.slot_id % num_users][c.actual_start : c.actual_end].tolist()
                toks.append(real + [0] * (chunk - len(real)))
            return runtime.make_forward_input(toks, [c.actual_start for c in chunks])

        for r, caches in zip(ranks, rank_caches):
            r.compile(caches)
        if not traced:
            ref.compile(ref_caches)
        if traced:
            ranks[0].set_d2h_ack_service(d2h)
            for r, caches in zip(ranks, rank_caches):
                r.capture_trace(caches)
            warm = _read_d2h_records(d2h, ranks[0].warmup_ack_count())
            logger.info(f"[pipeline] drained {len(warm)} warm-up D2H records")

        for f, chunks in enumerate(forwards):
            logger.info(f"[pipeline {'traced' if traced else 'eager'}] forward {f}: {chunks}")
            out = ranks[0].prefill_forward(tokens(ranks[0], chunks), rank_caches[0], chunks, d2h_service=d2h)
            # What the D2D sockets deliver: copies of rank 0's per-chip activation and packed indices.
            act, idx = ttnn.clone(out), ttnn.clone(ranks[0].outbound_indexer_indices)
            if not traced:
                ttnn.deallocate(out)
            assert ranks[1].prefill_forward(act, rank_caches[1], chunks, indexer_indices=idx) is None
            want = [
                (c.slot_id, c.actual_start, c.actual_end)
                for c in chunks
                if not c.is_empty
                for _ in range(acks_per_chunk)
            ]
            got = _read_d2h_records(d2h, len(want))
            assert got == want, f"forward {f}: D2H records {got} != {want}"
        if traced:
            # Eager reference on the same two models (their host-scalar programs, not the captured ones), slots 4..7.
            for chunks in forwards:
                shifted = [
                    c if c.is_empty else PrefillChunk(c.slot_id + num_users, c.actual_start, c.actual_end)
                    for c in chunks
                ]
                inp = tokens(ranks[0], shifted)
                h = ranks[0].model.forward_packed(
                    inp, rank_caches[0].kvpe, shifted, index_kv_cache=rank_caches[0].index
                )
                ttnn.deallocate(inp)
                ranks[1].model.forward_packed(
                    h,
                    rank_caches[1].kvpe,
                    shifted,
                    index_kv_cache=rank_caches[1].index,
                    seed_indexer_indices=ranks[0].model.last_indexer_indices,
                )
                ttnn.deallocate(h)
        else:
            for chunks in forwards:
                ref.prefill_forward(tokens(ref, chunks), ref_caches, chunks)
        ttnn.synchronize_device(mesh_device)

        failures = []
        full_layers = [l for l, t in enumerate(config.indexer_types[:NUM_LAYERS]) if t == "full"]
        if not traced:
            ref_kv = _read_user_slots(ref_caches.kvpe, mesh_device, NUM_LAYERS, ref.config.cache_num_users)
            ref_idx = _read_user_slots(ref_caches.index, mesh_device, len(full_layers), ref.config.cache_num_users)
        for (first, n), r, caches in zip(split, ranks, rank_caches):
            n_idx = full_indexer_rank(config, first + n) - full_indexer_rank(config, first)
            kv = _read_user_slots(caches.kvpe, mesh_device, n, r.config.cache_num_users)
            idx = _read_user_slots(caches.index, mesh_device, n_idx, r.config.cache_num_users)
            for u in range(num_users):
                if traced:  # this rank's traced slot u against its own eager slot u + num_users
                    pairs = [("kvpe", kv[u][i], kv[u + num_users][i], first + i) for i in range(n)]
                    pairs += [("indexer_k", idx[u][i], idx[u + num_users][i], None) for i in range(n_idx)]
                else:  # this rank's slot u against the one-rank model's, at the same global layer
                    pairs = [("kvpe", kv[u][i], ref_kv[u][first + i], first + i) for i in range(n)]
                    pairs += [
                        ("indexer_k", idx[u][i], ref_idx[u][full_indexer_rank(config, first) + i], None)
                        for i in range(n_idx)
                    ]
                for name, got, want, layer in pairs:
                    diff = (got - want).abs().max().item()
                    pcc = _pcc(want, got)
                    logger.info(
                        f"[pipeline] rank@{first} slot {u} {name} layer {layer} max|diff| {diff:.3e} PCC {pcc:.6f}"
                    )
                    if (diff != 0) if not traced else (pcc < TRACED_PCC):
                        failures.append(
                            f"rank@{first} slot {u} {name} layer {layer}: max|diff| {diff:.3e} PCC {pcc:.6f}"
                        )
        assert not failures, f"[pipeline] {len(failures)} mismatch(es):\n" + "\n".join(failures)
    finally:
        # Every rank's traces first: each is released under the sub-device managers it was captured with, and the
        # models on this one mesh share TT_CCL's sparse-MLA overlap manager, which the first release_trace() below
        # removes. (In the runner each rank is its own process.) controller.release() is idempotent.
        for r in ranks:
            if r._controller is not None:
                r._controller.release()
        # Then each model's sub-device managers (and the D2H service a traced rank holds, whose device-side
        # teardown needs the mesh still open).
        for r in ranks + ([ref] if ref is not None else []):
            r.release_trace()
        d2h = None
