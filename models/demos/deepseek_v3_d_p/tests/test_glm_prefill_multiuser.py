# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Multi-user chunked prefill for GLM-5.3 on synthetic weights: the oracle for packed (batched) forwards.

A packed forward holds several chunks, possibly from different users at different depths. Attention runs
once per chunk against the chunk's own KV slot; the FFN half (ffn_norm, MoE) runs once over every chunk's
rows. Whatever the packing, each chunk's result must equal what that chunk gets alone. These tests pin
that down on device, device against device, so they need no checkpoint, ttnn cache or golden trace, and
they run on a LoudBox.

* test_glm_multiuser_interleaved_matches_sequential: one chunk per forward. N users prefilled chunk by
  chunk in round-robin order into one set of cache slots, and each user start to finish into another,
  must agree. Checks per-user state on the one-chunk path: KVPE and indexer-K slots, and the top-k that
  GLM-5.3 "shared" layers reuse from the last "full" layer.
* test_glm_packed_forward_matches_one_chunk_forwards: four chunks per forward (PrefillChunk, as in
  MiniMax-M3's packed forwards). A schedule with partial chunks, a partial chunk followed by a real one,
  one user's consecutive chunks in the same forward, users at different depths and empty tail chunks must
  match the same chunks run each alone in its own forward.
* test_glm_packed_traced_matches_eager: the same schedule through TtPrefillRuntime.prefill_forward with
  use_trace (one capture, replayed per forward; per-chunk metadata and the MoE padding config read on
  device) must fill the same KV as the eager packed forward.

Mesh and chunk: each mesh runs the chunk that gives the production per-chip shape, 640 rows per SP row
(2x4 -> 1,280-token chunks, 8x4 -> 5,120). On 2x4 every chip sees what a galaxy chip sees in attention
(640 rows, 16 heads, 160 sparse_sdpa tokens) but holds 32 experts instead of 8.

Depth: 4 layers is the smallest stack with a MoE (layers 0-2 are dense and own full indexers; layer 3 is
MoE and reuses layer 2's top-k). Random weights hold one MoE layer in host RAM, about 19 GB.
"""

import copy

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
# Packing changes nothing per chunk except which rows share the MoE call (and so gate top-k ties and
# expert tile blocking).
PACKED_PCC = 0.999
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
    config = copy.deepcopy(config)
    config.max_seq_len = seq_cache

    state_dict = glm_random_transformer_state_dict(config, GLM53Config, NUM_LAYERS)
    transformer = TtPrefillTransformer(
        mesh_device=mesh_device,
        config=config,
        model_cfg=GLM53Config,
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
def test_glm_packed_forward_matches_one_chunk_forwards(variant, config_only, mesh_device, device_params, num_links):
    """Packed forwards (slots 0..3) must reproduce every chunk run alone in its own forward (slots 4..7).

    Both sides use the same packed model, so the reference forward is the chunk followed by empty chunks:
    the comparison isolates what sharing a forward with other chunks does to a chunk.
    """
    assert_gate_mode_matches_adapter(variant, GATE_MODE)
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
                got = packed[(c.slot_id, c.actual_start)]
                pcc = _pcc(ref, got)
                logger.info(f"[packed] {c} last-layer PCC {pcc:.6f} max|diff| {(ref - got).abs().max().item():.3e}")
                if pcc < PACKED_PCC:
                    failures.append(f"{c} hidden PCC {pcc:.6f} < {PACKED_PCC}")

        kv_slots = _read_user_slots(kvpe_cache, mesh_device, NUM_LAYERS, num_slots)
        idx_slots = _read_user_slots(index_kv_cache, mesh_device, full_indexer_rank(config, NUM_LAYERS), num_slots)
        failures += _compare_slots(
            "packed", kv_slots, idx_slots, [(u + num_users, u) for u in range(num_users)], PACKED_PCC
        )

        cross = _pcc(packed[(0, 0)], packed[(3, 0)])
        assert cross < DISTINCT_USERS_MAX_PCC, f"users A and D produced the same output (PCC {cross:.6f})"
        assert not failures, f"[packed] {len(failures)} mismatch(es):\n" + "\n".join(failures)
    finally:
        transformer.release_sub_device_managers()


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
    config = copy.deepcopy(config_only)
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
        model_cfg=GLM53Config,
        routing_use_l1_small_for_semaphores=GLM53Adapter.routing_use_l1_small_for_semaphores,
        sparse_kv_cache_format=MlaKvCacheFormat.BF16_RM,
        use_trace=True,
        tp_shard_kv=True,
    )
    mesh_device.enable_program_cache()
    state_dict = glm_random_transformer_state_dict(config, GLM53Config, NUM_LAYERS)
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
        failures = _compare_slots(
            "traced", kv_slots, idx_slots, [(u + num_users, u) for u in range(num_users)], TRACED_PCC
        )
        cross = _pcc(kv_slots[0][NUM_LAYERS - 1], kv_slots[3][NUM_LAYERS - 1])
        assert cross < DISTINCT_USERS_MAX_PCC, f"users A and D hold the same KV (PCC {cross:.6f})"
        assert not failures, f"[traced] {len(failures)} mismatch(es):\n" + "\n".join(failures)
    finally:
        runtime.release_trace()
