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
* test_glm_packed_pad_rows_skip_the_moe: partial chunks ahead of non-empty ones, one at a rotated (mid-slab) start.
  Every MoE layer must route each real row and give each pad row the sentinel, and the real rows must not move with
  what the pad rows hold.
* test_glm_packed_mtp_matches_one_chunk_forwards: the same schedule with an MTP layer (one level, random weights)
  after the trunk. Each chunk's MTP input comes from its own ids and lookahead block, prompt-ending chunks generate
  their own token, and every user's trunk and MTP KV must match the chunk run alone.
* test_glm_packed_traced_matches_eager: the same schedule, and one with every count of real chunks, through
  TtPrefillRuntime.prefill_forward with use_trace (one capture per real-chunk count, the one matching each
  forward replayed; per-chunk metadata and the MoE padding config read on device) must fill the same KV as the
  eager packed forward, and nothing may compile after the captures.
* test_glm_packed_presend_matches_inline_push: the packed schedule with two shared layers in a row; the sparse-KV
  push's per-step pre-sends must leave every shared step only its rest and match pushing every row in line.
* test_glm_packed_mtp_traced_matches_eager: traced packed forwards with MTP through prefill_forward (per-chunk level
  geometries, both MTP captures replayed) must fill the same trunk and MTP KV as the eager packed MTP forward.
* test_glm_packed_pipeline_matches_one_rank (8x4 only): the model split over two pipeline ranks on one mesh,
  packed forwards handed rank to rank, must match one rank, eager and traced.

Mesh and chunk: each mesh runs the chunk that gives the production per-chip shape, 640 rows per SP chip
(2x4 -> 1,280-token chunks, 8x4 -> 5,120, a 4x4 stage -> 2,560 with SP on its axis-1 ring; the pipeline test is
8x4 only). On 2x4 every chip sees what a galaxy chip sees in attention
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
    torus_x_device_params,
    torus_xy_device_params,
)
from models.demos.deepseek_v3_d_p.tests.glm_synthetic_weights import glm_random_transformer_state_dict
from models.demos.deepseek_v3_d_p.tt.mla.indexer import full_indexer_rank
from models.demos.deepseek_v3_d_p.tt.mla.utils import blockcyclic_positions
from models.demos.deepseek_v3_d_p.tt.moe.init_helpers import mesh_shard_dims
from models.demos.deepseek_v3_d_p.tt.moe.tt_moe_gate_prefill import GateComputeMode, assert_gate_mode_matches_adapter
from models.demos.deepseek_v3_d_p.tt.moe_block.moe_block import to_mesh_order
from models.demos.deepseek_v3_d_p.tt.moe_block.moe_block_ffn import TtMoEBlockFfn, release_blocks
from models.demos.deepseek_v3_d_p.tt.moe_impl import BLOCK, moe_block_fabric_payload_size
from models.demos.deepseek_v3_d_p.tt.runners.adapters.glm_5_3 import GLM53Adapter
from models.demos.deepseek_v3_d_p.tt.runners.kv_caches import MlaKvCaches
from models.demos.deepseek_v3_d_p.tt.tt_ccl import per_axis_topology
from models.demos.deepseek_v3_d_p.tt.tt_prefill_runtime import TtPrefillRuntime, TtPrefillRuntimeConfig
from models.demos.deepseek_v3_d_p.tt.tt_prefill_transformer import TtPrefillTransformer
from models.demos.deepseek_v3_d_p.utils.chunk_config import PREFILL_CHUNK_TOKENS_PER_CHIP
from models.demos.deepseek_v3_d_p.utils.kv_cache_utils import MlaKvCacheFormat, init_kvpe_cache, init_mla_kv_cache
from models.demos.deepseek_v3_d_p.utils.test_utils import gather_cache_natural, unrotate_cache_layer
from models.demos.minimax_m3.tt.packed_rows import packed_real_row_mask, packed_routed_pad_rows
from tt_py_test_utils_common.utils_for_testing import comp_pcc

NUM_LAYERS = 4


class _TestGLM53Config(GLM53Config):
    """GLM-5.3 with two dense layers instead of three, so 4 layers hold every layer kind (module docstring)."""

    NUM_DENSE_LAYERS = 2


_TEST_INDEXER_TYPES = ["full", "full", "full", "shared"]


def _test_config(config, indexer_types=None):
    """A copy of ``config`` with the test's layer kinds; config_only is shared process-wide, so never mutate it."""
    config = copy.deepcopy(config)
    config.first_k_dense_replace = _TestGLM53Config.NUM_DENSE_LAYERS
    config.indexer_types = list(indexer_types or _TEST_INDEXER_TYPES)
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
# The same forwards with random tokens in the pad rows instead of token 0. Every pad row is masked out of the MoE and
# sits after its chunk's real rows (causal attention), so nothing a pad row holds reaches a real row: the layers before
# the first MoE are bit-exact, the rest at a bar well above the ~0.997 that routed pad rows give (#63555).
PAD_TOKENS_PCC = 0.9999
# Traced and eager run the same programs (the traced ones read their scalars on device), so the caches
# should agree exactly.
TRACED_PCC = 0.9999
GLM_TRACE_REGION_SIZE = 512 * 1024 * 1024
# The runner's L1_SMALL plus 512 B: these tests build several models on one device (packed and forward_chunks=1,
# traced and eager, pipeline ranks). At the runner's 1,536 B, the packed test's second build (forward_chunks=1) ran
# out of L1_SMALL on galaxy 8x4 after the 2026-11-06 main merge, though the first model was freed.
TEST_L1_SMALL_SIZE = GLM53Adapter.l1_small_size + 512
# GLM-5.3's MoE is the MoE block (tt/moe_impl.py). The test's config class (_TestGLM53Config) is not one the block
# knows by name, so the models here ask for it, and the meshes open with its fabric payload (one combine row).
MOE_IMPL = BLOCK
GLM_FABRIC_PAYLOAD = moe_block_fabric_payload_size(GLM53Config, moe_fabric_payload(GLM53Config), MOE_IMPL)

MESH_PARAMS = [
    pytest.param(
        (2, 4),
        fabric2d_device_params(fabric_payload_size=GLM_FABRIC_PAYLOAD, l1_small_size=TEST_L1_SMALL_SIZE),
        1,
        marks=pytest.mark.requires_mesh_topology(mesh_shape=(2, 4), topology="mesh-2x4"),
        id="fabric2d-2x4",
    ),
    pytest.param(
        (8, 4),
        torus_xy_device_params(fabric_payload_size=GLM_FABRIC_PAYLOAD, l1_small_size=TEST_L1_SMALL_SIZE),
        2,
        marks=pytest.mark.requires_mesh_topology(mesh_shape=(8, 4), topology="mesh-8x4"),
        id="torus-xy-8x4",
    ),
    # One 4x4 stage (a galaxy tray half, GLM-5.3's serving stage): SP on the axis-1 ring at its 2,560-token chunk,
    # TP=4 on the axis-0 line. Needs the galaxy narrowed to one tray half (TT_VISIBLE_DEVICES and the 4x4
    # descriptor); unnarrowed, requires_mesh_topology skips it.
    pytest.param(
        (4, 4),
        torus_x_device_params(fabric_payload_size=GLM_FABRIC_PAYLOAD, l1_small_size=TEST_L1_SMALL_SIZE),
        2,
        marks=pytest.mark.requires_mesh_topology(mesh_shape=(4, 4), topology="mesh-4x4"),
        id="torus-x-4x4-sp1",
    ),
]


@pytest.fixture(autouse=True)
def _release_moe_blocks(request):
    """Free the MoE blocks a test built (their L1_SMALL words, arena and DRAM) before its mesh closes."""
    mesh = request.getfixturevalue("mesh_device") if "mesh_device" in request.fixturenames else None
    yield
    if mesh is not None:
        release_blocks(mesh)


def _sp_axis(mesh_device):
    """The mesh's SP axis: 1 on a 4x4 stage (SP on the tray half's ring), else 0."""
    return 1 if tuple(mesh_device.shape) == (4, 4) else 0


def _traced_mesh_params(l1_small_extra: int = 0) -> list:
    """MESH_PARAMS with a trace region; ``l1_small_extra`` grows L1_SMALL as the runner does for MTP."""
    l1_small = TEST_L1_SMALL_SIZE + l1_small_extra
    params = [
        pytest.param(
            (2, 4),
            fabric2d_device_params(
                fabric_payload_size=GLM_FABRIC_PAYLOAD,
                l1_small_size=l1_small,
                trace_region_size=GLM_TRACE_REGION_SIZE,
            ),
            1,
            marks=pytest.mark.requires_mesh_topology(mesh_shape=(2, 4), topology="mesh-2x4"),
            id="fabric2d-2x4",
        ),
        pytest.param(
            (8, 4),
            torus_xy_device_params(
                fabric_payload_size=GLM_FABRIC_PAYLOAD,
                l1_small_size=l1_small,
                trace_region_size=GLM_TRACE_REGION_SIZE,
            ),
            2,
            marks=pytest.mark.requires_mesh_topology(mesh_shape=(8, 4), topology="mesh-8x4"),
            id="torus-xy-8x4",
        ),
    ]
    params.append(
        pytest.param(
            (4, 4),
            torus_x_device_params(
                fabric_payload_size=GLM_FABRIC_PAYLOAD,
                l1_small_size=l1_small,
                trace_region_size=GLM_TRACE_REGION_SIZE,
            ),
            2,
            marks=pytest.mark.requires_mesh_topology(mesh_shape=(4, 4), topology="mesh-4x4"),
            id="torus-x-4x4-sp1",
        )
    )
    return params


TRACED_MESH_PARAMS = _traced_mesh_params()
# L1_SMALL headroom for the traced MTP test, which builds a traced and an eager MTP model on one device. The runner adds
# 512 B for MTP (prefill_runner._L1_SMALL_SIZE). Since #64657 every sparse-MLA q_a high_bw_all_gather program holds
# its own global semaphores in L1_SMALL, so the two models need more: 3,072 B in all ran out on galaxy 8x4, 4,096 B
# fits.
MTP_L1_SMALL_EXTRA = 2048
# Packed MTP runs on galaxy 8x4, a 4x4 stage and LoudBox (CI selects the 8x4 rows).
MTP_MESH_PARAMS = MESH_PARAMS
TRACED_MTP_MESH_PARAMS = _traced_mesh_params(MTP_L1_SMALL_EXTRA)


def _build(mesh_device, config, device_params, num_links, chunk, num_slots, forward_chunks=1, indexer_types=None):
    """Random-weight chunked transformer plus KVPE and indexer-K caches with ``num_slots`` user slots."""
    sp_axis = _sp_axis(mesh_device)
    tp_axis = 1 - sp_axis
    mesh_shape = list(mesh_device.shape)
    seq_cache = N_CHUNKS * chunk
    # _resolve_config_only is lru_cached and shared process-wide; don't leak max_seq_len into it.
    config = _test_config(config, indexer_types)
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
        moe_impl=MOE_IMPL,
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


def _forward_input(mesh_device, chunks, ids_by_slot, chunk, pad_ids=None):
    """A forward's tokens in the packed per-chip layout (pack_forward_tokens), SP-sharded on device. A
    partial chunk's tail and an empty chunk are padded with token 0, or with ``pad_ids`` (one id per pad row)."""
    sp_axis = _sp_axis(mesh_device)
    sp = mesh_device.shape[sp_axis]
    chunk_tokens = []
    pads = iter(pad_ids.tolist()) if pad_ids is not None else None
    for c in chunks:
        real = [] if c.is_empty else ids_by_slot[c.slot_id][c.actual_start : c.actual_end].tolist()
        pad = [0] * (chunk - len(real)) if pads is None else [next(pads) for _ in range(chunk - len(real))]
        chunk_tokens.append(real + pad)
    packed = pack_forward_tokens(chunk_tokens, [c.actual_start for c in chunks], sp)
    return ttnn.from_torch(
        torch.tensor(packed, dtype=torch.int64).reshape(sp, 1, len(chunks) * chunk // sp),
        device=mesh_device,
        dtype=ttnn.uint32,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        mesh_mapper=ttnn.ShardTensor2dMesh(
            mesh_device, mesh_shape=tuple(mesh_device.shape), dims=mesh_shard_dims(0, sp_axis)
        ),
    )


def _run_forward(transformer, kvpe_cache, index_kv_cache, mesh_device, chunks, ids_by_slot, chunk, pad_ids=None):
    """One forward of ``chunks``; returns the last layer's output on host, [1, len(chunks) * chunk, emb],
    in the packed per-chip row order. A one-chunk forward goes through the regular (non-packed) path."""
    tt_tokens = _forward_input(mesh_device, chunks, ids_by_slot, chunk, pad_ids)
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
    rows, _ = gather_cache_natural(cache, mesh_device, tp_shard_kv=True, sp_axis=_sp_axis(mesh_device))
    assert rows.shape[0] == num_slots * layers_per_user, (rows.shape, num_slots, layers_per_user)
    return [rows[s * layers_per_user : (s + 1) * layers_per_user] for s in range(num_slots)]


def _read_real_rows(cache, mesh_device, layers_per_user, chunk, lengths):
    """Per slot, per layer, the cache rows of positions [0, lengths[slot]) in natural order: what a later chunk reads.
    An eager forward without zero_pad leaves its pad rows' KV past actual_end, which nothing reads."""
    rows, stripes = gather_cache_natural(cache, mesh_device, tp_shard_kv=True, sp_axis=_sp_axis(mesh_device))
    positions = blockcyclic_positions(stripes, chunk, rows.shape[1])
    return [
        [unrotate_cache_layer(rows[s * layers_per_user + l], positions, n) for l in range(layers_per_user)]
        for s, n in enumerate(lengths)
    ]


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
    chunk = PREFILL_CHUNK_TOKENS_PER_CHIP * mesh_device.shape[_sp_axis(mesh_device)]
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
    slot u. Covers a partial chunk followed by a real one (its pad rows masked per row), one user's
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
    the comparison isolates what sharing a forward with other chunks does to a chunk. A second model built for
    one chunk per forward (forward_chunks=1, the regular path) then runs the same chunks, and the packed model's
    one-chunk forwards must match it: that catches an error common to every packed forward (its row layout,
    the MoE sized for FORWARD_CHUNKS chunks, padding), which the first comparison cannot see.
    """
    assert_gate_mode_matches_adapter(variant, GATE_MODE)
    sp = mesh_device.shape[_sp_axis(mesh_device)]
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
    release_blocks(mesh_device)  # (and its MoE block's L1_SMALL words / arena: the one-chunk model builds its own)

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


# A real row with a zero route word: its top-k experts all live on its own chip (8 or 16 of GLM-5.3's 256 here), so
# well under 1% of rows; more means real rows were dropped.
MAX_LOCAL_ONLY_REAL_ROWS = 0.01


def _capture_packed_counts(monkeypatch):
    """Wrap the MoE block's FFN to read its routing after every eager packed MoE call. One entry per MoE call:

    - ``total``: how many (row, expert) picks the routing planner counted, each chip's slice of the head's per-expert
      counts (its own experts) summed over the mesh. Dispatch, the routed experts and the combine run that plan.
    - per row (packed_real_row_mask), from the head's route words: one per owned row, naming the other chips the row
      is dispatched to, mapped back to its SP row's packed order. ``routed_pad_rows``: pad rows with a nonzero word,
      which must be none. ``unrouted_real_rows``: real rows with a zero word. A real row whose top-k experts all live
      on its own chip has one too, so these are counted rather than failed one by one; the pick total stays exact."""
    calls = []
    call = TtMoEBlockFfn.__call__

    def capture(self, x, *args, chunks=None, forward_chunks=None, **kwargs):
        assert chunks is not None, "this hook reads eager packed forwards only"
        y = call(self, x, *args, chunks=chunks, forward_chunks=forward_chunks, **kwargs)
        shape = tuple(self.mesh.shape)
        R, C = shape[self.options.sp_axis], shape[1 - self.options.sp_axis]  # SP rows, TP chips a row
        n_chips = self.mesh.get_num_devices()
        rows = int(x.shape[-2])  # an SP row's packed rows (every chunk's chunk_local rows back to back)
        F = forward_chunks if forward_chunks is not None else len(chunks)
        block = self._block(rows * R, F)
        assert block.splits == 1, "the routing check reads one head call per MoE call"
        head = block.last_head
        epc = self.dims.num_experts // n_chips
        # (device i holds the block's chip chips[i]: the identity with SP on axis 0, transposed on a 4x4 sp1 stage)
        chips = to_mesh_order(list(range(n_chips)), self.mesh.shape, self.options.sp_axis)
        total = 0
        for i, d in enumerate(ttnn.get_device_tensors(head.counts)):
            n = chips[i]
            total += int(ttnn.to_torch(d).reshape(-1)[n * epc : (n + 1) * epc].to(torch.int64).sum())

        # Block chip n = r C + p (SP row r, TP position p) owns, in each of the head's k row blocks, s rows of its SP
        # row: block b's rows [b s C + p s, b s C + (p + 1) s) (moe_head README, Row blocks).
        rw = ttnn._ttnn.operations.experimental.moe_block.route_words(n_chips)
        k = int(head.row_blocks)
        mask = packed_real_row_mask(chunks, R, rows // F)
        routed = [[None] * rows for _ in range(R)]
        for i, d in enumerate(ttnn.get_device_tensors(head.route_words)):
            r, p = divmod(chips[i], C)
            words = ttnn.to_torch(d).reshape(-1).to(torch.int64) & 0xFFFFFFFF
            owned = words.numel() // rw
            assert owned * C == rows and owned % k == 0, (owned, C, rows, k)
            s = owned // k
            # A row's destination chips: the low 16 bits of its one word, or the first of its two words past 16 chips.
            dests = words[0::rw] & 0xFFFF if rw == 1 else words[0::rw]
            for o in range(owned):
                b, t = divmod(o, s)
                routed[r][b * s * C + p * s + t] = bool(dests[o])
        cells = [(r, j) for r in range(R) for j in range(rows)]
        calls.append(
            {
                "total": total,
                "real_rows": sum(int(mask[r][j]) for r, j in cells),
                "routed_pad_rows": [(r, j) for r, j in cells if routed[r][j] and not mask[r][j]],
                "unrouted_real_rows": [(r, j) for r, j in cells if mask[r][j] and not routed[r][j]],
            }
        )
        return y

    monkeypatch.setattr(TtMoEBlockFfn, "__call__", capture)
    return calls


@pytest.mark.parametrize(
    "mesh_device, device_params, num_links", MESH_PARAMS, indirect=["mesh_device", "device_params"]
)
@pytest.mark.parametrize("variant", ["glm_5_3"], indirect=True, ids=["glm53"])
@pytest.mark.skipif(not is_blackhole(), reason="GLM DSA ops (indexer / sparse SDPA) are Blackhole-only")
@pytest.mark.timeout(0)
def test_glm_packed_pad_rows_skip_the_moe(variant, config_only, mesh_device, device_params, num_links, monkeypatch):
    """Partial chunks ahead of non-empty ones in a packed forward: their pad rows sit between two chunks' real rows on
    every chip, so only the forward's chunk bounds (MoEBlockPacking) keep them out of the MoE block's experts.

    Forward 0 puts user A's first turn (896 tokens) ahead of B's and C's full chunks; forward 1 puts A's second turn,
    which starts mid-slab at 896 (a rotated chunk), and B's partial chunk ahead of C's full one. Every MoE layer's
    routing planner must count exactly the real rows' picks (real rows x NUM_EXPERTS_PER_TOKEN): a routed pad row or a
    dropped real row changes the total. Then the same forwards run in fresh slots with random tokens in the pad rows:
    the dense layers' KV and indexer-K must be bit-exact and the rest within PAD_TOKENS_PCC."""
    assert_gate_mode_matches_adapter(variant, GATE_MODE)
    sp = mesh_device.shape[_sp_axis(mesh_device)]
    chunk = PREFILL_CHUNK_TOKENS_PER_CHIP * sp
    chunk_local = chunk // sp
    first, second = 896, chunk - 704  # A's two turns; the second starts at 896, mid-slab
    E = PrefillChunk(0, 0, 0)
    forwards = [
        [PrefillChunk(0, 0, first), PrefillChunk(1, 0, chunk), PrefillChunk(2, 0, chunk), E],
        [
            PrefillChunk(0, first, first + second),
            PrefillChunk(1, chunk, chunk + 704),
            PrefillChunk(2, chunk, 2 * chunk),
            E,
        ],
    ]
    num_users = 3
    num_slots = 2 * num_users
    mesh_device.enable_program_cache()
    config, transformer, kvpe_cache, index_kv_cache = _build(
        mesh_device, config_only, device_params, num_links, chunk, num_slots, forward_chunks=FORWARD_CHUNKS
    )
    calls = _capture_packed_counts(monkeypatch)
    try:
        for chunks in forwards:
            validate_chunks(
                chunks,
                chunk_size=chunk,
                forward_chunks=FORWARD_CHUNKS,
                num_users=num_slots,
                max_seq_len=N_CHUNKS * chunk,
            )
            assert packed_routed_pad_rows(chunks, sp, chunk_local) > 0, f"{chunks} has no pad rows ahead of real ones"
        ids = _token_ids(config, num_users, 2 * chunk)
        ids_by_slot = {s: ids[s % num_users] for s in range(num_slots)}

        failures = []
        outputs = []
        for f, chunks in enumerate(forwards):
            calls.clear()
            hidden = _run_forward(transformer, kvpe_cache, index_kv_cache, mesh_device, chunks, ids_by_slot, chunk)
            outputs.append(
                {
                    k: _chunk_real_rows(hidden, k, FORWARD_CHUNKS, c, sp, chunk_local)
                    for k, c in enumerate(chunks)
                    if not c.is_empty
                }
            )
            real = sum(c.actual_end - c.actual_start for c in chunks)
            topk = _TestGLM53Config.NUM_EXPERTS_PER_TOKEN
            assert calls, "no MoE layer ran"
            for layer, got in enumerate(calls):
                total, pad_routed, real_unrouted = got["total"], got["routed_pad_rows"], got["unrouted_real_rows"]
                logger.info(
                    f"[pad rows] forward {f} MoE call {layer}: planner counted {total} picks, "
                    f"{real} real rows x {topk} = {real * topk}; routed pad rows {len(pad_routed)}, "
                    f"real rows with no remote expert {len(real_unrouted)} of {got['real_rows']}"
                )
                if total != real * topk:
                    failures.append(f"forward {f} MoE call {layer}: {total} picks counted, want {real * topk}")
                if got["real_rows"] != real:
                    failures.append(f"forward {f} MoE call {layer}: {got['real_rows']} rows masked real, want {real}")
                if pad_routed:
                    failures.append(
                        f"forward {f} MoE call {layer}: {len(pad_routed)} pad rows dispatched "
                        f"(SP row, row): {pad_routed[:8]}"
                    )
                if len(real_unrouted) > MAX_LOCAL_ONLY_REAL_ROWS * real:
                    failures.append(
                        f"forward {f} MoE call {layer}: {len(real_unrouted)} of {real} real rows not dispatched "
                        f"(SP row, row): {real_unrouted[:8]}"
                    )
        assert not failures, f"[pad rows] {len(failures)} routing error(s):\n" + "\n".join(failures)
        monkeypatch.undo()

        # The same forwards in slots 3..5 with random pad tokens: the real rows must not move.
        gen = torch.Generator().manual_seed(7)
        for f, chunks in enumerate(forwards):
            moved = [
                c if c.is_empty else PrefillChunk(c.slot_id + num_users, c.actual_start, c.actual_end) for c in chunks
            ]
            pad_ids = torch.randint(1, config.vocab_size, (FORWARD_CHUNKS * chunk,), generator=gen)
            hidden = _run_forward(
                transformer, kvpe_cache, index_kv_cache, mesh_device, moved, ids_by_slot, chunk, pad_ids
            )
            for k, ref in outputs[f].items():
                got = _chunk_real_rows(hidden, k, FORWARD_CHUNKS, moved[k], sp, chunk_local)
                pcc, diff = _pcc(ref, got), (ref - got).abs().max().item()
                logger.info(f"[pad rows] forward {f} {moved[k]} last-layer PCC {pcc:.6f} max|diff| {diff:.3e}")
                if pcc < PAD_TOKENS_PCC:
                    failures.append(f"forward {f} {moved[k]}: output PCC {pcc:.6f} < {PAD_TOKENS_PCC} with other pads")

        lengths = [first + second, chunk + 704, 2 * chunk] * 2  # per slot: slots 3..5 mirror 0..2
        kv_slots = _read_real_rows(kvpe_cache, mesh_device, NUM_LAYERS, chunk, lengths)
        idx_slots = _read_real_rows(index_kv_cache, mesh_device, full_indexer_rank(config, NUM_LAYERS), chunk, lengths)
        exact_through = _TestGLM53Config.NUM_DENSE_LAYERS  # KV of layer L comes from layer L-1's output
        full_layers = [l for l, t in enumerate(config.indexer_types[:NUM_LAYERS]) if t == "full"]
        for u in range(num_users):
            ref_slot, got_slot = u, u + num_users
            for name, slots, layers in (("kvpe", kv_slots, range(NUM_LAYERS)), ("indexer_k", idx_slots, full_layers)):
                for rank, layer in enumerate(layers):
                    ref, got = slots[ref_slot][rank], slots[got_slot][rank]
                    pcc, diff = _pcc(ref, got), (ref - got).abs().max().item()
                    logger.info(
                        f"[pad rows] slot {got_slot} vs {ref_slot} {name}[{layer}] PCC {pcc:.6f} max|diff| {diff:.3e}"
                    )
                    if layer <= exact_through and diff != 0:
                        failures.append(
                            f"slot {got_slot} {name} layer {layer} moved with the pads (max|diff| {diff:.3e})"
                        )
                    elif pcc < PAD_TOKENS_PCC:
                        failures.append(f"slot {got_slot} {name} layer {layer} PCC {pcc:.6f} < {PAD_TOKENS_PCC}")
        assert not failures, f"[pad rows] {len(failures)} mismatch(es):\n" + "\n".join(failures)
    finally:
        transformer.release_sub_device_managers()


# Two shared layers in a row, so the sparse-KV push hands a pre-send from a full layer to a shared one, within a shared
# layer from chunk to chunk, and from one shared layer to the next.
_PRESEND_INDEXER_TYPES = ["full", "full", "shared", "shared"]


@pytest.mark.parametrize(
    "mesh_device, device_params, num_links", MESH_PARAMS, indirect=["mesh_device", "device_params"]
)
@pytest.mark.parametrize("variant", ["glm_5_3"], indirect=True, ids=["glm53"])
@pytest.mark.skipif(not is_blackhole(), reason="GLM DSA ops (indexer / sparse SDPA) are Blackhole-only")
@pytest.mark.timeout(0)
def test_glm_packed_presend_matches_inline_push(variant, config_only, mesh_device, device_params, num_links):
    """Packed forwards pre-send each shared attention step's prefix rows beside the step before it (#62997's push,
    per step): with pre-sends (slots 0..3) every shared step pushes only its rest, and the KV caches and outputs must
    match the same forwards pushing every row in line (pre-sends off, slots 4..7)."""
    assert_gate_mode_matches_adapter(variant, GATE_MODE)
    sp = mesh_device.shape[_sp_axis(mesh_device)]
    chunk = PREFILL_CHUNK_TOKENS_PER_CHIP * sp
    chunk_local = chunk // sp
    lengths, forwards = _packed_schedule(chunk)
    num_users = len(lengths)
    num_slots = 2 * num_users
    mesh_device.enable_program_cache()
    config, transformer, kvpe_cache, index_kv_cache = _build(
        mesh_device,
        config_only,
        device_params,
        num_links,
        chunk,
        num_slots,
        forward_chunks=FORWARD_CHUNKS,
        indexer_types=_PRESEND_INDEXER_TYPES,
    )
    planner = transformer.layers[2].mla._push
    try:
        if planner is None:
            pytest.skip("the sparse-KV push does not support this mesh; shared layers gather the whole prefix")
        ids = _token_ids(config, num_users, max(lengths))
        ids_by_slot = {s: ids[s % num_users] for s in range(num_slots)}
        shared_layers = sum(t == "shared" for t in _PRESEND_INDEXER_TYPES)
        real_chunks = sum(not c.is_empty for chunks in forwards for c in chunks)
        shared_steps = shared_layers * real_chunks

        hidden, pushes, builds = {}, {}, {}
        for presend, offset in ((True, 0), (False, num_users)):
            planner.presend = presend
            planner.pushes.clear()
            planner.builds = 0
            for chunks in forwards:
                chunks = [
                    c if c.is_empty else PrefillChunk(c.slot_id + offset, c.actual_start, c.actual_end) for c in chunks
                ]
                out = _run_forward(transformer, kvpe_cache, index_kv_cache, mesh_device, chunks, ids_by_slot, chunk)
                for k, c in enumerate(chunks):
                    if not c.is_empty:
                        hidden[(c.slot_id, c.actual_start)] = _chunk_real_rows(
                            out, k, FORWARD_CHUNKS, c, sp, chunk_local
                        )
            pushes[presend] = dict(planner.pushes)
            builds[presend] = planner.builds
            logger.info(
                f"[presend={presend}] pushes {pushes[presend]}, {builds[presend]} plans built over {shared_steps} "
                "shared steps"
            )
        planner.presend = True
        # Every shared step got its prefix from the step before it (the first from the last full layer's last chunk).
        assert pushes[True] == {"PREFIX": shared_steps, "REST": shared_steps}, pushes[True]
        assert pushes[False] == {"ALL": shared_steps}, pushes[False]
        # One plan per chunk of each forward, reused by every shared layer (not rebuilt chunk after chunk per layer).
        assert builds == {True: real_chunks, False: real_chunks}, builds

        failures = []
        for (slot, start), got in hidden.items():
            if slot >= num_users:
                continue
            ref = hidden[(slot + num_users, start)]
            pcc, diff = _pcc(ref, got), (ref - got).abs().max().item()
            logger.info(f"[presend] slot {slot} @{start} last-layer PCC {pcc:.6f} max|diff| {diff:.3e}")
            if pcc < INTERLEAVED_PCC:
                failures.append(f"slot {slot} @{start} hidden PCC {pcc:.6f} < {INTERLEAVED_PCC}")
        kv_slots = _read_user_slots(kvpe_cache, mesh_device, NUM_LAYERS, num_slots)
        idx_slots = _read_user_slots(index_kv_cache, mesh_device, full_indexer_rank(config, NUM_LAYERS), num_slots)
        # KV through the first shared layer comes from layers that read no pushed rows, so it must not move at all.
        first_shared = _PRESEND_INDEXER_TYPES.index("shared")
        for u in range(num_users):
            for name, slots in (("kvpe", kv_slots), ("indexer_k", idx_slots)):
                for rank in range(slots[u].shape[0]):
                    ref, got = slots[u + num_users][rank], slots[u][rank]
                    diff = (ref - got).abs().max().item()
                    pcc = _pcc(ref, got)
                    logger.info(f"[presend] slot {u} {name}[{rank}] PCC {pcc:.6f} max|diff| {diff:.3e}")
                    if name == "kvpe" and rank <= first_shared and diff != 0:
                        failures.append(f"slot {u} kvpe layer {rank} not bit-exact (max|diff| {diff:.3e})")
                    elif pcc < INTERLEAVED_PCC:
                        failures.append(f"slot {u} {name}[{rank}] PCC {pcc:.6f} < {INTERLEAVED_PCC}")
        assert not failures, f"[presend] {len(failures)} mismatch(es):\n" + "\n".join(failures)
    finally:
        if planner is not None:
            planner.presend = True
        transformer.release_sub_device_managers()


# MTP levels the packed-MTP tests run: one, as serving GLM-5.3 does (glm53_mtp_kv_share.json), and four
# (glm53_mtp4.json), where each level's input is the level below's output.
MTP_LEVELS = pytest.mark.parametrize("mtp_levels", [1, 4], ids=["levels1", "levels4"])
# Packed vs one chunk per forward, MTP KV per level: each level's input is the level below's output, so the MoE's
# per-call rounding (#63555) compounds level by level. Measured minimum over users on galaxy 8x4 (job 196294): 0.99886,
# 0.99645, 0.99266, 0.98794; a misplaced window or generated token scores far lower.
PACKED_MTP_LEVEL_KV_PCC = (PACKED_KV_PCC, 0.994, 0.989, 0.983)


def _mtp_test_weights(config, levels):
    """Random weights for a 4-layer trunk with MTP: ``(trunk state dict with the norm / LM head tail MTP needs, the
    MTP predictor's state dict, its MTPConfig)``. The MTP decoder layer is layer NUM_LAYERS, an MoE layer sharing the
    trunk's experts."""
    import dataclasses

    from models.demos.deepseek_v3_d_p.tt.mtp_prefill.mtp_config import MTPConfig

    hidden = config.hidden_size
    g = torch.Generator().manual_seed(77)

    # One more layer than the trunk: its weights are the MTP decoder layer's (an MoE layer, sharing the experts).
    state_dict = glm_random_transformer_state_dict(config, _TestGLM53Config, NUM_LAYERS + 1, share_experts=True)
    mtp_layer = state_dict["layers"].pop()
    state_dict["norm_weight"] = (torch.randn(hidden, generator=g) * 0.1 + 1.0).to(torch.bfloat16)
    state_dict["lm_head_weight"] = (torch.randn(config.vocab_size, hidden, generator=g) * hidden**-0.5).to(
        torch.bfloat16
    )
    mtp_cfg = dataclasses.replace(
        MTPConfig.from_hf_config(config, num_levels=levels),
        mtp_layer_idx=NUM_LAYERS,
        first_k_dense_replace=_TestGLM53Config.NUM_DENSE_LAYERS,
    )
    mtp_weights = {
        "enorm": (torch.randn(hidden, generator=g) * 0.1 + 1.0).to(torch.bfloat16),
        "hnorm": (torch.randn(hidden, generator=g) * 0.1 + 1.0).to(torch.bfloat16),
        "eh_proj": (torch.randn(hidden, 2 * hidden, generator=g) * (2 * hidden) ** -0.5).to(torch.bfloat16),
        "shared_head_norm": (torch.randn(hidden, generator=g) * 0.1 + 1.0).to(torch.bfloat16),
    }
    return state_dict, {"mtp": mtp_weights, "layer": mtp_layer}, mtp_cfg


def _build_mtp(mesh_device, config, device_params, num_links, chunk, num_slots, forward_chunks, levels):
    """:func:`_build` with a random-weight MTP predictor (layer NUM_LAYERS, ``levels`` levels) on the transformer, the
    norm / LM head tail MTP needs, and caches widened by the MTP layer."""
    from models.demos.deepseek_v3_d_p.tt.mtp_prefill.tt_mtp import TtMTPPredictor
    from models.demos.deepseek_v3_d_p.tt.mtp_prefill.utils import enable_mtp_indexer_slot

    sp_axis = _sp_axis(mesh_device)
    tp_axis = 1 - sp_axis
    mesh_shape = list(mesh_device.shape)
    seq_cache = N_CHUNKS * chunk
    config = _test_config(config)
    config.max_seq_len = seq_cache
    enable_mtp_indexer_slot(config, NUM_LAYERS)
    topology = per_axis_topology(device_params["fabric_config"])
    state_dict, mtp_state, mtp_cfg = _mtp_test_weights(config, levels)
    common = dict(
        dispatch_buffer_capacity_factor=8,
        num_links=num_links,
        topology=topology,
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
        moe_impl=MOE_IMPL,
    )
    predictor = TtMTPPredictor(
        mesh_device,
        config,
        _TestGLM53Config,
        mtp_state,
        mtp_cfg,
        seq_len=chunk,
        num_levels=levels,
        layer_idx=NUM_LAYERS,
        first_cache_slot=NUM_LAYERS,
        layer_num=NUM_LAYERS + levels,
        **common,
    )
    transformer = TtPrefillTransformer(
        mesh_device=mesh_device,
        config=config,
        model_cfg=_TestGLM53Config,
        state_dict=state_dict,
        num_layers=NUM_LAYERS,
        seq_len=chunk,
        is_balanced=False,
        padding_side="right",
        lm_head_is_column_parallel=True,
        mtp_predictor=predictor,
        **common,
    )
    del state_dict, mtp_state
    assert transformer.num_kvpe_cache_layers == NUM_LAYERS + levels

    kvpe_cache = init_mla_kv_cache(
        cache_format=MlaKvCacheFormat.BF16_RM,
        hf_config=config,
        mesh_device=mesh_device,
        seq_len=seq_cache,
        mesh_shape=mesh_shape,
        sp_axis=sp_axis,
        tp_axis=tp_axis,
        num_kvpe_cache_layers=NUM_LAYERS + levels,
        num_users=num_slots,
    )
    index_kv_cache = init_kvpe_cache(
        kvpe_cache_head_dim=config.index_head_dim,
        mesh_device=mesh_device,
        seq_len=seq_cache,
        mesh_shape=mesh_shape,
        sp_axis=sp_axis,
        tp_axis=tp_axis,
        num_kvpe_cache_layers=full_indexer_rank(config, NUM_LAYERS + levels),
        num_users=num_slots,
        dtype=ttnn.bfloat8_b,
    )
    return config, transformer, kvpe_cache, index_kv_cache


def _mtp_chunk_inputs(c, ids_by_slot, lengths, chunk, sp, levels):
    """One chunk's MTP inputs as the producer and runner build them: ``(trunk ids, [sp, num_mtp_tokens] lookahead
    block, provided levels)``. Positions at or past the user's prompt length are MTP_PAD_TOKEN_ID."""
    from models.demos.common.prefill.runners.runner_utils import MTP_PAD_TOKEN_ID, num_mtp_tokens
    from models.demos.deepseek_v3_d_p.tt.mla.utils import mtp_lookahead_positions

    mtp = torch.full((sp, num_mtp_tokens(levels)), MTP_PAD_TOKEN_ID, dtype=torch.int64)
    if c.is_empty:
        return [0] * chunk, mtp, 0
    ids, length = ids_by_slot[c.slot_id], lengths[c.slot_id]

    def token(p):
        return int(ids[p]) if p < length else MTP_PAD_TOKEN_ID

    real = [token(p) if p < c.actual_end else 0 for p in range(c.actual_start, c.actual_start + chunk)]
    slots = mtp_lookahead_positions(c.actual_start, sp, chunk // sp, c.actual_end, levels)
    for chip, positions in enumerate(slots):
        mtp[chip, : len(positions)] = torch.tensor([token(p) for p in positions], dtype=torch.int64)
    # The runner's count: real lookahead ids on the chip holding the chunk's last position; 0 when partial.
    provided = 0
    if c.actual_end == c.actual_start + chunk:
        provided = sum(1 for p in range(c.actual_end, c.actual_end + levels) if p < length)
    return real, mtp, provided


def _mtp_forward(transformer, kvpe_cache, index_kv_cache, mesh_device, chunks, ids_by_slot, lengths, chunk, levels):
    """One eager packed forward of ``chunks`` with MTP, called on the transformer directly, its inputs from
    _mtp_chunk_inputs. Returns the chunks' provided levels."""
    from models.demos.deepseek_v3_d_p.tt.mtp_prefill.device_windows import MTPUnionEmbedding

    sp_axis = _sp_axis(mesh_device)
    sp = mesh_device.shape[sp_axis]

    def upload(rows):  # [sp, n] int64 -> [sp, 1, n] uint32, one row per SP chip
        return ttnn.from_torch(
            rows.reshape(sp, 1, -1).to(torch.int32),
            device=mesh_device,
            dtype=ttnn.uint32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.ShardTensor2dMesh(
                mesh_device, mesh_shape=tuple(mesh_device.shape), dims=mesh_shard_dims(0, sp_axis)
            ),
        )

    unions, provided_levels = [], []
    for c in chunks:
        real, mtp, provided = _mtp_chunk_inputs(c, ids_by_slot, lengths, chunk, sp, levels)
        provided_levels.append(provided)
        trunk = torch.tensor(pack_forward_tokens([real], [c.actual_start], sp), dtype=torch.int64).reshape(sp, -1)
        tt_trunk, tt_mtp = upload(trunk), upload(mtp)
        unions.append(MTPUnionEmbedding.from_ids(tt_trunk, tt_mtp, transformer.mtp_embed_ids, num_levels=levels))
        ttnn.deallocate(tt_trunk)
        ttnn.deallocate(tt_mtp)
    model_input = ttnn.concat([u.trunk for u in unions], dim=2)
    transformer.forward(
        model_input,
        kvpe_cache,
        actual_isl=chunk,
        chunks=chunks,
        index_kv_cache=index_kv_cache,
        mtp_unions=unions,
        provided_levels=provided_levels,
        input_is_embedded=True,
    )
    ttnn.synchronize_device(mesh_device)
    ttnn.deallocate(model_input)
    for u in unions:
        u.deallocate()
    return provided_levels


@pytest.mark.parametrize(
    "mesh_device, device_params, num_links", MTP_MESH_PARAMS, indirect=["mesh_device", "device_params"]
)
@pytest.mark.parametrize("variant", ["glm_5_3"], indirect=True, ids=["glm53"])
@pytest.mark.skipif(not is_blackhole(), reason="GLM DSA ops (indexer / sparse SDPA) are Blackhole-only")
@MTP_LEVELS
@pytest.mark.timeout(0)
def test_glm_packed_mtp_matches_one_chunk_forwards(
    variant, config_only, mesh_device, device_params, num_links, mtp_levels
):
    """Packed forwards with MTP (slots 0..3) must fill every user's trunk and MTP KV as each chunk does alone in its
    own forward (slots 4..7), with prompt-ending chunks mid-forward, real and all-pad lookahead, and empty tails.
    Bars as test_glm_packed_forward_matches_one_chunk_forwards (the MoE sees different rows per call, #63555)."""
    assert_gate_mode_matches_adapter(variant, GATE_MODE)
    sp = mesh_device.shape[_sp_axis(mesh_device)]
    chunk = PREFILL_CHUNK_TOKENS_PER_CHIP * sp
    lengths, forwards = _packed_schedule(chunk)
    num_users = len(lengths)
    num_slots = 2 * num_users
    mesh_device.enable_program_cache()
    config, transformer, kvpe_cache, index_kv_cache = _build_mtp(
        mesh_device, config_only, device_params, num_links, chunk, num_slots, FORWARD_CHUNKS, mtp_levels
    )
    try:
        ids = _token_ids(config, num_users, N_CHUNKS * chunk)
        ids_by_slot = {s: ids[s % num_users] for s in range(num_slots)}
        lengths_by_slot = {s: lengths[s % num_users] for s in range(num_slots)}
        empty = PrefillChunk(0, 0, 0)
        for chunks in forwards:
            validate_chunks(
                chunks,
                chunk_size=chunk,
                forward_chunks=FORWARD_CHUNKS,
                num_users=num_slots,
                max_seq_len=N_CHUNKS * chunk,
            )
            levels = _mtp_forward(
                transformer,
                kvpe_cache,
                index_kv_cache,
                mesh_device,
                chunks,
                ids_by_slot,
                lengths_by_slot,
                chunk,
                mtp_levels,
            )
            logger.info(f"[packed mtp] forward {chunks} provided_levels={levels}")
        for chunks in forwards:
            for c in chunks:
                if c.is_empty:
                    continue
                alone = [PrefillChunk(c.slot_id + num_users, c.actual_start, c.actual_end)]
                alone += [empty] * (FORWARD_CHUNKS - 1)
                _mtp_forward(
                    transformer,
                    kvpe_cache,
                    index_kv_cache,
                    mesh_device,
                    alone,
                    ids_by_slot,
                    lengths_by_slot,
                    chunk,
                    mtp_levels,
                )

        kv_layers = NUM_LAYERS + mtp_levels
        kv_slots = _read_user_slots(kvpe_cache, mesh_device, kv_layers, num_slots)
        idx_slots = _read_user_slots(index_kv_cache, mesh_device, full_indexer_rank(config, kv_layers), num_slots)
        failures = []
        for u in range(num_users):
            for name, slots in (("kvpe", kv_slots), ("indexer_k", idx_slots)):
                ref, got = slots[u + num_users], slots[u]
                for layer in range(ref.shape[0]):
                    dense = name == "kvpe" and layer < _TestGLM53Config.NUM_DENSE_LAYERS
                    bar = SINGLE_MODEL_FIRST_KV_PCC if dense else PACKED_KV_PCC
                    tag = f"{name}[{layer}]"
                    if name == "kvpe" and layer >= NUM_LAYERS:
                        bar = PACKED_MTP_LEVEL_KV_PCC[layer - NUM_LAYERS]
                        tag = f"mtp L{layer - NUM_LAYERS + 1} kvpe"
                    pcc = _pcc(ref[layer], got[layer])
                    logger.info(f"[packed mtp] user {u} {tag} PCC {pcc:.6f}")
                    if pcc < bar:
                        failures.append(f"user {u} {tag} PCC {pcc:.6f} < {bar}")
        # The MTP slots must hold real, user-specific KV, or the comparison above proves nothing.
        cross = _pcc(kv_slots[0][NUM_LAYERS], kv_slots[1][NUM_LAYERS])
        assert cross < DISTINCT_USERS_MAX_PCC, f"users 0 and 1 have the same MTP KV (PCC {cross:.6f})"
        assert not failures, f"[packed mtp] {len(failures)} mismatch(es):\n" + "\n".join(failures)
    finally:
        transformer.release_sub_device_managers()


def _traced_schedule(chunk):
    """(user lengths, forwards) for the traced test. Each real-chunk count 1..3 has its own capture, so each is replayed
    three times, in alternating order (3, 2, 1, 4, 1, 2, 3, 1, 2, 3 real). Inside forwards with 1, 2 and 3 real chunks:
    a partial chunk, a same-slot pair (the second chunk reads the first's KV), a pair ending in a partial; one chunk
    starts mid-chunk (user J: 800, after a partial first chunk)."""
    C = chunk
    E = PrefillChunk(0, 0, 0)
    P = PrefillChunk
    lengths = [3 * C, 704, C + 320, 512, 3 * C, 2 * C + 64, C + 100, C, 2 * C, 800 + C, C + 32]
    #           A=0   B=1  C=2      D=3  E=4    F=5         G=6      H=7 I=8    J=9       L=10
    forwards = [
        [P(0, 0, C), P(0, C, 2 * C), P(1, 0, 704), E],
        [P(2, 0, C), P(2, C, C + 320), E, E],
        [P(3, 0, 512), E, E, E],
        [P(0, 2 * C, 3 * C), P(4, 0, C), P(5, 0, C), P(9, 0, 800)],
        [P(9, 800, 800 + C), E, E, E],
        [P(4, C, 2 * C), P(5, C, 2 * C), E, E],
        [P(6, 0, C), P(6, C, C + 100), P(7, 0, C), E],
        [P(4, 2 * C, 3 * C), E, E, E],
        [P(5, 2 * C, 2 * C + 64), P(8, 0, C), E, E],
        [P(8, C, 2 * C), P(10, 0, C), P(10, C, C + 32), E],
    ]
    return lengths, forwards


# Two shared layers (1, 2) run before checked layer 3, so a wrong pre-sent buffer or top-k row of a shared layer shows
# up in later KV.
_TRACED_INDEXER_TYPES = ["full", "shared", "shared", "shared"]


@pytest.mark.parametrize(
    "mesh_device, device_params, num_links", TRACED_MESH_PARAMS, indirect=["mesh_device", "device_params"]
)
@pytest.mark.parametrize("variant", ["glm_5_3"], indirect=True, ids=["glm53"])
@pytest.mark.parametrize("schedule", ["packed", "every_real_count"])
@pytest.mark.skipif(not is_blackhole(), reason="GLM DSA ops (indexer / sparse SDPA) are Blackhole-only")
@pytest.mark.timeout(0)
def test_glm_packed_traced_matches_eager(variant, config_only, mesh_device, device_params, num_links, schedule):
    """Traced packed forwards (slots 0..num_users-1) must fill the same KVPE and indexer-K as the eager packed
    forward of the same chunks on the same model (the next num_users slots). ``packed``: _packed_schedule on the test's
    layer kinds; ``every_real_count``: _traced_schedule, two shared layers before a checked one."""
    assert_gate_mode_matches_adapter(variant, GATE_MODE)
    sp = mesh_device.shape[_sp_axis(mesh_device)]
    chunk = PREFILL_CHUNK_TOKENS_PER_CHIP * sp
    if schedule == "packed":
        (lengths, forwards), indexer_types = _packed_schedule(chunk), None
    else:
        (lengths, forwards), indexer_types = _traced_schedule(chunk), _TRACED_INDEXER_TYPES
    num_users = len(lengths)
    seq_cache = N_CHUNKS * chunk
    config = _test_config(config_only, indexer_types)
    config.max_seq_len = seq_cache
    sp_axis = _sp_axis(mesh_device)
    rt_config = TtPrefillRuntimeConfig(
        num_layers=NUM_LAYERS,
        max_seq_len=seq_cache,
        mesh_shape=tuple(mesh_device.shape),
        sp_axis=sp_axis,
        tp_axis=1 - sp_axis,
        chunk_size=chunk,
        forward_tokens_budget=FORWARD_CHUNKS * chunk,
        num_users=2 * num_users,
        num_links=num_links,
        topology=per_axis_topology(device_params["fabric_config"]),
        capacity_factor=8,
        gate_fallback_mode=GATE_MODE,
        model_cfg=_TestGLM53Config,
        moe_impl=MOE_IMPL,
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
            sp_axis=sp_axis,
            tp_axis=1 - sp_axis,
            num_kvpe_cache_layers=NUM_LAYERS,
            num_users=cache_slots,
        ),
        index=init_kvpe_cache(
            kvpe_cache_head_dim=config.index_head_dim,
            mesh_device=mesh_device,
            seq_len=seq_cache,
            mesh_shape=mesh_shape,
            sp_axis=sp_axis,
            tp_axis=1 - sp_axis,
            num_kvpe_cache_layers=full_indexer_rank(config, NUM_LAYERS),
            num_users=cache_slots,
            dtype=ttnn.bfloat8_b,
        ),
    )
    try:
        runtime.compile(caches)
        runtime.capture_trace(caches)
        logger.info(f"[traced] programs added by each capture, by real-chunk count: {runtime.capture_new_programs}")
        assert sorted(runtime.capture_new_programs) == list(range(1, FORWARD_CHUNKS + 1)), "one capture per count"
        assert all(v == 0 for v in runtime.capture_new_programs.values()), "a capture compiled a program"
        programs_after_capture = mesh_device.num_program_cache_entries()
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
        # Staging + replays only: nothing may compile after the captures.
        new_programs = mesh_device.num_program_cache_entries() - programs_after_capture
        logger.info(f"[traced] programs compiled by the traced forwards: {new_programs}")
        assert new_programs == 0, f"{new_programs} program(s) compiled after the trace captures"

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


def _mtp_traced_schedule(chunk):
    """(user lengths, forwards) for the traced packed-MTP test: _packed_schedule's chunks regrouped so forward 1
    generates no level (replays "provided") and the others do ("generate"). Partial chunks go last, as the producer
    puts them (_partial_runs_last)."""
    C = chunk
    lengths, _ = _packed_schedule(chunk)
    E = PrefillChunk(0, 0, 0)
    forwards = [
        [PrefillChunk(0, 0, C), PrefillChunk(1, 0, C), PrefillChunk(2, 0, 896), E],
        [PrefillChunk(0, C, 2 * C), PrefillChunk(3, 0, C), E, E],
        [PrefillChunk(3, C, 2 * C), PrefillChunk(1, C, C + 704), E, E],
        [PrefillChunk(0, 2 * C, 3 * C), E, E, E],
    ]
    return lengths, forwards


class _RandomMTPRuntime(TtPrefillRuntime):
    """TtPrefillRuntime whose MTP predictor is built from ``mtp_state`` (random weights) instead of the MTP weight
    cache, with the production runtime's settings otherwise."""

    mtp_state = None
    mtp_cfg = None

    def _build_mtp_predictor(self) -> None:
        from models.demos.deepseek_v3_d_p.tt.mtp_prefill.tt_mtp import TtMTPPredictor

        cfg = self.config
        self.mtp_predictor = TtMTPPredictor(
            self.mesh_device,
            self.hf_config,
            cfg.model_cfg,
            self.mtp_state,
            self.mtp_cfg,
            seq_len=cfg.chunk_size,
            num_levels=cfg.mtp_levels,
            layer_idx=NUM_LAYERS,
            first_cache_slot=cfg.num_layers,
            kv_only_last_level=cfg.kv_only_last_mtp_level,
            tp_axis=cfg.tp_axis,
            sp_axis=cfg.sp_axis,
            num_links=cfg.num_links,
            topology=cfg.topology,
            gate_fallback_mode=cfg.gate_fallback_mode,
            dispatch_buffer_capacity_factor=cfg.capacity_factor,
            routing_use_l1_small_for_semaphores=cfg.routing_use_l1_small_for_semaphores,
            sparse_kv_cache_format=cfg.sparse_kv_cache_format,
            tp_shard_kv=cfg.tp_shard_kv,
            is_chunked=True,
            max_seq_len=cfg.max_seq_len,
            slot_num=cfg.cache_num_users,
            layer_num=cfg.num_layers + cfg.mtp_levels,
            forward_chunks=cfg.forward_chunks,
            moe_impl=cfg.moe_impl,  # (the trunk's MoE: a packed forward's padding is one kind for both)
        )


@pytest.mark.parametrize(
    "mesh_device, device_params, num_links", TRACED_MTP_MESH_PARAMS, indirect=["mesh_device", "device_params"]
)
@pytest.mark.parametrize("variant", ["glm_5_3"], indirect=True, ids=["glm53"])
@pytest.mark.skipif(not is_blackhole(), reason="GLM DSA ops (indexer / sparse SDPA) are Blackhole-only")
@MTP_LEVELS
@pytest.mark.timeout(0)
def test_glm_packed_mtp_traced_matches_eager(variant, config_only, mesh_device, device_params, num_links, mtp_levels):
    """Traced packed forwards with MTP (TtPrefillRuntime.prefill_forward, slots 0..3) must fill every user's trunk and
    MTP KV as the eager packed forward of the same chunks does (slots 4..7), replaying both MTP captures
    ("generate" and "provided")."""
    from models.demos.deepseek_v3_d_p.tt.mtp_prefill.utils import enable_mtp_indexer_slot

    assert_gate_mode_matches_adapter(variant, GATE_MODE)
    sp_axis = _sp_axis(mesh_device)
    sp = mesh_device.shape[sp_axis]
    chunk = PREFILL_CHUNK_TOKENS_PER_CHIP * sp
    lengths, forwards = _mtp_traced_schedule(chunk)
    num_users = len(lengths)
    seq_cache = N_CHUNKS * chunk
    config = _test_config(config_only)
    config.max_seq_len = seq_cache
    # The runtime places the MTP layer after config.num_hidden_layers (enable_mtp_indexer_slot), so the test's depth
    # has to be the config's, or the MTP indexer addresses the real model's 78 index layers.
    config.num_hidden_layers = NUM_LAYERS
    enable_mtp_indexer_slot(config, NUM_LAYERS)
    rt_config = TtPrefillRuntimeConfig(
        num_layers=NUM_LAYERS,
        max_seq_len=seq_cache,
        mesh_shape=tuple(mesh_device.shape),
        sp_axis=sp_axis,
        tp_axis=1 - sp_axis,
        chunk_size=chunk,
        forward_tokens_budget=FORWARD_CHUNKS * chunk,
        num_users=2 * num_users,
        num_links=num_links,
        topology=per_axis_topology(device_params["fabric_config"]),
        capacity_factor=8,
        gate_fallback_mode=GATE_MODE,
        model_cfg=_TestGLM53Config,
        moe_impl=MOE_IMPL,
        routing_use_l1_small_for_semaphores=GLM53Adapter.routing_use_l1_small_for_semaphores,
        sparse_kv_cache_format=MlaKvCacheFormat.BF16_RM,
        use_trace=True,
        tp_shard_kv=True,
        mtp_levels=mtp_levels,
    )
    mesh_device.enable_program_cache()
    state_dict, mtp_state, mtp_cfg = _mtp_test_weights(config, mtp_levels)
    runtime_cls = type("_Runtime", (_RandomMTPRuntime,), {"mtp_state": mtp_state, "mtp_cfg": mtp_cfg})
    runtime = runtime_cls(mesh_device, config, state_dict, rt_config)
    del state_dict, mtp_state
    kv_layers = NUM_LAYERS + mtp_levels
    cache_slots = rt_config.cache_num_users  # the users plus the traced forward's scratch slot
    mesh_shape = list(mesh_device.shape)
    caches = MlaKvCaches(
        kvpe=init_mla_kv_cache(
            cache_format=MlaKvCacheFormat.BF16_RM,
            hf_config=config,
            mesh_device=mesh_device,
            seq_len=seq_cache,
            mesh_shape=mesh_shape,
            sp_axis=sp_axis,
            tp_axis=1 - sp_axis,
            num_kvpe_cache_layers=kv_layers,
            num_users=cache_slots,
        ),
        index=init_kvpe_cache(
            kvpe_cache_head_dim=config.index_head_dim,
            mesh_device=mesh_device,
            seq_len=seq_cache,
            mesh_shape=mesh_shape,
            sp_axis=sp_axis,
            tp_axis=1 - sp_axis,
            num_kvpe_cache_layers=full_indexer_rank(config, kv_layers),
            num_users=cache_slots,
            dtype=ttnn.bfloat8_b,
        ),
    )
    try:
        runtime.compile(caches)
        runtime.capture_trace(caches)
        ids = _token_ids(config, num_users, N_CHUNKS * chunk)
        ids_by_slot = {s: ids[s % num_users] for s in range(2 * num_users)}
        lengths_by_slot = {s: lengths[s % num_users] for s in range(2 * num_users)}

        variants = []
        for f, chunks in enumerate(forwards):
            inputs = [_mtp_chunk_inputs(c, ids_by_slot, lengths_by_slot, chunk, sp, mtp_levels) for c in chunks]
            levels = [provided for _, _, provided in inputs]
            generates = any(p < mtp_levels for c, p in zip(chunks, levels) if not c.is_empty)
            variants.append("generate" if generates else "provided")
            logger.info(f"[traced mtp] forward {f}: {chunks} provided_levels={levels} ({variants[-1]})")
            runtime.prefill_forward(
                runtime.make_forward_input([real for real, _, _ in inputs], [c.actual_start for c in chunks]),
                caches,
                chunks,
                mtp_tokens=runtime.make_forward_mtp_tokens([mtp for _, mtp, _ in inputs]),
                provided_levels=levels,
            )
        ttnn.synchronize_device(mesh_device)
        assert set(variants) == {"generate", "provided"}, f"the schedule replays only {set(variants)}"

        # Eager reference on the same model: the packed forward called directly, so it runs the host-scalar programs
        # and run_mtp_packed rather than the captured ones.
        for chunks in forwards:
            ref = [
                c if c.is_empty else PrefillChunk(c.slot_id + num_users, c.actual_start, c.actual_end) for c in chunks
            ]
            _mtp_forward(
                runtime.model,
                caches.kvpe,
                caches.index,
                mesh_device,
                ref,
                ids_by_slot,
                lengths_by_slot,
                chunk,
                mtp_levels,
            )

        kv_slots = _read_user_slots(caches.kvpe, mesh_device, kv_layers, cache_slots)
        idx_slots = _read_user_slots(caches.index, mesh_device, full_indexer_rank(config, kv_layers), cache_slots)
        failures = _compare_slots(
            "traced mtp", kv_slots, idx_slots, [(u + num_users, u) for u in range(num_users)], TRACED_PCC
        )
        # The MTP slots must hold real, user-specific KV, or the comparison above proves nothing.
        cross = _pcc(kv_slots[0][NUM_LAYERS], kv_slots[1][NUM_LAYERS])
        assert cross < DISTINCT_USERS_MAX_PCC, f"users 0 and 1 have the same MTP KV (PCC {cross:.6f})"
        assert not failures, f"[traced mtp] {len(failures)} mismatch(es):\n" + "\n".join(failures)
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
                fabric_payload_size=GLM_FABRIC_PAYLOAD,
                l1_small_size=TEST_L1_SMALL_SIZE,
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
    sp = mesh_device.shape[_sp_axis(mesh_device)]
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
            moe_impl=MOE_IMPL,
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
            # The runner's send warm-up (prefill_runner._forward_send_warmup) reads the packed top-k between compile()
            # and capture_trace(): compile() must leave it on the model.
            sent = ranks[0].outbound_indexer_indices
            assert sent is not None, "compile() left no packed top-k for the send warm-up"
            assert sent.is_allocated(), "compile() left a freed packed top-k for the send warm-up"
            sent.memory_config()  # what the runner's _stage_for reads
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
            for controller in r._controllers_by_real_count.values():
                controller.release()
            if r._controller is not None:
                r._controller.release()
        # Then each model's sub-device managers (and the D2H service a traced rank holds, whose device-side
        # teardown needs the mesh still open).
        for r in ranks + ([ref] if ref is not None else []):
            r.release_trace()
        d2h = None
