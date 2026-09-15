# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Correctness of the ring-fused indexer_score op (ttnn.experimental.ring_indexer_score_dsa) on Blackhole.
Coverage includes a LoudBox 2x4 -> 1x4 axis ring, the complete 2x4 mesh, exact-physical 2x2, and opt-in
8x4 Galaxy gates. One op co-schedules the ring_attention
all-gather with the score; the reader gates each K band on only the SP shards it touches and dual-sources its
own slab from k_local. Checked against the same DSA references as the two-op path, including both K layouts,
indexed caches, straddle, kv_len, program-cache reuse, placement, and host validation.

Run:  scripts/run_safe_pytest.sh tests/ttnn/nightly/unit_tests/operations/experimental/indexer_score/test_ring_indexer_score_dsa.py
"""

import os

import pytest
import torch
from loguru import logger

import ttnn

from tests.ttnn.nightly.unit_tests.operations.experimental.indexer_score.test_indexer_score import (
    assert_indexer_match,
    to_device,
    glx_config,
    indexer_score_dsa_ref,
    _global_inputs,
    _nd_sharded_dram_config,
    _per_sp_ref,
    _straddle_ref,
    _to_slab,
    QB_HISTORY,
    QB_SQ,
    QB_CASES,
    QB_IDS,
    ST_CHUNK,
    ST_CS,
    ST_T,
)
from tests.ttnn.nightly.unit_tests.operations.experimental.indexer_score.ring_indexer_score_test_utils import (
    _open_ring4_ccl,
    ring_parent_shape,
    _close_ring4_ccl,
    _persistent_buffer,
    _shard_k,
    _to_tp_inner_reconstructed,
    RING,
    SP_AXIS,
    CHUNK_GLOBAL,
    T,
)

pytestmark = [
    pytest.mark.skipif(not ttnn.device.is_blackhole(), reason="indexer_score is Blackhole-only"),
    # The ring is carved out of whatever system mesh the box exposes (LoudBox 2x4, galaxy 8x4, ...), so the
    # requirement is an axis-1 long enough to hold it -- not a specific box size.
    pytest.mark.skipif(
        ring_parent_shape()[1] < RING,
        reason="ring-of-4 needs a system mesh with axis-1 >= 4",
    ),
]


def _ring8_partial_router_config():
    config = ttnn.FabricRouterConfig()
    config.max_packet_payload_size_bytes = 14 * 1024
    return config


_RING8_PARTIAL_DEVICE_PARAMS = {
    "fabric_config": ttnn.FabricConfig.FABRIC_2D_TORUS_XY,
    "reliability_mode": ttnn.FabricReliabilityMode.STRICT_INIT,
    "fabric_tensix_config": ttnn.FabricTensixConfig.DISABLED,
    "fabric_router_config": _ring8_partial_router_config(),
    "require_exact_physical_num_devices": True,
}


def _fused_dev_inputs(submesh, q_g, w_g, k_host, *, k_dtype=ttnn.bfloat16):
    """Fused op inputs: SP-shard q/w (bf16) on dim 2, SP-shard k_local (the AG input), and a zero-seeded
    gathered buffer (AG fills remote bands; zeros prove the reader dual-sources the local band). k_dtype sets
    both k_local and k_gathered (the op requires them equal)."""
    shard = ttnn.ShardTensorToMesh(submesh, dim=2)
    q_dev = ttnn.from_torch(q_g, device=submesh, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, mesh_mapper=shard)
    w_dev = to_device(w_g, submesh, mesh_mapper=shard)
    k_local = _shard_k(submesh, k_host, dtype=k_dtype)  # [B,1,sll,D] per chip (the all-gather INPUT)
    # Indexed mode gathers one selected input slot into slot 0; batch-1 scratch also covers the ordinary B=1 path.
    k_gathered = _persistent_buffer(submesh, torch.zeros_like(k_host[:1]), dtype=k_dtype)
    return q_dev, w_dev, k_local, k_gathered


def _open_full_mesh_ccl(mesh_shape, *, fabric_config=ttnn.FabricConfig.FABRIC_2D_TORUS_XY):
    """Open the complete physical 2D mesh with the requested fabric configuration."""
    ttnn.set_fabric_config(
        fabric_config,
        ttnn.FabricReliabilityMode.STRICT_INIT,
        None,
        ttnn.FabricTensixConfig.DISABLED,
        ttnn.FabricUDMMode.DISABLED,
        ttnn.FabricManagerMode.DEFAULT,
    )
    mesh = None
    try:
        mesh = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(*mesh_shape))
        grid = mesh.compute_with_storage_grid_size()
        ccl_crs = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))})
        worker_sub_device = ttnn.SubDevice([ccl_crs])
        worker_sub_device_id = ttnn.SubDeviceId(0)
        stall_group = [worker_sub_device_id]
        manager = mesh.create_sub_device_manager([worker_sub_device], 0)
        mesh.load_sub_device_manager(manager)
        mesh.set_sub_device_stall_group(stall_group)
        semaphores = [ttnn.create_global_semaphore(mesh, ccl_crs, 0) for _ in range(2)]
        return mesh, semaphores, worker_sub_device_id, stall_group
    except Exception:
        if mesh is not None:
            ttnn.close_mesh_device(mesh)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
        raise


def _close_full_mesh_ccl(mesh):
    try:
        try:
            mesh.reset_sub_device_stall_group()
            mesh.clear_loaded_sub_device_manager()
        finally:
            ttnn.close_mesh_device(mesh)
    finally:
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


def _full_mesh_inputs(mesh, q_g, w_g, k_host, *, k_dtype=ttnn.bfloat16):
    """Canonical flat row-major sequence shards plus a complete-mesh replicated gather scratch."""
    shard = ttnn.ShardTensorToMesh(mesh, dim=2)
    q_dev = ttnn.from_torch(q_g, device=mesh, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, mesh_mapper=shard)
    w_dev = to_device(w_g, mesh, mesh_mapper=shard)
    k_local = ttnn.from_torch(k_host, device=mesh, layout=ttnn.TILE_LAYOUT, dtype=k_dtype, mesh_mapper=shard)
    k_gathered = ttnn.from_torch(
        torch.zeros_like(k_host[:1]),
        device=mesh,
        layout=ttnn.TILE_LAYOUT,
        dtype=k_dtype,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )
    return q_dev, w_dev, k_local, k_gathered


def _linear_full_mesh_ref(q_g, k_g, w_g, ring_size, local_sq, chunk_start):
    refs = []
    for tensor_rank in range(ring_size):
        sl = slice(tensor_rank * local_sq, (tensor_rank + 1) * local_sq)
        refs.append(
            indexer_score_dsa_ref(q_g[:, :, sl, :], k_g, w_g[:, :, sl, :], chunk_start + tensor_rank * local_sq)
        )
    return torch.cat(refs, dim=2)


def _assert_remote_gather_slots(k_local, k_gathered, ring_size, valid_local_rows=None, cache_batch_idx=None):
    """Every remote transport shard must land in its canonical row-major tensor slot."""
    local_shards = [ttnn.to_torch(shard) for shard in ttnn.get_device_tensors(k_local)]
    gathered_shards = [ttnn.to_torch(shard) for shard in ttnn.get_device_tensors(k_gathered)]
    assert len(local_shards) == len(gathered_shards) == ring_size
    local_rows = local_shards[0].shape[2]
    valid_rows = local_rows if valid_local_rows is None else valid_local_rows
    for destination_rank, gathered in enumerate(gathered_shards):
        for tensor_rank, local in enumerate(local_shards):
            if tensor_rank == destination_rank:
                continue  # the fused reader may direct-source its optimized local slot
            start = tensor_rank * local_rows
            expected = local if cache_batch_idx is None else local[cache_batch_idx : cache_batch_idx + 1]
            assert torch.equal(
                gathered[:, :, start : start + valid_rows, :], expected[:, :, :valid_rows, :]
            ), f"destination {destination_rank} stores tensor rank {tensor_rank} in the wrong K slot"


def _run_fused(
    heads,
    *,
    block_cyclic,
    num_links=1,
    k_dtype=ttnn.bfloat16,
):
    """Run the one fused op and check vs the per-SP reference. num_links only changes fabric routing, never
    the gathered result -> same reference."""
    submesh, parent, ccl_semaphores, subdevice_id, stall_group = _open_ring4_ccl()
    try:
        q_g, k_nat, w_g = _global_inputs(heads, CHUNK_GLOBAL, T, seed=42)
        k_host = _to_slab(k_nat, RING, CHUNK_GLOBAL) if block_cyclic else k_nat
        q_dev, w_dev, k_local, k_gathered = _fused_dev_inputs(submesh, q_g, w_g, k_host, k_dtype=k_dtype)

        bc_kwargs = dict(block_cyclic_sp_axis=SP_AXIS, block_cyclic_chunk_local=QB_SQ) if block_cyclic else {}
        out = ttnn.experimental.ring_indexer_score_dsa(
            q_dev,
            k_gathered,
            w_dev,
            k_local,
            ccl_semaphores,
            cluster_axis=SP_AXIS,
            topology=ttnn.Topology.Linear,
            num_links=num_links,
            ag_sub_device_id=subdevice_id,
            program_config=glx_config(heads),
            **bc_kwargs,
        )
        ttnn.synchronize_device(submesh, sub_device_ids=stall_group)
        out_t = ttnn.to_torch(out, mesh_composer=ttnn.ConcatMeshToTensor(submesh, dim=2))

        ref = _per_sp_ref(q_g, k_nat, w_g, RING, QB_HISTORY)
        assert_indexer_match(out_t, ref, CHUNK_GLOBAL, T, check_neg=True)
        layout = "block_cyclic" if block_cyclic else "contiguous"
        logger.info(f"ring4 fused {layout} (heads={heads}): fused all-gather + dual-source score matched reference")
    finally:
        _close_ring4_ccl(parent, submesh, stall_group)


@pytest.mark.parametrize("block_cyclic", [False, True], ids=["contiguous", "block_cyclic"])
@pytest.mark.parametrize("case_id, heads", QB_CASES, ids=QB_IDS)
def test_indexer_score_ring4_fused(case_id, heads, block_cyclic):
    """Base fused path, num_links=2 (the production Blackhole link count)."""
    _run_fused(heads, block_cyclic=block_cyclic, num_links=2)


@pytest.mark.parametrize("block_cyclic", [False, True], ids=["contiguous", "block_cyclic"])
@pytest.mark.parametrize("case_id, heads", QB_CASES, ids=QB_IDS)
def test_indexer_score_ring4_fused_bfp8_k(case_id, heads, block_cyclic):
    """Production dtype: bfloat8_b K (local shard + gathered buffer), q/w stay bf16. Same PCC floor."""
    _run_fused(heads, block_cyclic=block_cyclic, num_links=1, k_dtype=ttnn.bfloat8_b)


@pytest.mark.parametrize("case_id, heads", QB_CASES, ids=QB_IDS)
def test_indexer_score_ring4_fused_production_shape(case_id, heads):
    """All production knobs at once (each covered alone elsewhere): block-cyclic + non-zero chunk_start +
    kv_len < T_alloc + num_links=2 + bfloat8_b K. Guards their interaction (bfp8 gathered buffer + kv_len tail
    mask + block-cyclic invP on the nl2 schedule), which the model always drives together."""
    chunk_start = CHUNK_GLOBAL  # a later prefill chunk (rank r attends to chunk_start + (r+1)*QB_SQ)
    kv_len = chunk_start + CHUNK_GLOBAL  # fullest rank's causal window == kv_len exactly (validate's tightest bound)
    t_alloc = 4 * CHUNK_GLOBAL  # over-allocate so kv_len < T_alloc (ring-divisible, tile-aligned)
    submesh, parent, ccl_semaphores, subdevice_id, stall_group = _open_ring4_ccl()
    try:
        q_g, k_nat, w_g = _global_inputs(heads, CHUNK_GLOBAL, t_alloc, seed=42)
        k_bc = _to_slab(k_nat, RING, CHUNK_GLOBAL)  # block-cyclic physical layout the reader inverts
        # bfloat8_b K (the model's cache dtype) for both the local shard and the gathered buffer.
        q_dev, w_dev, k_local, k_gathered = _fused_dev_inputs(submesh, q_g, w_g, k_bc, k_dtype=ttnn.bfloat8_b)

        out = ttnn.experimental.ring_indexer_score_dsa(
            q_dev,
            k_gathered,
            w_dev,
            k_local,
            ccl_semaphores,
            cluster_axis=SP_AXIS,
            topology=ttnn.Topology.Linear,
            num_links=2,
            ag_sub_device_id=subdevice_id,
            chunk_start_idx=chunk_start,
            kv_len=kv_len,
            block_cyclic_sp_axis=SP_AXIS,
            block_cyclic_chunk_local=QB_SQ,
            program_config=glx_config(heads),
        )
        ttnn.synchronize_device(submesh, sub_device_ids=stall_group)
        out_t = ttnn.to_torch(out, mesh_composer=ttnn.ConcatMeshToTensor(submesh, dim=2))

        # Only [0, kv_len) is valid; each SP rank scores the valid key prefix at the non-zero chunk_start.
        ref = _per_sp_ref(q_g, k_nat[:, :, :kv_len, :], w_g, RING, chunk_start)
        assert_indexer_match(out_t[:, :, :, :kv_len], ref, CHUNK_GLOBAL, kv_len, check_neg=True)
        logger.info(
            f"ring4 fused production-shape (heads={heads}): block_cyclic+cs={chunk_start}+kv_len={kv_len}+nl2+bfp8 "
            f"matched reference"
        )
    finally:
        _close_ring4_ccl(parent, submesh, stall_group)


def _run_fused_multiuser(heads, *, num_users, cache_batch_idx, num_links=1):
    """Multi-user indexed cache: k_local [num_users,1,sll,D] and batch-1 gathered scratch. cache_batch_idx
    selects the single gathered slot and the reader applies the corresponding local-cache offset. Distinct
    user K values make either a gather-slot or local-slot addressing error fail PCC."""
    submesh, parent, ccl_semaphores, subdevice_id, stall_group = _open_ring4_ccl()
    try:
        # Shared q/w scoring distinct per-user caches (distinct seed per slot -> a wrong-slot read changes the score).
        q_g, _, w_g = _global_inputs(heads, CHUNK_GLOBAL, T, seed=42)
        k_multi = torch.cat([_global_inputs(heads, CHUNK_GLOBAL, T, seed=100 + u)[1] for u in range(num_users)], dim=0)
        q_dev, w_dev, k_local, k_gathered = _fused_dev_inputs(submesh, q_g, w_g, k_multi)

        out = ttnn.experimental.ring_indexer_score_dsa(
            q_dev,
            k_gathered,
            w_dev,
            k_local,
            ccl_semaphores,
            cluster_axis=SP_AXIS,
            topology=ttnn.Topology.Linear,
            num_links=num_links,
            ag_sub_device_id=subdevice_id,
            cache_batch_idx=cache_batch_idx,
            program_config=glx_config(heads),
        )
        ttnn.synchronize_device(submesh, sub_device_ids=stall_group)
        out_t = ttnn.to_torch(out, mesh_composer=ttnn.ConcatMeshToTensor(submesh, dim=2))

        ref = _per_sp_ref(q_g, k_multi[cache_batch_idx : cache_batch_idx + 1], w_g, RING, QB_HISTORY)
        assert_indexer_match(out_t, ref, CHUNK_GLOBAL, T, check_neg=True)
        logger.info(
            f"ring4 fused multi-user (heads={heads}, users={num_users}, slot={cache_batch_idx}): matched reference"
        )
    finally:
        _close_ring4_ccl(parent, submesh, stall_group)


def test_indexer_score_ring4_fused_indexed_cache():
    """cache_batch_idx=1 (2nd user slot). One representative case -- the slot offset is head-independent."""
    _run_fused_multiuser(16, num_users=2, cache_batch_idx=1)


def test_indexer_score_ring4_fused_indexed_cache_slot_metadata(expect_error):
    """Trace-safe slot select: cache_batch_idx_tensor + index_cache_num_layers/_layer_idx.

    The cache is user-major, so the op recomposes the slot ON-DEVICE as user * num_layers + layer_idx.
    The SCALAR path is the oracle: for each layer the same dispatch is run once with the host-computed
    cache_batch_idx and once with the 1-element user tensor, and the two must agree.

    Only index_cache_layer_idx moves between layers. It is a RUNTIME arg re-patched by
    override_runtime_arguments (reader_slot_base + 2) and deliberately absent from the program hash, and
    that patch is what this test exists for: dropping it degrades KV PCC silently with depth rather than
    raising. Hence the third assertion -- the two layers must differ. A slot that never moved would match
    layer 0 twice and slip past a per-layer check.

    Slot metadata requires KV-extent metadata (one-directional), so chunk_start_idx_tensor rides along;
    its scalar equivalent is chunk_start_idx + kv_len = chunk_start + sp * chunk_local.
    """
    heads, num_users, num_layers = 16, 2, 3
    user = 1  # non-zero so a dropped user term is not masked by user * L == 0
    # The query chunk sits AFTER the history (T == QB_HISTORY + CHUNK_GLOBAL), so the global chunk start is
    # QB_HISTORY; each SP rank adds its own sp * QB_SQ. kv_len is what the metadata path derives on-device.
    chunk_start = QB_HISTORY
    kv_len = chunk_start + RING * QB_SQ

    submesh, parent, ccl_semaphores, subdevice_id, stall_group = _open_ring4_ccl()
    try:
        q_g, _, w_g = _global_inputs(heads, CHUNK_GLOBAL, T, seed=42)
        # Distinct K per (user, layer) slot in user-major order, so ANY slot-arithmetic error -- wrong user
        # term, wrong layer term, or a stale layer from a missing patch -- lands on different keys.
        k_slabs = [
            _to_slab(_global_inputs(heads, CHUNK_GLOBAL, T, seed=100 + slot)[1], RING, CHUNK_GLOBAL)
            for slot in range(num_users * num_layers)
        ]
        q_dev, w_dev, k_local, k_gathered = _fused_dev_inputs(submesh, q_g, w_g, torch.cat(k_slabs, dim=0))

        slot_tensor = ttnn.from_torch(
            torch.tensor([[[[user]]]], dtype=torch.int64),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.ReplicateTensorToMesh(submesh),
            device=submesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        chunk_start_tensor = ttnn.from_torch(
            torch.tensor([[[[chunk_start]]]], dtype=torch.int64),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.ReplicateTensorToMesh(submesh),
            device=submesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        def run(layer_idx, *, metadata, scalar_slot=None):
            slot_kwargs = (
                {
                    "cache_batch_idx_tensor": slot_tensor,
                    "index_cache_num_layers": num_layers,
                    "index_cache_layer_idx": layer_idx,
                    "chunk_start_idx_tensor": chunk_start_tensor,
                }
                if metadata
                else {
                    "cache_batch_idx": user * num_layers + layer_idx if scalar_slot is None else scalar_slot,
                    "chunk_start_idx": chunk_start,
                    "kv_len": kv_len,
                }
            )
            out = ttnn.experimental.ring_indexer_score_dsa(
                q_dev,
                k_gathered,
                w_dev,
                k_local,
                ccl_semaphores,
                cluster_axis=SP_AXIS,
                topology=ttnn.Topology.Linear,
                num_links=1,
                ag_sub_device_id=subdevice_id,
                block_cyclic_sp_axis=SP_AXIS,
                block_cyclic_chunk_local=QB_SQ,
                program_config=glx_config(heads),
                **slot_kwargs,
            )
            ttnn.synchronize_device(submesh, sub_device_ids=stall_group)
            return ttnn.to_torch(out, mesh_composer=ttnn.ConcatMeshToTensor(submesh, dim=2))

        meta_out = {}
        for layer_idx in (0, 1):
            scalar_out = run(layer_idx, metadata=False)
            meta_out[layer_idx] = run(layer_idx, metadata=True)
            assert torch.equal(meta_out[layer_idx], scalar_out), (
                f"layer {layer_idx}: cache_batch_idx_tensor (user={user}, num_layers={num_layers}) did not "
                f"reproduce scalar cache_batch_idx={user * num_layers + layer_idx}"
            )
            logger.info(f"ring4 slot metadata: user={user} layer={layer_idx} matched the scalar slot")
            if layer_idx == 0:
                entries_after_first = submesh.num_program_cache_entries()

        assert not torch.equal(meta_out[0], meta_out[1]), (
            "layer 0 and layer 1 produced identical scores -- index_cache_layer_idx did not reach the reader "
            "(check the override_runtime_arguments patch at reader_slot_base + 2)"
        )
        # index_cache_layer_idx is a runtime arg, not hashed: switching layers must reuse the program.
        assert submesh.num_program_cache_entries() == entries_after_first, "switching index_cache_layer_idx recompiled"

        # The USER id lives in the tensor, and the layer checks above never move it -- they vary only the
        # plain runtime scalar. Rewrite the same buffer in place on the warm program: the slot must follow,
        # which is what makes the value trace-safe (read on-device per dispatch, not captured once).
        other_user = 0
        ttnn.copy_host_to_device_tensor(
            ttnn.from_torch(
                torch.tensor([[[[other_user]]]], dtype=torch.int64),
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                mesh_mapper=ttnn.ReplicateTensorToMesh(submesh),
            ),
            slot_tensor,
        )
        meta_other = run(0, metadata=True)
        scalar_other = run(0, metadata=False, scalar_slot=other_user * num_layers)
        assert torch.equal(meta_other, scalar_other), (
            f"after rewriting the slot tensor to user={other_user}, the metadata path did not reproduce scalar "
            f"cache_batch_idx={other_user * num_layers} -- the user id is not being re-read on-device"
        )
        assert not torch.equal(meta_other, meta_out[0]), (
            f"user {user} and user {other_user} produced identical scores at layer 0 -- the slot tensor's VALUE "
            "was not re-read (a captured or cached user id would look exactly like this)"
        )
        assert submesh.num_program_cache_entries() == entries_after_first, "rewriting the slot tensor recompiled"
        logger.info(f"ring4 slot metadata: in-place user rewrite {user} -> {other_user} tracked by the reader")

        # The slot tensor needs the extent tensor: the factory forwards the slot to the all-gather but
        # withholds kv_actual_isl without it, and the helper then fails at PROGRAM BUILD with a message
        # naming neither this op nor the missing kwarg. Pin that it is rejected up front instead.
        with expect_error(RuntimeError, "cache_batch_idx_tensor requires chunk_start_idx_tensor"):
            ttnn.experimental.ring_indexer_score_dsa(
                q_dev,
                k_gathered,
                w_dev,
                k_local,
                ccl_semaphores,
                cluster_axis=SP_AXIS,
                topology=ttnn.Topology.Linear,
                num_links=1,
                ag_sub_device_id=subdevice_id,
                cache_batch_idx_tensor=slot_tensor,
                index_cache_num_layers=num_layers,
                index_cache_layer_idx=0,
                chunk_start_idx=chunk_start,
                kv_len=kv_len,
                block_cyclic_sp_axis=SP_AXIS,
                block_cyclic_chunk_local=QB_SQ,
                program_config=glx_config(heads),
            )
    finally:
        _close_ring4_ccl(parent, submesh, stall_group)


@pytest.mark.parametrize("rows_per_shard", [32, 96], ids=["prod_rows32", "padded_rows96"])
def test_indexer_score_ring4_fused_nd_indexed_bounded_gather_cache_hit(rows_per_shard):
    """Production cache contract in one regression:

    * multi-slot k_local is ND-sharded across DRAM banks;
    * the gather selects one slot into a batch-1 scratch;
    * kv_len bounds transport to complete touched block-cyclic slabs; and
    * a second dispatch changes both slot and kv_len on the same cached program; and
    * the production two-link all-gather partition preserves those cache-hit results.
    """
    heads, num_users = 16, 3
    t_alloc = 4 * CHUNK_GLOBAL
    local_t = t_alloc // RING
    submesh, parent, ccl_semaphores, subdevice_id, stall_group = _open_ring4_ccl()
    try:
        q_g, _, w_g = _global_inputs(heads, CHUNK_GLOBAL, t_alloc, seed=42)
        k_nat = torch.cat(
            [_global_inputs(heads, CHUNK_GLOBAL, t_alloc, seed=100 + u)[1] for u in range(num_users)], dim=0
        )
        k_bc = torch.cat([_to_slab(k_nat[u : u + 1], RING, CHUNK_GLOBAL) for u in range(num_users)], dim=0)
        q_dev, w_dev, k_local_i, k_gathered = _fused_dev_inputs(submesh, q_g, w_g, k_bc, k_dtype=ttnn.bfloat8_b)
        k_local = ttnn.to_memory_config(k_local_i, _nd_sharded_dram_config(submesh, rows_per_shard=rows_per_shard))
        ttnn.deallocate(k_local_i)

        def _score(slot, chunk_start, kv_len):
            out = ttnn.experimental.ring_indexer_score_dsa(
                q_dev,
                k_gathered,
                w_dev,
                k_local,
                ccl_semaphores,
                cluster_axis=SP_AXIS,
                topology=ttnn.Topology.Linear,
                num_links=2,
                ag_sub_device_id=subdevice_id,
                chunk_start_idx=chunk_start,
                cache_batch_idx=slot,
                kv_len=kv_len,
                block_cyclic_sp_axis=SP_AXIS,
                block_cyclic_chunk_local=QB_SQ,
                program_config=glx_config(heads),
            )
            ttnn.synchronize_device(submesh, sub_device_ids=stall_group)
            return ttnn.to_torch(out, mesh_composer=ttnn.ConcatMeshToTensor(submesh, dim=2))

        # A tile past the first global-slab boundary requires TWO complete local slabs per rank. This
        # specifically exercises ceil(kv_len/chunk_global), not just exact-boundary truncation.
        large_kv_len = CHUNK_GLOBAL + 32
        out0 = _score(slot=1, chunk_start=32, kv_len=large_kv_len)
        entries_after_first = submesh.num_program_cache_entries()
        scratch_after_large = [ttnn.to_torch(t).clone() for t in ttnn.get_device_tensors(k_gathered.cpu())]
        ref0 = _straddle_ref(q_g, k_nat[1:2, :, :large_kv_len, :], w_g, RING, CHUNK_GLOBAL, 32, large_kv_len)
        assert_indexer_match(out0[:, :, :, :large_kv_len], ref0, CHUNK_GLOBAL, large_kv_len, check_neg=True)

        valid_local_large = 2 * QB_SQ
        for scratch_t in scratch_after_large:
            for rank in range(RING):
                tail = scratch_t[:, :, rank * local_t + valid_local_large : (rank + 1) * local_t, :]
                assert torch.count_nonzero(tail) == 0, "gather wrote beyond the slab-rounded kv_len extent"

        # Shrink the extent and switch users. Both are runtime values: this must reuse the same compiled
        # program, overwrite only slab 0 from slot 2, and leave slab 1 exactly as slot 1 wrote it.
        out1 = _score(slot=2, chunk_start=0, kv_len=CHUNK_GLOBAL)
        assert (
            submesh.num_program_cache_entries() == entries_after_first
        ), "slot/kv_len change recompiled instead of exercising the cache-hit runtime-arg patch"
        scratch_after_small = [ttnn.to_torch(t) for t in ttnn.get_device_tensors(k_gathered.cpu())]
        ref1 = _per_sp_ref(q_g, k_nat[2:3, :, :CHUNK_GLOBAL, :], w_g, RING, 0)
        assert_indexer_match(out1[:, :, :, :CHUNK_GLOBAL], ref1, CHUNK_GLOBAL, CHUNK_GLOBAL, check_neg=True)

        any_first_slab_changed = False
        for before, after in zip(scratch_after_large, scratch_after_small):
            for rank in range(RING):
                base = rank * local_t
                any_first_slab_changed |= not torch.equal(
                    before[:, :, base : base + QB_SQ, :], after[:, :, base : base + QB_SQ, :]
                )
                assert torch.equal(
                    before[:, :, base + QB_SQ : base + 2 * QB_SQ, :],
                    after[:, :, base + QB_SQ : base + 2 * QB_SQ, :],
                ), "shrinking kv_len rewrote the second slab on a cache hit"
        assert any_first_slab_changed, "switching cache slots did not update any gathered first-slab data"
    finally:
        _close_ring4_ccl(parent, submesh, stall_group)


@pytest.mark.parametrize("case_id, heads", QB_CASES, ids=QB_IDS)
def test_indexer_score_ring4_fused_straddle(case_id, heads):
    """Mid-slab straddle + block-cyclic rotation (rotated-prefill/multiturn): a non-slab-aligned chunk_start
    (704) makes the boundary chip's queries cross a slab boundary, so the causal diagonal jumps by
    (chunk_global - cl). Proves the band reorder + per-band gate + dual-source read compose with the straddled
    mask. Checked vs the per-SP rotated reference."""
    cl = ST_CHUNK // RING  # per-shard chunk / per-device query rows (block-cyclic SP-only)
    submesh, parent, ccl_semaphores, subdevice_id, stall_group = _open_ring4_ccl()
    try:
        q_g, k_nat, w_g = _global_inputs(heads, ST_CHUNK, ST_T, seed=42)
        k_bc = _to_slab(k_nat, RING, ST_CHUNK)  # block-cyclic physical layout the reader inverts

        q_dev, w_dev, k_local, k_gathered = _fused_dev_inputs(submesh, q_g, w_g, k_bc)  # [1,1,ST_T/RING,D]

        out = ttnn.experimental.ring_indexer_score_dsa(
            q_dev,
            k_gathered,
            w_dev,
            k_local,
            ccl_semaphores,
            cluster_axis=SP_AXIS,
            topology=ttnn.Topology.Linear,
            num_links=1,
            ag_sub_device_id=subdevice_id,
            chunk_start_idx=ST_CS,  # mid-slab (704 % cl != 0) -> rotation + straddle
            block_cyclic_sp_axis=SP_AXIS,
            block_cyclic_chunk_local=cl,
            program_config=glx_config(heads),
        )
        ttnn.synchronize_device(submesh, sub_device_ids=stall_group)
        out_t = ttnn.to_torch(out, mesh_composer=ttnn.ConcatMeshToTensor(submesh, dim=2))

        ref = _straddle_ref(q_g, k_nat, w_g, RING, ST_CHUNK, ST_CS, ST_T)
        assert_indexer_match(out_t, ref, ST_CHUNK, ST_T, check_neg=True)
        logger.info(f"ring4 fused straddle (heads={heads}): rotated-prefill causal diagonal matched reference")
    finally:
        _close_ring4_ccl(parent, submesh, stall_group)


def test_indexer_score_ring4_fused_runtime_kv_len():
    """Padded cache: k allocated at T_alloc but only a kv_len prefix is valid; only cols [0, kv_len) are
    written. Confirms the AG gathers full T_alloc and the compute masks beyond kv_len (band_count spans full T,
    so no shard is left un-delivered). heads=16 representative -- kv_len masking is head-independent."""
    heads = 16
    kv_len = QB_HISTORY + CHUNK_GLOBAL  # valid written extent (28160 keys, 880 tiles)
    t_alloc = kv_len + CHUNK_GLOBAL  # over-allocate one more global chunk (30720, ring-divisible, tile-aligned)
    submesh, parent, ccl_semaphores, subdevice_id, stall_group = _open_ring4_ccl()
    try:
        q_g, k_nat, w_g = _global_inputs(heads, CHUNK_GLOBAL, t_alloc, seed=42)
        q_dev, w_dev, k_local, k_gathered = _fused_dev_inputs(submesh, q_g, w_g, k_nat)  # [1,1,t_alloc/RING,D]

        out = ttnn.experimental.ring_indexer_score_dsa(
            q_dev,
            k_gathered,
            w_dev,
            k_local,
            ccl_semaphores,
            cluster_axis=SP_AXIS,
            topology=ttnn.Topology.Linear,
            num_links=1,
            ag_sub_device_id=subdevice_id,
            chunk_start_idx=QB_HISTORY,  # rank r attends up to QB_HISTORY + (r+1)*QB_SQ; fullest = kv_len exactly
            kv_len=kv_len,
            program_config=glx_config(heads),
        )
        ttnn.synchronize_device(submesh, sub_device_ids=stall_group)
        out_t = ttnn.to_torch(out, mesh_composer=ttnn.ConcatMeshToTensor(submesh, dim=2))

        # Only [0, kv_len) is valid; reference scores each rank against the valid key prefix (rest is stale tail).
        ref = _per_sp_ref(q_g, k_nat[:, :, :kv_len, :], w_g, RING, QB_HISTORY)
        assert_indexer_match(out_t[:, :, :, :kv_len], ref, CHUNK_GLOBAL, kv_len, check_neg=True)
        logger.info(
            f"ring4 fused runtime kv_len (kv_len={kv_len} of T_alloc={t_alloc}): valid prefix matched reference"
        )
    finally:
        _close_ring4_ccl(parent, submesh, stall_group)


# The chunked-prefill tail, in one geometry. A fixed chunk size means a sequence's last chunk pads its
# query window past what the cache holds, so update_padded_kv_cache clamps the WRITE to the real tokens
# and kv_len is the matching READ bound. One case carries every edge of that contract:
#   * the causal window ends past the populated prefix (1632 > 384) -- pad rows with no keys to attend;
#   * and past the cache itself (1632 > 1280), which used to be a validation error;
#   * kv_len sits below one k_chunk (384 < 512). k_chunk_size is HASHED so it keeps its tuned value; only
#     band 0 does partial work. Unreachable while kv_len was the padded window (never below a chunk);
#   * the start is mid-slab, so the block-cyclic staircase straddles at boundary chip 1 -- the only
#     boundary that exercises all three of its branches (chips below advance a slab, the boundary chip
#     by its offset, chips above stay at the base). Chip 0 leaves "below" empty, chip 3 leaves "above".
# T is one global chunk here rather than the module's 3: past-the-cache needs a small chunk_start, and
# chunk_start < kv_len < k_chunk forces it small.
_TAIL_T = ST_CHUNK  # 1280: one global chunk (T % chunk_global == 0 is required)
_TAIL_CHUNK_START = 352  # tile-aligned, 352 % 320 != 0 -> mid-slab, boundary chip 1
_TAIL_KV_LEN = 384  # tile-aligned, > chunk_start, < k_chunk


@pytest.mark.parametrize("case_id, heads", QB_CASES, ids=QB_IDS)
def test_indexer_score_ring4_fused_window_past_short_prefix(case_id, heads):
    """Tail chunk: the causal window ends past both the populated prefix and the cache, on a prefix
    shorter than one k_chunk. Real query rows keep their scores; pad rows saturate."""
    cl = ST_CHUNK // RING
    cfg = glx_config(heads)
    assert _TAIL_CHUNK_START + ST_CHUNK > _TAIL_KV_LEN, "the window must end past the populated prefix"
    assert _TAIL_CHUNK_START + ST_CHUNK > _TAIL_T, "the window must end past the cache"
    assert _TAIL_KV_LEN < cfg.k_chunk_size, "kv_len must sit below one k_chunk"
    assert (_TAIL_CHUNK_START // cl) % RING != 0, "the start must straddle a slab boundary"
    assert _TAIL_T % (RING * cl) == 0, "T must be a whole number of global chunks"

    submesh, parent, ccl_semaphores, subdevice_id, stall_group = _open_ring4_ccl()
    try:
        q_g, k_nat, w_g = _global_inputs(heads, ST_CHUNK, _TAIL_T, seed=42)
        k_bc = _to_slab(k_nat, RING, ST_CHUNK)
        q_dev, w_dev, k_local, k_gathered = _fused_dev_inputs(submesh, q_g, w_g, k_bc)

        out = ttnn.experimental.ring_indexer_score_dsa(
            q_dev,
            k_gathered,
            w_dev,
            k_local,
            ccl_semaphores,
            cluster_axis=SP_AXIS,
            topology=ttnn.Topology.Linear,
            num_links=1,
            ag_sub_device_id=subdevice_id,
            chunk_start_idx=_TAIL_CHUNK_START,
            block_cyclic_sp_axis=SP_AXIS,
            block_cyclic_chunk_local=cl,
            kv_len=_TAIL_KV_LEN,
            program_config=cfg,
        )
        ttnn.synchronize_device(submesh, sub_device_ids=stall_group)
        out_t = ttnn.to_torch(out, mesh_composer=ttnn.ConcatMeshToTensor(submesh, dim=2))

        # Reference over the populated prefix only. Rows whose block-cyclic home sits at or past kv_len
        # mask nothing -- every key they reach is in their past, the saturation the kernel must match.
        ref = _straddle_ref(q_g, k_nat[:, :, :_TAIL_KV_LEN, :], w_g, RING, ST_CHUNK, _TAIL_CHUNK_START, _TAIL_KV_LEN)
        assert_indexer_match(out_t[:, :, :, :_TAIL_KV_LEN], ref, ST_CHUNK, _TAIL_KV_LEN, check_neg=True)
        logger.info(
            f"ring4 fused tail (heads={heads}): chunk_start={_TAIL_CHUNK_START} window ends "
            f"{_TAIL_CHUNK_START + ST_CHUNK} past kv_len={_TAIL_KV_LEN} and cache {_TAIL_T}, "
            f"k_chunk={cfg.k_chunk_size} -- prefix matched reference"
        )
    finally:
        _close_ring4_ccl(parent, submesh, stall_group)


@pytest.mark.parametrize("k_dtype", [ttnn.bfloat16, ttnn.bfloat8_b], ids=["bf16", "bfp8"])
def test_indexer_score_ring4_fused_program_cache_reuse(k_dtype):
    """Two dispatches, identical shapes but different chunk_start/kv_len on the SAME device (2nd is a cache
    hit). chunk_start/kv_len and fused-AG semaphore identity are hash-excluded, so
    override_runtime_arguments must re-apply them; if not, the 2nd dispatch reuses the 1st's frozen offset or
    semaphore addresses. Regression guard for the program-cache stale-runtime-argument bugs (every other test
    dispatches cold). Both bf16 and production bfp8_b K."""
    heads = 16  # the scalar re-patch is head-independent, so one head count suffices (both dtypes kept)
    t_alloc = 4 * CHUNK_GLOBAL  # room for both chunks' causal windows (global block == CHUNK_GLOBAL)
    submesh, parent, ccl_semaphores, subdevice_id, stall_group = _open_ring4_ccl()
    try:
        q_g, k_nat, w_g = _global_inputs(heads, CHUNK_GLOBAL, t_alloc, seed=42)
        k_bc = _to_slab(k_nat, RING, CHUNK_GLOBAL)  # block-cyclic physical layout
        q_dev, w_dev, k_local, k_gathered = _fused_dev_inputs(submesh, q_g, w_g, k_bc, k_dtype=k_dtype)

        # A physically distinct pair exercises the semaphore-address cache-hit override. This is the same
        # A/B rotation used by model TT_CCL; changing only the addresses must not create another program.
        grid = submesh.compute_with_storage_grid_size()
        ccl_crs = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))})
        alternate_semaphores = [ttnn.create_global_semaphore(submesh, ccl_crs, 0) for _ in range(2)]

        def _score(chunk_start, kv_len, semaphores):  # identical shapes each call -> 2nd is a program-cache hit
            out = ttnn.experimental.ring_indexer_score_dsa(
                q_dev,
                k_gathered,
                w_dev,
                k_local,
                semaphores,
                cluster_axis=SP_AXIS,
                topology=ttnn.Topology.Linear,
                num_links=1,
                ag_sub_device_id=subdevice_id,
                chunk_start_idx=chunk_start,
                block_cyclic_sp_axis=SP_AXIS,
                block_cyclic_chunk_local=QB_SQ,
                kv_len=kv_len,
                program_config=glx_config(heads),
            )
            ttnn.synchronize_device(submesh, sub_device_ids=stall_group)
            return ttnn.to_torch(out, mesh_composer=ttnn.ConcatMeshToTensor(submesh, dim=2))

        # chunk@0 (cache miss/build) then chunk@CHUNK_GLOBAL (cache HIT -- must re-apply chunk_start + kv_len).
        out0 = _score(chunk_start=0, kv_len=CHUNK_GLOBAL, semaphores=ccl_semaphores)
        entries_after_first = submesh.num_program_cache_entries()
        out1 = _score(
            chunk_start=CHUNK_GLOBAL,
            kv_len=2 * CHUNK_GLOBAL,
            semaphores=alternate_semaphores,
        )
        assert (
            submesh.num_program_cache_entries() == entries_after_first
        ), "alternating fused-AG semaphore addresses must reuse the cached ring-indexer program"
        ref0 = _per_sp_ref(q_g, k_nat[:, :, :CHUNK_GLOBAL, :], w_g, RING, 0)
        ref1 = _per_sp_ref(q_g, k_nat[:, :, : 2 * CHUNK_GLOBAL, :], w_g, RING, CHUNK_GLOBAL)
        assert_indexer_match(out0[:, :, :, :CHUNK_GLOBAL], ref0, CHUNK_GLOBAL, CHUNK_GLOBAL, check_neg=True)
        assert_indexer_match(out1[:, :, :, : 2 * CHUNK_GLOBAL], ref1, CHUNK_GLOBAL, 2 * CHUNK_GLOBAL, check_neg=True)
        logger.info(
            f"ring4 fused program-cache reuse (heads={heads}): 2nd chunk_start and semaphore pair re-applied on cache hit"
        )
    finally:
        _close_ring4_ccl(parent, submesh, stall_group)


def _run_full_mesh_accuracy_case(mesh_shape, *, block_cyclic):
    """Exercise a non-identity snake permutation while retaining row-major causal and K-slot semantics."""
    ring_size = mesh_shape[0] * mesh_shape[1]
    local_sq = 64
    chunk_global = ring_size * local_sq
    if block_cyclic:
        # Enter slab 1, rotate ownership by one tensor rank, and make exactly that boundary rank straddle.
        chunk_start = chunk_global + local_sq + 32
        t_len = 3 * chunk_global
    else:
        chunk_start = chunk_global
        t_len = 2 * chunk_global

    mesh, semaphores, subdevice_id, stall_group = _open_full_mesh_ccl(mesh_shape)
    try:
        heads = 8
        q_g, k_nat, w_g = _global_inputs(heads, chunk_global, t_len, seed=2026)
        k_host = _to_slab(k_nat, ring_size, chunk_global) if block_cyclic else k_nat
        q_dev, w_dev, k_local, k_gathered = _full_mesh_inputs(mesh, q_g, w_g, k_host)
        kwargs = {"block_cyclic_chunk_local": local_sq} if block_cyclic else {}

        mesh.enable_program_cache()
        mesh.clear_program_cache()
        outputs = []
        entries_after_first = None
        for _ in range(2):
            out = ttnn.experimental.ring_indexer_score_dsa(
                q_dev,
                k_gathered,
                w_dev,
                k_local,
                semaphores,
                cluster_axis=None,
                topology=ttnn.Topology.Ring,
                num_links=2,
                ag_sub_device_id=subdevice_id,
                chunk_start_idx=chunk_start,
                program_config=glx_config(heads),
                **kwargs,
            )
            ttnn.synchronize_device(mesh, sub_device_ids=stall_group)
            outputs.append(ttnn.to_torch(out, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=2)))
            entries = mesh.num_program_cache_entries()
            if entries_after_first is None:
                assert entries > 0
                entries_after_first = entries
                _assert_remote_gather_slots(k_local, k_gathered, ring_size)
            else:
                assert entries == entries_after_first, "full-mesh replay added program-cache entries"

        assert torch.equal(outputs[0], outputs[1]), "full-mesh indexer replay is not bit-exact"
        if block_cyclic:
            ref = _straddle_ref(q_g, k_nat, w_g, ring_size, chunk_global, chunk_start, t_len)
        else:
            ref = _linear_full_mesh_ref(q_g, k_nat, w_g, ring_size, local_sq, chunk_start)
        assert_indexer_match(outputs[0], ref, chunk_global, t_len, check_neg=True)
        logger.info(
            f"full-mesh indexer {mesh_shape} {'block-cyclic rotated' if block_cyclic else 'contiguous'}: "
            "PCC, deterministic replay, cache reuse, and canonical remote K placement passed"
        )
    finally:
        if mesh is not None:
            mesh.disable_and_clear_program_cache()
        _close_full_mesh_ccl(mesh)


@pytest.mark.parametrize("block_cyclic", [False, True], ids=["contiguous", "block_cyclic_rotated"])
def test_indexer_score_full_mesh_loudbox_accuracy_placement_and_cache_reuse(block_cyclic):
    """Use every device on the physical 2x4 LoudBox as one eight-rank snake ring."""
    if ttnn.get_num_devices() != 8:
        pytest.skip("2x4 full-mesh indexer coverage requires the exact physical eight-device LoudBox")
    _run_full_mesh_accuracy_case((2, 4), block_cyclic=block_cyclic)


# ---- 2D SP x TP LoudBox: TP-inner reconstructed KV -----------------------------------------------
LB_SPTP_SP = 2
LB_SPTP_TP = 4
LB_SPTP_SP_AXIS = 0
LB_SPTP_TP_AXIS = 1
LB_SPTP_HEADS = 32
LB_SPTP_CHUNK = 1280
LB_SPTP_CHUNK_LOCAL = LB_SPTP_CHUNK // LB_SPTP_SP
LB_SPTP_K_CHUNK = 320


def _sptp_loudbox_inputs(
    mesh,
    k_capacity,
    seed,
    *,
    sp=LB_SPTP_SP,
    tp=LB_SPTP_TP,
    sp_axis=LB_SPTP_SP_AXIS,
    tp_axis=LB_SPTP_TP_AXIS,
    heads=LB_SPTP_HEADS,
    chunk_global=LB_SPTP_CHUNK,
):
    chunk_local = chunk_global // sp
    q_host, k_natural, w_host = _global_inputs(heads, chunk_global, k_capacity, seed=seed)
    k_reconstructed = _to_tp_inner_reconstructed(k_natural, sp=sp, tp=tp, chunk_local=chunk_local)
    mesh_shape = [0, 0]
    mesh_shape[sp_axis] = sp
    mesh_shape[tp_axis] = tp
    shard_dims = [None, None]
    shard_dims[sp_axis] = 2
    sp_shard = ttnn.ShardTensor2dMesh(mesh, mesh_shape=tuple(mesh_shape), dims=tuple(shard_dims))
    k_local = ttnn.from_torch(
        k_reconstructed,
        device=mesh,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat8_b,
        mesh_mapper=sp_shard,
    )
    k_gathered = ttnn.from_torch(
        torch.zeros_like(k_natural),
        device=mesh,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat8_b,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )
    q_dev = ttnn.from_torch(
        q_host,
        device=mesh,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat8_b,
        mesh_mapper=sp_shard,
    )
    w_dev = ttnn.from_torch(
        w_host,
        device=mesh,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        mesh_mapper=sp_shard,
    )
    q_dev = ttnn.mesh_partition(q_dev, dim=2, cluster_axis=tp_axis)
    w_dev = ttnn.mesh_partition(w_dev, dim=2, cluster_axis=tp_axis)
    return q_host, k_natural, w_host, q_dev, k_local, k_gathered, w_dev


def _sptp_loudbox_ref(
    q_host,
    k_natural,
    w_host,
    chunk_start,
    *,
    sp=LB_SPTP_SP,
    tp=LB_SPTP_TP,
    sp_axis=LB_SPTP_SP_AXIS,
    tp_axis=LB_SPTP_TP_AXIS,
):
    chunk_local = q_host.shape[2] // sp
    q_per_device = chunk_local // tp
    mesh_shape = [0, 0]
    mesh_shape[sp_axis] = sp
    mesh_shape[tp_axis] = tp
    refs = []
    for mesh_row in range(mesh_shape[0]):
        for mesh_col in range(mesh_shape[1]):
            coord = (mesh_row, mesh_col)
            sp_rank = coord[sp_axis]
            tp_rank = coord[tp_axis]
            q_start = sp_rank * chunk_local + tp_rank * q_per_device
            q_slice = slice(q_start, q_start + q_per_device)
            refs.append(
                indexer_score_dsa_ref(
                    q_host[:, :, q_slice, :],
                    k_natural,
                    w_host[:, :, q_slice, :],
                    chunk_start + q_start,
                )
            )
    return torch.cat(refs, dim=2)


def _run_sptp_loudbox_score(
    semaphores,
    subdevice_id,
    q_dev,
    k_local,
    k_gathered,
    w_dev,
    *,
    chunk_start,
    kv_len,
    tp_sharded,
    sp_axis=LB_SPTP_SP_AXIS,
    tp_axis=LB_SPTP_TP_AXIS,
    topology=ttnn.Topology.Linear,
    num_links=1,
    chunk_local=LB_SPTP_CHUNK_LOCAL,
):
    return ttnn.experimental.ring_indexer_score_dsa(
        q_dev,
        k_gathered,
        w_dev,
        k_local,
        semaphores,
        cluster_axis=sp_axis,
        topology=topology,
        num_links=num_links,
        ag_sub_device_id=subdevice_id,
        chunk_start_idx=chunk_start,
        kv_len=kv_len,
        seq_subshard_axis=tp_axis,
        block_cyclic_sp_axis=sp_axis,
        block_cyclic_chunk_local=chunk_local,
        block_cyclic_cache_tp_sharded=tp_sharded,
        program_config=ttnn.IndexerScoreProgramConfig(
            q_chunk_size=32,
            k_chunk_size=LB_SPTP_K_CHUNK,
            head_group_size=0,
        ),
    )


def _sptp_loudbox_output(out):
    shards = [ttnn.to_torch(shard) for shard in ttnn.get_device_tensors(out.cpu())]
    return torch.cat(shards, dim=2)


@pytest.mark.parametrize("tp_sharded", [False, True], ids=["control", "tp_sharded"])
def test_indexer_score_sptp_loudbox_tp_sharded_kv_repro(tp_sharded):
    """Score a TP-inner reconstructed K cache on a LoudBox SP2 x TP4 mesh."""
    if ttnn.get_num_devices() != 8:
        pytest.skip("SP2 x TP4 fused indexer reproduction requires an exact eight-device LoudBox")
    assert LB_SPTP_CHUNK_LOCAL == 640
    assert LB_SPTP_CHUNK_LOCAL // LB_SPTP_TP == 160
    assert LB_SPTP_K_CHUNK // 32 == 10

    mesh, semaphores, subdevice_id, stall_group = _open_full_mesh_ccl(
        (LB_SPTP_SP, LB_SPTP_TP), fabric_config=ttnn.FabricConfig.FABRIC_1D
    )
    try:
        q_host, k_natural, w_host, q_dev, k_local, k_gathered, w_dev = _sptp_loudbox_inputs(
            mesh, LB_SPTP_CHUNK, seed=42
        )
        out = _run_sptp_loudbox_score(
            semaphores,
            subdevice_id,
            q_dev,
            k_local,
            k_gathered,
            w_dev,
            chunk_start=0,
            kv_len=LB_SPTP_CHUNK,
            tp_sharded=tp_sharded,
        )
        ttnn.synchronize_device(mesh, sub_device_ids=stall_group)
        out_host = _sptp_loudbox_output(out)
        reference = _sptp_loudbox_ref(q_host, k_natural, w_host, chunk_start=0)
        assert_indexer_match(out_host, reference, LB_SPTP_CHUNK, LB_SPTP_CHUNK, check_neg=True)
    finally:
        _close_full_mesh_ccl(mesh)


def test_indexer_score_sptp_loudbox_tp_sharded_multislab_partial_prefix_cache_reuse():
    """Cover TP-stripe resets inside a KC unit and runtime-scalar cache hits."""
    if ttnn.get_num_devices() != 8:
        pytest.skip("SP2 x TP4 fused indexer reproduction requires an exact eight-device LoudBox")
    k_capacity = 3 * LB_SPTP_CHUNK
    # Each TP stripe is 15 tiles wide, so the KC=10 unit at physical offset 10 crosses a stripe
    # boundary. At kv_len=960 its first five columns are invalid and its last five reset to valid keys.
    assert (k_capacity // (LB_SPTP_SP * LB_SPTP_TP)) // 32 == 15

    mesh, semaphores, subdevice_id, stall_group = _open_full_mesh_ccl(
        (LB_SPTP_SP, LB_SPTP_TP), fabric_config=ttnn.FabricConfig.FABRIC_1D
    )
    try:
        q_host, k_natural, w_host, q_dev, k_local, k_gathered, w_dev = _sptp_loudbox_inputs(mesh, k_capacity, seed=43)
        mesh.enable_program_cache()
        entries_before = mesh.num_program_cache_entries()
        entries_after_compile = None
        for dispatch, (chunk_start, kv_len) in enumerate(((0, 960), (LB_SPTP_CHUNK, 1920))):
            out = _run_sptp_loudbox_score(
                semaphores,
                subdevice_id,
                q_dev,
                k_local,
                k_gathered,
                w_dev,
                chunk_start=chunk_start,
                kv_len=kv_len,
                tp_sharded=True,
            )
            ttnn.synchronize_device(mesh, sub_device_ids=stall_group)
            out_host = _sptp_loudbox_output(out)
            ttnn.deallocate(out)

            reference = _sptp_loudbox_ref(q_host, k_natural[:, :, :kv_len], w_host, chunk_start=chunk_start)
            assert_indexer_match(out_host[:, :, :, :kv_len], reference, LB_SPTP_CHUNK, kv_len, check_neg=True)
            if dispatch == 0:
                entries_after_compile = mesh.num_program_cache_entries()
                assert entries_after_compile > entries_before
            else:
                assert (
                    mesh.num_program_cache_entries() == entries_after_compile
                ), "changing kv_len/chunk_start recompiled the fused Ring program"
    finally:
        _close_full_mesh_ccl(mesh)


def test_indexer_score_sptp_loudbox_ring_partial_readiness():
    """Exercise TP-inner K on two four-rank rings, including the AG midpoint readiness gate."""
    if ttnn.get_num_devices() != 8:
        pytest.skip("SP4 x TP2 fused indexer readiness coverage requires an exact eight-device LoudBox")
    sp, tp = 4, 2
    sp_axis, tp_axis = 1, 0
    chunk_global = LB_SPTP_CHUNK
    chunk_local = chunk_global // sp
    k_capacity = 3 * chunk_global
    kv_len = 960
    assert chunk_local // tp == 160
    assert (k_capacity // (sp * tp)) // 32 == 15

    mesh, semaphores, subdevice_id, stall_group = _open_full_mesh_ccl((tp, sp))
    try:
        q_host, k_natural, w_host, q_dev, k_local, k_gathered, w_dev = _sptp_loudbox_inputs(
            mesh,
            k_capacity,
            seed=44,
            sp=sp,
            tp=tp,
            sp_axis=sp_axis,
            tp_axis=tp_axis,
            chunk_global=chunk_global,
        )
        out = _run_sptp_loudbox_score(
            semaphores,
            subdevice_id,
            q_dev,
            k_local,
            k_gathered,
            w_dev,
            chunk_start=0,
            kv_len=kv_len,
            tp_sharded=True,
            sp_axis=sp_axis,
            tp_axis=tp_axis,
            topology=ttnn.Topology.Ring,
            num_links=2,
            chunk_local=chunk_local,
        )
        ttnn.synchronize_device(mesh, sub_device_ids=stall_group)
        out_host = _sptp_loudbox_output(out)
        reference = _sptp_loudbox_ref(
            q_host,
            k_natural[:, :, :kv_len],
            w_host,
            chunk_start=0,
            sp=sp,
            tp=tp,
            sp_axis=sp_axis,
            tp_axis=tp_axis,
        )
        assert_indexer_match(out_host[:, :, :, :kv_len], reference, chunk_global, kv_len, check_neg=True)
    finally:
        _close_full_mesh_ccl(mesh)


@pytest.mark.skipif(
    not os.getenv("TT_METAL_SIMULATOR")
    and (os.getenv("MESH_DEVICE") != "TG" or os.getenv("TT_METAL_RING_INDEXER_RUN_32_RANK_ACCURACY") != "1"),
    reason="requires Galaxy/simulator opt-in for the 32-rank complete-mesh indexer test",
)
def test_indexer_score_full_mesh_galaxy_8x4_accuracy():
    """Exercise the fixed 32-entry readiness tables at their supported Galaxy limit."""
    if ttnn.get_num_devices() != 32:
        pytest.skip("8x4 full-mesh indexer coverage requires exactly 32 available devices")
    _run_full_mesh_accuracy_case((8, 4), block_cyclic=False)


def test_indexer_score_full_mesh_indexed_bounded_gather_cache_hit_and_determinism():
    """Combine indexed ND-sharded K, bounded transport, rotated causal patching, and cache reuse."""
    if ttnn.get_num_devices() != 8:
        pytest.skip("complete 2x4 cache-hit coverage requires the exact physical eight-device LoudBox")

    mesh_shape = (2, 4)
    ring_size = 8
    heads, num_users, local_sq = 8, 3, 64
    chunk_global = ring_size * local_sq
    t_alloc = 3 * chunk_global
    local_t = t_alloc // ring_size
    mesh, semaphores, subdevice_id, stall_group = _open_full_mesh_ccl(mesh_shape)
    try:
        q_g, _, w_g = _global_inputs(heads, chunk_global, t_alloc, seed=2027)
        k_nat = torch.cat(
            [_global_inputs(heads, chunk_global, t_alloc, seed=2100 + user)[1] for user in range(num_users)], dim=0
        )
        k_bc = torch.cat(
            [_to_slab(k_nat[user : user + 1], ring_size, chunk_global) for user in range(num_users)], dim=0
        )
        q_dev, w_dev, k_local_i, k_gathered = _full_mesh_inputs(mesh, q_g, w_g, k_bc)
        k_local = ttnn.to_memory_config(k_local_i, _nd_sharded_dram_config(mesh, rows_per_shard=32))
        ttnn.deallocate(k_local_i)

        mesh.enable_program_cache()
        mesh.clear_program_cache()

        def _score(slot, chunk_start, kv_len):
            out = ttnn.experimental.ring_indexer_score_dsa(
                q_dev,
                k_gathered,
                w_dev,
                k_local,
                semaphores,
                cluster_axis=None,
                topology=ttnn.Topology.Ring,
                num_links=2,
                ag_sub_device_id=subdevice_id,
                chunk_start_idx=chunk_start,
                cache_batch_idx=slot,
                kv_len=kv_len,
                block_cyclic_chunk_local=local_sq,
                program_config=glx_config(heads),
            )
            ttnn.synchronize_device(mesh, sub_device_ids=stall_group)
            return ttnn.to_torch(out, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=2))

        large_kv_len = chunk_global + 32
        out0 = _score(slot=1, chunk_start=32, kv_len=large_kv_len)
        entries_after_first = mesh.num_program_cache_entries()
        scratch_after_large = [ttnn.to_torch(tensor).clone() for tensor in ttnn.get_device_tensors(k_gathered)]
        ref0 = _straddle_ref(q_g, k_nat[1:2, :, :large_kv_len, :], w_g, ring_size, chunk_global, 32, large_kv_len)
        assert_indexer_match(out0[:, :, :, :large_kv_len], ref0, chunk_global, large_kv_len, check_neg=True)
        _assert_remote_gather_slots(
            k_local,
            k_gathered,
            ring_size,
            valid_local_rows=2 * local_sq,
            cache_batch_idx=1,
        )
        for scratch in scratch_after_large:
            for tensor_rank in range(ring_size):
                tail = scratch[:, :, tensor_rank * local_t + 2 * local_sq : (tensor_rank + 1) * local_t, :]
                assert torch.count_nonzero(tail) == 0, "bounded gather wrote beyond its slab-rounded extent"

        out1 = _score(slot=2, chunk_start=0, kv_len=chunk_global)
        out2 = _score(slot=2, chunk_start=0, kv_len=chunk_global)
        assert mesh.num_program_cache_entries() == entries_after_first, "runtime scalar changes recompiled"
        assert torch.equal(out1, out2), "cache-hit replay is not bit-exact"
        ref1 = _straddle_ref(q_g, k_nat[2:3, :, :chunk_global, :], w_g, ring_size, chunk_global, 0, chunk_global)
        assert_indexer_match(out1[:, :, :, :chunk_global], ref1, chunk_global, chunk_global, check_neg=True)
        _assert_remote_gather_slots(k_local, k_gathered, ring_size, valid_local_rows=local_sq, cache_batch_idx=2)
        scratch_after_small = [ttnn.to_torch(tensor) for tensor in ttnn.get_device_tensors(k_gathered)]
        any_first_slab_changed = False
        for before, after in zip(scratch_after_large, scratch_after_small):
            for tensor_rank in range(ring_size):
                base = tensor_rank * local_t
                any_first_slab_changed |= not torch.equal(
                    before[:, :, base : base + local_sq, :], after[:, :, base : base + local_sq, :]
                )
                assert torch.equal(
                    before[:, :, base + local_sq : base + 2 * local_sq, :],
                    after[:, :, base + local_sq : base + 2 * local_sq, :],
                ), "shrinking kv_len rewrote the second slab on a cache hit"
        assert any_first_slab_changed, "switching cache slots did not update any gathered first-slab data"
    finally:
        if mesh is not None:
            mesh.disable_and_clear_program_cache()
        _close_full_mesh_ccl(mesh)


def test_indexer_score_full_mesh_rejects_invalid_contracts(expect_error):
    """Reject invalid full-mesh topology, axis roles, placements, replication, and link requests on host."""
    if ttnn.get_num_devices() != 8:
        pytest.skip("complete 2x4 negative coverage requires the exact physical eight-device LoudBox")

    mesh, semaphores, subdevice_id, _ = _open_full_mesh_ccl((2, 4))
    try:
        heads, local_sq, ring_size = 8, 64, 8
        chunk_global, t_len = ring_size * local_sq, 2 * ring_size * local_sq
        q_g, k_nat, w_g = _global_inputs(heads, chunk_global, t_len, seed=2028)
        q_dev, w_dev, k_local, k_gathered = _full_mesh_inputs(mesh, q_g, w_g, k_nat)

        def _call(**overrides):
            q_arg = overrides.pop("q", q_dev)
            k_arg = overrides.pop("k", k_gathered)
            kwargs = dict(
                cluster_axis=None,
                topology=ttnn.Topology.Ring,
                num_links=2,
                ag_sub_device_id=subdevice_id,
                chunk_start_idx=chunk_global,
                program_config=glx_config(heads),
            )
            kwargs.update(overrides)
            return ttnn.experimental.ring_indexer_score_dsa(
                q_arg,
                k_arg,
                w_dev,
                k_local,
                semaphores,
                **kwargs,
            )

        with expect_error(RuntimeError, "requires Ring topology"):
            _call(topology=ttnn.Topology.Linear)
        with expect_error(RuntimeError, "does not allow seq_subshard_axis"):
            _call(seq_subshard_axis=0)
        with expect_error(RuntimeError, "does not allow block_cyclic_sp_axis"):
            _call(block_cyclic_sp_axis=0, block_cyclic_chunk_local=local_sq)
        with expect_error(RuntimeError, "requires num_links > 0"):
            _call(num_links=0)
        with expect_error(RuntimeError, "could not resolve a direct-neighbor full-mesh snake ring"):
            _call(num_links=99)

        axis_mapper = ttnn.ShardTensor2dMesh(mesh, mesh_shape=(2, 4), dims=(None, 2))
        axis_q = ttnn.from_torch(
            q_g, device=mesh, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, mesh_mapper=axis_mapper
        )
        with expect_error(RuntimeError, "sequence dim 2 to be sharded across all"):
            _call(q=axis_q)

        nonreplicated_k = ttnn.from_torch(
            torch.zeros_like(k_nat),
            device=mesh,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
            mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=2),
        )
        with expect_error(RuntimeError, "persistent gathered K buffer replicated"):
            _call(k=nonreplicated_k)
    finally:
        _close_full_mesh_ccl(mesh)


@pytest.mark.requires_host_iommu
@pytest.mark.parametrize("mesh_device", [(8, 1)], ids=["ring8"], indirect=True)
@pytest.mark.parametrize("device_params", [_RING8_PARTIAL_DEVICE_PARAMS], indirect=True)
def test_indexer_score_ring8_partial_readiness_reference_cache_hit(mesh_device):
    """Reference-check the production Ring two-marker protocol, including its cache-hit runtime patch.

    This test keeps query work small while using a large BF16 K capacity to exercise the bank-owned schedule's
    midpoint/completion protocol. Two runtime prefixes exercise different marker locations on one cached program.
    Sampled first/last rows on every rank cover both directions without constructing a capacity-sized CPU score.
    """
    sp = mesh_device.shape[0]
    heads = 4
    q_per_rank = 32
    chunk_global = sp * q_per_rank
    k_capacity = 128 * 1024
    kv_lens = (56320, 112640)
    dim = 128
    grid = mesh_device.compute_with_storage_grid_size()
    worker_cores = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))})
    subdevice_id = ttnn.SubDeviceId(0)
    stall_group = [subdevice_id]
    manager = mesh_device.create_sub_device_manager([ttnn.SubDevice([worker_cores])], 0)
    mesh_device.load_sub_device_manager(manager)
    mesh_device.set_sub_device_stall_group(stall_group)
    ccl_semaphores = [ttnn.create_global_semaphore(mesh_device, worker_cores, 0) for _ in range(2)]
    try:
        q_g, k_nat, w_g = _global_inputs(heads, chunk_global, k_capacity, seed=4242)
        k_bc = _to_slab(k_nat, sp, chunk_global)
        sp_shard = ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=(sp, 1), dims=(2, None))
        q_dev = ttnn.from_torch(
            q_g, device=mesh_device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, mesh_mapper=sp_shard
        )
        w_dev = to_device(w_g, mesh_device, mesh_mapper=sp_shard)
        k_local = ttnn.from_torch(
            k_bc,
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=sp_shard,
        )
        k_gathered = ttnn.from_torch(
            torch.zeros_like(k_nat),
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )
        program_config = ttnn.IndexerScoreProgramConfig(
            q_chunk_size=32,
            k_chunk_size=320,
            head_group_size=0,
        )

        def _score(kv_len):
            chunk_start = kv_len - chunk_global
            out = ttnn.experimental.ring_indexer_score_dsa(
                q_dev,
                k_gathered,
                w_dev,
                k_local,
                ccl_semaphores,
                cluster_axis=0,
                topology=ttnn.Topology.Ring,
                num_links=2,
                ag_sub_device_id=subdevice_id,
                chunk_start_idx=chunk_start,
                kv_len=kv_len,
                block_cyclic_sp_axis=0,
                block_cyclic_chunk_local=q_per_rank,
                program_config=program_config,
            )
            ttnn.synchronize_device(mesh_device, sub_device_ids=stall_group)
            out_t = ttnn.to_torch(out, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=2))
            ttnn.deallocate(out)
            return out_t, chunk_start

        entries_before = mesh_device.num_program_cache_entries()
        for dispatch, kv_len in enumerate(kv_lens):
            out_t, chunk_start = _score(kv_len)
            if dispatch == 0:
                entries_after_compile = mesh_device.num_program_cache_entries()
                assert entries_after_compile > entries_before
            else:
                assert (
                    mesh_device.num_program_cache_entries() == entries_after_compile
                ), "changing kv_len/midpoint recompiled instead of patching the cached Ring program"

            for rank in range(sp):
                for local_row in (0, q_per_rank - 1):
                    global_row = rank * q_per_rank + local_row
                    row = slice(global_row, global_row + 1)
                    ref = indexer_score_dsa_ref(
                        q_g[:, :, row, :],
                        k_nat[:, :, :kv_len, :],
                        w_g[:, :, row, :],
                        chunk_start + global_row,
                    )
                    assert_indexer_match(out_t[:, :, row, :kv_len], ref, sq=1, t=kv_len, check_neg=False)
        logger.info("Ring-8 partial readiness matched sampled reference across a kv_len cache hit")
    finally:
        mesh_device.reset_sub_device_stall_group()
        mesh_device.clear_loaded_sub_device_manager()


def test_indexer_score_ring4_fused_rejects_head_streaming(expect_error):
    """The fused path requires all heads resident; a streaming config (0 < head_group_size < Hi) must be
    rejected at validate, not silently mis-scheduled. head-independent -> one representative case."""
    heads = 16  # head_group_size=8 is a streaming config (0 < 8 < 16)
    submesh, parent, ccl_semaphores, subdevice_id, stall_group = _open_ring4_ccl()
    try:
        q_g, k_nat, w_g = _global_inputs(heads, CHUNK_GLOBAL, T, seed=42)
        q_dev, w_dev, k_local, k_gathered = _fused_dev_inputs(submesh, q_g, w_g, k_nat)
        base = glx_config(heads)
        streaming_cfg = ttnn.IndexerScoreProgramConfig(
            q_chunk_size=base.q_chunk_size, k_chunk_size=base.k_chunk_size, head_group_size=heads // 2
        )
        with expect_error(RuntimeError, "head_group_size must be 0 or Hi"):
            ttnn.experimental.ring_indexer_score_dsa(
                q_dev,
                k_gathered,
                w_dev,
                k_local,
                ccl_semaphores,
                cluster_axis=SP_AXIS,
                topology=ttnn.Topology.Linear,
                num_links=1,
                ag_sub_device_id=subdevice_id,
                program_config=streaming_cfg,
            )
    finally:
        _close_ring4_ccl(parent, submesh, stall_group)
