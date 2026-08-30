# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Multi-device indexer_score correctness on a four-device Blackhole box.

This module covers the classic and Ring-fused frontends on the mesh layouts that fit in four chips:

  * SP-only on a (1, 4) or (4, 1) mesh.
  * 2D SP×TP on a (2, 2) mesh: sp=2 ring (cluster_axis) × tp=2 query seq sub-shard (seq_subshard_axis). K
    stays SP-sharded + TP-replicated so the AG is unchanged; TP sub-shards the query rows and seq_subshard_axis
    threads each device's tp sub-offset into the causal score. Fused analogue of the classic-path
    test_indexer_score.py::test_indexer_score_sp2_tp2_seq_subshard_rotated.

The Ring tests open the target mesh directly. Gathered buffers are seeded with zeros so correct scores also
verify device-side local sourcing.

Run:  scripts/run_safe_pytest.sh tests/ttnn/nightly/unit_tests/operations/experimental/indexer_score/test_ring_indexer_score_dsa_4d.py
"""

import pytest
import torch
from loguru import logger

import ttnn

from tests.ttnn.nightly.unit_tests.operations.experimental.indexer_score.test_indexer_score import (
    assert_grouped_match,
    assert_indexer_match,
    assert_pooled_match,
    glx_config,
    indexer_score_dsa_ref,
    indexer_score_msa_ref,
    _make_paged_k,
    _msa_scale_w,
    _nd_sharded_dram_config,
    _global_inputs,
    _axis_dims,
    _msa_per_sp_ref,
    _per_sp_ref,
    _shard_1d,
    _straddle_msa_pooled_ref,
    _straddle_ref,
    _to_mesh,
    _to_slab,
    BLOCK_POOL_BS,
    M3_QB_HEADS,
    M3_QB_SCALE,
    QB_CHUNK,
    QB_SQ,
    QB_SP,
    QB_T,
    QB_HISTORY,
    QB_DIM,
    QB_CASES,
    QB_IDS,
    QB2_CHUNK,
    QB2_SP,
    QB2_SP_AXIS,
    QB2_T,
    QB2_TP,
    ST_CHUNK,
    ST_CS,
    ST_MSA_T,
    ST_T,
)
from tests.ttnn.nightly.unit_tests.operations.experimental.indexer_score.test_ring_indexer_score_dsa import (
    _run_full_mesh_accuracy_case,
)
from tests.ttnn.nightly.unit_tests.operations.experimental.indexer_score.ring_indexer_score_test_utils import (
    _to_tp_inner_reconstructed,
)

DRAM = ttnn.DRAM_MEMORY_CONFIG

pytestmark = [
    pytest.mark.skipif(not ttnn.device.is_blackhole(), reason="indexer_score is Blackhole-only"),
    pytest.mark.skipif(ttnn.get_num_devices() != 4, reason="needs an exact 4-device Blackhole box"),
]


def _open_ccl(mesh_shape, *, fabric_config=ttnn.FabricConfig.FABRIC_1D):
    """Open `mesh_shape` directly (no parent carve), load a worker sub-device, make 2 ccl semaphores (the two
    ring directions). Mirrors ring_indexer_score_test_utils._open_ring4_ccl without the (2,4)->(1,4) submesh
    step, so it
    runs on a 4-chip box. Returns (mesh, ccl_semaphores, worker_sub_device_id, stall_group)."""
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
        mgr = mesh.create_sub_device_manager([worker_sub_device], 0)
        mesh.load_sub_device_manager(mgr)
        mesh.set_sub_device_stall_group(stall_group)
        ccl_semaphores = [ttnn.create_global_semaphore(mesh, ccl_crs, 0) for _ in range(2)]
        return mesh, ccl_semaphores, worker_sub_device_id, stall_group
    except Exception:
        if mesh is not None:
            ttnn.close_mesh_device(mesh)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
        raise


def _close_ccl(mesh):
    try:
        try:
            mesh.reset_sub_device_stall_group()
            mesh.clear_loaded_sub_device_manager()
        finally:
            ttnn.close_mesh_device(mesh)
    finally:
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


# ---- Classic multi-device frontend ---------------------------------------------------------------


@pytest.mark.parametrize("mesh_device", [QB_SP], indirect=True)
@pytest.mark.parametrize("case_id, heads", QB_CASES, ids=QB_IDS)
def test_indexer_score_qb_per_device_chunk_start(mesh_device, case_id, heads):
    """Derive each SP rank's causal start from its mesh coordinate."""
    q_g, k_g, w_g, q_dev, k_dev, w_dev = _shard_1d(mesh_device, heads, seed=42)

    out = ttnn.experimental.indexer_score_dsa(q_dev, k_dev, w_dev, program_config=glx_config(heads))
    out_t = ttnn.to_torch(out, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=2))

    ref = _per_sp_ref(q_g, k_g, w_g, QB_SP, QB_HISTORY)
    assert_indexer_match(out_t, ref, QB_CHUNK, QB_T, check_neg=True)


@pytest.mark.parametrize("mesh_device", [QB_SP], indirect=True)
@pytest.mark.parametrize("case_id, heads", QB_CASES, ids=QB_IDS)
def test_indexer_score_qb_one_compile_all_chunk_starts(mesh_device, case_id, heads):
    """Patch different causal starts without recompiling the program."""
    _, _, _, q_dev, k_dev, w_dev = _shard_1d(mesh_device, heads, seed=7)
    bases = [QB_HISTORY, QB_HISTORY - QB_SQ, QB_HISTORY - 2 * QB_SQ]

    entries_before = mesh_device.num_program_cache_entries()
    for base in bases:
        ttnn.experimental.indexer_score_dsa(
            q_dev, k_dev, w_dev, chunk_start_idx=base, program_config=glx_config(heads)
        ).deallocate()

    added = mesh_device.num_program_cache_entries() - entries_before
    assert added == 1, f"expected 1 program-cache entry across 3 distinct chunk_start bases, got {added}"


@pytest.mark.parametrize("mesh_device", [(QB2_SP, QB2_TP)], ids=["2x2"], indirect=True)
@pytest.mark.parametrize("case_id, heads", QB_CASES, ids=QB_IDS)
def test_indexer_score_qb_sp2_tp2(mesh_device, case_id, heads):
    """Shard sequence over SP and heads over TP while keeping causal starts constant across TP."""
    q_g, k_g, w_g = _global_inputs(heads, QB2_CHUNK, QB2_T, seed=42)
    mesh_shape = tuple(mesh_device.shape)
    qw_dims = _axis_dims(sp_dim=2, tp_dim=1)
    shard_qw = ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=mesh_shape, dims=qw_dims)
    q_dev = _to_mesh(mesh_device, q_g, ttnn.bfloat16, shard_qw)
    w_dev = _to_mesh(mesh_device, w_g, ttnn.bfloat16, shard_qw)
    k_dev = _to_mesh(mesh_device, k_g, ttnn.bfloat8_b, ttnn.ReplicateTensorToMesh(mesh_device))

    out = ttnn.experimental.indexer_score_dsa(
        q_dev,
        k_dev,
        w_dev,
        seq_shard_axes=[QB2_SP_AXIS],
        program_config=glx_config(heads // QB2_TP),
    )
    out_t = ttnn.to_torch(
        out, mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, mesh_shape=mesh_shape, dims=qw_dims)
    )
    out_t = out_t.float().sum(dim=1, keepdim=True)

    ref = _per_sp_ref(q_g, k_g, w_g, QB2_SP, QB_HISTORY)
    assert_indexer_match(out_t, ref, QB2_CHUNK, QB2_T, check_neg=True)


@pytest.mark.parametrize("mesh_device", [QB_SP], indirect=True)
def test_indexer_score_qb_msa_per_device_chunk_start(mesh_device):
    """Run MSA over SP=4 with one causal start per rank."""
    q_g, k_g, _ = _global_inputs(M3_QB_HEADS, QB_CHUNK, QB_T, seed=42)
    shard = ttnn.ShardTensorToMesh(mesh_device, dim=2)
    q_dev = _to_mesh(mesh_device, q_g, ttnn.bfloat16, shard)
    k_dev = _to_mesh(mesh_device, k_g, ttnn.bfloat8_b, ttnn.ReplicateTensorToMesh(mesh_device))

    out = ttnn.experimental.indexer_score_msa(
        q_dev, k_dev, scale=M3_QB_SCALE, num_groups=1, program_config=glx_config(M3_QB_HEADS)
    )
    out_t = ttnn.to_torch(out, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=2))

    ref = _msa_per_sp_ref(q_g, k_g, QB_SP, QB_HISTORY)
    assert_indexer_match(out_t, ref, QB_CHUNK, QB_T, check_neg=True)


@pytest.mark.parametrize("mesh_device", [(QB2_SP, QB2_TP)], ids=["2x2"], indirect=True)
def test_indexer_score_qb_msa_sp2_tp2(mesh_device):
    """Run MSA on SP=2 x TP=2 and sum the TP head partials."""
    q_g, k_g, _ = _global_inputs(M3_QB_HEADS, QB2_CHUNK, QB2_T, seed=42)
    mesh_shape = tuple(mesh_device.shape)
    q_dims = _axis_dims(sp_dim=2, tp_dim=1)
    shard_q = ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=mesh_shape, dims=q_dims)
    q_dev = _to_mesh(mesh_device, q_g, ttnn.bfloat16, shard_q)
    k_dev = _to_mesh(mesh_device, k_g, ttnn.bfloat8_b, ttnn.ReplicateTensorToMesh(mesh_device))

    out = ttnn.experimental.indexer_score_msa(
        q_dev,
        k_dev,
        seq_shard_axes=[QB2_SP_AXIS],
        scale=M3_QB_SCALE,
        num_groups=1,
        program_config=glx_config(M3_QB_HEADS // QB2_TP),
    )
    out_t = ttnn.to_torch(out, mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, mesh_shape=mesh_shape, dims=q_dims))
    out_t = out_t.float().sum(dim=1, keepdim=True)

    ref = _msa_per_sp_ref(q_g, k_g, QB2_SP, QB_HISTORY)
    assert_indexer_match(out_t, ref, QB2_CHUNK, QB2_T, check_neg=True)


@pytest.mark.parametrize("mesh_device", [(QB_SP, 1)], ids=["sp4"], indirect=True)
@pytest.mark.parametrize("case_id, heads", QB_CASES, ids=QB_IDS)
def test_indexer_score_qb_block_cyclic(mesh_device, case_id, heads):
    """Read an SP=4 block-cyclic K cache in natural token order."""
    q_g, k_nat, w_g = _global_inputs(heads, QB_CHUNK, QB_T, seed=42)
    k_bc = _to_slab(k_nat, QB_SP, QB_CHUNK)
    mesh_shape = tuple(mesh_device.shape)
    shard = ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=mesh_shape, dims=(2, None))
    q_dev = _to_mesh(mesh_device, q_g, ttnn.bfloat16, shard)
    w_dev = _to_mesh(mesh_device, w_g, ttnn.bfloat16, shard)
    k_dev = _to_mesh(mesh_device, k_bc, ttnn.bfloat8_b, ttnn.ReplicateTensorToMesh(mesh_device))

    out = ttnn.experimental.indexer_score_dsa(
        q_dev,
        k_dev,
        w_dev,
        seq_shard_axes=[0],
        block_cyclic_sp_axis=0,
        block_cyclic_chunk_local=QB_SQ,
        program_config=glx_config(heads),
    )
    out_t = ttnn.to_torch(out, mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, mesh_shape=mesh_shape, dims=(2, 1)))
    ref = _per_sp_ref(q_g, k_nat, w_g, QB_SP, QB_HISTORY)
    assert_indexer_match(out_t, ref, QB_CHUNK, QB_T, check_neg=True)


@pytest.mark.parametrize("mesh_device", [(2, 2)], ids=["2x2"], indirect=True)
@pytest.mark.parametrize("case_id, heads", QB_CASES, ids=QB_IDS)
def test_indexer_score_qb_both_axes_seq(mesh_device, case_id, heads, expect_error):
    """Shard query sequence over both axes of a 2x2 mesh."""
    sp, chunk_global, t = 2, 256, 512
    q_g, k_nat, w_g = _global_inputs(heads, chunk_global, t, seed=42)
    k_bc = _to_slab(k_nat, sp, chunk_global)
    shard = ttnn.ShardTensorToMesh(mesh_device, dim=2)
    q_dev = _to_mesh(mesh_device, q_g, ttnn.bfloat16, shard)
    w_dev = _to_mesh(mesh_device, w_g, ttnn.bfloat16, shard)
    k_dev = _to_mesh(mesh_device, k_bc, ttnn.bfloat8_b, ttnn.ReplicateTensorToMesh(mesh_device))
    kwargs = {
        "block_cyclic_sp_axis": 0,
        "block_cyclic_chunk_local": chunk_global // sp,
        "program_config": glx_config(heads),
    }

    out = ttnn.experimental.indexer_score_dsa(q_dev, k_dev, w_dev, seq_shard_axes=[], **kwargs)
    out_t = ttnn.to_torch(out, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=2))
    ref = indexer_score_dsa_ref(q_g, k_nat, w_g, t - chunk_global)
    assert_indexer_match(out_t, ref, chunk_global, t, check_neg=True)

    with expect_error(RuntimeError, "needs the TP axis"):
        ttnn.experimental.indexer_score_dsa(q_dev, k_dev, w_dev, seq_shard_axes=[0], **kwargs)


@pytest.mark.parametrize("mesh_device", [(1, 4)], ids=["sp1xtp4"], indirect=True)
def test_indexer_score_sp1_tp4_seq_subshard(mesh_device):
    """Apply each TP rank's query offset when the SP extent is one."""
    heads, chunk, t, chunk_start = 8, 256, 512, 128
    q_g, k_g, w_g = _global_inputs(heads, chunk, t, seed=42)
    mesh_shape = tuple(mesh_device.shape)
    shard_tp_seq = ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=mesh_shape, dims=(None, 2))
    q_dev = _to_mesh(mesh_device, q_g, ttnn.bfloat16, shard_tp_seq)
    w_dev = _to_mesh(mesh_device, w_g, ttnn.bfloat16, shard_tp_seq)
    k_dev = _to_mesh(mesh_device, k_g, ttnn.bfloat16, ttnn.ReplicateTensorToMesh(mesh_device))
    kwargs = {
        "seq_shard_axes": [0, 1],
        "block_cyclic_sp_axis": 0,
        "block_cyclic_chunk_local": chunk,
        "program_config": ttnn.IndexerScoreProgramConfig(q_chunk_size=32, k_chunk_size=64, head_group_size=0),
    }

    out = ttnn.experimental.indexer_score_dsa(q_dev, k_dev, w_dev, chunk_start_idx=chunk_start, **kwargs)
    out_t = ttnn.to_torch(out, mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, mesh_shape=mesh_shape, dims=(1, 2)))
    ref = indexer_score_dsa_ref(q_g, k_g, w_g, chunk_start)
    assert_indexer_match(out_t, ref, chunk, t, check_neg=True)

    out_default = ttnn.experimental.indexer_score_dsa(q_dev, k_dev, w_dev, **kwargs)
    out_default_t = ttnn.to_torch(
        out_default, mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, mesh_shape=mesh_shape, dims=(1, 2))
    )
    ref_default = indexer_score_dsa_ref(q_g, k_g, w_g, t - chunk)
    assert_indexer_match(out_default_t, ref_default, chunk, t, check_neg=True)


@pytest.mark.parametrize("mesh_device", [(2, 2)], ids=["sp2xtp2"], indirect=True)
def test_indexer_score_sp2_tp2_seq_subshard_rotated(mesh_device):
    """Match the block-cyclic writer mapping for a rotated SP x TP chunk."""
    heads, sp, chunk, t, chunk_start = 8, 2, 256, 512, 160
    q_g, k_nat, w_g = _global_inputs(heads, chunk, t, seed=42)
    k_bc = _to_slab(k_nat, sp, chunk)
    mesh_shape = tuple(mesh_device.shape)
    shard_sp_seq = ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=mesh_shape, dims=(2, None))
    q_dev = _to_mesh(mesh_device, q_g, ttnn.bfloat16, shard_sp_seq)
    w_dev = _to_mesh(mesh_device, w_g, ttnn.bfloat16, shard_sp_seq)
    k_dev = _to_mesh(mesh_device, k_bc, ttnn.bfloat16, ttnn.ReplicateTensorToMesh(mesh_device))
    q_dev = ttnn.mesh_partition(q_dev, dim=2, cluster_axis=1)
    w_dev = ttnn.mesh_partition(w_dev, dim=2, cluster_axis=1)

    out = ttnn.experimental.indexer_score_dsa(
        q_dev,
        k_dev,
        w_dev,
        chunk_start_idx=chunk_start,
        seq_shard_axes=[0, 1],
        block_cyclic_sp_axis=0,
        block_cyclic_chunk_local=chunk // sp,
        program_config=ttnn.IndexerScoreProgramConfig(q_chunk_size=32, k_chunk_size=64, head_group_size=0),
    )
    shards = [ttnn.to_torch(shard) for shard in ttnn.get_device_tensors(out.cpu())]
    per_sp = [torch.cat(shards[r * 2 : (r + 1) * 2], dim=2) for r in range(sp)]
    out_t = torch.cat(per_sp, dim=2)
    ref = _straddle_ref(q_g, k_nat, w_g, sp, chunk, chunk_start, t)
    assert_indexer_match(out_t, ref, chunk, t, check_neg=True)


@pytest.mark.parametrize("mesh_device", [(2, 1)], ids=["sp2"], indirect=True)
@pytest.mark.parametrize("case_id, heads", QB_CASES, ids=QB_IDS)
def test_indexer_score_qb_straddle(mesh_device, case_id, heads):
    """Patch aligned and mid-slab causal geometry through one cached program."""
    sp = mesh_device.shape[0]
    q_g, k_nat, w_g = _global_inputs(heads, ST_CHUNK, ST_T, seed=42)
    k_bc = _to_slab(k_nat, sp, ST_CHUNK)
    mesh_shape = tuple(mesh_device.shape)
    shard = ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=mesh_shape, dims=(2, None))
    q_dev = _to_mesh(mesh_device, q_g, ttnn.bfloat16, shard)
    w_dev = _to_mesh(mesh_device, w_g, ttnn.bfloat16, shard)
    k_dev = _to_mesh(mesh_device, k_bc, ttnn.bfloat8_b, ttnn.ReplicateTensorToMesh(mesh_device))
    cfg = ttnn.IndexerScoreProgramConfig(q_chunk_size=32, k_chunk_size=128, head_group_size=0)

    def run(chunk_start):
        out = ttnn.experimental.indexer_score_dsa(
            q_dev,
            k_dev,
            w_dev,
            chunk_start_idx=chunk_start,
            seq_shard_axes=[0],
            block_cyclic_sp_axis=0,
            block_cyclic_chunk_local=ST_CHUNK // sp,
            program_config=cfg,
        )
        out_t = ttnn.to_torch(
            out, mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, mesh_shape=mesh_shape, dims=(2, 1))
        )
        ref = _straddle_ref(q_g, k_nat, w_g, sp, ST_CHUNK, chunk_start, ST_T)
        assert_indexer_match(out_t, ref, ST_CHUNK, ST_T, check_neg=True)

    mesh_device.enable_program_cache()
    run(ST_CS)
    entries = mesh_device.num_program_cache_entries()
    run(0)
    assert mesh_device.num_program_cache_entries() == entries, "changing causal geometry recompiled the program"


@pytest.mark.parametrize("mesh_device", [(QB_SP, 1)], ids=["sp4"], indirect=True)
def test_indexer_score_qb_msa_block_cyclic(mesh_device):
    """Read an SP=4 block-cyclic K cache through the MSA frontend."""
    q_g, k_nat, _ = _global_inputs(M3_QB_HEADS, QB_CHUNK, QB_T, seed=42)
    k_bc = _to_slab(k_nat, QB_SP, QB_CHUNK)
    mesh_shape = tuple(mesh_device.shape)
    shard = ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=mesh_shape, dims=(2, None))
    q_dev = _to_mesh(mesh_device, q_g, ttnn.bfloat16, shard)
    k_dev = _to_mesh(mesh_device, k_bc, ttnn.bfloat8_b, ttnn.ReplicateTensorToMesh(mesh_device))

    out = ttnn.experimental.indexer_score_msa(
        q_dev,
        k_dev,
        seq_shard_axes=[0],
        scale=M3_QB_SCALE,
        num_groups=1,
        block_cyclic_sp_axis=0,
        block_cyclic_chunk_local=QB_SQ,
        program_config=glx_config(M3_QB_HEADS),
    )
    out_t = ttnn.to_torch(out, mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, mesh_shape=mesh_shape, dims=(2, 1)))
    ref = _msa_per_sp_ref(q_g, k_nat, QB_SP, QB_HISTORY)
    assert_indexer_match(out_t, ref, QB_CHUNK, QB_T, check_neg=True)


@pytest.mark.parametrize("mesh_device", [(2, 1)], ids=["sp2"], indirect=True)
def test_indexer_score_qb_msa_block_cyclic_straddle(mesh_device):
    """Apply the MSA forced-local stamp across a block-cyclic slab boundary."""
    sp = 2
    block_size = BLOCK_POOL_BS
    q_g, k_nat, _ = _global_inputs(M3_QB_HEADS, ST_CHUNK, ST_MSA_T, seed=42)
    k_bc = _to_slab(k_nat, sp, ST_CHUNK)
    mesh_shape = tuple(mesh_device.shape)
    shard = ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=mesh_shape, dims=(2, None))
    q_dev = _to_mesh(mesh_device, q_g, ttnn.bfloat16, shard)
    k_dev = _to_mesh(mesh_device, k_bc, ttnn.bfloat8_b, ttnn.ReplicateTensorToMesh(mesh_device))

    out = ttnn.experimental.indexer_score_msa(
        q_dev,
        k_dev,
        chunk_start_idx=ST_CS,
        seq_shard_axes=[0],
        scale=M3_QB_SCALE,
        num_groups=1,
        block_size=block_size,
        block_cyclic_sp_axis=0,
        block_cyclic_chunk_local=ST_CHUNK // sp,
        program_config=ttnn.IndexerScoreProgramConfig(q_chunk_size=64, k_chunk_size=1024, head_group_size=0),
    )
    out_t = ttnn.to_torch(out, mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, mesh_shape=mesh_shape, dims=(2, 1)))
    ref = _straddle_msa_pooled_ref(q_g, k_nat, sp, ST_CHUNK, ST_CS, ST_MSA_T, M3_QB_SCALE, block_size)
    assert_pooled_match(out_t, ref, 1, ST_CHUNK, ST_MSA_T // block_size, pcc_floor=0.995)


# ---- SP-only ring-of-4 on a (1, 4) mesh ------------------------------------------------------------
RING4 = 4
SP4_AXIS = 1  # the length-4 axis of the (1, 4) mesh == cluster_axis
CHUNK4 = RING4 * QB_SQ  # 2560 global prefill chunk (chunk_local = QB_SQ per SP shard)
T4 = QB_HISTORY + CHUNK4  # 28160 all-gathered keys


@pytest.mark.parametrize("block_cyclic", [False, True], ids=["contiguous", "block_cyclic_rotated"])
def test_indexer_score_full_mesh_2x2_accuracy_placement_and_cache_reuse(block_cyclic):
    """Run the complete physical 2x2 QuietBox as one four-rank direct-neighbor snake."""
    if ttnn.get_num_devices() != 4:
        pytest.skip("2x2 full-mesh indexer coverage requires an exact four-device physical mesh")
    _run_full_mesh_accuracy_case((2, 2), block_cyclic=block_cyclic)


def _small_ring_inputs(mesh, heads, *, paged, page_size=64, seed=73):
    """Small ring-4 tensors; paged mode gives every rank the same table permutation over its own page pool."""
    sq, local_t, dim = 64, 256, QB_DIM
    chunk, total_t = RING4 * sq, RING4 * local_t
    gen = torch.Generator().manual_seed(seed)
    q = torch.randn(1, heads, chunk, dim, generator=gen, dtype=torch.bfloat16)
    k = torch.randn(1, 1, total_t, dim, generator=gen, dtype=torch.bfloat16)
    w = torch.randn(1, heads, chunk, 1, generator=gen, dtype=torch.bfloat16)
    seq_shard = ttnn.ShardTensorToMesh(mesh, dim=2)
    q_dev = ttnn.from_torch(q, device=mesh, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, mesh_mapper=seq_shard)
    w_dev = ttnn.from_torch(w, device=mesh, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, mesh_mapper=seq_shard)
    k_gathered = ttnn.from_torch(
        torch.zeros_like(k),
        device=mesh,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )
    if not paged:
        k_local = ttnn.from_torch(k, device=mesh, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, mesh_mapper=seq_shard)
        return q, k, w, q_dev, w_dev, k_local, k_gathered, {}

    num_layers, layer_idx = 3, 2
    pools = []
    table = None
    for rank in range(RING4):
        local = k[:, :, rank * local_t : (rank + 1) * local_t]
        pool, rank_table = _make_paged_k(local, page_size, num_layers=num_layers, layer_idx=layer_idx, seed=seed + 100)
        pools.append(pool)
        table = rank_table if table is None else table
        assert torch.equal(table, rank_table)
    physical_pool = torch.cat(pools, dim=0)
    pool_mapper = ttnn.ShardTensor2dMesh(mesh, mesh_shape=(1, RING4), dims=(None, 0))
    k_local = ttnn.from_torch(
        physical_pool,
        device=mesh,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        memory_config=_nd_sharded_dram_config(mesh, rows_per_shard=page_size),
        mesh_mapper=pool_mapper,
    )
    table_dev = ttnn.from_torch(
        table,
        device=mesh,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.uint16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )
    paged_kwargs = dict(
        kv_cache_num_layers=num_layers,
        kv_cache_layer_idx=layer_idx,
        page_bundle_indices=table_dev,
        kv_cache_page_size=page_size,
    )
    return q, k, w, q_dev, w_dev, k_local, k_gathered, paged_kwargs


def _small_ring_dsa_ref(q, k, w):
    sq = q.shape[2] // RING4
    history = k.shape[2] - q.shape[2]
    return torch.cat(
        [
            indexer_score_dsa_ref(
                q[:, :, rank * sq : (rank + 1) * sq],
                k,
                w[:, :, rank * sq : (rank + 1) * sq],
                history + rank * sq,
            )
            for rank in range(RING4)
        ],
        dim=2,
    )


def _small_ring_msa_ref(q, k, num_groups, block_size):
    sq = q.shape[2] // RING4
    history = k.shape[2] - q.shape[2]
    scale = q.shape[-1] ** -0.5
    return torch.cat(
        [
            indexer_score_msa_ref(
                q[:, :, rank * sq : (rank + 1) * sq],
                k,
                _msa_scale_w(q.shape[1], sq, scale),
                history + rank * sq,
                num_groups=num_groups,
                block_size=block_size,
            )
            for rank in range(RING4)
        ],
        dim=2,
    )


def test_indexer_score_ring4_fused_paged_dsa_4d():
    """Ring DSA gathers logical local shards from permuted multi-layer page pools."""
    mesh, ccl_semaphores, subdevice_id, stall_group = _open_ccl((1, RING4))
    try:
        q, k, w, q_dev, w_dev, k_local, k_gathered, paged_kwargs = _small_ring_inputs(mesh, 8, paged=True)
        cfg = ttnn.IndexerScoreProgramConfig(q_chunk_size=32, k_chunk_size=64, head_group_size=0)
        out = ttnn.experimental.ring_indexer_score_dsa(
            q_dev,
            k_gathered,
            w_dev,
            k_local,
            ccl_semaphores,
            cluster_axis=SP4_AXIS,
            topology=ttnn.Topology.Linear,
            ag_sub_device_id=subdevice_id,
            program_config=cfg,
            **paged_kwargs,
        )
        ttnn.synchronize_device(mesh, sub_device_ids=stall_group)
        out_t = ttnn.to_torch(out, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=2))
        assert_indexer_match(out_t, _small_ring_dsa_ref(q, k, w), q.shape[2], k.shape[2], check_neg=True)
    finally:
        _close_ccl(mesh)


def test_indexer_score_ring4_fused_paged_cache_hit_4d():
    """Ring cache hits repatch both the physical local pool and page-table addresses."""
    mesh, ccl_semaphores, subdevice_id, stall_group = _open_ccl((1, RING4))
    try:
        mesh.clear_program_cache()
        keep_alive = []
        entries = None
        cfg = ttnn.IndexerScoreProgramConfig(q_chunk_size=32, k_chunk_size=64, head_group_size=0)
        for seed in (103, 107):
            inputs = _small_ring_inputs(mesh, 8, paged=True, seed=seed)
            keep_alive.append(inputs)
            q, k, w, q_dev, w_dev, k_local, k_gathered, paged_kwargs = inputs
            out = ttnn.experimental.ring_indexer_score_dsa(
                q_dev,
                k_gathered,
                w_dev,
                k_local,
                ccl_semaphores,
                cluster_axis=SP4_AXIS,
                topology=ttnn.Topology.Linear,
                ag_sub_device_id=subdevice_id,
                program_config=cfg,
                **paged_kwargs,
            )
            ttnn.synchronize_device(mesh, sub_device_ids=stall_group)
            out_t = ttnn.to_torch(out, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=2))
            assert_indexer_match(out_t, _small_ring_dsa_ref(q, k, w), q.shape[2], k.shape[2], check_neg=True)
            if entries is None:
                entries = mesh.num_program_cache_entries()
            else:
                assert mesh.num_program_cache_entries() == entries, "paged ring K/table address change recompiled"
    finally:
        _close_ccl(mesh)


@pytest.mark.parametrize("paged", [False, True], ids=["contiguous", "paged"])
@pytest.mark.parametrize("num_groups,block_size", [(1, 0), (4, 0), (4, 128)], ids=["g1_fused", "g4", "g4_block_pool"])
def test_indexer_score_ring4_fused_msa_4d(paged, num_groups, block_size):
    """Ring MSA equivalence for fused-head, grouped, and pooled modes, with and without paging."""
    mesh, ccl_semaphores, subdevice_id, stall_group = _open_ccl((1, RING4))
    try:
        q, k, _, q_dev, _, k_local, k_gathered, paged_kwargs = _small_ring_inputs(
            mesh, num_groups, paged=paged, seed=83 + num_groups + block_size
        )
        k_chunk = 1024 if block_size else 64
        cfg = ttnn.IndexerScoreProgramConfig(q_chunk_size=32, k_chunk_size=k_chunk, head_group_size=0)
        out = ttnn.experimental.ring_indexer_score_msa(
            q_dev,
            k_gathered,
            k_local,
            ccl_semaphores,
            cluster_axis=SP4_AXIS,
            topology=ttnn.Topology.Linear,
            num_groups=num_groups,
            ag_sub_device_id=subdevice_id,
            scale=QB_DIM**-0.5,
            block_size=block_size,
            program_config=cfg,
            **paged_kwargs,
        )
        ttnn.synchronize_device(mesh, sub_device_ids=stall_group)
        out_t = ttnn.to_torch(out, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=2))
        ref = _small_ring_msa_ref(q, k, num_groups, block_size)
        if block_size:
            assert_pooled_match(out_t, ref, num_groups, q.shape[2], k.shape[2] // block_size, pcc_floor=0.995)
        else:
            assert_grouped_match(out_t, ref, num_groups, q.shape[2], k.shape[2])
    finally:
        _close_ccl(mesh)


@pytest.mark.parametrize("block_cyclic", [False, True], ids=["contiguous", "block_cyclic"])
@pytest.mark.parametrize("case_id, heads", QB_CASES, ids=QB_IDS)
def test_indexer_score_ring4_fused_4d(case_id, heads, block_cyclic):
    """SP-only ring-of-4 fused op on a directly-opened (1,4) mesh, checked vs the per-SP reference."""
    mesh, ccl_semaphores, subdevice_id, stall_group = _open_ccl((1, RING4))
    try:
        q_g, k_nat, w_g = _global_inputs(heads, CHUNK4, T4, seed=42)
        k_host = _to_slab(k_nat, RING4, CHUNK4) if block_cyclic else k_nat

        shard = ttnn.ShardTensorToMesh(mesh, dim=2)  # SP-shard seq over the 4 devices
        q_dev = ttnn.from_torch(q_g, device=mesh, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, mesh_mapper=shard)
        w_dev = ttnn.from_torch(w_g, device=mesh, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, mesh_mapper=shard)
        k_local = ttnn.from_torch(k_host, device=mesh, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, mesh_mapper=shard)
        # Gathered buffer: full T per device, zero-seeded (AG fills remote bands; zeros prove local sourcing).
        k_gathered = ttnn.from_torch(
            torch.zeros_like(k_nat),
            device=mesh,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        )

        bc_kwargs = dict(block_cyclic_sp_axis=SP4_AXIS, block_cyclic_chunk_local=QB_SQ) if block_cyclic else {}
        out = ttnn.experimental.ring_indexer_score_dsa(
            q_dev,
            k_gathered,
            w_dev,
            k_local,
            ccl_semaphores,
            cluster_axis=SP4_AXIS,
            topology=ttnn.Topology.Linear,
            num_links=1,
            ag_sub_device_id=subdevice_id,
            program_config=glx_config(heads),
            **bc_kwargs,
        )
        ttnn.synchronize_device(mesh, sub_device_ids=stall_group)
        out_t = ttnn.to_torch(out, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=2))

        ref = _per_sp_ref(q_g, k_nat, w_g, RING4, QB_HISTORY)
        assert_indexer_match(out_t, ref, CHUNK4, T4, check_neg=True)
        layout = "block_cyclic" if block_cyclic else "contiguous"
        logger.info(f"4d ring4 fused {layout} (heads={heads}): matched reference")
    finally:
        _close_ccl(mesh)


@pytest.mark.requires_host_iommu
def test_indexer_score_ring4_true_ring_bfp8_bank_owned_reference_cache_hit():
    """Reference-check the production BFP8 bank-owned path on a true QuietBox Ring.

    The large BFP8 K capacity exercises the bank-owned schedule's midpoint/completion protocol. Two runtime
    prefixes exercise distinct marker locations through one cached program.
    Sampled boundary rows on every rank cover both ring directions without constructing a capacity-sized CPU
    score tensor.
    """
    sp = 4
    sp_axis = 0
    heads = 4
    q_per_rank = 32
    chunk_global = sp * q_per_rank
    k_capacity = 256 * 1024
    kv_lens = (56320, 112640)
    dim = 128
    mesh, ccl_semaphores, subdevice_id, stall_group = _open_ccl(
        (sp, 1), fabric_config=ttnn.FabricConfig.FABRIC_2D_TORUS_XY
    )
    try:
        q_g, k_nat, w_g = _global_inputs(heads, chunk_global, k_capacity, seed=4343)
        k_bc = _to_slab(k_nat, sp, chunk_global)
        sp_shard = ttnn.ShardTensor2dMesh(mesh, mesh_shape=(sp, 1), dims=(2, None))
        q_dev = ttnn.from_torch(q_g, device=mesh, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, mesh_mapper=sp_shard)
        w_dev = ttnn.from_torch(w_g, device=mesh, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, mesh_mapper=sp_shard)
        k_local = ttnn.from_torch(
            k_bc,
            device=mesh,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat8_b,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=sp_shard,
        )
        k_gathered = ttnn.from_torch(
            torch.zeros_like(k_nat),
            device=mesh,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat8_b,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        )
        program_config = ttnn.IndexerScoreProgramConfig(q_chunk_size=32, k_chunk_size=320, head_group_size=0)

        def _score(kv_len):
            chunk_start = kv_len - chunk_global
            out = ttnn.experimental.ring_indexer_score_dsa(
                q_dev,
                k_gathered,
                w_dev,
                k_local,
                ccl_semaphores,
                cluster_axis=sp_axis,
                topology=ttnn.Topology.Ring,
                num_links=2,
                ag_sub_device_id=subdevice_id,
                chunk_start_idx=chunk_start,
                kv_len=kv_len,
                block_cyclic_sp_axis=sp_axis,
                block_cyclic_chunk_local=q_per_rank,
                program_config=program_config,
            )
            ttnn.synchronize_device(mesh, sub_device_ids=stall_group)
            out_t = ttnn.to_torch(out, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=2))
            ttnn.deallocate(out)
            return out_t, chunk_start

        entries_before = mesh.num_program_cache_entries()
        saw_negative_reference = False
        for dispatch, kv_len in enumerate(kv_lens):
            out_t, chunk_start = _score(kv_len)
            if dispatch == 0:
                entries_after_compile = mesh.num_program_cache_entries()
                assert entries_after_compile > entries_before
            else:
                assert (
                    mesh.num_program_cache_entries() == entries_after_compile
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
                    visible_ref = ref[ref != float("-inf")]
                    saw_negative_reference |= bool((visible_ref < 0).any())
                    assert_indexer_match(out_t[:, :, row, :kv_len], ref, sq=1, t=kv_len, check_neg=False)
        assert saw_negative_reference, "sampled rows must include a negative valid score"
        logger.info("True Ring-4 BFP8 bank-owned partial readiness matched sampled reference across a cache hit")
    finally:
        _close_ccl(mesh)


@pytest.mark.requires_host_iommu
def test_indexer_score_ring4_small_capacity_has_no_empty_lane_deadlock():
    """Keep every multicast participant productive when one shard has fewer KC units than the full grid.

    Eleven query groups force multiple phases while eighteen KC units per shard require reduced nonempty lane
    geometry. The sampled reference verifies the resulting score.
    """
    sp = 4
    sp_axis = 0
    heads = 4
    q_per_rank = 352  # 11 tiles; prime group count exceeds the 10-row QuietBox grid.
    chunk_global = sp * q_per_rank
    k_capacity = 22_528  # 5,632 tokens/rank = 18 KC units, below the physical lane count.
    assert (k_capacity // sp + 10 * 32 - 1) // (10 * 32) == 18

    mesh, ccl_semaphores, subdevice_id, stall_group = _open_ccl(
        (sp, 1), fabric_config=ttnn.FabricConfig.FABRIC_2D_TORUS_XY
    )
    try:
        q_g, k_nat, w_g = _global_inputs(heads, chunk_global, k_capacity, seed=4444)
        k_bc = _to_slab(k_nat, sp, chunk_global)
        sp_shard = ttnn.ShardTensor2dMesh(mesh, mesh_shape=(sp, 1), dims=(2, None))
        q_dev = ttnn.from_torch(q_g, device=mesh, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, mesh_mapper=sp_shard)
        w_dev = ttnn.from_torch(w_g, device=mesh, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, mesh_mapper=sp_shard)
        k_local = ttnn.from_torch(
            k_bc,
            device=mesh,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat8_b,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=sp_shard,
        )
        k_gathered = ttnn.from_torch(
            torch.zeros_like(k_nat),
            device=mesh,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat8_b,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        )
        out = ttnn.experimental.ring_indexer_score_dsa(
            q_dev,
            k_gathered,
            w_dev,
            k_local,
            ccl_semaphores,
            cluster_axis=sp_axis,
            topology=ttnn.Topology.Ring,
            num_links=2,
            ag_sub_device_id=subdevice_id,
            chunk_start_idx=0,
            kv_len=k_capacity,
            block_cyclic_sp_axis=sp_axis,
            block_cyclic_chunk_local=q_per_rank,
            program_config=ttnn.IndexerScoreProgramConfig(
                q_chunk_size=32,
                k_chunk_size=320,
                head_group_size=0,
            ),
        )
        ttnn.synchronize_device(mesh, sub_device_ids=stall_group)
        out_t = ttnn.to_torch(out, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=2))

        for rank in range(sp):
            # Start at the end of the first query tile: row zero has only one visible causal score, for which
            # correlation is undefined. This still samples both ends of every rank while keeping the PCC check
            # meaningful.
            for local_row in (31, q_per_rank - 1):
                global_row = rank * q_per_rank + local_row
                row = slice(global_row, global_row + 1)
                ref = indexer_score_dsa_ref(
                    q_g[:, :, row, :],
                    k_nat,
                    w_g[:, :, row, :],
                    global_row,
                )
                assert_indexer_match(out_t[:, :, row, :], ref, sq=1, t=k_capacity, check_neg=True)
        logger.info("True Ring-4 small-capacity multi-phase schedule matched sampled reference without a hang")
    finally:
        _close_ccl(mesh)


# ---- 2D SP×TP on a (2, 2) mesh: sp=2 ring × tp=2 sequence sub-shard ---------------------------------
SP2 = 2  # sequence-parallel ranks == ring size (cluster_axis extent)
TP2 = 2  # tensor-parallel ranks the QUERY sequence is ALSO sub-sharded over (seq_subshard_axis extent)
SP2_AXIS = 0  # mesh rows == SP ring (cluster_axis / block_cyclic_sp_axis)
TP2_AXIS = 1  # mesh cols == TP seq sub-shard (seq_subshard_axis)
CHUNK_SPTP = SP2 * QB_SQ  # 1280 global chunk; per-SP-shard chunk_local = QB_SQ (640), per-device Sq = 320
T_SPTP = QB_HISTORY + CHUNK_SPTP  # 26880 keys
K_CHUNK_SPTP = 320


def _per_sp_tp_ref(q_g, k_g, w_g, sp, tp, history, sq_sp):
    """Reference for a 2D SP×TP seq sub-shard, in row-major device order (SP outer). Device (r, t) owns query
    rows [r*sq_sp + t*sq_dev, r*sq_sp + (t+1)*sq_dev) and scores from causal start history + that row base."""
    sq_dev = sq_sp // tp
    refs = []
    for r in range(sp):
        for t in range(tp):
            g0 = r * sq_sp + t * sq_dev
            sl = slice(g0, g0 + sq_dev)
            refs.append(indexer_score_dsa_ref(q_g[:, :, sl, :], k_g, w_g[:, :, sl, :], history + g0))
    return torch.cat(refs, dim=2)


def _sptp_tp_sharded_inputs(
    mesh,
    k_capacity,
    seed,
    *,
    chunk_global=CHUNK_SPTP,
    sp=SP2,
    tp=TP2,
    sp_axis=SP2_AXIS,
    tp_axis=TP2_AXIS,
):
    """Build SP x TP query shards and a TP-inner reconstructed K cache."""
    chunk_local = chunk_global // sp
    q_g, k_nat, w_g = _global_inputs(32, chunk_global, k_capacity, seed=seed)
    k_reconstructed = _to_tp_inner_reconstructed(k_nat, sp=sp, tp=tp, chunk_local=chunk_local)
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
        torch.zeros_like(k_nat),
        device=mesh,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat8_b,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )
    q_dev = ttnn.from_torch(
        q_g,
        device=mesh,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat8_b,
        mesh_mapper=sp_shard,
    )
    w_dev = ttnn.from_torch(
        w_g,
        device=mesh,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        mesh_mapper=sp_shard,
    )
    if tp > 1:
        q_dev = ttnn.mesh_partition(q_dev, dim=2, cluster_axis=tp_axis)
        w_dev = ttnn.mesh_partition(w_dev, dim=2, cluster_axis=tp_axis)
    return q_g, k_nat, w_g, q_dev, k_local, k_gathered, w_dev


def _run_sptp_tp_sharded_score(
    ccl_semaphores,
    subdevice_id,
    q_dev,
    k_local,
    k_gathered,
    w_dev,
    *,
    chunk_start,
    kv_len,
    tp_sharded,
    chunk_global=CHUNK_SPTP,
    topology=ttnn.Topology.Linear,
    num_links=1,
    sp=SP2,
    sp_axis=SP2_AXIS,
    tp_axis=TP2_AXIS,
):
    return ttnn.experimental.ring_indexer_score_dsa(
        q_dev,
        k_gathered,
        w_dev,
        k_local,
        ccl_semaphores,
        cluster_axis=sp_axis,
        topology=topology,
        num_links=num_links,
        ag_sub_device_id=subdevice_id,
        chunk_start_idx=chunk_start,
        kv_len=kv_len,
        seq_subshard_axis=tp_axis,
        block_cyclic_sp_axis=sp_axis,
        block_cyclic_chunk_local=chunk_global // sp,
        block_cyclic_cache_tp_sharded=tp_sharded,
        program_config=ttnn.IndexerScoreProgramConfig(
            q_chunk_size=32,
            k_chunk_size=K_CHUNK_SPTP,
            head_group_size=0,
        ),
    )


def _sptp_output(out):
    return torch.cat([ttnn.to_torch(shard) for shard in ttnn.get_device_tensors(out.cpu())], dim=2)


@pytest.mark.parametrize("tp_sharded", [False, True], ids=["control", "tp_sharded"])
def test_indexer_score_sptp_tp_sharded_kv_repro_4d(tp_sharded):
    """Score a TP-inner reconstructed K cache on a QuietBox SP2 x TP2 mesh."""
    chunk_local = CHUNK_SPTP // SP2
    assert chunk_local == 640
    assert chunk_local // TP2 == 320
    assert K_CHUNK_SPTP // 32 == 10

    mesh, ccl_semaphores, subdevice_id, stall_group = _open_ccl((SP2, TP2))
    try:
        q_g, k_nat, w_g, q_dev, k_local, k_gathered, w_dev = _sptp_tp_sharded_inputs(mesh, CHUNK_SPTP, seed=45)
        out = _run_sptp_tp_sharded_score(
            ccl_semaphores,
            subdevice_id,
            q_dev,
            k_local,
            k_gathered,
            w_dev,
            chunk_start=0,
            kv_len=CHUNK_SPTP,
            tp_sharded=tp_sharded,
        )
        ttnn.synchronize_device(mesh, sub_device_ids=stall_group)
        out_t = _sptp_output(out)
        ref = _per_sp_tp_ref(q_g, k_nat, w_g, SP2, TP2, history=0, sq_sp=chunk_local)
        assert_indexer_match(out_t, ref, CHUNK_SPTP, CHUNK_SPTP, check_neg=True)
    finally:
        _close_ccl(mesh)


def test_indexer_score_sptp_tp_sharded_multislab_partial_prefix_cache_reuse_4d():
    """QuietBox coverage for a valid TP stripe after an invalid one and runtime-scalar cache reuse."""
    chunk_global = CHUNK_SPTP // 2
    chunk_local = chunk_global // SP2
    k_capacity = 3 * chunk_global
    kv_cases = ((0, 960), (chunk_global, 1280))
    # KC [10,20) crosses a 15-tile TP stripe boundary. At kv_len=960 its first five
    # columns are invalid [40,45) and its last five map to valid keys [5,10).
    assert (k_capacity // (SP2 * TP2)) // 32 == 15

    mesh, ccl_semaphores, subdevice_id, stall_group = _open_ccl((SP2, TP2))
    try:
        q_g, k_nat, w_g, q_dev, k_local, k_gathered, w_dev = _sptp_tp_sharded_inputs(
            mesh, k_capacity, seed=46, chunk_global=chunk_global
        )
        mesh.enable_program_cache()
        entries_before = mesh.num_program_cache_entries()
        entries_after_compile = None
        for dispatch, (chunk_start, kv_len) in enumerate(kv_cases):
            out = _run_sptp_tp_sharded_score(
                ccl_semaphores,
                subdevice_id,
                q_dev,
                k_local,
                k_gathered,
                w_dev,
                chunk_start=chunk_start,
                kv_len=kv_len,
                tp_sharded=True,
                chunk_global=chunk_global,
            )
            ttnn.synchronize_device(mesh, sub_device_ids=stall_group)
            out_t = _sptp_output(out)
            ttnn.deallocate(out)
            ref = _per_sp_tp_ref(
                q_g,
                k_nat[:, :, :kv_len],
                w_g,
                SP2,
                TP2,
                history=chunk_start,
                sq_sp=chunk_local,
            )
            assert_indexer_match(out_t[:, :, :, :kv_len], ref, chunk_global, kv_len, check_neg=True)
            if dispatch == 0:
                entries_after_compile = mesh.num_program_cache_entries()
                assert entries_after_compile > entries_before
            else:
                assert (
                    mesh.num_program_cache_entries() == entries_after_compile
                ), "changing kv_len/chunk_start recompiled the QuietBox fused Ring program"
    finally:
        mesh.disable_and_clear_program_cache()
        _close_ccl(mesh)


def test_indexer_score_ring_partial_readiness_4d():
    """Score a partial prefix while a four-rank SP-only ring advances its readiness state."""
    sp, tp = 4, 1
    sp_axis, tp_axis = 1, 0
    chunk_global = CHUNK_SPTP
    chunk_local = chunk_global // sp
    k_capacity = 3 * chunk_global
    kv_len = 960

    mesh, ccl_semaphores, subdevice_id, stall_group = _open_ccl(
        (tp, sp), fabric_config=ttnn.FabricConfig.FABRIC_2D_TORUS_XY
    )
    try:
        q_g, k_nat, w_g, q_dev, k_local, k_gathered, w_dev = _sptp_tp_sharded_inputs(
            mesh,
            k_capacity,
            seed=47,
            sp=sp,
            tp=tp,
            sp_axis=sp_axis,
            tp_axis=tp_axis,
        )
        out = _run_sptp_tp_sharded_score(
            ccl_semaphores,
            subdevice_id,
            q_dev,
            k_local,
            k_gathered,
            w_dev,
            chunk_start=0,
            kv_len=kv_len,
            tp_sharded=False,
            topology=ttnn.Topology.Ring,
            num_links=2,
            sp=sp,
            sp_axis=sp_axis,
            tp_axis=None,
        )
        ttnn.synchronize_device(mesh, sub_device_ids=stall_group)
        out_t = _sptp_output(out)
        ref = _per_sp_tp_ref(
            q_g,
            k_nat[:, :, :kv_len],
            w_g,
            sp,
            tp,
            history=0,
            sq_sp=chunk_local,
        )
        assert_indexer_match(out_t[:, :, :, :kv_len], ref, chunk_global, kv_len, check_neg=True)
    finally:
        _close_ccl(mesh)


@pytest.mark.parametrize("case_id, heads", QB_CASES, ids=QB_IDS)
def test_indexer_score_sptp_fused_4d(case_id, heads):
    """2D SP×TP fused op on a (2,2) mesh: sp=2 ring × tp=2 query seq sub-shard. Block-cyclic K (SP-sharded +
    TP-replicated, so the AG is unchanged); each device's causal diagonal starts at
    history + sp_rank*Sq_sp + tp_rank*Sq_dev -- proving the fused path threads the TP sub-offset into the score.
    Slab-aligned (no straddle); checked vs the per-(sp,tp) reference."""
    chunk_local = CHUNK_SPTP // SP2  # per-SP-shard chunk == QB_SQ (640); per-device query rows = 320
    chunk_start = QB_HISTORY  # slab-aligned (QB_HISTORY % CHUNK_SPTP == 0) -> no straddle
    mesh, ccl_semaphores, subdevice_id, stall_group = _open_ccl((SP2, TP2))
    try:
        q_g, k_nat, w_g = _global_inputs(heads, CHUNK_SPTP, T_SPTP, seed=42)
        k_bc = _to_slab(k_nat, SP2, CHUNK_SPTP)  # block-cyclic physical layout the reader inverts

        # K: block-cyclic slab sharded on the SP axis (dim 2), replicated across TP.
        k_shard = ttnn.ShardTensor2dMesh(mesh, mesh_shape=(SP2, TP2), dims=(2, None))
        k_local = ttnn.from_torch(k_bc, device=mesh, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, mesh_mapper=k_shard)
        # Gathered buffer: full T per device (replicated over both axes), zero-seeded.
        k_gathered = ttnn.from_torch(
            torch.zeros_like(k_nat),
            device=mesh,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        )
        # q/w: SP-shard seq (dim 2), then split those rows over TP (mesh_partition) -> each device owns Sq_sp/tp rows.
        qw_shard = ttnn.ShardTensor2dMesh(mesh, mesh_shape=(SP2, TP2), dims=(2, None))
        q_dev = ttnn.from_torch(q_g, device=mesh, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, mesh_mapper=qw_shard)
        w_dev = ttnn.from_torch(w_g, device=mesh, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, mesh_mapper=qw_shard)
        q_dev = ttnn.mesh_partition(q_dev, dim=2, cluster_axis=TP2_AXIS)
        w_dev = ttnn.mesh_partition(w_dev, dim=2, cluster_axis=TP2_AXIS)

        out = ttnn.experimental.ring_indexer_score_dsa(
            q_dev,
            k_gathered,
            w_dev,
            k_local,
            ccl_semaphores,
            cluster_axis=SP2_AXIS,
            topology=ttnn.Topology.Linear,
            num_links=1,
            ag_sub_device_id=subdevice_id,
            chunk_start_idx=chunk_start,
            seq_subshard_axis=TP2_AXIS,  # the SP×TP feature under test
            block_cyclic_sp_axis=SP2_AXIS,
            block_cyclic_chunk_local=chunk_local,
            program_config=glx_config(heads),
        )
        ttnn.synchronize_device(mesh, sub_device_ids=stall_group)
        # Compose device shards back to global chunk order (row-major: SP outer, TP inner).
        shards = [ttnn.to_torch(s) for s in ttnn.get_device_tensors(out.cpu())]
        out_t = torch.cat(shards, dim=2)

        ref = _per_sp_tp_ref(q_g, k_nat, w_g, SP2, TP2, chunk_start, chunk_local)
        assert_indexer_match(out_t, ref, CHUNK_SPTP, T_SPTP, check_neg=True)
        logger.info(f"4d SP×TP fused (heads={heads}): sp2 ring × tp2 seq sub-shard matched reference")
    finally:
        _close_ccl(mesh)
