# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Reproducer: paged_scaled_dot_product_attention_decode produces incorrect per-batch
output for OLMo-3.1-32B's shape configuration on TG (Galaxy 8x4 mesh).

For details and full diagnostics see PAGED_SDPA_DECODE_BATCH_BUG.md at the repo root.

CORE INVARIANT
--------------
Identical Q (across batch positions) + identical KV-cache contents (across batch
positions) ⇒ identical SDPA output (across batch positions). Causal SDPA decode
is a deterministic per-batch reduction; identical inputs MUST produce identical
outputs.

WHAT THIS TEST EXERCISES
------------------------
- Single-device variants (`_single_device` suffix): kernel correctly produces
  identical output across all batch positions with identical inputs. These PASS
  and serve as a sanity check that the kernel is correct in isolation.
- Galaxy / TG variant (`_tg`): kernel produces DIFFERENT output across batch
  positions despite bit-identical inputs. This is the OLMo bug — observed on
  the actual model demo, reproduced here in a minimal harness.

The Galaxy variant uses OLMo's exact configuration:
- cluster_shape = (8, 4)
- KV cache replicated to all 32 devices (cluster_axis=0 by head, cluster_axis=1 replicated)
- Q sharded by batch across cluster_axis=1 (4 cols × 8 col-local batch = 32 total)
- SCORES_BATCHED_MM_OUTPUT_MEMCFG sharded on sub_core_grids (50-core layout)
- GQA ratio 8:1 per device

Run:
    pytest tests/ttnn/nightly/unit_tests/operations/sdpa/test_paged_sdpa_decode_batch_collapse.py -xvs

Skip single-device tests (already known to pass) and only run the TG repro:
    pytest tests/ttnn/nightly/unit_tests/operations/sdpa/test_paged_sdpa_decode_batch_collapse.py::test_paged_sdpa_decode_olmo_tg_batch_identity -xvs
"""

import torch
import pytest
import ttnn

from loguru import logger
from tests.ttnn.unit_tests.operations.sdpa.sdpa_test_utils import (
    fa_rand,
    nearest_n,
    nearest_pow_2,
    num_to_corerange,
)


# OLMo-3.1-32B-Think on TG: per-device parameters seen by paged SDPA decode.
B_LOCAL = 8       # batch_size_per_device_group (col-local batch on Galaxy)
N_Q = 8           # n_local_heads padded to 8 (5 real + 3 phantom for OLMo)
N_KV = 1          # n_local_kv_heads (1 KV head per device, GQA ratio 8:1)
D = 128           # head_dim
BLOCK_SIZE = 64
MAX_SEQ = 256     # small for the test; keep cache compact
CUR_POS = 130     # past first cache update


def _build_paged_cache_and_pt_identical(batch, n_kv, max_seq, block_size, d):
    """
    Return (K_paged, V_paged, page_table) where every batch slot holds bit-identical KV.

    The first slot's K/V is generated with fa_rand and then replicated across all
    `batch` slots. After paging, `page_table[k]` points to physical blocks that
    contain identical data for every k.
    """
    blocks_per_seq = max_seq // block_size
    max_num_blocks = batch * blocks_per_seq

    K_one = fa_rand(1, n_kv, max_seq, d)
    V_one = fa_rand(1, n_kv, max_seq, d)
    K_ref = K_one.repeat(batch, 1, 1, 1)
    V_ref = V_one.repeat(batch, 1, 1, 1)

    def to_paged(cache):
        return (
            cache.reshape(batch, n_kv, blocks_per_seq, block_size, d)
            .transpose(1, 2)
            .reshape(max_num_blocks, n_kv, block_size, d)
        )

    permutation = torch.randperm(max_num_blocks)
    page_table = torch.argsort(permutation).reshape(batch, blocks_per_seq)
    K_paged = to_paged(K_ref)[permutation]
    V_paged = to_paged(V_ref)[permutation]
    return K_paged, V_paged, page_table


def _check_batch_identity(out_torch, label, atol=0.0):
    """Verify all rows of out_torch's batch dim are equal (within atol).

    out_torch shape: [1, B, NH, D] or [B, NH, 1, D].
    """
    if out_torch.dim() == 4 and out_torch.shape[0] == 1:
        per_batch = out_torch[0]
    elif out_torch.dim() == 4:
        per_batch = out_torch
    else:
        per_batch = out_torch.flatten(0, -2)
    b = per_batch.shape[0]

    sums = [float(per_batch[i].abs().sum()) for i in range(b)]
    logger.info(f"[{label}] per-batch abs sums: {[round(s, 4) for s in sums]}")

    fail = []
    for i in range(1, b):
        diff = (per_batch[0] - per_batch[i]).abs()
        max_diff = float(diff.max())
        if max_diff > atol:
            fail.append((i, max_diff, sums[i] - sums[0]))
    if fail:
        details = ", ".join([f"row{i}: max_diff={d:.4e}, sum_diff={sd:.4f}" for i, d, sd in fail])
        raise AssertionError(
            f"[{label}] paged SDPA decode produced different output across batch positions "
            f"despite identical inputs: {details}"
        )
    logger.info(f"[{label}] all {b} batch rows are within atol={atol} ✓")


# ---------------------------------------------------------------------------
# Single-device sanity (PASS on current main): correctness in isolation.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
def test_paged_sdpa_decode_batch_identity_dram_single_device(device):
    """Sanity: with DRAM output on a single device, identical inputs ⇒ identical outputs."""
    torch.manual_seed(42)

    K_paged, V_paged, page_table = _build_paged_cache_and_pt_identical(B_LOCAL, N_KV, MAX_SEQ, BLOCK_SIZE, D)

    Q_one = fa_rand(1, 1, N_Q, D)
    Q = Q_one.repeat(1, B_LOCAL, 1, 1)
    cur_pos = torch.full((B_LOCAL,), CUR_POS, dtype=torch.int32)

    dram_cfg = ttnn.DRAM_MEMORY_CONFIG
    padded_num_heads = nearest_pow_2(nearest_n(N_Q, n=32))
    in_grid = ttnn.CoreRangeSet({num_to_corerange(B_LOCAL)})
    in_spec = ttnn.ShardSpec(in_grid, (padded_num_heads, D), ttnn.ShardOrientation.ROW_MAJOR)
    in_cfg = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, in_spec)

    program_cfg = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=device.compute_with_storage_grid_size(),
        exp_approx_mode=False, q_chunk_size=0, k_chunk_size=0,
    )
    compute_kernel_cfg = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
        fp32_dest_acc_en=False, packer_l1_acc=False,
    )

    tt_K = ttnn.as_tensor(K_paged, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, memory_config=dram_cfg)
    tt_V = ttnn.as_tensor(V_paged, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, memory_config=dram_cfg)
    tt_pt = ttnn.as_tensor(page_table, device=device, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, memory_config=dram_cfg)
    tt_Q = ttnn.as_tensor(Q, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, memory_config=in_cfg)
    tt_cp = ttnn.as_tensor(cur_pos, device=device, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, memory_config=dram_cfg)

    tt_out = ttnn.transformer.paged_scaled_dot_product_attention_decode(
        tt_Q, tt_K, tt_V, tt_pt,
        cur_pos_tensor=tt_cp,
        scale=D**-0.5,
        program_config=program_cfg,
        compute_kernel_config=compute_kernel_cfg,
        memory_config=dram_cfg,
    )
    out = ttnn.to_torch(tt_out)[:, :, :N_Q, :]
    _check_batch_identity(out, label="single-device DRAM-out, B=8, NQ=8, NKV=1")


# ---------------------------------------------------------------------------
# TG / Galaxy: reproduces the OLMo decode bug.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "mesh_device", [pytest.param((8, 4), id="galaxy_8x4")], indirect=True
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576, "trace_region_size": 0}], indirect=True)
def test_paged_sdpa_decode_olmo_tg_batch_identity(mesh_device):
    """
    Reproduces the OLMo TG paged_scaled_dot_product_attention_decode bug.

    Builds the OLMo-shape inputs on Galaxy (8x4 mesh) with bit-identical KV cache
    and Q across batch positions, and asserts that the SDPA decode output is also
    identical across batch positions.

    EXPECTED on a correct kernel: all 8 col-local batch rows bit-identical.
    OBSERVED in OLMo demo (and here): batch outputs split into chunks of 4 (DRAM)
    or have 4 zero slots (height-sharded on sub_core_grids).

    This test is expected to FAIL until the upstream SDPA decode kernel is fixed.
    """
    cluster_shape = (8, 4)
    batch_global = B_LOCAL * cluster_shape[1]   # 8 col-local users × 4 cols = 32
    n_kv = N_KV
    n_q = N_Q
    d = D
    block_size = BLOCK_SIZE
    max_seq = MAX_SEQ

    torch.manual_seed(42)

    # KV cache on each device — replicated across the mesh, contains
    # bit-identical data per batch slot.
    K_paged, V_paged, page_table = _build_paged_cache_and_pt_identical(
        batch_global, n_kv, max_seq, block_size, d
    )

    # Page-table on TG decode: dims=(None, -2) — sharded along tensor dim 0
    # (batch) across cluster_axis=1 (4 cols), replicated across cluster_axis=0.
    tt_pt = ttnn.from_torch(
        page_table,
        device=mesh_device,
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(None, -2), mesh_shape=cluster_shape),
    )

    # KV cache: replicate to all 32 devices.
    tt_K = ttnn.from_torch(
        K_paged,
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    tt_V = ttnn.from_torch(
        V_paged,
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )

    # Q: identical for all 32 batch positions, shape [1, 32, n_q, d].
    Q_one = fa_rand(1, 1, n_q, d)
    Q = Q_one.repeat(1, batch_global, 1, 1)

    # Q sharded across cluster_axis=1 by batch dim (4 cols × 8 col-local users).
    # Cluster_axis=0 holds heads — we mock with replicated heads (1 head replicated
    # on each cluster_axis=0 device) since this is a single-layer SDPA test, not a
    # full transformer. The bug repro only needs the BATCH dim sharded correctly.
    tt_Q = ttnn.from_torch(
        Q,
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(None, 1), mesh_shape=cluster_shape),
    )

    # current_pos: [batch_global] → sharded along batch across cluster_axis=1.
    cur_pos = torch.full((batch_global,), CUR_POS, dtype=torch.int32)
    tt_cp = ttnn.from_torch(
        cur_pos,
        device=mesh_device,
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(None, 0), mesh_shape=cluster_shape),
    )

    # Output mem cfg: HEIGHT_SHARDED on OLMo's sub_core_grids 50-core layout
    # (cols {1,2,3,5,6} × rows {0..9} — col 4 skipped). This is the configuration
    # that triggers the bug in the OLMo demo.
    olmo_sub_core_grids = ttnn.CoreRangeSet(
        [
            ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(3, 9)),
            ttnn.CoreRange(ttnn.CoreCoord(5, 0), ttnn.CoreCoord(6, 9)),
        ]
    )
    olmo_start_core = ttnn.CoreCoord(1, 0)
    padded_num_heads = nearest_pow_2(nearest_n(n_q, n=32))
    out_grid = ttnn.num_cores_to_corerangeset_in_subcoregrids(
        olmo_start_core, B_LOCAL, olmo_sub_core_grids, row_wise=True
    )
    out_spec = ttnn.ShardSpec(out_grid, (padded_num_heads, d), ttnn.ShardOrientation.ROW_MAJOR)
    out_cfg = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, out_spec)

    # Q input also needs to be sharded on the same core layout (otherwise mem cfg
    # of Q and the SDPA kernel disagree). Re-create tt_Q as height-sharded.
    in_spec = ttnn.ShardSpec(out_grid, (padded_num_heads, d), ttnn.ShardOrientation.ROW_MAJOR)
    in_cfg = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, in_spec)
    tt_Q_sharded = ttnn.to_memory_config(tt_Q, in_cfg)
    ttnn.deallocate(tt_Q)
    tt_Q = tt_Q_sharded

    program_cfg = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=mesh_device.compute_with_storage_grid_size(),
        exp_approx_mode=False, q_chunk_size=0, k_chunk_size=0,
    )
    compute_kernel_cfg = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
        fp32_dest_acc_en=False, packer_l1_acc=False,
    )

    tt_out = ttnn.transformer.paged_scaled_dot_product_attention_decode(
        tt_Q, tt_K, tt_V, tt_pt,
        cur_pos_tensor=tt_cp,
        scale=d**-0.5,
        program_config=program_cfg,
        compute_kernel_config=compute_kernel_cfg,
        memory_config=out_cfg,
    )

    # Read back per-device, then compare batch positions WITHIN one column.
    # On col 0 (cluster_axis=1 idx 0), the local Q has 8 col-local batches.
    # We read device 0 (cluster_axis_0=0, cluster_axis_1=0) which holds col 0's first row.
    out_per_dev = ttnn.get_device_tensors(tt_out)
    out_dev0 = ttnn.to_torch(out_per_dev[0]).float()
    logger.info(f"col0 dev0 SDPA output shape: {tuple(out_dev0.shape)}")

    # The output on dev 0 is the col-local (B=8) result. Slice to N_Q heads.
    if out_dev0.shape[-2] >= n_q:
        out_dev0 = out_dev0[..., :n_q, :]

    _check_batch_identity(out_dev0, label="TG col-0 dev-0 SDPA output, B_local=8")


@pytest.mark.parametrize(
    "mesh_device", [pytest.param((8, 4), id="galaxy_8x4")], indirect=True
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576, "trace_region_size": 0}], indirect=True)
def test_paged_sdpa_decode_olmo_tg_batch_identity_rect_grid(mesh_device):
    """
    Same as test_paged_sdpa_decode_olmo_tg_batch_identity, but uses a CONTIGUOUS
    rectangular sub_core_grids for the OUTPUT mem cfg (cols 1-6, rows 0-9 = 60 cores;
    avoids col 0 reserved + col 7 dispatch + does NOT skip col 4). Q input uses
    matching rectangular sharding.

    If this test PASSES, the bug is specifically the irregular OLMo 50-core grid.
    If it FAILS, the bug is in the multi-device decode kernel itself, independent
    of grid shape.
    """
    cluster_shape = (8, 4)
    batch_global = B_LOCAL * cluster_shape[1]
    n_kv = N_KV
    n_q = N_Q
    d = D
    block_size = BLOCK_SIZE
    max_seq = MAX_SEQ

    torch.manual_seed(42)

    K_paged, V_paged, page_table = _build_paged_cache_and_pt_identical(
        batch_global, n_kv, max_seq, block_size, d
    )

    tt_pt = ttnn.from_torch(
        page_table, device=mesh_device, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(None, -2), mesh_shape=cluster_shape),
    )
    tt_K = ttnn.from_torch(
        K_paged, device=mesh_device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    tt_V = ttnn.from_torch(
        V_paged, device=mesh_device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )

    Q_one = fa_rand(1, 1, n_q, d)
    Q = Q_one.repeat(1, batch_global, 1, 1)
    tt_Q = ttnn.from_torch(
        Q, device=mesh_device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(None, 1), mesh_shape=cluster_shape),
    )
    cur_pos = torch.full((batch_global,), CUR_POS, dtype=torch.int32)
    tt_cp = ttnn.from_torch(
        cur_pos, device=mesh_device, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(None, 0), mesh_shape=cluster_shape),
    )

    # CONTIGUOUS rectangular sub_core_grids (cols 1-6 × rows 0-9 = 60 cores).
    # No col-4 skip. Avoids col 0 (reserved) and col 7 (dispatch).
    rect_sub_core_grids = ttnn.CoreRangeSet(
        [ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(6, 9))]
    )
    rect_start_core = ttnn.CoreCoord(1, 0)
    padded_num_heads = nearest_pow_2(nearest_n(n_q, n=32))
    out_grid = ttnn.num_cores_to_corerangeset_in_subcoregrids(
        rect_start_core, B_LOCAL, rect_sub_core_grids, row_wise=True
    )
    out_spec = ttnn.ShardSpec(out_grid, (padded_num_heads, d), ttnn.ShardOrientation.ROW_MAJOR)
    out_cfg = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, out_spec)

    in_spec = ttnn.ShardSpec(out_grid, (padded_num_heads, d), ttnn.ShardOrientation.ROW_MAJOR)
    in_cfg = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, in_spec)
    tt_Q_sharded = ttnn.to_memory_config(tt_Q, in_cfg)
    ttnn.deallocate(tt_Q)
    tt_Q = tt_Q_sharded

    program_cfg = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=mesh_device.compute_with_storage_grid_size(),
        exp_approx_mode=False, q_chunk_size=0, k_chunk_size=0,
    )
    compute_kernel_cfg = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
        fp32_dest_acc_en=False, packer_l1_acc=False,
    )

    tt_out = ttnn.transformer.paged_scaled_dot_product_attention_decode(
        tt_Q, tt_K, tt_V, tt_pt,
        cur_pos_tensor=tt_cp,
        scale=d**-0.5,
        program_config=program_cfg,
        compute_kernel_config=compute_kernel_cfg,
        memory_config=out_cfg,
    )

    out_per_dev = ttnn.get_device_tensors(tt_out)
    out_dev0 = ttnn.to_torch(out_per_dev[0]).float()
    logger.info(f"col0 dev0 (rect-grid) SDPA output shape: {tuple(out_dev0.shape)}")
    if out_dev0.shape[-2] >= n_q:
        out_dev0 = out_dev0[..., :n_q, :]

    _check_batch_identity(out_dev0, label="TG col-0 dev-0 SDPA output (RECT GRID), B_local=8")
