# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Per-op test: ``ttnn.linear`` — the OFT voxel-feature projection (the reference model's
``conv3d``, flattened to a linear over 7 height points x 256 channels).

Shape source (kernel cache): ``bmm_large_block_zm_fused_bias_activation/161146`` with
``reader_bmm_tile_layout_in0_sender_padding/602379`` and in1 sender/receiver
(113636 / 113458). From compile-time args and CB formats:

  * in0 [1, 1, 576, 1792] float32 TILE, HEIGHT_SHARDED (32, 1792) on the 5x4 grid
    (576 rows -> 18 of 20 shards used)                                               [H]
  * weights [1792, 128] bf16 TILE DRAM, bias [1, 128] bf16 DRAM                       [H]
  * output bf16 HEIGHT_SHARDED (32, 128)                                               [H]
  * ``MatmulMultiCoreReuseMultiCast1DProgramConfig``: grid (5, 4), in0_block_w=8,
    subblock 1x4, per_core_M=1, per_core_N=4, out_block 1x4, fuse_batch, fused RELU,
    mcast_in0=False                                                                     [H]
  * LoFi, math_approx_mode=True, fp32_dest_acc off, packer_l1_acc off                  [H]

Called once per BEV slice (44 slices of 576 rows).

Second cache (in-tree OFT at ``628a4797bf7``: 3 scales, 256 output channels, bf16 activations).
The linear variants are the two all-bf16 ``bmm_large_block_zm_fused_bias_activation`` kernels
(FUSE_BIAS + PACK_RELU, LoFi, no fp32 dest acc, no packer L1 acc) 136177 / 280461; the bfloat8_b
weight / HiFi2 / fp32-acc bmm variants in that cache are the conv 1x1 (lateral) matmul paths:

  * in0 [1, 1, rows, 1792] bf16 TILE HEIGHT_SHARDED (32 * per_core_M, 1792), rows = 1408 / 2112 /
    2304 (``reader_bmm_tile_layout_in0_sender_padding`` 708347 / 803838 / 120639: sharded
    accessor 20 banks, shard 168 / 224 / 224 tiles, tensor 2464 / 3696 / 4032 tiles)            [H]
  * weights [1792, 256] bf16 TILE DRAM, bias [1, 256] bf16 DRAM (in1 sender accessors)          [H]
  * output bf16 HEIGHT_SHARDED (32 * per_core_M, 256), 15 / 17 / 18 shards used                  [H]
  * 1D config: grid (5, 4), in0_block_w=8 (7 blocks), per_core_M = 3 / 4 / 4, per_core_N=8,
    subblock 1x8, out_block (per_core_M, 8), fuse_batch, fused RELU, mcast_in0=False              [H]
  * LoFi, math_approx_mode=True, fp32_dest_acc off, packer_l1_acc off (model ``Linear``)       [H/M]
"""

import pytest
import torch

import ttnn
from models.experimental.ops.quasar.tests.oft_ops import op_utils as U

K, N = U.VOX_C, U.TD_C


def _compute_config(mesh):
    return ttnn.init_device_compute_kernel_config(
        mesh.arch(),
        math_fidelity=ttnn.MathFidelity.LoFi,
        math_approx_mode=True,
        fp32_dest_acc_en=False,
        packer_l1_acc=False,
        dst_full_sync_en=False,
    )


def _inputs(mesh, m):
    x = U.torch_rand((1, 1, m, K), dtype=torch.float32)
    w = U.torch_rand((K, N), std=K**-0.5)
    b = U.torch_rand((1, N))
    tt_w = U.to_tt(w, mesh)
    tt_b = U.to_tt(b, mesh)
    ref = torch.relu(x.to(torch.bfloat16).float() @ w.float() + b.float())
    return x, tt_w, tt_b, ref


@pytest.mark.blackhole_scale
@U.with_default_mesh()
@pytest.mark.parametrize("shape", [pytest.param((1, 1, U.SLICE_ROWS, K), id="oft-linear-576x1792x128-161146")])
def test_linear_oft_program_config(ttnn_mesh_device, reset_seeds, shape):
    """Model-faithful: sharded fp32 in0, the cached 1D program config on the 5x4 grid."""
    mesh = ttnn_mesh_device
    m = shape[2]
    x, tt_w, tt_b, ref = _inputs(mesh, m)
    tt_x = U.to_tt_mem(x, mesh, ("hs", (U.TILE, K)), dtype=ttnn.float32)

    program_config = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=(U.MODEL_GRID_X, U.MODEL_GRID_Y),
        in0_block_w=8,
        out_subblock_h=1,
        out_subblock_w=4,
        out_block_h=1,
        out_block_w=4,
        per_core_M=1,
        per_core_N=4,
        fuse_batch=True,
        fused_activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
        mcast_in0=False,
    )
    out = U.op("linear")(
        tt_x,
        tt_w,
        bias=tt_b,
        program_config=program_config,
        memory_config=U.resolve_memcfg(("hs", (U.TILE, N)), mesh),
        dtype=ttnn.bfloat16,
        compute_kernel_config=_compute_config(mesh),
    )
    U.assert_pcc(ref, out, pcc=0.99, mesh_device=mesh)


@U.with_default_mesh()
@pytest.mark.parametrize(
    "shape",
    [
        pytest.param((1, 1, U.SLICE_ROWS, K), id="oft-linear-576x1792x128-auto"),
        pytest.param((1, 1, 64, K), id="proxy-oft-linear-64x1792x128"),
    ],
)
def test_linear_oft(ttnn_mesh_device, reset_seeds, shape):
    """Same op (fp32 in0, bf16 weights/bias/output, fused ReLU, LoFi) without the pinned grid."""
    mesh = ttnn_mesh_device
    m = shape[2]
    x, tt_w, tt_b, ref = _inputs(mesh, m)
    tt_x = U.to_tt_mem(x, mesh, "l1", dtype=ttnn.float32)
    out = U.op("linear")(
        tt_x,
        tt_w,
        bias=tt_b,
        activation="relu",
        memory_config=ttnn.L1_MEMORY_CONFIG,
        dtype=ttnn.bfloat16,
        compute_kernel_config=_compute_config(mesh),
    )
    U.assert_pcc(ref, out, pcc=0.99, mesh_device=mesh)


# =============================================================================
# Cache #2: bf16 in0, 256 output channels, per-scale slice heights
# =============================================================================

N2 = 2 * U.TD_C  # 256: the in-tree OFT linear keeps the lateral channel count


def _inputs2(mesh, m):
    x = U.torch_rand((1, 1, m, K))  # bf16 grid_sample corner sums
    w = U.torch_rand((K, N2), std=K**-0.5)
    b = U.torch_rand((1, N2))
    ref = torch.relu(x.float() @ w.float() + b.float())
    return x, U.to_tt(w, mesh), U.to_tt(b, mesh), ref


@pytest.mark.blackhole_scale
@U.with_default_mesh()
@pytest.mark.parametrize(
    "shape",
    [
        pytest.param((1, 1, U.BEV_ROWS // 18, K), id="oft8-linear-1408x1792x256-708347"),
        pytest.param((1, 1, U.BEV_ROWS // 12, K), id="oft16-linear-2112x1792x256-803838"),
        pytest.param((1, 1, U.BEV_ROWS // 11, K), id="oft32-linear-2304x1792x256-120639"),
    ],
)
def test_linear_oft2_program_config(ttnn_mesh_device, reset_seeds, shape):
    """Model-faithful (cache #2): sharded bf16 in0, per_core_M = ceil(rows / 32 / 20), N=256, 5x4 grid."""
    mesh = ttnn_mesh_device
    m = shape[2]
    per_core_m = U.hs_rows(m) // U.TILE  # 3 / 4 / 4 (bmm 136177 / 280461 / 280461)
    per_core_n = N2 // U.TILE
    x, tt_w, tt_b, ref = _inputs2(mesh, m)
    tt_x = U.to_tt_mem(x, mesh, ("hs", (per_core_m * U.TILE, K)))

    program_config = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=(U.MODEL_GRID_X, U.MODEL_GRID_Y),
        in0_block_w=8,
        out_subblock_h=1,
        out_subblock_w=per_core_n,
        out_block_h=per_core_m,
        out_block_w=per_core_n,
        per_core_M=per_core_m,
        per_core_N=per_core_n,
        fuse_batch=True,
        fused_activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
        mcast_in0=False,
    )
    out = U.op("linear")(
        tt_x,
        tt_w,
        bias=tt_b,
        program_config=program_config,
        memory_config=U.resolve_memcfg(("hs", (per_core_m * U.TILE, N2)), mesh),
        dtype=ttnn.bfloat16,
        compute_kernel_config=_compute_config(mesh),
    )
    U.assert_pcc(ref, out, pcc=0.99, mesh_device=mesh)


@U.with_default_mesh()
@pytest.mark.parametrize("shape", [pytest.param((1, 1, 64, K), id="proxy-oft2-linear-64x1792x256")])
def test_linear_oft2(ttnn_mesh_device, reset_seeds, shape):
    """Same op (bf16 in0 / weights / bias / output, N=256, fused ReLU, LoFi) without the pinned grid."""
    mesh = ttnn_mesh_device
    x, tt_w, tt_b, ref = _inputs2(mesh, shape[2])
    tt_x = U.to_tt_mem(x, mesh, "l1")
    out = U.op("linear")(
        tt_x,
        tt_w,
        bias=tt_b,
        activation="relu",
        memory_config=ttnn.L1_MEMORY_CONFIG,
        dtype=ttnn.bfloat16,
        compute_kernel_config=_compute_config(mesh),
    )
    U.assert_pcc(ref, out, pcc=0.99, mesh_device=mesh)
