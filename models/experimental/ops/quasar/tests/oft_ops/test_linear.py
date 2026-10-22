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
