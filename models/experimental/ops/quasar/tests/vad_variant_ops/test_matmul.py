# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Per-op test: ``ttnn.matmul`` — batched / attention matmuls (no weights, no bias).

Shape source (kernel cache):
  * Batched ``[24, 10000, 4] @ [24, n, 4]^T`` (``IN1_TRANSPOSE_TILE``) — 1D mcast_in1,
    per-core 16x1 tiles, HiFi2 (ttnn-chosen config), bf16 DRAM in/out. K=4 from
    ``last_ktile_w`` [H]; batch 24 = 6 cams x 4 Z-anchors and n <= 32 inferred [M].
    Compute ``bmm_large_block_zm_fused_bias_activation/358023``.
  * Multi-head attention bmms on the reuse factory (``reader_bmm_tile_layout_in0`` +
    ``reader_writer_bmm_tile_layout_in1``): in0_block_w 1, subblock 1x1, one batch per core,
    LoFi => called with ``core_grid=`` [H]. K = 32 (R1) and K = 16 (R2) [H]; batch <= 20,
    assumed 8 heads [L]. in0 in L1 or DRAM, out in L1 or DRAM (both variants cached).
  * Interleaved multicore ``bmm.cpp`` (HiFi4): in0 bf16, in1 bfp8 DRAM, Kt 8 and 16 [H];
    M x N only bounded by tiles-per-core (MC1 121-139 output tiles, MC2 61-79, MC3/MC4 <= 20)
    [L]. Representative M/N chosen inside those bounds.
"""

import pytest

import ttnn
from models.experimental.ops.quasar.tests.vad_variant_ops import op_utils as U


def _seen(tt, ref, mesh):
    return U.from_tt(tt, mesh).reshape(-1)[: ref.numel()].reshape(ref.shape)


@U.with_default_mesh()
@pytest.mark.parametrize(
    "a_shape, b_shape",
    [pytest.param((24, U.NUM_QUERY, 4), (24, 32, 4), id="L1-transpose_b-k4-358023")],
)
def test_matmul_transpose_b(ttnn_mesh_device, reset_seeds, a_shape, b_shape):
    mesh = ttnn_mesh_device
    a_torch, b_torch = U.torch_rand(a_shape), U.torch_rand(b_shape)
    a, b = U.to_tt(a_torch, mesh), U.to_tt(b_torch, mesh)
    out = ttnn.matmul(a, b, transpose_b=True)
    U.assert_pcc(a_torch.float() @ b_torch.float().transpose(-1, -2), out, pcc=0.999, mesh_device=mesh)


# fmt: off
# (a_shape, b_shape, a_mem, out_mem)
MHA_CASES = [
    pytest.param((1, U.NUM_HEADS, 32, 32), (1, U.NUM_HEADS, 32, 32), "l1", "l1", id="R1-k32-l1-818765"),
    pytest.param((1, U.NUM_HEADS, 32, 32), (1, U.NUM_HEADS, 32, 32), "l1", "dram", id="R1-k32-dram-out-818765"),
    pytest.param((1, U.NUM_HEADS, 32, 16), (1, U.NUM_HEADS, 16, 32), "l1", "l1", id="R2-k16-l1-818765"),
    pytest.param((1, U.NUM_HEADS, 32, 16), (1, U.NUM_HEADS, 16, 32), "dram", "dram", id="R2-k16-dram-818765"),
]
# fmt: on


@U.with_default_mesh()
@pytest.mark.parametrize("a_shape, b_shape, a_mem, out_mem", MHA_CASES)
def test_matmul_mha(ttnn_mesh_device, reset_seeds, a_shape, b_shape, a_mem, out_mem):
    mesh = ttnn_mesh_device
    U.model_grid(mesh)
    a_torch, b_torch = U.torch_rand(a_shape), U.torch_rand(b_shape)
    a = U.to_tt_mem(a_torch, mesh, a_mem)
    b = U.to_tt_mem(b_torch, mesh, "l1")
    out = ttnn.matmul(
        a,
        b,
        core_grid=ttnn.CoreGrid(y=U.MODEL_GRID_Y, x=U.MODEL_GRID_X),
        memory_config=U.resolve_memcfg(out_mem, mesh),
        compute_kernel_config=ttnn.init_device_compute_kernel_config(
            mesh.arch(), math_fidelity=ttnn.MathFidelity.LoFi, math_approx_mode=False, fp32_dest_acc_en=False
        ),
    )
    U.assert_pcc(a_torch.float() @ b_torch.float(), out, pcc=0.99, mesh_device=mesh)


# fmt: off
# (a_shape, b_shape, a_mem, out_mem)
MC_CASES = [
    pytest.param((1, 1, 512, 256), (256, 256), "l1", "l1", id="MC1-kt8-128tiles-128436"),
    pytest.param((1, 1, 256, 256), (256, 256), "dram", "dram", id="MC2-kt8-64tiles-139452"),
    pytest.param((1, 1, 64, 256), (256, 256), "l1", "dram", id="MC3-kt8-16tiles-111944"),
    pytest.param((1, 1, 32, 512), (512, 512), "dram", "l1", id="MC4-kt16-16tiles-160512"),
]
# fmt: on


@U.with_default_mesh()
@pytest.mark.parametrize("a_shape, b_shape, a_mem, out_mem", MC_CASES)
def test_matmul_bfp8_weights(ttnn_mesh_device, reset_seeds, a_shape, b_shape, a_mem, out_mem):
    mesh = ttnn_mesh_device
    a_torch = U.torch_rand(a_shape)
    b_torch = U.torch_rand(b_shape, std=b_shape[0] ** -0.5)
    a = U.to_tt_mem(a_torch, mesh, a_mem)
    b = U.to_tt(b_torch, mesh, dtype=ttnn.bfloat8_b)
    out = ttnn.matmul(
        a,
        b,
        memory_config=U.resolve_memcfg(out_mem, mesh),
        compute_kernel_config=ttnn.init_device_compute_kernel_config(
            mesh.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False
        ),
    )
    U.assert_pcc(a_torch.float() @ _seen(b, b_torch, mesh), out, pcc=0.999, mesh_device=mesh)
