# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Per-op test: binary eltwise (``ttnn.add`` / ``sub`` / ``mul`` and their in-place forms) —
every distinct binary call the OFT model ran.

Shape source (kernel cache, binary_ng factory at ``1088b10b014``): binary_ng compute variants
(``eltwise_binary_sfpu_no_bcast``, ``eltwise_binary_no_bcast``, ``eltwise_binary_sfpu`` /
``eltwise_binary`` with ``BCAST_INPUT``, ``eltwise_binary_scalar``) plus the legacy
``eltwise_binary_kernel``. Reader/writer TensorAccessor args give page size (dtype),
interleaved L1 vs DRAM vs sharded, and the shard shape in tiles on the 20-core grid.

Groups:
  * OFT corner accumulation, per BEV slice: ``vox = TL - TR + BR - BL; vox *= area`` as
    in-place ``sub_`` / ``add_`` / ``mul_`` on float32 [1, 1, 576, 1792] TILE, all operands
    HEIGHT_SHARDED (32, 1792) on the 5x4 grid (SFPU, fp32 dest)                         [H]
  * BasicBlock residual add with fused ReLU (``PACK_RELU``), sharded:
      layer1 bf16 (1536, 64) [H]; layer1.0 bf16 + bfloat8_b (legacy binary path) [H];
      layer2 float32 (384, 128) [H]; topdown bf16 (1280, 128) with b in DRAM [H/L]
  * Encoder decode: dims ``x * log_dim_std + log_dim_mean`` with per-channel scalar
    broadcast on [1, 3, 159, 159] (a L1, b DRAM) [H bcast, M shape]; positions add [L];
    ``atan2``'s ``+/- pi`` scalar adds on [1, 159, 159] L1 [H]

Hash prefixes in ids are the compute kernel variant.
"""

import math

import pytest
import torch

import ttnn
from models.experimental.ops.quasar.tests.oft_ops import op_utils as U

VOX = (1, 1, U.SLICE_ROWS, U.VOX_C)
VOX_HS = ("hs", (U.TILE, U.VOX_C))

_INPLACE = {
    "sub_": ("subtract_", lambda a, b: a - b),
    "add_": ("add_", lambda a, b: a + b),
    "mul_": ("multiply_", lambda a, b: a * b),
}


def _run_inplace(mesh, op, shape, mem):
    tt_op, torch_op = _INPLACE[op]
    a = U.torch_rand(shape, dtype=torch.float32)
    b = U.torch_rand(shape, dtype=torch.float32)
    tt_a = U.to_tt_mem(a, mesh, mem, dtype=ttnn.float32)
    tt_b = U.to_tt_mem(b, mesh, mem, dtype=ttnn.float32)
    U.op(tt_op)(tt_a, tt_b)
    U.assert_pcc(torch_op(a, b), tt_a, pcc=0.9999, mesh_device=mesh)


@pytest.mark.blackhole_scale
@U.with_default_mesh()
@pytest.mark.parametrize(
    "op, shape",
    [
        pytest.param("sub_", VOX, id="vox-sub_-fp32-hs-691867"),
        pytest.param("add_", VOX, id="vox-add_-fp32-hs-167572"),
        pytest.param("mul_", VOX, id="vox-mul_area-fp32-hs-175368"),
    ],
)
def test_oft_corner_inplace_sharded(ttnn_mesh_device, reset_seeds, op, shape):
    _run_inplace(ttnn_mesh_device, op, shape, VOX_HS)


@U.with_default_mesh()
@pytest.mark.parametrize(
    "op, shape",
    [
        pytest.param("sub_", (1, 1, 64, U.VOX_C), id="proxy-vox-sub_-fp32-l1"),
        pytest.param("add_", (1, 1, 64, U.VOX_C), id="proxy-vox-add_-fp32-l1"),
        pytest.param("mul_", (1, 1, 64, U.VOX_C), id="proxy-vox-mul_-fp32-l1"),
    ],
)
def test_oft_corner_inplace(ttnn_mesh_device, reset_seeds, op, shape):
    """Same fp32 SFPU in-place ops on an L1-interleaved slice (emulator-sized)."""
    _run_inplace(ttnn_mesh_device, op, shape, "l1")


# fmt: off
# (a_shape, b_shape, a_dt, b_dt, a_mem, b_mem, out_mem)
RESIDUAL_CASES = [
    pytest.param((1, 1, 30720, 64), (1, 1, 30720, 64), "bf16", "bf16", ("hs", (1536, 64)), ("hs", (1536, 64)), ("hs", (1536, 64)), id="layer1-bf16-852339"),
    pytest.param((1, 1, 30720, 64), (1, 1, 30720, 64), "bf16", "bfp8", ("hs", (1536, 64)), ("hs", (1536, 64)), ("hs", (1536, 64)), id="layer1.0-bf16+bfp8-legacy-154311"),
    pytest.param((1, 1, 7680, 128), (1, 1, 7680, 128), "fp32", "fp32", ("hs", (384, 128)), ("hs", (384, 128)), ("hs", (384, 128)), id="layer2-fp32-828293"),
    pytest.param((1, 1, U.BEV_NHW, U.TD_C), (1, 1, U.BEV_NHW, U.TD_C), "bf16", "bf16", ("hs", (1280, 128)), "dram", ("hs", (1280, 128)), id="topdown-bf16-hs+dram-852339"),
]

RESIDUAL_SMALL_CASES = [
    pytest.param((1, 1, 1024, 64), (1, 1, 1024, 64), "bf16", "bf16", "l1", "l1", "l1", id="proxy-bf16"),
    pytest.param((1, 1, 1024, 128), (1, 1, 1024, 128), "fp32", "fp32", "l1", "l1", "l1", id="proxy-fp32"),
    pytest.param((1, 1, 1024, 128), (1, 1, 1024, 128), "bf16", "bf16", "l1", "dram", "l1", id="proxy-bf16-l1+dram"),
]
# fmt: on

_RES_ARGS = "a_shape, b_shape, a_dt, b_dt, a_mem, b_mem, out_mem"


def _pcc(*dts):
    return 0.99 if "bfp8" in dts else 0.999


@pytest.mark.blackhole_scale
@U.with_default_mesh()
@pytest.mark.parametrize(_RES_ARGS, RESIDUAL_CASES)
def test_add_residual_relu_sharded(ttnn_mesh_device, reset_seeds, a_shape, b_shape, a_dt, b_dt, a_mem, b_mem, out_mem):
    """BasicBlock ``out + identity`` with ReLU fused (PACK_RELU), model shard specs."""
    U.run_binary(
        ttnn_mesh_device,
        U.op("add"),
        lambda x, y: x + y,
        a_shape,
        b_shape,
        a_dt,
        b_dt,
        a_mem,
        b_mem,
        out_mem,
        relu=True,
        pcc=_pcc(a_dt, b_dt),
    )


@U.with_default_mesh()
@pytest.mark.parametrize(_RES_ARGS, RESIDUAL_SMALL_CASES)
def test_add_residual_relu(ttnn_mesh_device, reset_seeds, a_shape, b_shape, a_dt, b_dt, a_mem, b_mem, out_mem):
    U.run_binary(
        ttnn_mesh_device,
        U.op("add"),
        lambda x, y: x + y,
        a_shape,
        b_shape,
        a_dt,
        b_dt,
        a_mem,
        b_mem,
        out_mem,
        relu=True,
        pcc=_pcc(a_dt, b_dt),
    )


DIMS = (1, 3, U.BEV, U.BEV)
CHAN_SCALAR = (1, 3, 1, 1)


@U.with_default_mesh()
@pytest.mark.parametrize(
    "op, a_shape, b_shape",
    [
        # dims decode: dim_offsets * log_dim_std (SCALAR_B bcast, SFPU mul) + log_dim_mean (FPU add)
        pytest.param("mul", DIMS, CHAN_SCALAR, id="dims-mul-std-scalarbcast-127824"),
        pytest.param("add", DIMS, CHAN_SCALAR, id="dims-add-mean-scalarbcast-156177"),
        # positions decode: offsets + centres, no bcast (a L1, b DRAM) [L shape]
        pytest.param("add", DIMS, DIMS, id="pos-add-centres-232375"),
    ],
)
def test_decode_binary(ttnn_mesh_device, reset_seeds, op, a_shape, b_shape):
    tt_op, torch_op = {"mul": (U.op("multiply"), lambda x, y: x * y), "add": (U.op("add"), lambda x, y: x + y)}[op]
    U.run_binary(ttnn_mesh_device, tt_op, torch_op, a_shape, b_shape, "bf16", "bf16", "l1", "dram", "l1", pcc=0.999)


@U.with_default_mesh()
@pytest.mark.parametrize(
    "op, shape",
    [
        pytest.param("add", U.HEATMAP, id="atan2-add-pi-584471"),
        pytest.param("sub", U.HEATMAP, id="atan2-sub-pi-110119"),
    ],
)
def test_binary_scalar(ttnn_mesh_device, reset_seeds, op, shape):
    """``atan2`` quadrant fix-ups: tensor +/- pi as a binary_ng scalar op, bf16 L1."""
    mesh = ttnn_mesh_device
    x = U.torch_rand(shape)
    tt_x = U.to_tt_mem(x, mesh, "l1")
    if op == "add":
        out, ref = U.op("add")(tt_x, math.pi, memory_config=ttnn.L1_MEMORY_CONFIG), x.float() + math.pi
    else:
        out, ref = U.op("subtract")(tt_x, math.pi, memory_config=ttnn.L1_MEMORY_CONFIG), x.float() - math.pi
    U.assert_pcc(ref, out, pcc=0.999, mesh_device=mesh)
