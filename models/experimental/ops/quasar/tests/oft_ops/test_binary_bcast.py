# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Per-op test: binary_ng broadcast / scalar calls from the **second** OFT kernel cache (in-tree
model at ``628a4797bf7``): input normalization and the encoder decode. ``test_binary.py`` does
not cover these: its decode cases come from cache #1 (NCHW decode with per-channel scalar
broadcast).

Shape source (kernel cache #2, binary_ng factory at ``628a4797bf7``). Reader / writer
TensorAccessor compile-time args give each operand's buffer (0 = L1 interleaved, 2 = DRAM
interleaved, 17 = sharded) and the 2048-B bf16 tile page. Every compute kernel is bf16 in and out,
HiFi4, approx off, no fp32 dest acc. ``mul`` without ``fast_and_approximate_mode`` routes to the
SFPU, and ``add`` / ``sub`` route to the FPU.

  * Input normalization ``(x - mean) / std`` (``tt_oftnet.forward_normalization``): x is
    [1, 384, 1280, 3] bf16 ROW_MAJOR DRAM and mean / std are 1-D [3] bf16 TILE DRAM. binary_ng
    tilizes x and broadcasts the row (``ROW_B``, LLK row broadcast). The reader is
    ``reader_interleaved_row_bcast/930959`` (a DRAM, b DRAM) and the output is DRAM.
      - sub: ``eltwise_binary_row_bcast/540389`` (FPU ``sub_tiles``)                       [H op/mem, M shape]
      - div: ``eltwise_binary_sfpu_row_bcast/172958`` (``div_binary_tile``)                 [H op/mem, M shape]
        The in-tree code passes ``fast_and_approximate_mode=True``, which would give an FPU
        ``mul`` with a reciprocal on the RHS. The cache has the SFPU divide, i.e. fast mode off.
        The cache is treated as ground truth.
  * Encoder decode (``tt_encoder``). Offsets come from the head as L1 bf16 TILE and are
    permuted to [1, 159, 159, 3] (``test_exp_decode.py``). The constants are bf16 TILE DRAM:
    ``pos_std`` is 1-D [3], and ``log_dim_std`` / ``log_dim_mean`` are [1, 3].
      - ``offsets * std``: ``eltwise_binary_sfpu_row_bcast/123169`` (SFPU ``mul_binary_tile``,
        ``ROW_B``) + ``reader_interleaved_row_bcast/736061`` (a L1, b DRAM), out L1. The
        positions and dimensions decodes compile to the same kernel                  [H op/mem, M shape]
      - ``+ log_dim_mean``: ``eltwise_binary_row_bcast/508195`` (FPU ``add_tiles``, ``ROW_B``),
        same reader, out L1                                                           [H op/mem, M shape]
      - ``centers = grid[1:, 1:] + grid[:-1, :-1]``: [159, 159, 3] slices of the [160, 160, 3]
        bf16 TILE DRAM BEV grid. Compute is ``eltwise_binary_no_bcast/761319`` and the reader is
        ``reader_interleaved_no_bcast`` with a and b both DRAM. The output is DRAM          [H op/mem, M shape]
      - ``divide(centers, 2.0, fast_and_approximate_mode=True)``: ``eltwise_binary_scalar/380187``
        (FPU ``mul_tiles`` with ``recip_tile`` as the RHS activation) and
        ``writer_interleaved_scalar/103210`` (DRAM out)                                 [H op/mem, M shape]
      - ``positions = offsets * pos_std + centers``: ``eltwise_binary_no_bcast/761319``. The
        reader is ``reader_interleaved_no_bcast/527748`` (a L1, b DRAM). b has rank 3, so only the
        batch dim broadcasts and the subtile broadcast is NONE. The output is L1          [H op/mem, M shape]

The integral-image scaler ``mul`` (``eltwise_binary_sfpu/149548``, ``SCALAR_B``) and the bf16 sharded corner
``mul_`` (``eltwise_binary_sfpu_no_bcast/492671``) come from the same cache. ``test_binary.py``
covers both (``test_oft2_intimg_scale_mul``, ``test_oft2_corner_inplace*``).

Hash suffixes are the compute kernel variant, or the reader where the compute kernel is shared.
"""

import pytest
import torch

import ttnn
from models.experimental.ops.quasar.tests.oft_ops import op_utils as U

# Model constants (in-tree OFT reference / encoder defaults).
IMG_MEAN = [0.485, 0.456, 0.406]
IMG_STD = [0.229, 0.224, 0.225]
POS_STD = [0.5, 0.36, 0.5]
LOG_DIM_MEAN = [[0.42, 0.48, 1.35]]
LOG_DIM_STD = [[0.085, 0.067, 0.115]]

IMG_NHWC = (1, U.IMG_H, U.IMG_W, 3)
DECODE = (1, U.BEV, U.BEV, 3)  # permuted pos / dim offsets
CENTERS = (U.BEV, U.BEV, 3)  # grid[1:, 1:] slice of the [160, 160, 3] BEV grid


def _const(values):
    """``b_transform`` that replaces the random operand with a model constant."""
    return lambda b: torch.tensor(values, dtype=torch.bfloat16).reshape(b.shape)


# =============================================================================
# Input normalization: row broadcast of a 1-D [3] constant over NHWC channels
# =============================================================================


@U.with_default_mesh()
@pytest.mark.parametrize(
    "op, a_shape",
    [
        pytest.param("sub", IMG_NHWC, id="norm-sub-mean-rm-a-rowbcast-540389"),
        pytest.param("div", IMG_NHWC, id="norm-div-std-rowbcast-172958"),
        pytest.param("sub", (1, 32, 64, 3), id="proxy-norm-sub-mean-rm-a-rowbcast"),
        pytest.param("div", (1, 32, 64, 3), id="proxy-norm-div-std-rowbcast"),
    ],
)
def test_input_normalization(ttnn_mesh_device, reset_seeds, op, a_shape):
    """``x - mean`` (x ROW_MAJOR, tilized by binary_ng) and ``x / std`` (TILE), all DRAM."""
    mesh = ttnn_mesh_device
    if op == "sub":
        x = torch.rand(a_shape).to(torch.bfloat16)  # image in [0, 1]
        b = torch.tensor(IMG_MEAN, dtype=torch.bfloat16)
        tt_x = U.to_tt(x, mesh, layout=ttnn.ROW_MAJOR_LAYOUT)
        out = U.op("subtract")(tt_x, U.to_tt(b, mesh))
        ref = x.float() - b.float()
    else:
        x = (torch.rand(a_shape) - 0.45).to(torch.bfloat16)  # mean-subtracted image
        b = torch.tensor(IMG_STD, dtype=torch.bfloat16)
        tt_x = U.to_tt(x, mesh)
        # Cache #2 has the exact SFPU divide (fast_and_approximate_mode off), not the in-tree fast mode.
        out = U.op("divide")(tt_x, U.to_tt(b, mesh))
        ref = x.float() / b.float()
    U.assert_shape_dtype(out, shape=a_shape, dtype=ttnn.bfloat16)
    U.assert_pcc(ref, out, pcc=0.999, mesh_device=mesh)


# =============================================================================
# Encoder decode: row broadcast of the [3] / [1, 3] decode constants, L1 activations
# =============================================================================

# (op, a_shape, b_shape, constant)
_ROW_BCAST_CASES = [
    pytest.param("mul", DECODE, (3,), POS_STD, id="pos-mul-std-1d-rowbcast-123169"),
    pytest.param("mul", DECODE, (1, 3), LOG_DIM_STD, id="dims-mul-std-rowbcast-123169"),
    pytest.param("add", DECODE, (1, 3), LOG_DIM_MEAN, id="dims-add-mean-rowbcast-508195"),
]

_TT_OPS = {
    "add": ("add", lambda x, y: x + y),
    "mul": ("multiply", lambda x, y: x * y),
}


@U.with_default_mesh()
@pytest.mark.parametrize("op, a_shape, b_shape, const", _ROW_BCAST_CASES)
def test_decode_row_bcast(ttnn_mesh_device, reset_seeds, op, a_shape, b_shape, const):
    """``offsets * std`` / ``+ mean``: a L1 [1, 159, 159, 3], b DRAM constant, out L1."""
    tt_op, torch_op = _TT_OPS[op]
    U.run_binary(
        ttnn_mesh_device,
        U.op(tt_op),
        torch_op,
        a_shape,
        b_shape,
        "bf16",
        "bf16",
        "l1",
        "dram",
        "l1",
        b_transform=_const(const),
        pcc=0.999,
    )


@U.with_default_mesh()
@pytest.mark.parametrize(
    "a_shape, b_shape, a_mem, out_mem",
    [
        # centers = grid[1:, 1:] + grid[:-1, :-1]: both slices in DRAM, out DRAM
        pytest.param(CENTERS, CENTERS, "dram", "dram", id="centers-add-grid-dram-761319"),
        # positions = (offsets * pos_std) + centers: rank-4 a in L1, rank-3 b in DRAM (batch broadcast only)
        pytest.param(DECODE, CENTERS, "l1", "l1", id="pos-add-centers-rank3b-527748"),
    ],
)
def test_decode_add(ttnn_mesh_device, reset_seeds, a_shape, b_shape, a_mem, out_mem):
    U.run_binary(
        ttnn_mesh_device,
        U.op("add"),
        lambda x, y: x + y,
        a_shape,
        b_shape,
        "bf16",
        "bf16",
        a_mem,
        "dram",
        out_mem,
        pcc=0.999,
    )


@U.with_default_mesh()
@pytest.mark.parametrize("shape", [pytest.param(CENTERS, id="centers-div2-fast-scalar-380187")])
def test_decode_divide_scalar_fast(ttnn_mesh_device, reset_seeds, shape):
    """``divide(centers, 2.0, fast_and_approximate_mode=True)``: FPU mul by ``recip(2.0)``, DRAM."""
    mesh = ttnn_mesh_device
    # Cell-corner sums: x in [-80, 79], y = 2 * 1.74, z in [0, 159].
    x = (torch.rand(shape) * 160 - 80).to(torch.bfloat16)
    out = U.op("divide")(U.to_tt(x, mesh), 2.0, fast_and_approximate_mode=True)
    U.assert_shape_dtype(out, shape=shape, dtype=ttnn.bfloat16)
    U.assert_pcc(x.float() / 2.0, out, pcc=0.9999, mesh_device=mesh)
