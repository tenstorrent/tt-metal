# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Per-op test: ``ttnn.cumsum`` on uint32 — the in-tree OFT integral image with
``integral_image_quantization_strategy="to_uint32"`` (``ttnn_integral_image_channel_last``):

    tmp = ttnn.cumsum(features_nhwc, dim=1, dtype=uint32)   # over H
    out = ttnn.cumsum(tmp, dim=2, dtype=uint32)             # over W

Shape source (second kernel cache, accumulation factory at ``628a4797bf7``):

  * one ``accumulation_compute`` variant (139294): ``BINARY_OP add_int_tile<DataFormat::UInt32>``,
    ``FILL_TILE fill_tile_int<UInt32>``, acc value 0, HiFi4, every CB uint32; reader / writer
    (152437 / 409864) accessors: DRAM interleaved 4096-B uint32 tiles                         [H]
  * dims: the factory keeps shape and dim in runtime args, so they come from the model code.
    Both dims are < rank - 2, so ``accumulation_invoke`` swaps the axis with dim 0 around the
    device op, and the matching uint32 permutes are in the cache: one
    ``reader/writer_unary_transpose_cn`` pair (dim=1, swap N<->C) and six
    ``transpose_hc`` / ``permute_..._row_invariant`` pairs (dim=2: 3 scales x in/out)      [H op, M dims]
  * features [1, H, W, 256] uint32 TILE DRAM at (48, 160) / (24, 80) / (12, 40)            [M]

The mainline accumulation factory is Metal 2.0 (``ProgramSpec``) and its internal permute is too,
so ``ttnn.cumsum`` is called directly. The full quantize -> cumsum x2 -> dequantize chain is
checked as well (pieces in test_binary.py and test_typecast.py).
"""

import pytest
import torch

import ttnn
from models.experimental.ops.quasar.tests.oft_ops import op_utils as U

OFT8 = (1, U.L2_H, U.L2_W, U.OFT_C)
OFT16 = (1, U.L2_H // 2, U.L2_W // 2, U.OFT_C)
OFT32 = (1, U.L2_H // 4, U.L2_W // 4, U.OFT_C)

PRESCALE = 2.0**20  # tt_oft.py prescaler 1024 * 1024


def _host_int(tt_tensor, shape):
    """Exact integer readback (``U.from_tt`` returns float32, which rounds above 2^24)."""
    return ttnn.to_torch(tt_tensor).to(torch.int64).reshape(-1)[: torch.Size(shape).numel()].reshape(shape)


@U.with_default_mesh()
@pytest.mark.parametrize(
    "shape, dim",
    [
        pytest.param(OFT8, 1, id="oft8-1x48x160x256-u32-dim1-139294"),
        pytest.param(OFT8, 2, id="oft8-1x48x160x256-u32-dim2-139294"),
        pytest.param(OFT16, 1, id="oft16-1x24x80x256-u32-dim1-139294"),
        pytest.param(OFT16, 2, id="oft16-1x24x80x256-u32-dim2-139294"),
        pytest.param(OFT32, 1, id="oft32-1x12x40x256-u32-dim1-139294"),
        pytest.param(OFT32, 2, id="oft32-1x12x40x256-u32-dim2-139294"),
    ],
)
def test_cumsum_uint32(ttnn_mesh_device, reset_seeds, shape, dim):
    """One integral-image pass. Integer add: checked exactly (sums kept below 2^31)."""
    mesh = ttnn_mesh_device
    x = torch.randint(0, 2**16, shape, dtype=torch.int32)
    tt_x = U.to_tt(x, mesh, dtype=ttnn.uint32)  # uint32 TILE DRAM
    out = ttnn.cumsum(tt_x, dim=dim, dtype=ttnn.uint32)
    U.assert_shape_dtype(out, shape=shape, dtype=ttnn.uint32, finite=False)
    got = _host_int(out, shape)
    ref = torch.cumsum(x.to(torch.int64), dim=dim)
    assert torch.equal(got, ref), f"{(got != ref).sum().item()}/{ref.numel()} elements differ"


@U.with_default_mesh()
@pytest.mark.parametrize(
    "shape",
    [
        pytest.param(OFT8, id="oft8-intimg-u32-chain-1x48x160x256-139294"),
        pytest.param(OFT32, id="oft32-intimg-u32-chain-1x12x40x256-139294"),
    ],
)
def test_integral_image_uint32_chain(ttnn_mesh_device, reset_seeds, shape):
    """``tt_oft.py`` "to_uint32" integral image end to end: mul 2^20 -> typecast uint32 -> cumsum
    dim 1 -> cumsum dim 2 -> typecast bf16 -> mul 2^-20, on bf16 TILE DRAM features."""
    mesh = ttnn_mesh_device
    _, h, w, _ = shape
    # Post-ReLU features, bounded so the full H x W sum of x * 2^20 stays below 2^31 (no uint32 wrap).
    x = (U.rand_nonneg(shape) * (2.0**31 / PRESCALE / (h * w)) * 0.9).to(torch.bfloat16)
    tt_x = U.to_tt(x, mesh)

    prescaler = ttnn.from_torch(torch.tensor(PRESCALE), device=mesh, dtype=ttnn.bfloat16)
    postscaler = ttnn.from_torch(torch.tensor(1.0 / PRESCALE), device=mesh, dtype=ttnn.bfloat16)

    q = U.op("multiply")(tt_x, prescaler, dtype=ttnn.bfloat16)
    q = U.op("typecast")(q, ttnn.uint32)
    q = ttnn.cumsum(q, dim=1, dtype=ttnn.uint32)
    q = ttnn.cumsum(q, dim=2, dtype=ttnn.uint32)
    out = U.op("typecast")(q, ttnn.bfloat16)
    out = U.op("multiply")(out, postscaler, dtype=ttnn.bfloat16)

    U.assert_shape_dtype(out, shape=shape, dtype=ttnn.bfloat16, finite=False)
    ref = torch.cumsum(torch.cumsum(x.double(), dim=1), dim=2)
    U.assert_pcc(ref, out, pcc=0.999, mesh_device=mesh)
