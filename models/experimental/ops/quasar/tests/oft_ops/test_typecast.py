# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Per-op test: ``ttnn.typecast`` bf16 <-> uint32 — the quantize / dequantize around the uint32
integral image of the in-tree OFT model (``integral_image_quantization_strategy="to_uint32"``):

    features = mul(features, 2^20); features = typecast(features, uint32)
    integral = cumsum(cumsum(features, 1), 2)          # see test_cumsum.py
    integral = typecast(integral, bf16); integral = mul(integral, 2^-20)

Shape source (second kernel cache, factories at ``628a4797bf7``): ten ``eltwise_typecast``
variants, ``TYPECAST_LLK typecast_tile<5u, 24u>`` (Float16_b -> UInt32) and ``<24u, 5u>``
(UInt32 -> Float16_b), HiFi4, CB formats fp16b / uint32. ct[0] (tiles per core) pairs per scale
on a 20-core ``split_work_to_cores``:

  * [1, 48, 160, 256]: 1920 tiles -> 96 per core (180382 / 165911)                        [H]
  * [1, 24, 80, 256] (W padded to 96): 576 tiles -> 29 | 28 (505250, 846936 / 182001, 276016) [H]
  * [1, 12, 40, 256] (W padded to 64): 192 tiles -> 10 | 9 (164563, 184194 / 399861, 246810)   [H]

Reader / writer ``*_unary_interleaved_start_id`` accessors: DRAM interleaved, 2048-B bf16 tiles
in and 4096-B uint32 tiles out (and the reverse)                                            [H]

The in-tree code defaults to ``None`` (``ttnn.experimental.intimg``), but no intimg kernel is in
this cache: the run used ``"to_uint32"``. Routed through ``U.op("typecast")`` (Quasar port exists).
"""

import pytest
import torch

import ttnn
from models.experimental.ops.quasar.tests.oft_ops import op_utils as U

OFT8 = (1, U.L2_H, U.L2_W, U.OFT_C)
OFT16 = (1, U.L2_H // 2, U.L2_W // 2, U.OFT_C)
OFT32 = (1, U.L2_H // 4, U.L2_W // 4, U.OFT_C)


def _host_int(tt_tensor, shape):
    """Exact integer readback (``U.from_tt`` returns float32, which rounds above 2^24)."""
    return ttnn.to_torch(tt_tensor).to(torch.int64).reshape(-1)[: torch.Size(shape).numel()].reshape(shape)


@U.with_default_mesh()
@pytest.mark.parametrize(
    "shape",
    [
        pytest.param(OFT8, id="oft8-1x48x160x256-bf16-u32-180382"),
        pytest.param(OFT16, id="oft16-1x24x80x256-bf16-u32-505250"),
        pytest.param(OFT32, id="oft32-1x12x40x256-bf16-u32-164563"),
    ],
)
def test_typecast_bf16_to_uint32(ttnn_mesh_device, reset_seeds, shape):
    """Prescaled features (x * 2^20, non-negative) -> uint32. Integer-valued bf16 inputs: exact."""
    mesh = ttnn_mesh_device
    x = torch.randint(0, 2**20, shape).to(torch.bfloat16)  # bf16 rounding keeps them integers
    tt_x = U.to_tt(x, mesh)  # bf16 TILE DRAM
    out = U.op("typecast")(tt_x, ttnn.uint32, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    U.assert_shape_dtype(out, shape=shape, dtype=ttnn.uint32, finite=False)
    got = _host_int(out, shape)
    ref = x.to(torch.int64)
    assert torch.equal(got, ref), f"{(got != ref).sum().item()}/{ref.numel()} elements differ"


@U.with_default_mesh()
@pytest.mark.parametrize(
    "shape",
    [
        pytest.param(OFT8, id="oft8-1x48x160x256-u32-bf16-165911"),
        pytest.param(OFT16, id="oft16-1x24x80x256-u32-bf16-182001"),
        pytest.param(OFT32, id="oft32-1x12x40x256-u32-bf16-399861"),
    ],
)
def test_typecast_uint32_to_bf16(ttnn_mesh_device, reset_seeds, shape):
    """uint32 integral image -> bf16 (values kept below 2^31 so the int32 host view is exact)."""
    mesh = ttnn_mesh_device
    x = torch.randint(0, 2**30, shape, dtype=torch.int32)
    tt_x = U.to_tt(x, mesh, dtype=ttnn.uint32)  # uint32 TILE DRAM
    out = U.op("typecast")(tt_x, ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    U.assert_shape_dtype(out, shape=shape, dtype=ttnn.bfloat16, finite=False)
    ref = x.double()
    got = U.from_tt(out, mesh).reshape(shape).double()
    U.assert_pcc(ref, host=got, pcc=0.9999)
    # bf16 keeps 8 significant bits: at most 2^-7 relative error whether the cast rounds or truncates.
    rel = ((got - ref).abs() / ref.clamp_min(1.0)).max().item()
    assert rel <= 2.0**-7, f"uint32 -> bf16 max relative error {rel:.3g}"
