# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Tests for the BFLOAT2_B (2-bit block float) TTNN data type.

BFLOAT2_B stores a sign and one magnitude bit per element, with an 8-bit exponent shared by each 16-element face row,
so it holds {-1, 0, +1} times a power of two exactly. The host tests need no device. The device tests need a Wormhole
or Blackhole device; Quasar has no BFP formats.
"""

import pytest
import torch

import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc


def _ternary(shape, scale=1.0, seed=0):
    torch.manual_seed(seed)
    return torch.randint(-1, 2, shape, dtype=torch.float32) * scale


def _round_trip(x):
    tt_tensor = ttnn.from_torch(x, dtype=ttnn.bfloat2_b, layout=ttnn.TILE_LAYOUT)
    assert tt_tensor.dtype == ttnn.bfloat2_b
    assert tt_tensor.layout == ttnn.TILE_LAYOUT
    return ttnn.to_torch(tt_tensor)


def test_bfp2_alias():
    assert ttnn.bfloat2_b == ttnn.DataType.BFLOAT2_B


@pytest.mark.parametrize("shape", [(32, 32), (64, 128), (2, 3, 64, 96), (14, 47)])
@pytest.mark.parametrize("scale", [1.0, 0.125, 8.0])
def test_bfp2_ternary_round_trip_is_exact(shape, scale):
    # Every face row of a scaled ternary tensor shares one exponent, so packing loses nothing.
    x = _ternary(shape, scale)

    y = _round_trip(x)

    assert y.shape == x.shape
    assert y.dtype == torch.float32
    assert torch.equal(y, x)


def test_bfp2_from_bfloat16():
    x = _ternary((64, 64))
    assert torch.equal(_round_trip(x.to(torch.bfloat16)), x)


def test_bfp2_quantizes_general_values():
    torch.manual_seed(0)
    x = torch.randn((64, 64), dtype=torch.float32)

    y = _round_trip(x)

    # Values that are already BFLOAT2_B-quantized must survive another round trip unchanged.
    assert torch.equal(_round_trip(y), y)
    nonzero = y != 0
    assert torch.equal(torch.sign(y[nonzero]), torch.sign(x[nonzero]))


def test_bfp2_requires_tile_layout(expect_error):
    x = _ternary((32, 32))
    with expect_error(RuntimeError, "Layout must be Layout::TILE"):
        ttnn.from_torch(x, dtype=ttnn.bfloat2_b, layout=ttnn.ROW_MAJOR_LAYOUT)


def test_bfp2_defaults_to_tile_layout():
    assert ttnn.from_torch(_ternary((32, 32)), dtype=ttnn.bfloat2_b).layout == ttnn.TILE_LAYOUT


@pytest.mark.parametrize("other", [ttnn.float32, ttnn.bfloat16, ttnn.bfloat8_b, ttnn.bfloat4_b])
def test_bfp2_to_dtype(other):
    # Ternary values are exact in every one of these formats, so conversions in both directions are lossless.
    x = _ternary((64, 64))
    bfp2_tensor = ttnn.from_torch(x, dtype=ttnn.bfloat2_b, layout=ttnn.TILE_LAYOUT)

    converted = ttnn.to_dtype(bfp2_tensor, other)
    assert converted.dtype == other
    assert torch.equal(ttnn.to_torch(converted).float(), x)

    back = ttnn.to_dtype(ttnn.from_torch(x, dtype=other, layout=ttnn.TILE_LAYOUT), ttnn.bfloat2_b)
    assert back.dtype == ttnn.bfloat2_b
    assert torch.equal(ttnn.to_torch(back), x)


def test_bfp2_to_dtype_from_row_major_switches_to_tile():
    x = _ternary((64, 64))
    row_major = ttnn.from_torch(x)
    assert row_major.layout == ttnn.ROW_MAJOR_LAYOUT

    converted = ttnn.to_dtype(row_major, ttnn.bfloat2_b)

    assert converted.layout == ttnn.TILE_LAYOUT
    assert torch.equal(ttnn.to_torch(converted), x)


def test_bfp2_serialization(tmp_path):
    x = _ternary((64, 96))
    tt_tensor = ttnn.from_torch(x, dtype=ttnn.bfloat2_b, layout=ttnn.TILE_LAYOUT)

    path = tmp_path / "bfp2_tensor.tensorbin"
    ttnn.dump_tensor(path, tt_tensor)
    loaded = ttnn.load_tensor(path)

    assert loaded.dtype == ttnn.bfloat2_b
    assert loaded.layout == ttnn.TILE_LAYOUT
    assert torch.equal(ttnn.to_torch(loaded), x)


def test_bfp2_print_item_and_to_list():
    x = torch.full((32, 32), 0.5, dtype=torch.float32)
    tt_tensor = ttnn.from_torch(x, dtype=ttnn.bfloat2_b, layout=ttnn.TILE_LAYOUT)

    text = str(tt_tensor)
    assert "0.5000" in text
    assert "bfloat4_b" not in text.lower()
    assert ttnn.from_torch(x[:1, :1], dtype=ttnn.bfloat2_b, layout=ttnn.TILE_LAYOUT).item() == 0.5
    assert tt_tensor.to_list()[3][7] == 0.5


def test_bfp2_full():
    tt_tensor = ttnn.full([64, 64], 0.5, dtype=ttnn.bfloat2_b)

    assert tt_tensor.dtype == ttnn.bfloat2_b
    assert tt_tensor.layout == ttnn.TILE_LAYOUT
    assert torch.equal(ttnn.to_torch(tt_tensor), torch.full((64, 64), 0.5))


def test_bfp2_device_round_trip(device):
    x = _ternary((64, 128))

    tt_tensor = ttnn.from_torch(x, dtype=ttnn.bfloat2_b, layout=ttnn.TILE_LAYOUT, device=device)

    assert tt_tensor.dtype == ttnn.bfloat2_b
    assert torch.equal(ttnn.to_torch(tt_tensor), x)


def test_bfp2_matmul_matches_bfp4(device):
    # Ternary weights are exact in both formats, so a matmul with BFLOAT2_B weights must match the same matmul with
    # BFLOAT4_B weights bit for bit.
    torch.manual_seed(0)
    x = torch.randn((32, 256), dtype=torch.bfloat16)
    w = _ternary((256, 128))
    tt_x = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    outputs = {}
    for weight_dtype in (ttnn.bfloat2_b, ttnn.bfloat4_b):
        tt_w = ttnn.from_torch(w, dtype=weight_dtype, layout=ttnn.TILE_LAYOUT, device=device)
        outputs[weight_dtype] = ttnn.to_torch(ttnn.matmul(tt_x, tt_w))

    assert torch.equal(outputs[ttnn.bfloat2_b], outputs[ttnn.bfloat4_b])
    assert_with_pcc(x.float() @ w, outputs[ttnn.bfloat2_b].float(), 0.999)
