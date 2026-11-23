# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Host-side tests for the MX (OCP microscaling) TTNN data types.

No device is needed: MX tensors are packed and unpacked on the host. Device tests (Quasar only) live in
tests/ttnn/nightly/unit_tests/operations/experimental/quasar/test_binary_ng_mx.py.
"""

import pytest
import torch

import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc

# Minimum PCC between the float input and its MX round trip, for torch.randn data.
_ROUND_TRIP_PCC = {
    ttnn.mxfp8_e4m3: 0.999,
    ttnn.mxfp8_e5m2: 0.995,
    ttnn.mxfp6_e2m3: 0.995,
    ttnn.mxfp6_e3m2: 0.99,
    ttnn.mxfp4: 0.95,
    ttnn.mxint8: 0.999,
    ttnn.mxint4: 0.97,
    ttnn.mxint2: 0.6,
}


def _to_blocks(x):
    """[H, W] (multiples of 32) -> [num_blocks, 32] in the order the MX packer groups elements.

    A 32x32 tile holds four 16x16 faces, and each MX block is 32 consecutive elements in face order,
    i.e. two consecutive 16-wide face rows.
    """
    H, W = x.shape
    # dims: tile_row, face_row, block, row_in_block, tile_col, face_col, col_in_face
    x = x.reshape(H // 32, 2, 8, 2, W // 32, 2, 16)
    return x.permute(0, 4, 1, 5, 2, 3, 6).reshape(-1, 32)


def _from_blocks(blocks, H, W):
    x = blocks.reshape(H // 32, W // 32, 2, 2, 8, 2, 16)
    return x.permute(0, 2, 4, 5, 1, 3, 6).reshape(H, W)


def _mx_quantize_reference(x, emax_elem, quantize_element):
    """OCP MX v1.0 quantization: shared scale 2^(floor(log2(amax)) - emax_elem), floored at 2^-127."""
    H, W = x.shape
    blocks = _to_blocks(x.float())
    amax = blocks.abs().amax(dim=1, keepdim=True)
    _, exponent = torch.frexp(amax)  # amax = m * 2^exponent with m in [0.5, 1)
    shared_exp = torch.clamp(exponent - 1 - emax_elem, min=-127)
    scale = torch.pow(2.0, shared_exp.float())
    quantized = quantize_element(blocks / scale) * scale
    return _from_blocks(quantized, H, W)


def _quantize_e4m3(values):
    # E4M3FN has no Inf, so out-of-range values saturate to +-448 (max normal).
    return torch.clamp(values, -448.0, 448.0).to(torch.float8_e4m3fn).to(torch.float32)


def _quantize_int8(values):
    # S1.6 two's complement with the -128 code left unused: k / 64 for k in [-127, 127].
    return torch.clamp(torch.round(values * 64.0), -127, 127) / 64.0


_REFERENCES = {
    ttnn.mxfp8_e4m3: (8, _quantize_e4m3),
    ttnn.mxint8: (0, _quantize_int8),
}


def _round_trip(x, dtype):
    tt_tensor = ttnn.from_torch(x, dtype=dtype, layout=ttnn.TILE_LAYOUT)
    assert tt_tensor.dtype == dtype
    assert tt_tensor.layout == ttnn.TILE_LAYOUT
    return ttnn.to_torch(tt_tensor)


@pytest.mark.parametrize("dtype", ttnn.MX_DTYPES)
@pytest.mark.parametrize("shape", [(32, 32), (64, 128), (2, 3, 64, 96), (40, 50)])
def test_mx_round_trip(dtype, shape):
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=torch.float32)

    y = _round_trip(x, dtype)

    assert y.shape == x.shape
    assert y.dtype == torch.float32
    assert_with_pcc(x, y, _ROUND_TRIP_PCC[dtype])
    # Values that are already MX-quantized must survive another round trip unchanged.
    assert torch.equal(_round_trip(y, dtype), y)


@pytest.mark.parametrize("dtype", ttnn.MX_DTYPES)
def test_mx_from_bfloat16(dtype):
    torch.manual_seed(0)
    x = torch.randn((64, 64), dtype=torch.bfloat16)

    y = _round_trip(x, dtype)

    # Packing from a bfloat16 buffer must give exactly what packing the same values as float32 gives.
    assert torch.equal(y, _round_trip(x.float(), dtype))


@pytest.mark.parametrize("dtype", list(_REFERENCES))
@pytest.mark.parametrize("scale", [1.0, 1e-3, 1e4])
def test_mx_matches_ocp_reference(dtype, scale):
    torch.manual_seed(0)
    x = torch.randn((64, 96), dtype=torch.float32) * scale
    # Give some blocks a wide dynamic range so element rounding and saturation paths are exercised.
    x[::7, ::5] *= 100.0

    emax_elem, quantize_element = _REFERENCES[dtype]
    expected = _mx_quantize_reference(x, emax_elem, quantize_element)

    assert torch.equal(_round_trip(x, dtype), expected)


@pytest.mark.parametrize("dtype", ttnn.MX_DTYPES)
def test_mx_zeros_stay_zero(dtype):
    x = torch.zeros((32, 64), dtype=torch.float32)
    x[0, 0] = 1.0  # one non-zero block next to an all-zero block

    y = _round_trip(x, dtype)

    assert y[0, 0] == 1.0
    assert torch.count_nonzero(y) == 1


@pytest.mark.parametrize("dtype", ttnn.MX_DTYPES)
def test_mx_requires_tile_layout(expect_error, dtype):
    x = torch.randn((32, 32), dtype=torch.float32)
    with expect_error(RuntimeError, "Layout must be Layout::TILE"):
        ttnn.from_torch(x, dtype=dtype, layout=ttnn.ROW_MAJOR_LAYOUT)


@pytest.mark.parametrize("dtype", ttnn.MX_DTYPES)
def test_mx_defaults_to_tile_layout(dtype):
    x = torch.randn((32, 32), dtype=torch.float32)
    assert ttnn.from_torch(x, dtype=dtype).layout == ttnn.TILE_LAYOUT


@pytest.mark.parametrize("dtype", [ttnn.mxfp8_e4m3, ttnn.mxint8, ttnn.mxfp4])
@pytest.mark.parametrize("other", [ttnn.float32, ttnn.bfloat16, ttnn.bfloat8_b, ttnn.mxint4])
def test_mx_to_dtype(dtype, other):
    torch.manual_seed(0)
    x = torch.randn((64, 64), dtype=torch.float32)
    mx_tensor = ttnn.from_torch(x, dtype=dtype, layout=ttnn.TILE_LAYOUT)
    mx_values = ttnn.to_torch(mx_tensor)

    # MX -> other: same as converting the MX-decoded values to `other`.
    converted = ttnn.to_dtype(mx_tensor, other)
    assert converted.dtype == other
    expected = ttnn.to_torch(ttnn.from_torch(mx_values, dtype=other, layout=ttnn.TILE_LAYOUT))
    assert torch.equal(ttnn.to_torch(converted).float(), expected.float())

    # other -> MX: same as packing the decoded `other` values straight to MX.
    other_tensor = ttnn.from_torch(x, dtype=other, layout=ttnn.TILE_LAYOUT)
    back = ttnn.to_dtype(other_tensor, dtype)
    assert back.dtype == dtype
    assert torch.equal(ttnn.to_torch(back), _round_trip(ttnn.to_torch(other_tensor).float(), dtype))


@pytest.mark.parametrize("dtype", ttnn.MX_DTYPES)
def test_mx_serialization(tmp_path, dtype):
    torch.manual_seed(0)
    x = torch.randn((64, 96), dtype=torch.float32)
    tt_tensor = ttnn.from_torch(x, dtype=dtype, layout=ttnn.TILE_LAYOUT)

    path = tmp_path / "mx_tensor.tensorbin"
    ttnn.dump_tensor(path, tt_tensor)
    loaded = ttnn.load_tensor(path)

    assert loaded.dtype == dtype
    assert loaded.layout == ttnn.TILE_LAYOUT
    assert torch.equal(ttnn.to_torch(loaded), ttnn.to_torch(tt_tensor))


@pytest.mark.parametrize("dtype", [ttnn.mxfp8_e4m3, ttnn.mxint8])
def test_mx_print_and_item(dtype):
    x = torch.full((32, 32), 0.5, dtype=torch.float32)
    tt_tensor = ttnn.from_torch(x, dtype=dtype, layout=ttnn.TILE_LAYOUT)

    assert "0.5000" in str(tt_tensor)
    assert ttnn.from_torch(x[:1, :1], dtype=dtype, layout=ttnn.TILE_LAYOUT).item() == 0.5
    assert tt_tensor.to_list()[3][7] == 0.5


@pytest.mark.parametrize("dtype", [ttnn.mxfp8_e4m3, ttnn.mxint8])
def test_mx_full(dtype):
    tt_tensor = ttnn.full([64, 64], 0.75, dtype=dtype)
    assert tt_tensor.dtype == dtype
    assert tt_tensor.layout == ttnn.TILE_LAYOUT
    assert torch.equal(ttnn.to_torch(tt_tensor), torch.full((64, 64), 0.75))
