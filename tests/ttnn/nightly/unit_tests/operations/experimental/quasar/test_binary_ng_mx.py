# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""
Eltwise add with MX (OCP microscaling) tensors through ttnn.experimental.quasar.add.

The unpacker expands MX to Float16_b and the packer converts back, so the FPU never sees MX. The goldens
therefore quantize on the host with the same MX encoder that from_torch uses.

Run on the Quasar simulator:
    TT_METAL_SIMULATOR=<path>/libttsim.so TT_SIMULATOR_LOCALHOST=1 ARCH_NAME=quasar CHIP_ARCH=quasar \
        TT_METAL_SLOW_DISPATCH_MODE=1 \
        pytest tests/ttnn/nightly/unit_tests/operations/experimental/quasar/test_binary_ng_mx.py

On Wormhole/Blackhole only test_mx_rejected_off_quasar runs.
"""

import pytest
import torch

import ttnn
from tests.ttnn.nightly.unit_tests.operations.experimental.quasar.binary_ng_quasar_test_utils import _on_quasar
from tests.ttnn.utils_for_testing import assert_with_pcc

quasar_only = pytest.mark.skipif(not _on_quasar(), reason="MX formats are Quasar only")

# One interleaved shape small enough for the simulator, and one that spreads tiles over several cores.
_SHAPES = [(32, 32), (4 * 32, 8 * 32)]


def _mx_round_trip(x, dtype):
    """Host MX quantization: the values an MX tensor built from `x` holds."""
    return ttnn.to_torch(ttnn.from_torch(x, dtype=dtype, layout=ttnn.TILE_LAYOUT))


def _assert_matches(expected, actual, min_exact_fraction=0.99):
    # The device adds in Float16_b before packing, while the golden adds in float32, so a value that lands
    # on a rounding boundary can differ by one MX step. Everything else must match exactly.
    assert actual.shape == expected.shape
    exact_fraction = (actual == expected).float().mean().item()
    assert exact_fraction >= min_exact_fraction, f"only {exact_fraction:.4f} of the elements match exactly"
    assert_with_pcc(expected, actual, 0.999)


@quasar_only
@pytest.mark.parametrize("dtype", ttnn.MX_DTYPES)
@pytest.mark.parametrize("shape", _SHAPES)
def test_add_mx_in_mx_out(device, dtype, shape):
    torch.manual_seed(0)
    a = torch.randn(shape, dtype=torch.float32)
    b = torch.randn(shape, dtype=torch.float32)

    ta = ttnn.from_torch(a, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    tb = ttnn.from_torch(b, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    tc = ttnn.experimental.quasar.add(ta, tb)

    assert tc.dtype == dtype
    golden = _mx_round_trip(_mx_round_trip(a, dtype) + _mx_round_trip(b, dtype), dtype)
    _assert_matches(golden, ttnn.to_torch(tc))


@quasar_only
@pytest.mark.parametrize("dtype", ttnn.MX_DTYPES)
@pytest.mark.parametrize("shape", _SHAPES)
def test_add_bf16_in_mx_out(device, dtype, shape):
    # MX produced on device by the packer must decode on the host like host-packed MX.
    torch.manual_seed(0)
    a = torch.randn(shape, dtype=torch.bfloat16)
    b = torch.randn(shape, dtype=torch.bfloat16)

    ta = ttnn.from_torch(a, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tb = ttnn.from_torch(b, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tc = ttnn.experimental.quasar.add(ta, tb, dtype=dtype)

    assert tc.dtype == dtype
    golden = _mx_round_trip((a + b).float(), dtype)
    _assert_matches(golden, ttnn.to_torch(tc))


@quasar_only
@pytest.mark.parametrize("dtype", ttnn.MX_DTYPES)
@pytest.mark.parametrize("shape", _SHAPES)
def test_add_mx_in_bf16_out(device, dtype, shape):
    torch.manual_seed(0)
    a = torch.randn(shape, dtype=torch.float32)
    b = torch.randn(shape, dtype=torch.float32)

    ta = ttnn.from_torch(a, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    tb = ttnn.from_torch(b, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    tc = ttnn.experimental.quasar.add(ta, tb, dtype=ttnn.bfloat16)

    assert tc.dtype == ttnn.bfloat16
    golden = (_mx_round_trip(a, dtype) + _mx_round_trip(b, dtype)).bfloat16()
    _assert_matches(golden, ttnn.to_torch(tc))


@pytest.mark.skipif(_on_quasar(), reason="checks the non-Quasar guard")
@pytest.mark.parametrize("dtype", [ttnn.mxfp8_e4m3, ttnn.mxint8])
def test_mx_rejected_off_quasar(device, expect_error, dtype):
    x = torch.randn((32, 32), dtype=torch.float32)
    with expect_error(RuntimeError, "is not supported on arch"):
        ttnn.from_torch(x, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
