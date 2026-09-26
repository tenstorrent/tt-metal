# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

# Regression test for https://github.com/tenstorrent/tt-metal/issues/52036
# The SFPU xlogy guard used `in1 == nan`, which is never true in IEEE-754,
# so xlogy(x, NaN) fell through to the log path and returned ~89 instead of NaN.
import torch
import ttnn
import pytest
from tests.ttnn.utils_for_testing import assert_with_pcc

pytestmark = pytest.mark.use_module_device

@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32])
def test_xlogy_positive_nan(device, dtype):
    # Positive NaN payload (sign bit clear) — the case the old guard missed.
    torch.manual_seed(0)
    shape = torch.Size([1, 1, 32, 32])
    a = torch.rand(shape, dtype=torch.float32) * 2.0 + 0.5
    b = torch.full(shape, float("nan"))
    golden = torch.xlogy(a, b)
    ta = ttnn.from_torch(a, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    tb = ttnn.from_torch(b, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    out = ttnn.to_torch(ttnn.xlogy(ta, tb))
    # NaN masks must match exactly (NaN == NaN is False, so compare isnan).
    assert torch.equal(torch.isnan(out), torch.isnan(golden))
    assert torch.isnan(out).all(), "xlogy(x, +NaN) should be NaN everywhere"

@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32])
def test_xlogy_negative_y_is_nan(device, dtype):
    torch.manual_seed(1)
    shape = torch.Size([1, 1, 32, 32])
    a = torch.rand(shape, dtype=torch.float32) * 2.0 + 0.5
    b = torch.full(shape, -1.0)
    golden = torch.xlogy(a, b)
    ta = ttnn.from_torch(a, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    tb = ttnn.from_torch(b, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    out = ttnn.to_torch(ttnn.xlogy(ta, tb))
    assert torch.equal(torch.isnan(out), torch.isnan(golden))
    assert torch.isnan(out).all(), "xlogy(x, y<0) should be NaN everywhere"

@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32])
def test_xlogy_inf_unchanged(device, dtype):
    # Regression guard: the fix keeps ±inf on the log path (inf has mantissa 0,
    # so a naive `exexp == 128`-only check must NOT return NaN for it).
    torch.manual_seed(3)
    shape = torch.Size([1, 1, 32, 32])
    a = torch.rand(shape, dtype=torch.float32) + 0.5  # all positive
    b = torch.full(shape, float("inf"))
    golden = torch.xlogy(a, b)
    ta = ttnn.from_torch(a, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    tb = ttnn.from_torch(b, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    out = ttnn.to_torch(ttnn.xlogy(ta, tb))
    assert torch.isinf(out).all(), "xlogy(x, +inf) should be +inf, not NaN"
    assert torch.equal(torch.isinf(out), torch.isinf(golden))

@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32])
def test_xlogy_normal_range_unchanged(device, dtype):
    torch.manual_seed(2)
    shape = torch.Size([1, 1, 32, 32])
    a = torch.rand(shape, dtype=torch.float32) * 2.0 + 0.5
    b = torch.rand(shape, dtype=torch.float32) * 5.0 + 1.0
    golden = torch.xlogy(a, b)
    ta = ttnn.from_torch(a, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    tb = ttnn.from_torch(b, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    out = ttnn.to_torch(ttnn.xlogy(ta, tb))
    assert_with_pcc(golden, out, 0.99)
