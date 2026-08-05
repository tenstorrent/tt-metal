# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

# Regression test for https://github.com/tenstorrent/tt-metal/issues/52038
# tril/triu were composed as multiply(input, mask): in IEEE-754, 0 * inf = NaN,
# so a masked-out infinity/NaN came out as NaN instead of 0 (fp32 path).
import torch
import ttnn
import pytest
from tests.ttnn.utils_for_testing import assert_with_pcc

pytestmark = pytest.mark.use_module_device

@pytest.mark.parametrize("fn", [ttnn.tril, ttnn.triu])
@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32])
def test_tril_triu_inf_masked_is_zero(device, fn, dtype):
    torch.manual_seed(0)
    shape = torch.Size([1, 1, 32, 32])
    # -inf in the masked region: 0 * -inf = NaN under the old multiply-based impl
    a = torch.full(shape, float("-inf"))
    golden = fn(a).to(dtype)  # cast to the device dtype so torch.equal compares like-for-like
    ta = ttnn.from_torch(a, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    out = ttnn.to_torch(fn(ta))
    # Masked region must be exactly 0 (never NaN), selected region keeps -inf
    assert torch.equal(out, golden), "tril/triu with inf must match torch exactly"
    assert not torch.isnan(out).any(), "masked-out inf became NaN - fix regression"
    assert (out == float("-inf")).any(), "selected region lost the -inf"

@pytest.mark.parametrize("fn", [ttnn.tril, ttnn.triu])
@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32])
def test_tril_triu_normal_unchanged(device, fn, dtype):
    torch.manual_seed(1)
    shape = torch.Size([1, 1, 32, 32])
    a = torch.rand(shape, dtype=torch.float32) * 2.0 - 1.0
    golden = fn(a)
    ta = ttnn.from_torch(a, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    out = ttnn.to_torch(fn(ta))
    assert_with_pcc(golden, out, 0.999)
