# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn

from tests.ttnn.utils_for_testing import assert_with_ulp

pytestmark = pytest.mark.use_module_device

H, W = 64, 256

_OPS = [
    pytest.param(ttnn.add, {}, False, id="add"),
    pytest.param(ttnn.subtract, {}, False, id="subtract"),
    pytest.param(ttnn.rsub, {}, False, id="rsub"),
    pytest.param(ttnn.squared_difference, {}, False, id="squared_difference"),
    pytest.param(ttnn.multiply, {"fast_and_approximate_mode": True}, False, id="multiply_fast"),
    pytest.param(ttnn.divide, {"fast_and_approximate_mode": True}, False, id="divide_fast"),
    pytest.param(ttnn.eq, {}, True, id="eq"),
    pytest.param(ttnn.gt, {}, True, id="gt"),
]
_NARROW = [
    pytest.param(ttnn.bfloat16, "none", id="bf16-none"),
    pytest.param(ttnn.bfloat16, "row", id="bf16-row"),
    pytest.param(ttnn.bfloat16, "col", id="bf16-col"),
    pytest.param(ttnn.bfloat16, "scalar", id="bf16-scalar"),
    pytest.param(ttnn.bfloat8_b, "none", id="bfp8-none"),
    pytest.param(ttnn.bfloat8_b, "row", id="bfp8-row"),
]
_BCAST = {"none": (H, W), "row": (1, W), "col": (H, 1), "scalar": (1, 1)}


@pytest.mark.parametrize("float32_first", [True, False], ids=["fp32-first", "narrow-first"])
@pytest.mark.parametrize("narrow_dtype, bcast", _NARROW)
@pytest.mark.parametrize("op, kwargs, is_comparison", _OPS)
def test_binary_ng_float32_with_narrow_float(device, op, kwargs, is_comparison, narrow_dtype, bcast, float32_first):
    torch.manual_seed(0)
    narrow = (torch.randn(H, W) * 4).bfloat16()[: _BCAST[bcast][0], : _BCAST[bcast][1]]
    if is_comparison:
        sign = torch.where(torch.rand(H, W) < 0.5, 1.0, -1.0)
        wide = (torch.randn(H, W) * 4).bfloat16().float()
        wide[: narrow.shape[0], : narrow.shape[1]] = narrow.float()
        wide = wide * (1 + sign * 2.0**-14)
    else:
        wide = torch.randn(H, W) * 4

    tt_wide = ttnn.from_torch(wide, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    tt_narrow = ttnn.from_torch(narrow.float(), dtype=narrow_dtype, layout=ttnn.TILE_LAYOUT, device=device)
    tt_narrow_as_fp32 = ttnn.typecast(tt_narrow, ttnn.float32)

    for out_dtype in (ttnn.float32, ttnn.bfloat16):
        if float32_first:
            mixed, reference = (tt_wide, tt_narrow), (tt_wide, tt_narrow_as_fp32)
        else:
            mixed, reference = (tt_narrow, tt_wide), (tt_narrow_as_fp32, tt_wide)
        out = ttnn.to_torch(op(*mixed, dtype=out_dtype, **kwargs))
        ref = ttnn.to_torch(op(*reference, dtype=out_dtype, **kwargs))
        if out_dtype == ttnn.float32 or is_comparison:
            assert torch.equal(out, ref), f"{(out != ref).sum().item()} of {out.numel()} differ ({out_dtype})"
        else:
            finite = torch.isfinite(ref)
            assert torch.equal(out[~finite].nan_to_num(), ref[~finite].nan_to_num())
            assert_with_ulp(expected_result=ref[finite], actual_result=out[finite], ulp_threshold=1)
