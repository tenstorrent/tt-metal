# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import ttnn

# sinh_bw and cosh_bw used to guard on the input domain at |input| > 88.5, which is
# log(FLT_MAX) -- where exp saturates. Both gradients are hyperbolic functions, which
# are (e^x +/- e^-x)/2 and do not overflow until log(2*FLT_MAX) = 89.4159862, so the
# guard fired a full 0.916 early and returned +/-inf for arguments the forward op
# still computes finitely. With grad = 0 it returned sign(0) * inf = NaN, where torch
# returns 0.
#
# Below 88.5 and above 89.4159862 nothing changes: the first is untouched by the
# guard and the second genuinely overflows, so inf is the right answer there. Those
# are the two controls.
#
# bfloat16 note: at this magnitude bfloat16 steps by 0.5, so 88.6 rounds to 88.5 --
# not past the old bound -- and 89.4 rounds to 89.5, which overflows for real. In
# that dtype only +/-89.0 lands inside the window, and the other points act as extra
# controls. They are kept rather than skipped because a test that silently ran fewer
# cases in one dtype would be worse than one that says why.
INPUTS = (88.0, 88.6, 89.0, 89.4, -89.0, 100.0)
INPUT_IDS = ["88.0_below", "88.6", "89.0", "89.4", "-89.0", "100.0_overflows"]

# grad = 0 is not padding: it is the case the guard turned into NaN.
GRADS = (1.0, 0.0)
GRAD_IDS = ["grad_1", "grad_0"]

DTYPES = ((torch.float32, ttnn.float32), (torch.bfloat16, ttnn.bfloat16))
DTYPE_IDS = ("float32", "bfloat16")

SHAPE = torch.Size([1, 1, 32, 32])


def _to_device(pt, ttnn_dtype, device):
    return ttnn.from_torch(pt.detach(), dtype=ttnn_dtype, layout=ttnn.TILE_LAYOUT, device=device)


@pytest.mark.parametrize("torch_dtype, ttnn_dtype", DTYPES, ids=DTYPE_IDS)
@pytest.mark.parametrize("grad_value", GRADS, ids=GRAD_IDS)
@pytest.mark.parametrize("input_value", INPUTS, ids=INPUT_IDS)
@pytest.mark.parametrize("ttnn_op", (ttnn.sinh_bw, ttnn.cosh_bw), ids=("sinh_bw", "cosh_bw"))
def test_bw_hyperbolic_upper_range(input_value, grad_value, torch_dtype, ttnn_dtype, ttnn_op, device):
    in_data = torch.full(SHAPE, input_value, dtype=torch_dtype).requires_grad_(True)
    grad_data = torch.full(SHAPE, grad_value, dtype=torch_dtype)

    tt_out = ttnn_op(_to_device(grad_data, ttnn_dtype, device), _to_device(in_data, ttnn_dtype, device))
    golden = ttnn.get_golden_function(ttnn_op)(grad_data, in_data)[0]

    got = ttnn.to_torch(tt_out[0]).float()
    want = golden.float()

    finite_ref = torch.isfinite(want)
    lost = int((finite_ref & ~torch.isfinite(got)).sum())
    assert lost == 0, (
        f"{ttnn_op.__name__}(grad={grad_value:g}, input={input_value:g}) [{torch_dtype}]: "
        f"{lost} of {got.numel()} gradients came back non-finite where the reference is "
        f"{float(want.flatten()[0]):g}"
    )
    torch.testing.assert_close(got, want, rtol=2e-2, atol=0.0, equal_nan=True)
