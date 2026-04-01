# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
import pytest

from tests.ttnn.utils_for_testing import assert_with_ulp


@pytest.mark.parametrize(
    "c_shape, t_shape, f_shape",
    [
        ((8, 4, 1), (8, 4, 768), (1, 1, 1)),  # Ccol, Tfull, Fscalar
        ((8, 1, 768), (8, 4, 768), (1, 1, 1)),  # Crow, Tfull, Fscalar
        ((8, 4, 768), (8, 4, 1), (8, 1, 768)),  # Cfull, Tcol, Frow
        ((8, 4, 768), (8, 1, 768), (8, 4, 1)),  # Cfull, Trow, Fcol
        ((8, 4, 1), (8, 1, 768), (8, 4, 768)),  # Ccol, Trow, Ffull
        ((8, 1, 768), (8, 4, 1), (8, 4, 768)),  # Crow, Tcol, Ffull
        ((8, 1, 768), (8, 4, 768), (8, 4, 1)),  # Crow, Tfull, Fcol
        ((8, 4, 1), (8, 4, 768), (8, 1, 768)),  # Ccol, Tfull, Frow
        ((1, 1, 1), (8, 4, 1), (8, 1, 768)),  # Cscalar, Tcol, Frow
        ((1, 1, 1), (8, 1, 768), (8, 4, 1)),  # Cscalar, Trow, Fcol
        ((8, 1, 768), (1, 1, 1), (8, 4, 1)),  # Crow, Tscalar, Fcol
        ((8, 4, 1), (1, 1, 1), (8, 1, 768)),  # Ccol, Tscalar, Frow
    ],
)
def test_ttnn_where_row_col_mixed_bcast(c_shape, t_shape, f_shape, device):
    torch.manual_seed(0)
    C = torch.randint(0, 2, c_shape).to(torch.bfloat16)
    T = torch.randn(t_shape, dtype=torch.bfloat16)
    F = torch.ones(f_shape, dtype=torch.bfloat16) * 10
    golden = torch.where(C.bool(), T, F)

    ttnn_C = ttnn.from_torch(C, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    ttnn_T = ttnn.from_torch(T, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    ttnn_F = ttnn.from_torch(F, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    ttnn_result = ttnn.where(ttnn_C, ttnn_T, ttnn_F)
    result = ttnn.to_torch(ttnn_result)

    assert torch.equal(result, golden)


@pytest.mark.parametrize(
    "c_shape, t_shape",
    [
        ((8, 4, 1), (8, 1, 768)),  # Ccol, Trow
        ((8, 1, 768), (8, 4, 1)),  # Crow, Tcol
    ],
)
def test_ttnn_where_row_col_mixed_bcast_tts(c_shape, t_shape, device):
    torch.manual_seed(0)
    C = torch.randint(0, 2, c_shape).to(torch.bfloat16)
    T = torch.randn(t_shape, dtype=torch.bfloat16)
    F = 10.0
    golden = torch.where(C.bool(), T, F)

    ttnn_C = ttnn.from_torch(C, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    ttnn_T = ttnn.from_torch(T, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    ttnn_result = ttnn.where(ttnn_C, ttnn_T, F)
    result = ttnn.to_torch(ttnn_result)

    assert torch.equal(result, golden)


@pytest.mark.parametrize(
    "c_shape, f_shape",
    [
        ((8, 4, 1), (8, 1, 768)),  # Ccol, Frow
        ((8, 1, 768), (8, 4, 1)),  # Crow, Fcol
    ],
)
def test_ttnn_where_row_col_mixed_bcast_tst(c_shape, f_shape, device):
    torch.manual_seed(0)
    C = torch.randint(0, 2, c_shape).to(torch.bfloat16)
    T = 10.0
    F = torch.randn(f_shape, dtype=torch.bfloat16)
    golden = torch.where(C.bool(), T, F)

    ttnn_C = ttnn.from_torch(C, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    ttnn_F = ttnn.from_torch(F, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    ttnn_result = ttnn.where(ttnn_C, T, ttnn_F)
    result = ttnn.to_torch(ttnn_result)

    assert torch.equal(result, golden)


@pytest.mark.parametrize(
    "a_shape, b_shape, c_shape",
    [
        ((8, 4, 768), (8, 4, 768), (8, 4, 768)),  # full, full, full
        ((8, 4, 1), (8, 4, 768), (8, 4, 768)),  # acol, bfull, cfull
        ((8, 1, 768), (8, 4, 768), (8, 4, 768)),  # arow, bfull, cfull
        ((8, 4, 768), (8, 4, 1), (8, 1, 768)),  # afull, bcol, crow
        ((8, 4, 768), (8, 1, 768), (8, 4, 1)),  # afull, brow, ccol
        ((8, 4, 1), (8, 1, 768), (8, 4, 768)),  # acol, brow, cfull
        ((8, 1, 768), (8, 4, 1), (8, 4, 768)),  # arow, bcol, cfull
        ((8, 1, 768), (8, 4, 768), (8, 4, 1)),  # arow, bfull, ccol
        ((8, 4, 1), (8, 4, 768), (8, 1, 768)),  # acol, bfull, crow
        ((1, 1, 1), (8, 4, 1), (8, 1, 768)),  # ascalar, bcol, crow
        ((1, 1, 1), (8, 1, 768), (8, 4, 1)),  # ascalar, brow, ccol
        ((8, 1, 768), (1, 1, 1), (8, 4, 1)),  # arow, bscalar, ccol
        ((8, 4, 1), (1, 1, 1), (8, 1, 768)),  # acol, bscalar, crow
    ],
)
@pytest.mark.parametrize("value", [1.5, 0.5, -0.25])
@pytest.mark.parametrize("ttnn_op", [ttnn.addcmul, ttnn.addcdiv])
def test_ttnn_addc_ops_row_col_mixed_bcast(a_shape, b_shape, c_shape, value, ttnn_op, device):
    torch.manual_seed(0)
    in_data1 = torch.empty(a_shape, dtype=torch.bfloat16).uniform_(-100, 100)
    in_data2 = torch.empty(b_shape, dtype=torch.bfloat16).uniform_(-100, 100)
    in_data3 = torch.empty(c_shape, dtype=torch.bfloat16).uniform_(-100, 100)

    golden_fn = ttnn.get_golden_function(ttnn_op)
    golden = golden_fn(in_data1, in_data2, in_data3, value=value)

    input_tensor1 = ttnn.from_torch(in_data1, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    input_tensor2 = ttnn.from_torch(in_data2, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    input_tensor3 = ttnn.from_torch(in_data3, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    ttnn_result = ttnn_op(input_tensor1, input_tensor2, input_tensor3, value=value)
    result = ttnn.to_torch(ttnn_result)

    # For addcdiv: if denominator (in_data3) is zero, golden is nan but ttnn may return inf; normalize to nan
    if ttnn_op is ttnn.addcdiv:
        zero_mask = in_data3 == 0
        golden_nan_mask = torch.isnan(golden)
        result_inf_mask = torch.isinf(result)
        normalize_mask = zero_mask & golden_nan_mask & result_inf_mask
        if normalize_mask.any():
            result = torch.where(
                normalize_mask,
                torch.tensor(float("nan"), dtype=result.dtype, device=result.device),
                result,
            )

    assert_with_ulp(golden, result, ulp_threshold=10, allow_nonfinite=True)


@pytest.mark.parametrize(
    "c_shape, t_shape",
    [
        ((8, 4, 1), (8, 1, 768)),  # Ccol, Trow
        ((8, 1, 768), (8, 4, 1)),  # Crow, Tcol
    ],
)
@pytest.mark.parametrize("weight", [0.0, 0.25, 0.5, 0.75, 1.0])
def test_ttnn_lerp_tts_scalar_weight(c_shape, t_shape, weight, device):
    torch.manual_seed(0)
    in_data1 = torch.empty(c_shape, dtype=torch.bfloat16).uniform_(-100, 100)
    in_data2 = torch.empty(t_shape, dtype=torch.bfloat16).uniform_(-100, 100)
    golden = torch.lerp(in_data1, in_data2, weight)

    input_tensor1 = ttnn.from_torch(in_data1, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    input_tensor2 = ttnn.from_torch(in_data2, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    ttnn_result = ttnn.lerp(input_tensor1, input_tensor2, weight)
    result = ttnn.to_torch(ttnn_result)

    assert_with_ulp(golden, result, ulp_threshold=10)


@pytest.mark.parametrize(
    "input_shape, end_shape, weight_shape",
    [
        ((8, 4, 768), (8, 4, 768), (8, 4, 768)),  # full, full, full
        ((8, 4, 1), (8, 4, 768), (8, 4, 768)),  # acol, bfull, cfull
        ((8, 1, 768), (8, 4, 768), (8, 4, 768)),  # arow, bfull, cfull
        ((8, 4, 768), (8, 4, 1), (8, 1, 768)),  # afull, bcol, crow
        ((8, 4, 768), (8, 1, 768), (8, 4, 1)),  # afull, brow, ccol
        ((8, 4, 1), (8, 1, 768), (8, 4, 768)),  # acol, brow, cfull
        ((8, 1, 768), (8, 4, 1), (8, 4, 768)),  # arow, bcol, cfull
        ((8, 1, 768), (8, 4, 768), (8, 4, 1)),  # arow, bfull, ccol
        ((8, 4, 1), (8, 4, 768), (8, 1, 768)),  # acol, bfull, crow
        ((1, 1, 1), (8, 4, 1), (8, 1, 768)),  # ascalar, bcol, crow
        ((1, 1, 1), (8, 1, 768), (8, 4, 1)),  # ascalar, brow, ccol
        ((8, 1, 768), (1, 1, 1), (8, 4, 1)),  # arow, bscalar, ccol
        ((8, 4, 1), (1, 1, 1), (8, 1, 768)),  # acol, bscalar, crow
    ],
)
def test_ttnn_lerp_ttt_row_col_mixed_bcast(input_shape, end_shape, weight_shape, device):
    torch.manual_seed(0)
    in_data1 = torch.empty(input_shape, dtype=torch.bfloat16).uniform_(-100, 100)
    in_data2 = torch.empty(end_shape, dtype=torch.bfloat16).uniform_(-100, 100)
    # Weight in [0, 1] for lerp
    in_data3 = torch.empty(weight_shape, dtype=torch.bfloat16).uniform_(0, 1)
    golden = torch.lerp(in_data1, in_data2, in_data3)

    input_tensor1 = ttnn.from_torch(in_data1, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    input_tensor2 = ttnn.from_torch(in_data2, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    input_tensor3 = ttnn.from_torch(in_data3, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    ttnn_result = ttnn.lerp(input_tensor1, input_tensor2, input_tensor3)
    result = ttnn.to_torch(ttnn_result)

    assert_with_ulp(golden, result, ulp_threshold=10)
