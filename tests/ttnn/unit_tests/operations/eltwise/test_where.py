# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
import pytest

from tests.ttnn.utils_for_testing import assert_with_pcc, assert_equal, assert_with_ulp, assert_allclose
from math import isnan


def torch_equal_nan(a, b):
    return torch.all((a == b) | (torch.isnan(a) & torch.isnan(b)))


# TTT,  // tensor-tensor-tensor
# TTS,  // tensor-tensor-scalar
# TST,  // tensor-scalar-tensor
# TSS,  // tensor-scalar-scalar


@pytest.mark.parametrize(
    "c_shape, t_shape, f_shape",
    [
        ((2, 3, 64, 128), (2, 3, 64, 128), (2, 3, 64, 128)),  # LLK
        ((3, 2, 3, 64, 128), (3, 2, 3, 64, 128), (3, 2, 3, 64, 128)),  # LLK
        ((256,), (256,), (256,)),  # LLK
    ],
)
@pytest.mark.parametrize("scalar", [15.5, float("nan"), float("inf"), 10.0, 5.0, -11.33])
@pytest.mark.parametrize("variant", ["TTS", "TST", "TTT"])
@pytest.mark.parametrize("condition", [1, 0])
def test_ttnn_where(c_shape, t_shape, f_shape, scalar, variant, condition, device):
    torch.manual_seed(0)
    C = torch.ones(c_shape, dtype=torch.float32) * condition
    if variant == "TTS":
        T = torch.randn(t_shape, dtype=torch.float32)
        F = scalar
    elif variant == "TST":
        T = scalar
        F = torch.randn(f_shape, dtype=torch.float32)
    elif variant == "TTT":
        T = torch.randn(t_shape, dtype=torch.float32)
        F = torch.ones(f_shape, dtype=torch.float32) * 10
    golden = torch.where(C.bool(), T, F)

    ttnn_C = ttnn.from_torch(C, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    if variant == "TTS":
        ttnn_T = ttnn.from_torch(T, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
        ttnn_F = scalar
    elif variant == "TST":
        ttnn_T = scalar
        ttnn_F = ttnn.from_torch(F, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    elif variant == "TTT":
        ttnn_T = ttnn.from_torch(T, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
        ttnn_F = ttnn.from_torch(F, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    ttnn_result = ttnn.where(ttnn_C, ttnn_T, ttnn_F)
    result = ttnn.to_torch(ttnn_result)

    assert torch_equal_nan(result, golden)


@pytest.mark.parametrize(
    "c_shape, t_shape, f_shape",
    [
        ((2, 3, 64, 128), (2, 3, 64, 128), (2, 3, 64, 128)),  # LLK
        ((3, 2, 3, 64, 128), (3, 2, 3, 64, 128), (3, 2, 3, 64, 128)),  # LLK
        ((256,), (256,), (256,)),  # LLK
    ],
)
@pytest.mark.parametrize("variant", ["TTS", "TST", "TTT"])
@pytest.mark.parametrize("condition", [1, 0])
@pytest.mark.parametrize("scalar", [-9, 10, 7])
def test_ttnn_where_int32(c_shape, t_shape, f_shape, variant, condition, scalar, device):
    torch.manual_seed(0)
    C = torch.ones(c_shape, dtype=torch.int32) * condition
    if variant == "TTS":
        T = torch.randint(-1000, 1000, t_shape, dtype=torch.int32)
        F = scalar
    elif variant == "TST":
        T = scalar
        F = torch.randint(-2000, 100, f_shape, dtype=torch.int32)
    elif variant == "TTT":
        T = torch.randint(-1000, 1000, t_shape, dtype=torch.int32)
        F = torch.randint(-2000, 100, f_shape, dtype=torch.int32)
    golden = torch.where(C.bool(), T, F)

    ttnn_C = ttnn.from_torch(C, dtype=ttnn.int32, layout=ttnn.TILE_LAYOUT, device=device)
    if variant == "TTS":
        ttnn_T = ttnn.from_torch(T, dtype=ttnn.int32, layout=ttnn.TILE_LAYOUT, device=device)
        ttnn_F = scalar
    elif variant == "TST":
        ttnn_T = scalar
        ttnn_F = ttnn.from_torch(F, dtype=ttnn.int32, layout=ttnn.TILE_LAYOUT, device=device)
    elif variant == "TTT":
        ttnn_T = ttnn.from_torch(T, dtype=ttnn.int32, layout=ttnn.TILE_LAYOUT, device=device)
        ttnn_F = ttnn.from_torch(F, dtype=ttnn.int32, layout=ttnn.TILE_LAYOUT, device=device)
    ttnn_result = ttnn.where(ttnn_C, ttnn_T, ttnn_F)
    result = ttnn.to_torch(ttnn_result)

    assert torch.equal(result, golden)


@pytest.mark.parametrize(
    "tor_dtype, ttnn_dtype",
    [
        (torch.bfloat16, ttnn.bfloat8_b),
    ],
)
@pytest.mark.parametrize(
    "c_shape, t_shape, f_shape",
    [
        ((2, 3, 64, 128), (2, 3, 64, 128), (2, 3, 64, 128)),  # LLK
        ((3, 2, 3, 64, 128), (3, 2, 3, 64, 128), (3, 2, 3, 64, 128)),  # LLK
        ((256,), (256,), (256,)),  # LLK
    ],
)
@pytest.mark.parametrize("scalar", [15.5, -11.5, 0, 55.5])
@pytest.mark.parametrize("variant", ["TTS", "TST", "TTT"])
@pytest.mark.parametrize("condition", [1, 0])
def test_ttnn_where_bfloat8b(tor_dtype, ttnn_dtype, c_shape, t_shape, f_shape, scalar, variant, condition, device):
    torch.manual_seed(0)
    C = torch.ones(c_shape, dtype=tor_dtype) * condition
    if variant == "TTS":
        T = torch.randn(t_shape, dtype=tor_dtype)
        F = scalar
    elif variant == "TST":
        T = scalar
        F = torch.randn(f_shape, dtype=tor_dtype)
    elif variant == "TTT":
        T = torch.randn(t_shape, dtype=tor_dtype)
        F = torch.ones(f_shape, dtype=tor_dtype) * 10

    ttnn_C = ttnn.from_torch(C, dtype=ttnn_dtype, layout=ttnn.TILE_LAYOUT, device=device)
    C = ttnn.to_torch(ttnn_C)
    if variant == "TTS":
        ttnn_T = ttnn.from_torch(T, dtype=ttnn_dtype, layout=ttnn.TILE_LAYOUT, device=device)
        T = ttnn.to_torch(ttnn_T)
        ttnn_F = scalar
    elif variant == "TST":
        ttnn_T = scalar
        ttnn_F = ttnn.from_torch(F, dtype=ttnn_dtype, layout=ttnn.TILE_LAYOUT, device=device)
        F = ttnn.to_torch(ttnn_F)
    elif variant == "TTT":
        ttnn_T = ttnn.from_torch(T, dtype=ttnn_dtype, layout=ttnn.TILE_LAYOUT, device=device)
        ttnn_F = ttnn.from_torch(F, dtype=ttnn_dtype, layout=ttnn.TILE_LAYOUT, device=device)
        T = ttnn.to_torch(ttnn_T)
        F = ttnn.to_torch(ttnn_F)

    ttnn_result = ttnn.where(ttnn_C, ttnn_T, ttnn_F)

    result = ttnn.to_torch(ttnn_result)
    golden = torch.where(C.bool(), T, F)

    assert torch.equal(result, golden)


def test_ttnn_where_forge():
    cond = torch.tensor([[1, 0, 1, 0]], dtype=torch.int32)
    x = torch.tensor([[10, 20, 30, 40]], dtype=torch.int32)
    y = torch.tensor([[100, 200, 300, 400]], dtype=torch.int32)
    expected = torch.where(cond != 0, x, y)

    with ttnn.manage_device(0) as dev:
        ttnn_cond = ttnn.from_torch(cond, dtype=ttnn.int32, layout=ttnn.TILE_LAYOUT, device=dev)
        real_cond = ttnn.eq(ttnn_cond, 1)
        ttnn_x = ttnn.from_torch(x, dtype=ttnn.int32, layout=ttnn.TILE_LAYOUT, device=dev)
        ttnn_y = ttnn.from_torch(y, dtype=ttnn.int32, layout=ttnn.TILE_LAYOUT, device=dev)
        ttnn_result = ttnn.where(real_cond, ttnn_x, ttnn_y)
        result = ttnn.to_torch(ttnn_result)

    assert torch.equal(result, expected), f"Expected {expected}, got {result}"


def test_ttnn_where_nan(device):
    tor_dtype = torch.float32

    condition = torch.tensor([1, 0, -2, 0, 5, 0, 0, 8, 0, -1, float("inf"), float("nan")], dtype=tor_dtype)
    condition_all_ones = torch.tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=tor_dtype)
    condition_all_zeros = torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=tor_dtype)

    # true and false value tensors
    true_values = torch.tensor(
        [1.0, float("nan"), 3.0, float("inf"), -float("inf"), -1.0, 0.0, -0.0, 42.49, -92.42, 111.0, 112.0],
        dtype=tor_dtype,
    )
    false_values = torch.tensor(
        [-1.0, 999.9, float("nan"), -float("inf"), float("inf"), 1.0, -0.0, 0.0, -3.14, 7.84, 222.0, 223.0],
        dtype=tor_dtype,
    )

    ttnn_condition = ttnn.from_torch(condition, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    ttnn_condition_all_ones = ttnn.from_torch(
        condition_all_ones, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device
    )
    ttnn_condition_all_zeros = ttnn.from_torch(
        condition_all_zeros, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device
    )

    ttnn_true_values = ttnn.from_torch(true_values, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    ttnn_false_values = ttnn.from_torch(false_values, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)

    ttnn_result1 = ttnn.where(ttnn_condition, ttnn_true_values, ttnn_false_values)
    ttnn_result2 = ttnn.where(ttnn_condition_all_ones, ttnn_true_values, ttnn_false_values)
    ttnn_result3 = ttnn.where(ttnn_condition_all_zeros, ttnn_true_values, ttnn_false_values)

    tt_result1 = ttnn.to_torch(ttnn_result1)
    tt_result2 = ttnn.to_torch(ttnn_result2)
    tt_result3 = ttnn.to_torch(ttnn_result3)

    # where operation in torch expects condition to be a boolean dtype, in ttnn.where we follow 0's & non-zero's (0's and 1's would be ideal)
    result1 = torch.where(condition.bool(), true_values, false_values)
    result2 = torch.where(condition_all_ones.bool(), true_values, false_values)
    result3 = torch.where(condition_all_zeros.bool(), true_values, false_values)

    assert torch_equal_nan(tt_result1, result1)
    assert torch_equal_nan(tt_result2, result2)
    assert torch_equal_nan(tt_result3, result3)


@pytest.mark.parametrize("h", [32])
@pytest.mark.parametrize("w", [32])
@pytest.mark.parametrize(
    "tor_dtype, ttnn_dtype", [(torch.bfloat16, ttnn.bfloat16), (torch.float32, ttnn.float32), (torch.int32, ttnn.int32)]
)
def test_ttnn_where_mcw(h, w, tor_dtype, ttnn_dtype, device):
    C = torch.arange(h * w, dtype=tor_dtype)
    C = (C % 2).float()  # Alternates 0, 1, 0, 1, ...
    C = C.reshape(1, 1, h, w)
    C = C.expand(1, 1, h, w).to(tor_dtype)  # Broadcast to (n, c, h, w)
    T = torch.ones(1, 1, h, w, dtype=tor_dtype) * 4.0
    F = torch.ones(1, 1, h, w, dtype=tor_dtype) * 10.0
    golden = torch.where(C != 0, T, F)

    ttnn_C = ttnn.from_torch(C, dtype=ttnn_dtype, layout=ttnn.TILE_LAYOUT, device=device)
    ttnn_T = ttnn.from_torch(T, dtype=ttnn_dtype, layout=ttnn.TILE_LAYOUT, device=device)
    ttnn_F = ttnn.from_torch(F, dtype=ttnn_dtype, layout=ttnn.TILE_LAYOUT, device=device)
    ttnn_result = ttnn.where(ttnn_C, ttnn_T, ttnn_F)
    result = ttnn.to_torch(ttnn_result)

    assert torch.equal(result, golden)


def test_ttnn_where_edge_cases_bf16(device):
    tor_dtype = torch.bfloat16
    ttnn_dtype = ttnn.bfloat16

    condition = torch.tensor([1, 0, -2, 0, 5, 0, 0, 8, 0, -1, float("inf"), float("nan")], dtype=tor_dtype)
    condition_all_ones = torch.tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=tor_dtype)
    condition_all_zeros = torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=tor_dtype)

    # true and false value tensors
    true_values = torch.tensor(
        [1.0, float("inf"), 3.0, float("inf"), -float("inf"), -1.0, 0.0, -0.0, 42.49, -92.42, 111.0, 112.0],
        dtype=tor_dtype,
    )
    false_values = torch.tensor(
        [-1.0, 999.9, float("inf"), -float("inf"), float("inf"), 1.0, -0.0, 0.0, -3.14, 7.84, 222.0, 223.0],
        dtype=tor_dtype,
    )

    ttnn_condition = ttnn.from_torch(condition, dtype=ttnn_dtype, layout=ttnn.TILE_LAYOUT, device=device)
    ttnn_condition_all_ones = ttnn.from_torch(
        condition_all_ones, dtype=ttnn_dtype, layout=ttnn.TILE_LAYOUT, device=device
    )
    ttnn_condition_all_zeros = ttnn.from_torch(
        condition_all_zeros, dtype=ttnn_dtype, layout=ttnn.TILE_LAYOUT, device=device
    )

    ttnn_true_values = ttnn.from_torch(true_values, dtype=ttnn_dtype, layout=ttnn.TILE_LAYOUT, device=device)
    ttnn_false_values = ttnn.from_torch(false_values, dtype=ttnn_dtype, layout=ttnn.TILE_LAYOUT, device=device)

    ttnn_result1 = ttnn.where(ttnn_condition, ttnn_true_values, ttnn_false_values)
    ttnn_result2 = ttnn.where(ttnn_condition_all_ones, ttnn_true_values, ttnn_false_values)
    ttnn_result3 = ttnn.where(ttnn_condition_all_zeros, ttnn_true_values, ttnn_false_values)

    tt_result1 = ttnn.to_torch(ttnn_result1)
    tt_result2 = ttnn.to_torch(ttnn_result2)
    tt_result3 = ttnn.to_torch(ttnn_result3)

    # where operation in torch expects condition to be a boolean dtype, in ttnn.where we follow 0's & non-zero's (0's and 1's would be ideal)
    result1 = torch.where(condition.bool(), true_values, false_values)
    result2 = torch.where(condition_all_ones.bool(), true_values, false_values)
    result3 = torch.where(condition_all_zeros.bool(), true_values, false_values)

    assert torch_equal_nan(tt_result1, result1)
    assert torch_equal_nan(tt_result2, result2)
    assert torch_equal_nan(tt_result3, result3)


def test_bf8b_exponent_behaviour(device):
    tor_dtype = torch.float32
    ttnn_dtype = ttnn.bfloat8_b

    condition = torch.tensor([1, 0, -2, 0, 5, 0, 0, 8, 0, -1, 1, 1], dtype=tor_dtype)
    condition_all_ones = torch.tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=tor_dtype)
    condition_all_zeros = torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=tor_dtype)

    # true and false value tensors
    true_values = torch.tensor(
        [1.0, 9090.0, 3.0, 1010.55, -1010.55, -1.0, 0.0, -0.0, 42.49, -92.42, 111.0, 112.0],
        dtype=tor_dtype,
    )
    false_values = torch.tensor(
        [-1.0, 999.9, 9090.0, -1010.55, 1010.55, 1.0, -0.0, 0.0, -3.14, 7.84, 222.0, 223.0],
        dtype=tor_dtype,
    )

    ttnn_condition = ttnn.from_torch(condition, dtype=ttnn_dtype, layout=ttnn.TILE_LAYOUT, device=device)
    ttnn_condition_all_ones = ttnn.from_torch(
        condition_all_ones, dtype=ttnn_dtype, layout=ttnn.TILE_LAYOUT, device=device
    )
    ttnn_condition_all_zeros = ttnn.from_torch(
        condition_all_zeros, dtype=ttnn_dtype, layout=ttnn.TILE_LAYOUT, device=device
    )

    ttnn_true_values = ttnn.from_torch(true_values, dtype=ttnn_dtype, layout=ttnn.TILE_LAYOUT, device=device)
    true_values = ttnn.to_torch(ttnn_true_values)
    ttnn_false_values = ttnn.from_torch(false_values, dtype=ttnn_dtype, layout=ttnn.TILE_LAYOUT, device=device)
    false_values = ttnn.to_torch(ttnn_false_values)

    ttnn_result1 = ttnn.where(ttnn_condition, ttnn_true_values, ttnn_false_values)
    ttnn_result2 = ttnn.where(ttnn_condition_all_ones, ttnn_true_values, ttnn_false_values)
    ttnn_result3 = ttnn.where(ttnn_condition_all_zeros, ttnn_true_values, ttnn_false_values)

    tt_result1 = ttnn.to_torch(ttnn_result1)
    tt_result2 = ttnn.to_torch(ttnn_result2)
    tt_result3 = ttnn.to_torch(ttnn_result3)

    # where operation in torch expects condition to be a boolean dtype, in ttnn.where we follow 0's & non-zero's (0's and 1's would be ideal)
    result1 = torch.where(condition.bool(), true_values, false_values)
    result2 = torch.where(condition_all_ones.bool(), true_values, false_values)
    result3 = torch.where(condition_all_zeros.bool(), true_values, false_values)

    assert torch_equal_nan(tt_result1, result1)
    assert torch_equal_nan(tt_result2, result2)
    assert torch_equal_nan(tt_result3, result3)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("h, w", [[64, 128]])
def test_where_ttt(device, dtype, h, w):
    torch.manual_seed(0)

    ttnn_dtype = ttnn.bfloat16
    if dtype == torch.float32:
        ttnn_dtype = ttnn.float32

    C = torch.rand((h, w), dtype=dtype).uniform_(-100, 100)
    T = torch.rand((h, w), dtype=dtype).uniform_(-100, 100)
    F = torch.rand((h, w), dtype=dtype).uniform_(-100, 100)

    C = (C > 0).float()
    ttnn_C = ttnn.from_torch(C, dtype=ttnn_dtype, layout=ttnn.TILE_LAYOUT, device=device)
    ttnn_T = ttnn.from_torch(T, dtype=ttnn_dtype, layout=ttnn.TILE_LAYOUT, device=device)
    ttnn_F = ttnn.from_torch(F, dtype=ttnn_dtype, layout=ttnn.TILE_LAYOUT, device=device)

    golden = torch.where(C.bool(), T, F)
    ttnn_result = ttnn.where(ttnn_C, ttnn_T, ttnn_F)
    result = ttnn.to_torch(ttnn_result)

    assert torch_equal_nan(result, golden)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("h, w", [[64, 128]])
@pytest.mark.parametrize("scalar", [15.5, float("nan"), float("inf"), -float("inf")])
def test_where_tts(device, dtype, h, w, scalar):
    if dtype == torch.bfloat16 and isnan(scalar):
        pytest.xfail("NaN is packed as inf for ttnn.bfloat16")

    torch.manual_seed(0)

    ttnn_dtype = ttnn.bfloat16
    if dtype == torch.float32:
        ttnn_dtype = ttnn.float32

    C = torch.rand((h, w), dtype=dtype).uniform_(-100, 100)
    T = torch.rand((h, w), dtype=dtype).uniform_(-100, 100)
    F = scalar

    C = (C > 0).float()
    ttnn_C = ttnn.from_torch(C, dtype=ttnn_dtype, layout=ttnn.TILE_LAYOUT, device=device)
    ttnn_T = ttnn.from_torch(T, dtype=ttnn_dtype, layout=ttnn.TILE_LAYOUT, device=device)

    golden = torch.where(C.bool(), T, F)
    ttnn_result = ttnn.where(ttnn_C, ttnn_T, F)
    result = ttnn.to_torch(ttnn_result)

    assert torch_equal_nan(result, golden)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("h, w", [[64, 128]])
@pytest.mark.parametrize("scalar", [15.5, float("nan"), float("inf"), -float("inf")])
def test_where_tst(device, dtype, h, w, scalar):
    if dtype == torch.bfloat16 and isnan(scalar):
        pytest.xfail("NaN is packed as inf for ttnn.bfloat16")

    torch.manual_seed(0)

    ttnn_dtype = ttnn.bfloat16
    if dtype == torch.float32:
        ttnn_dtype = ttnn.float32

    C = torch.rand((h, w), dtype=dtype).uniform_(-100, 100)
    T = scalar
    F = torch.rand((h, w), dtype=dtype).uniform_(-100, 100)

    C = (C > 0).float()
    ttnn_C = ttnn.from_torch(C, dtype=ttnn_dtype, layout=ttnn.TILE_LAYOUT, device=device)
    ttnn_F = ttnn.from_torch(F, dtype=ttnn_dtype, layout=ttnn.TILE_LAYOUT, device=device)

    golden = torch.where(C.bool(), T, F)
    ttnn_result = ttnn.where(ttnn_C, T, ttnn_F)
    result = ttnn.to_torch(ttnn_result)

    assert torch_equal_nan(result, golden)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("h, w", [[64, 128]])
@pytest.mark.parametrize(
    "scalar1, scalar2",
    [
        [15.5, 31.2],
        [15.5, float("nan")],
        [float("nan"), 31.2],
        [float("inf"), -float("inf")],
        [-float("inf"), float("nan")],
    ],
)
def test_where_tss(device, dtype, h, w, scalar1, scalar2):
    if dtype == torch.bfloat16 and (isnan(scalar1) or isnan(scalar2)):
        pytest.xfail("NaN is packed as inf for ttnn.bfloat16")

    torch.manual_seed(0)

    ttnn_dtype = ttnn.bfloat16
    if dtype == torch.float32:
        ttnn_dtype = ttnn.float32

    C = torch.rand((h, w), dtype=dtype).uniform_(-100, 100)
    T = scalar1
    F = scalar2

    C = (C > 0).float()
    ttnn_C = ttnn.from_torch(C, dtype=ttnn_dtype, layout=ttnn.TILE_LAYOUT, device=device)

    golden = torch.where(C.bool(), T, F)
    ttnn_result = ttnn.where(ttnn_C, T, F)
    result = ttnn.to_torch(ttnn_result)

    if dtype == torch.bfloat16:
        assert_with_pcc(result, golden)
    else:
        assert torch_equal_nan(result, golden)


@pytest.mark.parametrize(
    "torch_dtype, ttnn_dtype",
    [
        (torch.float32, ttnn.float32),
        (torch.bfloat16, ttnn.bfloat16),
    ],
)
@pytest.mark.parametrize(
    "scalars",
    [
        (1000.0, 700.0),
        (float("inf"), 1.0),
        (1.0, float("inf")),
        (float("nan"), 0.0),
        (2.0, float("nan")),
        (-0.0, 34.5),
    ],
)
@pytest.mark.parametrize(
    "input_shapes",
    [
        torch.Size([100]),
        torch.Size([64, 128]),
        torch.Size([3, 128, 32]),
        torch.Size([1, 3, 320, 384]),
        torch.Size([1, 1, 32, 320, 12]),
    ],
)
def test_where_TSS_float_types(torch_dtype, ttnn_dtype, scalars, input_shapes, device):
    condition = torch.tensor([[0, 1] * (input_shapes[-1] // 2)] * input_shapes[0], dtype=torch_dtype)
    scalar_true, scalar_false = scalars

    torch_result = torch.where(condition.bool(), scalar_true, scalar_false)
    if torch_dtype != torch.float32:
        torch_result = torch.where(
            torch.isnan(torch_result), torch.tensor(float("inf"), dtype=torch_dtype), torch_result
        )

    ttnn_condition = ttnn.from_torch(condition, dtype=ttnn_dtype, layout=ttnn.TILE_LAYOUT, device=device)
    ttnn_result = ttnn.where(ttnn_condition, scalar_true, scalar_false)

    tt_result = ttnn.to_torch(ttnn_result)
    assert torch_equal_nan(tt_result, torch_result)


@pytest.mark.parametrize(
    "scalars",
    [
        (3, 7),
        (-10, 42),
        (0, 1),
        (9999, -9999),
        (-24567, 16777216),
        (-16777216, 56789),
    ],
)
@pytest.mark.parametrize(
    "input_shapes",
    [
        torch.Size([100]),
        torch.Size([64, 128]),
        torch.Size([3, 128, 32]),
        torch.Size([1, 3, 320, 384]),
        torch.Size([1, 1, 32, 320, 12]),
    ],
)
def test_where_TSS_int_types(scalars, input_shapes, device):
    scalar_true, scalar_false = scalars
    condition = torch.tensor([[0, 1] * (input_shapes[-1] // 2)] * input_shapes[0], dtype=torch.int32)

    torch_result = torch.where(condition.bool(), scalar_true, scalar_false)

    ttnn_condition = ttnn.from_torch(condition, dtype=ttnn.int32, layout=ttnn.TILE_LAYOUT, device=device)
    ttnn_result = ttnn.where(ttnn_condition, scalar_true, scalar_false)

    tt_result = ttnn.to_torch(ttnn_result)

    assert torch.equal(tt_result, torch_result)


def test_div_edgcase(device):
    a = torch.tensor([1, 2, -4, 0, -6, 0], dtype=torch.bfloat16)
    b = torch.tensor([-1, 0, 0, 0, -2, 7], dtype=torch.bfloat16)

    ttnn_a = ttnn.from_torch(a, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    ttnn_b = ttnn.from_torch(b, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    output_tensor = ttnn.div(ttnn_a, ttnn_b, accurate_mode=True)
    golden_tensor = torch.div(a, b)

    output_tensor = ttnn.to_torch(output_tensor)

    # accurate_mode=True
    # output_tensor tensor([-1., inf, -inf, inf,  3.,  0.], dtype=torch.bfloat16)
    # golden_tensor tensor([-1., inf, -inf, nan,  3.,  0.], dtype=torch.bfloat16)

    # Replace NaN values in golden tensor with inf to match expected behavior of ttnn.bfloat16
    golden_tensor = torch.where(
        torch.isnan(golden_tensor), torch.tensor(float("inf"), dtype=golden_tensor.dtype), golden_tensor
    )

    assert torch.allclose(output_tensor, golden_tensor, equal_nan=False)


def test_addcdiv_edgcase(device):
    # Hardcoded input tensors
    a = torch.tensor([1, 2, -4, 0, -6, 0], dtype=torch.bfloat16)
    b = torch.tensor([-1, 0, 0, 0, -2, 7], dtype=torch.bfloat16)
    c = torch.tensor([0, 0, 0, 0, 0, 0], dtype=torch.bfloat16)

    ttnn_a = ttnn.from_torch(a, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    ttnn_b = ttnn.from_torch(b, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    ttnn_c = ttnn.from_torch(c, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    value = -0.5
    output_tensor = ttnn.addcdiv(ttnn_c, ttnn_a, ttnn_b, value=value)
    golden_tensor = torch.addcdiv(c, a, b, value=value)

    output_tensor = ttnn.to_torch(output_tensor)

    # output_tensor tensor([ 0.5000,    -inf,     inf,    -inf, -1.5000,  0.0000],dtype=torch.bfloat16)
    # golden_tensor tensor([ 0.5000,    -inf,     inf,     nan, -1.5000,  0.0000],dtype=torch.bfloat16)

    # Replace NaN values in golden tensor with inf to match expected behavior of ttnn.bfloat16
    golden_tensor = torch.where(
        torch.isnan(golden_tensor), value * torch.tensor(float("inf"), dtype=golden_tensor.dtype), golden_tensor
    )

    assert torch.allclose(output_tensor, golden_tensor, equal_nan=False)


def test_addcdiv_edgcase_fp32(device):
    a = torch.tensor([1, 2, -4, 0, -6, 0], dtype=torch.float32)
    b = torch.tensor([-1, 0, 0, 0, -2, 7], dtype=torch.float32)
    c = torch.tensor([0, 0, 0, 0, 0, 0], dtype=torch.float32)

    ttnn_a = ttnn.from_torch(a, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    ttnn_b = ttnn.from_torch(b, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    ttnn_c = ttnn.from_torch(c, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)

    output_tensor = ttnn.addcdiv(ttnn_c, ttnn_a, ttnn_b, value=-0.5)
    golden_tensor = torch.addcdiv(c, a, b, value=-0.5)

    output_tensor1 = ttnn.to_torch(output_tensor)

    # output_tensor tensor([ 0.5000,    -inf,     inf,     nan, -1.5000,  0.0000])
    # golden_tensor tensor([ 0.5000,    -inf,     inf,     nan, -1.5000,  0.0000])

    assert torch_equal_nan(output_tensor1, golden_tensor)
