# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import random
import ttnn
from tests.ttnn.unit_tests.operations.backward.utility_funcs import data_gen_with_range, compare_pcc, compare_equal
from models.utility_functions import is_grayskull, skip_for_grayskull, skip_for_wormhole_b0


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 1, 320, 384])),
        (torch.Size([1, 3, 320, 384])),
    ),
)
def test_binary_hypot_ttnn(input_shapes, device):
    in_data1, input_tensor1 = data_gen_with_range(input_shapes, -100, 100, device)
    in_data2, input_tensor2 = data_gen_with_range(input_shapes, -150, 150, device)

    output_tensor = ttnn.hypot(input_tensor1, input_tensor2)
    golden_function = ttnn.get_golden_function(ttnn.hypot)
    golden_tensor = golden_function(in_data1, in_data2)

    comp_pass = compare_pcc([output_tensor], [golden_tensor])
    assert comp_pass


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 1, 320, 384])),
        (torch.Size([1, 3, 320, 384])),
    ),
)
def test_binary_xlogy_ttnn(input_shapes, device):
    in_data1, input_tensor1 = data_gen_with_range(input_shapes, -100, 100, device)
    in_data2, input_tensor2 = data_gen_with_range(input_shapes, -150, 150, device)

    output_tensor = ttnn.xlogy(input_tensor1, input_tensor2)
    golden_function = ttnn.get_golden_function(ttnn.xlogy)
    golden_tensor = golden_function(in_data1, in_data2)

    comp_pass = compare_pcc([output_tensor], [golden_tensor])
    assert comp_pass


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 1, 320, 384])),
        (torch.Size([1, 3, 320, 384])),
    ),
)
def test_binary_nextafter_ttnn(input_shapes, device):
    in_data1, input_tensor1 = data_gen_with_range(input_shapes, -100, 100, device)
    in_data2, input_tensor2 = data_gen_with_range(input_shapes, -150, 150, device)

    output_tensor = ttnn.nextafter(input_tensor1, input_tensor2)
    golden_function = ttnn.get_golden_function(ttnn.nextafter)
    golden_tensor = golden_function(in_data1, in_data2)

    comp_pass = compare_pcc([output_tensor], [golden_tensor])
    assert comp_pass


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 1, 320, 384])),
        (torch.Size([1, 3, 320, 384])),
    ),
)
@pytest.mark.parametrize("atol", [1.0, 5.0, 10.0])
@pytest.mark.parametrize("rtol", [1.0, 5.0, 10.0])
@pytest.mark.parametrize("equal_nan", [True, False])
def test_binary_isclose_ttnn(input_shapes, atol, rtol, equal_nan, device):
    in_data1, input_tensor1 = data_gen_with_range(input_shapes, -100, 100, device)
    in_data2, input_tensor2 = data_gen_with_range(input_shapes, -150, 150, device)

    output_tensor = ttnn.isclose(input_tensor1, input_tensor2, rtol=rtol, atol=atol, equal_nan=equal_nan)

    golden_function = ttnn.get_golden_function(ttnn.isclose)
    golden_tensor = golden_function(in_data1, in_data2, rtol=rtol, atol=atol, equal_nan=equal_nan)

    comp_pass = compare_pcc([output_tensor], [golden_tensor])
    assert comp_pass


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 1, 320, 384])),
        (torch.Size([1, 3, 320, 384])),
    ),
)
def test_binary_minimum_ttnn(input_shapes, device):
    in_data1, input_tensor1 = data_gen_with_range(input_shapes, -100, 100, device)
    in_data2, input_tensor2 = data_gen_with_range(input_shapes, -150, 150, device)

    output_tensor = ttnn.minimum(input_tensor1, input_tensor2)
    golden_function = ttnn.get_golden_function(ttnn.minimum)
    golden_tensor = golden_function(in_data1, in_data2)

    comp_pass = compare_pcc([output_tensor], [golden_tensor])
    assert comp_pass


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 1, 320, 384])),
        (torch.Size([1, 3, 320, 384])),
    ),
)
def test_binary_maximum_ttnn(input_shapes, device):
    in_data1, input_tensor1 = data_gen_with_range(input_shapes, -100, 100, device)
    in_data2, input_tensor2 = data_gen_with_range(input_shapes, -150, 150, device)

    output_tensor = ttnn.maximum(input_tensor1, input_tensor2)
    golden_function = ttnn.get_golden_function(ttnn.maximum)
    golden_tensor = golden_function(in_data1, in_data2)

    comp_pass = compare_pcc([output_tensor], [golden_tensor])
    assert comp_pass


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 1, 320, 384])),
        (torch.Size([1, 3, 320, 384])),
    ),
)
def test_binary_atan2_ttnn(input_shapes, device):
    in_data1, input_tensor1 = data_gen_with_range(input_shapes, -100, 100, device)
    in_data2, input_tensor2 = data_gen_with_range(input_shapes, -150, 150, device)

    output_tensor = ttnn.atan2(input_tensor1, input_tensor2)
    golden_function = ttnn.get_golden_function(ttnn.atan2)
    golden_tensor = golden_function(in_data1, in_data2)

    comp_pass = compare_pcc([output_tensor], [golden_tensor])
    assert comp_pass


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 1, 320, 384])),
        (torch.Size([1, 3, 320, 384])),
    ),
)
def test_binary_logical_xor_ttnn(input_shapes, device):
    in_data1, input_tensor1 = data_gen_with_range(input_shapes, -100, 100, device)
    in_data2, input_tensor2 = data_gen_with_range(input_shapes, -150, 150, device)

    output_tensor = ttnn.logical_xor(input_tensor1, input_tensor2)
    golden_function = ttnn.get_golden_function(ttnn.logical_xor)
    golden_tensor = golden_function(in_data1, in_data2)

    comp_pass = compare_pcc([output_tensor], [golden_tensor])
    assert comp_pass


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 1, 320, 384])),
        (torch.Size([1, 3, 320, 384])),
    ),
)
@pytest.mark.parametrize("alpha", [1.0, 5.0, 10.0])
def test_binary_addalpha_ttnn(input_shapes, alpha, device):
    in_data1, input_tensor1 = data_gen_with_range(input_shapes, -100, 100, device)
    in_data2, input_tensor2 = data_gen_with_range(input_shapes, -150, 150, device)

    output_tensor = ttnn.addalpha(input_tensor1, input_tensor2, alpha)
    golden_function = ttnn.get_golden_function(ttnn.addalpha)
    golden_tensor = golden_function(in_data1, in_data2, alpha)

    comp_pass = compare_pcc([output_tensor], [golden_tensor])
    assert comp_pass


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 1, 320, 384])),
        (torch.Size([1, 3, 320, 384])),
    ),
)
@pytest.mark.parametrize("alpha", [1.0, 5.0, 10.0])
def test_binary_subalpha_ttnn(input_shapes, alpha, device):
    in_data1, input_tensor1 = data_gen_with_range(input_shapes, -100, 100, device)
    in_data2, input_tensor2 = data_gen_with_range(input_shapes, -150, 150, device)

    output_tensor = ttnn.subalpha(input_tensor1, input_tensor2, alpha)
    golden_function = ttnn.get_golden_function(ttnn.subalpha)
    golden_tensor = golden_function(in_data1, in_data2, alpha)

    comp_pass = compare_pcc([output_tensor], [golden_tensor])
    assert comp_pass


@pytest.mark.parametrize("accurate_mode", [False, True])
@pytest.mark.parametrize("round_mode", ["None", "trunc", "floor"])
@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 1, 320, 384])),
        (torch.Size([1, 3, 320, 384])),
    ),
)
def test_binary_div_ttnn(accurate_mode, round_mode, input_shapes, device):
    if is_grayskull():
        if round_mode in ["trunc", "floor"]:
            pytest.skip("does not work for Grayskull -skipping")
    if accurate_mode == False:  # If input_b is non-zero tensor
        in_data1, input_tensor1 = data_gen_with_range(input_shapes, -100, 100, device)
        in_data2, input_tensor2 = data_gen_with_range(input_shapes, -150, -1, device)
    else:
        in_data1, input_tensor1 = data_gen_with_range(input_shapes, -100, 100, device)
        in_data2, input_tensor2 = data_gen_with_range(input_shapes, -100, 100, device)

    output_tensor = ttnn.div(input_tensor1, input_tensor2, accurate_mode=accurate_mode, round_mode=round_mode)
    golden_function = ttnn.get_golden_function(ttnn.div)
    golden_tensor = golden_function(in_data1, in_data2, round_mode)

    comp_pass = compare_pcc([output_tensor], [golden_tensor])
    assert comp_pass


@pytest.mark.parametrize("accurate_mode", [False, True])
@pytest.mark.parametrize("round_mode", ["None", "trunc", "floor"])
@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 1, 320, 384])),
        (torch.Size([1, 3, 320, 384])),
    ),
)
@pytest.mark.parametrize("value", [-5.1, 0.0, 10.9])
def test_binary_div_overload_ttnn(accurate_mode, round_mode, input_shapes, value, device):
    if is_grayskull():
        if round_mode in ["trunc", "floor"]:
            pytest.skip("does not work for Grayskull -skipping")
    in_data1, input_tensor1 = data_gen_with_range(input_shapes, -100, 100, device)

    output_tensor = ttnn.div(input_tensor1, value, accurate_mode=accurate_mode, round_mode=round_mode)
    golden_function = ttnn.get_golden_function(ttnn.div)
    golden_tensor = golden_function(in_data1, value, round_mode)

    comp_pass = compare_pcc([output_tensor], [golden_tensor])
    assert comp_pass


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 1, 320, 384])),
        (torch.Size([1, 3, 320, 384])),
    ),
)
def test_binary_div_no_nan_ttnn(input_shapes, device):
    in_data1, input_tensor1 = data_gen_with_range(input_shapes, -100, 100, device)
    in_data2, input_tensor2 = data_gen_with_range(input_shapes, -150, 150, device)

    output_tensor = ttnn.div_no_nan(input_tensor1, input_tensor2)
    golden_function = ttnn.get_golden_function(ttnn.div_no_nan)
    golden_tensor = golden_function(in_data1, in_data2)

    comp_pass = compare_pcc([output_tensor], [golden_tensor])
    assert comp_pass


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 1, 320, 384])),
        (torch.Size([1, 3, 320, 384])),
    ),
)
@pytest.mark.parametrize("value", [-5.1, 0.0, 10.9])
def test_binary_div_no_nan_overload_ttnn(input_shapes, value, device):
    in_data1, input_tensor1 = data_gen_with_range(input_shapes, -100, 100, device)

    output_tensor = ttnn.div_no_nan(input_tensor1, value)
    golden_function = ttnn.get_golden_function(ttnn.div_no_nan)
    golden_tensor = golden_function(in_data1, value)

    comp_pass = compare_pcc([output_tensor], [golden_tensor])
    assert comp_pass


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 1, 320, 384])),
        (torch.Size([1, 3, 320, 384])),
    ),
)
@skip_for_grayskull("#ToDo: GS implementation needs to be done for Floor")
def test_binary_floor_div_ttnn(input_shapes, device):
    in_data1, input_tensor1 = data_gen_with_range(input_shapes, -350, 100, device)
    in_data2, input_tensor2 = data_gen_with_range(input_shapes, -100, 100, device)
    output_tensor = ttnn.floor_div(input_tensor1, input_tensor2)
    golden_function = ttnn.get_golden_function(ttnn.floor_div)
    golden_tensor = golden_function(in_data1, in_data2)

    comp_pass = compare_pcc([output_tensor], [golden_tensor])
    assert comp_pass


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 1, 320, 384])),
        (torch.Size([1, 3, 320, 384])),
    ),
)
@pytest.mark.parametrize("value", [-5.1, 0.0, 10.9])
@skip_for_grayskull("#ToDo: GS implementation needs to be done for Floor")
def test_binary_floor_div_overload_ttnn(input_shapes, value, device):
    in_data1, input_tensor1 = data_gen_with_range(input_shapes, -100, 100, device)

    output_tensor = ttnn.floor_div(input_tensor1, value)
    golden_function = ttnn.get_golden_function(ttnn.floor_div)
    golden_tensor = golden_function(in_data1, value)

    comp_pass = compare_pcc([output_tensor], [golden_tensor])
    assert comp_pass


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 1, 320, 384])),
        (torch.Size([1, 3, 320, 384])),
    ),
)
@skip_for_grayskull("#ToDo: GS implementation needs to be done for remainder")
def test_binary_remainder_ttnn(input_shapes, device):
    in_data1, input_tensor1 = data_gen_with_range(input_shapes, -150, 150, device)
    in_data2, input_tensor2 = data_gen_with_range(input_shapes, -100, 100, device)
    output_tensor = ttnn.remainder(input_tensor1, input_tensor2)
    golden_function = ttnn.get_golden_function(ttnn.remainder)
    golden_tensor = golden_function(in_data1, in_data2)

    comp_pass = compare_pcc([output_tensor], [golden_tensor])
    assert comp_pass


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 1, 320, 384])),
        (torch.Size([1, 3, 320, 384])),
    ),
)
@pytest.mark.parametrize(
    "scalar",
    {random.randint(-100, 100) + 0.5 for _ in range(5)},
)
@pytest.mark.skip(reason="#10942 Test fails for certain scalar values.")
def test_remainder_ttnn(input_shapes, scalar, device):
    in_data1, input_tensor1 = data_gen_with_range(input_shapes, -150, 150, device)
    output_tensor = ttnn.remainder(input_tensor1, scalar)
    golden_function = ttnn.get_golden_function(ttnn.remainder)
    golden_tensor = golden_function(in_data1, scalar)

    comp_pass = compare_pcc([output_tensor], [golden_tensor])
    assert comp_pass


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 32])),
        # (torch.Size([1, 1, 320, 384])),
        # (torch.Size([1, 3, 320, 384])),
    ),
)
@skip_for_grayskull("#ToDo: GS implementation needs to be done for fmod")
def test_binary_fmod_ttnn(input_shapes, device):
    in_data1, input_tensor1 = data_gen_with_range(input_shapes, -150, 150, device)
    in_data2, input_tensor2 = data_gen_with_range(input_shapes, -100, 100, device)

    output_tensor = ttnn.fmod(input_tensor1, input_tensor2)
    golden_function = ttnn.get_golden_function(ttnn.fmod)
    golden_tensor = golden_function(in_data1, in_data2)

    comp_pass = compare_pcc([output_tensor], [golden_tensor])
    assert comp_pass


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 1, 320, 384])),
        (torch.Size([1, 3, 320, 384])),
    ),
)
@pytest.mark.parametrize(
    "scalar",
    {random.randint(-100, 100) + 0.5 for _ in range(5)},
)
@skip_for_grayskull("#ToDo: GS implementation needs to be done for fmod")
def test_fmod_ttnn(input_shapes, scalar, device):
    in_data1, input_tensor1 = data_gen_with_range(input_shapes, -150, 150, device)

    output_tensor = ttnn.fmod(input_tensor1, scalar)
    golden_function = ttnn.get_golden_function(ttnn.fmod)
    golden_tensor = golden_function(in_data1, scalar)

    comp_pass = compare_pcc([output_tensor], [golden_tensor])
    assert comp_pass


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 1, 320, 384])),
        (torch.Size([1, 3, 320, 384])),
    ),
)
def test_binary_logical_and__ttnn(input_shapes, device):
    in_data1, input_tensor1 = data_gen_with_range(input_shapes, -150, 150, device)
    in_data2, input_tensor2 = data_gen_with_range(input_shapes, -100, 100, device)
    ttnn.logical_and_(input_tensor1, input_tensor2)
    golden_function = ttnn.get_golden_function(ttnn.logical_and_)
    golden_tensor = golden_function(in_data1, in_data2)

    comp_pass = compare_pcc([input_tensor1], [golden_tensor])
    assert comp_pass


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 1, 320, 384])),
        (torch.Size([1, 3, 320, 384])),
    ),
)
def test_binary_logical_or__ttnn(input_shapes, device):
    in_data1, input_tensor1 = data_gen_with_range(input_shapes, -150, 150, device)
    in_data2, input_tensor2 = data_gen_with_range(input_shapes, -100, 100, device)
    ttnn.logical_or_(input_tensor1, input_tensor2)
    golden_function = ttnn.get_golden_function(ttnn.logical_or_)
    golden_tensor = golden_function(in_data1, in_data2)

    comp_pass = compare_pcc([input_tensor1], [golden_tensor])
    assert comp_pass


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 1, 320, 384])),
        (torch.Size([1, 3, 320, 384])),
    ),
)
def test_binary_logical_xor__ttnn(input_shapes, device):
    in_data1, input_tensor1 = data_gen_with_range(input_shapes, -150, 150, device)
    in_data2, input_tensor2 = data_gen_with_range(input_shapes, -100, 100, device)
    ttnn.logical_xor_(input_tensor1, input_tensor2)
    golden_function = ttnn.get_golden_function(ttnn.logical_xor_)
    golden_tensor = golden_function(in_data1, in_data2)

    comp_pass = compare_pcc([input_tensor1], [golden_tensor])
    assert comp_pass


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 1, 320, 384])),
        (torch.Size([1, 3, 320, 384])),
    ),
)
def test_binary_scatter_ttnn(input_shapes, device):
    in_data1, input_tensor1 = data_gen_with_range(input_shapes, -100, 100, device)
    in_data2, input_tensor2 = data_gen_with_range(input_shapes, -150, 150, device)

    output_tensor = ttnn.scatter(input_tensor1, input_tensor2)
    golden_function = ttnn.get_golden_function(ttnn.scatter)
    golden_tensor = golden_function(in_data1, in_data2)

    comp_pass = compare_pcc([output_tensor], [golden_tensor])
    assert comp_pass


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 1, 320, 384])),
        (torch.Size([1, 3, 320, 384])),
    ),
)
@pytest.mark.parametrize("coeffs", [[0.0], [-5.0, 2.0], [-3.0, 0.0, 10.0], [-100.0, -25.0, 0.0, 15.0, 100.0]])
def test_binary_polyval_ttnn(input_shapes, coeffs, device):
    in_data1, input_tensor1 = data_gen_with_range(input_shapes, -100, 100, device)

    output_tensor = ttnn.polyval(input_tensor1, coeffs)
    golden_function = ttnn.get_golden_function(ttnn.polyval)
    golden_tensor = golden_function(in_data1, coeffs)

    comp_pass = compare_pcc([output_tensor], [golden_tensor])
    assert comp_pass


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 1, 320, 384])),
        (torch.Size([1, 3, 320, 384])),
    ),
)
def test_binary_gti_ttnn(input_shapes, device):
    in_data1, input_tensor1 = data_gen_with_range(input_shapes, -100, 100, device)
    in_data2, input_tensor2 = data_gen_with_range(input_shapes, -150, 150, device)
    ttnn.gt_(input_tensor1, input_tensor2)
    golden_function = ttnn.get_golden_function(ttnn.gt_)
    golden_tensor = golden_function(in_data1, in_data2)

    comp_pass = compare_equal([input_tensor1], [golden_tensor])
    assert comp_pass


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 1, 320, 384])),
        (torch.Size([1, 3, 320, 384])),
    ),
)
@pytest.mark.parametrize(
    "scalar",
    {random.randint(-100, 100) + 0.5 for _ in range(5)},
)
def test_gti_ttnn(input_shapes, scalar, device):
    in_data, input_tensor = data_gen_with_range(input_shapes, -100, 100, device)
    ttnn.gt_(input_tensor, scalar)
    golden_function = ttnn.get_golden_function(ttnn.gt_)
    golden_tensor = golden_function(in_data, scalar)

    comp_pass = compare_equal([input_tensor], [golden_tensor])
    assert comp_pass


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 1, 320, 384])),
        (torch.Size([1, 3, 320, 384])),
    ),
)
def test_binary_gei_ttnn(input_shapes, device):
    in_data1, input_tensor1 = data_gen_with_range(input_shapes, -100, 100, device)
    in_data2, input_tensor2 = data_gen_with_range(input_shapes, -150, 150, device)
    ttnn.ge_(input_tensor1, input_tensor2)
    golden_function = ttnn.get_golden_function(ttnn.ge_)
    golden_tensor = golden_function(in_data1, in_data2)

    comp_pass = compare_equal([input_tensor1], [golden_tensor])
    assert comp_pass


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 1, 320, 384])),
        (torch.Size([1, 3, 320, 384])),
    ),
)
@pytest.mark.parametrize(
    "scalar",
    {random.randint(-100, 100) + 0.5 for _ in range(5)},
)
def test_gei_ttnn(input_shapes, scalar, device):
    in_data, input_tensor = data_gen_with_range(input_shapes, -100, 100, device)
    ttnn.ge_(input_tensor, scalar)
    golden_function = ttnn.get_golden_function(ttnn.ge_)
    golden_tensor = golden_function(in_data, scalar)

    comp_pass = compare_equal([input_tensor], [golden_tensor])
    assert comp_pass


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 1, 320, 384])),
        (torch.Size([1, 3, 320, 384])),
    ),
)
def test_binary_lti_ttnn(input_shapes, device):
    in_data1, input_tensor1 = data_gen_with_range(input_shapes, -100, 100, device)
    in_data2, input_tensor2 = data_gen_with_range(input_shapes, -150, 150, device)
    ttnn.lt_(input_tensor1, input_tensor2)
    golden_function = ttnn.get_golden_function(ttnn.lt_)
    golden_tensor = golden_function(in_data1, in_data2)

    comp_pass = compare_equal([input_tensor1], [golden_tensor])
    assert comp_pass


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 1, 320, 384])),
        (torch.Size([1, 3, 320, 384])),
    ),
)
@pytest.mark.parametrize(
    "scalar",
    {random.randint(-100, 100) + 0.5 for _ in range(5)},
)
def test_lti_ttnn(input_shapes, scalar, device):
    in_data, input_tensor = data_gen_with_range(input_shapes, -100, 100, device)
    ttnn.lt_(input_tensor, scalar)
    golden_function = ttnn.get_golden_function(ttnn.lt_)
    golden_tensor = golden_function(in_data, scalar)

    comp_pass = compare_equal([input_tensor], [golden_tensor])
    assert comp_pass


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 1, 320, 384])),
        (torch.Size([1, 3, 320, 384])),
    ),
)
def test_binary_lei_ttnn(input_shapes, device):
    in_data1, input_tensor1 = data_gen_with_range(input_shapes, -100, 100, device)
    in_data2, input_tensor2 = data_gen_with_range(input_shapes, -150, 150, device)
    ttnn.le_(input_tensor1, input_tensor2)
    golden_function = ttnn.get_golden_function(ttnn.le_)
    golden_tensor = golden_function(in_data1, in_data2)

    comp_pass = compare_equal([input_tensor1], [golden_tensor])
    assert comp_pass


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 1, 320, 384])),
        (torch.Size([1, 3, 320, 384])),
    ),
)
@pytest.mark.parametrize(
    "scalar",
    {random.randint(-100, 100) + 0.5 for _ in range(5)},
)
def test_lei_ttnn(input_shapes, scalar, device):
    in_data, input_tensor = data_gen_with_range(input_shapes, -100, 100, device)
    ttnn.le_(input_tensor, scalar)
    golden_function = ttnn.get_golden_function(ttnn.le_)
    golden_tensor = golden_function(in_data, scalar)

    comp_pass = compare_equal([input_tensor], [golden_tensor])
    assert comp_pass


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 1, 320, 384])),
        (torch.Size([1, 3, 320, 384])),
    ),
)
def test_binary_eqi_ttnn(input_shapes, device):
    in_data1, input_tensor1 = data_gen_with_range(input_shapes, -100, 100, device)
    in_data2, input_tensor2 = data_gen_with_range(input_shapes, -150, 150, device)
    ttnn.eq_(input_tensor1, input_tensor2)
    golden_function = ttnn.get_golden_function(ttnn.eq_)
    golden_tensor = golden_function(in_data1, in_data2)

    comp_pass = compare_equal([input_tensor1], [golden_tensor])
    assert comp_pass


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 1, 320, 384])),
        (torch.Size([1, 3, 320, 384])),
    ),
)
@pytest.mark.parametrize(
    "scalar",
    {random.randint(-100, 100) + 0.5 for _ in range(5)},
)
def test_eqi_ttnn(input_shapes, scalar, device):
    in_data, input_tensor = data_gen_with_range(input_shapes, -100, 100, device)
    ttnn.eq_(input_tensor, scalar)
    golden_function = ttnn.get_golden_function(ttnn.eq_)
    golden_tensor = golden_function(in_data, scalar)

    comp_pass = compare_equal([input_tensor], [golden_tensor])
    assert comp_pass


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 1, 320, 384])),
        (torch.Size([1, 3, 320, 384])),
    ),
)
def test_binary_nei_ttnn(input_shapes, device):
    in_data1, input_tensor1 = data_gen_with_range(input_shapes, -100, 100, device)
    in_data2, input_tensor2 = data_gen_with_range(input_shapes, -150, 150, device)
    ttnn.ne_(input_tensor1, input_tensor2)
    golden_function = ttnn.get_golden_function(ttnn.ne_)
    golden_tensor = golden_function(in_data1, in_data2)

    comp_pass = compare_equal([input_tensor1], [golden_tensor])
    assert comp_pass


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 1, 320, 384])),
        (torch.Size([1, 3, 320, 384])),
    ),
)
@pytest.mark.parametrize(
    "scalar",
    {random.randint(-100, 100) + 0.5 for _ in range(5)},
)
def test_nei_ttnn(input_shapes, scalar, device):
    in_data, input_tensor = data_gen_with_range(input_shapes, -100, 100, device)
    ttnn.ne_(input_tensor, scalar)
    golden_function = ttnn.get_golden_function(ttnn.ne_)
    golden_tensor = golden_function(in_data, scalar)

    comp_pass = compare_equal([input_tensor], [golden_tensor])
    assert comp_pass
