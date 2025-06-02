# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import random
import ttnn
from tests.ttnn.unit_tests.operations.eltwise.backward.utility_funcs import (
    data_gen_with_range,
    data_gen_with_range_int,
    compare_pcc,
    compare_equal,
)
from tests.ttnn.utils_for_testing import assert_with_pcc
from tests.tt_eager.python_api_testing.sweep_tests import (
    comparison_funcs,
)


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
    num_elements = max(int(torch.prod(torch.tensor(input_shapes)).item()), 1)
    in_data1 = torch.linspace(-100, 100, num_elements, dtype=torch.bfloat16)
    in_data1 = in_data1[:num_elements].reshape(input_shapes).nan_to_num(0.0)
    in_data2 = torch.linspace(-150, 150, num_elements, dtype=torch.bfloat16)
    in_data2 = in_data2[:num_elements].reshape(input_shapes).nan_to_num(0.0)

    input_tensor1 = ttnn.from_torch(
        in_data1,
        dtype=ttnn.bfloat16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    input_tensor2 = ttnn.from_torch(
        in_data2,
        dtype=ttnn.bfloat16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

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
@pytest.mark.parametrize("round_mode", [None, "trunc", "floor"])
@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 1, 320, 384])),
        (torch.Size([1, 3, 320, 384])),
    ),
)
def test_binary_div_ttnn(accurate_mode, round_mode, input_shapes, device):
    if accurate_mode == False:  # If input_b is non-zero tensor
        in_data1, input_tensor1 = data_gen_with_range(input_shapes, -100, 100, device)
        in_data2, input_tensor2 = data_gen_with_range(input_shapes, -150, -1, device)
    else:
        in_data1, input_tensor1 = data_gen_with_range(input_shapes, -200, 150, device)
        in_data2, input_tensor2 = data_gen_with_range(input_shapes, -120, 200, device)

    output_tensor = ttnn.div(input_tensor1, input_tensor2, accurate_mode=accurate_mode, round_mode=round_mode)
    golden_function = ttnn.get_golden_function(ttnn.div)
    golden_tensor = golden_function(in_data1, in_data2, round_mode)

    comp_pass = compare_pcc([output_tensor], [golden_tensor])
    assert comp_pass


@pytest.mark.parametrize("accurate_mode", [False, True])
@pytest.mark.parametrize("round_mode", [None, "trunc", "floor"])
@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 1, 320, 384])),
        (torch.Size([1, 3, 320, 384])),
    ),
)
def test_binary_div_ttnn_ci(accurate_mode, round_mode, input_shapes, device):
    if accurate_mode == False:  # If input_b is non-zero tensor
        in_data1, input_tensor1 = data_gen_with_range(input_shapes, -1e6, 1e6, device)
        in_data2, input_tensor2 = data_gen_with_range(input_shapes, -1e6, -1, device)
    else:
        in_data1, input_tensor1 = data_gen_with_range(input_shapes, -2e6, 1e6, device)
        in_data2, input_tensor2 = data_gen_with_range(input_shapes, -1e6, 2e6, device)

    output_tensor = ttnn.div(input_tensor1, input_tensor2, accurate_mode=accurate_mode, round_mode=round_mode)
    golden_function = ttnn.get_golden_function(ttnn.div)
    golden_tensor = golden_function(in_data1, in_data2, round_mode)
    output_tensor = ttnn.to_torch(output_tensor)

    comp_pass = comparison_funcs.comp_pcc(golden_tensor, output_tensor)
    assert comp_pass


@pytest.mark.parametrize("accurate_mode", [False, True])
@pytest.mark.parametrize("round_mode", [None, "trunc", "floor"])
@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 1, 320, 384])),
        (torch.Size([1, 3, 320, 384])),
    ),
)
def test_binary_div_ttnn_opt(accurate_mode, round_mode, input_shapes, device):
    if accurate_mode == False:  # If input_b is non-zero tensor
        in_data1, input_tensor1 = data_gen_with_range(input_shapes, -100, 100, device)
        in_data2, input_tensor2 = data_gen_with_range(input_shapes, -150, -1, device)
    else:
        in_data1, input_tensor1 = data_gen_with_range(input_shapes, -200, 100, device)
        in_data2, input_tensor2 = data_gen_with_range(input_shapes, -150, 200, device)

    _, output_tensor = data_gen_with_range(input_shapes, -1, 1, device)

    cq_id = 0
    ttnn.div(
        input_tensor1,
        input_tensor2,
        accurate_mode=accurate_mode,
        round_mode=round_mode,
        output_tensor=output_tensor,
        queue_id=cq_id,
    )
    golden_function = ttnn.get_golden_function(ttnn.div)
    golden_tensor = golden_function(in_data1, in_data2, round_mode)

    comp_pass = compare_pcc([output_tensor], [golden_tensor])
    assert comp_pass


@pytest.mark.parametrize("accurate_mode", [False, True])
@pytest.mark.parametrize("round_mode", [None, "trunc", "floor"])
@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 1, 320, 384])),
        (torch.Size([1, 3, 320, 384])),
    ),
)
@pytest.mark.parametrize("value", [-5.1, 0.0, 10.9])
def test_binary_div_scalar_ttnn(accurate_mode, round_mode, input_shapes, value, device):
    in_data1, input_tensor1 = data_gen_with_range(input_shapes, -100, 100, device)

    output_tensor = ttnn.div(input_tensor1, value, accurate_mode=accurate_mode, round_mode=round_mode)
    golden_function = ttnn.get_golden_function(ttnn.div)
    golden_tensor = golden_function(in_data1, value, round_mode)

    comp_pass = compare_pcc([output_tensor], [golden_tensor])
    assert comp_pass


@pytest.mark.parametrize("accurate_mode", [False, True])
@pytest.mark.parametrize("round_mode", [None, "trunc", "floor"])
@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 1, 320, 384])),
        (torch.Size([1, 3, 320, 384])),
    ),
)
@pytest.mark.parametrize("value", [-5.1, 0.0, 10.9])
def test_binary_div_scalar_ttnn_opt(accurate_mode, round_mode, input_shapes, value, device):
    in_data1, input_tensor1 = data_gen_with_range(input_shapes, -100, 100, device)
    _, output_tensor = data_gen_with_range(input_shapes, -1, 1, device)

    cq_id = 0
    ttnn.div(input_tensor1, value, accurate_mode=accurate_mode, round_mode=round_mode, output_tensor=output_tensor)
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
def test_binary_remainder_ttnn(input_shapes, device):
    in_data1, input_tensor1 = data_gen_with_range(input_shapes, -150, 150, device)
    in_data2, input_tensor2 = data_gen_with_range(input_shapes, -100, 100, device)
    output_tensor = ttnn.remainder(input_tensor1, input_tensor2)
    golden_function = ttnn.get_golden_function(ttnn.remainder)
    golden_tensor = golden_function(in_data1, in_data2, device=device)

    comp_pass = compare_pcc([output_tensor], [golden_tensor])
    assert comp_pass


@pytest.mark.parametrize(
    "shapes",
    [
        [[1, 1, 3, 3], [1, 1, 3, 3]],
    ],
)
def test_shape_remainder(device, shapes):
    torch.manual_seed(0)
    high = 10
    low = -10

    torch_input_tensor_a = torch.rand(shapes[0], dtype=torch.bfloat16) * (high - low) + low

    high = 9
    low = -9
    torch_input_tensor_b = torch.rand(shapes[1], dtype=torch.bfloat16) * (high - low) + low

    golden_function = ttnn.get_golden_function(ttnn.remainder)
    torch_output_tensor = golden_function(torch_input_tensor_a, torch_input_tensor_b, device=device)

    input_tensor_a = ttnn.from_torch(
        torch_input_tensor_a, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    input_tensor_b = ttnn.from_torch(
        torch_input_tensor_b, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )

    output_tensor = ttnn.remainder(input_tensor_a, input_tensor_b, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    output_tensor = ttnn.to_torch(output_tensor)

    assert ttnn.pearson_correlation_coefficient(torch_output_tensor, output_tensor) >= 0.999


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
def test_remainder_ttnn(input_shapes, scalar, device):
    in_data1, input_tensor1 = data_gen_with_range(input_shapes, -150, 150, device)
    output_tensor = ttnn.remainder(input_tensor1, scalar)
    golden_function = ttnn.get_golden_function(ttnn.remainder)
    golden_tensor = golden_function(in_data1, scalar, device=device)

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
def test_binary_fmod_ttnn(input_shapes, device):
    in_data1, input_tensor1 = data_gen_with_range(input_shapes, -150, 150, device)
    in_data2, input_tensor2 = data_gen_with_range(input_shapes, -100, 100, device)

    output_tensor = ttnn.fmod(input_tensor1, input_tensor2)
    golden_function = ttnn.get_golden_function(ttnn.fmod)
    golden_tensor = golden_function(in_data1, in_data2, device=device)

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
# Input with more than two decimal places experience precision loss in bfloat16. use FP32 for better precision.
def test_binary_fmod_decimal_ttnn(input_shapes, device):
    in_data1 = torch.randn(input_shapes, dtype=torch.float32) * 9
    input_tensor1 = ttnn.Tensor(in_data1, ttnn.float32).to(ttnn.TILE_LAYOUT).to(device)
    in_data2 = torch.rand(input_shapes, dtype=torch.float32) - 2
    input_tensor2 = ttnn.Tensor(in_data2, ttnn.float32).to(ttnn.TILE_LAYOUT).to(device)
    output_tensor = ttnn.fmod(input_tensor1, input_tensor2)
    golden_function = ttnn.get_golden_function(ttnn.fmod)
    golden_tensor = golden_function(in_data1, in_data2, device=device)

    comp_pass = compare_pcc([output_tensor], [golden_tensor], 0.9999)
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
def test_fmod_ttnn(input_shapes, scalar, device):
    in_data1, input_tensor1 = data_gen_with_range(input_shapes, -150, 150, device)

    output_tensor = ttnn.fmod(input_tensor1, scalar)
    golden_function = ttnn.get_golden_function(ttnn.fmod)
    golden_tensor = golden_function(in_data1, scalar, device=device)

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
    num_elements = max(int(torch.prod(torch.tensor(input_shapes)).item()), 1)
    in_data1 = torch.linspace(-150, 150, num_elements, dtype=torch.bfloat16)
    in_data1 = in_data1[:num_elements].reshape(input_shapes).nan_to_num(0.0)
    in_data2 = torch.linspace(-100, 100, num_elements, dtype=torch.bfloat16)
    in_data2 = in_data2[:num_elements].reshape(input_shapes).nan_to_num(0.0)

    input_tensor1 = ttnn.from_torch(
        in_data1,
        dtype=ttnn.bfloat16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    input_tensor2 = ttnn.from_torch(
        in_data2,
        dtype=ttnn.bfloat16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

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
    num_elements = max(int(torch.prod(torch.tensor(input_shapes)).item()), 1)
    in_data1 = torch.linspace(-150, 150, num_elements, dtype=torch.bfloat16)
    in_data1 = in_data1[:num_elements].reshape(input_shapes).nan_to_num(0.0)
    in_data2 = torch.linspace(-100, 100, num_elements, dtype=torch.bfloat16)
    in_data2 = in_data2[:num_elements].reshape(input_shapes).nan_to_num(0.0)

    input_tensor1 = ttnn.from_torch(
        in_data1,
        dtype=ttnn.bfloat16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    input_tensor2 = ttnn.from_torch(
        in_data2,
        dtype=ttnn.bfloat16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

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


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 2, 32, 64, 64])),
        (torch.Size([1, 3, 7, 29, 127])),
        (torch.Size([1, 3, 2, 32])),
        (torch.Size([1, 6, 49, 97])),
        (torch.Size([1, 7, 320])),
        (torch.Size([1, 49, 321])),
        (torch.Size([4, 32])),
        (torch.Size([49, 321])),
    ),
)
def test_binary_prelu_ttnn(input_shapes, device):
    in_data1 = torch.rand(input_shapes, dtype=torch.bfloat16) * 200 - 100
    channels = input_shapes[1]
    in_data2 = torch.rand((channels,), dtype=torch.bfloat16) * 200 - 100

    input_tensor1 = ttnn.from_torch(in_data1, layout=ttnn.TILE_LAYOUT, device=device)
    input_tensor2 = ttnn.from_torch(in_data2, layout=ttnn.TILE_LAYOUT, device=device)

    output_tensor = ttnn.prelu(input_tensor1, input_tensor2)
    output_tensor = ttnn.to_torch(output_tensor)
    golden_function = ttnn.get_golden_function(ttnn.prelu)
    golden_tensor = golden_function(in_data1, in_data2)

    assert_with_pcc(golden_tensor, output_tensor, 0.999)


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 2, 32, 64, 64])),
        (torch.Size([1, 3, 7, 29, 127])),
        (torch.Size([1, 3, 2, 32])),
        (torch.Size([1, 6, 49, 97])),
        (torch.Size([1, 7, 320])),
        (torch.Size([1, 49, 321])),
        (torch.Size([4, 32])),
        (torch.Size([49, 321])),
    ),
)
@pytest.mark.parametrize(
    "scalar",
    {-0.25, -2.7, 0.45, 6.4},
)
def test_binary_prelu_scalar_ttnn(input_shapes, scalar, device):
    in_data1 = torch.rand(input_shapes, dtype=torch.bfloat16) * 200 - 100
    input_tensor1 = ttnn.from_torch(in_data1, layout=ttnn.TILE_LAYOUT, device=device)

    output_tensor = ttnn.prelu(input_tensor1, scalar)
    output_tensor = ttnn.to_torch(output_tensor)
    golden_function = ttnn.get_golden_function(ttnn.prelu)
    golden_tensor = golden_function(in_data1, scalar)

    assert_with_pcc(golden_tensor, output_tensor, 0.999)


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 2, 32, 64, 64])),
        (torch.Size([1, 3, 7, 29, 127])),
        (torch.Size([1, 3, 2, 32])),
        (torch.Size([1, 6, 49, 97])),
        (torch.Size([1, 7, 320])),
        (torch.Size([1, 49, 321])),
        (torch.Size([4, 32])),
        (torch.Size([49, 321])),
    ),
)
@pytest.mark.parametrize(
    "weight",
    [
        [-0.25],
        [-2.7],
        [0.45],
        [6.4],
        [2],
        [-1],
    ],
)
def test_binary_prelu_1D_weight(input_shapes, weight, device):
    in_data1 = torch.rand(input_shapes, dtype=torch.bfloat16) * 200 - 100
    input_tensor1 = ttnn.from_torch(in_data1, layout=ttnn.TILE_LAYOUT, device=device)

    output_tensor = ttnn.prelu(input_tensor1, weight)
    output_tensor = ttnn.to_torch(output_tensor)
    golden_function = ttnn.get_golden_function(ttnn.prelu)
    golden_tensor = golden_function(in_data1, weight)

    assert_with_pcc(golden_tensor, output_tensor, 0.999)


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([64, 64])),
        (torch.Size([1, 1, 320, 384])),
        (torch.Size([1, 3, 320, 384])),
    ),
)
def test_binary_left_shift(input_shapes, device):
    torch.manual_seed(213919)
    in_data1 = torch.randint(-1000, 1000, input_shapes, dtype=torch.int32)
    in_data2 = torch.randint(-20, 50, input_shapes, dtype=torch.int32)
    input_tensor1 = ttnn.from_torch(in_data1, dtype=ttnn.int32, layout=ttnn.TILE_LAYOUT, device=device)
    input_tensor2 = ttnn.from_torch(in_data2, dtype=ttnn.int32, layout=ttnn.TILE_LAYOUT, device=device)

    output_tensor = ttnn.bitwise_left_shift(input_tensor1, input_tensor2)
    golden_function = ttnn.get_golden_function(ttnn.bitwise_left_shift)
    golden_tensor = golden_function(in_data1, in_data2)
    output_tensor = ttnn.to_torch(output_tensor)

    pcc = ttnn.pearson_correlation_coefficient(golden_tensor, output_tensor)
    assert pcc >= 0.99


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([64, 64])),
        (torch.Size([1, 1, 320, 384])),
        (torch.Size([1, 3, 320, 384])),
    ),
)
def test_binary_right_shift(input_shapes, device):
    torch.manual_seed(213919)
    in_data1 = torch.randint(-1000, 1000, input_shapes, dtype=torch.int32)
    in_data2 = torch.randint(0, 31, input_shapes, dtype=torch.int32)
    input_tensor1 = ttnn.from_torch(in_data1, dtype=ttnn.int32, layout=ttnn.TILE_LAYOUT, device=device)
    input_tensor2 = ttnn.from_torch(in_data2, dtype=ttnn.int32, layout=ttnn.TILE_LAYOUT, device=device)

    output_tensor = ttnn.bitwise_right_shift(input_tensor1, input_tensor2)
    golden_function = ttnn.get_golden_function(ttnn.bitwise_right_shift)
    golden_tensor = golden_function(in_data1, in_data2)
    output_tensor = ttnn.to_torch(output_tensor)

    pcc = ttnn.pearson_correlation_coefficient(golden_tensor, output_tensor)
    assert pcc >= 0.99


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([64, 64])),
        (torch.Size([1, 1, 320, 384])),
        (torch.Size([1, 3, 320, 384])),
    ),
)
@pytest.mark.parametrize(
    "scalar",
    {random.randint(0, 31)},
)
def test_unary_left_shift(input_shapes, device, scalar):
    torch.manual_seed(213919)
    in_data1 = torch.randint(-1000, 1000, input_shapes, dtype=torch.int32)
    input_tensor1 = ttnn.from_torch(in_data1, dtype=ttnn.int32, layout=ttnn.TILE_LAYOUT, device=device)

    output_tensor = ttnn.bitwise_left_shift(input_tensor1, scalar)
    golden_function = ttnn.get_golden_function(ttnn.bitwise_left_shift)
    golden_tensor = golden_function(in_data1, scalar)
    output_tensor = ttnn.to_torch(output_tensor)

    pcc = ttnn.pearson_correlation_coefficient(golden_tensor, output_tensor)
    assert pcc >= 0.99


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([64, 64])),
        (torch.Size([1, 1, 320, 384])),
        (torch.Size([1, 3, 320, 384])),
    ),
)
@pytest.mark.parametrize(
    "scalar",
    {random.randint(0, 31)},
)
def test_unary_right_shift(input_shapes, device, scalar):
    torch.manual_seed(213919)
    in_data1 = torch.randint(-1000, 1000, input_shapes, dtype=torch.int32)
    input_tensor1 = ttnn.from_torch(in_data1, dtype=ttnn.int32, layout=ttnn.TILE_LAYOUT, device=device)

    output_tensor = ttnn.bitwise_right_shift(input_tensor1, scalar)
    golden_function = ttnn.get_golden_function(ttnn.bitwise_right_shift)
    golden_tensor = golden_function(in_data1, scalar)
    output_tensor = ttnn.to_torch(output_tensor)

    pcc = ttnn.pearson_correlation_coefficient(golden_tensor, output_tensor)
    assert pcc >= 0.99
