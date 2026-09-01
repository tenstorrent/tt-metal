# SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import random

import pytest
import torch
import ttnn
from models.common.utility_functions import is_blackhole, is_slow_dispatch

from tests.tt_eager.python_api_testing.sweep_tests import (
    comparison_funcs,
)
from tests.ttnn.nightly.unit_tests.operations.eltwise.backward.utility_funcs import (
    compare_equal,
    compare_pcc,
    data_gen_with_range,
)
from tests.ttnn.utils_for_testing import (
    assert_div_by_zero_outputs,
    assert_with_pcc,
    assert_with_ulp,
)


def _data_gen_div_scalar_input(input_shapes, low, high, device, divisor):
    in_data1, input_tensor1 = data_gen_with_range(input_shapes, low, high, device)
    if divisor == 0.0:
        # Avoid 0/0: bf16 fast_and_approximate divide returns 0 instead of NaN (#43209).
        zero_mask = in_data1 == 0
        if zero_mask.any():
            in_data1 = in_data1.clone()
            in_data1[zero_mask] = 1.0
            input_tensor1 = ttnn.from_torch(
                in_data1, dtype=input_tensor1.dtype, layout=input_tensor1.layout, device=device
            )
    return in_data1, input_tensor1


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
    in_data1, input_tensor1 = data_gen_with_range(input_shapes, -100, 100, device, seed=0)
    in_data2, input_tensor2 = data_gen_with_range(input_shapes, -150, 150, device, seed=42)

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
    ((torch.Size([1, 1, 32, 32])),),
)
def test_binary_atan2_special_values(input_shapes, device):
    """Regression test: atan2(±inf, ±0) must return ±π/2 per IEEE 754.

    The kernel's both-zero rescue checked `min == 0`, which also fired when
    only |x| was zero and |y| was infinite, overwriting the correct π/2
    result with 0 (or π when x was -0).
    """
    y_vals = [float("inf"), float("-inf"), 1.0, -1.0, 0.0, float("inf"), float("-inf"), 2.5]
    x_vals = [0.0, 0.0, float("inf"), float("-inf"), -0.0, 2.5, -2.5, -0.0]

    torch_input_y = torch.tensor([y_vals] * 32, dtype=torch.float32)
    torch_input_x = torch.tensor([x_vals] * 32, dtype=torch.float32)

    golden_function = ttnn.get_golden_function(ttnn.atan2)
    golden_tensor = golden_function(torch_input_y, torch_input_x)

    tt_input_y = ttnn.from_torch(torch_input_y, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    tt_input_x = ttnn.from_torch(torch_input_x, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    output_tensor = ttnn.atan2(tt_input_y, tt_input_x)
    output_tensor = ttnn.to_torch(output_tensor)

    torch.testing.assert_close(output_tensor, golden_tensor)


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
    in_data1 = in_data1[:num_elements].reshape(input_shapes)
    in_data2 = torch.linspace(-150, 150, num_elements, dtype=torch.bfloat16)
    in_data2 = in_data2[:num_elements].reshape(input_shapes)

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
    output_tensor = ttnn.to_torch(output_tensor)
    golden_function = ttnn.get_golden_function(ttnn.logical_xor)
    golden_tensor = golden_function(in_data1, in_data2)

    assert torch.equal(output_tensor, golden_tensor)


@pytest.mark.parametrize("fast_and_approximate_mode", [True, False])
@pytest.mark.parametrize("rounding_mode", [None, "trunc", "floor"])
@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 1, 320, 384])),
        (torch.Size([1, 3, 320, 384])),
    ),
)
def test_binary_div_ttnn(fast_and_approximate_mode, rounding_mode, input_shapes, device):
    if fast_and_approximate_mode == True:  # If input_b is non-zero tensor (fast/approximate mode)
        in_data1, input_tensor1 = data_gen_with_range(input_shapes, -100, 100, device)
        in_data2, input_tensor2 = data_gen_with_range(input_shapes, -150, -1, device)
    else:
        in_data1, input_tensor1 = data_gen_with_range(input_shapes, -200, 150, device)
        in_data2, input_tensor2 = data_gen_with_range(input_shapes, -120, 200, device)

    output_tensor = ttnn.div(
        input_tensor1, input_tensor2, fast_and_approximate_mode=fast_and_approximate_mode, rounding_mode=rounding_mode
    )
    golden_function = ttnn.get_golden_function(ttnn.div)
    golden_tensor = golden_function(in_data1, in_data2, rounding_mode)

    comp_pass = compare_pcc([output_tensor], [golden_tensor])
    assert comp_pass


@pytest.mark.parametrize("fast_and_approximate_mode", [True, False])
@pytest.mark.parametrize("rounding_mode", [None, "trunc", "floor"])
@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 1, 320, 384])),
        (torch.Size([1, 3, 320, 384])),
    ),
)
def test_binary_div_ttnn_ci(fast_and_approximate_mode, rounding_mode, input_shapes, device):
    if fast_and_approximate_mode == True:  # If input_b is non-zero tensor (fast/approximate mode)
        in_data1, input_tensor1 = data_gen_with_range(input_shapes, -1e6, 1e6, device)
        in_data2, input_tensor2 = data_gen_with_range(input_shapes, -1e6, -1, device)
    else:
        in_data1, input_tensor1 = data_gen_with_range(input_shapes, -2e6, 1e6, device)
        in_data2, input_tensor2 = data_gen_with_range(input_shapes, -1e6, 2e6, device)

    output_tensor = ttnn.div(
        input_tensor1, input_tensor2, fast_and_approximate_mode=fast_and_approximate_mode, rounding_mode=rounding_mode
    )
    golden_function = ttnn.get_golden_function(ttnn.div)
    golden_tensor = golden_function(in_data1, in_data2, rounding_mode)
    output_tensor = ttnn.to_torch(output_tensor)

    comp_pass = comparison_funcs.comp_pcc(golden_tensor, output_tensor)
    assert comp_pass


@pytest.mark.parametrize("fast_and_approximate_mode", [True, False])
@pytest.mark.parametrize("rounding_mode", [None, "trunc", "floor"])
@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 1, 320, 384])),
        (torch.Size([1, 3, 320, 384])),
    ),
)
def test_binary_div_ttnn_opt(fast_and_approximate_mode, rounding_mode, input_shapes, device):
    if fast_and_approximate_mode == True:  # If input_b is non-zero tensor (fast/approximate mode)
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
        fast_and_approximate_mode=fast_and_approximate_mode,
        rounding_mode=rounding_mode,
        output_tensor=output_tensor,
        queue_id=cq_id,
    )
    golden_function = ttnn.get_golden_function(ttnn.div)
    golden_tensor = golden_function(in_data1, in_data2, rounding_mode)

    comp_pass = compare_pcc([output_tensor], [golden_tensor])
    assert comp_pass


@pytest.mark.parametrize("fast_and_approximate_mode", [True, False])
@pytest.mark.parametrize("rounding_mode", [None, "trunc", "floor"])
@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 1, 320, 384])),
        (torch.Size([1, 3, 320, 384])),
    ),
)
@pytest.mark.parametrize("value", [-5.1, 0.0, 10.9])
def test_binary_div_scalar_ttnn(fast_and_approximate_mode, rounding_mode, input_shapes, value, device):
    # Skip only rounding_mode=None + fast_and_approximate: trunc/floor of non-zero/0.0
    # always yields ±inf and is verifiable; rounding_mode=None returns 0 instead (#43209).
    if value == 0.0 and rounding_mode is None and fast_and_approximate_mode:
        pytest.skip(
            "Skipping test case due to division by zero not being handled properly in bfloat16 with rounding_mode=None and fast_and_approximate_mode=True"
        )
    in_data1, input_tensor1 = _data_gen_div_scalar_input(input_shapes, -100, 100, device, value)

    output_tensor = ttnn.div(
        input_tensor1, value, fast_and_approximate_mode=fast_and_approximate_mode, rounding_mode=rounding_mode
    )
    golden_function = ttnn.get_golden_function(ttnn.div)
    golden_tensor = golden_function(in_data1, value, rounding_mode)

    if value == 0.0:
        assert_div_by_zero_outputs(golden_tensor, ttnn.to_torch(output_tensor))
    else:
        comp_pass = compare_pcc([output_tensor], [golden_tensor])
        assert comp_pass


@pytest.mark.parametrize("fast_and_approximate_mode", [True, False])
@pytest.mark.parametrize("rounding_mode", [None, "trunc", "floor"])
@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 1, 320, 384])),
        (torch.Size([1, 3, 320, 384])),
    ),
)
@pytest.mark.parametrize("value", [-5.1, 0.0, 10.9])
def test_binary_div_scalar_ttnn_opt(fast_and_approximate_mode, rounding_mode, input_shapes, value, device):
    # Skip only rounding_mode=None + fast_and_approximate: trunc/floor of non-zero/0.0
    # always yields ±inf and is verifiable; rounding_mode=None returns 0 instead (#43209).
    if value == 0.0 and rounding_mode is None and fast_and_approximate_mode:
        pytest.skip(
            "Skipping test case due to division by zero not being handled properly in bfloat16 with rounding_mode=None and fast_and_approximate_mode=True"
        )
    in_data1, input_tensor1 = _data_gen_div_scalar_input(input_shapes, -100, 100, device, value)
    _, output_tensor = data_gen_with_range(input_shapes, -1, 1, device)

    cq_id = 0
    ttnn.div(
        input_tensor1,
        value,
        fast_and_approximate_mode=fast_and_approximate_mode,
        rounding_mode=rounding_mode,
        output_tensor=output_tensor,
    )
    golden_function = ttnn.get_golden_function(ttnn.div)
    golden_tensor = golden_function(in_data1, value, rounding_mode)

    if value == 0.0:
        assert_div_by_zero_outputs(golden_tensor, ttnn.to_torch(output_tensor))
    else:
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


@pytest.mark.parametrize("torch_dtype,ttnn_dtype", [(torch.float32, ttnn.float32), (torch.int32, ttnn.int32)])
def test_binary_floor_div_exact_multiples(torch_dtype, ttnn_dtype, device):
    values = torch.tensor([41, 82, 164, -41, -82, -164], dtype=torch_dtype)
    input_tensor = values.repeat(171)[:1024].reshape(1, 1, 32, 32)
    input_tensor = ttnn.from_torch(input_tensor, dtype=ttnn_dtype, layout=ttnn.TILE_LAYOUT, device=device)

    output_tensor = ttnn.floor_div(input_tensor, value=41)
    golden_tensor = torch.floor_divide(values.repeat(171)[:1024].reshape(1, 1, 32, 32), 41)

    assert torch.equal(ttnn.to_torch(output_tensor), golden_tensor)


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 1, 320, 384])),
        (torch.Size([1, 3, 320, 384])),
    ),
)
def test_binary_logical_and__ttnn(input_shapes, device):
    num_elements = max(int(torch.prod(torch.tensor(input_shapes)).item()), 1)
    in_data1 = torch.linspace(-150, 150, num_elements, dtype=torch.bfloat16)
    in_data1 = in_data1[:num_elements].reshape(input_shapes)
    in_data2 = torch.linspace(-100, 100, num_elements, dtype=torch.bfloat16)
    in_data2 = in_data2[:num_elements].reshape(input_shapes)

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

    ttnn.logical_and_(input_tensor1, input_tensor2)
    golden_function = ttnn.get_golden_function(ttnn.logical_and_)
    golden_tensor = golden_function(in_data1, in_data2)

    assert_with_ulp(input_tensor1, golden_tensor)
    assert torch.equal(ttnn.to_torch(input_tensor1), golden_tensor)


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
    in_data1 = in_data1[:num_elements].reshape(input_shapes)
    in_data2 = torch.linspace(-100, 100, num_elements, dtype=torch.bfloat16)
    in_data2 = in_data2[:num_elements].reshape(input_shapes)

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

    assert_with_ulp(input_tensor1, golden_tensor)
    assert torch.equal(ttnn.to_torch(input_tensor1), golden_tensor)


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
    in_data1 = in_data1[:num_elements].reshape(input_shapes)
    in_data2 = torch.linspace(-100, 100, num_elements, dtype=torch.bfloat16)
    in_data2 = in_data2[:num_elements].reshape(input_shapes)

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

    assert_with_ulp(input_tensor1, golden_tensor)
    assert torch.equal(ttnn.to_torch(input_tensor1), golden_tensor)


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
def test_gti_ttnn(input_shapes, device):
    for scalar in [random.randint(-100, 100) + 0.5 for _ in range(5)]:
        in_data, input_tensor = data_gen_with_range(input_shapes, -100, 100, device)
        ttnn.gt_(input_tensor, scalar)
        golden_function = ttnn.get_golden_function(ttnn.gt_)
        golden_tensor = golden_function(in_data, scalar)

        comp_pass = compare_equal([input_tensor], [golden_tensor])
        assert comp_pass, f"Failed for scalar={scalar}"


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
def test_gei_ttnn(input_shapes, device):
    for scalar in [random.randint(-100, 100) + 0.5 for _ in range(5)]:
        in_data, input_tensor = data_gen_with_range(input_shapes, -100, 100, device)
        ttnn.ge_(input_tensor, scalar)
        golden_function = ttnn.get_golden_function(ttnn.ge_)
        golden_tensor = golden_function(in_data, scalar)

        comp_pass = compare_equal([input_tensor], [golden_tensor])
        assert comp_pass, f"Failed for scalar={scalar}"


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
def test_lti_ttnn(input_shapes, device):
    for scalar in [random.randint(-100, 100) + 0.5 for _ in range(5)]:
        in_data, input_tensor = data_gen_with_range(input_shapes, -100, 100, device)
        ttnn.lt_(input_tensor, scalar)
        golden_function = ttnn.get_golden_function(ttnn.lt_)
        golden_tensor = golden_function(in_data, scalar)

        comp_pass = compare_equal([input_tensor], [golden_tensor])
        assert comp_pass, f"Failed for scalar={scalar}"


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
def test_lei_ttnn(input_shapes, device):
    for scalar in [random.randint(-100, 100) + 0.5 for _ in range(5)]:
        in_data, input_tensor = data_gen_with_range(input_shapes, -100, 100, device)
        ttnn.le_(input_tensor, scalar)
        golden_function = ttnn.get_golden_function(ttnn.le_)
        golden_tensor = golden_function(in_data, scalar)

        comp_pass = compare_equal([input_tensor], [golden_tensor])
        assert comp_pass, f"Failed for scalar={scalar}"


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
def test_eqi_ttnn(input_shapes, device):
    for scalar in [random.randint(-100, 100) + 0.5 for _ in range(5)]:
        in_data, input_tensor = data_gen_with_range(input_shapes, -100, 100, device)
        ttnn.eq_(input_tensor, scalar)
        golden_function = ttnn.get_golden_function(ttnn.eq_)
        golden_tensor = golden_function(in_data, scalar)

        comp_pass = compare_equal([input_tensor], [golden_tensor])
        assert comp_pass, f"Failed for scalar={scalar}"


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
def test_nei_ttnn(input_shapes, device):
    for scalar in [random.randint(-100, 100) + 0.5 for _ in range(5)]:
        in_data, input_tensor = data_gen_with_range(input_shapes, -100, 100, device)
        ttnn.ne_(input_tensor, scalar)
        golden_function = ttnn.get_golden_function(ttnn.ne_)
        golden_tensor = golden_function(in_data, scalar)

        comp_pass = compare_equal([input_tensor], [golden_tensor])
        assert comp_pass, f"Failed for scalar={scalar}"


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
    (-2.7, -0.25, 0.45, 6.4),
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
def test_unary_left_shift(input_shapes, device):
    for scalar in [random.randint(0, 31) for _ in range(5)]:
        torch.manual_seed(213919)
        in_data1 = torch.randint(-1000, 1000, input_shapes, dtype=torch.int32)
        input_tensor1 = ttnn.from_torch(in_data1, dtype=ttnn.int32, layout=ttnn.TILE_LAYOUT, device=device)

        output_tensor = ttnn.bitwise_left_shift(input_tensor1, scalar)
        golden_function = ttnn.get_golden_function(ttnn.bitwise_left_shift)
        golden_tensor = golden_function(in_data1, scalar)
        output_tensor = ttnn.to_torch(output_tensor)

        pcc = ttnn.pearson_correlation_coefficient(golden_tensor, output_tensor)
        assert pcc >= 0.99, f"Failed for scalar={scalar}"


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([64, 64])),
        (torch.Size([1, 1, 320, 384])),
        (torch.Size([1, 3, 320, 384])),
    ),
)
def test_unary_right_shift(input_shapes, device):
    for scalar in [random.randint(0, 31) for _ in range(5)]:
        torch.manual_seed(213919)
        in_data1 = torch.randint(-1000, 1000, input_shapes, dtype=torch.int32)
        input_tensor1 = ttnn.from_torch(in_data1, dtype=ttnn.int32, layout=ttnn.TILE_LAYOUT, device=device)

        output_tensor = ttnn.bitwise_right_shift(input_tensor1, scalar)
        golden_function = ttnn.get_golden_function(ttnn.bitwise_right_shift)
        golden_tensor = golden_function(in_data1, scalar)
        output_tensor = ttnn.to_torch(output_tensor)

        pcc = ttnn.pearson_correlation_coefficient(golden_tensor, output_tensor)
        assert pcc >= 0.99, f"Failed for scalar={scalar}"


# Kimi K3 betas: 4 for the gate half, 25 for the up half.
SITU_GLU_BETA1 = 4.0
SITU_GLU_BETA2 = 25.0

# The bf16 arm is gated in ULP (measured worst case: 3.0 across three composed ops). bfp8_b
# re-quantizes every intermediate and shares one exponent per 16-element block, which costs
# hundreds of bf16 ULP on small elements regardless of op accuracy, so that arm is gated by PCC.
SITU_GLU_ULP = 6
SITU_GLU_BF16_PCC = 0.999
SITU_GLU_BFP8_PCC = 0.99

# Numerics on both sides of the L1/DRAM intermediate split, also covering both dtypes. The
# assertions below check the output placement and the values, not which branch ran -- output
# placement is pinned to the input's for both.
SITU_GLU_CASES = [
    (torch.Size([1, 1, 512, 3072]), ttnn.bfloat16),  # K3 routed expert (3072) <= 3072 -> L1
    (torch.Size([1, 1, 512, 6144]), ttnn.bfloat8_b),  # K3 shared expert (6144) > 3072 -> DRAM
]


@pytest.mark.skipif(not is_blackhole(), reason="situ_glu builds on softcap, which is Blackhole only")
@pytest.mark.parametrize("input_shape, ttnn_dtype", SITU_GLU_CASES, ids=["hidden_le_3072", "hidden_gt_3072"])
def test_situ_glu(input_shape, ttnn_dtype, device):
    torch.manual_seed(0)
    # Span the saturating and near-linear regions of both halves.
    gate = torch.empty(input_shape, dtype=torch.bfloat16).uniform_(-30.0, 30.0)
    up = torch.empty(input_shape, dtype=torch.bfloat16).uniform_(-30.0, 30.0)

    gate_tt = ttnn.from_torch(gate, dtype=ttnn_dtype, layout=ttnn.TILE_LAYOUT, device=device)
    up_tt = ttnn.from_torch(up, dtype=ttnn_dtype, layout=ttnn.TILE_LAYOUT, device=device)

    out = ttnn.situ_glu(gate_tt, up_tt, SITU_GLU_BETA1, SITU_GLU_BETA2)
    # Output placement follows the input, not the possibly-L1 intermediates.
    assert out.memory_config().buffer_type == gate_tt.memory_config().buffer_type
    tt_res = ttnn.to_torch(out)
    golden = ttnn.get_golden_function(ttnn.situ_glu)(gate, up, beta1=SITU_GLU_BETA1, beta2=SITU_GLU_BETA2)

    is_bfp8 = ttnn_dtype == ttnn.bfloat8_b
    # Both halves are bounded: |situ_a| <= beta1, |up_half| <= beta2.
    bound = SITU_GLU_BETA1 * SITU_GLU_BETA2 * (1.0 + (5e-2 if is_bfp8 else 2**-8))
    max_abs = tt_res.to(torch.float32).abs().max().item()
    assert max_abs <= bound, f"situ_glu overshoot: max |out| {max_abs:.4f} > bound {bound:.4f}"

    if is_bfp8:
        assert_with_pcc(golden, tt_res, pcc=SITU_GLU_BFP8_PCC)
    else:
        assert_with_ulp(golden, tt_res, ulp_threshold=SITU_GLU_ULP)
        assert_with_pcc(golden, tt_res, pcc=SITU_GLU_BF16_PCC)


@pytest.mark.skipif(not is_blackhole(), reason="situ_glu builds on softcap, which is Blackhole only")
def test_situ_glu_l1_intermediates_fall_back(device):
    # 8192 tokens at the routed-expert width is ~48 MB per intermediate and three are live at the
    # peak, which does not fit L1. Forcing the L1 branch on hidden alone made this a hard
    # allocator failure instead of a DRAM fallback.
    shape = ttnn.Shape([1, 1, 8192, 3072])
    gate = ttnn.zeros(shape, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    up = ttnn.zeros(shape, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    out = ttnn.situ_glu(gate, up, SITU_GLU_BETA1, SITU_GLU_BETA2)

    assert out.memory_config().buffer_type == gate.memory_config().buffer_type
    gate.deallocate()
    up.deallocate()
    out.deallocate()


@pytest.mark.skipif(not is_blackhole(), reason="situ_glu builds on softcap, which is Blackhole only")
@pytest.mark.parametrize(
    "sub_core_grid",
    [
        ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 3))]),
        ttnn.CoreRangeSet(
            [
                ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(1, 4)),
                ttnn.CoreRange(ttnn.CoreCoord(3, 2), ttnn.CoreCoord(4, 3)),
            ]
        ),
    ],
    ids=["contiguous", "disjoint"],
)
def test_situ_glu_sub_core_grids(device, sub_core_grid):
    torch.manual_seed(0)
    # A width under the 3072 L1 cutoff, so this also pins the core restriction's other effect: it
    # holds the intermediates in the output's memory space instead of taking interleaved L1 on
    # every core, the restricted-away ones included.
    shape = torch.Size([1, 1, 512, 3072])
    gate = torch.empty(shape, dtype=torch.bfloat16).uniform_(-30.0, 30.0)
    up = torch.empty(shape, dtype=torch.bfloat16).uniform_(-30.0, 30.0)

    gate_tt = ttnn.from_torch(gate, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    up_tt = ttnn.from_torch(up, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    out = ttnn.situ_glu(gate_tt, up_tt, SITU_GLU_BETA1, SITU_GLU_BETA2, sub_core_grids=sub_core_grid)

    assert out.memory_config().buffer_type == gate_tt.memory_config().buffer_type
    tt_res = ttnn.to_torch(out)
    golden = ttnn.get_golden_function(ttnn.situ_glu)(gate, up, beta1=SITU_GLU_BETA1, beta2=SITU_GLU_BETA2)
    assert_with_ulp(golden, tt_res, ulp_threshold=SITU_GLU_ULP)
    assert_with_pcc(golden, tt_res, pcc=SITU_GLU_BF16_PCC)


@pytest.mark.skipif(not is_blackhole(), reason="situ_glu builds on softcap, which is Blackhole only")
def test_situ_glu_sub_core_grids_conflict(device, expect_error):
    shape = torch.Size([1, 1, 32, 32])
    gate = ttnn.zeros(shape, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    # The composed unaries take only sub_core_grids, so situ_glu resolves sub_device_id into it and
    # cannot honour both.
    with expect_error(RuntimeError, "Cannot specify both sub_core_grids and sub_device_id"):
        ttnn.situ_glu(
            gate,
            gate,
            SITU_GLU_BETA1,
            SITU_GLU_BETA2,
            sub_core_grids=ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 1))]),
            sub_device_id=ttnn.SubDeviceId(0),
        )


@pytest.mark.skipif(not is_blackhole(), reason="situ_glu builds on softcap, which is Blackhole only")
@pytest.mark.parametrize("via", ["memory_config", "input_placement"])
def test_situ_glu_sub_core_grids_rejects_interleaved_l1(device, expect_error, via):
    shape = torch.Size([1, 1, 32, 32])
    cores = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 1))])

    # An interleaved-L1 buffer takes L1 on every worker core, including the ones a core restriction
    # exists to stay off. It reaches the output placement two ways -- asked for, or inherited from an
    # interleaved-L1 input when memory_config is omitted -- so both have to be rejected.
    l1 = ttnn.L1_MEMORY_CONFIG
    gate = ttnn.zeros(
        shape,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=l1 if via == "input_placement" else ttnn.DRAM_MEMORY_CONFIG,
    )
    with expect_error(RuntimeError, "core restriction cannot be combined with an interleaved-L1 output"):
        ttnn.situ_glu(
            gate,
            gate,
            SITU_GLU_BETA1,
            SITU_GLU_BETA2,
            memory_config=l1 if via == "memory_config" else None,
            sub_core_grids=cores,
        )


@pytest.mark.skipif(not is_blackhole(), reason="situ_glu builds on softcap, which is Blackhole only")
@pytest.mark.skipif(is_slow_dispatch(), reason="sub-device managers are unsupported with slow dispatch")
def test_situ_glu_requires_cores_when_sub_devices_loaded(device, expect_error):
    shape = torch.Size([1, 1, 32, 32])
    grid = device.compute_with_storage_grid_size()
    first = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, 0))})
    rest = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 1), ttnn.CoreCoord(grid.x - 1, grid.y - 1))})
    manager = device.create_sub_device_manager([ttnn.SubDevice([first]), ttnn.SubDevice([rest])], 0)
    device.load_sub_device_manager(manager)
    try:
        gate = ttnn.zeros(shape, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        # Unrestricted, the composed ops would take sub-device 0 -- the first strip -- instead of the
        # full grid, with no error to show it. Make the caller name the cores once the grid is split.
        with expect_error(RuntimeError, "sub-devices are loaded"):
            ttnn.situ_glu(gate, gate, SITU_GLU_BETA1, SITU_GLU_BETA2)
    finally:
        device.clear_loaded_sub_device_manager()
        device.remove_sub_device_manager(manager)


@pytest.mark.skipif(not is_blackhole(), reason="situ_glu builds on softcap, which is Blackhole only")
def test_situ_glu_zero_beta_guard(device, expect_error):
    shape = torch.Size([1, 1, 32, 32])
    gate = ttnn.from_torch(
        torch.zeros(shape, dtype=torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
    )

    # Both betas are inverted before reaching the SFPU, so neither may be zero.
    for beta1, beta2 in [(0.0, SITU_GLU_BETA2), (SITU_GLU_BETA1, 0.0)]:
        with expect_error(RuntimeError, "beta1 and beta2 must be non-zero"):
            ttnn.situ_glu(gate, gate, beta1, beta2)
