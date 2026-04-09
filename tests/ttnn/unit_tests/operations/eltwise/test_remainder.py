# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import ttnn
from models.common.utility_functions import torch_random
from functools import partial
from tests.tt_eager.python_api_testing.sweep_tests.generation_funcs import gen_func_with_cast_tt
from tests.ttnn.utils_for_testing import assert_with_ulp

pytestmark = pytest.mark.use_module_device


@pytest.mark.parametrize(
    "input_shapes",
    ((torch.Size([1, 1, 32, 32])),),
)
def test_broken_remainder(input_shapes, device):
    torch_lhs = torch.ones(32, 32, dtype=torch.bfloat16)
    torch_rhs = torch.zeros(32, 32, dtype=torch.bfloat16)

    golden_function = ttnn.get_golden_function(ttnn.remainder)
    golden = golden_function(torch_lhs, torch_rhs, device=device)

    tt_lhs = ttnn.from_torch(torch_lhs, layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.bfloat16)
    tt_rhs = ttnn.from_torch(torch_rhs, layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.bfloat16)
    tt_result = ttnn.remainder(tt_lhs, tt_rhs)
    output_tensor = ttnn.to_torch(tt_result)

    # Handle special case where TT returns -inf but PyTorch returns nan for remainder with zero divisor
    assert torch.all(torch.isinf(output_tensor))
    assert torch.all(torch.isnan(golden))


@pytest.mark.parametrize(
    "input_shapes",
    ((torch.Size([1, 1, 32, 32])),),
)
def test_broken_remainder1(input_shapes, device):
    torch_lhs = torch.ones(32, 32, dtype=torch.bfloat16) * 95
    torch_rhs = torch.ones(32, 32, dtype=torch.bfloat16) * (-94.5)

    golden_function = ttnn.get_golden_function(ttnn.remainder)  # all -94.0
    golden = golden_function(torch_lhs, torch_rhs, device=device)

    tt_lhs = ttnn.from_torch(torch_lhs, layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.bfloat16)
    tt_rhs = ttnn.from_torch(torch_rhs, layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.bfloat16)

    tt_result = ttnn.remainder(tt_lhs, tt_rhs)
    result = ttnn.to_torch(tt_result)  # all 0.5
    assert torch.allclose(result, golden, atol=0.01, rtol=0)


# This test was added for #17361
# If input is a multiple of the scalar, the result should be 0, but both Torch and TT output either 0 or the scalar value itself depending on the operands.
# This inconsistency is persistent due to some fp precision loss in both Torch and TT.
# Eg: torch.remainder of (3, 1.5) = 0.0 and of (3, 0.003) = 0.003
# Eg: ttnn.remainder of (4, 0.004) = 0.004 and of (3, 0.003) = 0.0
@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([6, 5, 320, 320])),
        (torch.Size([2, 1, 384, 320])),
        (torch.Size([3, 123, 115])),
        (torch.Size([69, 178])),
        (torch.Size([1024])),
        (torch.Size([])),
    ),
)
@pytest.mark.parametrize("scalar", [-0.002, -0.001, -0.0006, -0.0003, 0.0, 0.0005, 0.0007, 0.001, 0.002])
def test_remainder_scalar(input_shapes, scalar, device):
    torch.manual_seed(0)
    if len(input_shapes) == 0:
        torch_input_tensor = torch.tensor(5.0, dtype=torch.bfloat16)
    else:
        torch_input_tensor = gen_func_with_cast_tt(
            partial(torch_random, low=-100, high=100, dtype=torch.bfloat16), ttnn.bfloat16
        )(input_shapes)
    input_tensor = ttnn.from_torch(
        torch_input_tensor,
        dtype=ttnn.bfloat16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    golden_function = ttnn.get_golden_function(ttnn.remainder)
    torch_output_tensor = golden_function(torch_input_tensor, scalar, device=device)

    output_tensor = ttnn.remainder(input_tensor, scalar)
    output_tensor = ttnn.to_torch(output_tensor)

    # Handle special case where TT returns -inf but PyTorch returns nan for fmod with zero divisor
    if scalar == 0.0:
        output_tensor = torch.where(
            torch.isinf(output_tensor), torch.tensor(float("nan"), dtype=output_tensor.dtype), output_tensor
        )
        assert torch.allclose(output_tensor, torch_output_tensor, equal_nan=True)
    else:
        assert torch.allclose(output_tensor, torch_output_tensor, atol=0.001, rtol=0)


@pytest.mark.parametrize(
    "testing_dtype",
    ["bfloat16", "float32"],
)
def test_remainder_nan(testing_dtype, device):
    torch_dtype = getattr(torch, testing_dtype)
    ttnn_dtype = getattr(ttnn, testing_dtype)
    if testing_dtype == "bfloat16":
        pytest.xfail("NaN is packed as inf for ttnn.bfloat16")

    torch_input_a = torch.tensor([1.0, 0.0, -1.0], dtype=torch_dtype)
    torch_input_b = torch.tensor([0.0, 0.0, 0.0], dtype=torch_dtype)

    golden_function = ttnn.get_golden_function(ttnn.remainder)
    golden = golden_function(torch_input_a, torch_input_b, device=device)

    tt_in_a = ttnn.from_torch(
        torch_input_a,
        dtype=ttnn_dtype,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    tt_in_b = ttnn.from_torch(
        torch_input_b,
        dtype=ttnn_dtype,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    tt_result = ttnn.remainder(tt_in_a, tt_in_b)
    output_tensor = ttnn.to_torch(tt_result)

    assert torch.equal(torch.isnan(golden), torch.isnan(output_tensor))


@pytest.mark.parametrize("dtype", ["bfloat16", "float32"])
def test_remainder_binary_accuracy(device, dtype):
    """Test remainder binary operation with specific values."""
    torch_dtype = getattr(torch, dtype)
    ttnn_dtype = getattr(ttnn, dtype)

    torch_input_a = torch.tensor([[5.0, 7.0, -5.0, -7.0, 3.5, 10.0, 1.5, -1.5, 9.0, 15.0]], dtype=torch_dtype)
    torch_input_b = torch.tensor([[2.0, 4.0, 2.0, 4.0, 2.0, 4.0, 0.5, 0.5, -2.0, -4.0]], dtype=torch_dtype)

    golden_fn = ttnn.get_golden_function(ttnn.remainder)
    golden = golden_fn(torch_input_a, torch_input_b, device=device)

    input_tensor_a = ttnn.from_torch(torch_input_a, dtype=ttnn_dtype, layout=ttnn.TILE_LAYOUT, device=device)
    input_tensor_b = ttnn.from_torch(torch_input_b, dtype=ttnn_dtype, layout=ttnn.TILE_LAYOUT, device=device)

    output = ttnn.remainder(input_tensor_a, input_tensor_b)
    output = ttnn.to_torch(output)
    assert_with_ulp(golden, output, 1)
