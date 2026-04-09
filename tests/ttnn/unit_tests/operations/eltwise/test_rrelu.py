# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import ttnn
from tests.ttnn.utils_for_testing import assert_allclose
from tests.ttnn.nightly.unit_tests.operations.eltwise.backward.utility_funcs import (
    data_gen_with_range,
    compare_pcc,
)

pytestmark = pytest.mark.use_module_device


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 1, 320, 384])),
        (torch.Size([1, 3, 320, 384])),
    ),
)
def test_rrelu_default_params(input_shapes, device):
    """Test rrelu with default parameters (lower=0.125, upper=1/3, eval mode)."""
    in_data, input_tensor = data_gen_with_range(input_shapes, -10, 10, device)

    output_tensor = ttnn.rrelu(input_tensor)
    golden_tensor = torch.nn.functional.rrelu(in_data, lower=0.125, upper=1.0 / 3.0, training=False)

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
    "lower,upper",
    [
        (0.125, 1.0 / 3.0),  # default
        (0.01, 0.1),  # small slopes
        (0.2, 0.5),  # larger slopes
        (0.0, 0.0),  # zero slope (ReLU behavior)
        (1.0, 1.0),  # identity for negatives
    ],
)
def test_rrelu_eval_mode(input_shapes, lower, upper, device):
    """Test rrelu in eval mode with various lower/upper bounds.

    In eval mode, the slope is fixed: a = (lower + upper) / 2.
    """
    in_data, input_tensor = data_gen_with_range(input_shapes, -10, 10, device)

    output_tensor = ttnn.rrelu(input_tensor, lower=lower, upper=upper)
    golden_tensor = torch.nn.functional.rrelu(in_data, lower=lower, upper=upper, training=False)

    assert_allclose(golden_tensor, output_tensor, rtol=1.6e-2, atol=1e-2)


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 1, 320, 384])),
    ),
)
def test_rrelu_all_positive(input_shapes, device):
    """Test rrelu with all-positive input (should be identity)."""
    in_data, input_tensor = data_gen_with_range(input_shapes, 0.1, 10, device)

    output_tensor = ttnn.rrelu(input_tensor)
    golden_tensor = torch.nn.functional.rrelu(in_data, lower=0.125, upper=1.0 / 3.0, training=False)

    assert_allclose(golden_tensor, output_tensor, rtol=1.6e-2, atol=1e-2)


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 1, 320, 384])),
    ),
)
def test_rrelu_all_negative(input_shapes, device):
    """Test rrelu with all-negative input (exercises the slope path)."""
    in_data, input_tensor = data_gen_with_range(input_shapes, -10, -0.1, device)

    output_tensor = ttnn.rrelu(input_tensor, lower=0.125, upper=1.0 / 3.0)
    golden_tensor = torch.nn.functional.rrelu(in_data, lower=0.125, upper=1.0 / 3.0, training=False)

    assert_allclose(golden_tensor, output_tensor, rtol=1.6e-2, atol=1e-2)


@pytest.mark.parametrize(
    "input_shapes",
    ((torch.Size([1, 1, 32, 32])),),
)
def test_rrelu_output_tensor(input_shapes, device):
    """Test rrelu with preallocated output tensor."""
    in_data, input_tensor = data_gen_with_range(input_shapes, -10, 10, device)
    _, output_tensor = data_gen_with_range(input_shapes, -1, 1, device)

    ttnn.rrelu(input_tensor, lower=0.125, upper=1.0 / 3.0, output_tensor=output_tensor)
    golden_tensor = torch.nn.functional.rrelu(in_data, lower=0.125, upper=1.0 / 3.0, training=False)

    assert_allclose(golden_tensor, output_tensor, rtol=1.6e-2, atol=1e-2)
