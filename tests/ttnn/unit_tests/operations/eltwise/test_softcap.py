# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import ttnn
from tests.ttnn.utils_for_testing import assert_allclose, assert_with_ulp
from tests.ttnn.nightly.unit_tests.operations.eltwise.backward.utility_funcs import (
    data_gen_with_range,
)

pytestmark = pytest.mark.use_module_device


def golden_softcap(x, cap):
    return cap * torch.tanh(x / cap)


@pytest.mark.parametrize(
    "input_shapes",
    [
        torch.Size([1, 1, 32, 32]),
        torch.Size([1, 1, 320, 384]),
        torch.Size([1, 3, 320, 384]),
    ],
)
@pytest.mark.parametrize("cap", [1.0, 10.0, 50.0])
def test_softcap_bfloat16(input_shapes, cap, device):
    in_data, input_tensor = data_gen_with_range(input_shapes, -100, 100, device)
    output_tensor = ttnn.softcap(input_tensor, cap=cap)
    golden_tensor = golden_softcap(in_data, cap)

    assert_with_ulp(golden_tensor, output_tensor, ulp_threshold=10)
    assert_allclose(golden_tensor, output_tensor, rtol=5e-2, atol=0.35)


@pytest.mark.parametrize(
    "input_shapes",
    [
        torch.Size([1, 1, 32, 32]),
        torch.Size([1, 1, 320, 384]),
    ],
)
@pytest.mark.parametrize("cap", [1.0, 50.0])
def test_softcap_default_cap(input_shapes, cap, device):
    """Test that cap=50.0 matches the default."""
    in_data, input_tensor = data_gen_with_range(input_shapes, -100, 100, device)
    output_tensor = ttnn.softcap(input_tensor, cap=50.0)
    golden_tensor = golden_softcap(in_data, 50.0)

    assert_with_ulp(golden_tensor, output_tensor, ulp_threshold=10)


@pytest.mark.parametrize(
    "input_shapes",
    [
        torch.Size([1, 1, 32, 32]),
        torch.Size([1, 1, 320, 384]),
    ],
)
@pytest.mark.parametrize("cap", [1.0, 10.0, 50.0])
def test_softcap_output_tensor(input_shapes, cap, device):
    """Test with preallocated output tensor."""
    in_data, input_tensor = data_gen_with_range(input_shapes, -100, 100, device)
    _, output_tensor = data_gen_with_range(input_shapes, -1, 1, device)

    ttnn.softcap(input_tensor, cap=cap, output_tensor=output_tensor)
    golden_tensor = golden_softcap(in_data, cap)

    assert_with_ulp(golden_tensor, output_tensor, ulp_threshold=10)
    assert_allclose(golden_tensor, output_tensor, rtol=5e-2, atol=0.35)


@pytest.mark.parametrize("cap", [0.5, 2.0, 100.0])
def test_softcap_small_input(cap, device):
    """Test with small input values near zero where Taylor series is used."""
    input_shapes = torch.Size([1, 1, 32, 32])
    in_data, input_tensor = data_gen_with_range(input_shapes, -0.5, 0.5, device)
    output_tensor = ttnn.softcap(input_tensor, cap=cap)
    golden_tensor = golden_softcap(in_data, cap)

    assert_with_ulp(golden_tensor, output_tensor, ulp_threshold=10)
    assert_allclose(golden_tensor, output_tensor, rtol=5e-2, atol=0.35)


@pytest.mark.parametrize("cap", [1.0, 10.0, 50.0])
def test_softcap_large_input(cap, device):
    """Test with large input values where tanh saturates."""
    input_shapes = torch.Size([1, 1, 32, 32])
    in_data, input_tensor = data_gen_with_range(input_shapes, -500, 500, device)
    output_tensor = ttnn.softcap(input_tensor, cap=cap)
    golden_tensor = golden_softcap(in_data, cap)

    assert_with_ulp(golden_tensor, output_tensor, ulp_threshold=10)
    assert_allclose(golden_tensor, output_tensor, rtol=5e-2, atol=0.35)
