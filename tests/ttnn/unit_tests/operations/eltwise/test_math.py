# SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

import torch

import ttnn

from tests.ttnn.utils_for_testing import assert_equal, assert_with_ulp, assert_allclose

pytestmark = pytest.mark.use_module_device


@pytest.mark.parametrize("h", [32])
@pytest.mark.parametrize("w", [32])
@pytest.mark.parametrize("output_dtype", [ttnn.bfloat16, ttnn.uint16, ttnn.uint32])
def test_eq(device, h, w, output_dtype):
    torch.manual_seed(0)

    same = 50
    torch_input_tensor_a = torch.rand((h, w), dtype=torch.bfloat16)
    torch_input_tensor_a[0, 0] = same
    torch_input_tensor_a[0, 1] = same
    torch_input_tensor_a[0, 2] = same

    torch_input_tensor_b = torch.rand((h, w), dtype=torch.bfloat16)
    torch_input_tensor_b[0, 0] = same
    torch_input_tensor_b[0, 1] = same
    torch_input_tensor_b[0, 2] = same

    golden_function = ttnn.get_golden_function(ttnn.eq)
    torch_output_tensor = golden_function(torch_input_tensor_a, torch_input_tensor_b)

    input_tensor_a = ttnn.from_torch(
        torch_input_tensor_a, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.L1_MEMORY_CONFIG
    )
    input_tensor_b = ttnn.from_torch(
        torch_input_tensor_b, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.L1_MEMORY_CONFIG
    )

    pages_before = ttnn._ttnn.reports.get_buffer_pages(device)
    output_tensor = ttnn.eq(input_tensor_a, input_tensor_b, dtype=output_dtype)
    assert output_tensor.get_dtype() == output_dtype
    assert len(pages_before) == len(ttnn._ttnn.reports.get_buffer_pages(device)) - 1
    output_tensor = ttnn.to_torch(output_tensor)
    assert_equal(torch_output_tensor.float(), output_tensor.float())

    # EQ with a preallocated output tensor
    output_tensor_preallocated_bfloat16 = ttnn.ones(
        [h, w], ttnn.bfloat16, ttnn.TILE_LAYOUT, device, ttnn.L1_MEMORY_CONFIG
    )
    output_tensor_preallocated = output_tensor_preallocated_bfloat16
    if output_dtype != ttnn.bfloat16:
        output_tensor_preallocated = ttnn.typecast(
            output_tensor_preallocated_bfloat16, output_dtype, memory_config=ttnn.L1_MEMORY_CONFIG
        )

    pages_before = ttnn._ttnn.reports.get_buffer_pages(device)
    ttnn.eq(input_tensor_a, input_tensor_b, dtype=output_dtype, output_tensor=output_tensor_preallocated)
    assert len(pages_before) == len(ttnn._ttnn.reports.get_buffer_pages(device))
    torch_output_tensor_preallocated = ttnn.to_torch(output_tensor_preallocated)
    assert_equal(torch_output_tensor.float(), torch_output_tensor_preallocated.float())


def test_digamma_large_x(device):
    """Regression guard for digamma at large x (issue #45520: "behaves bad for x>1000").

    The LUT kernel is fit on [0.01, 102]; beyond it a Bernoulli asymptotic branch
    (ln(x) - 1/2x - 1/12x^2 + ...) restores the (1, inf) support the pre-LUT composite
    op had. ``test_digamma`` only exercises [2, 102], so this covers the LUT->asymptotic
    crossover (102) and several decades past x=1000.
    """
    xs = torch.tensor(
        [[101.0, 102.0, 103.0, 150.0, 500.0, 1000.0, 5000.0, 1e4, 5e4, 1e5, 5e5, 1e6, 1e7, float("inf")]],
        dtype=torch.bfloat16,
    )
    golden = torch.digamma(xs.to(torch.float64)).to(torch.float32)
    input_tensor = ttnn.from_torch(xs, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    output_tensor = ttnn.to_torch(ttnn.digamma(input_tensor))
    assert_with_ulp(expected_result=golden, actual_result=output_tensor, ulp_threshold=2, allow_nonfinite=True)


def test_digamma_small_x(device):
    """Guard the steep near-pole region [0.01, 2): psi has a pole at 0 (psi(x) ~ -1/x),
    the steepest part of the fitted domain. test_digamma only exercises [2, 102].
    Sample avoids the zero-crossing at x~=1.4616 where ULP is ill-defined.
    """
    xs = torch.tensor(
        [[0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 0.75, 1.0, 1.25, 1.75, 1.9, 1.99]],
        dtype=torch.bfloat16,
    )
    golden = torch.digamma(xs.to(torch.float64)).to(torch.float32)
    input_tensor = ttnn.from_torch(xs, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    output_tensor = ttnn.to_torch(ttnn.digamma(input_tensor))
    assert_with_ulp(expected_result=golden, actual_result=output_tensor, ulp_threshold=2)


@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
@pytest.mark.parametrize("fill_value", [0.0, 0.001, -0.001, 1.0, -1.0])
def test_recip_fixed(device, h, w, fill_value):
    torch.manual_seed(0)
    torch_input_tensor = torch.full((h, w), fill_value, dtype=torch.bfloat16)
    input_tensor = ttnn.from_torch(torch_input_tensor, layout=ttnn.TILE_LAYOUT, device=device)
    output_tensor = ttnn.reciprocal(input_tensor)
    output_tensor = ttnn.to_torch(output_tensor)
    if fill_value == 0.0:
        # 1/+0 must be +inf on device (the bf16 golden clamps to +max-bf16, which is incorrect).
        # Compare against an exact +inf tensor so that a -inf or +max-bf16 regression is caught.
        expected = torch.full_like(output_tensor, float("inf"))
        assert torch.equal(
            output_tensor, expected
        ), f"reciprocal(+0) should produce +inf, got unique values {output_tensor.unique()}"
    else:
        golden_function = ttnn.get_golden_function(ttnn.reciprocal)
        torch_output_tensor = golden_function(torch_input_tensor, device=device)
        assert_with_ulp(expected_result=torch_output_tensor, actual_result=output_tensor, ulp_threshold=1)
        assert_allclose(torch_output_tensor, output_tensor, atol=1e-2, rtol=1e-2)
