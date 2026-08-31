# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn

from tests.ttnn.utils_for_testing import (
    assert_allclose,
    flush_subnormal_values_to_zero,
    generate_all_bfloat16_bitpatterns,
)

GELU_APPROXIMATIONS = ("none", "tanh")
SHAPE_TEST_CASES = (
    pytest.param(torch.Size([32]), id="rank1-tile-aligned"),
    pytest.param(torch.Size([25, 34]), id="rank2-unaligned"),
    pytest.param(torch.Size([1, 32, 32]), id="rank3-tile-aligned"),
    pytest.param(torch.Size([1, 3, 323, 389]), id="rank4-unaligned"),
)
EXHAUSTIVE_TEST_CASES = (
    pytest.param("none", torch.bfloat16, ttnn.bfloat16, 2e-2, 9e-3, id="none-bf16"),
    # The tanh kernel has a 0.0134 absolute error at its BF16 zero crossing.
    pytest.param("tanh", torch.bfloat16, ttnn.bfloat16, 4.9e-2, 1.5e-2, id="tanh-bf16"),
    pytest.param("none", torch.float32, ttnn.float32, 1e-2, 9e-3, id="none-fp32"),
    pytest.param("tanh", torch.float32, ttnn.float32, 1e-4, 1e-4, id="tanh-fp32"),
)
SPECIAL_VALUE_DTYPES = (
    pytest.param(torch.bfloat16, ttnn.bfloat16, 4.9e-2, 9e-3, id="bf16"),
    pytest.param(torch.float32, ttnn.float32, 1e-4, 1e-4, id="fp32"),
)
SPECIAL_VALUE_CASES = (
    pytest.param("none", float("inf"), "one", id="none-pos-inf"),
    pytest.param("none", float("-inf"), "zero", id="none-neg-inf"),
    pytest.param(
        "none",
        float("nan"),
        "nan",
        id="none-nan",
        marks=pytest.mark.xfail(
            reason="GELU polynomial backward currently treats NaN as a large positive value and returns 1.0",
            strict=True,
        ),
    ),
    pytest.param("tanh", float("inf"), "one", id="tanh-pos-inf"),
    pytest.param("tanh", float("-inf"), "zero", id="tanh-neg-inf"),
    pytest.param("tanh", float("nan"), "nan", id="tanh-nan"),
)


def _gelu_bw_reference(input_tensor, grad_tensor, approximate):
    input_tensor = input_tensor.to(torch.float32).detach().requires_grad_(True)
    output_tensor = torch.nn.functional.gelu(input_tensor, approximate=approximate)
    output_tensor.backward(grad_tensor.to(torch.float32))
    return input_tensor.grad


def _make_exhaustive_inputs(torch_dtype):
    # These are every BF16 bit pattern, promoted losslessly for the FP32 path.
    # Tenstorrent hardware flushes subnormal values, so flush them before both the
    # host reference and device execution. Non-finite encodings are covered by the
    # dedicated special-value test below.
    input_tensor = generate_all_bfloat16_bitpatterns(torch_dtype)
    input_tensor = flush_subnormal_values_to_zero(input_tensor)
    input_tensor[~torch.isfinite(input_tensor)] = 0.0

    # This exhaustive test isolates the derivative. The preallocated-output and
    # program-cache tests below use non-constant gradients to validate scaling.
    return input_tensor, torch.ones_like(input_tensor)


def _bf16_tolerance(approximate):
    return (4.9e-2, 1.5e-2) if approximate == "tanh" else (2e-2, 9e-3)


@pytest.mark.parametrize("approximate,torch_dtype,ttnn_dtype,rtol,atol", EXHAUSTIVE_TEST_CASES)
def test_gelu_bw_exhaustive_allclose(device, approximate, torch_dtype, ttnn_dtype, rtol, atol):
    input_data, grad_data = _make_exhaustive_inputs(torch_dtype)
    expected = _gelu_bw_reference(input_data, grad_data, approximate)

    # PyTorch's tanh-GELU backward can itself overflow for the largest finite
    # inputs. Those encodings are still sent to the device, but have no finite
    # PyTorch reference for an allclose comparison.
    finite_reference = torch.isfinite(expected)

    input_tensor = ttnn.from_torch(input_data, dtype=ttnn_dtype, layout=ttnn.TILE_LAYOUT, device=device)
    grad_tensor = ttnn.from_torch(grad_data, dtype=ttnn_dtype, layout=ttnn.TILE_LAYOUT, device=device)
    actual = ttnn.to_torch(ttnn.gelu_bw(grad_tensor, input_tensor, approximate=approximate)[0])

    assert torch.isfinite(actual[finite_reference]).all(), "device output is non-finite for a finite reference"
    assert_allclose(expected[finite_reference], actual[finite_reference], rtol=rtol, atol=atol)


def test_gelu_bw_default_matches_none(device):
    input_data = torch.linspace(-5.0, 5.0, 32 * 32, dtype=torch.bfloat16).reshape(32, 32)
    grad_data = torch.linspace(-2.0, 2.0, 32 * 32, dtype=torch.bfloat16).reshape(32, 32)
    input_tensor = ttnn.from_torch(input_data, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    grad_tensor = ttnn.from_torch(grad_data, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    default_actual = ttnn.to_torch(ttnn.gelu_bw(grad_tensor, input_tensor)[0])
    none_actual = ttnn.to_torch(ttnn.gelu_bw(grad_tensor, input_tensor, approximate="none")[0])
    assert torch.equal(default_actual, none_actual)


@pytest.mark.parametrize("approximate", GELU_APPROXIMATIONS)
@pytest.mark.parametrize("shape", SHAPE_TEST_CASES)
def test_gelu_bw_shape_coverage(device, approximate, shape):
    torch.manual_seed(shape.numel())
    input_data = torch.rand(shape, dtype=torch.bfloat16) * 4 - 2
    grad_data = torch.rand(shape, dtype=torch.bfloat16) * 4 - 2
    input_tensor = ttnn.from_torch(input_data, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    grad_tensor = ttnn.from_torch(grad_data, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    actual = ttnn.to_torch(ttnn.gelu_bw(grad_tensor, input_tensor, approximate=approximate)[0])
    rtol, atol = _bf16_tolerance(approximate)
    assert_allclose(_gelu_bw_reference(input_data, grad_data, approximate), actual, rtol=rtol, atol=atol)


@pytest.mark.parametrize("torch_dtype,ttnn_dtype,rtol,atol", SPECIAL_VALUE_DTYPES)
@pytest.mark.parametrize("approximate,input_value,expected", SPECIAL_VALUE_CASES)
def test_gelu_bw_special_values(device, approximate, input_value, expected, torch_dtype, ttnn_dtype, rtol, atol):
    if approximate == "tanh" and torch.isinf(torch.tensor(input_value)) and ttnn_dtype == ttnn.float32:
        pytest.xfail("FP32 tanh GELU backward overflows for infinite inputs and produces NaN")
    if approximate == "tanh" and torch.isnan(torch.tensor(input_value)) and ttnn_dtype == ttnn.bfloat16:
        pytest.xfail("BF16 tanh GELU backward treats NaN as a large positive value and returns 1.0")

    input_data = torch.tensor([input_value] + [0.0] * 31, dtype=torch_dtype)
    grad_data = torch.ones_like(input_data)
    input_tensor = ttnn.from_torch(
        input_data, dtype=ttnn_dtype, layout=ttnn.TILE_LAYOUT, device=device, preserve_nan_values=True
    )
    grad_tensor = ttnn.from_torch(grad_data, dtype=ttnn_dtype, layout=ttnn.TILE_LAYOUT, device=device)
    actual = ttnn.to_torch(ttnn.gelu_bw(grad_tensor, input_tensor, approximate=approximate)[0])[0]

    if expected == "nan":
        assert torch.isnan(actual), "GELU backward must propagate NaN inputs"
    else:
        expected_value = 1.0 if expected == "one" else 0.0
        assert torch.isclose(actual, torch.tensor(expected_value, dtype=torch_dtype), rtol=rtol, atol=atol)


@pytest.mark.parametrize("approximate", GELU_APPROXIMATIONS)
def test_bw_gelu_opt_output(approximate, device):
    shape = torch.Size([1, 1, 320, 384])
    input_data = torch.linspace(-5.0, 5.0, shape.numel(), dtype=torch.bfloat16).reshape(shape)
    grad_data = torch.linspace(-2.0, 2.0, shape.numel(), dtype=torch.bfloat16).reshape(shape)
    input_tensor = ttnn.from_torch(input_data, ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    grad_tensor = ttnn.from_torch(grad_data, ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    input_grad = ttnn.from_torch(
        torch.zeros(shape, dtype=torch.bfloat16),
        ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )

    pages_before = ttnn._ttnn.reports.get_buffer_pages(device)
    ttnn.gelu_bw(grad_tensor, input_tensor, approximate=approximate, input_grad=input_grad, queue_id=0)
    assert len(pages_before) == len(ttnn._ttnn.reports.get_buffer_pages(device))
    rtol, atol = _bf16_tolerance(approximate)
    assert_allclose(
        _gelu_bw_reference(input_data, grad_data, approximate),
        input_grad.cpu().to(ttnn.ROW_MAJOR_LAYOUT).to_torch(),
        rtol=rtol,
        atol=atol,
    )


@pytest.mark.parametrize(
    "grad_dtype,input_dtype",
    (
        (ttnn.bfloat16, ttnn.bfloat16),
        (ttnn.float32, ttnn.bfloat16),
        (ttnn.bfloat16, ttnn.float32),
    ),
)
def test_bw_gelu_grad_dtype_must_match_input(grad_dtype, input_dtype, device, expect_error):
    shape = torch.Size([1, 1, 32, 32])
    input_data = torch.linspace(-5.0, 5.0, shape.numel(), dtype=torch.float32).reshape(shape)
    grad_data = torch.linspace(-2.0, 2.0, shape.numel(), dtype=torch.float32).reshape(shape)
    input_tensor = ttnn.from_torch(input_data, input_dtype, layout=ttnn.TILE_LAYOUT, device=device)
    grad_tensor = ttnn.from_torch(grad_data, grad_dtype, layout=ttnn.TILE_LAYOUT, device=device)

    if grad_dtype != input_dtype:
        with expect_error(RuntimeError, "grad_output and input data types to match"):
            ttnn.gelu_bw(grad_tensor, input_tensor)
        return

    assert ttnn.gelu_bw(grad_tensor, input_tensor)[0].dtype == input_dtype


def test_bw_gelu_program_cache_regression(device):
    device.enable_program_cache()
    device.clear_program_cache()
    shape = torch.Size([1, 1, 320, 384])

    def fresh_inputs(seed):
        torch.manual_seed(seed)
        input_data = (torch.rand(shape, dtype=torch.bfloat16) * 4 - 2).detach()
        grad_data = (torch.rand(shape, dtype=torch.bfloat16) * 4 - 2).detach()
        return (
            input_data,
            grad_data,
            ttnn.from_torch(input_data, ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device),
            ttnn.from_torch(grad_data, ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device),
        )

    try:
        for expected_entries, approximate in enumerate(GELU_APPROXIMATIONS, start=1):
            input_data, grad_data, input_tensor, grad_tensor = fresh_inputs(expected_entries)
            actual = ttnn.gelu_bw(grad_tensor, input_tensor, approximate=approximate)[0]
            rtol, atol = _bf16_tolerance(approximate)
            assert_allclose(
                _gelu_bw_reference(input_data, grad_data, approximate), ttnn.to_torch(actual), rtol=rtol, atol=atol
            )
            assert device.num_program_cache_entries() == expected_entries

        for seed, approximate in ((42, "none"), (99, "tanh")):
            input_data, grad_data, input_tensor, grad_tensor = fresh_inputs(seed)
            actual = ttnn.gelu_bw(grad_tensor, input_tensor, approximate=approximate)[0]
            rtol, atol = _bf16_tolerance(approximate)
            assert_allclose(
                _gelu_bw_reference(input_data, grad_data, approximate), ttnn.to_torch(actual), rtol=rtol, atol=atol
            )
            assert device.num_program_cache_entries() == 2

        for approximate in GELU_APPROXIMATIONS:
            entries_before = None
            for seed in (100, 200):
                input_data, grad_data, input_tensor, grad_tensor = fresh_inputs(seed)
                rtol, atol = _bf16_tolerance(approximate)
                input_grad = ttnn.from_torch(
                    torch.zeros(shape, dtype=torch.bfloat16),
                    ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    device=device,
                    memory_config=ttnn.L1_MEMORY_CONFIG,
                )
                ttnn.gelu_bw(grad_tensor, input_tensor, approximate=approximate, input_grad=input_grad, queue_id=0)
                assert_allclose(
                    _gelu_bw_reference(input_data, grad_data, approximate),
                    ttnn.to_torch(input_grad),
                    rtol=rtol,
                    atol=atol,
                )
                if entries_before is None:
                    entries_before = device.num_program_cache_entries()
                else:
                    assert device.num_program_cache_entries() == entries_before
    finally:
        device.disable_and_clear_program_cache()
