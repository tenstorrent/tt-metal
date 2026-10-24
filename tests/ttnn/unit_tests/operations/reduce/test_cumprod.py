# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

import torch

import ttnn

from tests.ttnn.utils_for_testing import assert_numeric_metrics

TEST_PADDING_VALUE = -42


@pytest.mark.parametrize("dim", [0, 2, -1])
@pytest.mark.parametrize(
    "shape",
    [
        [],
        [2],
        [8, 64, 64],
        [5, 5, 5, 5, 1, 1, 1],
        [10, 3, 18, 20],
    ],
)
@pytest.mark.parametrize(
    "dtypes",
    [
        (torch.float32, None),
        (torch.bfloat16, ttnn.float32),
    ],
)
def test_cumprod_normal(dim, shape, dtypes, device):
    torch.manual_seed(0)
    if dim < len(shape) and -len(shape) <= dim:
        for _ in range(2):
            torch_input_tensor = torch.randn(shape, dtype=dtypes[0])
            torch_result_tensor = torch.cumprod(torch_input_tensor, dim)
            ttnn_input_tensor = ttnn.from_torch(
                torch_input_tensor, ttnn.bfloat16, layout=ttnn.Layout.TILE, device=device
            )
            ttnn_input_tensor = ttnn.fill_implicit_tile_padding(ttnn_input_tensor, TEST_PADDING_VALUE)
            ttnn_result_tensor = ttnn.cumprod(ttnn_input_tensor, dim, dtype=dtypes[1])

            # assert metadata
            assert ttnn_input_tensor.shape == ttnn_result_tensor.shape
            assert torch_input_tensor.shape == ttnn_input_tensor.shape
            assert torch_result_tensor.shape == ttnn_input_tensor.shape
            assert torch_result_tensor.shape == ttnn_result_tensor.shape

            ttnn_result_torch = ttnn.to_torch(ttnn_result_tensor)
            if dtypes[0] == torch.float32:
                pcc_threshold = 0.999
                rtol = 0.09
                atol = 1.47
                frobenius_threshold = 0.02
            else:
                pcc_threshold = 0.999
                rtol = 0.03
                atol = 1.04
                frobenius_threshold = 0.01
            # test for equivalance
            assert_numeric_metrics(
                torch_result_tensor,
                ttnn_result_torch,
                pcc_threshold=pcc_threshold,
                rtol=rtol,
                atol=atol,
                frobenius_threshold=frobenius_threshold,
            )
        assert device.num_program_cache_entries() >= 1
    else:
        pytest.skip(f"skipping for dim == {dim} and shape == {shape}")


@pytest.mark.parametrize("dim", [0, 2, -1])
@pytest.mark.parametrize(
    "shape",
    [
        [],
        [2],
        [8, 64, 64],
        [5, 5, 5, 5, 1, 1, 1],
    ],
)
@pytest.mark.parametrize(
    "dtypes",
    [
        (torch.float32, None),
        (torch.bfloat16, ttnn.float32),
    ],
)
def test_cumprod_reverse_order(dim, shape, dtypes, device):
    torch.manual_seed(0)
    if dim < len(shape) and -len(shape) <= dim:
        _, ttnn_dtype = dtypes

        # Factors in {-2, -1, 1, 2} keep every product a nonzero signed power of two, exact in bfloat16
        torch_input = torch.tensor([-2.0, -1.0, 1.0, 2.0], dtype=torch.bfloat16)[torch.randint(0, 4, shape)]
        torch_output = torch.flip(torch.cumprod(torch.flip(torch_input.float(), [dim]), dim), [dim])

        tt_input = ttnn.from_torch(torch_input, ttnn.bfloat16, layout=ttnn.Layout.TILE, device=device)
        tt_input = ttnn.fill_implicit_tile_padding(tt_input, float("nan"))
        tt_output_cpu = ttnn.to_torch(ttnn.cumprod(tt_input, dim, dtype=ttnn_dtype, reverse_order=True)).float()

        assert tt_output_cpu.shape == torch_output.shape
        assert torch.equal(tt_output_cpu, torch_output)

    else:
        pytest.skip(f"skipping for dim == {dim} and shape == {shape}")


@pytest.mark.parametrize("dim", [0, 2, -1])
@pytest.mark.parametrize(
    "shape",
    [
        [],
        [1],
        [8, 64, 64],
        [5, 5, 5, 5, 1, 1, 1],
        [1, 1, 20, 16],
    ],
)
@pytest.mark.parametrize(
    "dtypes",
    [
        (torch.float32, None),
    ],
)
def test_cumprod_preallocated(dim, shape, dtypes, device):
    torch.manual_seed(0)
    if dim < len(shape) and -len(shape) <= dim:
        for _ in range(2):
            torch_input_tensor = torch.randn(shape, dtype=dtypes[0])
            torch_preallocated_tensor = torch.zeros_like(torch_input_tensor)
            torch_result_tensor = torch.cumprod(torch_input_tensor, dim, out=torch_preallocated_tensor)
            ttnn_input_tensor = ttnn.from_torch(
                torch_input_tensor, dtype=dtypes[1], layout=ttnn.Layout.TILE, device=device
            )
            ttnn_input_tensor = ttnn.fill_implicit_tile_padding(ttnn_input_tensor, TEST_PADDING_VALUE)
            ttnn_preallocated_tensor = ttnn.zeros_like(ttnn_input_tensor)
            ttnn_result_tensor = ttnn.cumprod(ttnn_input_tensor, dim, out=ttnn_preallocated_tensor)

            # assert metadata
            assert ttnn_input_tensor.shape == ttnn_result_tensor.shape
            assert ttnn_preallocated_tensor.shape == ttnn_result_tensor.shape
            assert ttnn_preallocated_tensor.dtype == ttnn_result_tensor.dtype
            assert torch_input_tensor.shape == ttnn_input_tensor.shape
            assert torch_result_tensor.shape == ttnn_input_tensor.shape
            assert torch_result_tensor.shape == ttnn_result_tensor.shape

            ttnn_result_torch = ttnn.to_torch(ttnn_result_tensor)
            ttnn_preallocated_torch = ttnn.to_torch(ttnn_preallocated_tensor)
            pcc_threshold = 0.999
            rtol = 0.09
            atol = 1.47
            frobenius_threshold = 0.01
            for expected, actual in (
                (torch_result_tensor, ttnn_result_torch),
                (torch_preallocated_tensor, ttnn_preallocated_torch),
            ):
                # test for equivalance
                assert_numeric_metrics(
                    expected,
                    actual,
                    pcc_threshold=pcc_threshold,
                    rtol=rtol,
                    atol=atol,
                    frobenius_threshold=frobenius_threshold,
                )
    else:
        pytest.skip(f"skipping for dim == {dim} and shape == {shape}")


# The preallocated `out` tensor must live on device: the on-device check in
# validate_output_tensor fires before any other validation, so a host `out`
# would mask the check each row targets. Each row asserts the exact message of
# the check it exercises.
@pytest.mark.parametrize(
    "dim, input_shape, output_shape, torch_dtype, input_dtype, output_dtype, memory_config, layout, error_msg",
    [
        (
            -10,
            [1, 2, 3, 4, 5, 6, 7, 8, 9],
            [1, 2, 3, 4, 5, 6, 7, 8, 9],
            torch.bfloat16,
            ttnn.bfloat16,
            ttnn.bfloat16,
            ttnn.DRAM_MEMORY_CONFIG,
            ttnn.Layout.TILE,
            "The requested accumulation axis is -10, while the input tensor has rank 9",
        ),  # input_rank vs dim
        (
            10,
            [1, 2, 3, 4, 5, 6, 7, 8, 9],
            [1, 2, 3, 4, 5, 6, 7, 8, 9],
            torch.bfloat16,
            ttnn.bfloat16,
            ttnn.bfloat16,
            ttnn.DRAM_MEMORY_CONFIG,
            ttnn.Layout.TILE,
            "The requested accumulation axis is 10, while the input tensor has rank 9",
        ),  # input_rank vs dim
        (
            3,
            [1, 2, 3, 4, 5, 6, 7, 8, 9],
            [1, 2, 3, 4, 5, 6, 7, 8],
            torch.bfloat16,
            ttnn.bfloat16,
            ttnn.bfloat16,
            ttnn.DRAM_MEMORY_CONFIG,
            ttnn.Layout.TILE,
            "Shape mismatch: input tensor shape",
        ),  # input_shape vs output_shape
        (
            3,
            [1, 2, 3, 4, 5, 6, 7, 8, 9],
            [1, 2, 3, 4, 5, 6, 7, 8, 1],
            torch.bfloat16,
            ttnn.bfloat16,
            ttnn.bfloat16,
            ttnn.DRAM_MEMORY_CONFIG,
            ttnn.Layout.TILE,
            "Shape mismatch: input tensor shape",
        ),  # input_shape vs output_shape
        (
            3,
            [1, 2, 3, 4, 5, 6, 7, 8, 9],
            [1, 2, 3, 4, 5, 6, 7, 8, 9],
            torch.bfloat16,
            ttnn.bfloat16,
            ttnn.bfloat16,
            ttnn.DRAM_MEMORY_CONFIG,
            ttnn.Layout.ROW_MAJOR,
            "The provided input tensor has a non-tile layout: ROW_MAJOR",
        ),  # unsupported layout
    ],
)
def test_cumprod_failing_cases(
    dim,
    input_shape,
    output_shape,
    torch_dtype,
    input_dtype,
    output_dtype,
    memory_config,
    layout,
    error_msg,
    device,
    expect_error,
):
    torch.manual_seed(0)
    torch_input_tensor = torch.randn(input_shape, dtype=torch_dtype)
    ttnn_input_tensor = ttnn.from_torch(
        torch_input_tensor, dtype=input_dtype, layout=layout, device=device, memory_config=memory_config
    )
    ttnn_preallocated_tensor = ttnn.zeros(output_shape, dtype=output_dtype, layout=ttnn.Layout.TILE, device=device)
    with expect_error(RuntimeError, error_msg):
        ttnn.cumprod(ttnn_input_tensor, memory_config=memory_config, dim=dim, out=ttnn_preallocated_tensor)
