# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc


# @pytest.fixture
# def setup():
#     """Fixture to prepare input data for ttnn.avg_pool2d() and torch.nn.functional.avg_pool2d()."""
#     def _prepare_test_input_data(device, input_shape):
#         # Generate random test input data - suitable for torch.nn.functional.avg_pool2d().
#         input_data = torch.rand(input_shape, dtype=torch.bfloat16)

#         # Preprocess the input data - suitable for ttnn.avg_pool2d().
#         preprocessed_input_data = torch.permute(input_data, (0, 2, 3, 1))
#         preprocessed_input_data = torch.reshape(preprocessed_input_data, (1, 1, -1, in_c))
#         preprocessed_input_data = ttnn.from_torch(preprocessed_input_data, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)

#         batch_size, in_c, in_h, in_w = input_shape
#         return input_data, preprocessed_input_data, batch_size, in_h, in_w, in_c
#     return _prepare_test_input_data

# def post_process_output_data(output_data, batch_size, out_h, out_w, out_c):
#     output_data = ttnn.to_torch(output_data)
#     output_data = torch.reshape(output_data, (batch_size, out_h, out_w, out_c))
#     output_data = torch.permute(output_data, (0, 3, 1, 2))
#     return output_data


# @pytest.mark.parametrize("device_params", [{"l1_small_size": 4096}], indirect=True)
# @pytest.mark.parametrize(
#     "input_shape, kernel_size, stride, padding, dilation",
#     [
#         ((1, 16, 56, 56), (2, 2), (2, 2), (0, 0), (1, 1)),
#         ((1, 128, 56, 56), (2, 2), (2, 2), (0, 0), (1, 1)),
#     ]
# )
# def test_avg_pool2d(setup, device, input_shape, kernel_size, stride, padding, dilation):
#     """Test ttnn.avg_pool2d() by comparing with torch.nn.functional.avg_pool2d()."""

#     # Using fixture, prepare the test input data for both ttnn.avg_pool2d() and torch.nn.functional.avg_pool2d().
#     input_data, preprocessed_input_data, batch_size, in_h, in_w, in_c = setup(device, input_shape, kernel_size, stride, padding, dilation)

#     # Get actual output data from ttnn.avg_pool2d().
#     actual_output_data = ttnn.avg_pool2d(
#         preprocessed_input_data, batch_size, in_h, in_w, in_c, kernel_size, stride, padding, dilation, applied_shard_scheme=None)

#     # Get expected output data from torch.nn.functional.avg_pool2d().
#     expected_output_data = torch.nn.functional.avg_pool2d(
#         input_data, kernel_size, stride, padding, ceil_mode=False, count_include_pad=True, divisor_override=None)

#     # Compare the expected and actual outputs.
#     _, out_c, out_h, out_w = expected_output_data.shape
#     post_processed_actual_output_data = post_process_output_data(actual_output_data, batch_size, out_h, out_w, out_c)
#     assert_with_pcc(expected_output_data, post_processed_actual_output_data, 0.99)


def run_once(device, input_shape, kernel_size, stride, padding, dilation, shard_scheme=None):
    batch_size, in_c, in_h, in_w = input_shape
    torch_input = torch.rand(input_shape, dtype=torch.bfloat16)

    input_tensor = torch.permute(torch_input, (0, 2, 3, 1))
    input_tensor = torch.reshape(input_tensor, (1, 1, -1, in_c))
    input_tensor = ttnn.from_torch(input_tensor, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)

    # Function under test
    output_tensor = ttnn.avg_pool2d(
        input_tensor,
        batch_size,
        in_h,
        in_w,
        in_c,
        kernel_size,
        stride,
        padding,
        dilation,
        applied_shard_scheme=shard_scheme,
    )

    expected_output = torch.nn.functional.avg_pool2d(
        torch_input, kernel_size, stride, padding, ceil_mode=False, count_include_pad=True, divisor_override=None
    )

    output_tensor = ttnn.to_torch(output_tensor)
    _, out_c, out_h, out_w = expected_output.shape
    output_tensor = torch.reshape(output_tensor, (batch_size, out_h, out_w, out_c))
    output_tensor = torch.permute(output_tensor, (0, 3, 1, 2))
    assert_with_pcc(expected_output, output_tensor, 0.99)


@pytest.mark.parametrize("device_params", [{"l1_small_size": 4096}], indirect=True)
@pytest.mark.parametrize(
    "input_shape, kernel_size, stride, padding, dilation",
    [
        ((1, 16, 56, 56), (2, 2), (2, 2), (0, 0), (1, 1)),
        ((1, 128, 56, 56), (2, 2), (2, 2), (0, 0), (1, 1)),
        ((1, 256, 28, 28), (2, 2), (2, 2), (0, 0), (1, 1)),
        ((1, 192, 56, 56), (2, 2), (2, 2), (0, 0), (1, 1)),
        ((1, 160, 7, 7), (2, 2), (2, 2), (0, 0), (1, 1)),
        ((1, 256, 56, 56), (13, 13), (1, 1), (0, 0), (1, 1)),
        pytest.param(
            (1, 512, 14, 14),
            (2, 2),
            (2, 2),
            (0, 0),
            (1, 1),
            marks=pytest.mark.xfail(reason="incorrect results (#14459)"),
        ),
        pytest.param(
            (1, 384, 28, 28),
            (2, 2),
            (2, 2),
            (0, 0),
            (1, 1),
            marks=pytest.mark.xfail(reason="in_ntiles_c % MAX_TILES_PER_REDUCTION != 0 (#13901)"),
        ),
        pytest.param(
            (1, 1056, 14, 14),
            (2, 2),
            (2, 2),
            (0, 0),
            (1, 1),
            marks=pytest.mark.xfail(reason="in_ntiles_c % MAX_TILES_PER_REDUCTION != 0 (#13901)"),
        ),
        pytest.param(
            (1, 640, 14, 14),
            (2, 2),
            (2, 2),
            (0, 0),
            (1, 1),
            marks=pytest.mark.xfail(reason="in_ntiles_c % MAX_TILES_PER_REDUCTION != 0 (#13901)"),
        ),
        pytest.param(
            (1, 896, 14, 14),
            (2, 2),
            (2, 2),
            (0, 0),
            (1, 1),
            marks=pytest.mark.xfail(reason="in_ntiles_c % MAX_TILES_PER_REDUCTION != 0 (#13901)"),
        ),
        pytest.param(
            (1, 24, 56, 56),
            (2, 2),
            (2, 2),
            (0, 0),
            (1, 1),
            marks=pytest.mark.xfail(reason="in_ntiles_c % MAX_TILES_PER_REDUCTION != 0 (#13901)"),
        ),
        pytest.param(
            (1, 40, 28, 28),
            (2, 2),
            (2, 2),
            (0, 0),
            (1, 1),
            marks=pytest.mark.xfail(reason="incorrect results (#15731)"),
        ),
        pytest.param(
            (1, 80, 14, 14),
            (2, 2),
            (2, 2),
            (0, 0),
            (1, 1),
            marks=pytest.mark.xfail(reason="incorrect results (#15731)"),
        ),
        pytest.param(
            (1, 112, 14, 14),
            (2, 2),
            (2, 2),
            (0, 0),
            (1, 1),
            marks=pytest.mark.xfail(reason="incorrect results (#15731)"),
        ),
        pytest.param(
            (1, 384, 35, 35),
            (3, 3),
            (1, 1),
            (1, 1),
            (1, 1),
            marks=pytest.mark.xfail(reason="in_ntiles_c % MAX_TILES_PER_REDUCTION != 0 (#13901)"),
        ),
        pytest.param(
            (1, 1024, 17, 17),
            (3, 3),
            (1, 1),
            (1, 1),
            (1, 1),
            marks=pytest.mark.xfail(reason="incorrect results (#14459)"),
        ),
        pytest.param(
            (1, 1536, 8, 8),
            (3, 3),
            (1, 1),
            (1, 1),
            (1, 1),
            marks=pytest.mark.xfail(reason="incorrect results (#14459)"),
        ),
    ],
)
def test_avg_pool2d(device, input_shape, kernel_size, stride, padding, dilation):
    run_once(device, input_shape, kernel_size, stride, padding, dilation)


@pytest.mark.parametrize("device_params", [{"l1_small_size": 8192}], indirect=True)
@pytest.mark.parametrize(
    "input_shape, kernel_size, stride, padding, dilation, shard_scheme",
    [
        ((4, 256, 28, 28), (2, 2), (2, 2), (0, 0), (1, 1), ttnn.TensorMemoryLayout.HEIGHT_SHARDED),
        ((4, 256, 56, 56), (13, 13), (1, 1), (0, 0), (1, 1), ttnn.TensorMemoryLayout.HEIGHT_SHARDED),
        ((1, 2048, 28, 28), (2, 2), (2, 2), (0, 0), (1, 1), ttnn.TensorMemoryLayout.WIDTH_SHARDED),
        ((1, 2048, 56, 56), (13, 13), (1, 1), (0, 0), (1, 1), ttnn.TensorMemoryLayout.WIDTH_SHARDED),
        ((2, 512, 28, 28), (2, 2), (2, 2), (0, 0), (1, 1), ttnn.TensorMemoryLayout.BLOCK_SHARDED),
        ((2, 512, 56, 56), (13, 13), (1, 1), (0, 0), (1, 1), ttnn.TensorMemoryLayout.BLOCK_SHARDED),
    ],
)
def test_avg_pool2d_with_specific_sharding(device, input_shape, kernel_size, stride, padding, dilation, shard_scheme):
    run_once(device, input_shape, kernel_size, stride, padding, dilation, shard_scheme)
