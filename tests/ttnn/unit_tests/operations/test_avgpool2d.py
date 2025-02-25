# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
import pytest

from tests.ttnn.utils_for_testing import assert_with_pcc


def run_avg_pool(device, input_shape, kernel_size, stride, padding, dilation):
    # Test setup for both.
    batch_size, in_c, in_h, in_w = input_shape
    torch_input = torch.rand(input_shape, dtype=torch.bfloat16)
    # Test setup for Actual.
    input_tensor = torch.permute(torch_input, (0, 2, 3, 1))
    input_tensor = torch.reshape(input_tensor, (1, 1, -1, in_c))
    input_tensor = ttnn.from_torch(input_tensor, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)

    # Get Expected output.
    expected_output = torch.nn.functional.avg_pool2d(
        torch_input, kernel_size, stride, padding, ceil_mode=False, count_include_pad=True, divisor_override=None
    )

    # Get Actual output
    output_tensor = ttnn.avg_pool2d(input_tensor, batch_size, in_h, in_w, in_c, kernel_size, stride, padding, dilation)

    # Test teardown for Actual.
    output_tensor = ttnn.to_torch(output_tensor)
    output_tensor = torch.permute(output_tensor, (0, 3, 1, 2))
    output_tensor = torch.reshape(output_tensor, expected_output.shape)

    # Assertion
    # assert_with_pcc(expected_output, output_tensor, 0.99)
    # print(input_tensor.shape)
    torch.set_printoptions(profile="full")
    print(expected_output.shape)
    print(output_tensor.shape)
    print(expected_output)
    print(output_tensor)
    assert torch.allclose(expected_output, output_tensor, atol=0.01)


@pytest.mark.parametrize("device_params", [{"l1_small_size": 4096}], indirect=True)
@pytest.mark.parametrize(
    "input_shape, kernel_size, stride, padding, dilation",
    [
        # ((4, 256, 40, 40), (2, 2), (2, 2), (0, 0), (1, 1)),
        ((1, 64, 16, 16), (2, 2), (2, 2), (0, 0), (1, 1)),
        # ((1, 128, 56, 56), (2, 2), (2, 2), (0, 0), (1, 1)),
        # ((1, 256, 28, 28), (2, 2), (2, 2), (0, 0), (1, 1)),
        # ((1, 192, 56, 56), (2, 2), (2, 2), (0, 0), (1, 1)),
        # ((1, 160, 7, 7), (2, 2), (2, 2), (0, 0), (1, 1)),
        # ((1, 256, 56, 56), (13, 13), (1, 1), (0, 0), (1, 1)),
        # pytest.param((1, 512, 14, 14), (2, 2), (2, 2), (0, 0), (1, 1), marks=pytest.mark.xfail(reason="incorrect results (#14459)")),
        # pytest.param((1, 384, 28, 28), (2, 2), (2, 2), (0, 0), (1, 1), marks=pytest.mark.xfail(reason="in_ntiles_c % MAX_TILES_PER_REDUCTION != 0 (#13901)")),
        # pytest.param((1, 1056, 14, 14), (2, 2), (2, 2), (0, 0), (1, 1), marks=pytest.mark.xfail(reason="in_ntiles_c % MAX_TILES_PER_REDUCTION != 0 (#13901)")),
        # pytest.param((1, 640, 14, 14), (2, 2), (2, 2), (0, 0), (1, 1), marks=pytest.mark.xfail(reason="in_ntiles_c % MAX_TILES_PER_REDUCTION != 0 (#13901)")),
        # pytest.param((1, 896, 14, 14), (2, 2), (2, 2), (0, 0), (1, 1), marks=pytest.mark.xfail(reason="in_ntiles_c % MAX_TILES_PER_REDUCTION != 0 (#13901)")),
        # pytest.param((1, 24, 56, 56), (2, 2), (2, 2), (0, 0), (1, 1), marks=pytest.mark.xfail(reason="in_ntiles_c % MAX_TILES_PER_REDUCTION != 0 (#13901)")),
        # pytest.param((1, 40, 28, 28), (2, 2), (2, 2), (0, 0), (1, 1), marks=pytest.mark.xfail(reason="incorrect results (#15731)")),
        # pytest.param((1, 80, 14, 14), (2, 2), (2, 2), (0, 0), (1, 1), marks=pytest.mark.xfail(reason="incorrect results (#15731)")),
        # pytest.param((1, 112, 14, 14), (2, 2), (2, 2), (0, 0), (1, 1), marks=pytest.mark.xfail(reason="incorrect results (#15731)")),
        # pytest.param((1, 384, 35, 35), (3, 3), (1, 1), (1, 1), (1, 1), marks=pytest.mark.xfail(reason="in_ntiles_c % MAX_TILES_PER_REDUCTION != 0 (#13901)")),
        # pytest.param((1, 1024, 17, 17), (3, 3), (1, 1), (1, 1), (1, 1), marks=pytest.mark.xfail(reason="incorrect results (#14459)")),
        # pytest.param((1, 1536, 8, 8), (3, 3), (1, 1), (1, 1), (1, 1), marks=pytest.mark.xfail(reason="incorrect results (#14459)")),
    ],
)
def test_run_avg_pool(device, input_shape, kernel_size, stride, padding, dilation):
    run_avg_pool(device, input_shape, kernel_size, stride, padding, dilation)
