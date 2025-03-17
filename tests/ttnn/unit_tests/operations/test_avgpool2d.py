# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
import pytest

from tests.ttnn.utils_for_testing import assert_with_pcc


def run_avg_pool(device, input_shape, kernel_size, stride, padding, dilation, shard_scheme=None, ceil_mode=False):
    ## Test setup for both.
    in_n, in_c, in_h, in_w = input_shape
    torch.manual_seed(0)
    torch_input = torch.rand(input_shape, dtype=torch.bfloat16)
    # torch_input = torch.zeros(input_shape, dtype=torch.bfloat16)
    # for n in range(input_shape[0]):
    #     for c in range(input_shape[1]):
    #         for h in range(input_shape[2]):
    #             for w in range(input_shape[3]):
    #                 torch_input[n, c, h, w] = h * in_w + w

    ## Test setup for Actual.
    input_tensor = torch.permute(torch_input, (0, 2, 3, 1))
    input_tensor = torch.reshape(input_tensor, (1, 1, in_n * in_h * in_w, in_c))
    input_tensor = ttnn.from_torch(input_tensor, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    tt_input_tensor = ttnn.to_device(input_tensor, device)

    ## Get Expected output.
    expected_output = torch.nn.functional.avg_pool2d(
        torch_input,
        kernel_size,
        stride,
        padding,
        ceil_mode=ceil_mode,
        count_include_pad=True,
        divisor_override=None,
    )

    ## Get Actual output
    output_tensor = ttnn.avg_pool2d(
        input_tensor=tt_input_tensor,
        batch_size=in_n,
        input_h=in_h,
        input_w=in_w,
        channels=in_c,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        # memory_config=memory_config,
        applied_shard_scheme=shard_scheme,
        ceil_mode=ceil_mode,
    )

    ## Test teardown for Actual.
    output_tensor = output_tensor.cpu()
    output_tensor = torch.Tensor(ttnn.to_torch(output_tensor))
    output_tensor = output_tensor[:, :, :, :in_c]
    output_tensor = output_tensor.reshape(
        expected_output.shape[0], expected_output.shape[2], expected_output.shape[3], expected_output.shape[1]
    )
    output_tensor = torch.permute(output_tensor, (0, 3, 1, 2))  # N, C, H, W

    ## Assertion
    assert_with_pcc(expected_output, output_tensor, 0.99)
    assert torch.allclose(output_tensor, expected_output, rtol=0.02)


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
@pytest.mark.parametrize(
    "input_shape",
    (
        [1, 32, 16, 16],
        [1, 32, 112, 112],
        [1, 512, 16, 16],
        [2, 32, 16, 16],
        [2, 32, 112, 112],
        [2, 512, 16, 16],
        [1, 800, 16, 16],
        [2, 800, 16, 16],
    ),
)
@pytest.mark.parametrize(
    "kernel_size",
    (
        (2, 2),
        (3, 3),
        (5, 5),
        (9, 9),
    ),
)
@pytest.mark.parametrize(
    "stride",
    (
        (1, 1),
        (2, 2),
    ),
)
@pytest.mark.parametrize(
    "padding",
    (
        (0, 0),
        (1, 1),
        (2, 2),
        (4, 4),
        (6, 6),
    ),
)
@pytest.mark.parametrize("dilation", ((1, 1),))
@pytest.mark.parametrize(
    "ceil_mode",
    [
        False,
        # True # TODO(jongbinlimTT): Need to support.
    ],
)
def test_run_avg_pool(device, input_shape, kernel_size, stride, padding, dilation, ceil_mode):
    known_issues = [
        ((2, 2), (2, 2)),
        ((2, 2), (3, 3)),
        ((4, 4), (2, 2)),
        ((4, 4), (3, 3)),
        ((4, 4), (5, 5)),
        ((6, 6), (2, 2)),
        ((6, 6), (3, 3)),
        ((6, 6), (5, 5)),
        ((6, 6), (9, 9)),
    ]
    if ((padding, kernel_size) in known_issues) or any(p > k // 2 for p, k in zip(padding, kernel_size)):
        pytest.skip(
            "Known issue with this combination of parameters - RuntimeError: pad should be at most half of kernel size."
        )

    run_avg_pool(
        device,
        input_shape,
        kernel_size,
        stride,
        padding,
        dilation,
        shard_scheme=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ceil_mode=ceil_mode,
    )
