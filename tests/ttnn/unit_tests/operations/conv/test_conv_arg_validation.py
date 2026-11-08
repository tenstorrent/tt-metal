# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""conv2d and conv_transpose2d must reject the argument combinations torch rejects.

The shared conv parameter validation only rejected zero-valued arguments. It accepted groups that do not
divide the channel counts, which the weight reshaping divides by; dilation zero, which multiplies the
effective kernel and the padding the output image size derives from; and an output_padding at least as
large as both the stride and the dilation, which torch rejects because the step would double-count.
Each of those produced an output instead of an error.
"""

import pytest
import torch
import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc

# The valid-config controls run a conv on device and need the conv tests' L1_SMALL region.
pytestmark = pytest.mark.parametrize("device_params", [{"l1_small_size": 64 * 1024}], indirect=True)


def _conv2d_kwargs(device, overrides):
    kwargs = dict(
        device=device,
        in_channels=32,
        out_channels=32,
        batch_size=1,
        input_height=16,
        input_width=32,
        kernel_size=(3, 3),
        stride=(1, 1),
        padding=(1, 1),
        dilation=(1, 1),
        groups=1,
    )
    kwargs.update(overrides)
    return kwargs


def _conv_transpose2d_kwargs(device, overrides):
    kwargs = dict(
        device=device,
        in_channels=32,
        out_channels=32,
        batch_size=1,
        input_height=16,
        input_width=32,
        kernel_size=(3, 3),
        stride=(2, 2),
        padding=(1, 1),
        dilation=(1, 1),
        groups=1,
    )
    kwargs.update(overrides)
    return kwargs


def _tt_inputs(device, weight_shape=(32, 32, 3, 3)):
    # NHWC input for the declared 16x32, 32-channel frames; the weight is a host tensor.
    torch_input = torch.randn(1, 16, 32, 32, dtype=torch.bfloat16)
    torch_weight = torch.randn(weight_shape, dtype=torch.bfloat16)
    tt_input = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16, device=device)
    tt_weight = ttnn.from_torch(torch_weight, dtype=ttnn.bfloat16)
    return torch_input, torch_weight, tt_input, tt_weight


@pytest.mark.parametrize(
    "overrides, expected",
    [
        ({"groups": 3}, "divisible by groups"),
        ({"groups": 0}, "groups must be greater than 0"),
        ({"dilation": (0, 1)}, "dilation must be greater than 0"),
        ({"dilation": (1, 0)}, "dilation must be greater than 0"),
    ],
    ids=["groups_not_dividing_channels", "groups_zero", "dilation_h_zero", "dilation_w_zero"],
)
def test_conv2d_rejects_invalid_params(device, overrides, expected, expect_error):
    _, _, tt_input, tt_weight = _tt_inputs(device)
    with expect_error(RuntimeError, expected):
        ttnn.conv2d(input_tensor=tt_input, weight_tensor=tt_weight, **_conv2d_kwargs(device, overrides))


@pytest.mark.parametrize(
    "overrides, expected",
    [
        ({"groups": 3}, "divisible by groups"),
        ({"groups": 0}, "groups must be greater than 0"),
        ({"dilation": (0, 1)}, "dilation must be greater than 0"),
    ],
    ids=["groups_not_dividing_channels", "groups_zero", "dilation_h_zero"],
)
def test_conv_transpose2d_rejects_invalid_params(device, overrides, expected, expect_error):
    _, _, tt_input, tt_weight = _tt_inputs(device)
    with expect_error(RuntimeError, expected):
        ttnn.conv_transpose2d(
            input_tensor=tt_input,
            weight_tensor=tt_weight,
            output_padding=(0, 0),
            **_conv_transpose2d_kwargs(device, overrides),
        )


@pytest.mark.parametrize(
    "output_padding, stride, dilation",
    [
        pytest.param((2, 2), (2, 2), (1, 1), id="equals_stride"),
        pytest.param((3, 2), (2, 2), (1, 1), id="exceeds_stride"),
        pytest.param((2, 2), (2, 2), (2, 2), id="equals_stride_and_dilation"),
    ],
)
def test_conv_transpose2d_rejects_output_padding(device, output_padding, stride, dilation, expect_error):
    _, _, tt_input, tt_weight = _tt_inputs(device)
    # Torch rejects a dimension only when output_padding is at least both the stride and the dilation.
    with expect_error(RuntimeError, "must be smaller than"):
        ttnn.conv_transpose2d(
            input_tensor=tt_input,
            weight_tensor=tt_weight,
            output_padding=output_padding,
            **_conv_transpose2d_kwargs(device, {"stride": stride, "dilation": dilation}),
        )


@pytest.mark.parametrize(
    "output_padding, stride, dilation",
    [
        ((1, 1), (2, 2), (1, 1)),
        ((1, 0), (2, 2), (1, 1)),
        ((0, 0), (2, 2), (1, 1)),
        ((2, 2), (2, 2), (3, 3)),
    ],
    ids=["below_stride", "below_stride_h", "zero", "below_dilation"],
)
def test_conv_transpose2d_accepts_valid_output_padding(device, output_padding, stride, dilation):
    # Controls that pass with and without the fix: below the stride, or below the dilation.
    torch_input, torch_weight, tt_input, tt_weight = _tt_inputs(device)
    output, (out_height, out_width) = ttnn.conv_transpose2d(
        input_tensor=tt_input,
        weight_tensor=tt_weight,
        output_padding=output_padding,
        return_output_dim=True,
        **_conv_transpose2d_kwargs(device, {"stride": stride, "dilation": dilation}),
    )

    ref = torch.nn.functional.conv_transpose2d(
        torch_input.permute(0, 3, 1, 2).float(),
        torch_weight.float(),
        padding=(1, 1),
        output_padding=output_padding,
        stride=stride,
        dilation=dilation,
    )
    # The op returns the flattened [1, 1, N*out_height*out_width, C] tensor.
    assert list(output.shape) == [1, 1, out_height * out_width, ref.shape[1]]
    output_nhwc = ttnn.to_torch(output).float().reshape(1, out_height, out_width, ref.shape[1])
    assert_with_pcc(ref, output_nhwc.permute(0, 3, 1, 2), 0.99)


@pytest.mark.parametrize("groups", [1, 2, 4])
def test_conv2d_accepts_valid_groups(device, groups):
    # Control that passes with and without the fix: groups that divide every channel count.
    torch_input, torch_weight, tt_input, tt_weight = _tt_inputs(device, weight_shape=(32, 32 // groups, 3, 3))

    output, (out_height, out_width) = ttnn.conv2d(
        input_tensor=tt_input,
        weight_tensor=tt_weight,
        return_output_dim=True,
        **_conv2d_kwargs(device, {"groups": groups}),
    )

    ref = torch.nn.functional.conv2d(
        torch_input.permute(0, 3, 1, 2).float(),
        torch_weight.float(),
        padding=(1, 1),
        groups=groups,
    )
    # The op returns the flattened [1, 1, N*out_height*out_width, C] tensor.
    assert list(output.shape) == [1, 1, out_height * out_width, ref.shape[1]]
    output_nhwc = ttnn.to_torch(output).float().reshape(1, out_height, out_width, ref.shape[1])
    assert_with_pcc(ref, output_nhwc.permute(0, 3, 1, 2), 0.99)
