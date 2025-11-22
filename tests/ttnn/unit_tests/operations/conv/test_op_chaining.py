# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
from tests.ttnn.nightly.unit_tests.operations.conv.test_conv2d import (
    torch_tensor_map,
    randomize_torch_tensor,
)
import ttnn
import torch
from loguru import logger
from tests.ttnn.utils_for_testing import check_with_pcc_without_tensor_printout


SliceHeight = ttnn.Conv2dDRAMSliceHeight
SliceWidth = ttnn.Conv2dDRAMSliceWidth


@pytest.mark.parametrize(
    "input_layout, dtype",
    [[ttnn.TILE_LAYOUT, ttnn.bfloat8_b], [ttnn.ROW_MAJOR_LAYOUT, ttnn.bfloat16]],
)
@pytest.mark.parametrize(
    "batch_size, input_channels, input_height, input_width, math_fidelity, slice_type, num_slices, parameters",
    # fmt: off
    (
        ( 1,  400,    192,   192,     ttnn.MathFidelity.HiFi4, SliceWidth, 6,
            [
                (512, (3, 3), (1, 1), (1, 1), (1, 1)),
                (256, (3, 3), (1, 1), (1, 1), (1, 1)),
                (128, (5, 5), (1, 1), (2, 2), (1, 1))
            ]
        ),
        ( 2,   13,    313,    71,     ttnn.MathFidelity.LoFi, SliceHeight, 6,
            [
                (256, (5, 5), (2, 2), (2, 2), (2, 2)),
                (256, (3, 3), (1, 1), (1, 1), (1, 1)),
                (256, (5, 5), (1, 1), (1, 1), (1, 1))
            ]
        ),
        ( 2,   63,    981,    39,     ttnn.MathFidelity.LoFi, SliceHeight, 8,
            [
                (256, (3, 3), (2, 2), (2, 2), (1, 1)),
                (256, (3, 3), (1, 1), (1, 1), (1, 1)),
                (256, (5, 5), (1, 1), (1, 1), (1, 1))
            ]
        ),
        ( 2,  512,    128,   128,     ttnn.MathFidelity.LoFi, SliceWidth, 4,
            [
                (256, (3, 3), (1, 1), (1, 1), (1, 1)),
                (256, (3, 3), (1, 1), (1, 1), (1, 1)),
                (256, (5, 5), (1, 1), (1, 1), (1, 1))
            ]
        ),
        ( 2,   64,    384,    64,     ttnn.MathFidelity.LoFi, SliceHeight, 12,
            [
                (256, (4, 4), (2, 2), (1, 1), (1, 1)),
                (256, (3, 3), (1, 1), (1, 1), (1, 1)),
                (256, (5, 5), (1, 1), (1, 1), (1, 1))
            ]
        ),
        ( 1,    16,   1024,  1024,     ttnn.MathFidelity.LoFi, SliceWidth, 8,
            [
                (32, (3, 3), (1, 1), (1, 1), (1, 1)),
                (32, (3, 3), (1, 1), (1, 1), (1, 1)),
                (32, (3, 3), (1, 1), (1, 1), (1, 1))
            ]
        ),
        ( 1, 2904,     48,    48,     ttnn.MathFidelity.HiFi4, SliceWidth, 2,
            [
                (256, (3, 3), (1, 1), (0, 0), (1, 1)),
                (256, (3, 3), (1, 1), (1, 1), (1, 1)),
                (256, (5, 5), (1, 1), (1, 1), (1, 1))
            ]
        ),
    )
    # fmt: on
)
def test_multi_conv(
    device,
    torch_tensor_map,
    batch_size,
    input_channels,
    input_height,
    input_width,
    input_layout,
    dtype,
    math_fidelity,
    parameters,
    slice_type,
    num_slices,
):
    if device.core_grid.y == 7:
        pytest.skip("Tests have been configured for N150.")

    compute_config = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=math_fidelity,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )
    op_slicing_attrs = []
    torch_weights_tensors = []
    torch_bias_tensors = []
    ttnn_weights_tensors = []
    ttnn_bias_tensors = []
    torch_input_tensor_nchw = randomize_torch_tensor(
        torch_tensor_map, (batch_size, input_channels, input_height, input_width)
    )
    current_input_channels = input_channels
    current_torch_output = torch_input_tensor_nchw
    torch_input_tensor = torch.permute(torch_input_tensor_nchw, (0, 2, 3, 1))
    tt_input_tensor = ttnn.from_torch(
        torch_input_tensor,
        dtype,
        layout=input_layout,
        device=device,
    )
    conv_config = ttnn.Conv2dConfig(
        weights_dtype=dtype,
        config_tensors_in_dram=True,
        output_layout=input_layout,
        activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
    )
    for output_channels, kernel, stride, padding, dilation in parameters:
        torch_weights_tensors.append(
            randomize_torch_tensor(
                torch_tensor_map,
                (output_channels, current_input_channels, kernel[0], kernel[1]),
            )
        )

        torch_bias_tensors.append(
            randomize_torch_tensor(
                torch_tensor_map,
                (1, 1, 1, output_channels),
            )
        )
        ttnn_weights_tensors.append(ttnn.from_torch(torch_weights_tensors[-1], ttnn.bfloat16))
        ttnn_bias_tensors.append(ttnn.from_torch(torch_bias_tensors[-1], ttnn.bfloat16))
        padding_n4 = (padding[0], padding[0], padding[1], padding[1])
        current_torch_output = torch.nn.functional.relu(
            torch.nn.functional.conv2d(
                current_torch_output,
                torch_weights_tensors[-1],
                bias=torch_bias_tensors[-1].reshape(-1),
                stride=stride,
                padding=padding,
                dilation=dilation,
            )
        )
        op_slicing_attrs.append(
            ttnn.Conv2dSliceAttr(
                batch_size=batch_size,
                input_shape=[input_height, input_width],
                input_channels=current_input_channels,
                output_channels=output_channels,
                kernel_size=tuple(kernel),
                stride=tuple(stride),
                padding_n4=tuple(padding_n4),
                dilation=tuple(dilation),
                groups=1,
                input_layout=input_layout,
                input_dtype=dtype,
                output_dtype=dtype,
                weight_tensor=ttnn_weights_tensors[-1],
                device=device,
                bias_tensor=ttnn_bias_tensors[-1],
                conv_config=conv_config,
                compute_config=compute_config,
            )
        )
        current_input_channels = output_channels
    print(current_torch_output.shape)
    ref = current_torch_output
    [out_batch, out_channels, out_height, out_width] = current_torch_output.shape
    tt_output_tensor = ttnn.zeros(
        [out_batch, out_height, out_width, out_channels],
        dtype=dtype,
        layout=input_layout,
        device=device,
    )
    ttnn.run_sliced_op(
        input_tensor=tt_input_tensor,
        output_tensor=tt_output_tensor,
        op_slice_attr=op_slicing_attrs,
        dram_slice_config=ttnn.Conv2dSliceConfig(slice_type=slice_type, num_slices=num_slices),
    )
    threshold = 0.99
    ref = torch.permute(ref, (0, 2, 3, 1))

    tt_output_tensor_host = ttnn.from_device(tt_output_tensor)
    out = ttnn.to_torch(tt_output_tensor_host)
    out = out.reshape(batch_size, out_height, out_width, out.shape[-1])
    out = out[:, :, :, : ref.shape[-1]]

    logger.info(f"Threshold: {threshold}")
    passing, pcc_msg = check_with_pcc_without_tensor_printout(out, ref, pcc=threshold)
    logger.info(f"PCC = {pcc_msg}. Threshold = {threshold}")
    if not passing:
        torch.set_printoptions(sci_mode=False)
        diff = torch.abs(out - ref) / ref.max()
        assert passing, f"Test failed with PCC = {pcc_msg}, below threshold {threshold}"
