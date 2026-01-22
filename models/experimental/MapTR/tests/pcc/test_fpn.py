# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn
from loguru import logger
from models.experimental.MapTR.reference.dependency import FPN
from models.experimental.MapTR.resources.download_chkpoint import ensure_checkpoint_downloaded, MAPTR_WEIGHTS_PATH
from models.experimental.MapTR.tt.ttnn_fpn import TtFPN
from models.tt_cnn.tt.builder import Conv2dConfiguration
from tests.ttnn.utils_for_testing import assert_with_pcc

FPN_LAYER = "img_neck."


def load_maptr_fpn_weights(weights_path: str = MAPTR_WEIGHTS_PATH):
    ensure_checkpoint_downloaded(weights_path)

    checkpoint = torch.load(weights_path, map_location="cpu")
    full_state_dict = checkpoint.get("state_dict", checkpoint)

    fpn_weights = {}
    for key, value in full_state_dict.items():
        if key.startswith(FPN_LAYER):
            relative_key = key[len(FPN_LAYER) :]
            fpn_weights[relative_key] = value

    logger.info(f"Loaded {len(fpn_weights)} weight tensors for FPN")
    return fpn_weights


def load_torch_model_maptr(torch_model: FPN, weights_path: str = MAPTR_WEIGHTS_PATH):
    fpn_weights = load_maptr_fpn_weights(weights_path)
    model_state_dict = torch_model.state_dict()
    new_state_dict = {}

    for model_key in model_state_dict.keys():
        if model_key in fpn_weights:
            new_state_dict[model_key] = fpn_weights[model_key]
        else:
            logger.warning(f"Weight not found in checkpoint for: {model_key}")
            new_state_dict[model_key] = model_state_dict[model_key]

    torch_model.load_state_dict(new_state_dict)
    torch_model.eval()
    return torch_model


def create_conv_config_from_conv(
    conv: torch.nn.Conv2d,
    input_height: int,
    input_width: int,
    batch_size: int,
    weight_ttnn: ttnn.Tensor,
    bias_ttnn: ttnn.Tensor = None,
    activation: ttnn.UnaryWithParam = None,
    deallocate_activation: bool = False,
) -> Conv2dConfiguration:
    kernel_size = conv.kernel_size if isinstance(conv.kernel_size, tuple) else (conv.kernel_size, conv.kernel_size)
    stride = conv.stride if isinstance(conv.stride, tuple) else (conv.stride, conv.stride)
    padding = conv.padding if isinstance(conv.padding, tuple) else (conv.padding, conv.padding)
    dilation = conv.dilation if isinstance(conv.dilation, tuple) else (conv.dilation, conv.dilation)

    return Conv2dConfiguration(
        input_height=input_height,
        input_width=input_width,
        in_channels=conv.in_channels,
        out_channels=conv.out_channels,
        batch_size=batch_size,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        groups=conv.groups,
        dilation=dilation,
        weight=weight_ttnn,
        bias=bias_ttnn,
        activation=activation,
        deallocate_activation=deallocate_activation,
    )


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
def test_maptr_fpn(device, reset_seeds):
    in_channels = [2048]
    out_channels = 256
    num_outs = 1

    torch_model = FPN(in_channels=in_channels, out_channels=out_channels, num_outs=num_outs)
    torch_model = load_torch_model_maptr(torch_model)

    batch_size = 1
    height = 12
    width = 20
    input_tensor = torch.randn(batch_size, in_channels[0], height, width)
    inputs = [input_tensor]

    torch_output = torch_model(inputs)

    _, _, input_height, input_width = input_tensor.shape

    # Extract weights and biases for lateral and FPN convolutions
    lateral_conv = torch_model.lateral_convs[0].conv
    lateral_weight_ttnn = ttnn.from_torch(lateral_conv.weight.data, dtype=ttnn.float32)
    lateral_bias_ttnn = None
    if lateral_conv.bias is not None:
        lateral_bias_ttnn = ttnn.from_torch(lateral_conv.bias.data.reshape(1, 1, 1, -1), dtype=ttnn.float32)

    fpn_conv = torch_model.fpn_convs[0].conv
    fpn_weight_ttnn = ttnn.from_torch(fpn_conv.weight.data, dtype=ttnn.float32)
    fpn_bias_ttnn = None
    if fpn_conv.bias is not None:
        fpn_bias_ttnn = ttnn.from_torch(fpn_conv.bias.data.reshape(1, 1, 1, -1), dtype=ttnn.float32)

    lateral_conv_config = create_conv_config_from_conv(
        conv=lateral_conv,
        input_height=input_height,
        input_width=input_width,
        batch_size=batch_size,
        weight_ttnn=lateral_weight_ttnn,
        bias_ttnn=lateral_bias_ttnn,
        deallocate_activation=True,
    )

    fpn_conv_config = create_conv_config_from_conv(
        conv=fpn_conv,
        input_height=input_height,
        input_width=input_width,
        batch_size=batch_size,
        weight_ttnn=fpn_weight_ttnn,
        bias_ttnn=fpn_bias_ttnn,
        deallocate_activation=False,
    )

    tt_model = TtFPN(
        lateral_conv_config=lateral_conv_config,
        fpn_conv_config=fpn_conv_config,
        device=device,
    )

    input_tt = ttnn.from_torch(
        input_tensor.permute(0, 2, 3, 1), layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.bfloat16
    )
    inputs_tt = [input_tt]

    tt_output = tt_model(inputs_tt)

    # Convert TTNN output to PyTorch format and reshape to match expected shape
    tt_output_list = []
    for i, out in enumerate(tt_output):
        out = ttnn.to_layout(out, layout=ttnn.ROW_MAJOR_LAYOUT)

        # Handle sharded tensors if needed
        try:
            if hasattr(out, "is_sharded") and out.is_sharded():
                out = ttnn.sharded_to_interleaved(out)
        except:
            pass

        out_torch = ttnn.to_torch(out)
        expected_shape = torch_output[i].shape

        # Convert TTNN output format to PyTorch NCHW format
        if list(out_torch.shape) != list(expected_shape):
            if len(out_torch.shape) == 4:
                # Handle NHWC format: [1, H, W, C] -> [1, C, H, W]
                if (
                    out_torch.shape[0] == expected_shape[0]
                    and out_torch.shape[1] == expected_shape[2]
                    and out_torch.shape[2] == expected_shape[3]
                    and out_torch.shape[3] == expected_shape[1]
                ):
                    out_torch = out_torch.permute(0, 3, 1, 2)
                # Handle flattened format: [1, C, 1, H*W] -> [1, C, H, W]
                elif (
                    out_torch.shape[0] == expected_shape[0]
                    and out_torch.shape[1] == expected_shape[1]
                    and out_torch.shape[2] == 1
                    and out_torch.shape[3] == input_height * input_width
                ):
                    out_torch = out_torch.squeeze(2).reshape(
                        expected_shape[0], expected_shape[1], input_height, input_width
                    )
                    out_torch = out_torch.permute(0, 2, 3, 1).permute(0, 3, 1, 2)  # NCHW -> NHWC -> NCHW
                # Fallback: reshape to NHWC then permute
                elif out_torch.numel() == torch_output[i].numel():
                    out_torch = out_torch.reshape(
                        expected_shape[0], expected_shape[2], expected_shape[3], expected_shape[1]
                    )
                    out_torch = out_torch.permute(0, 3, 1, 2)

        assert list(out_torch.shape) == list(
            expected_shape
        ), f"Shape mismatch: got {out_torch.shape}, expected {expected_shape}"
        tt_output_list.append(out_torch)

    # Compare TTNN and PyTorch outputs
    for tt_out, torch_out in zip(tt_output_list, torch_output):
        pcc_passed, pcc_message = assert_with_pcc(tt_out, torch_out, 0.99)
        assert pcc_passed, pcc_message
