# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import ttnn

from loguru import logger

from models.experimental.transfuser.reference.config import GlobalConfig
from models.experimental.transfuser.reference.lidar_center_net import LidarCenterNetHead
from models.experimental.transfuser.tt.head import TTLidarCenterNetHead

from ttnn.model_preprocessing import preprocess_model_parameters
from models.common.utility_functions import (
    tt2torch_tensor,
)
from tests.ttnn.utils_for_testing import check_with_pcc


def create_lidar_center_net_head_preprocessor(device, weight_dtype=ttnn.bfloat16):
    def custom_preprocessor(torch_model, name, ttnn_module_args):
        parameters = {}

        # Process each head's parameters
        for head_name in [
            "heatmap_head",
            "wh_head",
            "offset_head",
            "yaw_class_head",
            "yaw_res_head",
            "velocity_head",
            "brake_head",
        ]:
            if head_name == "heatmap_head":
                weight_dtype = ttnn.float32
            if hasattr(torch_model, head_name):
                head = getattr(torch_model, head_name)

                # Get output channels for this head
                out_channels = head[2].weight.shape[0]  # From second conv layer

                # Note: We cannot use prepare_conv_weights here because we need
                # the full conv2d parameters (batch_size, input_height, etc.)
                # which are only available at runtime, not during preprocessing.
                # So we keep weights in PyTorch format and convert at runtime.
                parameters[head_name] = {}

                # Store weights in PyTorch format - will be prepared during first forward pass
                parameters[head_name]["conv1_weight"] = ttnn.from_torch(
                    head[0].weight, dtype=weight_dtype, layout=ttnn.ROW_MAJOR_LAYOUT
                )
                parameters[head_name]["conv1_bias"] = ttnn.from_torch(
                    head[0].bias.reshape(1, 1, 1, -1), dtype=weight_dtype, layout=ttnn.ROW_MAJOR_LAYOUT
                )

                parameters[head_name]["conv2_weight"] = ttnn.from_torch(
                    head[2].weight, dtype=weight_dtype, layout=ttnn.ROW_MAJOR_LAYOUT
                )
                parameters[head_name]["conv2_bias"] = ttnn.from_torch(
                    head[2].bias.reshape(1, 1, 1, -1), dtype=weight_dtype, layout=ttnn.ROW_MAJOR_LAYOUT
                )

        return parameters

    return custom_preprocessor


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
@pytest.mark.parametrize(
    "batch_size, in_channels, feat_height, feat_width, num_classes",
    [
        (1, 64, 64, 64, 1),
    ],
)
@pytest.mark.parametrize("input_dtype", [ttnn.bfloat16])
@pytest.mark.parametrize("weight_dtype", [ttnn.float32])
def test_lidar_center_net_head(
    device,
    batch_size,
    in_channels,
    feat_height,
    feat_width,
    num_classes,
    input_dtype,
    weight_dtype,
):
    # Create synthetic feature input (NCHW format)
    feat_input = torch.randn(batch_size, in_channels, feat_height, feat_width)

    # Create config
    config = GlobalConfig(setting="eval")

    # Initialize reference model
    ref_layer = LidarCenterNetHead(
        in_channel=in_channels,
        feat_channel=in_channels,
        num_classes=num_classes,
        train_cfg=config,
    ).eval()

    # Run reference forward pass - pass as list to test forward() method
    with torch.no_grad():
        ref_outputs = ref_layer.forward([feat_input])

    # Unpack list outputs (each contains one tensor since we have single scale)
    (
        ref_center_heatmap_list,
        ref_wh_list,
        ref_offset_list,
        ref_yaw_class_list,
        ref_yaw_res_list,
        ref_velocity_list,
        ref_brake_list,
    ) = ref_outputs

    # Extract single tensors from lists
    ref_center_heatmap = ref_center_heatmap_list[0]
    ref_wh = ref_wh_list[0]
    ref_offset = ref_offset_list[0]
    ref_yaw_class = ref_yaw_class_list[0]
    ref_yaw_res = ref_yaw_res_list[0]
    ref_velocity = ref_velocity_list[0]
    ref_brake = ref_brake_list[0]

    # Preprocess model parameters
    parameters = preprocess_model_parameters(
        initialize_model=lambda: ref_layer,
        custom_preprocessor=create_lidar_center_net_head_preprocessor(device, weight_dtype),
        device=device,
    )

    # Initialize TTNN model
    tt_layer = TTLidarCenterNetHead(
        device=device,
        parameters=parameters,
        in_channel=in_channels,
        feat_channel=in_channels,
        num_classes=num_classes,
        num_dir_bins=config.num_dir_bins,
    )

    # Convert input to TTNN format (NCHW -> NHWC -> flattened)
    feat_input_nhwc = feat_input.permute(0, 2, 3, 1)  # (1, 64, 64, 64) -> (1, 64, 64, 64)
    feat_input_flattened = feat_input_nhwc.reshape(
        1, 1, batch_size * feat_height * feat_width, in_channels
    )  # (1, 1, 4096, 64)

    tt_feat_input = ttnn.from_torch(
        feat_input_flattened,  # Use flattened version
        device=device,
        layout=ttnn.TILE_LAYOUT,
        dtype=input_dtype,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )

    # Run TTNN forward pass - pass as list to test forward() method
    tt_outputs = tt_layer.forward(
        [tt_feat_input],
        batch_size=batch_size,
        height=feat_height,
        width=feat_width,
    )

    # Unpack list outputs
    (
        tt_center_heatmap_list,
        tt_wh_list,
        tt_offset_list,
        tt_yaw_class_list,
        tt_yaw_res_list,
        tt_velocity_list,
        tt_brake_list,
    ) = tt_outputs

    # Extract single tensors from lists
    tt_center_heatmap = tt_center_heatmap_list[0]
    tt_wh = tt_wh_list[0]
    tt_offset = tt_offset_list[0]
    tt_yaw_class = tt_yaw_class_list[0]
    tt_yaw_res = tt_yaw_res_list[0]
    tt_velocity = tt_velocity_list[0]
    tt_brake = tt_brake_list[0]

    # Convert TTNN outputs back to torch (NHWC -> NCHW)
    tt_center_heatmap_torch = tt2torch_tensor(tt_center_heatmap).permute(0, 3, 1, 2)
    tt_center_heatmap_torch = tt_center_heatmap_torch.reshape(ref_center_heatmap.shape)
    tt_wh_torch = tt2torch_tensor(tt_wh).permute(0, 3, 1, 2)
    tt_wh_torch = tt_wh_torch.reshape(ref_wh.shape)
    tt_offset_torch = tt2torch_tensor(tt_offset).permute(0, 3, 1, 2)
    tt_offset_torch = tt_offset_torch.reshape(ref_offset.shape)
    tt_yaw_class_torch = tt2torch_tensor(tt_yaw_class).permute(0, 3, 1, 2)
    tt_yaw_class_torch = tt_yaw_class_torch.reshape(ref_yaw_class.shape)
    tt_yaw_res_torch = tt2torch_tensor(tt_yaw_res).permute(0, 3, 1, 2)
    tt_yaw_res_torch = tt_yaw_res_torch.reshape(ref_yaw_res.shape)
    tt_velocity_torch = tt2torch_tensor(tt_velocity).permute(0, 3, 1, 2)
    tt_velocity_torch = tt_velocity_torch.reshape(ref_velocity.shape)
    tt_brake_torch = tt2torch_tensor(tt_brake).permute(0, 3, 1, 2)
    tt_brake_torch = tt_brake_torch.reshape(ref_brake.shape)

    # Validate center heatmap
    does_pass, heatmap_pcc_message = check_with_pcc(ref_center_heatmap, tt_center_heatmap_torch, 0.95)
    logger.info(f"Center Heatmap PCC: {heatmap_pcc_message}")
    assert does_pass, f"Center Heatmap PCC check failed: {heatmap_pcc_message}"

    # Validate WH prediction
    does_pass, wh_pcc_message = check_with_pcc(ref_wh, tt_wh_torch, 0.95)
    logger.info(f"WH PCC: {wh_pcc_message}")
    assert does_pass, f"WH PCC check failed: {wh_pcc_message}"

    # Validate offset prediction
    does_pass, offset_pcc_message = check_with_pcc(ref_offset, tt_offset_torch, 0.95)
    logger.info(f"Offset PCC: {offset_pcc_message}")
    assert does_pass, f"Offset PCC check failed: {offset_pcc_message}"

    # Validate yaw class prediction
    does_pass, yaw_class_pcc_message = check_with_pcc(ref_yaw_class, tt_yaw_class_torch, 0.95)
    logger.info(f"Yaw Class PCC: {yaw_class_pcc_message}")
    assert does_pass, f"Yaw Class PCC check failed: {yaw_class_pcc_message}"

    # Validate yaw residual prediction
    does_pass, yaw_res_pcc_message = check_with_pcc(ref_yaw_res, tt_yaw_res_torch, 0.95)
    logger.info(f"Yaw Residual PCC: {yaw_res_pcc_message}")
    assert does_pass, f"Yaw Residual PCC check failed: {yaw_res_pcc_message}"

    # Validate velocity prediction
    does_pass, velocity_pcc_message = check_with_pcc(ref_velocity, tt_velocity_torch, 0.95)
    logger.info(f"Velocity PCC: {velocity_pcc_message}")
    assert does_pass, f"Velocity PCC check failed: {velocity_pcc_message}"

    # Validate brake prediction
    does_pass, brake_pcc_message = check_with_pcc(ref_brake, tt_brake_torch, 0.95)
    logger.info(f"Brake PCC: {brake_pcc_message}")
    assert does_pass, f"Brake PCC check failed: {brake_pcc_message}"

    if does_pass:
        logger.info("LidarCenterNetHead Passed!")
    else:
        logger.warning("LidarCenterNetHead Failed!")
