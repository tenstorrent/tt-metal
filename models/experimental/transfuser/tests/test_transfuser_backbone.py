# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import torch
import pytest

import ttnn
from loguru import logger

from models.experimental.transfuser.reference.config import GlobalConfig
from models.experimental.transfuser.reference.transfuser_backbone import TransfuserBackbone
from models.experimental.transfuser.tt.custom_preprocessing import create_custom_mesh_preprocessor
from models.experimental.transfuser.tt.transfuser_backbone import TtTransfuserBackbone
from ttnn.model_preprocessing import (
    preprocess_model_parameters,
)
from models.utility_functions import (
    tt2torch_tensor,
)
from tests.ttnn.utils_for_testing import check_with_pcc


def get_mesh_mappers(device):
    if device.get_num_devices() != 1:
        return (
            ttnn.ShardTensorToMesh(device, dim=0),
            None,
            ttnn.ConcatMeshToTensor(device, dim=0),
        )
    return None, None, None


@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
@pytest.mark.parametrize(
    "image_architecture, lidar_architecture, n_layer, use_velocity, use_target_point_image, img_input_shape, lidar_input_shape",
    [("regnety_032", "regnety_032", 4, False, True, (1, 3, 160, 704), (1, 3, 256, 256))],  # GPT-SelfAttention 1
)
# @pytest.mark.parametrize("input_dtype", [ttnn.bfloat16])
# @pytest.mark.parametrize("weight_dtype", [ttnn.bfloat16])
def test_transfuser_backbone(
    device,
    image_architecture,
    lidar_architecture,
    n_layer,
    use_velocity,
    use_target_point_image,
    img_input_shape,
    lidar_input_shape,
    # input_dtype,
    # weight_dtype,
    # model_config,
):
    num_devices = device.get_num_devices()
    # batch_size = batch_size * num_devices
    inputs_mesh_mapper, weights_mesh_mapper, output_mesh_composer = get_mesh_mappers(device)
    # self.name = name

    image_input = torch.randn(img_input_shape)
    lidar_input = torch.randn(lidar_input_shape)
    velocity_input = torch.randn(1, 1)

    # setting machine to avoid loading files
    config = GlobalConfig(setting="eval")
    config.n_layer = n_layer
    if use_target_point_image:
        config.use_target_point_image = use_target_point_image

    ref_layer = TransfuserBackbone(
        config,
        image_architecture=image_architecture,
        lidar_architecture=lidar_architecture,
        use_velocity=use_velocity,
    ).eval()

    features = ref_layer(image_input, lidar_input, velocity_input)
    # features, image_features_grid, fused_features = ref_layer(image_input, lidar_input, velocity_input)

    parameters = preprocess_model_parameters(
        initialize_model=lambda: ref_layer,
        custom_preprocessor=create_custom_mesh_preprocessor(weights_mesh_mapper),
        # custom_preprocessor=create_gpt_preprocessor(device, n_layer, weight_dtype),
        device=device,
    )
    # print(f"{parameters=}")

    # pytest.skip("Skipping test_transfuser_backbone")

    tt_layer = TtTransfuserBackbone(
        parameters=parameters,
        stride=2,
        # model_config =model_config,
    )
    tt_image_input = ttnn.from_torch(
        image_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        mesh_mapper=inputs_mesh_mapper,
    )
    output_tensor = tt_layer(tt_image_input, device=device)

    # ttnn.deallocate(output_tensor)
    expected_shape = features.shape
    print(".............................................................")
    print(expected_shape)
    # tt_output_tensor_torch = torch.reshape(
    #         tt_output_tensor_torch,
    #         (expected_shape[0], expected_shape[2], expected_shape[3], expected_shape[1]),
    #     )

    # Convert to PyTorch format (N,C,H,W) for comparison
    # output_tensor = ttnn.permute(output_tensor, SmallVector<int64_t>{0, 3, 1, 2})

    tt_torch_output = tt2torch_tensor(output_tensor)

    does_pass, image_out_pcc_message = check_with_pcc(tt_torch_output, features, 0.95)

    logger.info(f"Image Output PCC: {image_out_pcc_message}")
    assert does_pass, f"PCC check failed: {image_out_pcc_message}"

    if does_pass:
        logger.info("GPT Passed!")
    else:
        logger.warning("GPT Failed!")
