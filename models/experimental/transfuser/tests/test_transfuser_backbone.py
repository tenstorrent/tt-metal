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
from tests.ttnn.utils_for_testing import check_with_pcc


class TransfuserBackboneInfra:
    def __init__(
        self,
        device,
        image_architecture,
        lidar_architecture,
        n_layer,
        use_velocity,
        use_target_point_image,
        img_input_shape,
        lidar_input_shape,
        model_config,
    ):
        super().__init__()
        self._init_seeds()
        self.device = device
        self.n_layer = n_layer
        self.image_arch = image_architecture
        self.lidar_arch = lidar_architecture
        self.use_velocity = use_velocity
        self.img_input_shape = img_input_shape
        self.lidar_input_shape = lidar_input_shape
        self.num_devices = device.get_num_devices()
        # self.batch_size = batch_size * self.num_devices
        self.inputs_mesh_mapper, self.weights_mesh_mapper, self.output_mesh_composer = self.get_mesh_mappers(device)
        # self.name = name

        # setting machine to avoid loading files
        self.config = GlobalConfig(setting="eval")
        self.config.n_layer = self.n_layer
        if use_target_point_image:
            self.config.use_target_point_image = use_target_point_image

        # Build reference torch model
        torch_model = TransfuserBackbone(
            self.config,
            image_architecture=self.image_arch,
            lidar_architecture=self.lidar_arch,
            use_velocity=self.use_velocity,
        )

        # Preprocess parameters for TTNN
        parameters = preprocess_model_parameters(
            initialize_model=lambda: torch_model,
            custom_preprocessor=create_custom_mesh_preprocessor(self.weights_mesh_mapper),
            device=None,
        )

        # Prepare golden inputs/outputs
        self.torch_image_input = torch.randn(self.img_input_shape)
        self.torch_lidar_input = torch.randn(self.lidar_input_shape)
        self.torch_velocity_input = torch.randn(1, 1)
        self.torch_image_output, self.torch_lidar_output = torch_model(
            # self.torch_output_tensor = torch_model(
            self.torch_image_input,
            self.torch_lidar_input,
            self.torch_velocity_input,
        )

        # Convert input to TTNN format
        tt_image_input = ttnn.from_torch(
            self.torch_image_input,
            # self.torch_image_input.permute(0, 2, 3, 1),
            dtype=ttnn.bfloat16,
            mesh_mapper=self.inputs_mesh_mapper,
        )
        tt_lidar_input = ttnn.from_torch(
            self.torch_lidar_input,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            mesh_mapper=self.inputs_mesh_mapper,
        )
        self.input_image_tensor = ttnn.to_device(tt_image_input, device)
        self.input_lidar_tensor = ttnn.to_device(tt_lidar_input, device)

        # Build TTNN model
        self.ttnn_model = TtTransfuserBackbone(
            parameters=parameters,
            stride=2,
            model_config=model_config,
        )

        # Run + validate
        self.run()
        self.validate(model_config)

    def _init_seeds(self):
        if not hasattr(self, "_model_initialized"):
            torch.manual_seed(42)
            torch.cuda.manual_seed_all(42)
            torch.backends.cudnn.deterministic = True
            self._model_initialized = True

    def get_mesh_mappers(self, device):
        if device.get_num_devices() != 1:
            return (
                ttnn.ShardTensorToMesh(device, dim=0),
                None,
                ttnn.ConcatMeshToTensor(device, dim=0),
            )
        return None, None, None

    def run(self):
        self.output_image_tensor, self.output_lidar_tensor = self.ttnn_model(
            self.input_image_tensor, self.input_lidar_tensor, self.device
        )
        return self.output_image_tensor, self.output_lidar_tensor

    def validate(self, model_config, output_tensor=None):
        # Validate image output
        tt_image_tensor_torch = ttnn.to_torch(
            self.output_image_tensor,
            device=self.device,
            mesh_composer=self.output_mesh_composer,
        )

        # Validate lidar output
        tt_lidar_tensor_torch = ttnn.to_torch(
            self.output_lidar_tensor,
            device=self.device,
            mesh_composer=self.output_mesh_composer,
        )

        # Deallocate output tensors
        ttnn.deallocate(self.output_image_tensor)
        ttnn.deallocate(self.output_lidar_tensor)

        # Reshape + permute image output back to NCHW
        expected_image_shape = self.torch_image_output.shape
        tt_image_tensor_torch = torch.reshape(
            tt_image_tensor_torch,
            (expected_image_shape[0], expected_image_shape[2], expected_image_shape[3], expected_image_shape[1]),
        )
        tt_image_tensor_torch = torch.permute(tt_image_tensor_torch, (0, 3, 1, 2))

        # Reshape + permute lidar output back to NCHW
        expected_lidar_shape = self.torch_lidar_output.shape
        tt_lidar_tensor_torch = torch.reshape(
            tt_lidar_tensor_torch,
            (expected_lidar_shape[0], expected_lidar_shape[2], expected_lidar_shape[3], expected_lidar_shape[1]),
        )
        tt_lidar_tensor_torch = torch.permute(tt_lidar_tensor_torch, (0, 3, 1, 2))

        # PCC validation for both outputs
        image_pcc_passed, image_pcc_message = check_with_pcc(self.torch_image_output, tt_image_tensor_torch, pcc=0.99)
        lidar_pcc_passed, lidar_pcc_message = check_with_pcc(self.torch_lidar_output, tt_lidar_tensor_torch, pcc=0.99)

        logger.info(f"Image Output PCC: {image_pcc_message}")
        logger.info(f"LiDAR Output PCC: {lidar_pcc_message}")

        # Both outputs must pass for overall validation to pass
        overall_pcc_passed = image_pcc_passed and lidar_pcc_passed

        assert overall_pcc_passed, logger.error(
            f"PCC check failed - Image: {image_pcc_message}, LiDAR: {lidar_pcc_message}"
        )

        logger.info(
            f"act_dtype={model_config['ACTIVATIONS_DTYPE']}, "
            f"weight_dtype={model_config['WEIGHTS_DTYPE']}, "
            f"math_fidelity={model_config['MATH_FIDELITY']}, "
            f"Image PCC={image_pcc_message}, LiDAR PCC={lidar_pcc_message}"
        )

        return overall_pcc_passed, f"Image: {image_pcc_message}, LiDAR: {lidar_pcc_message}"


# Default model config
model_config = {
    "MATH_FIDELITY": ttnn.MathFidelity.LoFi,
    "WEIGHTS_DTYPE": ttnn.bfloat8_b,
    "ACTIVATIONS_DTYPE": ttnn.bfloat8_b,
}


@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
@pytest.mark.parametrize(
    "image_architecture, lidar_architecture, n_layer, use_velocity, use_target_point_image, img_input_shape, lidar_input_shape",
    [("regnety_032", "regnety_032", 4, False, True, (1, 3, 160, 704), (1, 3, 256, 256))],
)
def test_stem(
    device,
    image_architecture,
    lidar_architecture,
    n_layer,
    use_velocity,
    use_target_point_image,
    img_input_shape,
    lidar_input_shape,
):
    TransfuserBackboneInfra(
        device,
        image_architecture,
        lidar_architecture,
        n_layer,
        use_velocity,
        use_target_point_image,
        img_input_shape,
        lidar_input_shape,
        model_config,
    )
