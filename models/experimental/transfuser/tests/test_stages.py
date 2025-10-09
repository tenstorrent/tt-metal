# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import torch
import pytest

import ttnn
from loguru import logger

from models.experimental.transfuser.reference.config import GlobalConfig
from models.experimental.transfuser.reference.stage import Stage
from models.experimental.transfuser.tt.custom_preprocessing import create_custom_mesh_preprocessor
from models.experimental.transfuser.tt.stages import Ttstages
from ttnn.model_preprocessing import (
    preprocess_model_parameters,
)
from tests.ttnn.utils_for_testing import check_with_pcc


class StageInfra:
    def __init__(
        self,
        device,
        stage_name,
        input_shape,
    ):
        super().__init__()
        self._init_seeds()
        self.device = device
        self.stage_name = stage_name
        self.input_shape = input_shape
        self.num_devices = device.get_num_devices()
        # self.batch_size = batch_size * self.num_devices
        self.inputs_mesh_mapper, self.weights_mesh_mapper, self.output_mesh_composer = self.get_mesh_mappers(device)
        self.config = GlobalConfig(setting="eval")

        # Assert that only layer1 to layer2 is specifically selected
        assert stage_name in [
            "layer1",
            "layer2",
            "layer3",
            "layer4",
        ], f"Only layer1 to layer4 exists, got: {stage_name}"

        # Build reference torch model
        torch_model = Stage(
            self.config,
            stage_name=stage_name,
            image_architecture="regnety_032",
        )
        torch_model.eval()

        # Prepare golden inputs/outputs
        self.torch_input = torch.randn(self.input_shape)
        with torch.no_grad():
            self.torch_output = torch_model(
                self.torch_input,
            )

        # Preprocess parameters for TTNN
        parameters = preprocess_model_parameters(
            initialize_model=lambda: torch_model,
            custom_preprocessor=create_custom_mesh_preprocessor(self.weights_mesh_mapper),
            device=None,
        )
        parameters = getattr(parameters, stage_name)
        # Build TTNN model
        self.ttnn_model = Ttstages(
            parameters=parameters,
            stride=2,
            model_config=model_config,
            stage_name=stage_name,
        )

        # Convert input to TTNN format
        self.tt_input = ttnn.from_torch(
            # self.torch_input.permute(0, 2, 3, 1),
            self.torch_input,
            dtype=ttnn.bfloat16,
            mesh_mapper=self.inputs_mesh_mapper,
        )
        # self.input_tensor = ttnn.to_device(tt_input, device)

        self.tt_input = ttnn.to_device(self.tt_input, device)
        self.tt_input = ttnn.permute(self.tt_input, (0, 2, 3, 1))

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
        self.output_tensor = self.ttnn_model(self.tt_input, self.device)
        return self.output_tensor

    def validate(self, model_config, output_tensor=None):
        # Validate image output
        tt_tensor_torch = ttnn.to_torch(
            self.output_tensor,
            device=self.device,
            mesh_composer=self.output_mesh_composer,
        )

        # Deallocate output tensors
        ttnn.deallocate(self.output_tensor)

        # Reshape + permute image output back to NCHW
        expected_shape = self.torch_output.shape
        tt_tensor_torch = torch.reshape(
            tt_tensor_torch,
            (expected_shape[0], expected_shape[2], expected_shape[3], expected_shape[1]),
        )
        tt_tensor_torch = torch.permute(tt_tensor_torch, (0, 3, 1, 2))

        # PCC validation for both outputs
        image_pcc_passed, image_pcc_message = check_with_pcc(self.torch_output, tt_tensor_torch, pcc=0.99)

        logger.info(f"Image Output PCC: {image_pcc_message}")
        assert image_pcc_passed, logger.error(f"PCC check failed - pcc_message: {image_pcc_message}")

        logger.info(
            f"act_dtype={model_config['ACTIVATIONS_DTYPE']}, "
            f"weight_dtype={model_config['WEIGHTS_DTYPE']}, "
            f"math_fidelity={model_config['MATH_FIDELITY']}, "
            f"Image PCC={image_pcc_message},"
        )

        return image_pcc_passed, f"Image: {image_pcc_message}"


# High accuracy model config
model_config = {
    "MATH_FIDELITY": ttnn.MathFidelity.HiFi4,
    "WEIGHTS_DTYPE": ttnn.bfloat16,
    "ACTIVATIONS_DTYPE": ttnn.bfloat16,
    "fp32_dest_acc_en": True,
    "packer_l1_acc": True,
    "math_approx_mode": False,
}


@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
@pytest.mark.parametrize(
    "stage_name,input_shape",
    [
        # ImageCNN Tests
        ("layer1", (1, 32, 80, 352)),
        ("layer2", (1, 72, 40, 176)),
        ("layer3", (1, 216, 20, 88)),
        ("layer4", (1, 576, 10, 44)),
        # LidarEncoder Tests
        ("layer1", (1, 32, 128, 128)),
        ("layer2", (1, 72, 64, 64)),
        ("layer3", (1, 216, 32, 32)),
        ("layer4", (1, 576, 16, 16)),
    ],
)
def test_stage(
    device,
    stage_name,
    input_shape,
):
    StageInfra(
        device,
        stage_name,
        input_shape,
    )
