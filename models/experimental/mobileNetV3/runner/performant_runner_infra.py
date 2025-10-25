# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.

# SPDX-License-Identifier: Apache-2.0

import torch
from loguru import logger

import ttnn
import os
from models.experimental.mobileNetV3.tt.ttnn_mobileNetV3 import ttnn_MobileNetV3
from models.experimental.mobileNetV3.tests.pcc.common import inverted_residual_setting, last_channel
from torchvision import models
from torchvision.models import MobileNet_V3_Small_Weights
from models.experimental.mobileNetV3.tt.custom_preprocessor import create_custom_preprocessor
from ttnn.model_preprocessing import preprocess_model_parameters
from tests.ttnn.utils_for_testing import check_with_pcc
from PIL import Image
import torchvision.transforms as transforms


class MobileNetV3PerformanceRunnerInfra:
    def __init__(
        self,
        device,
        batch_size,
        model_location_generator=None,
        resolution=(224, 224),
        torch_input_tensor=None,
        inputs_mesh_mapper=None,
        weights_mesh_mapper=None,
        outputs_mesh_composer=None,
        input_path=None,
    ):
        if not hasattr(self, "_model_initialized"):
            torch.manual_seed(42)
            self._model_initialized = True
            torch.cuda.manual_seed_all(42)
            torch.backends.cudnn.deterministic = True

        self.resolution = resolution
        self.pcc_passed = False
        self.pcc_message = "Did you forget to call validate()?"
        self.device = device
        self.batch_size = batch_size
        self.model_location_generator = model_location_generator
        self.torch_input_tensor = torch_input_tensor

        self.num_devices = device.get_num_devices()
        self.inputs_mesh_mapper = inputs_mesh_mapper
        self.weights_mesh_mapper = weights_mesh_mapper
        self.outputs_mesh_composer = outputs_mesh_composer
        self.real_input_path = input_path

        self.torch_model = models.mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1)

        # Create input tensor
        if self.real_input_path and os.path.exists(self.real_input_path):
            img = Image.open(self.real_input_path).convert("RGB")

            preprocess = transforms.Compose(
                [
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ]
            )

            input_tensor = preprocess(img)
            self.torch_input_tensor = input_tensor.unsqueeze(0)

            # Verify shape matches expected dimensions
            expected_shape = (batch_size * self.num_devices, 3, self.resolution[0], self.resolution[1])
            if self.torch_input_tensor.shape != expected_shape:
                logger.warning(
                    f"Input shape mismatch. Expected: {expected_shape}, Got: {self.torch_input_tensor.shape}"
                )
        else:
            self.torch_input_tensor = torch.randn(
                (self.batch_size, 3, self.resolution[0], self.resolution[1]), dtype=torch.float32
            )

        # Preprocess model parameters
        self.parameters = preprocess_model_parameters(
            initialize_model=lambda: self.torch_model,
            custom_preprocessor=create_custom_preprocessor(self.weights_mesh_mapper),
            device=None,
        )

        self.ttnn_model = ttnn_MobileNetV3(
            inverted_residual_setting=inverted_residual_setting, last_channel=last_channel, parameters=self.parameters
        )

        self.torch_output = self.torch_model(self.torch_input_tensor)

        self.torch_input_tensor = self.torch_input_tensor.permute(0, 2, 3, 1)

    def setup_dram_interleaved_input(self, torch_input_tensor=None, mesh_mapper=None):
        # Inputs to MobileNetV3 need to be in ttnn.DRAM_MEMORY_CONFIG for supporting DRAM sliced Conv2d
        mesh_mapper = self.inputs_mesh_mapper if mesh_mapper is None else mesh_mapper
        torch_input_tensor = self.torch_input_tensor if torch_input_tensor is None else torch_input_tensor
        tt_inputs_host = ttnn.from_torch(torch_input_tensor, dtype=ttnn.bfloat16, mesh_mapper=self.inputs_mesh_mapper)
        return tt_inputs_host, ttnn.DRAM_MEMORY_CONFIG

    def run(self):
        self.tt_output = self.ttnn_model(self.device, self.input_tensor)

    def validate(self, tt_output=None):
        """Validate output in a uniform loop."""

        tt_output = self.tt_output if tt_output is None else tt_output

        tt_output = ttnn.reshape(tt_output, (1, -1))
        tt_output = ttnn.to_torch(tt_output)

        self._PCC_THRESH = 0.98
        self.pcc_passed = self.pcc_message = []

        logger.info(f"MobileNet V3: batch_size={self.batch_size}, ")

        passed, msg = check_with_pcc(self.torch_output, tt_output, pcc=self._PCC_THRESH)
        self.pcc_passed.append(passed)
        self.pcc_message.append(msg)
        logger.info(f"MobileNet V3 : " f"PCC {msg}, " f"shape={tt_output.shape}")

        assert all(self.pcc_passed), logger.error(f"MobileNet V3 PCC check failed: {self.pcc_message}")

    def dealloc_output(self):
        ttnn.deallocate(self.tt_output)
