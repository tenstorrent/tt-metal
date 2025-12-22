# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.

# SPDX-License-Identifier: Apache-2.0

import os

import torch
import torchvision.transforms as transforms
from loguru import logger
from PIL import Image
from torchvision import models
from torchvision.models import MobileNet_V3_Small_Weights
from ttnn.model_preprocessing import preprocess_model_parameters

import ttnn
from models.experimental.mobileNetV3.tests.pcc.common import inverted_residual_setting, last_channel
from models.tt_cnn.tt.pipeline import get_memory_config_for_persistent_dram_tensor
from models.experimental.mobileNetV3.tt.custom_preprocessor import create_custom_preprocessor
from models.experimental.mobileNetV3.tt.ttnn_mobileNetV3 import ttnn_MobileNetV3
from tests.ttnn.utils_for_testing import check_with_pcc


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
            inverted_residual_setting=inverted_residual_setting,
            last_channel=last_channel,
            parameters=self.parameters,
            device=self.device,
        )

        self.torch_output = self.torch_model(self.torch_input_tensor)
        self.torch_input_tensor = self.torch_input_tensor.permute(0, 2, 3, 1)

    def setup_dram_interleaved_input(self, torch_input_tensor=None, mesh_mapper=None):
        # Inputs to MobileNetV3 need to be in ttnn.DRAM_MEMORY_CONFIG for supporting DRAM sliced Conv2d
        mesh_mapper = self.inputs_mesh_mapper if mesh_mapper is None else mesh_mapper
        torch_input_tensor = self.torch_input_tensor if torch_input_tensor is None else torch_input_tensor
        tt_inputs_host = ttnn.from_torch(torch_input_tensor, dtype=ttnn.bfloat16, mesh_mapper=mesh_mapper)
        return tt_inputs_host, ttnn.DRAM_MEMORY_CONFIG

    def setup_sharded_input(self, device, torch_input_tensor=None, mesh_mapper=None, pad_channels=16):
        # Inputs to MobileNetV3 need to be in ttnn.L1_MEMORY_CONFIG for supporting L1 sharded input
        mesh_mapper = self.inputs_mesh_mapper if mesh_mapper is None else mesh_mapper
        torch_input_tensor = self.torch_input_tensor if torch_input_tensor is None else torch_input_tensor

        original_channels = torch_input_tensor.shape[-1]
        if pad_channels and original_channels < pad_channels:
            torch_input_tensor = torch.nn.functional.pad(
                torch_input_tensor, (0, pad_channels - original_channels), value=0
            )

        tt_inputs_host = ttnn.from_torch(torch_input_tensor, dtype=ttnn.bfloat16, mesh_mapper=mesh_mapper)

        # ttnn tensor shape reflects per-device shape when using ShardTensorToMesh
        batch = tt_inputs_host.shape[0]
        height = tt_inputs_host.shape[1]
        width = tt_inputs_host.shape[2]
        channels = tt_inputs_host.shape[3]

        tt_inputs_host = ttnn.reshape(tt_inputs_host, (1, 1, batch * height * width, channels))

        dram_input_mem_config = get_memory_config_for_persistent_dram_tensor(
            tt_inputs_host.shape, ttnn.TensorMemoryLayout.HEIGHT_SHARDED, device.dram_grid_size()
        )

        input_l1_core_grid = ttnn.CoreGrid(x=8, y=8)
        height_dim = tt_inputs_host.shape[-2]

        if height_dim % input_l1_core_grid.num_cores != 0:
            num_cores = input_l1_core_grid.num_cores
            while height_dim % num_cores != 0 and num_cores > 1:
                num_cores -= 1
            y = min(8, num_cores)
            while num_cores % y != 0:
                y -= 1
            x = num_cores // y
            input_l1_core_grid = ttnn.CoreGrid(x=x, y=y)

        l1_input_mem_config = ttnn.create_sharded_memory_config(
            shape=(height_dim // input_l1_core_grid.num_cores, channels),
            core_grid=input_l1_core_grid,
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )

        return tt_inputs_host, dram_input_mem_config, l1_input_mem_config, channels

    def run(self):
        self.tt_output = self.ttnn_model(self.device, self.input_tensor)

    def validate(self, tt_output=None):
        # Validate output tensor
        tt_output = self.tt_output if tt_output is None else tt_output
        tt_output = ttnn.reshape(tt_output, (1, -1))
        tt_output = ttnn.to_torch(tt_output, mesh_composer=self.outputs_mesh_composer)

        self._PCC_THRESH = 0.98
        self.pcc_passed = self.pcc_message = []

        logger.info(f"MobileNet V3: batch_size={self.batch_size}, ")

        torch_output_ref = self.torch_output
        if tt_output.shape[0] > torch_output_ref.shape[0]:
            # Multi-device validation
            for i in range(tt_output.shape[0]):
                passed, msg = check_with_pcc(torch_output_ref[0:1], tt_output[i : i + 1], pcc=self._PCC_THRESH)
                self.pcc_passed.append(passed)
                self.pcc_message.append(msg)
                logger.info(f"MobileNet V3 device {i}: PCC {msg}, shape={tt_output[i:i+1].shape}")
        else:
            # Single-device validation
            passed, msg = check_with_pcc(torch_output_ref, tt_output, pcc=self._PCC_THRESH)
            self.pcc_passed.append(passed)
            self.pcc_message.append(msg)
            logger.info(f"MobileNet V3: PCC {msg}, shape={tt_output.shape}")

        assert all(self.pcc_passed), logger.error(f"MobileNet V3 PCC check failed: {self.pcc_message}")

    def dealloc_output(self):
        ttnn.deallocate(self.tt_output)
