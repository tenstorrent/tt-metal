# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import torch
from loguru import logger
from ttnn.model_preprocessing import preprocess_model_parameters
from ultralytics import YOLO

import ttnn
from models.demos.yolov8s.tt.tt_yolov8s_utils import create_custom_mesh_preprocessor
from models.demos.yolov8s.tt.ttnn_yolov8s import TtYolov8sModel
from models.utility_functions import divup, is_wormhole_b0
from tests.ttnn.utils_for_testing import assert_with_pcc


def load_torch_model():
    torch_model = YOLO("yolov8s.pt")
    torch_model = torch_model.model
    torch_model.eval()
    return torch_model


class YOLOv8sPerformanceRunnerInfra:
    def __init__(
        self,
        device,
        batch_size,
        model_location_generator=None,
        mesh_mapper=None,
        mesh_composer=None,
        weights_mesh_mapper=None,
    ):
        super().__init__()
        torch.manual_seed(0)
        self.pcc_passed = False
        self.pcc_message = "Did you forget to call validate()?"
        self.device = device
        self.batch_size = batch_size
        self.model_location_generator = model_location_generator
        torch_model = load_torch_model()
        self.mesh_mapper = mesh_mapper
        self.mesh_composer = mesh_composer
        self.weights_mesh_mapper = weights_mesh_mapper
        self.num_devices = device.get_num_devices()
        input_shape = (batch_size, 640, 640, 3)
        inp_h, inp_w = input_shape[1], input_shape[2]

        state_dict = torch_model.state_dict()

        parameters = preprocess_model_parameters(
            initialize_model=lambda: torch_model,
            custom_preprocessor=create_custom_mesh_preprocessor(device, self.weights_mesh_mapper),
            device=device,
        )
        print(f"parameters: {parameters['model.0']}")
        self.ttnn_yolov8_model = TtYolov8sModel(device=device, parameters=parameters, res=(inp_h, inp_w))

        torch_input_tensor = torch.randn(input_shape, dtype=torch.float32)
        for i in range(self.num_devices - 1):
            torch_input_tensor = torch.cat([torch_input_tensor] * self.num_devices, dim=0)

        self.tt_input_tensor = ttnn.from_torch(torch_input_tensor, ttnn.bfloat16, mesh_mapper=self.mesh_mapper)
        self.torch_input_tensor = torch_input_tensor.permute(0, 3, 1, 2)
        self.torch_output_tensor = torch_model(self.torch_input_tensor)

    def run(self):
        self.output_tensor = self.ttnn_yolov8_model(self.input_tensor)  # [0]

    def _setup_l1_sharded_input(self, device, torch_input_tensor=None):
        if is_wormhole_b0():
            core_grid = ttnn.CoreGrid(y=8, x=8)
        else:  # BH
            core_grid = ttnn.CoreGrid(y=12, x=10)
            # exit("Unsupported device")
        torch_input_tensor = self.torch_input_tensor if torch_input_tensor is None else torch_input_tensor

        n, c, h, w = torch_input_tensor.shape
        if c == 3:
            c = 16
        input_mem_config = ttnn.create_sharded_memory_config(
            [n // self.num_devices, c, h, w],
            ttnn.CoreGrid(x=8, y=8),
            ttnn.ShardStrategy.HEIGHT,
        )

        tt_inputs_host = ttnn.from_torch(
            torch_input_tensor, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, mesh_mapper=self.mesh_mapper
        )

        return tt_inputs_host, input_mem_config

    def _setup_dram_sharded_input(self, device, torch_input_tensor=None):
        tt_inputs_host, input_mem_config = self._setup_l1_sharded_input(device)
        dram_grid_size = device.dram_grid_size()
        dram_shard_spec = ttnn.ShardSpec(
            ttnn.CoreRangeSet(
                {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(dram_grid_size.x - 1, dram_grid_size.y - 1))}
            ),
            [
                divup(tt_inputs_host.volume() // tt_inputs_host.shape[-1], (dram_grid_size.x * dram_grid_size.y)),
                tt_inputs_host.shape[-1],
            ],
            ttnn.ShardOrientation.ROW_MAJOR,
        )
        sharded_mem_config_DRAM = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.DRAM, dram_shard_spec
        )

        return tt_inputs_host, sharded_mem_config_DRAM, input_mem_config

    def validate(self, output_tensor=None, torch_output_tensor=None):
        ttnn_output_tensor = self.output_tensor if output_tensor is None else output_tensor
        torch_output_tensor = self.torch_output_tensor if torch_output_tensor is None else torch_output_tensor
        output_tensor = ttnn.to_torch(ttnn_output_tensor[0], mesh_composer=self.mesh_composer)
        self.pcc_passed, self.pcc_message = assert_with_pcc(self.torch_output_tensor[0], output_tensor, pcc=0.99)

        logger.info(f"Yolov8s - batch_size={self.batch_size}, PCC={self.pcc_message}")

    def dealloc_output(self):
        ttnn.deallocate(self.output_tensor[0])
        for t in self.output_tensor[1]:
            if isinstance(t, list):
                for sub_t in t:
                    ttnn.deallocate(sub_t)
            else:
                ttnn.deallocate(t)
