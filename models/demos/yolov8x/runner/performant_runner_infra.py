# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0


import torch
from loguru import logger

import ttnn
from models.demos.yolov8x.common import load_torch_model
from models.demos.yolov8x.tt.ttnn_yolov8x import TtYolov8xModel
from models.demos.yolov8x.tt.ttnn_yolov8x_utils import custom_preprocessor
from models.utility_functions import divup, is_wormhole_b0
from tests.ttnn.utils_for_testing import assert_with_pcc


def load_ttnn_model(device, torch_model, weights_mesh_mapper=None):
    state_dict = torch_model.state_dict()
    parameters = custom_preprocessor(device, state_dict, mesh_mapper=weights_mesh_mapper)
    ttnn_model = TtYolov8xModel(device=device, parameters=parameters)
    return ttnn_model


class YOLOv8xPerformanceRunnerInfra:
    def __init__(
        self,
        device,
        batch_size,
        model_location_generator=None,
        inputs_mesh_mapper=None,
        weights_mesh_mapper=None,
        outputs_mesh_composer=None,
    ):
        super().__init__()
        torch.manual_seed(0)
        self.pcc_passed = False
        self.pcc_message = "Did you forget to call validate()?"
        self.device = device
        self.batch_size = batch_size
        self.num_devices = device.get_num_devices()
        self.inputs_mesh_mapper = inputs_mesh_mapper
        self.weights_mesh_mapper = weights_mesh_mapper
        self.outputs_mesh_composer = outputs_mesh_composer
        self.model_location_generator = model_location_generator
        torch_model = load_torch_model(model_location_generator=self.model_location_generator)
        self.ttnn_yolov8_model = load_ttnn_model(
            device=self.device, torch_model=torch_model, weights_mesh_mapper=self.weights_mesh_mapper
        )
        input_shape = (self.batch_size, 3, 640, 640)
        self.torch_input_tensor = torch.randn(input_shape, dtype=torch.float32)
        self.tt_input_tensor = ttnn.from_torch(
            self.torch_input_tensor, ttnn.bfloat16, mesh_mapper=self.inputs_mesh_mapper
        )
        self.torch_output_tensor = torch_model(self.torch_input_tensor)[0]

    def run(self):
        self.output_tensor = self.ttnn_yolov8_model(self.input_tensor)[0]

    def setup_l1_sharded_input(self, device, torch_input_tensor=None, min_channels=16):
        if is_wormhole_b0():
            core_grid = ttnn.CoreGrid(y=8, x=8)
        else:
            exit("Unsupported device")

        torch_input_tensor = self.torch_input_tensor if torch_input_tensor is None else torch_input_tensor

        n, c, h, w = torch_input_tensor.shape
        if c < min_channels:
            c = min_channels
        elif c % min_channels != 0:
            c = ((c // min_channels) + 1) * min_channels
        n = n // self.num_devices if n // self.num_devices != 0 else n

        input_mem_config = ttnn.create_sharded_memory_config(
            [n, c, h, w],
            ttnn.CoreGrid(x=8, y=8),
            ttnn.ShardStrategy.HEIGHT,
        )

        assert torch_input_tensor.ndim == 4, "Expected input tensor to have shape (BS, C, H, W)"
        tt_inputs_host = ttnn.from_torch(torch_input_tensor, ttnn.bfloat16, mesh_mapper=self.inputs_mesh_mapper)

        return tt_inputs_host, input_mem_config

    def setup_dram_sharded_input(self, device, torch_input_tensor=None, mesh_mapper=None, mesh_composer=None):
        tt_inputs_host, input_mem_config = self.setup_l1_sharded_input(device)
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

    def validate(self, output_tensor=None):
        if output_tensor is None:
            output_tensor = self.output_tensor
        else:
            if isinstance(output_tensor, (list, tuple)):
                output_tensor = output_tensor[0]

        output_tensor = ttnn.to_torch(output_tensor, mesh_composer=self.outputs_mesh_composer)

        valid_pcc = 0.978
        self.pcc_passed, self.pcc_message = assert_with_pcc(self.torch_output_tensor, output_tensor, pcc=valid_pcc)

        logger.info(f"Yolov8x batch_size={self.batch_size}, PCC={self.pcc_message}")

    def dealloc_output(self):
        if hasattr(self, "output_tensor") and self.output_tensor is not None:
            ttnn.deallocate(self.output_tensor)
