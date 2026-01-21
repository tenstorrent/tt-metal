# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.

# SPDX-License-Identifier: Apache-2.0

import torch
from loguru import logger

import ttnn
from models.common.utility_functions import divup, is_wormhole_b0
from models.demos.yolov7.common import load_torch_model
from models.demos.yolov7.tt.ttnn_yolov7 import ttnn_yolov7
from models.demos.yolov7.ttnn_yolov7_utils import custom_preprocessor
from tests.ttnn.utils_for_testing import assert_with_pcc


class YOLOv7PerformanceRunnerInfra:
    def __init__(
        self,
        device,
        batch_size,
        act_dtype,
        weight_dtype,
        model_location_generator=None,
        resolution=(640, 640),
        torch_input_tensor=None,
        inputs_mesh_mapper=None,
        weights_mesh_mapper=None,
        outputs_mesh_composer=None,
    ):
        torch.manual_seed(0)
        self.resolution = resolution
        self.pcc_passed = False
        self.pcc_message = "Did you forget to call validate()?"
        self.device = device
        self.batch_size = batch_size
        self.act_dtype = act_dtype
        self.weight_dtype = weight_dtype
        self.model_location_generator = model_location_generator
        self.torch_input_tensor = torch_input_tensor

        self.num_devices = device.get_num_devices()
        self.inputs_mesh_mapper = inputs_mesh_mapper
        self.weights_mesh_mapper = weights_mesh_mapper
        self.outputs_mesh_composer = outputs_mesh_composer

        self.torch_model = load_torch_model(self.model_location_generator)
        self.torch_input_tensor = torch.randn((self.batch_size, 3, 640, 640), dtype=torch.float32)
        self.parameters = custom_preprocessor(model=self.torch_model, mesh_mapper=self.weights_mesh_mapper)
        nx_ny = [80, 40, 20]
        grid_tensors = []
        for i in range(3):
            yv, xv = torch.meshgrid([torch.arange(nx_ny[i]), torch.arange(nx_ny[i])])
            grid_tensors.append(torch.stack((xv, yv), 2).view((1, 1, nx_ny[i], nx_ny[i], 2)).float())

        self.ttnn_yolov7_model = ttnn_yolov7(self.device, self.parameters, grid_tensors)
        self.torch_output_tensor = self.torch_model(self.torch_input_tensor)[0]

    def _setup_l1_sharded_input(self, device, torch_input_tensor=None, min_channels=16):
        if is_wormhole_b0():
            core_grid = ttnn.CoreGrid(y=8, x=8)
        else:
            exit("Unsupported device")

        num_devices = device.get_num_devices()
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

        tt_inputs_host = ttnn.from_torch(torch_input_tensor, dtype=ttnn.bfloat16, mesh_mapper=self.inputs_mesh_mapper)
        return tt_inputs_host, input_mem_config

    def setup_dram_sharded_input(self, device, torch_input_tensor=None, mesh_mapper=None, mesh_composer=None):
        tt_inputs_host, input_mem_config = self._setup_l1_sharded_input(device)
        dram_grid_size = device.dram_grid_size()
        dram_shard_spec = ttnn.ShardSpec(
            ttnn.CoreRangeSet(
                {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(dram_grid_size.x - 1, dram_grid_size.y - 1))}
            ),
            [
                divup(tt_inputs_host.volume() // tt_inputs_host.shape[-1], dram_grid_size.x * dram_grid_size.y),
                tt_inputs_host.shape[-1],
            ],
            ttnn.ShardOrientation.ROW_MAJOR,
        )
        sharded_mem_config_DRAM = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.DRAM, dram_shard_spec
        )

        return tt_inputs_host, sharded_mem_config_DRAM, input_mem_config

    def run(self):
        self.output_tensor = self.ttnn_yolov7_model(self.input_tensor)[0]

    def validate(self, output_tensor=None, torch_output_tensor=None):
        if output_tensor is None:
            output_tensor = self.output_tensor
        else:
            if isinstance(output_tensor, (list, tuple)):
                output_tensor = output_tensor[0]

        torch_output_tensor = self.torch_output_tensor if torch_output_tensor is None else torch_output_tensor
        output_tensor = ttnn.to_torch(output_tensor, mesh_composer=self.outputs_mesh_composer)
        self.pcc_passed, self.pcc_message = assert_with_pcc(self.torch_output_tensor, output_tensor, pcc=0.99)

        logger.info(
            f"Yolov7 - batch_size={self.batch_size}, act_dtype={self.act_dtype}, weight_dtype={self.weight_dtype}, PCC={self.pcc_message}"
        )

    def dealloc_output(self):
        ttnn.deallocate(self.output_tensor)
