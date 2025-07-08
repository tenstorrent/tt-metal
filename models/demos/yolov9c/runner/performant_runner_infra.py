# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0


import torch
from loguru import logger

import ttnn
from models.demos.yolov9c.demo.demo_utils import load_torch_model
from models.demos.yolov9c.tt.model_preprocessing import create_yolov9c_model_parameters
from models.demos.yolov9c.tt.ttnn_yolov9c import YoloV9
from models.utility_functions import divup, is_wormhole_b0
from tests.ttnn.utils_for_testing import assert_with_pcc


class YOLOv9PerformanceRunnerInfra:
    def __init__(
        self,
        device,
        batch_size,
        act_dtype,
        weight_dtype,
        model_task="segment",
        resolution=(640, 640),
        torch_input_tensor=None,
        mesh_mapper=None,
        weights_mesh_mapper=None,
        mesh_composer=None,
    ):
        torch.manual_seed(0)
        self.resolution = resolution
        self.pcc_passed = False
        self.pcc_message = "Did you forget to call validate()?"

        self.device = device
        self.num_devices = self.device.get_num_devices()
        self.mesh_mapper = mesh_mapper
        self.weights_mesh_mapper = weights_mesh_mapper
        self.mesh_composer = mesh_composer

        self.batch_size = batch_size
        self.act_dtype = act_dtype
        self.weight_dtype = weight_dtype
        self.model_task = model_task
        self.enable_segment = self.model_task == "segment"
        self.torch_input_tensor = torch_input_tensor
        self.torch_model = load_torch_model(model_task=self.model_task)

        self.torch_input_tensor = (
            torch.randn((batch_size * self.num_devices, 3, 640, 640), dtype=torch.float32)
            if self.torch_input_tensor is None
            else self.torch_input_tensor
        )
        self.torch_input_params = torch.randn((batch_size, 3, 640, 640), dtype=torch.float32)
        self.parameters = create_yolov9c_model_parameters(self.torch_model, self.torch_input_params, device=self.device)

        self.ttnn_yolov9c_model = YoloV9(
            self.device, self.parameters, enable_segment=self.enable_segment
        )  # Set enable_segment to False for detection

        self.torch_output_tensor = self.torch_model(self.torch_input_tensor)

    def _setup_l1_sharded_input(self, device, torch_input_tensor=None):
        if is_wormhole_b0():
            core_grid = ttnn.CoreGrid(y=8, x=8)
        else:
            exit("Unsupported device")

        torch_input_tensor = self.torch_input_tensor if torch_input_tensor is None else torch_input_tensor
        n, c, h, w = torch_input_tensor.shape
        ## Converting from image based channels (3) to min channels (16)
        if c == 3:
            c = 16
        n = n // self.num_devices if n // self.num_devices != 0 else n
        input_mem_config = ttnn.create_sharded_memory_config(
            [n, c, h, w],
            ttnn.CoreGrid(x=8, y=8),
            ttnn.ShardStrategy.HEIGHT,
        )
        tt_inputs_host = ttnn.from_torch(
            torch_input_tensor, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, mesh_mapper=self.mesh_mapper
        )

        return tt_inputs_host, input_mem_config

    def setup_dram_sharded_input(self, device, torch_input_tensor=None, mesh_mapper=None, mesh_composer=None):
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

    def run(self):
        self.output_tensor = self.ttnn_yolov9c_model(self.input_tensor)

    def validate(self, output_tensor=None, torch_output_tensor=None):
        ttnn_output_tensor = self.output_tensor if output_tensor is None else output_tensor
        torch_output_tensor = self.torch_output_tensor if torch_output_tensor is None else torch_output_tensor
        output_tensor = ttnn.to_torch(ttnn_output_tensor[0], mesh_composer=self.mesh_composer)
        self.pcc_passed, self.pcc_message = assert_with_pcc(self.torch_output_tensor[0], output_tensor, pcc=0.99)

        logger.info(
            f"Yolov9c - batch_size={self.batch_size}, act_dtype={self.act_dtype}, weight_dtype={self.weight_dtype}, PCC={self.pcc_message}"
        )

    def dealloc_output(self):
        ttnn.deallocate(self.output_tensor[0])
        for t in self.output_tensor[1]:
            if isinstance(t, list):
                for sub_t in t:
                    ttnn.deallocate(sub_t)
            else:
                ttnn.deallocate(t)
