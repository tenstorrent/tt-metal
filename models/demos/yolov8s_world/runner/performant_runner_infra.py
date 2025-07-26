# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0


import torch
from loguru import logger
from ttnn.model_preprocessing import preprocess_model_parameters

import ttnn
from models.demos.yolov8s_world.demo.demo_utils import load_torch_model
from models.demos.yolov8s_world.tt.ttnn_yolov8s_world import TtYOLOWorld
from models.demos.yolov8s_world.tt.ttnn_yolov8s_world_utils import create_custom_preprocessor, move_to_device
from models.utility_functions import divup, is_wormhole_b0
from tests.ttnn.utils_for_testing import assert_with_pcc


class YOLOv8sWorldPerformanceRunnerInfra:
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

        self.num_devices = self.device.get_num_devices()
        self.inputs_mesh_mapper = inputs_mesh_mapper
        self.weights_mesh_mapper = weights_mesh_mapper
        self.outputs_mesh_composer = outputs_mesh_composer

        self.torch_model = load_torch_model()
        self.torch_input_tensor = (
            torch.randn((batch_size, 3, 640, 640), dtype=torch.float32)
            if self.torch_input_tensor is None
            else self.torch_input_tensor
        )

        parameters = preprocess_model_parameters(
            initialize_model=lambda: self.torch_model, custom_preprocessor=create_custom_preprocessor(device)
        )

        for i in [12, 15, 19, 22]:
            parameters["model"][i]["attn"]["gl"]["weight"] = ttnn.to_device(
                parameters["model"][i]["attn"]["gl"]["weight"], device=device
            )
            parameters["model"][i]["attn"]["gl"]["bias"] = ttnn.to_device(
                parameters["model"][i]["attn"]["gl"]["bias"], device=device
            )
            parameters["model"][i]["attn"]["bias"] = ttnn.to_device(
                parameters["model"][i]["attn"]["bias"], device=device
            )

            parameters["model"][16] = move_to_device(parameters["model"][16], device)

            parameters["model"][23]["cv4"] = move_to_device(parameters["model"][23]["cv4"], device)

        self.parameters = parameters

        self.ttnn_yolov8s_world_model = TtYOLOWorld(self.device, self.parameters)

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
        input_mem_config = ttnn.create_sharded_memory_config(
            [n // self.num_devices, c, h, w],
            ttnn.CoreGrid(x=8, y=8),
            ttnn.ShardStrategy.HEIGHT,
        )
        tt_inputs_host = ttnn.from_torch(
            torch_input_tensor, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, mesh_mapper=self.inputs_mesh_mapper
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
        self.output_tensor = self.ttnn_yolov8s_world_model(self.input_tensor)

    def validate(self, output_tensor=None, torch_output_tensor=None):
        ttnn_output_tensor = self.output_tensor if output_tensor is None else output_tensor
        torch_output_tensor = self.torch_output_tensor if torch_output_tensor is None else torch_output_tensor
        output_tensor = ttnn.to_torch(ttnn_output_tensor[0], mesh_composer=self.outputs_mesh_composer)

        self.pcc_passed, self.pcc_message = assert_with_pcc(self.torch_output_tensor[0], output_tensor, pcc=0.99)

        logger.info(
            f"yolov8s_world - batch_size={self.batch_size}, act_dtype={self.act_dtype}, weight_dtype={self.weight_dtype}, PCC={self.pcc_message}"
        )

    def dealloc_output(self):
        ttnn.deallocate(self.output_tensor[0])
        for t in self.output_tensor[1]:
            if isinstance(t, list):
                for sub_t in t:
                    ttnn.deallocate(sub_t)
            else:
                ttnn.deallocate(t)
