# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
from loguru import logger

import ttnn
from models.demos.yolov4.common import YOLOV4_BOXES_PCC, YOLOV4_CONFS_PCC, get_model_result, load_torch_model
from models.demos.yolov4.post_processing import gen_yolov4_boxes_confs, get_region_boxes
from models.demos.yolov4.tt.model_preprocessing import create_yolov4_model_parameters
from models.demos.yolov4.tt.yolov4 import TtYOLOv4
from models.utility_functions import divup, is_wormhole_b0
from tests.ttnn.utils_for_testing import assert_with_pcc


class YOLOv4PerformanceRunnerInfra:
    def __init__(
        self,
        device,
        batch_size,
        act_dtype,
        weight_dtype,
        model_location_generator=None,
        resolution=(320, 320),
        mesh_mapper=None,
        mesh_composer=None,
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
        self.num_devices = device.get_num_devices()
        self.inputs_mesh_mapper = mesh_mapper
        self.output_mesh_composer = mesh_composer

        self.torch_model = load_torch_model(self.model_location_generator)

        input_shape = (batch_size * self.num_devices, *resolution, 3)
        torch_input_shape = (batch_size, *resolution, 3)
        torch_input_tensor = torch.randn(input_shape, dtype=torch.float32)
        torch_input_tensor_params = torch.randn(torch_input_shape, dtype=torch.float32)
        self.input_tensor = ttnn.from_torch(torch_input_tensor, ttnn.bfloat16, mesh_mapper=self.inputs_mesh_mapper)
        self.torch_input_tensor = torch_input_tensor.permute(0, 3, 1, 2)
        self.torch_input_tensor_params = torch_input_tensor_params.permute(0, 3, 1, 2)

        parameters = create_yolov4_model_parameters(
            self.torch_model, self.torch_input_tensor_params, resolution, device
        )

        self.ttnn_yolov4_model = TtYOLOv4(parameters, device)

        self.torch_output_tensor = self.torch_model(self.torch_input_tensor)
        ref1, ref2, ref3 = gen_yolov4_boxes_confs(self.torch_output_tensor)
        self.ref_boxes, self.ref_confs = get_region_boxes([ref1, ref2, ref3])

    def run(self):
        self.output_tensor = self.ttnn_yolov4_model(self.input_tensor)

    def _setup_l1_sharded_input(self, device, torch_input_tensor=None):
        if is_wormhole_b0():
            core_grid = ttnn.CoreGrid(y=8, x=8)
        else:
            exit("Unsupported device")
        num_devices = device.get_num_devices()
        # torch tensor
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
        output_tensor = self.output_tensor if output_tensor is None else output_tensor
        result_boxes, result_confs = get_model_result(
            ttnn_output_tensor=output_tensor, resolution=self.resolution, mesh_composer=self.output_mesh_composer
        )

        self.pcc_passed, self.pcc_message = assert_with_pcc(self.ref_boxes, result_boxes, pcc=YOLOV4_BOXES_PCC)
        logger.info(
            f"Yolov4 - Bboxes. batch_size={self.batch_size}, act_dtype={self.act_dtype}, weight_dtype={self.weight_dtype}, PCC={self.pcc_message}"
        )

        self.pcc_passed, self.pcc_message = assert_with_pcc(self.ref_confs, result_confs, pcc=YOLOV4_CONFS_PCC)
        logger.info(
            f"Yolov4 - Confs. batch_size={self.batch_size}, act_dtype={self.act_dtype}, weight_dtype={self.weight_dtype}, PCC={self.pcc_message}"
        )

    def dealloc_output(self):
        ttnn.deallocate(self.output_tensor[0])
        ttnn.deallocate(self.output_tensor[1])
