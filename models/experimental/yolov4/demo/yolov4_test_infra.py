# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from loguru import logger
import os
import torch
import torchvision

import ttnn
from ttnn.model_preprocessing import (
    preprocess_model_parameters,
)
from models.utility_functions import (
    is_wormhole_b0,
    is_grayskull,
    divup,
    _nearest_y,
)

from tests.ttnn.utils_for_testing import assert_with_pcc
from models.demos.ttnn_resnet.tt.custom_preprocessing import create_custom_mesh_preprocessor

from models.experimental.yolov4.reference.yolov4 import Yolov4
from models.experimental.yolov4.ttnn.yolov4 import TtYOLOv4
from models.experimental.yolov4.ttnn.weight_parameter_update import update_weight_parameters
from collections import OrderedDict
import cv2


class yolov4TestInfra:
    def __init__(
        self,
        device,
        batch_size,
        inputs_mesh_mapper=None,
        weights_mesh_mapper=None,
        output_mesh_composer=None,
    ):
        super().__init__()
        torch.manual_seed(0)
        self.pcc_passed = False
        self.pcc_message = "Did you forget to call validate()?"
        self.device = device
        self.batch_size = batch_size
        self.inputs_mesh_mapper = inputs_mesh_mapper
        self.weights_mesh_mapper = weights_mesh_mapper
        self.output_mesh_composer = output_mesh_composer

        if not os.path.exists("tests/ttnn/integration_tests/yolov4/yolov4.pth"):  # check if yolov4.th is availble
            os.system("tests/ttnn/integration_tests/yolov4/yolov4_weights_download.sh")
        weights_pth = "tests/ttnn/integration_tests/yolov4/yolov4.pth"

        self.ttnn_model = TtYOLOv4(weights_pth)

        n_classes = 80
        namesfile = "models/experimental/yolov4/demo/coco.names"
        imgfile = "models/experimental/yolov4/demo/giraffe_320.jpg"

        width = 320
        height = 320

        img = cv2.imread(imgfile)
        sized = cv2.resize(img, (width, height))
        torch_pixel_values = cv2.cvtColor(sized, cv2.COLOR_BGR2RGB)
        torch_pixel_values = torch.from_numpy(torch_pixel_values.transpose(2, 0, 1)).float().div(255.0).unsqueeze(0)
        torch_pixel_values = torch.autograd.Variable(torch_pixel_values)
        torch_pixel_values = torch.permute(torch_pixel_values, (0, 2, 3, 1))
        torch_pixel_values = torch.nn.functional.pad(torch_pixel_values, (0, 13, 0, 0, 0, 0, 0, 0))
        N, H, W, C = torch_pixel_values.shape
        self.torch_pixel_values = torch.reshape(torch_pixel_values, (N, 1, H * W, C))
        self.input_tensor = None

    def setup_l1_sharded_input(self, device, torch_pixel_values=None, mesh_mapper=None, mesh_composer=None):
        N, _, HW, C = self.torch_pixel_values.shape

        shard_grid = ttnn.CoreRangeSet(
            {
                ttnn.CoreRange(
                    ttnn.CoreCoord(0, 0),
                    ttnn.CoreCoord(7, 7),
                ),
            }
        )
        n_cores = 64
        shard_spec = ttnn.ShardSpec(shard_grid, [N * HW // n_cores, C], ttnn.ShardOrientation.ROW_MAJOR, False)
        input_mem_config = ttnn.MemoryConfig(
            ttnn.types.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.types.BufferType.L1, shard_spec
        )

        tt_inputs_host = ttnn.from_torch(
            self.torch_pixel_values,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )

        return tt_inputs_host, input_mem_config

    def setup_dram_sharded_input(self, device, torch_input_tensor=None, mesh_mapper=None, mesh_composer=None):
        tt_inputs_host, input_mem_config = self.setup_l1_sharded_input(
            device, self.torch_pixel_values, mesh_mapper=mesh_mapper, mesh_composer=mesh_composer
        )
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
            False,
        )
        sharded_mem_config_DRAM = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.DRAM, dram_shard_spec
        )

        return tt_inputs_host, sharded_mem_config_DRAM, input_mem_config

    def run(self, tt_input_tensor=None):
        self.output_tensor = None
        self.output_tensor = self.ttnn_model(self.device, self.input_tensor)
        return self.output_tensor


def create_test_infra(
    device,
    batch_size,
    inputs_mesh_mapper=None,
    weights_mesh_mapper=None,
    output_mesh_composer=None,
):
    return yolov4TestInfra(
        device,
        batch_size,
        inputs_mesh_mapper,
        weights_mesh_mapper,
        output_mesh_composer,
    )
