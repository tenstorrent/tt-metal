# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import os
import ttnn
import torch
import requests
from loguru import logger
import torch.nn.functional as F


from models.experimental.yolov10.tests.common import (
    load_torch_model,
)
from models.experimental.yolov10.tt.model_preprocessing import (
    create_yolov10x_model_parameters,
)
from models.experimental.yolov10.tt.yolov10 import TtnnYolov10
from models.utility_functions import divup, is_wormhole_b0
from tests.ttnn.utils_for_testing import assert_with_pcc


def load_coco_class_names():
    url = "https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names"
    path = f"models/demos/yolov4/demo/coco.names"
    response = requests.get(url)
    try:
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            return response.text.strip().split("\n")
    except requests.RequestException:
        pass
    if os.path.exists(path):
        with open(path, "r") as f:
            return [line.strip() for line in f.readlines()]

    raise Exception("Failed to fetch COCO class names from both online and local sources.")


class YOLOv10PerformanceRunnerInfra:
    def __init__(
        self,
        device,
        batch_size,
        act_dtype,
        weight_dtype,
        model_location_generator=None,
        resolution=(640, 640),
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

        self.torch_model = load_torch_model()

        self.torch_input_tensor = torch.randn((1, 3, 640, 640), dtype=torch.float32)
        model_type = f"torch_model"

        self.torch_output_tensor = self.torch_model(self.torch_input_tensor)

        self.parameters = create_yolov10x_model_parameters(
            self.torch_model, self.torch_input_tensor, device=self.device
        )

        self.ttnn_yolov10x_model = TtnnYolov10(self.device, self.parameters, self.parameters)

    def run(self):
        self.output_tensor = self.ttnn_yolov10x_model(self.input_tensor)

    def _setup_l1_sharded_input(self, device, torch_input_tensor=None):
        if is_wormhole_b0():
            core_grid = ttnn.CoreGrid(y=8, x=8)
        else:
            exit("Unsupported device")

        num_devices = 1 if isinstance(device, ttnn.Device) else device.get_num_devices()
        # torch tensor
        torch_input_tensor = self.torch_input_tensor if torch_input_tensor is None else torch_input_tensor

        n, c, h, w = torch_input_tensor.shape

        torch_input_tensor = torch_input_tensor.permute(0, 2, 3, 1)
        torch_input_tensor = F.pad(torch_input_tensor, (0, 29))
        input_mem_config = ttnn.create_sharded_memory_config(
            [6400, 32],
            core_grid=device.core_grid,
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
        tt_inputs_host = ttnn.from_torch(torch_input_tensor, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)

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

    def validate(self, output_tensor=None, torch_output_tensor=None):
        output_tensor = self.output_tensor if output_tensor is None else output_tensor
        torch_output_tensor = self.torch_output_tensor if torch_output_tensor is None else torch_output_tensor
        output_tensor = ttnn.to_torch(output_tensor)
        self.pcc_passed, self.pcc_message = assert_with_pcc(self.torch_output_tensor, output_tensor, pcc=0.99)

        logger.info(
            f"Yolov10x batch_size={self.batch_size}, act_dtype={self.act_dtype}, weight_dtype={self.weight_dtype}, PCC={self.pcc_passed}"
        )

    def dealloc_output(self):
        ttnn.deallocate(self.output_tensor)
