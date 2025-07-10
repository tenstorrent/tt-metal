# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0


import torch
import torch.nn.functional as F
from loguru import logger
import os
import sys

import ttnn
from models.utility_functions import divup, is_wormhole_b0
from tests.ttnn.utils_for_testing import assert_with_pcc

from models.experimental.yolov6l.reference.yolov6l_utils import fuse_model
from models.experimental.yolov6l.tt.model_preprocessing import create_yolov6l_model_parameters
from models.experimental.yolov6l.tt.ttnn_yolov6l import TtYolov6l

sys.path.append("models/experimental/yolov6l/reference/")


class YOLOv6lPerformanceRunnerInfra:
    def __init__(
        self,
        device,
        batch_size,
        act_dtype,
        weight_dtype,
        model_location_generator=None,
        resolution=(640, 640),
        torch_input_tensor=None,
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

        weights = "tests/ttnn/integration_tests/yolov6l/yolov6l.pt"
        if not os.path.exists(weights):
            os.system("bash models/experimental/yolov6l/weights_download.sh")

        ckpt = torch.load(weights, map_location=torch.device("cpu"), weights_only=False)
        model = ckpt["ema" if ckpt.get("ema") else "model"].float()
        model = fuse_model(model).eval()

        self.torch_model = model
        self.torch_input_tensor = (
            torch.randn((1, 3, 640, 640), dtype=torch.float32)
            if self.torch_input_tensor is None
            else self.torch_input_tensor
        )

        parameters = create_yolov6l_model_parameters(model, self.torch_input_tensor, device)

        self.parameters = parameters

        self.ttnn_yolov6l_model = TtYolov6l(device, parameters, parameters.model_args)

        self.torch_output_tensor = self.torch_model(self.torch_input_tensor)

    def _setup_l1_sharded_input(self, device, torch_input_tensor=None):
        if is_wormhole_b0():
            core_grid = ttnn.CoreGrid(y=8, x=8)
        else:
            exit("Unsupported device")

        num_devices = device.get_num_devices()
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

    def run(self):
        self.output_tensor = self.ttnn_yolov6l_model(self.input_tensor)

    def validate(self, output_tensor=None, torch_output_tensor=None):
        ttnn_output_tensor = self.output_tensor if output_tensor is None else output_tensor
        torch_output_tensor = self.torch_output_tensor if torch_output_tensor is None else torch_output_tensor
        output_tensor = ttnn.to_torch(ttnn_output_tensor)

        self.pcc_passed, self.pcc_message = assert_with_pcc(self.torch_output_tensor[0], output_tensor, pcc=0.99)

        logger.info(
            f"yolov6l - batch_size={self.batch_size}, act_dtype={self.act_dtype}, weight_dtype={self.weight_dtype}, PCC={self.pcc_message}"
        )

    def dealloc_output(self):
        ttnn.deallocate(self.output_tensor)
