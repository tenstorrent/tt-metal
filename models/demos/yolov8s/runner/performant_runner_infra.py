# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import torch
from loguru import logger
from ultralytics import YOLO

import ttnn
from models.demos.yolov8s.tt.tt_yolov8s_utils import custom_preprocessor
from models.demos.yolov8s.tt.ttnn_yolov8s import TtYolov8sModel
from models.utility_functions import divup, is_wormhole_b0
from tests.ttnn.utils_for_testing import assert_with_pcc


def load_torch_model():
    torch_model = YOLO("yolov8s.pt")
    torch_model = torch_model.model
    torch_model.eval()
    return torch_model


def load_ttnn_model(device, torch_model, inp_h, inp_w):
    state_dict = torch_model.state_dict()
    parameters = custom_preprocessor(device, state_dict, inp_h=inp_h, inp_w=inp_w)
    ttnn_model = TtYolov8sModel(device=device, parameters=parameters, res=(inp_h, inp_w))
    return ttnn_model


class YOLOv8sPerformanceRunnerInfra:
    def __init__(
        self,
        device,
        batch_size,
        model_location_generator=None,
    ):
        super().__init__()
        torch.manual_seed(0)
        self.pcc_passed = False
        self.pcc_message = "Did you forget to call validate()?"
        self.device = device
        self.batch_size = batch_size
        self.model_location_generator = model_location_generator
        torch_model = load_torch_model()
        input_shape = (1, 640, 640, 3)
        inp_h, inp_w = input_shape[1], input_shape[2]
        self.ttnn_yolov8_model = load_ttnn_model(device=self.device, torch_model=torch_model, inp_h=inp_h, inp_w=inp_w)
        torch_input_tensor = torch.randn(input_shape, dtype=torch.float32)
        self.tt_input_tensor = ttnn.from_torch(torch_input_tensor, ttnn.bfloat16)
        self.torch_input_tensor = torch_input_tensor.permute(0, 3, 1, 2)
        self.torch_output_tensor = torch_model(self.torch_input_tensor)[0]

    def run(self):
        self.output_tensor = self.ttnn_yolov8_model(self.input_tensor)[0]

    def setup_l1_sharded_input(self, device, torch_input_tensor=None):
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
            [n, c, h, w],
            ttnn.CoreGrid(x=8, y=8),
            ttnn.ShardStrategy.HEIGHT,
        )

        tt_inputs_host = ttnn.from_torch(torch_input_tensor, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)

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
        output_tensor = self.output_tensor if output_tensor is None else output_tensor
        output_tensor = ttnn.to_torch(self.output_tensor)

        valid_pcc = 0.978
        self.pcc_passed, self.pcc_message = assert_with_pcc(self.torch_output_tensor, output_tensor, pcc=valid_pcc)

        logger.info(f"Yolov8s batch_size={self.batch_size}, PCC={self.pcc_message}")

    def dealloc_output(self):
        ttnn.deallocate(self.output_tensor)
