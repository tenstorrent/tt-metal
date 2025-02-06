# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from loguru import logger
import os
import sys
import pytest
import torch
import torch.nn as nn
import torchvision
from tests.ttnn.utils_for_testing import assert_with_pcc
import ttnn
from models.experimental.functional_yolov8m.tt.ttnn_optimized_yolov8m import YOLOv8m
from models.experimental.functional_yolov8m.reference import yolov8m_utils

from models.utility_functions import (
    is_wormhole_b0,
    is_grayskull,
    divup,
)

from models.experimental.functional_yolov8m.tt.ttnn_yolov8m_utils import (
    ttnn_decode_bboxes,
    custom_preprocessor,
)

try:
    sys.modules["ultralytics"] = yolov8m_utils
    sys.modules["ultralytics.nn.tasks"] = yolov8m_utils
    sys.modules["ultralytics.nn.modules.conv"] = yolov8m_utils
    sys.modules["ultralytics.nn.modules.block"] = yolov8m_utils
    sys.modules["ultralytics.nn.modules.head"] = yolov8m_utils

except KeyError:
    print("models.experimental.functional_yolov8m.reference.yolov8m_utils not found.")


class Ensemble(nn.ModuleList):
    def __init__(self):
        super(Ensemble, self).__init__()

    def forward(self, x, augment=False):
        y = []
        for module in self:
            y.append(module(x, augment)[0])
        y = torch.cat(y, 1)
        return y, None


def attempt_load(weights, map_location=None):
    model = Ensemble()
    for w in weights if isinstance(weights, list) else [weights]:
        w = "models/experimental/functional_yolov8m/demo/yolov8m.pt"
        ckpt = torch.load(w, map_location=map_location)
        model.append(ckpt["ema" if ckpt.get("ema") else "model"].float().eval())
    for m in model.modules():
        if type(m) in [nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU]:
            m.inplace = True
        elif type(m) is nn.Upsample:
            m.recompute_scale_factor = None

    if len(model) == 1:
        return model[-1]
    else:
        for k in ["names", "stride"]:
            setattr(model, k, getattr(model[-1], k))
        return model


class Yolov8mTestInfra:
    def __init__(
        self,
        device,
        batch_size,
    ):
        super().__init__()
        torch.manual_seed(0)
        self.pcc_passed = False
        self.pcc_message = "Did you forget to call validate()?"
        self.device = device
        self.batch_size = batch_size
        # self.batch_size = batch_size
        # self.act_dtype = act_dtype
        # self.weight_dtype = weight_dtype
        # self.moclassdel_location_generator = model_location_generator
        # self.ttnn_yolov4_model = TtYOLOv4(device, load_yolov4_weight(self.model_location_generator))

        torch_model = attempt_load("yolov8m.pt", map_location="cpu")
        state_dict = torch_model.state_dict()

        self.parameters = custom_preprocessor(device, state_dict)
        input_shape = (1, 640, 640, 3)
        torch_input_tensor = torch.randn(input_shape, dtype=torch.float32)
        # self.input_tensor = ttnn.from_torch(torch_input_tensor, ttnn.bfloat16)
        # self.torch_input_tensor = torch_input_tensor.permute(0, 3, 1, 2)
        # self.torch_output_tensor = torch_model(self.torch_input_tensor)

        self.input_tensor = ttnn.from_torch(
            torch_input_tensor, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device
        )
        self.torch_input_tensor = torch_input_tensor.permute(0, 3, 1, 2)
        self.torch_output_tensor = torch_model(self.torch_input_tensor)

    def run(self, tt_input_tensor=None):
        self.output_tensor = None
        self.output_tensor = YOLOv8m(self.device, self.input_tensor, parameters=self.parameters)
        return self.output_tensor

    def setup_l1_sharded_input(self, device, torch_input_tensor=None):
        if is_wormhole_b0():
            core_grid = ttnn.CoreGrid(y=8, x=8)
        else:
            exit("Unsupported device")
        num_devices = 1 if isinstance(device, ttnn.Device) else device.get_num_devices()
        # torch tensor
        torch_input_tensor = self.torch_input_tensor if torch_input_tensor is None else torch_input_tensor

        n, c, h, w = torch_input_tensor.shape
        # sharded mem config for fold input
        num_cores = core_grid.x * core_grid.y
        shard_h = (n * w * h + num_cores - 1) // num_cores
        grid_size = core_grid
        grid_coord = ttnn.CoreCoord(grid_size.x - 1, grid_size.y - 1)
        shard_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), grid_coord)})
        shard_spec = ttnn.ShardSpec(shard_grid, (shard_h, 16), ttnn.ShardOrientation.ROW_MAJOR)
        input_mem_config = ttnn.MemoryConfig(
            ttnn.types.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.types.BufferType.L1, shard_spec
        )
        torch_input_tensor = torch_input_tensor.permute(0, 2, 3, 1)
        torch_input_tensor = torch_input_tensor.reshape(1, 1, h * w * n, c)
        tt_inputs_host = ttnn.from_torch(
            torch_input_tensor, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device
        )
        # tt_inputs_host = ttnn.pad(tt_inputs_host, [1, 1, n * h * w, 16], [0, 0, 0, 0], 0)
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
                16,
            ],
            ttnn.ShardOrientation.ROW_MAJOR,
        )
        sharded_mem_config_DRAM = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.DRAM, dram_shard_spec
        )

        return tt_inputs_host, sharded_mem_config_DRAM, input_mem_config

    def validate(self, output_tensor=None):
        output_tensor = self.output_tensor if output_tensor is None else output_tensor
        output_tensor = ttnn.to_torch(self.output_tensor[0])

        valid_pcc = 0.96
        self.pcc_passed, self.pcc_message = assert_with_pcc(self.torch_output_tensor[0], output_tensor, pcc=valid_pcc)

    def dealloc_output(self):
        ttnn.deallocate(self.output_tensor[0])


def create_test_infra(device, batch_size):
    return Yolov8mTestInfra(device, batch_size)
