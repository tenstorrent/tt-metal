# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from loguru import logger
import os
import pytest
import torch
import torchvision
from tests.ttnn.utils_for_testing import assert_with_pcc
import ttnn
from models.demos.yolov4.reference.yolov4 import Yolov4
from models.demos.yolov4.ttnn.yolov4 import TtYOLOv4
from models.demos.yolov4.demo.demo import YoloLayer, get_region_boxes, gen_yolov4_boxes_confs


from models.utility_functions import (
    is_wormhole_b0,
    is_grayskull,
    divup,
)


def load_yolov4_weight(model_location_generator=None):
    if model_location_generator == None:
        model_path = "models"
    else:
        model_path = model_location_generator("models", model_subdir="Yolo")
    if model_path == "models":
        if not os.path.exists("tests/ttnn/integration_tests/yolov4/yolov4.pth"):  # check if yolov4.th is availble
            os.system(
                "tests/ttnn/integration_tests/yolov4/yolov4_weights_download.sh"
            )  # execute the yolov4_weights_download.sh file

        weights_pth = "tests/ttnn/integration_tests/yolov4/yolov4.pth"
    else:
        weights_pth = str(model_path / "yolov4.pth")

    model = torch.load(weights_pth)
    return model


def load_yolov4_model(ttnn_model):
    torch_model = Yolov4()
    new_state_dict = dict(zip(torch_model.state_dict().keys(), ttnn_model.torch_model.values()))
    torch_model.load_state_dict(new_state_dict)
    torch_model.eval()
    return torch_model


class Yolov4TestInfra:
    def __init__(
        self,
        device,
        batch_size,
        act_dtype,
        weight_dtype,
        model_location_generator=None,
    ):
        super().__init__()
        torch.manual_seed(0)
        self.pcc_passed = False
        self.pcc_message = "Did you forget to call validate()?"
        self.device = device
        self.batch_size = batch_size
        self.act_dtype = act_dtype
        self.weight_dtype = weight_dtype
        self.model_location_generator = model_location_generator
        self.ttnn_yolov4_model = TtYOLOv4(load_yolov4_weight(self.model_location_generator), device)

        torch_model = load_yolov4_model(self.ttnn_yolov4_model)
        input_shape = (1, 320, 320, 3)
        torch_input_tensor = torch.randn(input_shape, dtype=torch.float32)
        self.input_tensor = ttnn.from_torch(torch_input_tensor, ttnn.bfloat16)
        self.torch_input_tensor = torch_input_tensor.permute(0, 3, 1, 2)
        self.torch_output_tensor = torch_model(self.torch_input_tensor)
        ref1, ref2, ref3 = gen_yolov4_boxes_confs(self.torch_output_tensor)
        self.ref_boxes, self.ref_confs = get_region_boxes([ref1, ref2, ref3])

    def run(self):
        self.output_tensor = self.ttnn_yolov4_model(self.input_tensor)

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
        tt_inputs_host = ttnn.from_torch(torch_input_tensor, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
        tt_inputs_host = ttnn.pad(tt_inputs_host, [1, 1, n * h * w, 16], [0, 0, 0, 0], 0)
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
        result_boxes_padded = ttnn.to_torch(self.output_tensor[0])
        result_confs = ttnn.to_torch(self.output_tensor[1])

        result_boxes_padded = result_boxes_padded.permute(0, 2, 1, 3)
        result_boxes_list = []
        # That ttnn tensor is the concat output of 3 padded tensors
        # As a perf workaround I'm doing the unpadding on the torch output here.
        # TODO: cleaner ttnn code when ttnn.untilize() is fully optimized
        box_1_start_i = 0
        box_1_end_i = 6100
        box_2_start_i = 6128
        box_2_end_i = 6228
        box_3_start_i = 6256
        box_3_end_i = 6356
        result_boxes_list.append(result_boxes_padded[:, box_1_start_i:box_1_end_i])
        result_boxes_list.append(result_boxes_padded[:, box_2_start_i:box_2_end_i])
        result_boxes_list.append(result_boxes_padded[:, box_3_start_i:box_3_end_i])
        result_boxes = torch.cat(result_boxes_list, dim=1)

        valid_pcc = 0.99
        self.pcc_passed, self.pcc_message = assert_with_pcc(self.ref_boxes, result_boxes, pcc=valid_pcc)

        logger.info(
            f"Yolov4 - Bboxes. batch_size={self.batch_size}, act_dtype={self.act_dtype}, weight_dtype={self.weight_dtype}, PCC={self.pcc_message}"
        )

        valid_pcc = 0.71
        self.pcc_passed, self.pcc_message = assert_with_pcc(self.ref_confs, result_confs, pcc=valid_pcc)

        logger.info(
            f"Yolov4 - Confs. batch_size={self.batch_size}, act_dtype={self.act_dtype}, weight_dtype={self.weight_dtype}, PCC={self.pcc_message}"
        )

    def dealloc_output(self):
        ttnn.deallocate(self.output_tensor[0])
        ttnn.deallocate(self.output_tensor[1])


def create_test_infra(
    device,
    batch_size,
    act_dtype,
    weight_dtype,
    model_location_generator=None,
):
    return Yolov4TestInfra(
        device,
        batch_size,
        act_dtype,
        weight_dtype,
        model_location_generator,
    )
