# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0


import os

import torch
from loguru import logger

import ttnn
from models.demos.mobilenetv2.reference.mobilenetv2 import Mobilenetv2
from models.demos.mobilenetv2.tt.model_preprocessing import create_mobilenetv2_model_parameters, get_mesh_mappers
from models.demos.mobilenetv2.tt.ttnn_mobilenetv2 import TtMobileNetV2
from models.utility_functions import divup, is_wormhole_b0
from tests.ttnn.utils_for_testing import assert_with_pcc


def load_torch_model():
    weights_path = "models/demos/mobilenetv2/mobilenet_v2-b0353104.pth"
    if not os.path.exists(weights_path):
        os.system("bash models/demos/mobilenetv2/weights_download.sh")

    state_dict = torch.load(weights_path)
    ds_state_dict = {k: v for k, v in state_dict.items()}

    torch_model = Mobilenetv2()
    new_state_dict = {
        name1: parameter2
        for (name1, parameter1), (name2, parameter2) in zip(torch_model.state_dict().items(), ds_state_dict.items())
        if isinstance(parameter2, torch.FloatTensor)
    }
    torch_model.load_state_dict(new_state_dict)
    torch_model.eval()
    return torch_model


def load_ttnn_model(device, torch_model, batch_size):
    model_parameters = create_mobilenetv2_model_parameters(torch_model, device=device)
    ttnn_model = TtMobileNetV2(model_parameters, device, batchsize=batch_size)
    return ttnn_model


class MobileNetv2TestInfra:
    def __init__(
        self,
        device,
        batch_size,
        model_location_generator=None,
        mesh_mapper=None,
        mesh_composer=None,
    ):
        super().__init__()
        torch.manual_seed(0)
        self.pcc_passed = False
        self.pcc_message = "Did you forget to call validate()?"
        self.device = device
        self.batch_size = batch_size
        self.model_location_generator = model_location_generator
        torch_model = load_torch_model()
        self.num_devices = device.get_num_devices()
        self.mesh_mapper = mesh_mapper
        self.mesh_composer = mesh_composer
        self.batch_size_per_device = self.batch_size
        self.batch_size = self.batch_size * self.num_devices
        self.ttnn_mobilenetv2_model = load_ttnn_model(
            device=self.device, torch_model=torch_model, batch_size=self.batch_size_per_device
        )
        input_shape = (self.batch_size, 224, 224, 3)
        self.inputs_mesh_mapper, _, self.output_mesh_composer = get_mesh_mappers(self.device)
        torch_input_tensor = torch.randn(input_shape, dtype=torch.float32)
        self.input_tensor = ttnn.from_torch(torch_input_tensor, ttnn.bfloat16, mesh_mapper=self.inputs_mesh_mapper)
        self.torch_input_tensor = torch_input_tensor.permute(0, 3, 1, 2)
        self.torch_output_tensor = torch_model(self.torch_input_tensor)

    def run(self):
        self.output_tensor = self.ttnn_mobilenetv2_model(self.input_tensor)

    def setup_l1_sharded_input(self, device, torch_input_tensor=None, channels_length=16):
        if is_wormhole_b0():
            core_grid = ttnn.CoreGrid(y=8, x=8)
        else:
            exit("Unsupported device")
        torch_input_tensor = self.torch_input_tensor if torch_input_tensor is None else torch_input_tensor

        n, c, h, w = torch_input_tensor.shape
        # sharded mem config for fold input
        num_cores = core_grid.x * core_grid.y
        shard_h = ((n // self.num_devices) * w * h + num_cores - 1) // num_cores
        grid_size = core_grid
        grid_coord = ttnn.CoreCoord(grid_size.x - 1, grid_size.y - 1)
        shard_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), grid_coord)})
        shard_spec = ttnn.ShardSpec(shard_grid, (shard_h, channels_length), ttnn.ShardOrientation.ROW_MAJOR)
        input_mem_config = ttnn.MemoryConfig(
            ttnn.types.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.types.BufferType.L1, shard_spec
        )
        torch_input_tensor = torch_input_tensor.permute(0, 2, 3, 1)
        tt_inputs_host = ttnn.from_torch(
            torch_input_tensor, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, mesh_mapper=self.inputs_mesh_mapper
        )
        tt_inputs_host = ttnn.reshape(tt_inputs_host, (1, 1, ((n // self.num_devices) * h * w), c))
        tt_inputs_host = ttnn.pad(
            tt_inputs_host, [1, 1, (n // self.num_devices) * h * w, channels_length], [0, 0, 0, 0], 0
        )
        return tt_inputs_host, input_mem_config

    def setup_dram_sharded_input(
        self, device, torch_input_tensor=None, mesh_mapper=None, mesh_composer=None, channels_length=16
    ):
        tt_inputs_host, input_mem_config = self.setup_l1_sharded_input(device)
        dram_grid_size = device.dram_grid_size()
        dram_shard_spec = ttnn.ShardSpec(
            ttnn.CoreRangeSet(
                {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(dram_grid_size.x - 1, dram_grid_size.y - 1))}
            ),
            [
                divup(tt_inputs_host.volume() // tt_inputs_host.shape[-1], (dram_grid_size.x * dram_grid_size.y)),
                channels_length,
            ],
            ttnn.ShardOrientation.ROW_MAJOR,
        )
        sharded_mem_config_DRAM = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.DRAM, dram_shard_spec
        )

        return tt_inputs_host, sharded_mem_config_DRAM, input_mem_config

    def validate(self, output_tensor=None):
        output_tensor = self.output_tensor if output_tensor is None else output_tensor
        output_tensor = ttnn.to_torch(self.output_tensor, mesh_composer=self.output_mesh_composer)
        valid_pcc = 0.94
        self.pcc_passed, self.pcc_message = assert_with_pcc(self.torch_output_tensor, output_tensor, pcc=valid_pcc)

        logger.info(f"mobilenetv2 batch_size={self.batch_size}, PCC={self.pcc_message}")

    def dealloc_output(self):
        ttnn.deallocate(self.output_tensor)


def create_test_infra(
    device,
    batch_size,
):
    return MobileNetv2TestInfra(
        device,
        batch_size,
    )
