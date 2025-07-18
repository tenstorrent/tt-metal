# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import torch
from loguru import logger

import ttnn
from models.demos.vgg_unet.reference.vgg_unet import UNetVGG19
from models.demos.vgg_unet.ttnn.model_preprocessing import create_vgg_unet_model_parameters, get_mesh_mappers
from models.demos.vgg_unet.ttnn.ttnn_vgg_unet import Tt_vgg_unet
from models.utility_functions import divup, is_wormhole_b0
from tests.ttnn.utils_for_testing import assert_with_pcc


class VGG_UnetTestInfra:
    def __init__(self, device, model_location_generator=None, use_pretrained_weight=False, batch_size=1):
        super().__init__()
        torch.manual_seed(0)
        input_mesh_mapper, weights_mesh_mapper, output_mesh_composer = get_mesh_mappers(device)
        self.pcc_passed = False
        self.batch_size = batch_size
        self.mesh_mapper = input_mesh_mapper
        self.mesh_composer = output_mesh_composer
        self.device = device
        self.num_devices = self.device.get_num_devices()
        self.batch_size_per_device = self.batch_size // self.num_devices
        self.pcc_message = "Did you forget to call validate()?"
        self.model_location_generator = model_location_generator
        self.torch_input = torch.randn((self.batch_size, 3, 256, 256), dtype=torch.bfloat16)
        self.torch_input_per_device = torch.randn((self.batch_size_per_device, 3, 256, 256), dtype=torch.float32)
        self.torch_input = self.torch_input.float()
        torch_model = UNetVGG19()
        if use_pretrained_weight:
            torch_model.load_state_dict(torch.load("models/demos/vgg_unet/vgg_unet_torch.pth"))
            torch_model.eval()  # Set to evaluation mode
        parameters = create_vgg_unet_model_parameters(torch_model, self.torch_input_per_device, device=device)
        self.torch_output = torch_model(self.torch_input)
        self.ttnn_vgg_unet_model = Tt_vgg_unet(device, parameters, parameters.conv_args)

    def run(self):
        self.output_tensor = self.ttnn_vgg_unet_model(self.input_tensor)

    def setup_l1_sharded_input(self, device, torch_input_tensor=None, min_channels=16):
        if is_wormhole_b0():
            core_grid = ttnn.CoreGrid(y=8, x=8)
        else:
            exit("Unsupported device")

        torch_input_tensor = self.torch_input if self.torch_input is None else self.torch_input

        n, c, h, w = torch_input_tensor.shape
        if c == 3:
            c = min_channels
        input_mem_config = ttnn.create_sharded_memory_config(
            [n // self.num_devices, c, h, w],
            ttnn.CoreGrid(x=8, y=8),
            ttnn.ShardStrategy.HEIGHT,
        )
        tt_inputs_host = ttnn.from_torch(
            torch_input_tensor, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, mesh_mapper=self.mesh_mapper
        )
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
        output_tensor = ttnn.to_torch(self.output_tensor, mesh_composer=self.mesh_composer)
        output_tensor = torch.permute(output_tensor, (0, 3, 1, 2))
        output_tensor = output_tensor.reshape((self.torch_output).shape)

        valid_pcc = 0.98
        self.pcc_passed, self.pcc_message = assert_with_pcc(self.torch_output, output_tensor, pcc=valid_pcc)

        logger.info(f"VGG_Unet, PCC={self.pcc_message}")

    def dealloc_output(self):
        ttnn.deallocate(self.output_tensor)


def create_test_infra(device, model_location_generator=None, use_pretrained_weight=False, batch_size=1):
    return VGG_UnetTestInfra(device, model_location_generator, use_pretrained_weight, batch_size)
