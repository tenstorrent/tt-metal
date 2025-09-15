# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import torch
from loguru import logger

import ttnn
from models.common.utility_functions import divup, is_wormhole_b0
from models.demos.vgg_unet.common import load_torch_model
from models.demos.vgg_unet.reference.vgg_unet import UNetVGG19
from models.demos.vgg_unet.ttnn.model_preprocessing import create_vgg_unet_model_parameters, get_mesh_mappers
from models.demos.vgg_unet.ttnn.ttnn_vgg_unet import Tt_vgg_unet
from tests.ttnn.utils_for_testing import assert_with_pcc


class VGG_UnetTestInfra:
    def __init__(
        self,
        device,
        model_location_generator=None,
        use_pretrained_weight=False,
        device_batch_size=1,
        channels=3,
        resolution=256,
    ):
        super().__init__()
        torch.manual_seed(0)
        input_mesh_mapper, weights_mesh_mapper, output_mesh_composer = get_mesh_mappers(device)
        self.pcc_passed = False
        self.device_batch_size = device_batch_size
        self.mesh_mapper = input_mesh_mapper
        self.mesh_composer = output_mesh_composer
        self.device = device
        self.channels = channels
        self.num_devices = self.device.get_num_devices()
        self.batch_size = self.device_batch_size * self.num_devices
        self.pcc_message = "Did you forget to call validate()?"
        self.model_location_generator = model_location_generator
        self.torch_input = torch.randn((self.batch_size, channels, resolution, resolution))
        self.torch_input_per_device = torch.randn((self.device_batch_size, channels, resolution, resolution))
        torch_model = UNetVGG19()
        if use_pretrained_weight:
            torch_model = load_torch_model(torch_model, model_location_generator)
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
            raise RuntimeError("Unsupported device: Only Wormhole B0 is currently supported.")

        torch_input_tensor = self.torch_input if torch_input_tensor is None else torch_input_tensor

        n, c, h, w = torch_input_tensor.shape
        if c < min_channels:
            c = min_channels
        elif c % min_channels != 0:
            c = ((c // min_channels) + 1) * min_channels
        n = n // self.num_devices if n // self.num_devices != 0 else n
        input_mem_config = ttnn.create_sharded_memory_config(
            [n, c, h, w],
            ttnn.CoreGrid(x=8, y=8),
            ttnn.ShardStrategy.HEIGHT,
        )
        input_tensor = [torch_input_tensor[i].unsqueeze(0) for i in range(torch_input_tensor.shape[0])]
        tt_inputs_host = ttnn.from_host_shards(
            [ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT) for t in input_tensor], device.shape
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


def create_test_infra(device, model_location_generator=None, use_pretrained_weight=False, device_batch_size=1):
    return VGG_UnetTestInfra(device, model_location_generator, use_pretrained_weight, device_batch_size)
