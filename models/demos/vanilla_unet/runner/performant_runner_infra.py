# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0


import torch
from loguru import logger
from ttnn.model_preprocessing import preprocess_model_parameters

import ttnn
from models.common.utility_functions import divup, is_wormhole_b0
from models.demos.vanilla_unet.common import load_torch_model
from models.demos.vanilla_unet.tests.pcc.test_ttnn_unet import create_custom_preprocessor
from models.demos.vanilla_unet.ttnn.common import get_mesh_mappers
from models.demos.vanilla_unet.ttnn.ttnn_unet import TtUnet
from tests.ttnn.utils_for_testing import assert_with_pcc


class VanillaUNetPerformanceRunnerInfra:
    def __init__(
        self,
        device,
        batch_size,
        act_dtype,
        weight_dtype,
        model_location_generator=None,
        resolution=(480, 640),
        channels=3,
        torch_input_tensor=None,
    ):
        torch.manual_seed(0)
        self.resolution = resolution
        self.pcc_passed = False
        self.channels = channels
        self.pcc_message = "Did you forget to call validate()?"
        self.device = device
        self.num_devices = self.device.get_num_devices()
        self.batch_size = batch_size * self.device.get_num_devices()
        self.act_dtype = act_dtype
        self.weight_dtype = weight_dtype
        self.model_location_generator = model_location_generator
        self.torch_input_tensor = torch_input_tensor
        in_mapper, wt_mapper, out_composer = get_mesh_mappers(self.device)
        self.inputs_mesh_mapper = in_mapper
        self.weights_mesh_mapper = wt_mapper
        self.output_mesh_composer = out_composer

        self.torch_input_tensor = (
            torch.randn((self.batch_size, self.channels, resolution[0], resolution[1]))
            if self.torch_input_tensor is None
            else self.torch_input_tensor
        )

        self.torch_model = load_torch_model(model_location_generator)
        self.torch_output_tensor = self.torch_model(self.torch_input_tensor)

        self.parameters = preprocess_model_parameters(
            initialize_model=lambda: self.torch_model,
            custom_preprocessor=create_custom_preprocessor(device),
            device=None,
        )

        self.ttnn_model = TtUnet(device=device, parameters=self.parameters, model=self.torch_model)

    def run(self):
        self.output_tensor = self.ttnn_model(self.device, self.input_tensor)

    def _setup_l1_sharded_input(self, device, torch_input_tensor=None, min_channels=16):
        if is_wormhole_b0():
            core_grid = ttnn.CoreGrid(y=8, x=8)
        else:
            raise RuntimeError("Unsupported device: Only Wormhole B0 is currently supported.")

        torch_input_tensor = self.torch_input_tensor if torch_input_tensor is None else torch_input_tensor

        n, c, h, w = torch_input_tensor.shape

        if c < min_channels:
            c = min_channels
        elif c % min_channels != 0:
            c = ((c // min_channels) + 1) * min_channels
        n = n // self.num_devices if n // self.num_devices != 0 else n
        input_mem_config = ttnn.create_sharded_memory_config(
            [n, c, self.resolution[1], w],
            ttnn.CoreGrid(x=8, y=8),
            ttnn.ShardStrategy.HEIGHT,
        )
        input_tensor = [torch_input_tensor[i].unsqueeze(0) for i in range(torch_input_tensor.shape[0])]
        tt_inputs_host = ttnn.from_host_shards(
            [ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT) for t in input_tensor], device.shape
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
        output_tensor = ttnn.to_torch(output_tensor, mesh_composer=self.output_mesh_composer)
        output_tensor = output_tensor.permute(0, 3, 1, 2)
        output_tensor = output_tensor.reshape(torch_output_tensor.shape)
        self.pcc_passed, self.pcc_message = assert_with_pcc(self.torch_output_tensor, output_tensor, pcc=0.94)

        logger.info(
            f"Vanilla Unet batch_size={self.batch_size}, act_dtype={self.act_dtype}, weight_dtype={self.weight_dtype}, PCC={self.pcc_passed}"
        )

    def dealloc_output(self):
        ttnn.deallocate(self.output_tensor)
