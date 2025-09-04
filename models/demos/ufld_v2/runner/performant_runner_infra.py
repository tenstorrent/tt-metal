# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0


import torch
from loguru import logger
from ttnn.model_preprocessing import infer_ttnn_module_args, preprocess_model_parameters

import ttnn
from models.demos.ufld_v2.common import load_torch_model
from models.demos.ufld_v2.tests.pcc.test_ttnn_ufld_v2 import create_custom_mesh_preprocessor, get_mesh_mappers
from models.demos.ufld_v2.ttnn.ttnn_ufld_v2 import TtnnUFLDv2
from models.utility_functions import divup, is_wormhole_b0
from tests.ttnn.utils_for_testing import assert_with_pcc


def load_ttnn_model(device, torch_model, torch_input_tensor, weights_mesh_mapper):
    parameters = preprocess_model_parameters(
        initialize_model=lambda: torch_model,
        custom_preprocessor=create_custom_mesh_preprocessor(mesh_mapper=weights_mesh_mapper),
        device=device,
    )
    parameters.conv_args = {}
    parameters.conv_args = infer_ttnn_module_args(
        model=torch_model, run_model=lambda model: torch_model(torch_input_tensor), device=device
    )
    ttnn_model = TtnnUFLDv2(conv_args=parameters.conv_args, conv_pth=parameters, device=device)
    return ttnn_model


class UFLDPerformanceRunnerInfra:
    def __init__(
        self,
        device,
        model_location_generator,
        device_batch_size=1,
        act_dtype=ttnn.bfloat8_b,
        weight_dtype=ttnn.bfloat8_b,
        resolution=(320, 800),
        torch_input_tensor=None,
        channels=3,
    ):
        torch.manual_seed(0)
        self.device = device
        self.channels = channels
        self.num_devices = device.get_num_devices()
        self.inputs_mesh_mapper, self.weights_mesh_mapper, self.output_mesh_composer = get_mesh_mappers(self.device)
        self.resolution = resolution
        self.pcc_passed = False
        self.pcc_message = "Did you forget to call validate()?"
        self.device_batch_size = device_batch_size
        self.batch_size = self.device_batch_size * self.num_devices
        self.act_dtype = act_dtype
        self.weight_dtype = weight_dtype
        self.model_location_generator = model_location_generator
        self.torch_input_tensor = torch_input_tensor
        self.torch_input_tensor = (
            torch.randn((self.batch_size, self.channels, resolution[0], resolution[1]), dtype=torch.float32)
            if self.torch_input_tensor is None
            else self.torch_input_tensor
        )
        self.torch_input_tensor_per_device = torch.randn(
            (self.device_batch_size, self.channels, resolution[0], resolution[1]), dtype=torch.float32
        )
        self.torch_model = load_torch_model(model_location_generator, use_pretrained_weight=True)
        self.ttnn_ufld_v2_model = load_ttnn_model(
            self.device, self.torch_model, self.torch_input_tensor_per_device, self.weights_mesh_mapper
        )
        self.torch_output_tensor_1, self.torch_output_tensor_2 = self.torch_model(self.torch_input_tensor)

    def setup_l1_sharded_input(self, device, torch_input_tensor=None, min_channels=16):
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
            [n, c, h, w],
            ttnn.CoreGrid(x=8, y=8),
            ttnn.ShardStrategy.HEIGHT,
        )
        assert torch_input_tensor.ndim == 4, "Expected input tensor to have shape (BS, C, H, W)"
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

    def run(self):
        self.output_tensor_1 = self.ttnn_ufld_v2_model(input=self.input_tensor)

    def validate(self, output_tensor_1=None, torch_output_tensor_1=None):
        ttnn_output_tensor = self.output_tensor_1 if output_tensor_1 is None else output_tensor_1
        torch_output_tensor = self.torch_output_tensor_1 if torch_output_tensor_1 is None else torch_output_tensor_1
        output_tensor = ttnn.to_torch(ttnn_output_tensor, mesh_composer=self.output_mesh_composer).squeeze(1).squeeze(1)
        self.valid_pcc = 0.976
        self.pcc_passed, self.pcc_message = assert_with_pcc(torch_output_tensor, output_tensor, pcc=self.valid_pcc)
        logger.info(
            f"ufld_v2 - batch_size={self.batch_size}, act_dtype={self.act_dtype}, weight_dtype={self.weight_dtype}, PCC={self.pcc_message}"
        )

    def dealloc_output(self):
        ttnn.deallocate(self.output_tensor_1)


def create_test_infra(
    device,
    batch_size=1,
    act_dtype=ttnn.bfloat8_b,
    weight_dtype=ttnn.bfloat8_b,
    model_location_generator=None,
    resolution=(320, 800),
    torch_input_tensor=None,
):
    return UFLDPerformanceRunnerInfra(
        device,
        batch_size,
        act_dtype,
        weight_dtype,
        model_location_generator,
        resolution,
        torch_input_tensor,
    )
