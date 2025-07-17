# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import os

import torch
from loguru import logger
from ttnn.model_preprocessing import preprocess_model_parameters

import ttnn
from models.demos.vanilla_unet.reference.unet import UNet
from models.demos.vanilla_unet.ttnn.ttnn_unet import TtUnet
from models.utility_functions import divup, is_wormhole_b0
from tests.ttnn.integration_tests.vanilla_unet.test_ttnn_unet import create_custom_preprocessor
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

        weights_path = "models/demos/vanilla_unet/unet.pt"
        if not os.path.exists(weights_path):
            os.system("bash models/demos/vanilla_unet/weights_download.sh")

        state_dict = torch.load(
            weights_path,
            map_location=torch.device("cpu"),
        )
        ds_state_dict = {k: v for k, v in state_dict.items()}

        self.torch_model = UNet()

        self.torch_input_tensor = (
            torch.randn((1, 3, 480, 640)) if self.torch_input_tensor is None else self.torch_input_tensor
        )

        new_state_dict = {}
        keys = [name for name, parameter in self.torch_model.state_dict().items()]
        values = [parameter for name, parameter in ds_state_dict.items()]
        for i in range(len(keys)):
            new_state_dict[keys[i]] = values[i]

        self.torch_model.load_state_dict(new_state_dict)
        self.torch_model.eval()

        self.torch_output_tensor = self.torch_model(self.torch_input_tensor)

        self.parameters = preprocess_model_parameters(
            initialize_model=lambda: self.torch_model,
            custom_preprocessor=create_custom_preprocessor(device),
            device=None,
        )

        self.ttnn_model = TtUnet(device=device, parameters=self.parameters, model=self.torch_model)

    def run(self):
        self.output_tensor = self.ttnn_model(self.device, self.input_tensor)

    def _setup_l1_sharded_input(self, device, torch_input_tensor=None):
        if is_wormhole_b0():
            core_grid = ttnn.CoreGrid(y=8, x=8)
        else:
            exit("Unsupported device")

        num_devices = 1 if isinstance(device, ttnn.Device) else device.get_num_devices()
        # torch tensor
        torch_input_tensor = self.torch_input_tensor if torch_input_tensor is None else torch_input_tensor

        n, c, h, w = torch_input_tensor.shape

        ## Converting from image based channels (3) to min channels (16)
        if c == 3:
            c = 16
        input_mem_config = ttnn.create_sharded_memory_config(
            [n, c, 640, w],
            ttnn.CoreGrid(x=8, y=8),
            ttnn.ShardStrategy.HEIGHT,
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
        output_tensor = output_tensor.permute(0, 3, 1, 2)
        output_tensor = output_tensor.reshape(torch_output_tensor.shape)
        self.pcc_passed, self.pcc_message = assert_with_pcc(self.torch_output_tensor, output_tensor, pcc=0.94)

        logger.info(
            f"Vanilla Unet batch_size={self.batch_size}, act_dtype={self.act_dtype}, weight_dtype={self.weight_dtype}, PCC={self.pcc_passed}"
        )

    def dealloc_output(self):
        ttnn.deallocate(self.output_tensor)
