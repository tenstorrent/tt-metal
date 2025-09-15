# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import torch.nn.functional as F
import torch
import ttnn
from loguru import logger
from models.experimental.vovnet.tt.model_preprocessing import custom_preprocessor
from models.experimental.vovnet.tt.vovnet import TtVoVNet
from models.common.utility_functions import divup, is_wormhole_b0
from tests.ttnn.utils_for_testing import assert_with_pcc
from models.experimental.vovnet.common import load_torch_model
from models.demos.utils.common_demo_utils import get_mesh_mappers


class VovnetPerformanceRunnerInfra:
    def __init__(
        self,
        device,
        device_batch_size,
        act_dtype,
        weight_dtype,
        model_location_generator=None,
        resolution=(224, 224),
        torch_input_tensor=None,
        channels=3,
    ):
        torch.manual_seed(0)
        self.device_batch_size = device_batch_size
        self.resolution = resolution
        self.channels = channels
        self.pcc_passed = False
        self.pcc_message = "Did you forget to call validate()?"
        self.device = device
        self.inputs_mesh_mapper, _, self.output_mesh_composer = get_mesh_mappers(self.device)
        self.batch_size = self.device_batch_size * self.device.get_num_devices()
        self.act_dtype = act_dtype
        self.weight_dtype = weight_dtype
        self.model_location_generator = model_location_generator
        self.torch_input_tensor = torch_input_tensor
        self.torch_model = load_torch_model(self.model_location_generator).eval()
        state_dict = self.torch_model.state_dict()

        self.parameters = custom_preprocessor(device, state_dict)

        self.torch_input_tensor = (
            torch.randn((self.batch_size, self.channels, self.resolution[0], self.resolution[1]), dtype=torch.float32)
            if self.torch_input_tensor is None
            else self.torch_input_tensor
        )

        self.ttnn_vovnet_model = TtVoVNet(device=self.device, parameters=self.parameters, base_address="")

        self.torch_output_tensor = self.torch_model(self.torch_input_tensor)

    def _setup_l1_sharded_input(self, device, torch_input_tensor=None, min_channels=16):
        if is_wormhole_b0():
            core_grid = ttnn.CoreGrid(y=8, x=8)
        else:
            exit("Unsupported device")

        torch_input_tensor = self.torch_input_tensor if torch_input_tensor is None else torch_input_tensor

        n, c, h, w = torch_input_tensor.shape
        n = n // self.device.get_num_devices() if n // self.device.get_num_devices() != 0 else n
        num_cores = core_grid.x * core_grid.y
        shard_h = (n * w * h + num_cores - 1) // num_cores
        grid_size = core_grid
        grid_coord = ttnn.CoreCoord(grid_size.x - 1, grid_size.y - 1)
        shard_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), grid_coord)})
        shard_spec = ttnn.ShardSpec(shard_grid, (shard_h, min_channels), ttnn.ShardOrientation.ROW_MAJOR)
        input_mem_config = ttnn.MemoryConfig(
            ttnn.types.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.types.BufferType.L1, shard_spec
        )
        torch_input_tensor = torch_input_tensor.permute(0, 2, 3, 1)
        torch_input_tensor = torch_input_tensor.reshape(n * self.device.get_num_devices(), 1, h * w, c)
        if c < min_channels:
            padding_c = min_channels - c
        torch_input_tensor = F.pad(torch_input_tensor, (0, padding_c), "constant", 0)
        assert torch_input_tensor.ndim == 4, "Expected input tensor to have shape (BS, C, H, W)"
        input_tensor = [torch_input_tensor[i].unsqueeze(0) for i in range(torch_input_tensor.shape[0])]
        tt_inputs_host = ttnn.from_host_shards(
            [ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT) for t in input_tensor], device.shape
        )
        return tt_inputs_host, input_mem_config

    def setup_dram_sharded_input(
        self, device, torch_input_tensor=None, mesh_mapper=None, mesh_composer=None, min_channels=16
    ):
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
        self.output_tensor = self.ttnn_vovnet_model.forward(self.input_tensor)

    def validate(self, output_tensor=None, torch_output_tensor=None):
        ttnn_output_tensor = self.output_tensor if output_tensor is None else output_tensor
        torch_output_tensor = self.torch_output_tensor if torch_output_tensor is None else torch_output_tensor
        output_tensor = ttnn.to_torch(ttnn_output_tensor, mesh_composer=self.output_mesh_composer)
        self.pcc_passed, self.pcc_message = assert_with_pcc(self.torch_output_tensor, output_tensor, pcc=0.5)

        logger.info(
            f"Vovnet - batch_size={self.batch_size}, act_dtype={self.act_dtype}, weight_dtype={self.weight_dtype}, PCC={self.pcc_message}"
        )

    def dealloc_output(self):
        ttnn.deallocate(self.output_tensor)
