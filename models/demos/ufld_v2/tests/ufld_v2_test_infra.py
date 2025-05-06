# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
import torch
from loguru import logger
from tests.ttnn.utils_for_testing import assert_with_pcc
from models.utility_functions import is_wormhole_b0, divup
from models.demos.ufld_v2.ttnn.ttnn_ufld_v2 import TtnnUFLDv2
from models.demos.ufld_v2.reference.ufld_v2_model import TuSimple34
from ttnn.model_preprocessing import preprocess_model_parameters, infer_ttnn_module_args
from tests.ttnn.integration_tests.ufld_v2.test_ttnn_ufld_v2 import custom_preprocessor_whole_model


def load_torch_model():
    torch_model = TuSimple34(input_height=320, input_width=800)
    torch_model.eval()
    return torch_model


def load_ttnn_model(device, torch_model, torch_input_tensor):
    parameters = preprocess_model_parameters(
        initialize_model=lambda: torch_model,
        custom_preprocessor=custom_preprocessor_whole_model,
        device=device,
    )
    parameters.conv_args = {}
    parameters.conv_args = infer_ttnn_module_args(
        model=torch_model, run_model=lambda model: torch_model(torch_input_tensor), device=device
    )
    ttnn_model = TtnnUFLDv2(conv_args=parameters.conv_args, conv_pth=parameters, device=device)
    return ttnn_model


class UFLDv2TestInfra:
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
        input_shape = (batch_size, 3, 320, 800)
        self.torch_input_tensor = torch.randn(input_shape, dtype=torch.float32)
        torch_model = load_torch_model()
        self.ttnn_ufld_v2_model = load_ttnn_model(self.device, torch_model, self.torch_input_tensor)
        self.tt_input_tensor = self.torch_input_tensor.permute(0, 3, 1, 2)
        self.tt_input_tensor = ttnn.from_torch(self.tt_input_tensor, ttnn.bfloat16)
        self.torch_output_tensor_1, self.torch_output_tensor_2 = torch_model(self.torch_input_tensor)

    def run(self):
        self.output_tensor_1 = self.ttnn_ufld_v2_model(input=self.input_tensor, batch_size=self.batch_size)

    def setup_l1_sharded_input(self, device, torch_input_tensor=None):
        if is_wormhole_b0():
            core_grid = ttnn.CoreGrid(y=8, x=8)
        else:
            exit("Unsupported device")
        torch_input_tensor = self.torch_input_tensor if torch_input_tensor is None else torch_input_tensor

        n, c, h, w = torch_input_tensor.shape
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
        output_tensor_1 = ttnn.to_torch(self.output_tensor_1).squeeze(dim=0).squeeze(dim=0)
        valid_pcc = 0.96
        self.pcc_passed, self.pcc_message = assert_with_pcc(self.torch_output_tensor_1, output_tensor_1, pcc=valid_pcc)

        logger.info(f"UFLD_V2 batch_size={self.batch_size}, PCC={self.pcc_message}")

    def dealloc_output(self):
        ttnn.deallocate(self.output_tensor_1)


def create_test_infra(
    device,
    batch_size,
):
    return UFLDv2TestInfra(
        device,
        batch_size,
    )
