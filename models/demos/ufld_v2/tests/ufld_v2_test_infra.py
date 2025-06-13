# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import os

import torch
import torch.nn.functional as F
from loguru import logger
from ttnn.model_preprocessing import infer_ttnn_module_args, preprocess_model_parameters

import ttnn
from models.demos.ufld_v2.reference.ufld_v2_model import TuSimple34
from models.demos.ufld_v2.ttnn.ttnn_ufld_v2 import TtnnUFLDv2
from models.utility_functions import divup, is_wormhole_b0
from tests.ttnn.integration_tests.ufld_v2.test_ttnn_ufld_v2 import custom_preprocessor_whole_model
from tests.ttnn.utils_for_testing import assert_with_pcc


def load_torch_model():
    torch_model = TuSimple34(input_height=320, input_width=800)
    torch_model.eval()
    weights_path = "models/demos/ufld_v2/tusimple_res34.pth"
    if not os.path.exists(weights_path):
        os.system("bash models/demos/ufld_v2/weights_download.sh")
    state_dict = torch.load(weights_path)
    new_state_dict = {}
    for key, value in state_dict["model"].items():
        new_key = key.replace("model.", "res_model.")
        new_state_dict[new_key] = value
    torch_model.load_state_dict(new_state_dict)
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


class UFLDPerformanceRunnerInfra:
    def __init__(
        self,
        device,
        batch_size=1,
        act_dtype=ttnn.bfloat8_b,
        weight_dtype=ttnn.bfloat8_b,
        model_location_generator=None,
        resolution=(320, 800),
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
        self.torch_input_tensor = (
            torch.randn((1, 3, 320, 800), dtype=torch.float32)
            if self.torch_input_tensor is None
            else self.torch_input_tensor
        )
        self.torch_model = load_torch_model()
        self.ttnn_ufld_v2_model = load_ttnn_model(self.device, self.torch_model, self.torch_input_tensor)
        self.torch_output_tensor_1, self.torch_output_tensor_2 = self.torch_model(self.torch_input_tensor)

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
        torch_input_tensor = F.pad(torch_input_tensor, (0, 13))
        torch_input_tensor = torch_input_tensor.reshape(
            1,
            1,
            (torch_input_tensor.shape[0] * torch_input_tensor.shape[1] * torch_input_tensor.shape[2]),
            torch_input_tensor.shape[3],
        )
        tt_inputs_host = ttnn.from_torch(torch_input_tensor, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
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
        self.output_tensor_1 = self.ttnn_ufld_v2_model(input=self.input_tensor, batch_size=self.batch_size)

    def validate(self, output_tensor_1=None, torch_output_tensor_1=None):
        ttnn_output_tensor = self.output_tensor_1 if output_tensor_1 is None else output_tensor_1
        torch_output_tensor = self.torch_output_tensor_1 if torch_output_tensor_1 is None else torch_output_tensor_1
        output_tensor = ttnn.to_torch(ttnn_output_tensor).squeeze(0).squeeze(0)
        self.valid_pcc = 0.99
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
        device, batch_size, act_dtype, weight_dtype, model_location_generator, resolution, torch_input_tensor
    )
