# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0
import os
import sys
import ttnn
import torch
import torch.nn as nn
from torchvision import models
from loguru import logger
from tests.ttnn.utils_for_testing import assert_with_pcc
from ttnn.model_preprocessing import preprocess_model_parameters
from tests.ttnn.integration_tests.swin_s.test_ttnn_swin_transformer import (
    preprocess_attn_mask,
    create_custom_preprocessor,
)
from models.experimental.functional_swin_s.reference.swin_transformer import SwinTransformer
from models.experimental.functional_swin_s.tt.tt_swin_transformer import TtSwinTransformer

# from models.experimental.functional_yolov8x.tt.ttnn_yolov8x import YOLOv8xModel
# from models.experimental.functional_yolov8x.reference import yolov8x_utils
# from models.experimental.functional_yolov8x.tt.ttnn_yolov8x_utils import custom_preprocessor

from models.utility_functions import (
    is_wormhole_b0,
    divup,
)


def load_torch_model():
    model = models.swin_s(weights="IMAGENET1K_V1")
    state_dict = model.state_dict()

    torch_model = SwinTransformer(
        patch_size=[4, 4], embed_dim=96, depths=[2, 2, 18, 2], num_heads=[3, 6, 12, 24], window_size=[7, 7]
    )
    torch_model.load_state_dict(state_dict)
    torch_model.eval()
    return torch_model


def load_ttnn_model(device, torch_model):
    parameters = preprocess_model_parameters(
        initialize_model=lambda: torch_model, custom_preprocessor=create_custom_preprocessor(device), device=device
    )

    attn_mask_tuple = preprocess_attn_mask([1, 3, 512, 512], [4, 4], [7, 7], [3, 3], device)

    # Convert the model to TTNN
    ttnn_model = TtSwinTransformer(
        device,
        parameters,
        patch_size=[4, 4],
        embed_dim=96,
        depths=[2, 2, 18, 2],
        num_heads=[3, 6, 12, 24],
        window_size=[7, 7],
        attn_mask_tuple=attn_mask_tuple,
    )
    return ttnn_model


class Swin_S_TestInfra:
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
        torch_model = load_torch_model()
        self.ttnn_model = load_ttnn_model(device=self.device, torch_model=torch_model)
        input_shape = (1, 3, 512, 512)
        self.torch_input_tensor = torch.randn(input_shape)
        self.input_tensor = ttnn.from_torch(
            self.torch_input_tensor,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        self.torch_output_tensor = torch_model(self.torch_input_tensor)

    def run(self):
        # input_tensor = ttnn.to_device(self.input_tensor, device=device, memory_config=ttnn.L1_MEMORY_CONFIG)
        self.output_tensor = self.ttnn_model(self.input_tensor)

    def setup_l1_sharded_input(self, device, torch_input_tensor=None):
        if is_wormhole_b0():
            core_grid = ttnn.CoreGrid(y=8, x=8)
        else:
            exit("Unsupported device")
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
        # torch_input_tensor = torch_input_tensor.permute(0, 2, 3, 1)
        # torch_input_tensor = torch_input_tensor.reshape(1, 1, h * w * n, c)
        tt_inputs_host = ttnn.from_torch(
            torch_input_tensor,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        # tt_inputs_host = ttnn.pad(tt_inputs_host, [1, 1, n * h * w, 16], [0, 0, 0, 0], 0)

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
        output_tensor = ttnn.to_torch(self.output_tensor)

        valid_pcc = 0.8
        self.pcc_passed, self.pcc_message = assert_with_pcc(self.torch_output_tensor, output_tensor, pcc=valid_pcc)

        logger.info(f"Yolov8x batch_size={self.batch_size}, PCC={self.pcc_message}")

    def dealloc_output(self):
        ttnn.deallocate(self.output_tensor)


def create_test_infra(
    device,
    batch_size,
):
    return Swin_S_TestInfra(
        device,
        batch_size,
    )
