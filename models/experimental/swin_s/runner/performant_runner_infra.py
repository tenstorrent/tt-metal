# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0


import torch
from loguru import logger
import ttnn
from torchvision import models
from tests.ttnn.integration_tests.swin_s.test_ttnn_swin_transformer import (
    create_custom_preprocessor,
    preprocess_attn_mask,
)
from models.experimental.swin_s.reference.swin_transformer import SwinTransformer
from models.experimental.swin_s.tt.tt_swin_transformer import TtSwinTransformer
from models.utility_functions import divup, is_wormhole_b0
from tests.ttnn.utils_for_testing import assert_with_pcc
from ttnn.model_preprocessing import preprocess_model_parameters


def load_torch_model():
    model = models.swin_s(weights="IMAGENET1K_V1")
    state_dict = model.state_dict()

    model.load_state_dict(state_dict)
    model.eval()
    torch_model = SwinTransformer(
        patch_size=[4, 4], embed_dim=96, depths=[2, 2, 18, 2], num_heads=[3, 6, 12, 24], window_size=[7, 7]
    )

    torch_model.load_state_dict(state_dict)
    torch_model.eval()

    return torch_model


class SwinSPerformanceRunnerInfra:
    def __init__(
        self,
        device,
        batch_size,
        act_dtype,
        weight_dtype,
        model_location_generator=None,
        resolution=(512, 512),
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

        self.torch_model = load_torch_model()

        self.torch_input_tensor = (
            torch.randn((1, 3, 512, 512), dtype=torch.float32)
            if self.torch_input_tensor is None
            else self.torch_input_tensor
        )

        self.parameters = preprocess_model_parameters(
            initialize_model=lambda: self.torch_model,
            custom_preprocessor=create_custom_preprocessor(self.device),
            device=self.device,
        )

        attn_mask_tuple = preprocess_attn_mask([1, 3, 512, 512], [4, 4], [7, 7], [3, 3], device)

        self.ttnn_swin_model = TtSwinTransformer(
            self.device,
            self.parameters,
            patch_size=[4, 4],
            embed_dim=96,
            depths=[2, 2, 18, 2],
            num_heads=[3, 6, 12, 24],
            window_size=[7, 7],
            attn_mask_tuple=attn_mask_tuple,
        )

        self.torch_output_tensor = self.torch_model(self.torch_input_tensor)

    def _setup_l1_sharded_input(self, device, torch_input_tensor=None):
        if is_wormhole_b0():
            core_grid = ttnn.CoreGrid(y=8, x=8)
        else:
            raise RuntimeError("Unsupported device: This implementation currently supports only Wormhole B0 devices.")

        num_devices = device.get_num_devices()
        torch_input_tensor = self.torch_input_tensor if torch_input_tensor is None else torch_input_tensor

        n, c, h, w = torch_input_tensor.shape

        padded_c = 16 if c < 16 else c  # If the channels < 16, pad the channels to 16 to run the Conv layer

        input_mem_config = ttnn.create_sharded_memory_config(
            [n, padded_c, h, w],
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
        self.output_tensor = self.ttnn_swin_model(self.input_tensor)

    def validate(self, output_tensor=None, torch_output_tensor=None):
        ttnn_output_tensor = self.output_tensor if output_tensor is None else output_tensor
        torch_output_tensor = self.torch_output_tensor if torch_output_tensor is None else torch_output_tensor
        output_tensor = ttnn.to_torch(ttnn_output_tensor)
        self.pcc_passed, self.pcc_message = assert_with_pcc(
            self.torch_output_tensor, output_tensor, pcc=0.95
        )  # PCC:0.953924198820544

        logger.info(
            f"Swin S - batch_size={self.batch_size}, act_dtype={self.act_dtype}, weight_dtype={self.weight_dtype}, PCC={self.pcc_message}"
        )

    def dealloc_output(self):
        ttnn.deallocate(self.output_tensor)
