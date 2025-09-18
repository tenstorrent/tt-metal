# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0


import torch
from loguru import logger
import ttnn
from models.experimental.swin_s.tests.pcc.test_ttnn_swin_transformer import (
    create_custom_mesh_preprocessor,
    preprocess_attn_mask,
)
from models.experimental.swin_s.reference.swin_transformer import SwinTransformer
from models.experimental.swin_s.tt.tt_swin_transformer import TtSwinTransformer
from models.common.utility_functions import divup, is_wormhole_b0
from tests.ttnn.utils_for_testing import assert_with_pcc
from ttnn.model_preprocessing import preprocess_model_parameters
from models.experimental.swin_s.common import load_torch_model
from models.demos.utils.common_demo_utils import get_mesh_mappers


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
        channels=3,
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
        self.num_devices = self.device.get_num_devices()
        self.mesh_mapper, self.weights_mesh_mapper, self.mesh_composer = get_mesh_mappers(self.device)
        torch_model = SwinTransformer(
            patch_size=[4, 4], embed_dim=96, depths=[2, 2, 18, 2], num_heads=[3, 6, 12, 24], window_size=[7, 7]
        )
        self.torch_model = load_torch_model(torch_model=torch_model, model_location_generator=model_location_generator)

        self.torch_input_tensor = (
            torch.randn(
                (self.batch_size * self.num_devices, channels, resolution[0], resolution[1]), dtype=torch.float32
            )
            if self.torch_input_tensor is None
            else self.torch_input_tensor
        )

        self.parameters = preprocess_model_parameters(
            initialize_model=lambda: self.torch_model,
            custom_preprocessor=create_custom_mesh_preprocessor(self.weights_mesh_mapper),
            device=self.device,
        )

        attn_mask_tuple = preprocess_attn_mask(
            [1, 3, 512, 512], [4, 4], [7, 7], [3, 3], device, weights_mesh_mapper=self.weights_mesh_mapper
        )

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

    def _setup_l1_sharded_input(self, device, torch_input_tensor=None, channels=3, padded_channels=16):
        if is_wormhole_b0():
            core_grid = ttnn.CoreGrid(y=8, x=8)
        else:
            raise RuntimeError("Unsupported device: This implementation currently supports only Wormhole B0 devices.")

        num_devices = device.get_num_devices()
        torch_input_tensor = self.torch_input_tensor if torch_input_tensor is None else torch_input_tensor

        n, c, h, w = torch_input_tensor.shape
        assert n % self.num_devices == 0, f"n isn't evenly divided by the available number of devices"
        n = n // self.num_devices if n // self.num_devices != 0 else n
        if c == channels:
            c = padded_channels

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
        output_tensor = ttnn.to_torch(ttnn_output_tensor, mesh_composer=self.mesh_composer)
        self.pcc_passed, self.pcc_message = assert_with_pcc(self.torch_output_tensor, output_tensor, pcc=0.94)

        logger.info(
            f"Swin S - batch_size={self.batch_size* self.num_devices}, act_dtype={self.act_dtype}, weight_dtype={self.weight_dtype}, PCC={self.pcc_message}"
        )

    def dealloc_output(self):
        ttnn.deallocate(self.output_tensor)
