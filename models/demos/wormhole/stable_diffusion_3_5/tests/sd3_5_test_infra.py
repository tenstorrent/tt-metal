# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from loguru import logger
import os
import pytest
import torch
import ttnn
import numpy as np

from tests.ttnn.utils_for_testing import assert_with_pcc
from models.utility_functions import (
    is_wormhole_b0,
    is_grayskull,
    divup,
)

from ttnn.model_preprocessing import preprocess_model_parameters
from tests.ttnn.integration_tests.stable_diffusion3_5.test_ttnn_sd3_transformer_2d_model import (
    create_custom_preprocessor,
)
from models.experimental.functional_stable_diffusion3_5.ttnn.ttnn_sd3_transformer_2d_model import (
    ttnn_SD3Transformer2DModel,
)

# from models.experimental.functional_stable_diffusion3_5.demo.ttnn_pipeline import ttnnStableDiffusion3Pipeline
from diffusers import StableDiffusion3Pipeline
from models.experimental.functional_stable_diffusion3_5.reference.sd3_transformer_2d_model import SD3Transformer2DModel


class SD35mTestInfra:
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

        pipe = StableDiffusion3Pipeline.from_pretrained(
            "stabilityai/stable-diffusion-3.5-medium", torch_dtype=torch.bfloat16
        )
        config = pipe.transformer.config

        reference_model = SD3Transformer2DModel(
            sample_size=128,
            patch_size=2,
            in_channels=16,
            num_layers=24,
            attention_head_dim=64,
            num_attention_heads=24,
            joint_attention_dim=4096,
            caption_projection_dim=1536,
            pooled_projection_dim=2048,
            out_channels=16,
            pos_embed_max_size=384,
            dual_attention_layers=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
            qk_norm="rms_norm",
            config=config,
        )
        reference_model.load_state_dict(pipe.transformer.state_dict())
        reference_model.eval()
        self.parameters = preprocess_model_parameters(
            initialize_model=lambda: reference_model,
            custom_preprocessor=create_custom_preprocessor(device),
            device=device,
        )
        self.parameters["pos_embed"]["proj"]["weight"] = ttnn.from_device(
            self.parameters["pos_embed"]["proj"]["weight"]
        )
        self.parameters["pos_embed"]["proj"]["bias"] = ttnn.from_device(self.parameters["pos_embed"]["proj"]["bias"])

        self.ttnn_model = ttnn_SD3Transformer2DModel(
            sample_size=128,
            patch_size=2,
            in_channels=16,
            num_layers=24,
            attention_head_dim=64,
            num_attention_heads=24,
            joint_attention_dim=4096,
            caption_projection_dim=1536,
            pooled_projection_dim=2048,
            out_channels=16,
            pos_embed_max_size=384,
            dual_attention_layers=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
            qk_norm="rms_norm",
            config=config,
            parameters=self.parameters,
        )

        ################

        """
        j = 0
        numpy_array = np.load(
            "../../sd35_512_unopt/tt-metal/models/experimental/functional_stable_diffusion3_5/demo/demo_unoptimized_512x512__hidden_states_"
            + str(j)
            + ".npy"
        )
        self.torch_input_hidden_states = torch.from_numpy(numpy_array)  # .to(dtype=torch.bfloat16)
        numpy_array = np.load(
            "../../sd35_512_unopt/tt-metal/models/experimental/functional_stable_diffusion3_5/demo/demo_unoptimized_512x512__encoder_hidden_"
            + str(j)
            + ".npy"
        )
        self.torch_input_encoder_hidden_states = torch.from_numpy(numpy_array)  # .to(dtype=torch.bfloat16)
        numpy_array = np.load(
            "../../sd35_512_unopt/tt-metal/models/experimental/functional_stable_diffusion3_5/demo/demo_unoptimized_512x512__pooled_proj_"
            + str(j)
            + ".npy"
        )
        self.torch_input_pooled_projections = torch.from_numpy(numpy_array)  # .to(dtype=torch.bfloat16)

        numpy_array = np.load(
                "../../sd35_512_unopt/tt-metal/models/experimental/functional_stable_diffusion3_5/demo/demo_unoptimized_512x512___timesteps_proj_"
                + str(j)
                + ".npy"
            )
        self.torch_input_timesteps_proj = torch.from_numpy(numpy_array)  # .to(dtype=torch.bfloat16)


        self.torch_output_tensor = torch.from_numpy(
            np.load(
                "../../sd35_512_unopt/tt-metal/models/experimental/functional_stable_diffusion3_5/demo/demo_unoptimized_512x512__noise_pred_"
                + str(j)
                + ".npy"
            )
        )

        """

        j = 0

        self.torch_input_hidden_states = torch.load("pt_demo_trace/demo_in0_noTrace_" + str(j) + ".pt")
        self.torch_input_encoder_hidden_states = torch.load("pt_demo_trace/demo_in1_noTrace_" + str(j) + ".pt")
        self.torch_input_pooled_projections = torch.load("pt_demo_trace/demo_in2_noTrace_" + str(j) + ".pt")
        self.torch_input_timesteps_proj = torch.load("pt_demo_trace/demo_in3_noTrace_" + str(j) + ".pt")
        self.torch_output_tensor = torch.load("pt_demo_trace/noise_pred_noTrace_" + str(j) + ".pt")

    def run(
        self,
        input_tensor_hidden_state,
        input_tensor_encoder_hidden_state,
        input_tensor_pooled_proj,
        input_tensor_timestep,
    ):
        self.output_tensor = self.ttnn_model(
            input_tensor_hidden_state,
            input_tensor_encoder_hidden_state,
            input_tensor_pooled_proj,
            input_tensor_timestep,
            None,
            None,
            None,
            parameters=self.parameters,
        )

    def setup_l1_sharded_input_hidden_state(self, device, torch_input_tensor=None):
        if is_wormhole_b0():
            core_grid = ttnn.CoreGrid(y=8, x=8)
        else:
            exit("Unsupported device")
        num_devices = 1 if isinstance(device, ttnn.Device) else device.get_num_devices()
        # torch tensor
        torch_input_tensor = self.torch_input_hidden_states if torch_input_tensor is None else torch_input_tensor
        ## shape = ttnn.Shape([2, 16, 64, 64])

        n, c, h, w = torch_input_tensor.shape
        # sharded mem config for fold input
        num_cores = core_grid.x * core_grid.y
        shard_h = (n * w * h + num_cores - 1) // num_cores
        grid_size = core_grid
        grid_coord = ttnn.CoreCoord(grid_size.x - 1, grid_size.y - 1)
        shard_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), grid_coord)})
        shard_spec = ttnn.ShardSpec(shard_grid, (shard_h, 16), ttnn.ShardOrientation.ROW_MAJOR)  # , False)
        input_mem_config = ttnn.MemoryConfig(
            ttnn.types.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.types.BufferType.L1, shard_spec
        )
        torch_input_tensor = torch_input_tensor.permute(0, 2, 3, 1)
        tt_inputs_host = ttnn.from_torch(
            torch_input_tensor, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT
        )  # , layout=ttnn.TILE_LAYOUT) #
        return tt_inputs_host, input_mem_config

    def setup_dram_sharded_input_hidden_state(
        self, device, torch_input_tensor=None, mesh_mapper=None, mesh_composer=None
    ):
        tt_inputs_host, input_mem_config = self.setup_l1_sharded_input_hidden_state(device)
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
            # False,
        )
        sharded_mem_config_DRAM = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.DRAM, dram_shard_spec
        )

        return tt_inputs_host, sharded_mem_config_DRAM, input_mem_config

    def setup_l1_sharded_input_encoder_hidden_state(self, device, torch_input_tensor=None):
        if is_wormhole_b0():
            core_grid = ttnn.CoreGrid(y=8, x=8)
        else:
            exit("Unsupported device")
        num_devices = 1 if isinstance(device, ttnn.Device) else device.get_num_devices()
        # torch tensor
        torch_input_tensor = (
            self.torch_input_encoder_hidden_states if torch_input_tensor is None else torch_input_tensor
        )
        ## shape = ttnn.Shape([2, 154[160], 4096])

        if len(torch_input_tensor.shape) == 3:
            torch_input_tensor = torch_input_tensor.unsqueeze(1)

        n, c, h, w = torch_input_tensor.shape
        # sharded mem config for fold input
        num_cores = core_grid.x * core_grid.y
        shard_w = (w + num_cores - 1) // num_cores
        # shard_h = n * c * h
        shard_h = 320  # 2 x 154[160]
        grid_size = core_grid
        grid_coord = ttnn.CoreCoord(grid_size.x - 1, grid_size.y - 1)
        shard_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), grid_coord)})
        shard_spec = ttnn.ShardSpec(shard_grid, (shard_h, shard_w), ttnn.ShardOrientation.ROW_MAJOR)  # , False)
        input_mem_config = ttnn.MemoryConfig(
            ttnn.types.TensorMemoryLayout.WIDTH_SHARDED, ttnn.types.BufferType.L1, shard_spec
        )
        tt_inputs_host = ttnn.from_torch(
            torch_input_tensor, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
        )  # , layout=ttnn.ROW_MAJOR_LAYOUT)
        return tt_inputs_host, input_mem_config

    def setup_dram_sharded_input_encoder_hidden_state(
        self, device, torch_input_tensor=None, mesh_mapper=None, mesh_composer=None
    ):
        tt_inputs_host, input_mem_config = self.setup_l1_sharded_input_encoder_hidden_state(device)
        dram_grid_size = device.dram_grid_size()
        dram_shard_spec = ttnn.ShardSpec(
            ttnn.CoreRangeSet(
                {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(dram_grid_size.x - 1, dram_grid_size.y - 1))}
            ),
            [
                320,
                divup(tt_inputs_host.shape[-1], (dram_grid_size.x * dram_grid_size.y)),
            ],
            ttnn.ShardOrientation.ROW_MAJOR,
            # False,
        )
        sharded_mem_config_DRAM = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.DRAM, dram_shard_spec
        )

        return tt_inputs_host, sharded_mem_config_DRAM, input_mem_config

    def setup_l1_sharded_input_pooled_proj(self, device, torch_input_tensor=None):
        if is_wormhole_b0():
            core_grid = ttnn.CoreGrid(y=8, x=8)
        else:
            exit("Unsupported device")
        num_devices = 1 if isinstance(device, ttnn.Device) else device.get_num_devices()
        # torch tensor
        torch_input_tensor = self.torch_input_pooled_projections if torch_input_tensor is None else torch_input_tensor
        ## shape = ttnn.Shape([2[32], 2048])

        if len(torch_input_tensor.shape) == 3:
            torch_input_tensor = torch_input_tensor.unsqueeze(0)
        elif len(torch_input_tensor.shape) == 2:
            torch_input_tensor = torch_input_tensor.unsqueeze(0).unsqueeze(0)

        n, c, h, w = torch_input_tensor.shape
        # sharded mem config for fold input
        num_cores = core_grid.x * core_grid.y
        shard_w = (w + num_cores - 1) // num_cores
        shard_h = 32  # n*c*h
        grid_size = core_grid
        grid_coord = ttnn.CoreCoord(grid_size.x - 1, grid_size.y - 1)
        shard_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), grid_coord)})
        shard_spec = ttnn.ShardSpec(shard_grid, (shard_h, shard_w), ttnn.ShardOrientation.ROW_MAJOR)  # , False)
        input_mem_config = ttnn.MemoryConfig(
            ttnn.types.TensorMemoryLayout.WIDTH_SHARDED, ttnn.types.BufferType.L1, shard_spec
        )
        tt_inputs_host = ttnn.from_torch(
            torch_input_tensor, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
        )  # , layout=ttnn.ROW_MAJOR_LAYOUT)
        return tt_inputs_host, input_mem_config

    def setup_l1_sharded_input_timestep(self, device, torch_input_tensor=None):
        if is_wormhole_b0():
            core_grid = ttnn.CoreGrid(y=8, x=1)
        else:
            exit("Unsupported device")
        num_devices = 1 if isinstance(device, ttnn.Device) else device.get_num_devices()
        # torch tensor
        torch_input_tensor = self.torch_input_timesteps_proj if torch_input_tensor is None else torch_input_tensor
        ## shape = ttnn.Shape([2[32], 256])

        if len(torch_input_tensor.shape) == 3:
            torch_input_tensor = torch_input_tensor.unsqueeze(0)
        elif len(torch_input_tensor.shape) == 2:
            torch_input_tensor = torch_input_tensor.unsqueeze(0).unsqueeze(0)

        n, c, h, w = torch_input_tensor.shape
        # sharded mem config for fold input
        num_cores = core_grid.x * core_grid.y
        shard_w = (w + num_cores - 1) // num_cores
        shard_h = 32  # n*c*h
        grid_size = core_grid
        grid_coord = ttnn.CoreCoord(grid_size.x - 1, grid_size.y - 1)
        shard_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), grid_coord)})
        shard_spec = ttnn.ShardSpec(shard_grid, (shard_h, shard_w), ttnn.ShardOrientation.ROW_MAJOR)  # , False)
        input_mem_config = ttnn.MemoryConfig(
            ttnn.types.TensorMemoryLayout.WIDTH_SHARDED, ttnn.types.BufferType.L1, shard_spec
        )
        # input_mem_config = ttnn.L1_MEMORY_CONFIG
        tt_inputs_host = ttnn.from_torch(torch_input_tensor, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
        return tt_inputs_host, input_mem_config

    def validate(self, output_tensor=None):
        output_tensor = self.output_tensor if output_tensor is None else output_tensor

        ttnn_output_tensor = ttnn.to_torch(output_tensor[0])
        valid_pcc = 0.3
        self.pcc_passed, self.pcc_message = assert_with_pcc(self.torch_output_tensor, ttnn_output_tensor, pcc=valid_pcc)

        logger.info(f"SD3.5m batch_size={self.batch_size}, PCC={self.pcc_message}")

    def dealloc_output(self):
        ttnn.deallocate(self.output_tensor[0])


def create_test_infra(
    device,
    batch_size,
    model_location_generator=None,
):
    return SD35mTestInfra(
        device,
        batch_size,
        model_location_generator,
    )
