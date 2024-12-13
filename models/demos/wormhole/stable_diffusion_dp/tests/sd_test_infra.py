# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from loguru import logger
import os
import pytest
import torch
import torchvision
import math

import ttnn
from ttnn.model_preprocessing import (
    preprocess_model_parameters,
)
from models.utility_functions import (
    is_wormhole_b0,
    is_grayskull,
    divup,
)
from tqdm.auto import tqdm
from ttnn import unsqueeze_to_4D
from diffusers import StableDiffusionPipeline
from tests.ttnn.utils_for_testing import assert_with_pcc, comp_pcc
from models.demos.wormhole.stable_diffusion.sd_pndm_scheduler import TtPNDMScheduler
from models.demos.wormhole.stable_diffusion.tt.ttnn_functional_unet_2d_condition_model_new_conv import (
    UNet2DConditionModel as UNet2D,
)
from models.demos.wormhole.stable_diffusion_dp.tests.custom_preprocessing import create_custom_mesh_preprocessor
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import (
    AutoencoderKL,
    UNet2DConditionModel,
    StableDiffusionPipeline,
)


def unsqueeze_all_params_to_4d(params):
    if isinstance(params, dict):
        for key in params.keys():
            params[key] = unsqueeze_all_params_to_4d(params[key])
    elif isinstance(params, ttnn.ttnn.model_preprocessing.ParameterList):
        for i in range(len(params)):
            params[i] = unsqueeze_all_params_to_4d(params[i])
    elif isinstance(params, ttnn.Tensor):
        params = unsqueeze_to_4D(params)

    return params


def constant_prop_time_embeddings(timesteps, sample, time_proj):
    timesteps = timesteps[None]
    timesteps = timesteps.expand(sample.shape[0])
    t_emb = time_proj(timesteps)
    return t_emb


class SDTestInfra:
    def __init__(
        self,
        device,
        batch_size,
        input_shape,
        num_inference_steps,
    ):
        super().__init__()
        torch.manual_seed(0)
        self.pcc_passed = False
        self.pcc_message = "Did you forget to call validate()?"
        self.device = device
        self.batch_size = batch_size
        self.input_shape = input_shape
        self.num_inference_steps = num_inference_steps
        self.inputs_mesh_mapper, self.weights_mesh_mapper, self.output_mesh_composer = self.get_mesh_mappers(device)

        # setup pytorch model
        model_name = "CompVis/stable-diffusion-v1-4"
        pipe = StableDiffusionPipeline.from_pretrained(model_name, torch_dtype=torch.float32)
        torch_model = pipe.unet
        torch_model.eval()
        self.config = torch_model.config

        encoder_hidden_states_shape = [1, 2, 77, 768]
        self.class_labels = None
        self.attention_mask = None
        self.cross_attention_kwargs = None
        self.return_dict = True

        batch_size, in_channels, input_height, input_width = self.input_shape
        hidden_states_shape = [batch_size, in_channels, input_height, input_width]

        self.torch_input_tensor = torch.randn(hidden_states_shape)

        # 1. Load the autoencoder model which will be used to decode the latents into image space.
        vae = AutoencoderKL.from_pretrained(model_name, subfolder="vae")
        vae.to("cpu")
        vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)

        # 2. Load the tokenizer and text encoder to tokenize and encode the text.
        tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
        text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14")
        text_encoder.to("cpu")
        torch_model.to("cpu")

        # 3. load the PNDMScheduler with some fitting parameters.
        self.ttnn_scheduler = TtPNDMScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            num_train_timesteps=1000,
            skip_prk_steps=True,
            steps_offset=1,
            device=device,
        )

        BS, C, height, width = hidden_states_shape
        guidance_scale = 7.5  # Scale for classifier-free guidance
        generator = torch.manual_seed(174)  # 10233 Seed generator to create the inital latent noise
        # Initial random noise
        latents = torch.randn(
            (batch_size, self.config.in_channels, height // vae_scale_factor, width // vae_scale_factor),
            generator=generator,
        )
        latents = latents.to("cpu")
        self.ttnn_scheduler.set_timesteps(self.num_inference_steps)

        rand_latents = torch.tensor(latents)
        rand_latents = ttnn.from_torch(
            rand_latents,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            mesh_mapper=self.inputs_mesh_mapper,
        )
        ttnn_latent_model_input = ttnn.concat([rand_latents, rand_latents], dim=0)

        self.encoder_hidden_states = torch.randn(encoder_hidden_states_shape)
        parameters = preprocess_model_parameters(
            model_name=model_name,
            initialize_model=lambda: torch_model,
            custom_preprocessor=create_custom_mesh_preprocessor(self.weights_mesh_mapper),
            device=device,
        )
        # unsqueeze weight tensors to 4D for generating perf dump
        parameters = unsqueeze_all_params_to_4d(parameters)

        self.input = ttnn.from_torch(
            self.torch_input_tensor,
            ttnn.bfloat16,
            device=device,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            mesh_mapper=self.inputs_mesh_mapper,
        )
        self._tlist = []
        for t in self.ttnn_scheduler.timesteps:
            _t = constant_prop_time_embeddings(t, self.input, torch_model.time_proj)
            _t = _t.unsqueeze(0).unsqueeze(0)
            _t = _t.permute(2, 0, 1, 3)  # pre-permute temb
            _t = ttnn.from_torch(
                _t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, mesh_mapper=self.weights_mesh_mapper
            )
            self._tlist.append(_t)

        self.time_step = self.ttnn_scheduler.timesteps.tolist()

        encoder_hidden_states = torch.nn.functional.pad(self.encoder_hidden_states, (0, 0, 0, 19))
        self.ttnn_encoder_hidden_states = ttnn.from_torch(
            encoder_hidden_states,
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            mesh_mapper=self.inputs_mesh_mapper,
        )

        # golden
        self.torch_output_tensor = torch_model(
            self.torch_input_tensor,
            timestep=self.time_step[0],
            encoder_hidden_states=self.encoder_hidden_states.squeeze(0),
        ).sample

        # ttnn
        reader_patterns_cache = {}
        self.ttnn_model = UNet2D(device, parameters, batch_size, input_height, input_width, reader_patterns_cache)

    def get_mesh_mappers(self, device):
        is_mesh_device = isinstance(device, ttnn.MeshDevice)
        if is_mesh_device:
            inputs_mesh_mapper = ttnn.ShardTensorToMesh(device, dim=0)
            weights_mesh_mapper = ttnn.ReplicateTensorToMesh(
                device
            )  # causes unnecessary replication/takes more time on the first pass
            output_mesh_composer = ttnn.ConcatMeshToTensor(device, dim=0)
        else:
            inputs_mesh_mapper = None
            weights_mesh_mapper = None
            output_mesh_composer = None
        return inputs_mesh_mapper, weights_mesh_mapper, output_mesh_composer

    def setup_l1_sharded_input(self, device, torch_input_tensor=None):
        # 16329: trace_region_size issue
        max_grid_size = (8, 8)
        torch_input_tensor = self.torch_input_tensor if torch_input_tensor is None else torch_input_tensor
        batch_size, in_channels, input_height, input_width = torch_input_tensor.shape
        shard_grid = ttnn.CoreRangeSet(
            {
                ttnn.CoreRange(
                    ttnn.CoreCoord(0, 0),
                    ttnn.CoreCoord(max_grid_size[1] - 1, max_grid_size[0] - 1),
                )
            }
        )
        shard_shape = [
            divup(math.prod(torch_input_tensor.shape) // torch_input_tensor.shape[-1], max_grid_size[1]),
            torch_input_tensor.shape[-1],
        ]

        shard_orientation = ttnn.ShardOrientation.ROW_MAJOR
        shard_spec = ttnn.ShardSpec(shard_grid, shard_shape, shard_orientation, False)

        input_mem_config = ttnn.MemoryConfig(
            ttnn.types.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.types.BufferType.L1, shard_spec
        )
        tt_inputs_host = ttnn.from_torch(
            torch_input_tensor, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, mesh_mapper=self.inputs_mesh_mapper
        )
        return tt_inputs_host, input_mem_config

    def setup_dram_sharded_input(self, device, torch_input_tensor=None):
        torch_input_tensor = self.torch_input_tensor if torch_input_tensor is None else torch_input_tensor
        tt_inputs_host, input_mem_config = self.setup_l1_sharded_input(device, torch_input_tensor)
        dram_grid_size = device.dram_grid_size()
        dram_shard_spec = ttnn.ShardSpec(
            ttnn.CoreRangeSet(
                {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(dram_grid_size.x - 1, dram_grid_size.y - 1))}
            ),
            [
                divup(tt_inputs_host.volume() // tt_inputs_host.shape[-1], dram_grid_size.x),
                tt_inputs_host.shape[-1],
            ],
            ttnn.ShardOrientation.ROW_MAJOR,
            False,
        )
        sharded_mem_config_DRAM = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.DRAM, dram_shard_spec
        )

        return tt_inputs_host, sharded_mem_config_DRAM, input_mem_config

    def run(self, tt_input_tensor=None):
        self.output_tensor = self.ttnn_model(
            self.input_tensor,
            timestep=self._tlist[0],
            encoder_hidden_states=self.ttnn_encoder_hidden_states,
            class_labels=self.class_labels,
            attention_mask=self.attention_mask,
            cross_attention_kwargs=self.cross_attention_kwargs,
            return_dict=self.return_dict,
            config=self.config,
        )
        return self.output_tensor

    def validate(self, output_tensor=None, torch_output_tensor=None):
        output_tensor = self.output_tensor if output_tensor is None else output_tensor
        output_tensor = ttnn.to_torch(output_tensor, mesh_composer=self.output_mesh_composer)
        torch_output_tensor = self.torch_output_tensor if torch_output_tensor is None else torch_output_tensor
        valid_pcc = 0.98
        self.pcc_passed, self.pcc_message = assert_with_pcc(torch_output_tensor, output_tensor, pcc=valid_pcc)

        logger.info(f"Stable Diffusion batch_size={self.batch_size}, PCC={self.pcc_message}")


def create_test_infra(device, batch_size, input_shape, num_inference_steps):
    return SDTestInfra(device, batch_size, input_shape, num_inference_steps)
