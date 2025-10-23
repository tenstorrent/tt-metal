# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import ttnn
import torch
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
from diffusers.models.modeling_outputs import AutoencoderKLOutput
from models.common.lightweightmodule import LightweightModule
from models.experimental.stable_diffusion_xl_base.vae.tt.tt_decoder import TtDecoder
from models.experimental.stable_diffusion_xl_base.vae.tt.tt_encoder import TtEncoder
from models.experimental.stable_diffusion_xl_base.tt.sdxl_utility import prepare_linear_params


class TtAutoencoderKL(LightweightModule):
    def __init__(
        self,
        device,
        state_dict,
        model_config,
    ):
        super().__init__()

        self.device = device
        self.model_config = model_config

        self.stride = (1, 1)
        self.padding = (0, 0)
        self.dilation = (1, 1)
        self.groups = 1

        self.decoder = TtDecoder(
            device,
            state_dict,
            model_config,
        )

        self.encoder = TtEncoder(
            device,
            state_dict,
            model_config,
        )

        quant_conv_weights = state_dict[f"quant_conv.weight"].squeeze()
        quant_conv_bias = state_dict[f"quant_conv.bias"]

        self.tt_quant_conv_weights, self.tt_quant_conv_bias = prepare_linear_params(
            device, quant_conv_weights, quant_conv_bias, model_config.conv_w_dtype
        )

        post_quant_conv_weights = state_dict[f"post_quant_conv.weight"].squeeze()
        post_quant_conv_bias = state_dict[f"post_quant_conv.bias"]

        self.tt_post_quant_conv_weights, self.tt_post_quant_conv_bias = prepare_linear_params(
            device, post_quant_conv_weights, post_quant_conv_bias, model_config.conv_w_dtype
        )

    def encode(self, hidden_states):
        assert isinstance(
            hidden_states, torch.Tensor
        ), "encode only supports torch tensors as input, conversion to ttnn done within"

        B, C, H, W = hidden_states.shape

        initial_batch_size = hidden_states.shape[0]
        hidden_states = torch.permute(hidden_states, (0, 2, 3, 1))
        hidden_states = torch.reshape(hidden_states, (B, 1, H * W, C))
        hidden_states = ttnn.from_torch(
            hidden_states,
            dtype=ttnn.bfloat16,
            device=self.device,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(self.device, list(self.device.shape), dims=(None, 0)),
        )
        B = hidden_states.shape[0]

        hidden_states, [C, H, W] = self.encoder(hidden_states, [B, C, H, W])
        hidden_states = ttnn.linear(
            hidden_states,
            self.tt_quant_conv_weights,
            bias=self.tt_quant_conv_bias,
        )

        h = ttnn.to_torch(hidden_states, mesh_composer=ttnn.ConcatMeshToTensor(self.device, dim=0)).float()[
            :initial_batch_size, ...
        ]
        B = h.shape[0]
        h = h.reshape(B, H, W, C)
        h = torch.permute(h, (0, 3, 1, 2))

        h = h.chunk(B, dim=0)
        posterior = [DiagonalGaussianDistribution(h_i) for h_i in h]
        return AutoencoderKLOutput(latent_dist=posterior)

    def decode(self, hidden_states, input_shape):
        B, C, H, W = input_shape

        pre_conv_hidden_states = hidden_states
        hidden_states = ttnn.linear(
            hidden_states,
            self.tt_post_quant_conv_weights,
            bias=self.tt_post_quant_conv_bias,
        )
        ttnn.deallocate(pre_conv_hidden_states)

        hidden_states = ttnn.sharded_to_interleaved(hidden_states, ttnn.L1_MEMORY_CONFIG)
        hidden_states, [C, H, W] = self.decoder(hidden_states, [B, C, H, W])

        return hidden_states, [C, H, W]

    def forward(self, *args, **kwargs):
        raise NotImplementedError("Use decode() or encode() instead of forward()")
