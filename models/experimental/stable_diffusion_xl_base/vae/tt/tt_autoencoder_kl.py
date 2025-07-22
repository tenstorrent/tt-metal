# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
import torch.nn as nn
from models.experimental.stable_diffusion_xl_base.vae.tt.tt_decoder import TtDecoder
from models.experimental.stable_diffusion_xl_base.tt.sdxl_utility import prepare_linear_params


class TtAutoencoderKL(nn.Module):
    def __init__(self, device, state_dict, model_config, batch_size=1):
        super().__init__()

        self.device = device
        self.model_config = model_config

        self.stride = (1, 1)
        self.padding = (0, 0)
        self.dilation = (1, 1)
        self.groups = 1

        self.decoder = TtDecoder(device, state_dict, model_config, batch_size=batch_size)

        post_quant_conv_weights = state_dict[f"post_quant_conv.weight"].squeeze()
        post_quant_conv_bias = state_dict[f"post_quant_conv.bias"]

        self.tt_post_quant_conv_weights, self.tt_post_quant_conv_bias = prepare_linear_params(
            device, post_quant_conv_weights, post_quant_conv_bias, model_config.conv_w_dtype
        )

    def forward(self, hidden_states, input_shape):
        B, C, H, W = input_shape

        pre_conv_hidden_states = hidden_states
        hidden_states = ttnn.linear(
            hidden_states,
            self.tt_post_quant_conv_weights,
            bias=self.tt_post_quant_conv_bias,
        )
        ttnn.deallocate(pre_conv_hidden_states)

        hidden_states = ttnn.sharded_to_interleaved(hidden_states, ttnn.L1_MEMORY_CONFIG)
        hidden_states = self.decoder(hidden_states, [B, C, H, W])

        return hidden_states
