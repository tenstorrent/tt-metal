# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
import torch.nn as nn
from models.experimental.stable_diffusion_xl_base.vae.tt.tt_decoder import TtDecoder
from models.experimental.stable_diffusion_xl_base.tt.sdxl_utility import prepare_conv_params


class TtAutoencoderKL(nn.Module):
    def __init__(self, device, state_dict):
        super().__init__()

        self.device = device

        self.stride = (1, 1)
        self.padding = (0, 0)
        self.dilation = (1, 1)
        self.groups = 1

        self.decoder = TtDecoder(device, state_dict)

        post_quant_conv_weights = state_dict[f"post_quant_conv.weight"]
        post_quant_conv_bias = state_dict[f"post_quant_conv.bias"].unsqueeze(0).unsqueeze(0).unsqueeze(0)

        (
            self.compute_config,
            self.conv_config,
            self.tt_post_quant_conv_weights,
            self.tt_post_quant_conv_bias,
            self.conv_params,
        ) = prepare_conv_params(
            device,
            post_quant_conv_weights,
            post_quant_conv_bias,
            ttnn.bfloat16,
            act_block_h_override=0,
            fp32_dest_acc_en=True,
            math_fidelity=ttnn.MathFidelity.LoFi,
        )
        self.conv_config.deallocate_activation = True

    def forward(self, hidden_states, input_shape):
        B, C, H, W = input_shape

        [hidden_states, [H, W], [d_w, d_b]] = ttnn.conv2d(
            input_tensor=hidden_states,
            weight_tensor=self.tt_post_quant_conv_weights,
            in_channels=self.conv_params["input_channels"],
            out_channels=self.conv_params["output_channels"],
            device=self.device,
            bias_tensor=self.tt_post_quant_conv_bias,
            kernel_size=self.conv_params["kernel_size"],
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            batch_size=B,
            input_height=H,
            input_width=W,
            conv_config=self.conv_config,
            compute_config=self.compute_config,
            groups=self.groups,
            memory_config=None,
            return_output_dim=True,
            return_weights_and_bias=True,
        )
        C = self.conv_params["output_channels"]

        self.tt_post_quant_conv_weights = d_w
        self.tt_post_quant_conv_bias = d_b

        hidden_states = ttnn.sharded_to_interleaved(hidden_states, ttnn.L1_MEMORY_CONFIG)
        hidden_states = self.decoder(hidden_states, [B, C, H, W])

        return hidden_states
