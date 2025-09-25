# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from typing import Optional, Tuple
import torch.nn as nn

from models.utility_functions import (
    torch_to_tt_tensor_rm,
)

import ttnn
from models.experimental.distilbert.tt.distilbert_multihead_self_attention import (
    TtMultiHeadSelfAttention,
)
from models.experimental.distilbert.tt.distilbert_ffn import TtFFN


class TtTransformerBlock(nn.Module):
    def __init__(self, config, state_dict=None, base_address="", device=None):
        super().__init__()
        self.config = config
        self.state_dict = state_dict
        self.base_address = base_address
        self.device = device

        if config.dim % config.n_heads != 0:
            raise ValueError(f"config.n_heads {config.n_heads} must divide config.dim {config.dim} evenly")

        self.attention = TtMultiHeadSelfAttention(
            self.config,
            state_dict=self.state_dict,
            base_address=self.base_address,
            device=self.device,
        )

        self.sa_gamma = torch_to_tt_tensor_rm(state_dict[f"{base_address}.sa_layer_norm.weight"], self.device)
        self.sa_beta = torch_to_tt_tensor_rm(state_dict[f"{base_address}.sa_layer_norm.bias"], self.device)

        self.sa_LayerNorm = ttnn.layer_norm

        self.ffn = TtFFN(
            self.config,
            state_dict=self.state_dict,
            base_address=self.base_address,
            device=self.device,
        )

        self.output_gamma = torch_to_tt_tensor_rm(state_dict[f"{base_address}.output_layer_norm.weight"], self.device)
        self.output_beta = torch_to_tt_tensor_rm(state_dict[f"{base_address}.output_layer_norm.bias"], self.device)

        self.output_LayerNorm = ttnn.layer_norm

    def forward(
        self,
        input: ttnn.Tensor,
        attn_mask: Optional[ttnn.Tensor] = None,
        head_mask: Optional[ttnn.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[ttnn.Tensor, ...]:
        sa_output = self.attention(
            query=input,
            key=input,
            value=input,
            mask=attn_mask,
            head_mask=head_mask,
            output_attention=output_attentions,
        )

        if output_attentions:
            (
                sa_output,
                sa_weights,
            ) = sa_output
        else:
            if type(sa_output) != tuple:
                raise TypeError(f"sa_output must be a tuple but it is {type(sa_output)} type")

            sa_output = sa_output[0]

        sa_output = self.sa_LayerNorm(
            ttnn.add(sa_output, input),
            epsilon=1e-12,
            weight=self.sa_gamma,
            bias=self.sa_beta,
        )

        ffn_output = self.ffn(sa_output)

        ffn_output = self.output_LayerNorm(
            ttnn.add(ffn_output, sa_output),
            epsilon=1e-12,
            weight=self.output_gamma,
            bias=self.output_beta,
        )

        output = (ffn_output,)
        if output_attentions:
            output = (sa_weights,) + output

        return output
