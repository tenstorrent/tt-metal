# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.experimental.devstarl2_small.tt.tt_pixtralattn import TtMistralImageAttention
from models.experimental.devstarl2_small.tt.tt_pixtralmlp import MistralTTVisionMLP
from models.experimental.devstarl2_small.tt.tt_pixtralnorm import TtPixtralRMSNorm


class TtPixtralAttentionLayer(LightweightModule):
    """TT composition of HF ``PixtralAttentionLayer``: attention_norm -> attention + residual -> ffn_norm -> mlp + residual."""

    def __init__(
        self,
        mesh_device,
        tt_ccl,
        state_dict,
        state_dict_prefix,
        weight_cache_path,
        dtype,
        configuration,
    ):
        super().__init__()
        self.attention_norm = TtPixtralRMSNorm(
            mesh_device=mesh_device,
            state_dict=state_dict,
            eps=1e-5,
            weight_key=f"{state_dict_prefix}attention_norm.weight",
            dtype=dtype,
        )
        self.attention = TtMistralImageAttention(
            mesh_device=mesh_device,
            tt_ccl=tt_ccl,
            state_dict=state_dict,
            state_dict_prefix=f"{state_dict_prefix}attention.",
            weight_cache_path=weight_cache_path,
            dtype=dtype,
            configuration=configuration,
        )
        self.ffn_norm = TtPixtralRMSNorm(
            mesh_device=mesh_device,
            state_dict=state_dict,
            eps=1e-5,
            weight_key=f"{state_dict_prefix}ffn_norm.weight",
            dtype=dtype,
        )
        self.mlp = MistralTTVisionMLP(
            mesh_device=mesh_device,
            args=configuration,
            state_dict=state_dict,
            weight_cache_path=weight_cache_path,
            dtype=dtype,
            state_dict_prefix=f"{state_dict_prefix}feed_forward.",
        )

    def forward(self, hidden_states: ttnn.Tensor, attention_mask=None, position_embeddings=None) -> ttnn.Tensor:
        residual = hidden_states
        x = self.attention_norm(hidden_states)
        x = self.attention(x, position_embeddings=position_embeddings)
        x = ttnn.add(residual, x, use_legacy=None)

        residual = x
        x = self.ffn_norm(x)
        x = self.mlp(x)
        x = ttnn.add(residual, x, use_legacy=None)
        return x


__all__ = ["TtPixtralAttentionLayer"]
