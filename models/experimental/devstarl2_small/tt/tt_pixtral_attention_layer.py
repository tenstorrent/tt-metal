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
            tt_ccl=tt_ccl,
        )

    def forward(self, hidden_states: ttnn.Tensor, attention_mask=None, position_embeddings=None) -> ttnn.Tensor:
        residual = hidden_states
        x = self.attention_norm(hidden_states)
        x = self.attention(x, position_embeddings=position_embeddings)
        ttnn.add(residual, x, output_tensor=residual, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(x)

        x = self.ffn_norm(residual)
        x = self.mlp(x)
        ttnn.add(residual, x, output_tensor=residual, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(x)
        return residual


__all__ = ["TtPixtralAttentionLayer"]
