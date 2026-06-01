# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Hybrid decoder layer: DeltaNet or GQA Attention + MLP."""

import torch
import ttnn
from models.common.lightweightmodule import LightweightModule
from models.demos.qwen36_27b.tt.deltanet import TtGatedDeltaNet
from models.demos.qwen36_27b.tt.attention import TtGatedAttention
from models.demos.qwen36_27b.tt.mlp import TtMLP


TILE = 32


class SimpleRMSNorm(LightweightModule):
    """Device-side RMSNorm using ttnn.rms_norm."""

    def __init__(self, device, dim, state_dict, key, eps=1e-6, dtype=ttnn.bfloat16):
        super().__init__()
        self.device = device
        self.eps = eps

        w = state_dict[f"{key}.weight"]
        torch_weight = (w + 1.0).unsqueeze(0).view(1, 1, dim).reshape(1, 1, dim // TILE, TILE)
        self.weight = ttnn.from_torch(torch_weight, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)

    def forward(self, x):
        return ttnn.rms_norm(x, epsilon=self.eps, weight=self.weight)


class TtHybridDecoderLayer(LightweightModule):
    def __init__(self, device, state_dict, layer_idx, config, dtype=ttnn.bfloat16):
        super().__init__()
        self.layer_idx = layer_idx
        self.layer_type = config.layer_types[layer_idx]

        if self.layer_type == "linear_attention":
            self.token_mixer = TtGatedDeltaNet(device, state_dict, layer_idx, config, dtype=dtype)
        else:
            self.token_mixer = TtGatedAttention(device, state_dict, layer_idx, config, dtype=dtype)

        self.mlp = TtMLP(device, state_dict, layer_idx, config, dtype=dtype)

        prefix = f"model.layers.{layer_idx}"
        self.input_layernorm = SimpleRMSNorm(
            device, config.hidden_size, state_dict, f"{prefix}.input_layernorm",
            eps=config.rms_norm_eps, dtype=dtype,
        )
        self.post_attention_layernorm = SimpleRMSNorm(
            device, config.hidden_size, state_dict, f"{prefix}.post_attention_layernorm",
            eps=config.rms_norm_eps, dtype=dtype,
        )

    def forward(self, hidden_states, deltanet_state=None, cos=None, sin=None, kv_cache=None, mode="decode"):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        new_kv_cache = None
        if self.layer_type == "linear_attention":
            hidden_states = self.token_mixer(hidden_states, deltanet_state, mode=mode)
        else:
            hidden_states, new_kv_cache = self.token_mixer(hidden_states, cos, sin, kv_cache, mode=mode)

        hidden_states = ttnn.add(residual, hidden_states)

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = ttnn.add(residual, hidden_states)

        return hidden_states, new_kv_cache
