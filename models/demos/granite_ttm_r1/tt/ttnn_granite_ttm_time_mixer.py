# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from models.demos.granite_ttm_r1.tt.common import TorchModuleFallback


class TtnnGraniteTTMTimeMixer:
    """Implements PatchMixerBlock with TTNN.

    Input shape:  [B, C, P, d_model]
    Output shape: [B, C, P, d_model]

    Forward pass (gated_attn=True):
        1. Save residual
        2. LayerNorm on last dim
        3. Permute (0,1,3,2) -> [B, C, d_model, P]
        4. MLP: fc1 -> GELU -> fc2
        5. GatedAttention: linear -> softmax(dim=-1) -> element-wise multiply
        6. Permute (0,1,3,2) back -> [B, C, P, d_model]
        7. Add residual
    """

    def __init__(self, *, parameters=None, config=None, torch_module=None):
        self._params = parameters
        self._config = config
        self._fallback = TorchModuleFallback(torch_module) if torch_module is not None and parameters is None else None

    def __call__(self, hidden_states, *, device=None, **kwargs):
        if self._params is not None:
            return self._forward_ttnn(hidden_states, device=device)
        if self._fallback is not None:
            return self._fallback(hidden_states, device=device, **kwargs)
        return hidden_states

    def _forward_ttnn(self, hidden_states, *, device):
        import ttnn

        residual = hidden_states

        # LayerNorm on last dim
        hidden_states = ttnn.layer_norm(
            hidden_states,
            weight=self._params.norm.norm.weight,
            bias=self._params.norm.norm.bias,
            epsilon=1e-5,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )

        # Transpose dims 2 and 3: [B, C, P, d_model] -> [B, C, d_model, P]
        hidden_states = ttnn.permute(hidden_states, (0, 1, 3, 2))

        # MLP: fc1 -> GELU -> fc2
        hidden_states = ttnn.linear(
            hidden_states,
            self._params.mlp.fc1.weight,
            bias=self._params.mlp.fc1.bias,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            dtype=ttnn.bfloat16,
        )
        hidden_states = ttnn.gelu(hidden_states, memory_config=ttnn.L1_MEMORY_CONFIG)
        hidden_states = ttnn.linear(
            hidden_states,
            self._params.mlp.fc2.weight,
            bias=self._params.mlp.fc2.bias,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            dtype=ttnn.bfloat16,
        )

        # Gated attention: linear -> softmax(dim=-1) -> element-wise multiply
        attn = ttnn.linear(
            hidden_states,
            self._params.gating_block.attn_layer.weight,
            bias=self._params.gating_block.attn_layer.bias,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            dtype=ttnn.bfloat16,
        )
        attn = ttnn.softmax(attn, dim=-1, memory_config=ttnn.L1_MEMORY_CONFIG)
        hidden_states = ttnn.mul(hidden_states, attn, memory_config=ttnn.L1_MEMORY_CONFIG)

        # Transpose back: [B, C, d_model, P] -> [B, C, P, d_model]
        hidden_states = ttnn.permute(hidden_states, (0, 1, 3, 2))

        # Residual connection
        out = ttnn.add(hidden_states, residual, memory_config=ttnn.L1_MEMORY_CONFIG)
        return out
