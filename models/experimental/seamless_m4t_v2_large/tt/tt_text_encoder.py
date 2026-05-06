# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""TTNN [`SeamlessM4Tv2Encoder`] (prefill / inference)."""

from __future__ import annotations

import math
from typing import Optional

import ttnn

from models.common.utility_functions import nearest_32


def _core_grid(device: ttnn.Device) -> ttnn.CoreGrid:
    grid = device.compute_with_storage_grid_size()
    return ttnn.CoreGrid(y=grid.y, x=grid.x)


class TTSeamlessM4Tv2Encoder:
    """
    Device port of Hugging Face ``SeamlessM4Tv2Encoder``.

    ``forward`` takes tensors already placed on the device. Use
    ``create_text_encoder_parameters`` to build ``parameters`` from the PyTorch encoder.
    """

    def __init__(
        self,
        device: ttnn.Device,
        parameters,
        *,
        layer_norm_eps: float,
        num_hidden_layers: int,
        num_attention_heads: int,
        hidden_size: int,
    ):
        self.device = device
        self.parameters = parameters
        self.layer_norm_eps = layer_norm_eps
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_size = hidden_size
        self._sdpa_compute_cfg = ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )
        self._linear_ln_compute_cfg = ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )

    def _sdpa_program_config(self, seq_q: int, seq_k: int) -> ttnn.SDPAProgramConfig:
        q_chunk = max(64, min(256, nearest_32(seq_q)))
        k_chunk = max(64, min(256, nearest_32(seq_k)))
        return ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=self.device.compute_with_storage_grid_size(),
            q_chunk_size=q_chunk,
            k_chunk_size=k_chunk,
            exp_approx_mode=False,
        )

    def _linear(self, x: ttnn.Tensor, weight: ttnn.Tensor, bias: ttnn.Tensor) -> ttnn.Tensor:
        return ttnn.linear(
            x,
            weight,
            bias=bias,
            core_grid=_core_grid(self.device),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self._linear_ln_compute_cfg,
        )

    def _layer_norm(self, x: ttnn.Tensor, *, weight: ttnn.Tensor, bias: ttnn.Tensor) -> ttnn.Tensor:
        return ttnn.layer_norm(
            x,
            weight=weight,
            bias=bias,
            epsilon=self.layer_norm_eps,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    @staticmethod
    def _heads(x: ttnn.Tensor, batch: int, seq: int, num_heads: int, head_dim: int) -> ttnn.Tensor:
        x = ttnn.reshape(x, (batch, seq, num_heads, head_dim))
        return ttnn.permute(x, (0, 2, 1, 3))

    @staticmethod
    def _merge_heads(
        x: ttnn.Tensor, batch: int, seq: int, num_heads: int, head_dim: int, hidden_size: int
    ) -> ttnn.Tensor:
        x = ttnn.permute(x, (0, 2, 1, 3))
        return ttnn.reshape(x, (batch, seq, hidden_size))

    def _attention(
        self,
        hidden_states: ttnn.Tensor,
        attn_module,
        attn_mask: Optional[ttnn.Tensor],
        *,
        batch: int,
        seq_q: int,
        seq_k: int,
        num_heads: int,
        head_dim: int,
        hidden_size: int,
        sdpa_cfg: ttnn.SDPAProgramConfig,
    ) -> ttnn.Tensor:
        q = self._linear(hidden_states, attn_module.q_proj.weight, attn_module.q_proj.bias)
        k = self._linear(hidden_states, attn_module.k_proj.weight, attn_module.k_proj.bias)
        v = self._linear(hidden_states, attn_module.v_proj.weight, attn_module.v_proj.bias)

        qh = self._heads(q, batch, seq_q, num_heads, head_dim)
        kh = self._heads(k, batch, seq_k, num_heads, head_dim)
        vh = self._heads(v, batch, seq_k, num_heads, head_dim)

        ttnn.deallocate(q)
        ttnn.deallocate(k)
        ttnn.deallocate(v)

        qh = ttnn.to_memory_config(qh, ttnn.DRAM_MEMORY_CONFIG)
        kh = ttnn.to_memory_config(kh, ttnn.DRAM_MEMORY_CONFIG)
        vh = ttnn.to_memory_config(vh, ttnn.DRAM_MEMORY_CONFIG)

        # Match HF ``SeamlessM4Tv2Attention``: scale Q by head_dim**-0.5 before QKᵀ.
        qh = ttnn.multiply(qh, 1.0 / math.sqrt(head_dim), memory_config=ttnn.DRAM_MEMORY_CONFIG)

        attn_out = ttnn.transformer.scaled_dot_product_attention(
            qh,
            kh,
            vh,
            attn_mask=attn_mask,
            is_causal=False,
            scale=1.0,
            program_config=sdpa_cfg,
            compute_kernel_config=self._sdpa_compute_cfg,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(qh)
        ttnn.deallocate(kh)
        ttnn.deallocate(vh)

        attn_out = ttnn.slice(
            attn_out,
            [0, 0, 0, 0],
            [batch, num_heads, seq_q, head_dim],
            [1, 1, 1, 1],
        )

        merged = self._merge_heads(attn_out, batch, seq_q, num_heads, head_dim, hidden_size)
        ttnn.deallocate(attn_out)
        proj = self._linear(merged, attn_module.out_proj.weight, attn_module.out_proj.bias)
        ttnn.deallocate(merged)
        return proj

    def forward(
        self,
        input_ids: ttnn.Tensor,
        position_ids: ttnn.Tensor,
        attention_mask: Optional[ttnn.Tensor] = None,
    ) -> ttnn.Tensor:
        """
        Args:
            input_ids: ``uint32`` ``[batch, seq]`` on device.
            position_ids: ``uint32`` ``[batch, seq]`` on device.
            attention_mask: optional additive mask ``[batch, 1, seq, seq]`` (bfloat16).

        Returns:
            Last hidden states ``bfloat16`` ``[batch, seq, hidden_size]`` on device.
        """
        parameters = self.parameters
        num_heads = self.num_attention_heads
        hidden_size = self.hidden_size
        head_dim = hidden_size // num_heads
        num_layers = self.num_hidden_layers

        batch = int(input_ids.shape[0])
        seq = int(input_ids.shape[1])

        tok = ttnn.embedding(
            input_ids,
            weight=parameters.embed_tokens.weight,
            layout=ttnn.TILE_LAYOUT,
        )
        pos = ttnn.embedding(
            position_ids,
            weight=parameters.embed_positions.weight,
            layout=ttnn.TILE_LAYOUT,
        )

        hidden = ttnn.add(tok, pos, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(tok)
        ttnn.deallocate(pos)

        sdpa_self = self._sdpa_program_config(seq, seq)

        for i in range(num_layers):
            layer = parameters.layers[i]

            normed = self._layer_norm(
                hidden,
                weight=layer.self_attn_layer_norm.weight,
                bias=layer.self_attn_layer_norm.bias,
            )
            attn_out = self._attention(
                normed,
                layer.self_attn,
                attention_mask,
                batch=batch,
                seq_q=seq,
                seq_k=seq,
                num_heads=num_heads,
                head_dim=head_dim,
                hidden_size=hidden_size,
                sdpa_cfg=sdpa_self,
            )
            ttnn.deallocate(normed)
            hidden = ttnn.add(hidden, attn_out, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            ttnn.deallocate(attn_out)

            normed = self._layer_norm(
                hidden,
                weight=layer.ffn_layer_norm.weight,
                bias=layer.ffn_layer_norm.bias,
            )
            ff = self._linear(normed, layer.ffn.fc1.weight, layer.ffn.fc1.bias)
            ttnn.deallocate(normed)
            ff = ttnn.relu(ff)
            ff = self._linear(ff, layer.ffn.fc2.weight, layer.ffn.fc2.bias)
            hidden = ttnn.add(hidden, ff, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            ttnn.deallocate(ff)

        hidden = self._layer_norm(
            hidden,
            weight=parameters.layer_norm.weight,
            bias=parameters.layer_norm.bias,
        )
        return hidden
