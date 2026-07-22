"""ESPnet Transformer-XL relative position self-attention (TTNN).

Implements `RelPositionMultiHeadedAttention` from ESPnet/CosyVoice:
  - linear_q/k/v/out projections
  - linear_pos (position projection, no bias)
  - pos_bias_u / pos_bias_v (learnable per-head biases)
  - matrix_ac = (q + pos_bias_u) @ k^T
  - matrix_bd = (q + pos_bias_v) @ p^T  (p = linear_pos(pos_emb))
  - rel_shift(matrix_bd) when shapes differ
  - scores = (matrix_ac + matrix_bd) / sqrt(d_k)
  - softmax + mask + @ v → output

Reference: cosyvoice/transformer/attention.py::RelPositionMultiHeadedAttention
"""

from __future__ import annotations

import math
from typing import Dict, Optional

import torch

import ttnn


class EspnetRelPosAttention:
    """ESPnet rel-pos multi-head attention on TTNN (single-device, batch=1)."""

    def __init__(
        self,
        mesh_device: ttnn.MeshDevice,
        weights: Dict[str, torch.Tensor],
        n_heads: int = 8,
        d_model: int = 512,
        dtype=ttnn.bfloat16,
    ):
        self.mesh_device = mesh_device
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.d_model = d_model
        self.dtype = dtype
        self.scale = 1.0 / math.sqrt(self.d_k)

        self.w_q = self._to_device(weights["linear_q.weight"].t(), bias=weights.get("linear_q.bias"))
        self.w_k = self._to_device(weights["linear_k.weight"].t(), bias=weights.get("linear_k.bias"))
        self.w_v = self._to_device(weights["linear_v.weight"].t(), bias=weights.get("linear_v.bias"))
        self.w_out = self._to_device(weights["linear_out.weight"].t(), bias=weights.get("linear_out.bias"))
        self.w_pos = self._to_device(weights["linear_pos.weight"].t())

        self.pos_bias_u = weights["pos_bias_u"]
        self.pos_bias_v = weights["pos_bias_v"]

    def _to_device(self, weight: torch.Tensor, bias: Optional[torch.Tensor] = None):
        w = ttnn.from_torch(
            weight.unsqueeze(0).unsqueeze(0),
            dtype=self.dtype,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        b = None
        if bias is not None:
            b = ttnn.from_torch(
                bias.unsqueeze(0).unsqueeze(0).unsqueeze(0),
                dtype=self.dtype,
                layout=ttnn.TILE_LAYOUT,
                device=self.mesh_device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        return (w, b)

    def _linear(self, x: ttnn.Tensor, wb) -> ttnn.Tensor:
        w, b = wb
        if b is not None:
            return ttnn.linear(x, w, bias=b)
        return ttnn.matmul(x, w)

    def forward(
        self,
        x: ttnn.Tensor,
        pos_emb: ttnn.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> ttnn.Tensor:
        """Forward pass.

        Args:
            x: [1, 1, T, d_model] device tensor (TILE_LAYOUT)
            pos_emb: [1, 1, T_pos, d_model] device tensor
            mask: [1, 1, T, T] additive mask (0 / -inf) on host, or None

        Returns:
            [1, 1, T, d_model] device tensor
        """
        B, _, T, D = x.shape

        q = self._linear(x, self.w_q)
        k = self._linear(x, self.w_k)
        v = self._linear(x, self.w_v)
        p = self._linear(pos_emb, self.w_pos)

        q_torch = ttnn.to_torch(q).float()
        k_torch = ttnn.to_torch(k).float()
        v_torch = ttnn.to_torch(v).float()
        p_torch = ttnn.to_torch(p).float()

        q_torch = q_torch.view(B, T, self.n_heads, self.d_k)
        k_torch = k_torch.view(B, T, self.n_heads, self.d_k).permute(0, 2, 1, 3)
        v_torch = v_torch.view(B, T, self.n_heads, self.d_k).permute(0, 2, 1, 3)

        T_pos = p_torch.shape[2]
        p_torch = p_torch.view(B, T_pos, self.n_heads, self.d_k).permute(0, 2, 1, 3)

        pos_bias_u = self.pos_bias_u.float().unsqueeze(0).unsqueeze(2)
        pos_bias_v = self.pos_bias_v.float().unsqueeze(0).unsqueeze(2)

        q_with_bias_u = (q_torch + pos_bias_u).permute(0, 2, 1, 3)
        q_with_bias_v = (q_torch + pos_bias_v).permute(0, 2, 1, 3)

        matrix_ac = torch.matmul(q_with_bias_u, k_torch.transpose(-2, -1))
        matrix_bd = torch.matmul(q_with_bias_v, p_torch.transpose(-2, -1))

        if matrix_ac.shape != matrix_bd.shape:
            matrix_bd = self._rel_shift(matrix_bd, T)

        scores = (matrix_ac + matrix_bd) * self.scale

        if mask is not None:
            scores = scores + mask.float()

        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, v_torch)

        out = out.permute(0, 2, 1, 3).contiguous().view(B, T, D)

        out_tt = ttnn.from_torch(
            out.unsqueeze(0),
            dtype=self.dtype,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh_device,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        result = self._linear(out_tt, self.w_out)
        return result

    @staticmethod
    def _rel_shift(x: torch.Tensor, time1: int) -> torch.Tensor:
        """ESPnet rel_shift: [B, H, T, 2T-1] → [B, H, T, T]."""
        B, H, T1, T2 = x.shape
        zero_pad = torch.zeros(B, H, T1, 1, device=x.device, dtype=x.dtype)
        x_padded = torch.cat([zero_pad, x], dim=-1)
        x_padded = x_padded.view(B, H, T2 + 1, T1)
        x = x_padded[:, :, 1:].view_as(x)[:, :, :, : T2 // 2 + 1]
        return x
