"""UpsampleConformerEncoder — Stage 1 implementation (host-side torch).

Architecture (from cosyvoice2.yaml + source):
  1. embed: Linear(512,512) + LayerNorm + ESPnet rel-pos encoding
  2. pre_lookahead_layer: Conv1d(k=4) + leaky_relu + Conv1d(k=3) + residual
  3. 6× ConformerEncoderLayer: LN → rel-pos attn → residual → LN → FFN → residual
  4. up_layer: Upsample1D(stride=2) = interpolate(nearest,×2) + pad(left,4) + Conv1d(k=5,s=1)
  5. up_embed: same as embed
  6. 4× ConformerEncoderLayer
  7. after_norm: LayerNorm(512)

Non-streaming: chunk_size=0 → full causal attention mask (lower-triangular).

Stage 1: runs entirely on host (torch). The encoder is not perf-critical
(runs once per utterance). Device optimization is Stage 2.
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F


def _make_espnet_pos_emb(seq_len: int, d_model: int) -> torch.Tensor:
    """Generate ESPnet sinusoidal relative position encoding.

    Replicates EspnetRelPositionalEncoding.position_encoding(offset=0, size=seq_len).
    Returns: [1, 2*seq_len-1, d_model]
    """
    pe_positive = torch.zeros(seq_len, d_model)
    pe_negative = torch.zeros(seq_len, d_model)
    position = torch.arange(0, seq_len, dtype=torch.float32).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * -(math.log(10000.0) / d_model))
    pe_positive[:, 0::2] = torch.sin(position * div_term)
    pe_positive[:, 1::2] = torch.cos(position * div_term)
    pe_negative[:, 0::2] = torch.sin(-1 * position * div_term)
    pe_negative[:, 1::2] = torch.cos(-1 * position * div_term)

    pe_positive = torch.flip(pe_positive, [0]).unsqueeze(0)
    pe_negative = pe_negative[1:].unsqueeze(0)
    pe = torch.cat([pe_positive, pe_negative], dim=1)
    return pe


def _rel_shift(x: torch.Tensor) -> torch.Tensor:
    """ESPnet rel_shift: [B, H, T, 2T-1] → [B, H, T, T]."""
    B, H, T1, T2 = x.shape
    zero_pad = torch.zeros(B, H, T1, 1, device=x.device, dtype=x.dtype)
    x_padded = torch.cat([zero_pad, x], dim=-1)
    x_padded = x_padded.view(B, H, T2 + 1, T1)
    x = x_padded[:, :, 1:].view_as(x)[:, :, :, : T2 // 2 + 1]
    return x


class RelPosAttention(torch.nn.Module):
    """ESPnet Transformer-XL relative position attention (torch)."""

    def __init__(self, weights: Dict[str, torch.Tensor], n_heads: int, d_model: int):
        super().__init__()
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.scale = 1.0 / math.sqrt(self.d_k)

        self.linear_q = torch.nn.Linear(d_model, d_model)
        self.linear_q.weight = torch.nn.Parameter(weights["linear_q.weight"])
        self.linear_q.bias = torch.nn.Parameter(weights["linear_q.bias"])

        self.linear_k = torch.nn.Linear(d_model, d_model)
        self.linear_k.weight = torch.nn.Parameter(weights["linear_k.weight"])
        self.linear_k.bias = torch.nn.Parameter(weights["linear_k.bias"])

        self.linear_v = torch.nn.Linear(d_model, d_model)
        self.linear_v.weight = torch.nn.Parameter(weights["linear_v.weight"])
        self.linear_v.bias = torch.nn.Parameter(weights["linear_v.bias"])

        self.linear_out = torch.nn.Linear(d_model, d_model)
        self.linear_out.weight = torch.nn.Parameter(weights["linear_out.weight"])
        self.linear_out.bias = torch.nn.Parameter(weights["linear_out.bias"])

        self.linear_pos = torch.nn.Linear(d_model, d_model, bias=False)
        self.linear_pos.weight = torch.nn.Parameter(weights["linear_pos.weight"])

        self.pos_bias_u = torch.nn.Parameter(weights["pos_bias_u"])
        self.pos_bias_v = torch.nn.Parameter(weights["pos_bias_v"])

    def forward(self, x: torch.Tensor, pos_emb: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, T, D = x.shape
        H, d_k = self.n_heads, self.d_k

        q = self.linear_q(x).view(B, T, H, d_k)
        k = self.linear_k(x).view(B, T, H, d_k).permute(0, 2, 1, 3)
        v = self.linear_v(x).view(B, T, H, d_k).permute(0, 2, 1, 3)

        T_pos = pos_emb.shape[1]
        p = self.linear_pos(pos_emb).view(B, T_pos, H, d_k).permute(0, 2, 1, 3)

        q_with_bias_u = (q + self.pos_bias_u).permute(0, 2, 1, 3)
        q_with_bias_v = (q + self.pos_bias_v).permute(0, 2, 1, 3)

        matrix_ac = torch.matmul(q_with_bias_u, k.transpose(-2, -1))
        matrix_bd = torch.matmul(q_with_bias_v, p.transpose(-2, -1))

        if matrix_ac.shape != matrix_bd.shape:
            matrix_bd = _rel_shift(matrix_bd)

        scores = (matrix_ac + matrix_bd) * self.scale

        if mask is not None:
            scores = scores + mask

        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)
        out = out.permute(0, 2, 1, 3).contiguous().view(B, T, D)
        return self.linear_out(out)


class ConformerBlock(torch.nn.Module):
    """LN → rel-pos attn → residual → LN → FFN(silu) → residual."""

    def __init__(self, weights: Dict[str, torch.Tensor], prefix: str, n_heads: int, d_model: int, ffn_dim: int):
        super().__init__()
        self.d_model = d_model

        self.norm_mha_w = weights[f"{prefix}.norm_mha.weight"]
        self.norm_mha_b = weights[f"{prefix}.norm_mha.bias"]
        self.norm_ff_w = weights[f"{prefix}.norm_ff.weight"]
        self.norm_ff_b = weights[f"{prefix}.norm_ff.bias"]

        attn_weights = {
            "linear_q.weight": weights[f"{prefix}.self_attn.linear_q.weight"],
            "linear_q.bias": weights[f"{prefix}.self_attn.linear_q.bias"],
            "linear_k.weight": weights[f"{prefix}.self_attn.linear_k.weight"],
            "linear_k.bias": weights[f"{prefix}.self_attn.linear_k.bias"],
            "linear_v.weight": weights[f"{prefix}.self_attn.linear_v.weight"],
            "linear_v.bias": weights[f"{prefix}.self_attn.linear_v.bias"],
            "linear_out.weight": weights[f"{prefix}.self_attn.linear_out.weight"],
            "linear_out.bias": weights[f"{prefix}.self_attn.linear_out.bias"],
            "linear_pos.weight": weights[f"{prefix}.self_attn.linear_pos.weight"],
            "pos_bias_u": weights[f"{prefix}.self_attn.pos_bias_u"],
            "pos_bias_v": weights[f"{prefix}.self_attn.pos_bias_v"],
        }
        self.attn = RelPosAttention(attn_weights, n_heads, d_model)

        self.ff_w1 = weights[f"{prefix}.feed_forward.w_1.weight"]
        self.ff_b1 = weights[f"{prefix}.feed_forward.w_1.bias"]
        self.ff_w2 = weights[f"{prefix}.feed_forward.w_2.weight"]
        self.ff_b2 = weights[f"{prefix}.feed_forward.w_2.bias"]

    def forward(self, x: torch.Tensor, pos_emb: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        residual = x
        x_norm = F.layer_norm(x, (self.d_model,), self.norm_mha_w, self.norm_mha_b)
        x = residual + self.attn(x_norm, pos_emb, mask)

        residual = x
        x_norm = F.layer_norm(x, (self.d_model,), self.norm_ff_w, self.norm_ff_b)
        h = F.linear(x_norm, self.ff_w1, self.ff_b1)
        h = F.silu(h)
        h = F.linear(h, self.ff_w2, self.ff_b2)
        x = residual + h
        return x


class UpsampleConformerEncoder(torch.nn.Module):
    """Full UpsampleConformerEncoder (host-side torch, Stage 1)."""

    def __init__(
        self,
        encoder_weights: Dict[str, torch.Tensor],
        n_heads: int = 8,
        d_model: int = 512,
        ffn_dim: int = 2048,
        n_blocks: int = 6,
        n_up_blocks: int = 4,
    ):
        super().__init__()
        self.d_model = d_model

        self.embed_linear_w = encoder_weights["encoder.embed.out.0.weight"]
        self.embed_linear_b = encoder_weights["encoder.embed.out.0.bias"]
        self.embed_ln_w = encoder_weights["encoder.embed.out.1.weight"]
        self.embed_ln_b = encoder_weights["encoder.embed.out.1.bias"]

        self.up_embed_linear_w = encoder_weights["encoder.up_embed.out.0.weight"]
        self.up_embed_linear_b = encoder_weights["encoder.up_embed.out.0.bias"]
        self.up_embed_ln_w = encoder_weights["encoder.up_embed.out.1.weight"]
        self.up_embed_ln_b = encoder_weights["encoder.up_embed.out.1.bias"]

        self.pre_la_conv1_w = encoder_weights["encoder.pre_lookahead_layer.conv1.weight"]
        self.pre_la_conv1_b = encoder_weights["encoder.pre_lookahead_layer.conv1.bias"]
        self.pre_la_conv2_w = encoder_weights["encoder.pre_lookahead_layer.conv2.weight"]
        self.pre_la_conv2_b = encoder_weights["encoder.pre_lookahead_layer.conv2.bias"]

        self.up_conv_w = encoder_weights["encoder.up_layer.conv.weight"]
        self.up_conv_b = encoder_weights["encoder.up_layer.conv.bias"]

        self.after_norm_w = encoder_weights["encoder.after_norm.weight"]
        self.after_norm_b = encoder_weights["encoder.after_norm.bias"]

        self.blocks = torch.nn.ModuleList(
            [
                ConformerBlock(encoder_weights, f"encoder.encoders.{i}", n_heads, d_model, ffn_dim)
                for i in range(n_blocks)
            ]
        )
        self.up_blocks = torch.nn.ModuleList(
            [
                ConformerBlock(encoder_weights, f"encoder.up_encoders.{i}", n_heads, d_model, ffn_dim)
                for i in range(n_up_blocks)
            ]
        )

    def _embed(self, x: torch.Tensor, w, b, ln_w, ln_b) -> Tuple[torch.Tensor, torch.Tensor]:
        x = F.linear(x, w, b)
        x = F.layer_norm(x, (self.d_model,), ln_w, ln_b)
        x = x * math.sqrt(self.d_model)
        T = x.shape[1]
        pos_emb = _make_espnet_pos_emb(T, self.d_model)
        return x, pos_emb

    def _pre_lookahead(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        h = x.transpose(1, 2)
        h = F.pad(h, (0, 3), mode="constant", value=0.0)
        h = F.conv1d(h, self.pre_la_conv1_w, self.pre_la_conv1_b)
        h = F.leaky_relu(h)
        h = F.pad(h, (2, 0), mode="constant", value=0.0)
        h = F.conv1d(h, self.pre_la_conv2_w, self.pre_la_conv2_b)
        h = h.transpose(1, 2)
        return residual + h

    def _upsample(self, x: torch.Tensor) -> torch.Tensor:
        h = x.transpose(1, 2)
        h = F.interpolate(h, scale_factor=2.0, mode="nearest")
        h = F.pad(h, (4, 0), value=0.0)
        h = F.conv1d(h, self.up_conv_w, self.up_conv_b)
        return h.transpose(1, 2)

    @torch.no_grad()
    def forward(self, token_emb: torch.Tensor) -> torch.Tensor:
        """Full encoder forward (non-streaming).

        Args:
            token_emb: [1, T_tokens, 512] — output of input_embedding * mask

        Returns:
            [1, T_mel, 512] — encoder output (before encoder_proj)
        """
        x, pos_emb = self._embed(token_emb, self.embed_linear_w, self.embed_linear_b, self.embed_ln_w, self.embed_ln_b)
        x = self._pre_lookahead(x)

        for block in self.blocks:
            x = block(x, pos_emb, None)

        x = self._upsample(x)

        x, pos_emb = self._embed(
            x, self.up_embed_linear_w, self.up_embed_linear_b, self.up_embed_ln_w, self.up_embed_ln_b
        )

        for block in self.up_blocks:
            x = block(x, pos_emb, None)

        x = F.layer_norm(x, (self.d_model,), self.after_norm_w, self.after_norm_b)
        return x
