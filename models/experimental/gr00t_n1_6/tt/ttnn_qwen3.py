# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Qwen3-1.7B Language Model - TTNN Implementation for GR00T N1.6.

On-device operations: linear projections, RMSNorm, attention (Q@K^T, softmax, @V),
GQA expansion, MLP. CPU-assist only for QK-norm + RoPE (small per-head ops on
[B, heads, seq, 128] tensors) and token embedding.

Qwen3-1.7B (Eagle-Block2A backbone):
    - 28 layers (GR00T uses first 16 via select_layer)
    - 2048 hidden, 16 Q heads / 8 KV heads, head_dim=128
    - QK normalization + RoPE (theta=1M)
    - SiLU-gated MLP, RMSNorm pre-norm, no bias
"""

import math
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

import ttnn
from models.experimental.gr00t_n1_6.common.configs import Qwen3Config
from models.experimental.gr00t_n1_6.tt.ttnn_common import (
    CORE_GRID_BH,
    preprocess_linear_weight,
    preprocess_rmsnorm_params,
)


# ---------------------------------------------------------------------------
# RoPE utilities (CPU) — applied to small Q/K tensors per-layer
# ---------------------------------------------------------------------------

def precompute_freqs_cis(
    head_dim: int,
    max_seq_len: int,
    theta: float = 1_000_000.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Precompute RoPE cos/sin: [max_seq, head_dim//2] in float32."""
    half = head_dim // 2
    freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    t = torch.arange(max_seq_len, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    return freqs.cos(), freqs.sin()


def apply_rotary_emb_cpu(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
) -> torch.Tensor:
    """Apply RoPE on CPU. x: [B, heads, seq, head_dim], cos/sin: [seq, half]."""
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    c = cos.unsqueeze(0).unsqueeze(0)
    s = sin.unsqueeze(0).unsqueeze(0)
    return torch.cat([x1 * c - x2 * s, x2 * c + x1 * s], dim=-1)


def qk_norm_cpu(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """RMSNorm per-head on CPU. x: [..., head_dim], weight: [head_dim]."""
    rms = x.float().pow(2).mean(dim=-1, keepdim=True).add(eps).sqrt()
    return (x.float() / rms * weight).to(torch.bfloat16)


# ---------------------------------------------------------------------------
# Qwen3 MLP — fully on device
# ---------------------------------------------------------------------------

class Qwen3MLPTTNN:
    """SiLU-gated MLP: down_proj(silu(gate_proj(x)) * up_proj(x)). No bias."""

    def __init__(self, gate_w: torch.Tensor, up_w: torch.Tensor,
                 down_w: torch.Tensor, device: Any):
        self.gate_weight = preprocess_linear_weight(gate_w, device, dtype=ttnn.bfloat16)
        self.up_weight = preprocess_linear_weight(up_w, device, dtype=ttnn.bfloat16)
        self.down_weight = preprocess_linear_weight(down_w, device, dtype=ttnn.bfloat16)

    def __call__(self, x: ttnn.Tensor) -> ttnn.Tensor:
        gate = ttnn.linear(x, self.gate_weight, memory_config=ttnn.L1_MEMORY_CONFIG,
                           dtype=ttnn.bfloat16, core_grid=CORE_GRID_BH, activation="silu")
        up = ttnn.linear(x, self.up_weight, memory_config=ttnn.L1_MEMORY_CONFIG,
                         dtype=ttnn.bfloat16, core_grid=CORE_GRID_BH)
        hidden = ttnn.mul(gate, up, memory_config=ttnn.L1_MEMORY_CONFIG, dtype=ttnn.bfloat16)
        ttnn.deallocate(gate)
        ttnn.deallocate(up)
        out = ttnn.linear(hidden, self.down_weight, memory_config=ttnn.L1_MEMORY_CONFIG,
                          dtype=ttnn.bfloat16, core_grid=CORE_GRID_BH)
        ttnn.deallocate(hidden)
        return out


# ---------------------------------------------------------------------------
# Qwen3 Attention — projections + attention on device, QK-norm + RoPE on CPU
# ---------------------------------------------------------------------------

class Qwen3AttentionTTNN:
    """
    GQA attention for Qwen3-1.7B.

    On device: Q/K/V/O projections, scaled attention (Q@K^T, softmax, @V).
    CPU-assist: QK normalization (RMSNorm per-head) and RoPE on small tensors
    [B, heads, seq, 128]. This avoids 4D slice issues while keeping the
    expensive matmuls entirely on device.
    """

    def __init__(
        self, q_proj_w: torch.Tensor, k_proj_w: torch.Tensor,
        v_proj_w: torch.Tensor, o_proj_w: torch.Tensor,
        q_norm_w: torch.Tensor, k_norm_w: torch.Tensor,
        config: Qwen3Config, device: Any,
    ):
        self.config = config
        self.device = device
        self.num_heads = config.num_attention_heads      # 16
        self.num_kv_heads = config.num_key_value_heads   # 8
        self.head_dim = config.head_dim                  # 128
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.kv_repeat = self.num_heads // self.num_kv_heads  # 2

        self.q_weight = preprocess_linear_weight(q_proj_w, device, dtype=ttnn.bfloat16)
        self.k_weight = preprocess_linear_weight(k_proj_w, device, dtype=ttnn.bfloat16)
        self.v_weight = preprocess_linear_weight(v_proj_w, device, dtype=ttnn.bfloat16)
        self.o_weight = preprocess_linear_weight(o_proj_w, device, dtype=ttnn.bfloat16)

        # QK norm weights kept on CPU (applied per-head)
        self.q_norm_w_cpu = q_norm_w.to(torch.float32)
        self.k_norm_w_cpu = k_norm_w.to(torch.float32)

    def __call__(
        self, x: ttnn.Tensor,
        cos: torch.Tensor, sin: torch.Tensor,
        causal_mask: ttnn.Tensor,
    ) -> ttnn.Tensor:
        """
        Args:
            x:           [B, seq, hidden] on device
            cos, sin:    [seq, half] RoPE tables on CPU (pre-sliced)
            causal_mask: [1, 1, seq, seq] additive mask on device
        """
        batch, seq_len, _ = x.shape

        # --- Q/K/V projections on device ---
        q = ttnn.linear(x, self.q_weight, memory_config=ttnn.L1_MEMORY_CONFIG,
                        dtype=ttnn.bfloat16, core_grid=CORE_GRID_BH)
        k = ttnn.linear(x, self.k_weight, memory_config=ttnn.L1_MEMORY_CONFIG,
                        dtype=ttnn.bfloat16, core_grid=CORE_GRID_BH)
        v = ttnn.linear(x, self.v_weight, memory_config=ttnn.L1_MEMORY_CONFIG,
                        dtype=ttnn.bfloat16, core_grid=CORE_GRID_BH)

        # --- Bring Q/K to CPU for QK-norm + RoPE (small tensors) ---
        q_cpu = ttnn.to_torch(q).to(torch.bfloat16)
        k_cpu = ttnn.to_torch(k).to(torch.bfloat16)
        ttnn.deallocate(q)
        ttnn.deallocate(k)

        # Reshape: [B, seq, heads*hd] -> [B, heads, seq, hd]
        q_cpu = q_cpu.reshape(batch, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k_cpu = k_cpu.reshape(batch, seq_len, self.num_kv_heads, self.head_dim).permute(0, 2, 1, 3)

        # QK normalization
        q_cpu = qk_norm_cpu(q_cpu, self.q_norm_w_cpu, self.config.rms_norm_eps)
        k_cpu = qk_norm_cpu(k_cpu, self.k_norm_w_cpu, self.config.rms_norm_eps)

        # RoPE
        q_cpu = apply_rotary_emb_cpu(q_cpu, cos, sin)
        k_cpu = apply_rotary_emb_cpu(k_cpu, cos, sin)

        # GQA: expand KV heads 8 -> 16
        k_cpu = k_cpu.repeat_interleave(self.kv_repeat, dim=1)

        # Scale Q
        q_cpu = q_cpu * self.scale

        # --- Transfer Q/K back to device for attention matmuls ---
        q = ttnn.from_torch(q_cpu.contiguous(), dtype=ttnn.bfloat16,
                            layout=ttnn.TILE_LAYOUT, device=self.device,
                            memory_config=ttnn.L1_MEMORY_CONFIG)
        k = ttnn.from_torch(k_cpu.contiguous(), dtype=ttnn.bfloat16,
                            layout=ttnn.TILE_LAYOUT, device=self.device,
                            memory_config=ttnn.L1_MEMORY_CONFIG)

        # V: reshape on device (no QK-norm or RoPE needed)
        v_cpu = ttnn.to_torch(v).to(torch.bfloat16)
        ttnn.deallocate(v)
        v_cpu = v_cpu.reshape(batch, seq_len, self.num_kv_heads, self.head_dim).permute(0, 2, 1, 3)
        v_cpu = v_cpu.repeat_interleave(self.kv_repeat, dim=1)
        v = ttnn.from_torch(v_cpu.contiguous(), dtype=ttnn.bfloat16,
                            layout=ttnn.TILE_LAYOUT, device=self.device,
                            memory_config=ttnn.L1_MEMORY_CONFIG)

        # --- Attention on device: Q @ K^T + mask -> softmax -> @ V ---
        k_t = ttnn.permute(k, (0, 1, 3, 2))  # Transpose K
        ttnn.deallocate(k)

        attn = ttnn.matmul(q, k_t, memory_config=ttnn.L1_MEMORY_CONFIG,
                           dtype=ttnn.bfloat16, core_grid=CORE_GRID_BH)
        ttnn.deallocate(q)
        ttnn.deallocate(k_t)

        # Causal mask
        attn = ttnn.add(attn, causal_mask, memory_config=ttnn.L1_MEMORY_CONFIG)

        # Softmax
        attn = ttnn.softmax_in_place(attn, numeric_stable=True)

        # Context: attn @ V
        context = ttnn.matmul(attn, v, memory_config=ttnn.L1_MEMORY_CONFIG,
                              dtype=ttnn.bfloat16, core_grid=CORE_GRID_BH)
        ttnn.deallocate(attn)
        ttnn.deallocate(v)

        # Reshape: [B, heads, seq, hd] -> [B, seq, heads*hd]
        context = ttnn.permute(context, (0, 2, 1, 3))
        context = ttnn.reshape(context, (batch, seq_len, self.num_heads * self.head_dim))

        # Output projection on device
        out = ttnn.linear(context, self.o_weight, memory_config=ttnn.L1_MEMORY_CONFIG,
                          dtype=ttnn.bfloat16, core_grid=CORE_GRID_BH)
        ttnn.deallocate(context)
        return out


# ---------------------------------------------------------------------------
# Qwen3 Decoder Layer
# ---------------------------------------------------------------------------

class Qwen3DecoderLayerTTNN:
    """Single Qwen3 layer: RMSNorm -> Attention -> Residual -> RMSNorm -> MLP -> Residual."""

    def __init__(self, layer_weights: Dict[str, torch.Tensor],
                 config: Qwen3Config, device: Any):
        self._eps = config.rms_norm_eps
        self.input_ln_weight = preprocess_rmsnorm_params(
            layer_weights["input_layernorm.weight"], device)
        self.attn = Qwen3AttentionTTNN(
            q_proj_w=layer_weights["self_attn.q_proj.weight"],
            k_proj_w=layer_weights["self_attn.k_proj.weight"],
            v_proj_w=layer_weights["self_attn.v_proj.weight"],
            o_proj_w=layer_weights["self_attn.o_proj.weight"],
            q_norm_w=layer_weights["self_attn.q_norm.weight"],
            k_norm_w=layer_weights["self_attn.k_norm.weight"],
            config=config, device=device,
        )
        self.post_attn_ln_weight = preprocess_rmsnorm_params(
            layer_weights["post_attention_layernorm.weight"], device)
        self.mlp = Qwen3MLPTTNN(
            gate_w=layer_weights["mlp.gate_proj.weight"],
            up_w=layer_weights["mlp.up_proj.weight"],
            down_w=layer_weights["mlp.down_proj.weight"],
            device=device,
        )

    def __call__(self, hidden: ttnn.Tensor, cos: torch.Tensor,
                 sin: torch.Tensor, causal_mask: ttnn.Tensor) -> ttnn.Tensor:
        residual = hidden

        normed = ttnn.rms_norm(hidden, weight=self.input_ln_weight,
                               epsilon=self._eps, memory_config=ttnn.L1_MEMORY_CONFIG)
        attn_out = self.attn(normed, cos, sin, causal_mask)
        ttnn.deallocate(normed)

        hidden = ttnn.add(residual, attn_out,
                          memory_config=ttnn.L1_MEMORY_CONFIG, dtype=ttnn.bfloat16)
        ttnn.deallocate(attn_out)
        residual = hidden

        normed = ttnn.rms_norm(hidden, weight=self.post_attn_ln_weight,
                               epsilon=self._eps, memory_config=ttnn.L1_MEMORY_CONFIG)
        mlp_out = self.mlp(normed)
        ttnn.deallocate(normed)

        hidden = ttnn.add(residual, mlp_out,
                          memory_config=ttnn.L1_MEMORY_CONFIG, dtype=ttnn.bfloat16)
        ttnn.deallocate(mlp_out)
        return hidden


# ---------------------------------------------------------------------------
# Full Qwen3 Model
# ---------------------------------------------------------------------------

class Qwen3ModelTTNN:
    """
    Qwen3-1.7B for GR00T N1.6 on TTNN.

    CPU-only: token embedding (vocab=151680), vision feature splicing,
    QK-norm + RoPE per layer (small [B, heads, seq, 128] tensors).

    On-device: all linear projections, RMSNorm, attention matmuls,
    softmax, MLP, residual connections.

    Runs first select_layer (16) of 28 layers. Returns hidden states
    for the DiT action head.
    """

    def __init__(self, config: Qwen3Config, weights: Dict[str, torch.Tensor],
                 device: Any):
        self.config = config
        self.device = device
        self._eps = config.rms_norm_eps
        self._select_layer = config.select_layer

        # Token embedding (CPU)
        embed_w = weights.get("model.embed_tokens.weight")
        if embed_w is None:
            raise KeyError("Missing 'model.embed_tokens.weight'")
        self._embed_weight_cpu = embed_w.to(torch.float32)

        # RoPE tables (CPU, reused across layers)
        self._rope_cos, self._rope_sin = precompute_freqs_cis(
            config.head_dim, config.max_position_embeddings, theta=config.rope_theta)

        # Causal mask on device (precomputed, sliced per forward)
        max_seq = 1024
        mask_cpu = torch.triu(
            torch.full((max_seq, max_seq), -1e9, dtype=torch.bfloat16), diagonal=1,
        ).unsqueeze(0).unsqueeze(0)
        self._causal_mask_tt = ttnn.from_torch(
            mask_cpu, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
            device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        # Decoder layers
        num_layers = min(config.select_layer, config.num_hidden_layers)
        self.layers: List[Qwen3DecoderLayerTTNN] = []
        for i in range(num_layers):
            p = f"model.layers.{i}."
            layer_w = {
                "input_layernorm.weight":          weights[f"{p}input_layernorm.weight"],
                "self_attn.q_proj.weight":         weights[f"{p}self_attn.q_proj.weight"],
                "self_attn.k_proj.weight":         weights[f"{p}self_attn.k_proj.weight"],
                "self_attn.v_proj.weight":         weights[f"{p}self_attn.v_proj.weight"],
                "self_attn.o_proj.weight":         weights[f"{p}self_attn.o_proj.weight"],
                "self_attn.q_norm.weight":         weights[f"{p}self_attn.q_norm.weight"],
                "self_attn.k_norm.weight":         weights[f"{p}self_attn.k_norm.weight"],
                "post_attention_layernorm.weight": weights[f"{p}post_attention_layernorm.weight"],
                "mlp.gate_proj.weight":            weights[f"{p}mlp.gate_proj.weight"],
                "mlp.up_proj.weight":              weights[f"{p}mlp.up_proj.weight"],
                "mlp.down_proj.weight":            weights[f"{p}mlp.down_proj.weight"],
            }
            self.layers.append(Qwen3DecoderLayerTTNN(layer_w, config, device))

        # Final RMSNorm
        norm_w = weights.get("model.norm.weight")
        if norm_w is None:
            raise KeyError("Missing 'model.norm.weight'")
        self.final_norm_weight = preprocess_rmsnorm_params(norm_w, device)

    def __call__(
        self, input_ids: torch.Tensor,
        image_features: Optional[torch.Tensor] = None,
        image_token_positions: Optional[List[int]] = None,
    ) -> ttnn.Tensor:
        """
        Forward through first select_layer Qwen3 layers.

        Args:
            input_ids: [B, seq] token IDs (CPU).
            image_features: Optional [B, num_img, hidden] vision features.
            image_token_positions: Positions to splice image features.

        Returns:
            [B, seq, hidden] on device after final RMSNorm.
        """
        batch, seq_len = input_ids.shape

        # Embed on CPU
        hidden_cpu = F.embedding(input_ids.long(), self._embed_weight_cpu).to(torch.bfloat16)

        # Splice vision features
        if image_features is not None:
            if image_token_positions is None:
                raise ValueError("image_token_positions required with image_features")
            if isinstance(image_features, ttnn.Tensor):
                image_features = ttnn.to_torch(image_features).to(torch.bfloat16)
            else:
                image_features = image_features.to(torch.bfloat16)
            for local_idx, seq_idx in enumerate(image_token_positions):
                hidden_cpu[:, seq_idx, :] = image_features[:, local_idx, :]

        # Transfer to device (single CPU->device transfer)
        hidden_tt = ttnn.from_torch(
            hidden_cpu, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
            device=self.device, memory_config=ttnn.L1_MEMORY_CONFIG,
        )

        # Slice causal mask to seq_len (on device)
        causal_mask = ttnn.slice(
            self._causal_mask_tt, [0, 0, 0, 0], [1, 1, seq_len, seq_len],
        )

        # RoPE tables sliced to seq_len (CPU)
        cos = self._rope_cos[:seq_len]
        sin = self._rope_sin[:seq_len]

        # Run layers (projections + attention on device, QK-norm + RoPE on CPU)
        for layer in self.layers:
            hidden_tt = layer(hidden_tt, cos, sin, causal_mask)

        ttnn.deallocate(causal_mask)

        # Final RMSNorm on device
        output = ttnn.rms_norm(
            hidden_tt, weight=self.final_norm_weight,
            epsilon=self._eps, memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        if output is not hidden_tt:
            ttnn.deallocate(hidden_tt)

        return output
