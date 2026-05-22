# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
VibeVoice Language Model (Qwen2-1.5B backbone) — TTNN port.

Implements a Qwen2-compatible Transformer (28 layers, hidden 1536, 12 heads, 2 KV heads)
using ttnn ops directly. Designed for prefill (inputs_embeds path) and greedy decode.

Host-side:
  load_vibevoice_lm_weights() → load + remap weights
  preprocess_lm_weights()     → convert to device tensors

Device forward:
  TTVibeVoiceLM.prefill()  → [B, S, vocab] logits  (or hidden states)
  TTVibeVoiceLM.decode()   → [B, 1, vocab] logits
"""

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import ttnn

from models.experimental.vibevoice.tt.vibevoice_config import DecoderConfig


_HIFI4 = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi4,
    math_approx_mode=False,
    fp32_dest_acc_en=True,
    packer_l1_acc=False,
)


# ──────────────────────────────────────────────────────────────
# Host-side weight preparation
# ──────────────────────────────────────────────────────────────


def load_vibevoice_lm_weights(model_path: str) -> Dict[str, torch.Tensor]:
    """Load and remap VibeVoice LM weights to tt-friendly naming (host only)."""
    from models.experimental.vibevoice.tt.load_weights import (
        load_vibevoice_state_dict,
        split_submodule_weights,
        remap_lm_keys_to_tt_transformers,
    )

    state_dict = load_vibevoice_state_dict(model_path)
    sub = split_submodule_weights(state_dict)
    return remap_lm_keys_to_tt_transformers(sub["lm"])


# ──────────────────────────────────────────────────────────────
# Weight containers
# ──────────────────────────────────────────────────────────────


@dataclass
class LayerWeights:
    wq: ttnn.Tensor  # [n_heads*head_dim, hidden]
    wk: ttnn.Tensor  # [n_kv_heads*head_dim, hidden]
    wv: ttnn.Tensor  # [n_kv_heads*head_dim, hidden]
    wo: ttnn.Tensor  # [hidden, n_heads*head_dim]
    w1: ttnn.Tensor  # [ffn_dim, hidden]  gate
    w2: ttnn.Tensor  # [hidden, ffn_dim]  down
    w3: ttnn.Tensor  # [ffn_dim, hidden]  up
    attn_norm_w: ttnn.Tensor  # [1,1,1,hidden]
    ffn_norm_w: ttnn.Tensor  # [1,1,1,hidden]
    # Qwen2 qkv biases
    q_bias: Optional[ttnn.Tensor] = None
    k_bias: Optional[ttnn.Tensor] = None
    v_bias: Optional[ttnn.Tensor] = None


@dataclass
class LMWeights:
    tok_embeddings: ttnn.Tensor  # [vocab, hidden] — on host for embed lookup, then to device
    tok_embeddings_torch: torch.Tensor  # host copy for embedding lookup
    norm_w: ttnn.Tensor  # [1,1,1,hidden]
    lm_head_w: ttnn.Tensor  # [hidden, vocab] transposed for linear
    layers: List[LayerWeights]
    config: DecoderConfig


def _tile(t: torch.Tensor, device, dtype=ttnn.bfloat16) -> ttnn.Tensor:
    """Convert 2D [out, in] weight to TTNN TILE layout, transposed for x@W semantics."""
    # ttnn.linear computes x @ W (no implicit transpose), so store as [in, out]
    t_4d = t.to(torch.bfloat16).t().unsqueeze(0).unsqueeze(0)
    return ttnn.as_tensor(
        t_4d,
        device=device,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def _norm_weight(t: torch.Tensor, device) -> ttnn.Tensor:
    """Convert 1D norm weight to [1,1,dim//32,32] ROW_MAJOR for ttnn.rms_norm."""
    dim = t.shape[0]
    return ttnn.as_tensor(
        t.to(torch.bfloat16).view(1, 1, dim // 32, 32),
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def preprocess_lm_weights(
    state_dict: Dict[str, torch.Tensor],
    device,
    config: DecoderConfig,
) -> LMWeights:
    """Convert remapped LM state dict to device tensors.

    state_dict is keyed using tt_transformers names:
      tok_embeddings.weight, norm.weight
      layers.N.attention.wq.weight, .wk.weight, .wv.weight, .wo.weight
      layers.N.attention.wq.bias, .wk.bias, .wv.bias  (optional in Qwen2)
      layers.N.feed_forward.w1.weight, .w2.weight, .w3.weight
      layers.N.attention_norm.weight, .ffn_norm.weight
    """
    tok_emb_torch = state_dict["tok_embeddings.weight"].to(torch.bfloat16)  # [vocab, hidden]
    tok_emb_tt = _tile(tok_emb_torch, device)

    norm_tt = _norm_weight(state_dict["norm.weight"], device)

    # lm_head — Qwen2 uses tied weights (same as tok_embeddings) but may have separate key
    if "lm_head.weight" in state_dict:
        lm_head_w = state_dict["lm_head.weight"].to(torch.bfloat16)
    else:
        lm_head_w = tok_emb_torch  # tied weights
    lm_head_tt = _tile(lm_head_w, device)

    layers: List[LayerWeights] = []
    for i in range(config.num_hidden_layers):
        prefix = f"layers.{i}"

        def _w(key: str) -> ttnn.Tensor:
            return _tile(state_dict[f"{prefix}.{key}.weight"], device)

        def _b(key: str) -> Optional[ttnn.Tensor]:
            bias_key = f"{prefix}.{key}.bias"
            if bias_key in state_dict:
                b = state_dict[bias_key].to(torch.bfloat16)
                return ttnn.as_tensor(
                    b.view(1, 1, 1, -1),
                    device=device,
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
            return None

        lw = LayerWeights(
            wq=_w("attention.wq"),
            wk=_w("attention.wk"),
            wv=_w("attention.wv"),
            wo=_w("attention.wo"),
            w1=_w("feed_forward.w1"),
            w2=_w("feed_forward.w2"),
            w3=_w("feed_forward.w3"),
            attn_norm_w=_norm_weight(state_dict[f"{prefix}.attention_norm.weight"], device),
            ffn_norm_w=_norm_weight(state_dict[f"{prefix}.ffn_norm.weight"], device),
            q_bias=_b("attention.wq"),
            k_bias=_b("attention.wk"),
            v_bias=_b("attention.wv"),
        )
        layers.append(lw)

    return LMWeights(
        tok_embeddings=tok_emb_tt,
        tok_embeddings_torch=tok_emb_torch,
        norm_w=norm_tt,
        lm_head_w=lm_head_tt,
        layers=layers,
        config=config,
    )


# ──────────────────────────────────────────────────────────────
# RoPE helpers (host precomputation, device application)
# ──────────────────────────────────────────────────────────────


def _build_rope_cache(
    seq_len: int,
    head_dim: int,
    rope_theta: float = 1_000_000.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build cos/sin cache for RoPE on host."""
    half = head_dim // 2
    inv_freq = 1.0 / (rope_theta ** (torch.arange(0, half, dtype=torch.float32) * 2 / head_dim))
    positions = torch.arange(seq_len, dtype=torch.float32)
    freqs = torch.outer(positions, inv_freq)  # [S, head_dim//2]
    emb = torch.cat([freqs, freqs], dim=-1)  # [S, head_dim]
    cos = emb.cos().to(torch.bfloat16)
    sin = emb.sin().to(torch.bfloat16)
    return cos, sin  # [S, head_dim]


def _apply_rope_torch(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply RoPE to [B, n_heads, S, head_dim] tensor (host, for weight precomputation)."""

    def rotate_half(t):
        half = t.shape[-1] // 2
        return torch.cat([-t[..., half:], t[..., :half]], dim=-1)

    cos = cos.unsqueeze(0).unsqueeze(0)  # [1, 1, S, head_dim]
    sin = sin.unsqueeze(0).unsqueeze(0)
    return x * cos + rotate_half(x) * sin


# ──────────────────────────────────────────────────────────────
# KV cache
# ──────────────────────────────────────────────────────────────


@dataclass
class KVCache:
    """Simple KV cache for TTVibeVoiceLM."""

    keys: List[Optional[ttnn.Tensor]]  # per-layer, [B, n_kv_heads, S_past, head_dim]
    values: List[Optional[ttnn.Tensor]]  # per-layer
    seq_len: int = 0


def create_kv_cache(n_layers: int) -> KVCache:
    return KVCache(keys=[None] * n_layers, values=[None] * n_layers, seq_len=0)


# ──────────────────────────────────────────────────────────────
# Main TT LM class
# ──────────────────────────────────────────────────────────────


class TTVibeVoiceLM:
    """TTNN Qwen2-1.5B language model for VibeVoice.

    forward() methods operate exclusively on ttnn.Tensor.
    """

    def __init__(self, weights: LMWeights, device):
        self.w = weights
        self.device = device
        self.cfg = weights.config

    def _embed(self, input_ids: torch.Tensor) -> ttnn.Tensor:
        """Lookup embeddings on host → move to device. Returns [B, S, hidden]."""
        # input_ids: [B, S] long on CPU
        emb = self.w.tok_embeddings_torch[input_ids.view(-1)]  # [B*S, hidden]
        B, S = input_ids.shape
        emb = emb.view(B, S, -1)  # [B, S, hidden]
        return ttnn.as_tensor(
            emb.unsqueeze(1),  # [B, 1, S, hidden] for ttnn
            device=self.device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def _attention_layer(
        self,
        x: ttnn.Tensor,
        layer_w: LayerWeights,
        cos_sin: Optional[Tuple[torch.Tensor, torch.Tensor]],
        kv_cache: Optional[KVCache],
        layer_idx: int,
        start_pos: int = 0,
    ) -> ttnn.Tensor:
        """Single Qwen2 attention block (no torch on device).

        x: [B, 1, S, hidden]
        Returns: [B, 1, S, hidden]
        """
        cfg = self.cfg
        B = x.shape[0]
        S = x.shape[2]
        head_dim = cfg.head_dim
        n_heads = cfg.num_attention_heads
        n_kv = cfg.num_key_value_heads

        # QKV projections
        q = ttnn.linear(x, layer_w.wq, compute_kernel_config=_HIFI4, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        k = ttnn.linear(x, layer_w.wk, compute_kernel_config=_HIFI4, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        v = ttnn.linear(x, layer_w.wv, compute_kernel_config=_HIFI4, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        # Add QKV biases if present (Qwen2 has them)
        if layer_w.q_bias is not None:
            q = ttnn.add(q, layer_w.q_bias, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        if layer_w.k_bias is not None:
            k = ttnn.add(k, layer_w.k_bias, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        if layer_w.v_bias is not None:
            v = ttnn.add(v, layer_w.v_bias, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        # Reshape to [B, n_heads, S, head_dim] — need to go through host for reshape
        # For correctness: bring to host, reshape, apply RoPE, return to device
        q_torch = ttnn.to_torch(q).to(torch.float32).view(B, S, n_heads, head_dim).transpose(1, 2)  # [B,nh,S,hd]
        k_torch = ttnn.to_torch(k).to(torch.float32).view(B, S, n_kv, head_dim).transpose(1, 2)  # [B,nkv,S,hd]
        v_torch = ttnn.to_torch(v).to(torch.float32).view(B, S, n_kv, head_dim).transpose(1, 2)  # [B,nkv,S,hd]

        # Apply RoPE on host
        if cos_sin is not None:
            cos, sin = cos_sin
            # cos/sin: [S_total, head_dim]; slice for this sequence position
            c = cos[start_pos : start_pos + S]
            s = sin[start_pos : start_pos + S]
            q_torch = _apply_rope_torch(q_torch, c, s)
            k_torch = _apply_rope_torch(k_torch, c, s)

        # Update KV cache if provided
        if kv_cache is not None:
            if kv_cache.keys[layer_idx] is None:
                kv_cache.keys[layer_idx] = k_torch
                kv_cache.values[layer_idx] = v_torch
            else:
                kv_cache.keys[layer_idx] = torch.cat([kv_cache.keys[layer_idx], k_torch], dim=2)
                kv_cache.values[layer_idx] = torch.cat([kv_cache.values[layer_idx], v_torch], dim=2)
            k_torch = kv_cache.keys[layer_idx]
            v_torch = kv_cache.values[layer_idx]

        # GQA: repeat KV heads to match Q heads
        repeat = n_heads // n_kv
        k_torch = k_torch.repeat_interleave(repeat, dim=1)  # [B, n_heads, S_total, hd]
        v_torch = v_torch.repeat_interleave(repeat, dim=1)

        # Scaled dot-product attention
        scale = 1.0 / math.sqrt(head_dim)
        scores = torch.matmul(q_torch, k_torch.transpose(-2, -1)) * scale  # [B,nh,S,S_total]
        # Causal mask for prefill
        S_total = k_torch.shape[2]
        if S > 1:
            mask = torch.triu(torch.full((S, S_total), float("-inf")), diagonal=S_total - S + 1)
            scores = scores + mask
        attn = torch.softmax(scores.float(), dim=-1)
        out = torch.matmul(attn, v_torch)  # [B, nh, S, hd]
        out = out.transpose(1, 2).contiguous().view(B, S, n_heads * head_dim)  # [B, S, hidden]

        # Back to device
        out_tt = ttnn.as_tensor(
            out.unsqueeze(1),  # [B, 1, S, hidden]
            device=self.device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        # Output projection
        out_tt = ttnn.linear(out_tt, layer_w.wo, compute_kernel_config=_HIFI4, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        return out_tt

    def _ffn_layer(self, x: ttnn.Tensor, layer_w: LayerWeights) -> ttnn.Tensor:
        """SwiGLU FFN: gate_proj(x) * silu(gate_proj(x)) → down_proj."""
        gate = ttnn.linear(x, layer_w.w1, compute_kernel_config=_HIFI4, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        up = ttnn.linear(x, layer_w.w3, compute_kernel_config=_HIFI4, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        gate = ttnn.silu(gate, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        hidden = ttnn.mul(gate, up, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        out = ttnn.linear(hidden, layer_w.w2, compute_kernel_config=_HIFI4, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        return out

    def _transformer_layer(
        self,
        x: ttnn.Tensor,
        layer_idx: int,
        cos_sin: Optional[Tuple[torch.Tensor, torch.Tensor]],
        kv_cache: Optional[KVCache],
        start_pos: int = 0,
    ) -> ttnn.Tensor:
        """Full transformer layer with pre-norm residuals."""
        lw = self.w.layers[layer_idx]

        # Pre-norm + attention
        x_norm = ttnn.rms_norm(
            x,
            weight=lw.attn_norm_w,
            epsilon=self.cfg.rms_norm_eps,
            compute_kernel_config=_HIFI4,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        attn_out = self._attention_layer(x_norm, lw, cos_sin, kv_cache, layer_idx, start_pos)
        x = ttnn.add(x, attn_out, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        # Pre-norm + FFN
        x_norm = ttnn.rms_norm(
            x,
            weight=lw.ffn_norm_w,
            epsilon=self.cfg.rms_norm_eps,
            compute_kernel_config=_HIFI4,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ffn_out = self._ffn_layer(x_norm, lw)
        x = ttnn.add(x, ffn_out, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        return x

    def forward(
        self,
        inputs_embeds: ttnn.Tensor,
        start_pos: int = 0,
        kv_cache: Optional[KVCache] = None,
        return_last_hidden: bool = False,
    ) -> Tuple[ttnn.Tensor, Optional[ttnn.Tensor]]:
        """Run transformer forward pass.

        Args:
            inputs_embeds: [B, 1, S, hidden] bfloat16 TILE on device
            start_pos: position offset for RoPE (for decode mode)
            kv_cache: optional KVCache for decode
            return_last_hidden: if True, return (last_hidden, logits) else (logits, None)

        Returns:
            (logits [B, 1, S, vocab], last_hidden or None)
        """
        B = inputs_embeds.shape[0]
        S = inputs_embeds.shape[2]
        cfg = self.cfg

        # Build RoPE cache on host
        cos, sin = _build_rope_cache(start_pos + S, cfg.head_dim, cfg.rope_theta)

        x = inputs_embeds
        for layer_idx in range(cfg.num_hidden_layers):
            x = self._transformer_layer(x, layer_idx, (cos, sin), kv_cache, start_pos)

        # Final norm
        x = ttnn.rms_norm(
            x,
            weight=self.w.norm_w,
            epsilon=cfg.rms_norm_eps,
            compute_kernel_config=_HIFI4,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        last_hidden = x if return_last_hidden else None

        # LM head projection → logits
        logits = ttnn.linear(
            x,
            self.w.lm_head_w,
            compute_kernel_config=_HIFI4,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        return logits, last_hidden

    def prefill(
        self,
        input_ids: torch.Tensor,
        kv_cache: Optional[KVCache] = None,
        return_last_hidden: bool = False,
    ) -> Tuple[ttnn.Tensor, Optional[ttnn.Tensor]]:
        """Prefill: embed input_ids and run forward pass."""
        inputs_embeds = self._embed(input_ids)
        return self.forward(inputs_embeds, start_pos=0, kv_cache=kv_cache, return_last_hidden=return_last_hidden)

    def decode_step(
        self,
        input_id: torch.Tensor,
        start_pos: int,
        kv_cache: KVCache,
    ) -> ttnn.Tensor:
        """Single decode step returning logits [B, 1, 1, vocab]."""
        inputs_embeds = self._embed(input_id)
        logits, _ = self.forward(inputs_embeds, start_pos=start_pos, kv_cache=kv_cache)
        return logits
