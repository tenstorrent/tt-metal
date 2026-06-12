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

import numpy as np
import torch
import ttnn

from models.experimental.vibevoice.tt.vibevoice_config import DecoderConfig


_HIFI4 = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi4,
    math_approx_mode=False,
    fp32_dest_acc_en=True,
    packer_l1_acc=False,
)

# Without program_config, SDPA decode uses the full device grid; on Blackhole that
# can exceed the 64-core/head tree-reduction cap (MAX_TREE_REDUCTION_ROUNDS=6).
_SDPA_DECODE_CFG = ttnn.SDPAProgramConfig(
    compute_with_storage_grid_size=ttnn.CoreCoord(8, 8),
    q_chunk_size=0,
    k_chunk_size=0,
    exp_approx_mode=False,
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
    tok_embeddings: ttnn.Tensor  # [1, 1, hidden, vocab] TILE — kept for compatibility
    tok_embeddings_embed: ttnn.Tensor  # [vocab, hidden] ROW_MAJOR — for ttnn.embedding
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
    # ROW_MAJOR [vocab, hidden] for ttnn.embedding lookup
    tok_emb_embed = ttnn.as_tensor(
        tok_emb_torch,
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

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
        tok_embeddings_embed=tok_emb_embed,
        norm_w=norm_tt,
        lm_head_w=lm_head_tt,
        layers=layers,
        config=config,
    )


# ──────────────────────────────────────────────────────────────
# RoPE helpers (host precomputation, device application)
# ──────────────────────────────────────────────────────────────


def _build_rope_cache(seq_len: int, head_dim: int, rope_theta: float = 1_000_000.0):
    """Build cos/sin RoPE tables using numpy. Returns numpy arrays [S, head_dim]."""
    half = head_dim // 2
    inv_freq = (1.0 / (rope_theta ** (np.arange(0, half, dtype=np.float32) * 2.0 / head_dim))).astype(np.float32)
    positions = np.arange(seq_len, dtype=np.float32)
    freqs = np.outer(positions, inv_freq)  # [S, half]
    emb = np.concatenate([freqs, freqs], axis=-1).astype(np.float32)  # [S, head_dim]
    return np.cos(emb).astype(np.float32), np.sin(emb).astype(np.float32)


def _build_rope_cache_tt(
    seq_len: int,
    head_dim: int,
    device,
    rope_theta: float = 1_000_000.0,
) -> Tuple[ttnn.Tensor, ttnn.Tensor]:
    """Build RoPE cos/sin on device. Returns [1, 1, seq_len, head_dim] TILE."""
    cos, sin = _build_rope_cache(seq_len, head_dim, rope_theta)  # numpy [S, hd]
    cos_4d = cos[np.newaxis, np.newaxis, :, :]  # [1, 1, S, head_dim]
    sin_4d = sin[np.newaxis, np.newaxis, :, :]
    cos_tt = ttnn.as_tensor(
        cos_4d, device=device, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    sin_tt = ttnn.as_tensor(
        sin_4d, device=device, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    return cos_tt, sin_tt


def _rotate_half_ttnn(x: ttnn.Tensor) -> ttnn.Tensor:
    """Rotate half: [B, n, S, hd] → [-x2, x1] where x = [x1 | x2], hd split in half."""
    sh = x.shape
    B, n, S, hd = sh[0], sh[1], sh[2], sh[3]
    half = hd // 2
    x1 = ttnn.slice(x, [0, 0, 0, 0], [B, n, S, half], memory_config=ttnn.DRAM_MEMORY_CONFIG)
    x2 = ttnn.slice(x, [0, 0, 0, half], [B, n, S, hd], memory_config=ttnn.DRAM_MEMORY_CONFIG)
    return ttnn.concat(
        [ttnn.neg(x2, memory_config=ttnn.DRAM_MEMORY_CONFIG), x1], dim=3, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )


def _apply_rope_ttnn(x: ttnn.Tensor, cos: ttnn.Tensor, sin: ttnn.Tensor) -> ttnn.Tensor:
    """Apply RoPE in float32 (matches reference fp32 RoPE numerics)."""
    x_f32 = ttnn.typecast(x, ttnn.float32)
    rotated = ttnn.add(
        ttnn.mul(x_f32, cos, memory_config=ttnn.DRAM_MEMORY_CONFIG),
        ttnn.mul(_rotate_half_ttnn(x_f32), sin, memory_config=ttnn.DRAM_MEMORY_CONFIG),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    return ttnn.typecast(rotated, ttnn.bfloat16)


def _reshape_tt(x: ttnn.Tensor, shape: list) -> ttnn.Tensor:
    """Reshape via ROW_MAJOR intermediary to avoid tile layout conflicts."""
    x = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    x = ttnn.reshape(x, shape)
    return ttnn.to_layout(x, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)


# ──────────────────────────────────────────────────────────────
# KV cache
# ──────────────────────────────────────────────────────────────


@dataclass
class KVCache:
    """Fixed-size KV cache for TTVibeVoiceLM.

    Each layer keeps a preallocated DRAM tensor ``[B, n_kv_heads, max_seq, head_dim]``
    (TILE, bf16).  Prefill writes its slice with ``ttnn.fill_cache`` (offset 0) and
    decode writes one token per step with ``ttnn.update_cache`` at the absolute
    position.  ``ttnn.transformer.scaled_dot_product_attention_decode`` reads the
    valid prefix bounded by ``cur_pos`` — so the tensor shape stays static (trace-
    friendly) and per-step cost is O(1) in emitted-token count.  This is what lets
    the model scale to 64k context / ~40k generated tokens without the old
    concat-grown cache (O(S) realloc/step) or the fp32 GQA materialization.
    """

    keys: List[Optional[ttnn.Tensor]]  # per-layer, [B, n_kv_heads, max_seq, head_dim]
    values: List[Optional[ttnn.Tensor]]  # per-layer
    max_seq: int = 0


def _round_up(x: int, m: int) -> int:
    return ((x + m - 1) // m) * m


def create_kv_cache(n_layers: int) -> KVCache:
    """Empty cache (tensors allocated lazily by TTVibeVoiceLM.alloc_kv_cache)."""
    return KVCache(keys=[None] * n_layers, values=[None] * n_layers, max_seq=0)


# ──────────────────────────────────────────────────────────────
# Main TT LM class
# ──────────────────────────────────────────────────────────────


class TTVibeVoiceLM:
    """TTNN Qwen2-1.5B language model for VibeVoice.

    forward() methods operate exclusively on ttnn.Tensor.
    """

    # KV-cache seq length is rounded up to this multiple so the fused SDPA-decode
    # auto k_chunk_size (must be a multiple of 32 and divide the padded length)
    # always has clean divisors.
    _KV_ALIGN = 256

    def __init__(self, weights: LMWeights, device):
        self.w = weights
        self.device = device
        self.cfg = weights.config
        self.scale = 1.0 / math.sqrt(self.cfg.head_dim)
        # Precompute full RoPE tables on device once (sliced per call via ttnn.slice)
        max_len = self.cfg.max_position_embeddings
        self._cos_tt, self._sin_tt = _build_rope_cache_tt(max_len, self.cfg.head_dim, device, self.cfg.rope_theta)
        # Causal-mask cache for the fp32 prefill path, keyed by (S, S_total).
        self._mask_cache: Dict[Tuple[int, int], ttnn.Tensor] = {}

    def alloc_kv_cache(self, max_seq: int, dtype: ttnn.DataType = ttnn.bfloat16) -> KVCache:
        """Preallocate a fixed-size KV cache sized to ``max_seq`` (rounded up).

        Shape per layer: ``[1, n_kv_heads, max_seq_aligned, head_dim]`` TILE/DRAM.
        """
        cfg = self.cfg
        max_seq_aligned = _round_up(max(max_seq, self._KV_ALIGN), self._KV_ALIGN)
        n_kv = cfg.num_key_value_heads
        head_dim = cfg.head_dim
        keys: List[ttnn.Tensor] = []
        values: List[ttnn.Tensor] = []
        for _ in range(cfg.num_hidden_layers):
            keys.append(
                ttnn.zeros(
                    [1, n_kv, max_seq_aligned, head_dim],
                    dtype=dtype,
                    layout=ttnn.TILE_LAYOUT,
                    device=self.device,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
            )
            values.append(
                ttnn.zeros(
                    [1, n_kv, max_seq_aligned, head_dim],
                    dtype=dtype,
                    layout=ttnn.TILE_LAYOUT,
                    device=self.device,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
            )
        return KVCache(keys=keys, values=values, max_seq=max_seq_aligned)

    def _embed(self, input_ids) -> ttnn.Tensor:
        """Device embedding lookup via ttnn.embedding. Returns [B, 1, S, hidden] TILE.

        input_ids: torch.Tensor, numpy array, or any array-like [B, S] of token ids.
        """
        ids_np = np.asarray(input_ids, dtype=np.int32)
        B, S = ids_np.shape
        ids_tt = ttnn.as_tensor(
            ids_np,
            device=self.device,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        # ttnn.embedding: [B, S] uint32 + [vocab, hidden] ROW_MAJOR → [B, S, hidden]
        emb = ttnn.embedding(ids_tt, self.w.tok_embeddings_embed, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        # Reshape [B, S, hidden] → [B, 1, S, hidden] and convert to TILE
        return _reshape_tt(emb, [B, 1, S, self.cfg.hidden_size])

    def _attention_layer(
        self,
        x: ttnn.Tensor,
        layer_w: LayerWeights,
        cos_sin_tt: Optional[Tuple[ttnn.Tensor, ttnn.Tensor]],
        kv_cache: Optional[KVCache],
        layer_idx: int,
        start_pos: int = 0,
    ) -> ttnn.Tensor:
        """Single Qwen2 attention block — all ops on device.

        x: [B, 1, S, hidden]
        Returns: [B, 1, S, hidden]
        """
        cfg = self.cfg
        B = x.shape[0]
        S = x.shape[2]
        head_dim = cfg.head_dim
        n_heads = cfg.num_attention_heads
        n_kv = cfg.num_key_value_heads

        # QKV projections [B, 1, S, n*hd]
        q = ttnn.linear(x, layer_w.wq, compute_kernel_config=_HIFI4, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        k = ttnn.linear(x, layer_w.wk, compute_kernel_config=_HIFI4, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        v = ttnn.linear(x, layer_w.wv, compute_kernel_config=_HIFI4, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        if layer_w.q_bias is not None:
            q = ttnn.add(q, layer_w.q_bias, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        if layer_w.k_bias is not None:
            k = ttnn.add(k, layer_w.k_bias, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        if layer_w.v_bias is not None:
            v = ttnn.add(v, layer_w.v_bias, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        # Reshape [B, 1, S, n*hd] → [B, S, n, hd] then permute → [B, n, S, hd]
        # Matches PyTorch: view(B, S, n, hd).transpose(1, 2)
        q = ttnn.permute(_reshape_tt(q, [B, S, n_heads, head_dim]), (0, 2, 1, 3))  # [B, n_heads, S, hd]
        k = ttnn.permute(_reshape_tt(k, [B, S, n_kv, head_dim]), (0, 2, 1, 3))  # [B, n_kv, S, hd]
        v = ttnn.permute(_reshape_tt(v, [B, S, n_kv, head_dim]), (0, 2, 1, 3))  # [B, n_kv, S, hd]

        # Apply RoPE on device (validated fp32 path); cos/sin sliced [start_pos : start_pos+S].
        if cos_sin_tt is not None:
            cos_tt, sin_tt = cos_sin_tt
            c = ttnn.slice(
                cos_tt, [0, 0, start_pos, 0], [1, 1, start_pos + S, head_dim], memory_config=ttnn.DRAM_MEMORY_CONFIG
            )
            s = ttnn.slice(
                sin_tt, [0, 0, start_pos, 0], [1, 1, start_pos + S, head_dim], memory_config=ttnn.DRAM_MEMORY_CONFIG
            )
            q = _apply_rope_ttnn(q, c, s)
            k = _apply_rope_ttnn(k, c, s)

        if S > 1:
            # ── Prefill: fp32 manual attention reading the fixed-cache prefix ──
            # The chunk's K/V is written into the preallocated cache at its (tile-
            # aligned) offset, then we read the whole [0:start_pos+S] prefix and run
            # the reference-parity fp32 path (GQA materialize + fp32 matmul/softmax).
            # This keeps prefill numerics identical to the original (PCC >= 0.99);
            # bf16 flash-SDPA prefill compounds to ~0.984 over 28 layers.  Prefill is
            # one-time, so the fp32 cost is acceptable.
            if kv_cache is not None and kv_cache.keys[layer_idx] is not None:
                # Write this chunk's K/V into the fixed cache and attend over the prefix.
                ttnn.fill_cache(kv_cache.keys[layer_idx], k, 0, update_idx=start_pos)
                ttnn.fill_cache(kv_cache.values[layer_idx], v, 0, update_idx=start_pos)
                S_total = start_pos + S
                k_all = ttnn.slice(
                    kv_cache.keys[layer_idx],
                    [0, 0, 0, 0],
                    [B, n_kv, S_total, head_dim],
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
                v_all = ttnn.slice(
                    kv_cache.values[layer_idx],
                    [0, 0, 0, 0],
                    [B, n_kv, S_total, head_dim],
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
            else:
                # No allocated cache (single-shot prefill, e.g. PCC tests): attend within
                # this forward only.  start_pos is 0 in this case.
                S_total = S
                k_all, v_all = k, v

            # GQA: repeat_interleave KV heads → [B, n_heads, S_total, hd]
            repeat = n_heads // n_kv
            k_slices, v_slices = [], []
            for kv_idx in range(n_kv):
                kh = ttnn.slice(
                    k_all, [0, kv_idx, 0, 0], [B, kv_idx + 1, S_total, head_dim], memory_config=ttnn.DRAM_MEMORY_CONFIG
                )
                vh = ttnn.slice(
                    v_all, [0, kv_idx, 0, 0], [B, kv_idx + 1, S_total, head_dim], memory_config=ttnn.DRAM_MEMORY_CONFIG
                )
                for _ in range(repeat):
                    k_slices.append(kh)
                    v_slices.append(vh)
            k_rep = ttnn.concat(k_slices, dim=1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            v_rep = ttnn.concat(v_slices, dim=1, memory_config=ttnn.DRAM_MEMORY_CONFIG)

            q_f32 = ttnn.typecast(q, ttnn.float32)
            k_f32 = ttnn.typecast(k_rep, ttnn.float32)
            v_f32 = ttnn.typecast(v_rep, ttnn.float32)
            k_t = ttnn.permute(k_f32, (0, 1, 3, 2))  # [B, n_heads, hd, S_total]
            scores = ttnn.matmul(q_f32, k_t, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            scores = ttnn.mul(scores, self.scale, memory_config=ttnn.DRAM_MEMORY_CONFIG)

            cache_key = (S, S_total)
            if cache_key not in self._mask_cache:
                mask = np.triu(np.full((S, S_total), float("-inf"), dtype=np.float32), k=S_total - S + 1)
                self._mask_cache[cache_key] = ttnn.as_tensor(
                    mask[np.newaxis, np.newaxis, :, :],
                    device=self.device,
                    dtype=ttnn.float32,
                    layout=ttnn.TILE_LAYOUT,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
            scores = ttnn.add(scores, self._mask_cache[cache_key], memory_config=ttnn.DRAM_MEMORY_CONFIG)

            attn = ttnn.softmax(scores, dim=-1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            out = ttnn.matmul(attn, v_f32, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            out = ttnn.typecast(out, ttnn.bfloat16)
            out = _reshape_tt(ttnn.permute(out, (0, 2, 1, 3)), [B, 1, S, n_heads * head_dim])
        else:
            # ── Decode: write one token at start_pos, then fused flash-decode over the
            # cache prefix.  GQA handled natively (no KV-head materialization, no fp32
            # blow-up); reads only the valid prefix bounded by ``cur_pos``.  ~flat in
            # emitted-token count → scales to 64k ctx / ~40k tokens, and is trace-ready.
            #
            # Precision note: the op is bf16-only (rejects fp32).  Its attention is
            # numerically excellent (decode hidden PCC 0.9997 vs HF Qwen2) but, being
            # bf16 vs the fp32 CPU reference, it flips a few *greedy near-ties* among the
            # constrained tokens (free-running token_match ~0.977).  For this generative
            # TTS that is a different-but-valid generation, not degraded audio — validated
            # by the forced-token audio-parity gate (test_e2e_generate_pcc.py).  A grouped
            # fp32 manual decode matches tokens exactly but measured 358 ms/step (slower
            # than the old 202 ms), so it is not used.  See PERF_OPTIMIZATION_NOTES.md.
            assert kv_cache is not None and kv_cache.keys[layer_idx] is not None, "decode needs an allocated KV cache"
            ttnn.update_cache(kv_cache.keys[layer_idx], k, start_pos)  # k: [1, n_kv, 1, hd]
            ttnn.update_cache(kv_cache.values[layer_idx], v, start_pos)
            q_dec = ttnn.permute(q, (0, 2, 1, 3))  # [1, B, n_heads, hd] for sdpa_decode
            attn = ttnn.transformer.scaled_dot_product_attention_decode(
                q_dec,
                kv_cache.keys[layer_idx],
                kv_cache.values[layer_idx],
                cur_pos=[start_pos],
                scale=self.scale,
                program_config=_SDPA_DECODE_CFG,
                compute_kernel_config=_HIFI4,
            )  # [1, B, n_heads, hd]
            out = _reshape_tt(attn, [B, 1, S, n_heads * head_dim])

        # Output projection
        out = ttnn.linear(out, layer_w.wo, compute_kernel_config=_HIFI4, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        return out

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
        cos_sin_tt: Optional[Tuple[ttnn.Tensor, ttnn.Tensor]],
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
        attn_out = self._attention_layer(x_norm, lw, cos_sin_tt, kv_cache, layer_idx, start_pos)
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

        # Use precomputed full RoPE tables; _attention_layer slices [start_pos:start_pos+S]
        cos_tt, sin_tt = self._cos_tt, self._sin_tt

        x = inputs_embeds
        if x.dtype == ttnn.float32:
            x = ttnn.typecast(x, ttnn.bfloat16)
        for layer_idx in range(cfg.num_hidden_layers):
            x = self._transformer_layer(x, layer_idx, (cos_tt, sin_tt), kv_cache, start_pos)

        # Final norm
        x = ttnn.rms_norm(
            x,
            weight=self.w.norm_w,
            epsilon=cfg.rms_norm_eps,
            compute_kernel_config=_HIFI4,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        last_hidden = ttnn.typecast(x, ttnn.float32) if return_last_hidden else None

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
        return self.prefill_embeds(inputs_embeds, kv_cache=kv_cache, return_last_hidden=return_last_hidden)

    def prefill_embeds(
        self,
        inputs_embeds: ttnn.Tensor,
        kv_cache: Optional[KVCache] = None,
        chunk_size: int = 256,
        return_last_hidden: bool = False,
    ) -> Tuple[ttnn.Tensor, Optional[ttnn.Tensor]]:
        """Chunked prefill (fp32 manual attention, reference-parity precision).

        Each chunk's K/V is written into the fixed cache at its (tile-aligned, multiple
        of ``chunk_size``) offset and the chunk attends to the whole prefix read back
        from the cache — bounding the fp32 score matrix to ``[n_heads, chunk, S_total]``.
        ``chunk_size`` must be a multiple of 32 (fill_cache offset alignment).
        """
        S = inputs_embeds.shape[2]
        if S <= chunk_size:
            return self.forward(
                inputs_embeds,
                start_pos=0,
                kv_cache=kv_cache,
                return_last_hidden=return_last_hidden,
            )

        logits = None
        last_hidden = None
        hidden_dim = inputs_embeds.shape[-1]
        for start in range(0, S, chunk_size):
            end = min(start + chunk_size, S)
            chunk = ttnn.slice(
                inputs_embeds,
                [0, 0, start, 0],
                [1, 1, end, hidden_dim],
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            logits, last_hidden = self.forward(
                chunk,
                start_pos=start,
                kv_cache=kv_cache,
                return_last_hidden=return_last_hidden,
            )
        return logits, last_hidden

    def decode_step(
        self,
        input_id: torch.Tensor,
        start_pos: int,
        kv_cache: KVCache,
        return_last_hidden: bool = False,
    ):
        """Single decode step.

        Returns logits [B, 1, 1, vocab], or (logits, last_hidden) when return_last_hidden=True.
        """
        inputs_embeds = self._embed(input_id)
        logits, last_hidden = self.forward(
            inputs_embeds,
            start_pos=start_pos,
            kv_cache=kv_cache,
            return_last_hidden=return_last_hidden,
        )
        if return_last_hidden:
            return logits, last_hidden
        return logits
