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
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import ttnn

from models.common.tensor_utils import get_rot_transformation_mat
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

# Decode-only program config for the wq/wo 1536x1536 projections (single-token step, Mt=1):
# 1D mcast_in0, 8x3=24 cores, in0_block_w=4, per_core_N=2, out_subblock 1x2, width-sharded
# output -> 12.25 µs vs 25.4 µs on the auto config (2.08x).  per_core_M=1 makes it valid only
# for S==1 decode; prefill (S>1, Mt>1) keeps the auto config.
_QO_DECODE_PROGCFG = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
    compute_with_storage_grid_size=ttnn.CoreCoord(8, 3),
    in0_block_w=4,
    out_subblock_h=1,
    out_subblock_w=2,
    per_core_M=1,
    per_core_N=2,
    fuse_batch=True,
    fused_activation=None,
    mcast_in0=True,
)
# Width-sharded L1 output (the winning layout); the shard spec is derived from the
# program config.  Downstream ops that need interleaved input reshard automatically.
_QO_DECODE_OUT_MEMCFG = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1)

# B=2 variant of _QO_DECODE_PROGCFG for the CFG batch-2 fused decode (pos+neg rows folded
# into M).  Identical in0_block_w / mcast / subblock; per_core_M=2 so it is valid for M=2.
# Byte-identical per row to the per_core_M=1 B=1 config (row 0 maxabsdiff==0), i.e. the
# K-reduction order is preserved — long-form-safe.
_QO_DECODE_PROGCFG_B2 = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
    compute_with_storage_grid_size=ttnn.CoreCoord(8, 3),
    in0_block_w=4,
    out_subblock_h=1,
    out_subblock_w=2,
    per_core_M=2,
    per_core_N=2,
    fuse_batch=True,
    fused_activation=None,
    mcast_in0=True,
)

# Decode-only fused-KV projection (32 x 1536 x 512).  Auto lands this on 16 cores at ~22 µs;
# 1D mcast_in0 over 8x1=8 cores with in0_block_w=2, per_core_N=2, out_subblock 1x2 and a
# width-sharded L1 output (plus L1 in0 from the attn rms_norm) reaches ~16 µs.  in0_block_w=2
# matches auto's K-reduction (maxabsdiff==0), so the whole q/k/v path stays byte-identical.
# Bias-add then reshard to DRAM for NlpCreateHeads.  Prefill (S>1) keeps auto.
_WKV_DECODE_PROGCFG = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
    compute_with_storage_grid_size=ttnn.CoreCoord(8, 1),
    in0_block_w=2,
    out_subblock_h=1,
    out_subblock_w=2,
    per_core_M=1,
    per_core_N=2,
    fuse_batch=True,
    fused_activation=None,
    mcast_in0=True,
)
_WKV_DECODE_PROGCFG_B2 = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
    compute_with_storage_grid_size=ttnn.CoreCoord(8, 1),
    in0_block_w=2,
    out_subblock_h=1,
    out_subblock_w=2,
    per_core_M=2,
    per_core_N=2,
    fuse_batch=True,
    fused_activation=None,
    mcast_in0=True,
)
_WKV_DECODE_OUT_MEMCFG = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1)

# Decode-only FFN program configs (S==1), byte-identical to the auto config: in0_block_w=2 is the
# K-reduction block auto uses for these shapes (maxabsdiff==0), so the reduction order — hence the
# bf16 rounding — is preserved.
#
# These extend the CFG batch-2 weight-read-once pattern to the FFN.  The batch-2 LM fusion batched
# the wq/wo matmuls (per_core_M=2) but left the FFN on auto, which reads each FFN weight matrix
# once per CFG row.  per_core_M=2 folds both rows into M so each weight is read once: ~1.9x on the
# down-proj and ~1.85x each on gate/up at B=2.
#
# per_core_M makes these valid only for S==1 decode; prefill (S>1) keeps auto.
_FFN_DOWN_DECODE_PROGCFG_B1 = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
    compute_with_storage_grid_size=ttnn.CoreCoord(8, 3),
    in0_block_w=2,
    out_subblock_h=1,
    out_subblock_w=2,
    per_core_M=1,
    per_core_N=2,
    fuse_batch=True,
    fused_activation=None,
    mcast_in0=True,
)
_FFN_DOWN_DECODE_PROGCFG_B2 = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
    compute_with_storage_grid_size=ttnn.CoreCoord(8, 3),
    in0_block_w=2,
    out_subblock_h=1,
    out_subblock_w=2,
    per_core_M=2,
    per_core_N=2,
    fuse_batch=True,
    fused_activation=None,
    mcast_in0=True,
)
# gate/up (1536x8960): N=8960=280 tiles, so per_core_N=4 over an 11x8=88 grid (352>=280).  Only the
# B=2 batched case beats auto (B=1 gate/up candidates were slower than auto).
_FFN_GATEUP_DECODE_PROGCFG_B2 = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
    compute_with_storage_grid_size=ttnn.CoreCoord(11, 8),
    in0_block_w=2,
    out_subblock_h=1,
    out_subblock_w=2,
    per_core_M=2,
    per_core_N=4,
    fuse_batch=True,
    fused_activation=None,
    mcast_in0=True,
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
    wq: ttnn.Tensor  # TILE [1,1,hidden,n_heads*head_dim]
    wkv: ttnn.Tensor  # TILE [1,1,hidden,2*n_kv*head_dim] — fused wk|wv (byte-ident vs separate)
    wo: ttnn.Tensor  # TILE [1,1,n_heads*head_dim,hidden]
    w1: ttnn.Tensor  # [ffn_dim, hidden]  gate
    w2: ttnn.Tensor  # [hidden, ffn_dim]  down
    w3: ttnn.Tensor  # [ffn_dim, hidden]  up
    attn_norm_w: ttnn.Tensor  # [1,1,1,hidden]
    ffn_norm_w: ttnn.Tensor  # [1,1,1,hidden]
    # Qwen2 qkv biases
    q_bias: Optional[ttnn.Tensor] = None
    kv_bias: Optional[ttnn.Tensor] = None  # fused k_bias|v_bias on the out dim


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
    return ttnn.as_tensor(
        t.to(torch.bfloat16).t().unsqueeze(0).unsqueeze(0),
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

    # Adjacent-pair head_dim reorder for the fused RoPE kernel (see _FUSED_ROPE).  Only the ROTATED
    # projections move — wq, wk and their biases — so wv/wo and the residual stream are untouched.
    _perm = _interleave_perm(config.head_dim) if _FUSED_ROPE else None

    def _rope_perm(key: str, t: torch.Tensor) -> torch.Tensor:
        if _perm is None or not key.endswith(("attention.wq", "attention.wk")):
            return t
        return _permute_head_dim(t, config.head_dim, _perm)

    layers: List[LayerWeights] = []
    for i in range(config.num_hidden_layers):
        prefix = f"layers.{i}"

        def _w(key: str) -> ttnn.Tensor:
            return _tile(_rope_perm(key, state_dict[f"{prefix}.{key}.weight"]), device)

        def _b(key: str) -> Optional[ttnn.Tensor]:
            bias_key = f"{prefix}.{key}.bias"
            if bias_key in state_dict:
                b = _rope_perm(key, state_dict[bias_key]).to(torch.bfloat16)
                return ttnn.as_tensor(
                    b.view(1, 1, 1, -1),
                    device=device,
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
            return None

        # Fuse wk|wv (and biases) on the out dim once at load — decode keeps the fast wq
        # progcfg, while nlp_create_qkv_heads(q, input_kv=kv) drops the runtime concat +
        # separate K/V matmuls, and is byte-identical.
        wk_t = _rope_perm("attention.wk", state_dict[f"{prefix}.attention.wk.weight"])
        wv_t = state_dict[f"{prefix}.attention.wv.weight"]
        wkv_tt = _tile(torch.cat([wk_t, wv_t], dim=0), device)
        k_b = _b("attention.wk")
        v_b = _b("attention.wv")
        if k_b is not None and v_b is not None:
            kv_bias = ttnn.concat([k_b, v_b], dim=3, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        else:
            kv_bias = None

        lw = LayerWeights(
            wq=_w("attention.wq"),
            wkv=wkv_tt,
            wo=_w("attention.wo"),
            w1=_w("feed_forward.w1"),
            w2=_w("feed_forward.w2"),
            w3=_w("feed_forward.w3"),
            attn_norm_w=_norm_weight(state_dict[f"{prefix}.attention_norm.weight"], device),
            ffn_norm_w=_norm_weight(state_dict[f"{prefix}.ffn_norm.weight"], device),
            q_bias=_b("attention.wq"),
            kv_bias=kv_bias,
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

# VV_FUSED_ROPE=1 replaces the 9-op fp32 RoPE chain in the hot batch-2 decode with
# ttnn.experimental.rotary_embedding_llama (3.97 ms → 0.31 ms per speech frame) and builds/reads
# the cos/sin tables entirely on device.  The kernel pairs ADJACENT head_dim elements while
# VibeVoice/HF pair (i, i + hd/2), so the bridge is a relabelling of head_dim: permuting the
# ROTATED projections' out dim (wq, wk and their biases) by _interleave_perm at load makes q/k
# emerge already adjacent-paired, and permuting the cos/sin tables the same way makes RoPE commute
# with it.  Attention only ever contracts q against k — both permuted identically — and v/wo are
# untouched, so every model output is unchanged.  Prefill keeps its fp32 mul/add — the adjacent-pair
# rotate is a signed permutation, hence exact as an fp32 matmul on the bf16-valued q/k, so its RoPE
# is bit-exact for a given table; it shifts only by the device table's 1.2e-07 deviation from numpy
# (measured: prefill hidden PCC 0.999946).  The decode paths take the kernel's bf16 RoPE, which
# leaves greedy tokens unchanged over a synthetic 8-step check but is not bit-exact.
_FUSED_ROPE = os.environ.get("VV_FUSED_ROPE", "0") == "1"


def _interleave_perm(head_dim: int) -> np.ndarray:
    """head_dim index map from VibeVoice's half-split order to the fused kernel's adjacent-pair
    order: out[2i] = in[i], out[2i+1] = in[i + hd/2]."""
    p = np.empty(head_dim, dtype=np.int64)
    p[0::2] = np.arange(head_dim // 2)
    p[1::2] = np.arange(head_dim // 2, head_dim)
    return p


def _permute_head_dim(t: torch.Tensor, head_dim: int, perm: np.ndarray) -> torch.Tensor:
    """Reorder head_dim WITHIN each head of a projection weight [n*hd, in] or bias [n*hd]."""
    return t.reshape(-1, head_dim, *t.shape[1:])[:, perm].reshape(t.shape)


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


def _build_rope_tables_dev(
    seq_len: int,
    head_dim: int,
    device,
    rope_theta: float,
) -> Tuple[ttnn.Tensor, ttnn.Tensor, ttnn.Tensor, ttnn.Tensor]:
    """Build the adjacent-pair RoPE tables ON DEVICE (PART 1).

    ttnn.arange + a broadcast multiply + ttnn.cos/sin do the whole [seq_len, head_dim] table on
    device: 2.1 ms vs 160.8 ms for the numpy build, max |err| 1.2e-07 vs numpy (60 differing
    elements out of 16.8M after the bf16 rounding the fused kernel needs).  Only inv_freq's
    ``head_dim`` constants stay on host — deriving them on device from exp/pow carries ~5e-7
    relative error, which ``pos``·inv_freq amplifies to a 3.9e-3 absolute angle error at
    pos≈65k, i.e. a full bf16 ulp (measured: 690k differing elements).

    Returns (cos_tt, sin_tt, cos_emb, sin_emb): fp32 [1,1,seq_len,head_dim] TILE for the fp32
    applies to slice, plus bf16 [seq_len,head_dim] ROW_MAJOR for the per-position ttnn.embedding
    row gather (PART 2).
    """
    half = head_dim // 2
    inv_freq = (1.0 / (rope_theta ** (np.arange(0, half, dtype=np.float32) * 2.0 / head_dim))).astype(np.float32)
    # Adjacent-pair order repeats each frequency twice (half-split concat([f, f]) reordered).
    inv_tt = ttnn.as_tensor(
        torch.from_numpy(np.repeat(inv_freq, 2)).reshape(1, head_dim),
        device=device,
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    pos = ttnn.arange(0, seq_len, 1, dtype=ttnn.float32, device=device)
    pos = ttnn.reshape(ttnn.to_layout(pos, ttnn.TILE_LAYOUT), [seq_len, 1])
    ang = ttnn.multiply(pos, inv_tt, memory_config=ttnn.DRAM_MEMORY_CONFIG)  # [seq_len, head_dim]
    cos_2d = ttnn.cos(ang, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    sin_2d = ttnn.sin(ang, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    ttnn.deallocate(ang)
    ttnn.deallocate(pos)
    ttnn.deallocate(inv_tt)
    cos_emb = ttnn.to_layout(ttnn.typecast(cos_2d, ttnn.bfloat16), ttnn.ROW_MAJOR_LAYOUT)
    sin_emb = ttnn.to_layout(ttnn.typecast(sin_2d, ttnn.bfloat16), ttnn.ROW_MAJOR_LAYOUT)
    cos_tt = ttnn.reshape(cos_2d, [1, 1, seq_len, head_dim])
    sin_tt = ttnn.reshape(sin_2d, [1, 1, seq_len, head_dim])
    return cos_tt, sin_tt, cos_emb, sin_emb


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


def _apply_rope_interleaved_ttnn(
    x: ttnn.Tensor, cos: ttnn.Tensor, sin: ttnn.Tensor, trans_mat: ttnn.Tensor
) -> ttnn.Tensor:
    """Adjacent-pair RoPE in float32 — for equal cos/sin values, byte-identical to _apply_rope_ttnn
    on the permuted layout.

    The adjacent-pair rotate [-x1, x0, -x3, x2, …] is a signed permutation, so ``x @ trans_mat``
    reproduces it exactly for the bf16-valued q/k the projections emit (verified bit-exact at every
    decode and prefill shape); the surrounding fp32 mul/add/typecasts match _apply_rope_ttnn op for
    op.  Used by prefill, which therefore keeps its fp32 RoPE numerics.
    """
    x_f32 = ttnn.typecast(x, ttnn.float32)
    rotated = ttnn.add(
        ttnn.mul(x_f32, cos, memory_config=ttnn.DRAM_MEMORY_CONFIG),
        ttnn.mul(
            ttnn.matmul(x_f32, trans_mat, compute_kernel_config=_HIFI4, memory_config=ttnn.DRAM_MEMORY_CONFIG),
            sin,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        ),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    return ttnn.typecast(rotated, ttnn.bfloat16)


def _reshape_tt(x: ttnn.Tensor, shape: list) -> ttnn.Tensor:
    """Reshape landing in TILE layout.

    For a TILE input (every attention head-split / head-merge), ttnn.reshape reshapes it
    directly — byte-identical to the ROW_MAJOR round-trip (reshape is value-preserving;
    verified maxabsdiff==0 across the decode + prefill shapes) and ~3x cheaper on the
    decode path (drops the per-reshape untilize + tilize, ~1.6 ms / B=2 LM forward).
    A non-TILE input (the ROW_MAJOR embedding output) keeps the round-trip that also
    lands the result in TILE.
    """
    if x.layout == ttnn.TILE_LAYOUT:
        r = ttnn.reshape(x, shape)
        if r.layout != ttnn.TILE_LAYOUT:
            r = ttnn.to_layout(r, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        return r
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


def _max_pow2_divisor(n: int) -> int:
    """Largest power-of-two dividing n (matches ttnn sdpa_decode get_chunk_size)."""
    if n <= 0:
        return 1
    i = 1
    while i < n and n % (1 << (i + 1)) == 0:
        i += 1
    return 1 << i


def _k_chunk_from_cache_seq(cache_seq: int) -> int:
    """Auto k_chunk_size for fused SDPA-decode given fixed cache length S."""
    return min(512, _max_pow2_divisor(cache_seq))


def _fused_sdpa_decode_safe(valid_len: int, k_chunk: int) -> bool:
    """Return True when ``scaled_dot_product_attention_decode`` is safe on Blackhole."""
    if k_chunk >= 512:
        return True
    n_chunks = valid_len // k_chunk
    rem = valid_len % k_chunk
    if n_chunks < 2:
        return True
    return n_chunks == 2 and rem == 0


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

    # KV-cache seq length is rounded up to this multiple so fused SDPA-decode's auto
    # k_chunk_size (largest pow2 divisor of S, cap 512) avoids known kernel hangs.
    # 256-aligned caches pick k_chunk=256, and a valid_len of 513 then lays out as 2x256+1, which
    # hangs on Blackhole.  1024-aligned caches pick k_chunk=512, so 513 lays out as 1x512+1, which
    # is safe.
    _KV_ALIGN = 1024

    def __init__(self, weights: LMWeights, device):
        self.w = weights
        self.device = device
        self.cfg = weights.config
        self.scale = 1.0 / math.sqrt(self.cfg.head_dim)
        # Precompute full RoPE tables on device once (sliced per call via ttnn.slice)
        max_len = self.cfg.max_position_embeddings
        self._fused_rope = _FUSED_ROPE
        if not self._fused_rope:
            self._cos_tt, self._sin_tt = _build_rope_cache_tt(max_len, self.cfg.head_dim, device, self.cfg.rope_theta)
        # Causal-mask state for the fp32 prefill path (see _causal_mask).  The host builds only
        # the [S, S] triangular block, keyed by chunk length; the widened per-chunk mask lives in
        # a single slot so device DRAM stays flat across a chunked prefill.
        self._tri_cache: Dict[int, ttnn.Tensor] = {}
        self._mask_key: Optional[Tuple[int, int]] = None
        self._mask_tt: Optional[ttnn.Tensor] = None

        # ── Trace-safe decode state (Phase C) ──────────────────────────────
        # Host RoPE rows: the traced decode writes a per-position [1,1,1,hd] cos/sin
        # row into a persistent device buffer each step (instead of slicing the device
        # table with a Python-int position, which would bake into the trace).
        if self._fused_rope:
            # PART 1 on device: adjacent-pair tables built by ttnn arange/mul/cos/sin, no numpy
            # tables at all (so no host RoPE rows exist — PART 2 gathers them on device below).
            self._cos_tt, self._sin_tt, self._cos_emb, self._sin_emb = _build_rope_tables_dev(
                max_len, self.cfg.head_dim, device, self.cfg.rope_theta
            )
            self._cos_np = self._sin_np = None
        else:
            self._cos_np, self._sin_np = _build_rope_cache(
                max_len, self.cfg.head_dim, self.cfg.rope_theta
            )  # [max_len, hd]
            # On-device bf16 RoPE tables [max_len, hd] ROW_MAJOR for the llama-style path: the row
            # for a DEVICE position is gathered on-device via ttnn.embedding (bf16-only), so the
            # position can advance on-device (plus_one) with no per-step host RoPE write.  bf16 RoPE
            # is ~0.9999 PCC vs the fp32 host rows and does not flip greedy tokens.
            self._cos_emb = ttnn.as_tensor(
                torch.from_numpy(self._cos_np).to(torch.bfloat16),
                device=device,
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            self._sin_emb = ttnn.as_tensor(
                torch.from_numpy(self._sin_np).to(torch.bfloat16),
                device=device,
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        # Height-sharded L1 memcfg for the paged_update_cache input [1,1,n_kv,hd]
        # (heads tile-padded to 32, one batch row => one core).  paged_update_cache
        # takes a device-tensor write index so the KV write position varies per replay.
        _grid = device.compute_with_storage_grid_size()
        _shard_grid = ttnn.num_cores_to_corerangeset(1, _grid, True)
        self._kv_update_shard_mc = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(_shard_grid, [32, self.cfg.head_dim], ttnn.ShardOrientation.ROW_MAJOR),
        )
        if self._fused_rope:
            # PART 3 resources.  rotary_embedding_llama's decode mode wants [1,B,heads,hd]
            # height-sharded L1 with a TILE_HEIGHT-tall shard — which _kv_update_shard_mc already
            # is, so the rotated K feeds paged_update_cache with no extra conversion.  The kernel's
            # ±1 matrix is tile-local (32×32, bf16); the fp32 applies use the full head_dim one.
            self._rope_trans_bf16 = ttnn.as_tensor(
                get_rot_transformation_mat(dhead=32),
                device=device,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
                    ttnn.BufferType.L1,
                    ttnn.ShardSpec(_shard_grid, [32, 32], ttnn.ShardOrientation.ROW_MAJOR),
                ),
            )
            self._rope_trans_f32 = ttnn.as_tensor(
                get_rot_transformation_mat(dhead=self.cfg.head_dim),
                device=device,
                dtype=ttnn.float32,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

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

    def _causal_mask(self, S: int, S_total: int) -> ttnn.Tensor:
        """Additive causal mask [1, 1, S, S_total] for a prefill chunk at ``start_pos = S_total - S``.

        Only the trailing S columns are triangular: every column left of them is the chunk's
        strict past, so it is 0.0 for all S rows.  The host therefore builds one [S, S] block —
        the same block for every chunk of the same length — and the zero prefix is prepended on
        device.  Byte-identical to the per-chunk ``np.triu((S, S_total), k=S_total - S + 1)``
        upload it replaces (verified maxabsdiff 0.0, matching -inf counts, S_total up to 23040).

        Chunked prefill walks S_total monotonically and all 28 layers of a chunk share one key, so
        the widened mask is held in a single slot rather than a per-key dict: device DRAM stays at
        one mask instead of growing by [S, S_total] fp32 for all ~90 chunks.
        """
        if self._mask_key == (S, S_total):
            return self._mask_tt

        tri = self._tri_cache.get(S)
        if tri is None:
            tri = ttnn.as_tensor(
                np.triu(np.full((S, S), float("-inf"), dtype=np.float32), k=1)[np.newaxis, np.newaxis, :, :],
                device=self.device,
                dtype=ttnn.float32,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            self._tri_cache[S] = tri

        if S_total == S:
            mask = tri  # no prefix to prepend (single-shot prefill)
        else:
            zeros = ttnn.zeros(
                [1, 1, S, S_total - S],
                dtype=ttnn.float32,
                layout=ttnn.TILE_LAYOUT,
                device=self.device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            mask = ttnn.concat([zeros, tri], dim=-1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            ttnn.deallocate(zeros)

        self._release_causal_mask()  # the previous chunk's layers are all done with it
        self._mask_key, self._mask_tt = (S, S_total), mask
        return mask

    def _release_causal_mask(self) -> None:
        """Free the widened mask slot.  A slot whose key had ``S_total == S`` aliases a cached
        [S, S] block, which is kept (256 KB total) for the next prefill."""
        if self._mask_tt is not None and self._mask_key[0] != self._mask_key[1]:
            ttnn.deallocate(self._mask_tt)
        self._mask_key = self._mask_tt = None

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

        # Q + fused KV projections.  wq keeps the swept decode progcfg (S==1); wk|wv are
        # one matmul (wkv) so nlp_create_qkv_heads can take ``input_kv`` and skip the
        # runtime Q|K|V concat (byte-identical vs separate K/V + concat).  wkv uses its
        # own decode progcfg + width-sharded L1 out (S==1); bias-add below returns DRAM
        # for NlpCreateHeads.
        q = ttnn.linear(
            x,
            layer_w.wq,
            compute_kernel_config=_HIFI4,
            program_config=_QO_DECODE_PROGCFG if S == 1 else None,
            memory_config=_QO_DECODE_OUT_MEMCFG if S == 1 else ttnn.DRAM_MEMORY_CONFIG,
        )
        kv = ttnn.linear(
            x,
            layer_w.wkv,
            compute_kernel_config=_HIFI4,
            program_config=_WKV_DECODE_PROGCFG if S == 1 else None,
            memory_config=_WKV_DECODE_OUT_MEMCFG if S == 1 else ttnn.DRAM_MEMORY_CONFIG,
        )
        if layer_w.q_bias is not None:
            q = ttnn.add(q, layer_w.q_bias, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        if layer_w.kv_bias is not None:
            kv = ttnn.add(kv, layer_w.kv_bias, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        q, k, v = ttnn.experimental.nlp_create_qkv_heads(
            q,
            input_kv=kv,
            num_heads=n_heads,
            num_kv_heads=n_kv,
            transpose_k_heads=False,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )  # [B, n_heads/n_kv, S, hd]

        # Apply RoPE on device (validated fp32 path).
        # cos_sin_tt is the already-sliced [1,1,S,hd] window when hoisted by forward();
        # fall back to slicing the full table if a caller still passes the raw cache.
        if cos_sin_tt is not None:
            cos_tt, sin_tt = cos_sin_tt
            if cos_tt.shape[2] != S:
                cos_tt = ttnn.slice(
                    cos_tt, [0, 0, start_pos, 0], [1, 1, start_pos + S, head_dim], memory_config=ttnn.DRAM_MEMORY_CONFIG
                )
                sin_tt = ttnn.slice(
                    sin_tt, [0, 0, start_pos, 0], [1, 1, start_pos + S, head_dim], memory_config=ttnn.DRAM_MEMORY_CONFIG
                )
            if self._fused_rope:
                q = _apply_rope_interleaved_ttnn(q, cos_tt, sin_tt, self._rope_trans_f32)
                k = _apply_rope_interleaved_ttnn(k, cos_tt, sin_tt, self._rope_trans_f32)
            else:
                q = _apply_rope_ttnn(q, cos_tt, sin_tt)
                k = _apply_rope_ttnn(k, cos_tt, sin_tt)

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

            scores = ttnn.add(scores, self._causal_mask(S, S_total), memory_config=ttnn.DRAM_MEMORY_CONFIG)

            attn = ttnn.softmax(scores, dim=-1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            out = ttnn.matmul(attn, v_f32, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            out = ttnn.typecast(out, ttnn.bfloat16)
            # [B, n_heads, S, hd] → [B, 1, S, n_heads*hd]; byte-identical to permute+reshape
            # (maxabsdiff==0 on S=1 and S=256).
            out = ttnn.experimental.nlp_concat_heads(out, memory_config=ttnn.DRAM_MEMORY_CONFIG)
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
            # by a forced-token audio-parity check.  A grouped
            # fp32 manual decode matches tokens exactly but measured 358 ms/step (slower
            # than the old 202 ms), so it is not used.
            assert kv_cache is not None and kv_cache.keys[layer_idx] is not None, "decode needs an allocated KV cache"
            ttnn.update_cache(kv_cache.keys[layer_idx], k, start_pos)  # k: [1, n_kv, 1, hd]
            ttnn.update_cache(kv_cache.values[layer_idx], v, start_pos)

            cache_seq = kv_cache.max_seq or kv_cache.keys[layer_idx].shape[2]
            valid_len = start_pos + S
            k_chunk = _k_chunk_from_cache_seq(cache_seq)

            if _fused_sdpa_decode_safe(valid_len, k_chunk):
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
            else:
                # Fallback: fp32 manual GQA decode over cache prefix (slower but no hang).
                k_all = ttnn.slice(
                    kv_cache.keys[layer_idx],
                    [0, 0, 0, 0],
                    [B, n_kv, valid_len, head_dim],
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
                v_all = ttnn.slice(
                    kv_cache.values[layer_idx],
                    [0, 0, 0, 0],
                    [B, n_kv, valid_len, head_dim],
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
                repeat = n_heads // n_kv
                k_slices, v_slices = [], []
                for kv_idx in range(n_kv):
                    kh = ttnn.slice(
                        k_all,
                        [0, kv_idx, 0, 0],
                        [B, kv_idx + 1, valid_len, head_dim],
                        memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    )
                    vh = ttnn.slice(
                        v_all,
                        [0, kv_idx, 0, 0],
                        [B, kv_idx + 1, valid_len, head_dim],
                        memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    )
                    for _ in range(repeat):
                        k_slices.append(kh)
                        v_slices.append(vh)
                k_rep = ttnn.concat(k_slices, dim=1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
                v_rep = ttnn.concat(v_slices, dim=1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
                q_f32 = ttnn.typecast(q, ttnn.float32)
                k_f32 = ttnn.typecast(k_rep, ttnn.float32)
                v_f32 = ttnn.typecast(v_rep, ttnn.float32)
                k_t = ttnn.permute(k_f32, (0, 1, 3, 2))
                scores = ttnn.matmul(q_f32, k_t, memory_config=ttnn.DRAM_MEMORY_CONFIG)
                scores = ttnn.mul(scores, self.scale, memory_config=ttnn.DRAM_MEMORY_CONFIG)
                attn = ttnn.softmax(scores, dim=-1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
                out = ttnn.matmul(attn, v_f32, memory_config=ttnn.DRAM_MEMORY_CONFIG)
                out = ttnn.typecast(out, ttnn.bfloat16)
                out = _reshape_tt(out, [B, 1, S, n_heads * head_dim])

        # Output projection (1536x1536; same decode fast-path as wq).
        out = ttnn.linear(
            out,
            layer_w.wo,
            compute_kernel_config=_HIFI4,
            program_config=_QO_DECODE_PROGCFG if S == 1 else None,
            memory_config=_QO_DECODE_OUT_MEMCFG if S == 1 else ttnn.DRAM_MEMORY_CONFIG,
        )
        return out

    def _ffn_layer(self, x: ttnn.Tensor, layer_w: LayerWeights) -> ttnn.Tensor:
        """SwiGLU FFN: gate_proj(x) * silu(gate_proj(x)) → down_proj.

        Decode (S==1) uses byte-identical program configs that batch the CFG rows so the FFN
        weights are read once (see _FFN_*_DECODE_PROGCFG_*); prefill (S>1) keeps auto.
        """
        B, S = x.shape[0], x.shape[2]
        if S == 1 and B == 2:  # cfg-batch-2 deploy decode
            gate_pc, down_pc = _FFN_GATEUP_DECODE_PROGCFG_B2, _FFN_DOWN_DECODE_PROGCFG_B2
        elif S == 1 and B == 1:  # eager / B=1 traced decode (gate/up: no win over auto)
            gate_pc, down_pc = None, _FFN_DOWN_DECODE_PROGCFG_B1
        else:  # prefill (S>1) → auto
            gate_pc, down_pc = None, None
        # L1-island for gate/up matmul outputs + silu:
        # - Decode B==2: explicit progcfg + L1, ~1.15x and maxabsdiff==0.
        # - Prefill S>1: auto + L1 interleaved is byte-identical and ~1.13x on the 5-op chain.
        #   Prefill down must keep a DRAM in0 — auto with an L1 in0 re-picks the K-reduction
        #   (maxabsdiff≠0) — so silu→mul stays in DRAM below.
        # - Decode B==1: auto + L1 gate is byte-identical in isolation, but the full chain is not
        #   faster, so it stays in DRAM.
        gateup_mc = ttnn.L1_MEMORY_CONFIG if ((S == 1 and B == 2) or S > 1) else ttnn.DRAM_MEMORY_CONFIG
        gate = ttnn.linear(x, layer_w.w1, compute_kernel_config=_HIFI4, program_config=gate_pc, memory_config=gateup_mc)
        up = ttnn.linear(x, layer_w.w3, compute_kernel_config=_HIFI4, program_config=gate_pc, memory_config=gateup_mc)
        gate = ttnn.silu(gate, memory_config=gateup_mc)
        # Place the SwiGLU product (down_proj's in0) in L1 for decode: down_proj is the biggest LM
        # matmul (K=8960, 24 cores, ~45% DRAM BW) and reads in0 faster from L1 (151 -> 134 µs at
        # B=2, 1.13x).  Placement only — same in0_block_w K-reduction, so byte-identical
        # (maxabsdiff==0).  Prefill (S>1) keeps DRAM.
        hidden_mc = ttnn.L1_MEMORY_CONFIG if S == 1 else ttnn.DRAM_MEMORY_CONFIG
        hidden = ttnn.mul(gate, up, memory_config=hidden_mc)
        out = ttnn.linear(
            hidden,
            layer_w.w2,
            compute_kernel_config=_HIFI4,
            program_config=down_pc,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
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

        # Pre-norm + attention.  Decode (S==1): emit attn-norm into L1 so wq/wkv read
        # in0 from L1 (byte-identical memory placement; pairs with _WKV_DECODE_*).
        S = x.shape[2]
        x_norm = ttnn.rms_norm(
            x,
            weight=lw.attn_norm_w,
            epsilon=self.cfg.rms_norm_eps,
            compute_kernel_config=_HIFI4,
            memory_config=ttnn.L1_MEMORY_CONFIG if S == 1 else ttnn.DRAM_MEMORY_CONFIG,
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
        compute_logits: bool = True,
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
        S = inputs_embeds.shape[2]
        B = inputs_embeds.shape[0]
        cfg = self.cfg

        # Hoist RoPE cos/sin slice once per forward (same window for all 28 layers).
        # Avoids 2×num_layers redundant Slice ops on the decode/prefill path.
        head_dim = cfg.head_dim
        cos_row = ttnn.slice(
            self._cos_tt, [0, 0, start_pos, 0], [1, 1, start_pos + S, head_dim], memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        sin_row = ttnn.slice(
            self._sin_tt, [0, 0, start_pos, 0], [1, 1, start_pos + S, head_dim], memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        cos_sin_tt = (cos_row, sin_row)

        x = inputs_embeds
        if x.dtype == ttnn.float32:
            x = ttnn.typecast(x, ttnn.bfloat16)
        for layer_idx in range(cfg.num_hidden_layers):
            x = self._transformer_layer(x, layer_idx, cos_sin_tt, kv_cache, start_pos)

        # Final norm
        x = ttnn.rms_norm(
            x,
            weight=self.w.norm_w,
            epsilon=cfg.rms_norm_eps,
            compute_kernel_config=_HIFI4,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        last_hidden = ttnn.typecast(x, ttnn.float32) if return_last_hidden else None

        # Non-final prefill chunks: their logits are discarded by the sampler, so skip the
        # vocab-151936 matmul entirely (~1.7 ms per intermediate chunk).
        if not compute_logits:
            return None, last_hidden

        # Only the LAST position's logits are consumed (greedy argmax of the next token),
        # so for prefill (S>1) run lm_head on just the last row — bit-exact, and cuts the
        # S×1536×151936 matmul's M from S to 1 (1752→1215 µs; the 467 MB weight read is the
        # remaining floor).  last_hidden stays full-S.
        x_head = (
            x
            if S == 1
            else ttnn.slice(x, [0, 0, S - 1, 0], [B, 1, S, x.shape[-1]], memory_config=ttnn.DRAM_MEMORY_CONFIG)
        )

        # LM head projection → logits
        logits = ttnn.linear(
            x_head,
            self.w.lm_head_w,
            compute_kernel_config=_HIFI4,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        return logits, last_hidden

    # ── Trace-safe decode (Phase C) ────────────────────────────────────────
    # Mirrors the eager S==1 decode but is fully driven by device tensors so the
    # 28-layer step can be captured once and replayed: KV write position and the
    # SDPA read bound come from ``cur_pos`` (a device int32 tensor) via
    # paged_update_cache / sdpa cur_pos_tensor, and RoPE comes from a host-written
    # per-position [1,1,1,hd] row.  Numerically equivalent to the eager fused path.
    def _attention_decode_traced(
        self,
        x: ttnn.Tensor,
        layer_w: LayerWeights,
        cos_row: ttnn.Tensor,
        sin_row: ttnn.Tensor,
        cur_pos: ttnn.Tensor,
        kv_cache: KVCache,
        layer_idx: int,
    ) -> ttnn.Tensor:
        cfg = self.cfg
        B, S = 1, 1
        head_dim = cfg.head_dim
        n_heads = cfg.num_attention_heads
        n_kv = cfg.num_key_value_heads

        q = ttnn.linear(
            x,
            layer_w.wq,
            compute_kernel_config=_HIFI4,
            program_config=_QO_DECODE_PROGCFG,
            memory_config=_QO_DECODE_OUT_MEMCFG,
        )
        kv = ttnn.linear(
            x,
            layer_w.wkv,
            compute_kernel_config=_HIFI4,
            program_config=_WKV_DECODE_PROGCFG,
            memory_config=_WKV_DECODE_OUT_MEMCFG,
        )
        if layer_w.q_bias is not None:
            q = ttnn.add(q, layer_w.q_bias, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        if layer_w.kv_bias is not None:
            kv = ttnn.add(kv, layer_w.kv_bias, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        q, k, v = ttnn.experimental.nlp_create_qkv_heads(
            q,
            input_kv=kv,
            num_heads=n_heads,
            num_kv_heads=n_kv,
            transpose_k_heads=False,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )  # [1, n_heads/n_kv, 1, hd]

        # RoPE via the per-position row (broadcasts over the head dim; same numerics
        # as the eager sliced-table path).  cos_row/sin_row are the height-sharded bf16 rows on the
        # fused path, where the rotate also delivers the [1,B,heads,hd] layout used below.
        if self._fused_rope:
            q_dec = self._rope_decode_fused(q, cos_row, sin_row)
            k_1bkd = self._rope_decode_fused(k, cos_row, sin_row)
        else:
            q = _apply_rope_ttnn(q, cos_row, sin_row)
            k = _apply_rope_ttnn(k, cos_row, sin_row)
            q_dec = ttnn.permute(q, (0, 2, 1, 3))  # [1, B, n_heads, hd]
            # KV write at cur_pos: paged_update_cache needs input [1,B,n_kv,hd] height-sharded L1.
            k_1bkd = ttnn.to_memory_config(ttnn.permute(k, (0, 2, 1, 3)), self._kv_update_shard_mc)
        v_1bkd = ttnn.to_memory_config(ttnn.permute(v, (0, 2, 1, 3)), self._kv_update_shard_mc)
        ttnn.experimental.paged_update_cache(
            kv_cache.keys[layer_idx], k_1bkd, update_idxs_tensor=cur_pos, page_table=None
        )
        ttnn.experimental.paged_update_cache(
            kv_cache.values[layer_idx], v_1bkd, update_idxs_tensor=cur_pos, page_table=None
        )

        attn = ttnn.transformer.scaled_dot_product_attention_decode(
            q_dec,
            kv_cache.keys[layer_idx],
            kv_cache.values[layer_idx],
            cur_pos_tensor=cur_pos,
            scale=self.scale,
            program_config=_SDPA_DECODE_CFG,
            compute_kernel_config=_HIFI4,
        )  # [1, B, n_heads, hd]
        out = _reshape_tt(attn, [B, 1, S, n_heads * head_dim])
        out = ttnn.linear(
            out,
            layer_w.wo,
            compute_kernel_config=_HIFI4,
            program_config=_QO_DECODE_PROGCFG,
            memory_config=_QO_DECODE_OUT_MEMCFG,
        )
        return out

    def _transformer_layer_traced(
        self,
        x: ttnn.Tensor,
        layer_idx: int,
        cos_row: ttnn.Tensor,
        sin_row: ttnn.Tensor,
        cur_pos: ttnn.Tensor,
        kv_cache: KVCache,
    ) -> ttnn.Tensor:
        lw = self.w.layers[layer_idx]
        x_norm = ttnn.rms_norm(
            x,
            weight=lw.attn_norm_w,
            epsilon=self.cfg.rms_norm_eps,
            compute_kernel_config=_HIFI4,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        attn_out = self._attention_decode_traced(x_norm, lw, cos_row, sin_row, cur_pos, kv_cache, layer_idx)
        x = ttnn.add(x, attn_out, memory_config=ttnn.DRAM_MEMORY_CONFIG)
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

    def build_lm_head_subset(self, token_ids) -> ttnn.Tensor:
        """Return a [1,1,hidden,N] tiled lm_head weight holding ONLY the columns for ``token_ids``
        (in the given order).  For a constrained greedy decode where only a handful of tokens are
        selectable, projecting hidden by this subset and argmax over the N logits is IDENTICAL to
        argmax over the full vocab with all other tokens masked to -inf — but replaces the
        [hidden x 151936] matmul + full-vocab mask-add + full-vocab argmax with a [hidden x N]
        matmul + N-wide argmax.  Pass token_ids sorted ascending so argmax tie-breaking matches the
        full-vocab argmax exactly."""
        full = ttnn.to_torch(self.w.lm_head_w).to(torch.float32)  # [1,1,hidden,vocab]
        sub = full[:, :, :, list(token_ids)].contiguous()  # [1,1,hidden,N]
        return ttnn.as_tensor(
            sub, device=self.device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )

    def forward_decode_traced_embeds(
        self,
        inputs_embeds: ttnn.Tensor,
        cos_row: ttnn.Tensor,
        sin_row: ttnn.Tensor,
        cur_pos: ttnn.Tensor,
        kv_cache: KVCache,
        return_last_hidden: bool = False,
        need_logits: bool = True,
        lm_head_w: Optional[ttnn.Tensor] = None,
    ) -> Tuple[Optional[ttnn.Tensor], Optional[ttnn.Tensor]]:
        """Capturable single-token decode over an already-embedded input [1,1,1,hidden].

        ``need_logits=False`` skips the lm_head projection entirely (used by the negative-CFG
        forward, whose logits are discarded — bit-exact, saves the full lm_head).  ``lm_head_w``
        (a [1,1,hidden,N] column subset) projects only the selectable tokens for a constrained
        decode — argmax over its N logits == argmax over the full vocab masked to the same tokens."""
        cfg = self.cfg
        x = inputs_embeds
        if x.dtype == ttnn.float32:
            x = ttnn.typecast(x, ttnn.bfloat16)
        if self._fused_rope:
            # PART 2 on device: cos/sin come from cur_pos via the device table, so the caller's
            # host-written rows are unused (and are not even built — see __init__).
            cos_row, sin_row = self._rope_rows_sharded(cur_pos)
        for layer_idx in range(cfg.num_hidden_layers):
            x = self._transformer_layer_traced(x, layer_idx, cos_row, sin_row, cur_pos, kv_cache)
        x = ttnn.rms_norm(
            x,
            weight=self.w.norm_w,
            epsilon=cfg.rms_norm_eps,
            compute_kernel_config=_HIFI4,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        last_hidden = ttnn.typecast(x, ttnn.float32) if return_last_hidden else None
        logits = None
        if need_logits:
            head_w = lm_head_w if lm_head_w is not None else self.w.lm_head_w
            logits = ttnn.linear(x, head_w, compute_kernel_config=_HIFI4, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        return logits, last_hidden

    def _rope_rows_from_pos(self, cur_pos: ttnn.Tensor) -> Tuple[ttnn.Tensor, ttnn.Tensor]:
        """Gather the bf16 RoPE cos/sin rows for a DEVICE position (llama-style, on-device).

        cur_pos: [1] int32 device tensor.  Returns cos/sin [1,1,1,hd] bf16.  Uses ttnn.embedding
        (bf16-only) so the position can be a device tensor advanced by plus_one — no host RoPE
        write.  Numerically = the fp32 sinusoid table rounded to bf16 (~0.9999 PCC vs fp32 rows).
        """
        hd = self.cfg.head_dim
        idx = ttnn.reshape(ttnn.typecast(cur_pos, ttnn.uint32), [1, 1])
        cos = ttnn.embedding(idx, self._cos_emb, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        sin = ttnn.embedding(idx, self._sin_emb, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        return ttnn.reshape(cos, [1, 1, 1, hd]), ttnn.reshape(sin, [1, 1, 1, hd])

    def _rope_rows_sharded(self, cur_pos: ttnn.Tensor) -> Tuple[ttnn.Tensor, ttnn.Tensor]:
        """PART 2 on device: gather this position's adjacent-pair cos/sin row and land it in the
        height-sharded L1 layout the fused kernel needs.  Called ONCE per decode step (not per
        layer) — the 28 layers all rotate against the same row."""
        cos, sin = self._rope_rows_from_pos(cur_pos)
        return (
            ttnn.to_memory_config(cos, self._kv_update_shard_mc),
            ttnn.to_memory_config(sin, self._kv_update_shard_mc),
        )

    def _rope_decode_fused(self, t: ttnn.Tensor, cos_sh: ttnn.Tensor, sin_sh: ttnn.Tensor) -> ttnn.Tensor:
        """PART 3 on device: adjacent-pair RoPE on a decode q or k via the fused kernel.

        Takes [1, n, 1, hd] and returns [1, 1, n, hd] height-sharded L1 — exactly what sdpa-decode
        (q) and paged_update_cache (k) consume next, so the permute the fp32 path did afterwards is
        absorbed here and no conversion is added.
        """
        t = ttnn.to_memory_config(ttnn.permute(t, (0, 2, 1, 3)), self._kv_update_shard_mc)
        return ttnn.experimental.rotary_embedding_llama(t, cos_sh, sin_sh, self._rope_trans_bf16, is_decode_mode=True)

    def forward_decode_traced_embeds_dev_rope(
        self,
        inputs_embeds: ttnn.Tensor,
        cur_pos: ttnn.Tensor,
        kv_cache: KVCache,
        return_last_hidden: bool = False,
    ) -> Tuple[ttnn.Tensor, Optional[ttnn.Tensor]]:
        """Like forward_decode_traced_embeds but the RoPE rows are gathered ON DEVICE from
        cur_pos (bf16) instead of supplied as host-written fp32 rows — so the whole step,
        including RoPE-row selection, is driven by the device position tensor (llama pattern)."""
        cos_row, sin_row = self._rope_rows_from_pos(cur_pos)
        return self.forward_decode_traced_embeds(inputs_embeds, cos_row, sin_row, cur_pos, kv_cache, return_last_hidden)

    # ── CFG batch-2 fused decode (pos row0 + neg row1 in one B=2 forward) ─────────
    # The two CFG forwards (pos-LM, neg-LM) are weight-DRAM-bound at M=1, so batching
    # their inputs into [2,1,1,H] reads each layer's weights ONCE for both rows.  Only
    # the weight-bound MATMULS are batched (qkv/o/gate/up/down); attention (rope / KV
    # write / sdpa) stays per-row on the two SEPARATE [1,..] caches, i.e. byte-identical
    # to the B=1 attention (no batched KV cache, no extra DRAM).  Every batched op is
    # byte-identical per row.
    def _attention_decode_traced_b2(
        self,
        x: ttnn.Tensor,  # [2,1,1,H]  row0=pos, row1=neg
        layer_w: LayerWeights,
        rope_rows,  # [(cos0,sin0),(cos1,sin1)]  per-row [1,1,1,hd]
        cur_positions,  # [cur_pos0, cur_pos1]  per-row [1] int32
        kv_caches,  # [kv0, kv1]  separate [1,..] caches
        layer_idx: int,
    ) -> ttnn.Tensor:
        cfg = self.cfg
        head_dim = cfg.head_dim
        n_heads = cfg.num_attention_heads
        n_kv = cfg.num_key_value_heads

        # Batched weight-bound projections — read wq/wkv once for both rows.
        q = ttnn.linear(
            x,
            layer_w.wq,
            compute_kernel_config=_HIFI4,
            program_config=_QO_DECODE_PROGCFG_B2,
            memory_config=_QO_DECODE_OUT_MEMCFG,
        )
        kv = ttnn.linear(
            x,
            layer_w.wkv,
            compute_kernel_config=_HIFI4,
            program_config=_WKV_DECODE_PROGCFG_B2,
            memory_config=_WKV_DECODE_OUT_MEMCFG,
        )
        if layer_w.q_bias is not None:
            q = ttnn.add(q, layer_w.q_bias, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        if layer_w.kv_bias is not None:
            kv = ttnn.add(kv, layer_w.kv_bias, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        q, k, v = ttnn.experimental.nlp_create_qkv_heads(
            q,
            input_kv=kv,
            num_heads=n_heads,
            num_kv_heads=n_kv,
            transpose_k_heads=False,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )  # [2, n_heads/n_kv, 1, hd]

        # Per-row attention on the two separate caches — identical ops to the B=1 path.
        attn_rows = []
        for b in range(2):
            cos_row, sin_row = rope_rows[b]
            cur_pos = cur_positions[b]
            kv_cache = kv_caches[b]
            qb = ttnn.slice(q, [b, 0, 0, 0], [b + 1, n_heads, 1, head_dim], memory_config=ttnn.DRAM_MEMORY_CONFIG)
            kb = ttnn.slice(k, [b, 0, 0, 0], [b + 1, n_kv, 1, head_dim], memory_config=ttnn.DRAM_MEMORY_CONFIG)
            vb = ttnn.slice(v, [b, 0, 0, 0], [b + 1, n_kv, 1, head_dim], memory_config=ttnn.DRAM_MEMORY_CONFIG)
            if self._fused_rope:
                q_dec = self._rope_decode_fused(qb, cos_row, sin_row)
                k_1bkd = self._rope_decode_fused(kb, cos_row, sin_row)
            else:
                qb = _apply_rope_ttnn(qb, cos_row, sin_row)
                kb = _apply_rope_ttnn(kb, cos_row, sin_row)
                q_dec = ttnn.permute(qb, (0, 2, 1, 3))  # [1, 1, n_heads, hd]
                k_1bkd = ttnn.to_memory_config(ttnn.permute(kb, (0, 2, 1, 3)), self._kv_update_shard_mc)
            v_1bkd = ttnn.to_memory_config(ttnn.permute(vb, (0, 2, 1, 3)), self._kv_update_shard_mc)
            ttnn.experimental.paged_update_cache(
                kv_cache.keys[layer_idx], k_1bkd, update_idxs_tensor=cur_pos, page_table=None
            )
            ttnn.experimental.paged_update_cache(
                kv_cache.values[layer_idx], v_1bkd, update_idxs_tensor=cur_pos, page_table=None
            )
            attn = ttnn.transformer.scaled_dot_product_attention_decode(
                q_dec,
                kv_cache.keys[layer_idx],
                kv_cache.values[layer_idx],
                cur_pos_tensor=cur_pos,
                scale=self.scale,
                program_config=_SDPA_DECODE_CFG,
                compute_kernel_config=_HIFI4,
            )
            attn_rows.append(_reshape_tt(attn, [1, 1, 1, n_heads * head_dim]))

        attn = ttnn.concat(attn_rows, dim=0, memory_config=ttnn.DRAM_MEMORY_CONFIG)  # [2,1,1,n_heads*hd]
        out = ttnn.linear(
            attn,
            layer_w.wo,
            compute_kernel_config=_HIFI4,
            program_config=_QO_DECODE_PROGCFG_B2,
            memory_config=_QO_DECODE_OUT_MEMCFG,
        )
        return out  # [2,1,1,H]

    def _transformer_layer_traced_b2(self, x, layer_idx, rope_rows, cur_positions, kv_caches) -> ttnn.Tensor:
        lw = self.w.layers[layer_idx]
        x_norm = ttnn.rms_norm(
            x,
            weight=lw.attn_norm_w,
            epsilon=self.cfg.rms_norm_eps,
            compute_kernel_config=_HIFI4,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        attn_out = self._attention_decode_traced_b2(x_norm, lw, rope_rows, cur_positions, kv_caches, layer_idx)
        x = ttnn.add(x, attn_out, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        x_norm = ttnn.rms_norm(
            x,
            weight=lw.ffn_norm_w,
            epsilon=self.cfg.rms_norm_eps,
            compute_kernel_config=_HIFI4,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ffn_out = self._ffn_layer(x_norm, lw)  # batched [2,..] — auto matmuls, batch-independent
        x = ttnn.add(x, ffn_out, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        return x

    def forward_decode_traced_embeds_b2(
        self,
        embeds_b2: ttnn.Tensor,  # [2,1,1,H]  row0=pos input, row1=neg input
        rope_rows,  # [(cos0,sin0),(cos1,sin1)]
        cur_positions,  # [cur_pos0, cur_pos1]
        kv_caches,  # [kv0, kv1]
        lm_head_w: Optional[ttnn.Tensor] = None,
    ) -> Tuple[ttnn.Tensor, ttnn.Tensor]:
        """Fused batch-2 decode: row0 = pos-LM (input+pos+kv0), row1 = neg-LM (input+pos+kv1).

        Returns (row0_logits, hidden_b2[2,1,1,H] fp32).  The lm_head is projected on ROW0 ONLY
        (pos-LM produces the token; neg logits are discarded, exactly like the B=1 need_logits=False
        neg forward).  hidden_b2[0] == the B=1 pos last_hidden, hidden_b2[1] == the B=1 neg
        last_hidden, both byte-identical (per-row batch independence)."""
        cfg = self.cfg
        x = embeds_b2
        if x.dtype == ttnn.float32:
            x = ttnn.typecast(x, ttnn.bfloat16)
        if self._fused_rope:
            # PART 2 on device, once per step: both CFG rows' cos/sin gathered from their device
            # positions, so the caller's host-written rows are unused (and are not even built).
            rope_rows = [self._rope_rows_sharded(p) for p in cur_positions]
        for layer_idx in range(cfg.num_hidden_layers):
            x = self._transformer_layer_traced_b2(x, layer_idx, rope_rows, cur_positions, kv_caches)
        x = ttnn.rms_norm(
            x,
            weight=self.w.norm_w,
            epsilon=cfg.rms_norm_eps,
            compute_kernel_config=_HIFI4,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        hidden_b2 = ttnn.typecast(x, ttnn.float32)  # [2,1,1,H]
        x0 = ttnn.slice(x, [0, 0, 0, 0], [1, 1, 1, cfg.hidden_size], memory_config=ttnn.DRAM_MEMORY_CONFIG)
        head_w = lm_head_w if lm_head_w is not None else self.w.lm_head_w
        logits0 = ttnn.linear(x0, head_w, compute_kernel_config=_HIFI4, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        return logits0, hidden_b2

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
                compute_logits=(end >= S),  # only the final chunk's logits are consumed
            )
        self._release_causal_mask()  # decode never reads it; don't hold [S, S_total] fp32 for the run
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
