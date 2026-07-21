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

# Decode-only program config for the wq/wo 1536x1536 projections (single-token
# step, Mt=1).  Sweep winner (matmul/test_matmul_32x1536x1536_sweep.py): 1D
# mcast_in0, 8x3=24 cores, in0_block_w=4, per_core_N=2, out_subblock 1x2,
# width-sharded output -> 12.25us vs 25.4us auto baseline (2.08x).  per_core_M=1
# makes it valid ONLY for S==1 decode; prefill (S>1, Mt>1) keeps the auto config.
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
# Proven byte-identical per row to the per_core_M=1 B=1 config (cfg_batch2_byteident_probe.py:
# row0 maxabsdiff==0), i.e. the K-reduction order is preserved — long-form-safe.
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

# Decode-only FFN program configs (S==1).  BYTE-IDENTICAL to the auto config (in0_block_w=2 is the
# K-reduction block auto uses for these shapes — proven maxabsdiff==0 vs auto in
# tests/perf/ffn_byteident_ibw_sweep.py), so the K-reduction order — hence the bf16 rounding — is
# preserved.  The win is the cfg-batch-2 weight-read-once pattern extended to the FFN: the cfg-b2
# LM fusion batched the wq/wo matmuls (per_core_M=2) but left the FFN on auto, which reads each
# FFN weight matrix TWICE (once per CFG row).  per_core_M=2 folds both rows into M so the weights
# are read once -> ~1.9x on the down-proj + ~1.85x each on gate/up at B=2, all long-form-safe (Tier-0).
# per_core_M makes these valid ONLY for S==1 decode; prefill (S>1) keeps auto.
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
    # 256-aligned caches pick k_chunk=256; valid_len=513 → layout 2×256+1 HANGs on
    # Blackhole (see scripts/vv_sdpa_decode_sweep.py). 1024-aligned → k_chunk=512;
    # valid_len=513 → 1×512+1 is OK (sweep-verified).
    _KV_ALIGN = 1024

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

        # ── Trace-safe decode state (Phase C) ──────────────────────────────
        # Host RoPE rows: the traced decode writes a per-position [1,1,1,hd] cos/sin
        # row into a persistent device buffer each step (instead of slicing the device
        # table with a Python-int position, which would bake into the trace).
        self._cos_np, self._sin_np = _build_rope_cache(max_len, self.cfg.head_dim, self.cfg.rope_theta)  # [max_len, hd]
        # On-device bf16 RoPE tables [max_len, hd] ROW_MAJOR for the llama-style path: the row
        # for a DEVICE position is gathered on-device via ttnn.embedding (bf16-only), so the
        # position can advance on-device (plus_one) with no per-step host RoPE write.  bf16 RoPE
        # is ~0.9999 PCC vs the fp32 host rows and does not flip greedy tokens (bf16_rope_accuracy.py).
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
        # wq is 1536x1536; on a single-token decode step (S==1) use the swept fast
        # config (2.08x), else the auto config for prefill chunks.
        q = ttnn.linear(
            x,
            layer_w.wq,
            compute_kernel_config=_HIFI4,
            program_config=_QO_DECODE_PROGCFG if S == 1 else None,
            memory_config=_QO_DECODE_OUT_MEMCFG if S == 1 else ttnn.DRAM_MEMORY_CONFIG,
        )
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
        gate = ttnn.linear(
            x, layer_w.w1, compute_kernel_config=_HIFI4, program_config=gate_pc, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        up = ttnn.linear(
            x, layer_w.w3, compute_kernel_config=_HIFI4, program_config=gate_pc, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        gate = ttnn.silu(gate, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        hidden = ttnn.mul(gate, up, memory_config=ttnn.DRAM_MEMORY_CONFIG)
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
        k = ttnn.linear(x, layer_w.wk, compute_kernel_config=_HIFI4, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        v = ttnn.linear(x, layer_w.wv, compute_kernel_config=_HIFI4, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        if layer_w.q_bias is not None:
            q = ttnn.add(q, layer_w.q_bias, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        if layer_w.k_bias is not None:
            k = ttnn.add(k, layer_w.k_bias, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        if layer_w.v_bias is not None:
            v = ttnn.add(v, layer_w.v_bias, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        q = ttnn.permute(_reshape_tt(q, [B, S, n_heads, head_dim]), (0, 2, 1, 3))  # [1, n_heads, 1, hd]
        k = ttnn.permute(_reshape_tt(k, [B, S, n_kv, head_dim]), (0, 2, 1, 3))  # [1, n_kv, 1, hd]
        v = ttnn.permute(_reshape_tt(v, [B, S, n_kv, head_dim]), (0, 2, 1, 3))

        # RoPE via the per-position row (broadcasts over the head dim; same numerics
        # as the eager sliced-table path).
        q = _apply_rope_ttnn(q, cos_row, sin_row)
        k = _apply_rope_ttnn(k, cos_row, sin_row)

        # KV write at cur_pos: paged_update_cache needs input [1,B,n_kv,hd] height-sharded L1.
        k_1bkd = ttnn.to_memory_config(ttnn.permute(k, (0, 2, 1, 3)), self._kv_update_shard_mc)
        v_1bkd = ttnn.to_memory_config(ttnn.permute(v, (0, 2, 1, 3)), self._kv_update_shard_mc)
        ttnn.experimental.paged_update_cache(
            kv_cache.keys[layer_idx], k_1bkd, update_idxs_tensor=cur_pos, page_table=None
        )
        ttnn.experimental.paged_update_cache(
            kv_cache.values[layer_idx], v_1bkd, update_idxs_tensor=cur_pos, page_table=None
        )

        q_dec = ttnn.permute(q, (0, 2, 1, 3))  # [1, B, n_heads, hd]
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
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
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
    # to the current B=1 attention (no batched KV cache, no extra DRAM).  Every batched
    # op is proven byte-identical per row (cfg_batch2_byteident_probe.py +
    # cfg_batch2_sdpa_byteident_probe.py) → Tier-0.
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

        # Batched weight-bound projections — read wq/wk/wv once for both rows.
        q = ttnn.linear(
            x,
            layer_w.wq,
            compute_kernel_config=_HIFI4,
            program_config=_QO_DECODE_PROGCFG_B2,
            memory_config=_QO_DECODE_OUT_MEMCFG,
        )
        k = ttnn.linear(x, layer_w.wk, compute_kernel_config=_HIFI4, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        v = ttnn.linear(x, layer_w.wv, compute_kernel_config=_HIFI4, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        if layer_w.q_bias is not None:
            q = ttnn.add(q, layer_w.q_bias, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        if layer_w.k_bias is not None:
            k = ttnn.add(k, layer_w.k_bias, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        if layer_w.v_bias is not None:
            v = ttnn.add(v, layer_w.v_bias, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        # [2,1,1,X] → [2, X_heads, 1, hd]; the reshape reshards the width-sharded q to DRAM.
        q = ttnn.permute(_reshape_tt(q, [2, 1, n_heads, head_dim]), (0, 2, 1, 3))  # [2, n_heads, 1, hd]
        k = ttnn.permute(_reshape_tt(k, [2, 1, n_kv, head_dim]), (0, 2, 1, 3))  # [2, n_kv, 1, hd]
        v = ttnn.permute(_reshape_tt(v, [2, 1, n_kv, head_dim]), (0, 2, 1, 3))

        # Per-row attention on the two separate caches — identical ops to the B=1 path.
        attn_rows = []
        for b in range(2):
            cos_row, sin_row = rope_rows[b]
            cur_pos = cur_positions[b]
            kv_cache = kv_caches[b]
            qb = ttnn.slice(q, [b, 0, 0, 0], [b + 1, n_heads, 1, head_dim], memory_config=ttnn.DRAM_MEMORY_CONFIG)
            kb = ttnn.slice(k, [b, 0, 0, 0], [b + 1, n_kv, 1, head_dim], memory_config=ttnn.DRAM_MEMORY_CONFIG)
            vb = ttnn.slice(v, [b, 0, 0, 0], [b + 1, n_kv, 1, head_dim], memory_config=ttnn.DRAM_MEMORY_CONFIG)
            qb = _apply_rope_ttnn(qb, cos_row, sin_row)
            kb = _apply_rope_ttnn(kb, cos_row, sin_row)
            k_1bkd = ttnn.to_memory_config(ttnn.permute(kb, (0, 2, 1, 3)), self._kv_update_shard_mc)
            v_1bkd = ttnn.to_memory_config(ttnn.permute(vb, (0, 2, 1, 3)), self._kv_update_shard_mc)
            ttnn.experimental.paged_update_cache(
                kv_cache.keys[layer_idx], k_1bkd, update_idxs_tensor=cur_pos, page_table=None
            )
            ttnn.experimental.paged_update_cache(
                kv_cache.values[layer_idx], v_1bkd, update_idxs_tensor=cur_pos, page_table=None
            )
            q_dec = ttnn.permute(qb, (0, 2, 1, 3))  # [1, 1, n_heads, hd]
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
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
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
