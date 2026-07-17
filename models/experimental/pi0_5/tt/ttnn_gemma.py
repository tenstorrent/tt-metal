# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Gemma transformer blocks - TTNN Implementation (Optimized).

This module implements Gemma 2B style transformer layers using TTNN operations:
    - RMSNorm (using native ttnn.rms_norm)
    - Multi-Query Attention (MQA) with fused QKV and native RoPE
    - GeGLU MLP (gated GELU activation)
    - Native head operations (nlp_create_qkv_heads, nlp_concat_heads)

Architecture configurations:
    - Gemma 2B (VLM): width=2048, depth=18, mlp_dim=16384, heads=8, kv_heads=1
    - Gemma 300M (Expert): width=1024, depth=18, mlp_dim=4096, heads=8, kv_heads=1

Optimizations over baseline:
    1. Fused QKV projection (1 linear instead of 3)
    2. Native ttnn.experimental.nlp_create_qkv_heads
    3. Native ttnn.experimental.rotary_embedding (split-half pattern)
    4. Native ttnn.experimental.nlp_concat_heads for output
    5. Native ttnn.rms_norm (single fused kernel)
    6. Pure TTNN RoPE precomputation
"""

import math
from typing import Dict, Optional, Tuple

import torch
import ttnn

from models.experimental.pi0_5.common.configs import GemmaConfig
from models.experimental.pi0_5.tt._ttnn_compat import concat_heads_matmul, kv_sdpa, nlp_create_qkv_heads_rope
from models.experimental.pi0_5.tt.ttnn_common import (
    get_sdpa_compute_kernel_config,
    get_sdpa_exp_approx_mode,
    sdpa_prefill_chunk_sizes,
)


# ============================================================================
# RMSNorm (TTNN - Optimized)
# ============================================================================


def rms_norm_ttnn(
    x: ttnn.Tensor,
    weight: ttnn.Tensor,
    eps: float = 1e-6,
    sharded_pcfg=None,
    sharded_memcfg=None,
) -> ttnn.Tensor:
    """
    OPTIMIZED: RMSNorm using ttnn.rms_norm fused operation.

    NOTE: The weight tensor should already have the Gemma-style +1 offset
    pre-applied during initialization (not computed here every forward pass).

    Args:
        x: TTNN tensor (batch_size, seq_len, hidden_dim)
        weight: TTNN weight tensor with +1 offset already applied (1, hidden_dim)
        eps: Epsilon for numerical stability
        sharded_pcfg / sharded_memcfg: optional sharded RMSNorm path (ViT-BH
            pattern, see _modulated_rms_norm). If both provided, input is moved
            into the block-sharded layout and rms_norm runs on the configured
            grid. Returns a block-sharded tensor — caller must convert back to
            interleaved if downstream consumer needs it.

    Returns:
        Normalized TTNN tensor (bfloat16)
    """
    if sharded_pcfg is not None and sharded_memcfg is not None:
        x_sh = ttnn.to_memory_config(x, sharded_memcfg)
        out = ttnn.rms_norm(
            x_sh,
            weight=weight,
            epsilon=eps,
            program_config=sharded_pcfg,
            memory_config=sharded_memcfg,
            compute_kernel_config=_RMS_NORM_COMPUTE_CONFIG,
        )
        if x_sh is not x:
            ttnn.deallocate(x_sh)
        return out
    return ttnn.rms_norm(
        x,
        weight=weight,
        epsilon=eps,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )


# ============================================================================
# Rotary Position Embeddings (TTNN Meta Format)
# ============================================================================


def precompute_freqs_cis_meta_format(
    head_dim: int,
    max_seq_len: int,
    device: ttnn.Device,
    base: float = 10000.0,
) -> Tuple[ttnn.Tensor, ttnn.Tensor]:
    """
    Precompute cos and sin for rotary embeddings using pure TTNN operations.

    ttnn.experimental.rotary_embedding uses the split-half pattern (same as Gemma):
    - rotate_half(x) = cat(-x[..., dim/2:], x[..., :dim/2])
    - result = x * cos + rotate_half(x) * sin

    For this to work correctly, cos/sin must have shape [1, 1, max_seq_len, head_dim]
    where the values are repeated: [c0, c1, ..., c_{n/2-1}, c0, c1, ..., c_{n/2-1}]

    This matches how the rotation pairs x[i] with x[i+dim/2] for i < dim/2.

    Args:
        head_dim: Dimension per head (must be even)
        max_seq_len: Maximum sequence length
        device: TTNN device
        base: Base for frequency computation

    Returns:
        Tuple of (cos, sin) each of shape (1, 1, max_seq_len, head_dim) as TTNN tensors
    """
    half_dim = head_dim // 2

    # Compute inverse frequencies using ttnn.arange
    # indices: [0, 2, 4, ..., head_dim-2]
    indices = ttnn.arange(0, head_dim, 2, device=device, dtype=ttnn.float32)
    # Convert to TILE_LAYOUT early (required for unary ops like pow, reciprocal, cos, sin)
    indices = ttnn.to_layout(indices, ttnn.TILE_LAYOUT)

    # freqs = 1.0 / (base ** (indices / head_dim))
    exponents = ttnn.multiply(indices, 1.0 / head_dim)
    ttnn.deallocate(indices)
    base_powers = ttnn.pow(base, exponents)
    ttnn.deallocate(exponents)
    freqs = ttnn.reciprocal(base_powers)  # Shape: [half_dim]
    ttnn.deallocate(base_powers)

    # Compute positions: [0, 1, 2, ..., max_seq_len-1]
    t = ttnn.arange(0, max_seq_len, 1, device=device, dtype=ttnn.float32)  # Shape: [max_seq_len]
    t = ttnn.to_layout(t, ttnn.TILE_LAYOUT)

    # Outer product: t[i] * freqs[j] -> [max_seq_len, half_dim]
    # Reshape for broadcasting: t -> [max_seq_len, 1], freqs -> [1, half_dim]
    t_col = ttnn.reshape(t, (max_seq_len, 1))
    ttnn.deallocate(t)
    freqs_row = ttnn.reshape(freqs, (1, half_dim))
    ttnn.deallocate(freqs)
    freqs_outer = ttnn.multiply(t_col, freqs_row)  # Shape: [max_seq_len, half_dim]
    ttnn.deallocate(t_col)
    ttnn.deallocate(freqs_row)

    # Compute cos/sin: [max_seq_len, half_dim]
    cos_half = ttnn.cos(freqs_outer)
    sin_half = ttnn.sin(freqs_outer)
    ttnn.deallocate(freqs_outer)

    # Repeat for full head_dim: [c0, c1, ..., c_{n/2-1}, c0, c1, ..., c_{n/2-1}]
    # This matches the split-half rotation where x[i] pairs with x[i+dim/2]
    cos_2d = ttnn.concat([cos_half, cos_half], dim=-1)  # [seq, head_dim]
    sin_2d = ttnn.concat([sin_half, sin_half], dim=-1)  # [seq, head_dim]
    ttnn.deallocate(cos_half)
    ttnn.deallocate(sin_half)

    # Reshape to add batch and head dimensions: [1, 1, seq, head_dim]
    cos = ttnn.reshape(cos_2d, (1, 1, max_seq_len, head_dim))
    sin = ttnn.reshape(sin_2d, (1, 1, max_seq_len, head_dim))
    ttnn.deallocate(cos_2d)
    ttnn.deallocate(sin_2d)

    # Convert to bfloat16 for use with rotary_embedding
    cos = ttnn.typecast(cos, ttnn.bfloat16)
    sin = ttnn.typecast(sin, ttnn.bfloat16)

    return cos, sin


# ============================================================================
# Shared sharded-matmul program config helper
# ============================================================================


_pcfg_cache: Dict[Tuple[int, int, int, int, int, str], object] = {}


def build_matmul_pcfg(
    m_tiles: int,
    k_tiles: int,
    n_tiles: int,
    grid_x: int,
    grid_y: int,
    *,
    in0_block_w: Optional[int] = None,
    activation=None,
    dst_budget: int = 8,
):
    """Build a sharded-matmul program config.

    For small M (suffix-side action expert: M=2 tiles), we'd waste rows of the
    2D grid (only m_tiles of grid_y rows get work → 12×2 = 24 of 120 cores).
    For these cases we return a 1D **width-sharded** config instead
    (MatmulMultiCoreReuseMultiCast1DProgramConfig with mcast_in0=True), which
    multicasts activations and spreads N across all 120 cores. That brings the
    expert MLP/QKV/o_proj from ~22-24 cores to 60-120.

    For larger M (VLM prefix: M=9 tiles, SigLIP: M=8 tiles) the 2D config
    already uses 96-108 cores, so we keep it.

    Tuned for Blackhole Galaxy (compute grid 12x10): use grid=(12,10) and
    in0_block_w=4 for large MLP matmuls (CBs fit under the ~200 KB L1 headroom
    left by trace-persistent KV cache buffers). For smaller attention matmuls,
    pass in0_block_w=8.

    Returns None if shapes don't admit a clean config.
    """
    if m_tiles == 0 or k_tiles == 0 or n_tiles == 0:
        return None
    total_cores = grid_x * grid_y

    # --- 1D width-shard path: small M, big N -----------------------------
    # When m_tiles is much smaller than grid_y, the 2D grid wastes rows.
    # Switch to 1D width-shard so all 120 cores work on N slices.
    if m_tiles * 4 <= grid_y and n_tiles >= total_cores // 4:
        # PI0_DENOISE_MM_TUNE=1 enables per-shape overrides discovered by
        # tests/perf/test_denoise_matmul_sweep.py for the 4 expert matmul
        # shapes (M=32). Wall-clock wins of -10% to -42% on the sweep;
        # device-kernel-time delta needs tracy verification before keeping
        # by default.
        import os as _os

        _tune = _os.environ.get("PI0_DENOISE_MM_TUNE", "").lower() in ("1", "true", "yes", "on")
        # PI0_MM_SWEEP_V2=1 — proposed new override table from 2026-06-05
        # sharding sweep. See [[pi05-matmul-sharding-sweep-results]] memory.
        # Production picker ranked bottom-half on all swept shapes; sweep
        # top-1 configs predict +13 to +62% wall-clock speedup (subject to
        # in-model tracy translation). These are UNVERIFIED — use only for
        # A/B testing, never default-on.
        _tune_v2 = _os.environ.get("PI0_MM_SWEEP_V2", "").lower() in ("1", "true", "yes", "on")
        if _tune and m_tiles == 1:
            # (k_tiles, n_tiles) -> (num_cores, in0_block_w)
            # Only shapes where tracy verified a real device-kernel-time WIN
            # vs the production picker. Other denoise shapes (qkv_fused,
            # mlp_gate_up) regressed when their wall-clock-sweep "winners"
            # were applied — the wall-clock proxy doesn't generalize at
            # small K. Tracy-verified wins for large K only.
            if _tune_v2:
                # Sweep-derived configs (2026-06-05). Tracy verification on the
                # v10 perf trace (52.432 ms) showed only 1 of 3 candidates
                # translated to a real device-kernel-time win:
                #   - qkv_fused: 9.04 → 8.22 µs/call (-9%, -0.147 ms) ✓ KEPT
                #   - mlp_gate_up: 12.50 → 13.62 µs/call (+0.40 ms) ✗ REVERTED
                #   - mlp_down:    13.60 → 14.41 µs/call (+0.15 ms) ✗ REVERTED
                # Same wall-clock-doesn't-translate pattern as prior sweeps.
                # Keeping the env knob so V2 stays a clean opt-in for future
                # A/B testing of additional candidates; only the verified
                # qkv_fused override is in the table by default.
                _DENOISE_TUNE_TABLE = {
                    (64, 32): (120, 32),  # o_proj:     keep verified config
                    (128, 32): (24, 32),  # mlp_down:   keep verified config
                    (32, 80): (64, 8),  # qkv_fused:  tracy-verified -9%
                }
                # BH-recalibration attempts 2026-06-05 (all REVERTED).
                # Verified against true baseline 51.023 ms (full flag set incl.
                # QWEN_NLP_*_HEAD_SPLIT=1 + PI0_UPSTREAM_MASKS=1 + PI0_LN_WEIGHTS_L1=1):
                # - qkv_fused (32,80): (80,8) — matmul +0.186 ms (40→80 active cores too granular)
                # - mlp_down (128,32): (32,32) — matmul +0.197 ms (16→32 active cores too granular)
                # - mlp_gate_up (32,128): (32,16) — matmul +0.482 ms (fewer cores raises µs/call)
                # - mlp_gate_up (32,128): (64,32) — matmul +0.04 ms (ibw=32 vs 16, neutral)
                # Conclusion: V11 picker outputs are at the BH dispatch floor; pushing toward
                # per_core_N=1 in any of these shapes worsens dispatch/work balance.
            else:
                _DENOISE_TUNE_TABLE = {
                    (64, 32): (120, 32),  # o_proj:   M=32 K=2048 N=1024 — verified -10% kernel
                    (128, 32): (24, 32),  # mlp_down: M=32 K=4096 N=1024 — verified -6% kernel
                }
            override = _DENOISE_TUNE_TABLE.get((k_tiles, n_tiles))
            if override is not None:
                tuned_cores, tuned_bw = override
                num_cores = min(tuned_cores, total_cores)
                if n_tiles % num_cores != 0:
                    per_core_N_1d = (n_tiles + num_cores - 1) // num_cores
                else:
                    per_core_N_1d = n_tiles // num_cores
                in0_bw = tuned_bw
                # Validate before applying — same checks as the regular path
                if k_tiles % in0_bw == 0:
                    # dst_budget=4 here matches the sweep winner's fp32_dest=True;
                    # the compute_kernel_config side is set in the matmul forward.
                    eff_budget = 4
                    out_sw = min(per_core_N_1d, eff_budget)
                    while out_sw > 1 and per_core_N_1d % out_sw != 0:
                        out_sw -= 1
                    out_sh = max(1, eff_budget // out_sw)
                    out_sh = min(m_tiles, out_sh)
                    while out_sh > 1 and m_tiles % out_sh != 0:
                        out_sh -= 1
                    cfg_gx = min(grid_x, num_cores)
                    cfg_gy = min(grid_y, (num_cores + cfg_gx - 1) // cfg_gx)
                    key = (m_tiles, k_tiles, n_tiles, grid_x, grid_y, str(activation), 4, "1d-tuned")
                    if key in _pcfg_cache:
                        return _pcfg_cache[key]
                    cfg = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                        compute_with_storage_grid_size=(cfg_gx, cfg_gy),
                        in0_block_w=in0_bw,
                        out_subblock_h=out_sh,
                        out_subblock_w=out_sw,
                        per_core_M=m_tiles,
                        per_core_N=per_core_N_1d,
                        fuse_batch=True,
                        fused_activation=activation,
                        mcast_in0=True,
                    )
                    _pcfg_cache[key] = cfg
                    return cfg
                # else: fall through to the default 1D logic below
        # Pick num_cores: largest divisor of n_tiles ≤ total_cores, and not
        # smaller than half the grid (otherwise stick with 2D).
        num_cores = min(total_cores, n_tiles)
        while num_cores > total_cores // 2 and n_tiles % num_cores != 0:
            num_cores -= 1
        if n_tiles % num_cores != 0:
            # No clean divisor available — fall back to ceil division on full grid
            num_cores = total_cores
            per_core_N_1d = (n_tiles + num_cores - 1) // num_cores
        else:
            per_core_N_1d = n_tiles // num_cores

        # Choose in0_block_w for 1D (full M per core → larger CBs needed).
        # 1D width-shard with small M (≤2 tiles) and small per_core_N (often 1-2)
        # has tiny CBs — we can use a much larger block_w than the 2D default of
        # 4-8 to reduce K-loop iterations. Cap at 16 to keep L1 happy when
        # per_core_N is bigger (e.g. n_tiles=128, num_cores=64, per_core_N=2).
        if in0_block_w is None:
            in0_bw = 16
        else:
            in0_bw = in0_block_w
        # Constrain by per_core_N for L1 safety: CB ~ in0_bw * per_core_N tiles.
        while in0_bw > 1 and in0_bw * per_core_N_1d > 32:
            in0_bw //= 2
        while k_tiles % in0_bw != 0 and in0_bw > 1:
            in0_bw //= 2
        if in0_bw < 2:
            in0_bw = 1

        # DST budget: out_subblock_w * out_subblock_h ≤ dst_budget.
        out_subblock_w_1d = min(per_core_N_1d, dst_budget)
        while out_subblock_w_1d > 1 and per_core_N_1d % out_subblock_w_1d != 0:
            out_subblock_w_1d -= 1
        out_subblock_h_1d = max(1, dst_budget // out_subblock_w_1d)
        out_subblock_h_1d = min(m_tiles, out_subblock_h_1d)
        while out_subblock_h_1d > 1 and m_tiles % out_subblock_h_1d != 0:
            out_subblock_h_1d -= 1

        # Grid: num_cores arranged as (grid_x, ceil(num_cores/grid_x)).
        cfg_gx = min(grid_x, num_cores)
        cfg_gy = min(grid_y, (num_cores + cfg_gx - 1) // cfg_gx)

        key = (m_tiles, k_tiles, n_tiles, grid_x, grid_y, str(activation), dst_budget, "1d")
        if key in _pcfg_cache:
            return _pcfg_cache[key]
        cfg = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=(cfg_gx, cfg_gy),
            in0_block_w=in0_bw,
            out_subblock_h=out_subblock_h_1d,
            out_subblock_w=out_subblock_w_1d,
            per_core_M=m_tiles,
            per_core_N=per_core_N_1d,
            fuse_batch=True,
            fused_activation=activation,
            mcast_in0=True,
        )
        _pcfg_cache[key] = cfg
        return cfg

    # --- 2D block-shard path (default, large M) --------------------------
    # PI0_PREFILL_MM_TUNE=1 enables per-shape overrides from the sweep at
    # tests/perf/test_prefill_matmul_sweep.py. Same caution applies as the
    # denoise sweep: wall-clock numbers are a proxy that needs tracy
    # verification before keeping by default. Each entry was tracy-verified
    # to give a real device-kernel-time win.
    import os as _os

    _tune2d = _os.environ.get("PI0_PREFILL_MM_TUNE", "").lower() in ("1", "true", "yes", "on")
    if _tune2d:
        # (m_tiles, k_tiles, n_tiles) -> (grid_x, grid_y, in0_block_w)
        # Includes all 2D-path shapes the sweep flagged with wall-clock wins;
        # the tracy verifier filters this down to the real wins.
        # Only entries where tracy verified a real device-kernel-time win
        # over the production picker. Sweep wall-clock predictions for the
        # other shapes (vlm_qkv_fused, vlm_o_proj, siglip_*) either showed
        # noise (±0.007 ms total) or didn't match real production shapes
        # (e.g. the sweep's "siglip_attn_proj" at K=N=1152 doesn't exist —
        # production SigLIP Q/K/V are K=1152 N=1536 since head_dim=72
        # padded to 96 × num_heads=16 = 1536).
        #
        # The gate_up override (16, 64, 512) was tried with bw=8 and trips
        # a runtime CB-clash in production (clean-L1 sweep didn't catch it).
        _PREFILL_TUNE_TABLE = {
            (16, 512, 64): (12, 8, 16),  # vlm_mlp_down: M=512 K=16384 N=2048
            # Tracy-verified: 3.254 -> 2.956 ms (-0.298 ms)
        }
        override = _PREFILL_TUNE_TABLE.get((m_tiles, k_tiles, n_tiles))
        if override is not None:
            tg_x, tg_y, tg_bw = override
            if k_tiles % tg_bw == 0:
                per_core_M_t = (m_tiles + tg_y - 1) // tg_y
                per_core_N_t = (n_tiles + tg_x - 1) // tg_x
                if per_core_M_t > 0 and per_core_N_t > 0:
                    eff_budget = 4  # matches fp32_dest=True (the common sweep winner)
                    sub_w = min(per_core_N_t, eff_budget)
                    while sub_w > 1 and per_core_N_t % sub_w != 0:
                        sub_w -= 1
                    sub_h = max(1, eff_budget // sub_w)
                    sub_h = min(per_core_M_t, sub_h)
                    while sub_h > 1 and per_core_M_t % sub_h != 0:
                        sub_h -= 1
                    key = (m_tiles, k_tiles, n_tiles, grid_x, grid_y, str(activation), 4, "2d-tuned")
                    if key in _pcfg_cache:
                        return _pcfg_cache[key]
                    cfg = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                        compute_with_storage_grid_size=(tg_x, tg_y),
                        in0_block_w=tg_bw,
                        out_subblock_h=sub_h,
                        out_subblock_w=sub_w,
                        per_core_M=per_core_M_t,
                        per_core_N=per_core_N_t,
                        transpose_mcast=False,
                        fused_activation=activation,
                    )
                    _pcfg_cache[key] = cfg
                    return cfg
                # else fall through to default

    per_core_M = (m_tiles + grid_y - 1) // grid_y
    per_core_N = (n_tiles + grid_x - 1) // grid_x
    if per_core_M == 0 or per_core_N == 0:
        return None

    # in0_block_w must divide K_tiles. SigLIP MLP has K=4304 padded to 4320
    # (135 tiles) which doesn't divide cleanly into 4 or 2 — return None so
    # caller falls back to the default core_grid path.
    if in0_block_w is None:
        # Adaptive choice: per_core_N drives CB size. block_w=8 is fastest
        # but overflows L1 when per_core_N is large (CB ≈ block_w * per_core_N
        # tiles, doubled for double-buffer). With pi0's trace KV pinning
        # ~1.3 MB of L1, we have ~150 KB CB headroom per core.
        # block_w=8 cap: per_core_N * 8 * 1024 * 2 ≤ 150 KB → per_core_N ≤ 9
        if per_core_N <= 12:
            in0_block_w = 8
        else:
            in0_block_w = 4
    while k_tiles % in0_block_w != 0 and in0_block_w > 1:
        in0_block_w //= 2
    if in0_block_w == 1 and k_tiles > 32:
        # Tiny block_w on a large K is unlikely to win — bail out.
        return None

    key = (m_tiles, k_tiles, n_tiles, grid_x, grid_y, str(activation), dst_budget)
    if key in _pcfg_cache:
        return _pcfg_cache[key]

    # out_subblock_w * out_subblock_h <= dst_budget (DST register tile budget).
    # fp32_dest_acc_en=True halves the budget to 4. bf16 dest accum allows 8.
    out_subblock_w = min(per_core_N, dst_budget)
    while out_subblock_w > 1 and per_core_N % out_subblock_w != 0:
        out_subblock_w -= 1
    out_subblock_h_budget = max(1, dst_budget // out_subblock_w)
    out_subblock_h = min(per_core_M, out_subblock_h_budget)
    while out_subblock_h > 1 and per_core_M % out_subblock_h != 0:
        out_subblock_h -= 1

    cfg = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=(grid_x, grid_y),
        in0_block_w=in0_block_w,
        out_subblock_h=out_subblock_h,
        out_subblock_w=out_subblock_w,
        per_core_M=per_core_M,
        per_core_N=per_core_N,
        transpose_mcast=False,
        fused_activation=activation,
    )
    _pcfg_cache[key] = cfg
    return cfg


# ============================================================================
# Multi-Query Attention (TTNN - Optimized)
# ============================================================================


class GemmaAttentionTTNN:
    """
    Gemma Multi-Query Attention using TTNN operations.

    OPTIMIZED:
    1. Fused QKV projection (1 linear instead of 3)
    2. Native ttnn.experimental.nlp_create_qkv_heads
    3. Native ttnn.experimental.rotary_embedding (split-half pattern)
    4. Native ttnn.experimental.nlp_concat_heads for output
    """

    def __init__(
        self,
        config: GemmaConfig,
        weights: Dict[str, ttnn.Tensor],
        layer_idx: int,
        device: ttnn.Device,
        cos_meta: Optional[ttnn.Tensor] = None,
        sin_meta: Optional[ttnn.Tensor] = None,
    ):
        """
        Initialize attention layer with TTNN weights.

        Args:
            config: Gemma configuration
            weights: TTNN weight tensors (including fused wqkv)
            layer_idx: Layer index
            device: TTNN device
            cos_meta: Precomputed cos for native TTNN RoPE [1, 1, max_seq, head_dim]
            sin_meta: Precomputed sin for native TTNN RoPE [1, 1, max_seq, head_dim]
        """
        self.config = config
        self.layer_idx = layer_idx
        self.device = device

        # Query device grid to use all available cores (P150: up to 13x10, N150: 8x8)
        device_grid = device.compute_with_storage_grid_size()
        self.grid_size = (device_grid.x, device_grid.y)
        self.core_grid = ttnn.CoreGrid(y=device_grid.y, x=device_grid.x)

        # OPTIMIZATION: Use fused QKV weight (single linear instead of 3)
        self.wqkv = weights["self_attn.wqkv"]
        self.o_proj = weights["self_attn.o_proj.weight"]

        self.num_heads = config.num_heads
        self.num_kv_heads = config.num_kv_heads
        self.head_dim = config.head_dim
        self.hidden_size = config.width
        self.scale = 1.0 / math.sqrt(self.head_dim)

        # Store meta format cos/sin for native TTNN RoPE (split-half pattern)
        self.cos_meta = cos_meta
        self.sin_meta = sin_meta

        # HiFi2 config for projections (faster, less precision needed).
        # PI0_EXPERT_MM_LOFI=1 drops fidelity to LoFi for the expert matmul
        # path (PERF_PLAYBOOKS/05 §6: BGE-M3 saw -6 ms FF1, -2 ms FF2 on
        # bf8b weights from this walk). PCC must be verified end-to-end
        # via LIBERO rollouts before promoting default. fp32_dest_acc_en
        # stays False to keep the subblock cap at 8 (01 §3).
        import os as _os

        _expert_lofi = _os.environ.get("PI0_EXPERT_MM_LOFI", "").lower() in ("1", "true", "yes", "on")
        self.compute_kernel_config_hifi2 = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.LoFi if _expert_lofi else ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=True,
        )

        # SDPA compute config — env-controllable for A/B testing. See ttnn_common.py.
        # Default = HiFi2 + fp32_dest_acc=True + packer_l1_acc=True. Bump
        # PI0_SDPA_HIFI=4 to match ViT-BH-hiRes precision regime.
        self.compute_kernel_config_sdpa = get_sdpa_compute_kernel_config()

    def forward(
        self,
        hidden_states: ttnn.Tensor,
        cos: ttnn.Tensor,
        sin: ttnn.Tensor,
        attention_mask: Optional[ttnn.Tensor] = None,
        position_ids: Optional[ttnn.Tensor] = None,
        past_key_value: Optional[Tuple[ttnn.Tensor, ttnn.Tensor]] = None,
        use_cache: bool = False,
        keep_padded: bool = False,
        *,
        bs_norm_factory=None,
        bs_grid: Optional[Tuple[int, int]] = None,
    ) -> Tuple[ttnn.Tensor, Optional[Tuple[ttnn.Tensor, ttnn.Tensor]]]:
        """
        OPTIMIZED forward pass using fused QKV and native TTNN operations.

        Key optimizations:
        1. Single fused QKV linear (3x fewer linear ops)
        2. Native ttnn.experimental.nlp_create_qkv_heads
        3. Native ttnn.experimental.rotary_embedding (split-half pattern)
        4. Native ttnn.experimental.nlp_concat_heads for output

        Args:
            hidden_states: TTNN tensor (batch, seq_len, hidden_dim)
            cos, sin: Unused (kept for API compatibility, native RoPE uses self.cos_meta/sin_meta)
            attention_mask: Attention mask
            position_ids: Position indices
            past_key_value: Cached KV
            keep_padded: When True, do not slice q_rope/k_rope back to logical
                seq_len after rotary_embedding. Used by the expert path where the
                suffix is already tile-aligned (logical=physical=64) and the SDPA
                mask handles phantom positions. Reapplied from reverted commit
                3d597a3b8e6 with the FINITE-mask hybrid fix (-1e4 vs -inf) per
                the revert message's "TIER B will be revisited" note.
            use_cache: Whether to return cache

        Returns:
            Tuple of (output, optional_cache)
        """
        batch_size = hidden_states.shape[0]
        if len(hidden_states.shape) == 4:
            seq_len = hidden_states.shape[2]
        else:
            seq_len = hidden_states.shape[1]
            hidden_states = ttnn.reshape(hidden_states, (batch_size, 1, seq_len, -1))

        # OPTIMIZATION 1: Single fused QKV linear (instead of 3 separate)
        # Output: [batch, 1, seq, Q_dim + K_dim + V_dim]
        m_tiles = (batch_size * seq_len) // 32
        k_tiles_in = self.hidden_size // 32
        n_tiles_qkv = self.wqkv.shape[-1] // 32
        use_bs_qkv = bs_norm_factory is not None and bs_grid is not None
        if use_bs_qkv:
            gx, gy = bs_grid
            if not bs_matmul_divisible(m_tiles, k_tiles_in, n_tiles_qkv, gx, gy):
                hidden_states = ttnn.sharded_to_interleaved(hidden_states, memory_config=ttnn.L1_MEMORY_CONFIG)
                use_bs_qkv = False
            else:
                bs_mc_qkv = make_bs_memcfg(batch_size, seq_len, self.wqkv.shape[-1], gx, gy)
                qkv_pcfg = build_bs_matmul_pcfg(m_tiles, k_tiles_in, n_tiles_qkv, gx, gy, dst_budget=4)
                xqkv = ttnn.linear(
                    hidden_states,
                    self.wqkv,
                    dtype=ttnn.bfloat8_b,
                    memory_config=bs_mc_qkv,
                    compute_kernel_config=self.compute_kernel_config_hifi2,
                    program_config=qkv_pcfg,
                )
                xqkv = ttnn.sharded_to_interleaved(xqkv, memory_config=ttnn.L1_MEMORY_CONFIG)

        if not use_bs_qkv:
            m_tiles_interleaved = (seq_len + 31) // 32
            wqkv_pcfg = build_matmul_pcfg(
                m_tiles_interleaved, k_tiles_in, n_tiles_qkv, self.grid_size[0], self.grid_size[1], in0_block_w=8
            )

            if wqkv_pcfg is not None:
                xqkv = ttnn.linear(
                    hidden_states,
                    self.wqkv,
                    dtype=ttnn.bfloat8_b,
                    memory_config=ttnn.L1_MEMORY_CONFIG,
                    compute_kernel_config=self.compute_kernel_config_hifi2,
                    program_config=wqkv_pcfg,
                )
            else:
                xqkv = ttnn.linear(
                    hidden_states,
                    self.wqkv,
                    dtype=ttnn.bfloat8_b,
                    memory_config=ttnn.L1_MEMORY_CONFIG,
                    compute_kernel_config=self.compute_kernel_config_hifi2,
                    core_grid=self.core_grid,
                )

        # OPTIMIZATION 2: Native TTNN head splitting (no PyTorch transfers!)
        # This splits the fused QKV into separate Q, K, V with proper head layout
        # Output shapes: q=[batch, num_heads, seq, head_dim], k/v=[batch, num_kv_heads, seq, head_dim]
        #
        # MQA-bypass experiment (DON'T re-try without first reading this note):
        # When num_kv_heads=1 and M-tile=2 (the expert: action_horizon=50
        # padded to 64 → 2 tiles), this op's work-distribution heuristic
        # caps dispatch at 2 cores even with QWEN_NLP_CREATE_HEADS_HEAD_SPLIT=1
        # (the multiplier is num_kv_heads, which is 1, so head-split is a
        # no-op for MQA). At 180 calls/chunk (18 expert layers × 10 denoise
        # steps) that "looks" wasteful in the visualizer.
        #
        # I tried bypassing with slice(Q)+slice(K)+slice(V)+reshape(Q)+
        # transpose(Q) (K, V are already in the right layout once sliced).
        # Result: +6 ms/chunk regression (61 → 67 ms perf-trace baseline).
        # Tracy breakdown (run_mqa_bypass_l1c vs baseline run2):
        #   - Saved: -198× NlpCreateHeads, -4788 µs device
        #   - Added: +198× ReshapeView (~36 µs/call, 7287 µs total),
        #            +198× Transpose (1521 µs),
        #            +594× Slice (1071 µs),
        #            +792× Typecast fp32→bf16 (3529 µs, auto-inserted
        #                  because ReshapeView promotes bf8_b→fp32 internally)
        #   - Net: +8.6 ms device-side
        # Pinning memory_config=L1_MEMORY_CONFIG on every intermediate shrank
        # op-to-op latency 5-47× but DIDN'T touch device compute — the
        # ReshapeView cost is from physically restitching tiles whose inner
        # boundaries change ((B,1,M,H*D)→(B,M,H,D)), not from buffer location.
        #
        # Net: the fused nlp_create_qkv_heads at 2 cores beats the
        # decomposed slice+reshape+transpose chain at 120 cores. Don't undo
        # this without an upstream TTNN fix to either:
        #   (a) make ReshapeView on tile-layout bf8_b not promote to fp32, or
        #   (b) extend nlp_create_qkv_heads to parallelize across num_q_heads
        #       (not just num_kv_heads × M-tile).
        q, k, v = ttnn.experimental.nlp_create_qkv_heads(
            xqkv,
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            transpose_k_heads=False,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        # NOTE: do NOT deallocate xqkv — nlp_create_qkv_heads may view-alias
        # into its input buffer rather than allocating fresh q/k/v tensors.
        # Tried `ttnn.deallocate(xqkv)` here in commit attempt 2026-06-04 22:30
        # and it crashed with "Tensor is not allocated" at trace replay.

        # OPTIMIZATION 3: Apply RoPE using native TTNN (split-half pattern).
        # If the caller passed in `cos`/`sin` overrides (position-aware tables
        # gathered at cumsum-based or offset positions for upstream-openpi
        # compat), use them directly. Otherwise fall back to slicing the
        # default cos_meta/sin_meta at sequential positions [0..seq_len-1].
        if cos is not None and sin is not None:
            # Caller-supplied: already shaped [1, 1, seq_len, head_dim]
            # at the right positions; no slice + no deallocate (caller owns them).
            cos_for_rope = cos
            sin_for_rope = sin
            _own_rope_tensors = False
        else:
            cos_for_rope = ttnn.slice(
                self.cos_meta,
                [0, 0, 0, 0],
                [1, 1, seq_len, self.head_dim],
            )
            sin_for_rope = ttnn.slice(
                self.sin_meta,
                [0, 0, 0, 0],
                [1, 1, seq_len, self.head_dim],
            )
            _own_rope_tensors = True

        # ttnn.experimental.rotary_embedding uses split-half pattern like Gemma
        # NOTE: do NOT deallocate q/k after RoPE — rotary_embedding may alias
        # its input. Same caveat applies to q_rope_padded after slice. Both
        # were tried in commit attempt 2026-06-04 22:30 → "Tensor is not
        # allocated" at trace replay.
        q_rope_padded = ttnn.experimental.rotary_embedding(q, cos_for_rope, sin_for_rope)
        k_rope_padded = ttnn.experimental.rotary_embedding(k, cos_for_rope, sin_for_rope)

        if _own_rope_tensors:
            ttnn.deallocate(cos_for_rope)
            ttnn.deallocate(sin_for_rope)

        if keep_padded:
            # Expert/suffix fast path: q/k are already tile-aligned (logical=physical),
            # SDPA mask handles phantom positions. Skipping the slice removes the
            # UntilizeWithUnpadding ops per chunk along the suffix-attention path.
            q_rope = q_rope_padded
            k_rope = k_rope_padded
        else:
            # rotary_embedding pads output to tile boundary, slice back to original seq_len
            q_rope = ttnn.slice(q_rope_padded, [0, 0, 0, 0], [batch_size, self.num_heads, seq_len, self.head_dim])
            k_rope = ttnn.slice(k_rope_padded, [0, 0, 0, 0], [batch_size, self.num_kv_heads, seq_len, self.head_dim])

        # Handle KV cache. Force concat output to L1 — the result feeds SDPA
        # directly so keeping it on-chip avoids DRAM round-trips.
        if past_key_value is not None:
            past_k, past_v = past_key_value
            k_rope = ttnn.concat([past_k, k_rope], dim=2, memory_config=ttnn.L1_MEMORY_CONFIG)
            v = ttnn.concat([past_v, v], dim=2, memory_config=ttnn.L1_MEMORY_CONFIG)

        new_cache = (k_rope, v) if use_cache else None

        # SDPA chunk sizes aligned with models/tt_transformers/tt/model_config.py prefill defaults
        kv_seq_len = k_rope.shape[2]
        q_chunk, k_chunk = sdpa_prefill_chunk_sizes(seq_len, kv_seq_len)

        sdpa_cfg = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=self.grid_size,
            q_chunk_size=q_chunk,
            k_chunk_size=k_chunk,
            exp_approx_mode=get_sdpa_exp_approx_mode(kv_seq_len),
        )

        attn_output = ttnn.transformer.scaled_dot_product_attention(
            q_rope,
            k_rope,
            v,
            attn_mask=attention_mask,
            is_causal=False,
            scale=self.scale,
            program_config=sdpa_cfg,
            compute_kernel_config=self.compute_kernel_config_sdpa,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        # NOTE: do not deallocate q_rope here — it may alias q_rope_padded
        # (and thus q for keep_padded=True). Tried it 2026-06-04 22:30 → crash.

        # OPTIMIZATION 4: Native TTNN head concatenation (no PyTorch transfers!)
        # attn_output: [batch, num_heads, seq, head_dim] -> [batch, 1, seq, num_heads * head_dim]
        attn_concat = ttnn.experimental.nlp_concat_heads(
            attn_output,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        # NOTE: do not deallocate attn_output — nlp_concat_heads may alias.

        # Output projection — 2D BLOCK_SHARDED program config.
        # K = num_heads*head_dim (concatenated heads), N = hidden_size.
        oproj_k = (self.num_heads * self.head_dim) // 32
        oproj_n = self.hidden_size // 32
        oproj_pcfg = build_matmul_pcfg(m_tiles, oproj_k, oproj_n, self.grid_size[0], self.grid_size[1], in0_block_w=8)
        # PI0_VLM_MLP_BF8_OUT=1 also controls the attention O-proj output dtype.
        # Saves ~17 KB / core (33 KB bf16 → 16 KB bf8) at chunk=1024 — needed to
        # close the ~32 KB CB-clash gap that blocks chunk=1024 at LIBERO bs=3.
        # Same PCC-safety expectation as MLP gate/up outputs (validated 0.9964 PCC).
        import os as _os_oproj

        _oproj_dtype = (
            ttnn.bfloat8_b
            if _os_oproj.environ.get("PI0_VLM_MLP_BF8_OUT", "").lower() in ("1", "true", "yes", "on")
            else ttnn.bfloat16
        )
        if oproj_pcfg is not None:
            output = ttnn.linear(
                attn_concat,
                self.o_proj,
                dtype=_oproj_dtype,
                memory_config=ttnn.L1_MEMORY_CONFIG,
                compute_kernel_config=self.compute_kernel_config_hifi2,
                program_config=oproj_pcfg,
            )
        else:
            output = ttnn.linear(
                attn_concat,
                self.o_proj,
                dtype=_oproj_dtype,
                memory_config=ttnn.L1_MEMORY_CONFIG,
                compute_kernel_config=self.compute_kernel_config_hifi2,
                core_grid=self.core_grid,
            )
        # NOTE: linear typically allocates a new output, so attn_concat
        # would be safe to free here in principle — but skipping until
        # we have a way to test aliasing semantics op-by-op.

        # Reshape back to 3D: [batch, 1, seq, hidden] -> [batch, seq, hidden]
        output = ttnn.reshape(output, (batch_size, seq_len, self.hidden_size))

        return output, new_cache


class AdaRMSExpertAttentionTTNN(GemmaAttentionTTNN):
    """Fused-only attention for the pi0.5 action-expert (denoise) path.

    Overrides forward() with the fused-op sequence (commit a63765d8fd) and carries NO unfused
    fallback: create-qkv-heads + q/k RoPE fuse into ``nlp_create_qkv_heads_rope`` (1 dispatch), and
    concat-heads + O-projection fuse into ``concat_heads_matmul`` (1 dispatch, bf16 out). Used ONLY
    by the expert block (AdaRMSGemmaBlockTTNN); the shared GemmaAttentionTTNN (VLM prefill) keeps its
    own unfused path because concat_heads_matmul is valid only for a <=1-tile suffix (the 32-token
    denoise suffix) — the 1024-token prefill would be incorrect.
    """

    def forward(
        self,
        hidden_states: ttnn.Tensor,
        cos: ttnn.Tensor,
        sin: ttnn.Tensor,
        attention_mask: Optional[ttnn.Tensor] = None,
        position_ids: Optional[ttnn.Tensor] = None,
        past_key_value: Optional[Tuple[ttnn.Tensor, ttnn.Tensor]] = None,
        use_cache: bool = False,
        keep_padded: bool = False,
        *,
        bs_norm_factory=None,
        bs_grid: Optional[Tuple[int, int]] = None,
    ) -> Tuple[ttnn.Tensor, Optional[ttnn.Tensor]]:
        batch_size = hidden_states.shape[0]
        if len(hidden_states.shape) == 4:
            seq_len = hidden_states.shape[2]
        else:
            seq_len = hidden_states.shape[1]
            hidden_states = ttnn.reshape(hidden_states, (batch_size, 1, seq_len, -1))
        # The fused concat_heads_matmul O-projection is only valid for a <=1-tile suffix (contiguous
        # head tiles). Fail loudly rather than silently mis-compute if used outside that regime.
        assert seq_len <= 32, (
            f"AdaRMSExpertAttentionTTNN requires suffix seq_len<=32 (1 tile), got {seq_len}. "
            f"action_horizon>32 is unsupported on the fused expert path."
        )

        # Fused QKV linear (bf8_b), interleaved path (the expert never passes bs_norm_factory).
        m_tiles = (batch_size * seq_len) // 32
        k_tiles_in = self.hidden_size // 32
        n_tiles_qkv = self.wqkv.shape[-1] // 32
        m_tiles_interleaved = (seq_len + 31) // 32
        wqkv_pcfg = build_matmul_pcfg(
            m_tiles_interleaved, k_tiles_in, n_tiles_qkv, self.grid_size[0], self.grid_size[1], in0_block_w=8
        )
        if wqkv_pcfg is not None:
            xqkv = ttnn.linear(
                hidden_states,
                self.wqkv,
                dtype=ttnn.bfloat8_b,
                memory_config=ttnn.L1_MEMORY_CONFIG,
                compute_kernel_config=self.compute_kernel_config_hifi2,
                program_config=wqkv_pcfg,
            )
        else:
            xqkv = ttnn.linear(
                hidden_states,
                self.wqkv,
                dtype=ttnn.bfloat8_b,
                memory_config=ttnn.L1_MEMORY_CONFIG,
                compute_kernel_config=self.compute_kernel_config_hifi2,
                core_grid=self.core_grid,
            )

        # RoPE tables: caller-supplied position-aware override, else sliced sequential meta.
        if cos is not None and sin is not None:
            cos_for_rope, sin_for_rope, own_rope = cos, sin, False
        else:
            cos_for_rope = ttnn.slice(self.cos_meta, [0, 0, 0, 0], [1, 1, seq_len, self.head_dim])
            sin_for_rope = ttnn.slice(self.sin_meta, [0, 0, 0, 0], [1, 1, seq_len, self.head_dim])
            own_rope = True

        # Fused create-qkv-heads + q/k RoPE in ONE dispatch (byte-identical to nlp_create_qkv_heads
        # + 2x rotary_embedding, PCC ~1.0). Output is roped, head-split, and tile-aligned.
        q_rope, k_rope, v = nlp_create_qkv_heads_rope(
            xqkv, cos_for_rope, sin_for_rope, self.num_heads, self.num_kv_heads, memory_config=ttnn.L1_MEMORY_CONFIG
        )
        if own_rope:
            ttnn.deallocate(cos_for_rope)
            ttnn.deallocate(sin_for_rope)

        # Expert SDPA is always the specialized fused-flash ttnn.kv_sdpa (the same op the L1 decode_all
        # path uses — see tt_pipeline/denoise_block.py). It honors attn_mask (additive bf16 mask over the
        # full folded KV) and, when prefix-KV is present and we are NOT caching, folds past_k/past_v into
        # its reader as two ranges so the two ttnn.concat ops are skipped. The op requires the small-query
        # MQA shape: Sq == 1 tile (== 32, already asserted above) and a single KV head. compute_kernel_config
        # is passed so the flash accumulation runs at fp32_dest_acc (HiFi2) — kv_sdpa's own default is bf16
        # dest accumulation, which loses expert-attention precision.
        assert int(q_rope.shape[-2]) == 32, f"kv_sdpa requires Sq == 32 (1 tile); got {int(q_rope.shape[-2])}"
        assert int(self.num_kv_heads) == 1, f"kv_sdpa requires num_kv_heads == 1 (MQA); got {self.num_kv_heads}"
        if past_key_value is not None and not use_cache:
            # Fold path: kv_sdpa reads past_k/past_v + suffix k/v as two KV ranges — no pre-concat.
            past_k, past_v = past_key_value
            attn_output = kv_sdpa(
                q_rope,
                k_rope,
                v,
                attn_mask=attention_mask,
                scale=self.scale,
                past_k=past_k,
                past_v=past_v,
                compute_kernel_config=self.compute_kernel_config_sdpa,
            )
            new_cache = None
        else:
            # use_cache (or no prefix): materialize the full KV so new_cache holds [prefix ; suffix].
            if past_key_value is not None:
                past_k, past_v = past_key_value
                k_rope = ttnn.concat([past_k, k_rope], dim=2, memory_config=ttnn.L1_MEMORY_CONFIG)
                v = ttnn.concat([past_v, v], dim=2, memory_config=ttnn.L1_MEMORY_CONFIG)
            new_cache = (k_rope, v) if use_cache else None
            attn_output = kv_sdpa(
                q_rope,
                k_rope,
                v,
                attn_mask=attention_mask,
                scale=self.scale,
                compute_kernel_config=self.compute_kernel_config_sdpa,
            )

        # Fused concat-heads + O-projection in ONE dispatch (bf16 out so it can feed the bf16 fused
        # addcmul gated residual). Uses the tuned 1D-mcast O-matmul program config.
        oproj_k = (self.num_heads * self.head_dim) // 32
        oproj_n = self.hidden_size // 32
        oproj_pcfg = build_matmul_pcfg(m_tiles, oproj_k, oproj_n, self.grid_size[0], self.grid_size[1], in0_block_w=8)
        output = concat_heads_matmul(
            attn_output, self.o_proj, memory_config=ttnn.L1_MEMORY_CONFIG, program_config=oproj_pcfg
        )
        output = ttnn.reshape(output, (batch_size, seq_len, self.hidden_size))
        return output, new_cache


# ============================================================================
# GeGLU MLP (TTNN)
# ============================================================================


class GemmaMLPTTNN:
    """
    Gemma MLP with GeGLU activation using TTNN.

    Uses chunking along sequence dimension combined with auto L1 sharding
    to fit large intermediate tensors (mlp_dim=16384) in L1 memory.

    Strategy:
    - Chunk input along sequence dimension (e.g., 544 → 3 chunks of 256)
    - Let matmul auto-compute optimal sharding for L1
    - Subsequent ops inherit the sharding from matmul output
    - Accumulate results in L1, concatenate at end
    """

    def __init__(
        self,
        config: GemmaConfig,
        weights: Dict[str, torch.Tensor],
        device: ttnn.Device,
        force_bf16_out: bool = False,
    ):
        """
        Initialize MLP with weights.

        Args:
            config: Gemma configuration
            weights: PyTorch weight tensors (will be converted to TTNN)
            device: TTNN device
            force_bf16_out: emit bf16 gate/up/down outputs (ignoring PI0_VLM_MLP_BF8_OUT). Set by the
                action-expert (denoise) block so the bf16 mlp_output feeds the fused addcmul gated
                residual. The tiny denoise suffix has no L1 pressure, so the bf8 footprint saving
                (which matters for the 1024-token VLM prefill) is unneeded here.
        """
        self.config = config
        self.device = device
        self._force_bf16_out = force_bf16_out

        # Convert weights to TTNN as BF8_B for 2x DRAM bandwidth savings
        def to_ttnn(w):
            if isinstance(w, torch.Tensor):
                return ttnn.from_torch(
                    w.T.contiguous(),
                    dtype=ttnn.bfloat8_b,
                    layout=ttnn.TILE_LAYOUT,
                    device=device,
                )
            return w

        self.gate_proj = to_ttnn(weights["mlp.gate_proj.weight"])
        self.up_proj = to_ttnn(weights["mlp.up_proj.weight"])
        self.down_proj = to_ttnn(weights["mlp.down_proj.weight"])
        self.hidden_size = config.width
        self.intermediate_size = config.mlp_dim

        # PI0_DRAM_SHARDED_MLP_DOWN=1 — DRAM width-sharded weight variant for
        # down_proj. Per PERF_PLAYBOOKS/05 §3b + 08 §2 (decode matmul recipe).
        # Scoped to expert MLP (mlp_dim ≤ 4096) — VLM (mlp_dim=16384) doesn't
        # benefit from this pattern at its larger M.
        import os as _os
        from .ttnn_common import build_dram_width_sharded_memcfg

        self.down_proj_dram_sharded = None
        self.down_proj_dram_sharded_padded_n = None
        self.down_proj_dram_sharded_dram_cores = None
        # =====================================================================
        # ⚠ DO NOT ENABLE FOR PRODUCTION ⚠
        # The DRAM-width-sharded mlp_down path was empirically VERIFIED to
        # regress total kernel time on the pi0.5 denoise expert shape
        # (M=32 K=4096 N=1024) at every compute-core count tested. See
        # [[pi05-dram-sharded-mlp-down-attempt]] memory for the full data.
        #
        # Quick summary of why this is parked, not deleted:
        #   - The matmul itself does win (~-0.3 ms over 180 calls at 8c).
        #   - But the per-call activation I2S reshard costs ~+1.88 ms.
        #   - BH has only 8 DRAM banks (not 12 — that's Wormhole), so the
        #     bandwidth ceiling is half of what the reference benchmark had.
        #   - Our shape is too small for the L1-handoff workaround
        #     (gate/up matmuls would need to drop to 8 compute cores, which
        #     adds more time than mlp_down saves).
        #
        # KEPT AS SCAFFOLDING because the infrastructure (helper, runtime
        # branch, isolated test) is reusable for any future experiment on
        # larger shapes (e.g. VLM mlp_down on Wormhole, or pi0.5 if shape
        # changes). Default OFF; the flag is intentionally opt-in only.
        # =====================================================================
        # PI0_DRAM_SHARDED_MLP_DOWN values: 1|true|yes|on|8 → 8 banks (no N-pad);
        # 12 → 12 banks (N-padded to 1152, only works on hardware with ≥12 banks).
        # Empty/0/false → disabled.
        _flag = _os.environ.get("PI0_DRAM_SHARDED_MLP_DOWN", "")
        _enabled = _flag.lower() in ("1", "true", "yes", "on", "8", "12")
        if _enabled:
            if self.intermediate_size <= 4096:
                # K = mlp_dim, N = hidden_size for down_proj (after transpose).
                # PI0_DRAM_SHARDED_MLP_DOWN value selects bank count:
                #   "1" / "8"  → dram_cores=8 (no N-padding, 67% BW; compute=8)
                #   "12"       → dram_cores=12 (N=1024→1152 padding, 100% BW; compute=4)
                # Both keep clean K/N divisibility for our shape.
                _banks = 12 if _flag == "12" else 8
                k_dim = self.intermediate_size
                n_dim = self.hidden_size
                memcfg, padded_n, dram_cores_used = build_dram_width_sharded_memcfg(
                    device, k_dim, n_dim, dram_cores=_banks
                )
                w_raw = weights["mlp.down_proj.weight"]
                if isinstance(w_raw, torch.Tensor):
                    # VLM path: weight is host torch tensor. Pad + upload sharded.
                    w_torch = w_raw.T.contiguous()
                    if padded_n > n_dim:
                        pad_cols = padded_n - n_dim
                        w_torch = torch.nn.functional.pad(w_torch, (0, pad_cols), mode="constant", value=0.0)
                    self.down_proj_dram_sharded = ttnn.from_torch(
                        w_torch,
                        dtype=ttnn.bfloat8_b,
                        layout=ttnn.TILE_LAYOUT,
                        device=device,
                        memory_config=memcfg,
                    )
                else:
                    # Expert path: weight already on-device (DRAM interleaved).
                    # For dram_cores=12 + N=1024, need to pad on-device first.
                    w_to_shard = self.down_proj
                    if padded_n > n_dim:
                        # self.down_proj shape: [..., K, N]; pad last dim
                        # ttnn.pad takes per-dim (low, high) tuples
                        pad_low = (0,) * len(self.down_proj.shape)
                        pad_high = tuple(
                            (padded_n - n_dim) if i == len(self.down_proj.shape) - 1 else 0
                            for i in range(len(self.down_proj.shape))
                        )
                        w_to_shard = ttnn.pad(self.down_proj, padding=list(zip(pad_low, pad_high)), value=0.0)
                    self.down_proj_dram_sharded = ttnn.to_memory_config(w_to_shard, memcfg)
                self.down_proj_dram_sharded_padded_n = padded_n
                self.down_proj_dram_sharded_dram_cores = dram_cores_used

        # Query device grid to size chunks for available cores
        device_grid = device.compute_with_storage_grid_size()
        self.grid_size = (device_grid.x, device_grid.y)
        self.core_grid = ttnn.CoreGrid(y=device_grid.y, x=device_grid.x)
        num_cores = device_grid.x * device_grid.y

        # Chunk size must be tile-aligned (multiple of 32). Scale with core count.
        #
        # P150 / BH (>=100 cores): default 768. pi0.5 LIBERO production is bs=2
        # single-arm (base + wrist; the 3rd zero-padded right_wrist slot from the
        # bimanual training convention is masked off in single-arm — see
        # [[pi05-siglip-bs3-production]]). Prefix at bs=2 = 2·256 + ≤256 lang ≤ 768
        # tokens, so 768 fits in one chunk. Single-pass VLM saves -7.7 ms tracy-
        # verified at bs=2 (vs the prior 544-chunk default which forced two
        # 18-layer MLP dispatches at 544 + 224 = 768).
        # At bs=3 (PI0_NUM_CAMERAS=3, prefix=1024) this chunks as 768 + 256.
        # Override via PI0_VLM_CHUNK_SIZE to test other values:
        #   1024 → single-pass at bs=3 (CB clash today; see OPEN_ISSUE_MLP_CB_CLASH.md)
        #   544  → previous default (preserves bs=1 production behavior)
        #
        # N150 (64 cores): 256, unchanged.
        import os as _os

        _user_chunk = _os.environ.get("PI0_VLM_CHUNK_SIZE", "").strip()
        if _user_chunk:
            try:
                self.chunk_size = int(_user_chunk)
                assert self.chunk_size % 32 == 0, f"PI0_VLM_CHUNK_SIZE={self.chunk_size} must be tile-aligned (mod 32)"
            except (ValueError, AssertionError) as e:
                raise RuntimeError(f"Invalid PI0_VLM_CHUNK_SIZE={_user_chunk!r}: {e}") from e
        elif num_cores >= 100:
            self.chunk_size = 768
        else:
            self.chunk_size = 256

        # BH Galaxy compute grid is 12x10 = 120 cores. Try full 12x10 first
        # (uses all compute cores, smallest per_core_N = best L1 fit).
        # Falls back to 12x8 / 8x8 on smaller devices.
        if self.grid_size[0] >= 12 and self.grid_size[1] >= 10:
            self._pcfg_grid = (12, 10)
        elif self.grid_size[0] >= 12:
            self._pcfg_grid = (12, 8)
        else:
            self._pcfg_grid = (8, 8)

    def forward(self, x) -> ttnn.Tensor:
        """
        Forward pass using chunked processing with auto L1 sharding.

        Strategy:
        1. Chunk input along sequence dimension (e.g., 544 → 3 chunks of 256)
        2. Let matmul auto-compute sharding (typically WIDTH or BLOCK in L1)
        3. Subsequent ops inherit sharding from matmul output
        4. Accumulate results in L1, concatenate at end

        Args:
            x: Input tensor [batch, seq, hidden] or [batch, 1, seq, hidden] (PyTorch or TTNN)

        Returns:
            TTNN output tensor [batch, seq, hidden] or [batch, 1, seq, hidden]
        """
        # Convert PyTorch to TTNN if needed
        was_torch = isinstance(x, torch.Tensor)
        if was_torch:
            x = ttnn.from_torch(
                x,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=self.device,
            )

        batch_size = x.shape[0]
        was_3d = len(x.shape) == 3

        # Always work with 4D tensors (ttnn.slice requires 4D coordinates)
        if was_3d:
            x = ttnn.reshape(x, [batch_size, 1, x.shape[1], x.shape[2]])

        seq_len = x.shape[2]
        hidden = x.shape[3]

        # Calculate number of chunks (tile-aligned)
        num_chunks = (seq_len + self.chunk_size - 1) // self.chunk_size
        output_chunks = []

        for chunk_idx in range(num_chunks):
            chunk_start = chunk_idx * self.chunk_size
            chunk_end = min(chunk_start + self.chunk_size, seq_len)
            actual_chunk_size = chunk_end - chunk_start

            # BUGFIX: Previously this padded to self.chunk_size (e.g. 544 on BH Galaxy).
            # That meant the expert MLP (seq=51) was running matmuls on (1, 544, 1024)
            # tensors — ~10x more work than necessary. Now pad only to the next
            # tile multiple (32). Verified savings: ~26 ms / inference (expert MLP
            # gate+up+down accounted for 38 ms of matmul time, mostly wasted
            # padding compute).
            tile = 32
            padded_chunk_size = ((actual_chunk_size + tile - 1) // tile) * tile
            needs_chunk_padding = padded_chunk_size > actual_chunk_size

            # Slice input chunk (always 4D)
            x_chunk = ttnn.slice(x, [0, 0, chunk_start, 0], [batch_size, 1, chunk_end, hidden])

            # Pad chunk if needed for tile alignment
            # Move to DRAM for multicore pad support (avoids L1 fallback warning)
            if needs_chunk_padding:
                pad_amount = padded_chunk_size - actual_chunk_size
                x_chunk = ttnn.to_memory_config(x_chunk, ttnn.DRAM_MEMORY_CONFIG)
                x_chunk = ttnn.pad(x_chunk, padding=((0, 0), (0, 0), (0, pad_amount), (0, 0)), value=0.0)

            # OPTIMIZATION: precomputed program_config for 2D BLOCK_SHARDED matmul.
            # Cached by (M_tiles, K_tiles, N_tiles) so the same config is reused
            # across all 18 layers per MLP shape. The pcfg helper returns None for
            # awkward shapes; we fall back to the default `core_grid` path then.
            m_tiles = padded_chunk_size // 32
            k_to_intermediate = self.hidden_size // 32
            k_to_hidden = self.intermediate_size // 32
            n_intermediate = self.intermediate_size // 32
            n_hidden = self.hidden_size // 32

            # MLP gate/up/down — let the helper auto-pick in0_block_w based on
            # per_core_N. Large-N (gate/up, per_core_N=43) keeps block_w=4;
            # small-N (down, per_core_N=6-22 depending on width) gets block_w=8.
            #
            # PI0_VLM_MLP_IBW=<N> opt-in: forces in0_block_w on gate/up only.
            # Use to shrink in0 CB when fitting larger chunks (e.g. chunk=992
            # CB-clashes with default ibw=4; ibw=2 halves the in0 CB at the
            # cost of more K-trips per matmul). Must divide K_tiles=64.
            import os as _os_ibw

            _ibw_env = _os_ibw.environ.get("PI0_VLM_MLP_IBW", "").strip()
            _ibw_override = int(_ibw_env) if _ibw_env.isdigit() else None
            _gate_up_kwargs = {"in0_block_w": _ibw_override} if _ibw_override else {}
            gate_pcfg = build_matmul_pcfg(
                m_tiles,
                k_to_intermediate,
                n_intermediate,
                self._pcfg_grid[0],
                self._pcfg_grid[1],
                activation=(ttnn.UnaryOpType.GELU, True),
                **_gate_up_kwargs,
            )
            up_pcfg = build_matmul_pcfg(
                m_tiles,
                k_to_intermediate,
                n_intermediate,
                self._pcfg_grid[0],
                self._pcfg_grid[1],
                **_gate_up_kwargs,
            )
            down_pcfg = build_matmul_pcfg(
                m_tiles,
                k_to_hidden,
                n_hidden,
                self._pcfg_grid[0],
                self._pcfg_grid[1],
            )

            # PI0_VLM_MLP_BF8_OUT=1 opt-in: store gate/up matmul outputs as bf8_b
            # instead of bf16. Halves the peak L1 footprint of the two
            # intermediate (1, M, 16384) tensors (~660 KB/core → ~330 KB/core at
            # chunk_size=1024). Required to make chunk_size=1024 fit alongside
            # the matmul kernel's static CB region. PCC risk: bf8 quantization
            # of post-GELU activations compounds across 18 layers — validate via
            # test_pcc_pi05_model_libero.py before promoting to default.
            import os as _os_mlp

            _mlp_bf8_out = _os_mlp.environ.get("PI0_VLM_MLP_BF8_OUT", "").lower() in ("1", "true", "yes", "on")
            # The action-expert block forces bf16 outputs so mlp_output can feed the fused addcmul
            # gated residual (ternary addcmul requires all operands share a dtype).
            if self._force_bf16_out:
                _mlp_bf8_out = False
            common_kwargs = dict(
                dtype=ttnn.bfloat8_b if _mlp_bf8_out else ttnn.bfloat16,
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )

            # PI0_VLM_MLP_MINIMAL=1 opt-in: switch gate/up from ttnn.linear
            # (2D-mcast) to ttnn.experimental.minimal_matmul. minimal_matmul
            # streams K in K_block_size chunks via ONE reused in0 CB instead
            # of the 2D-mcast's per_core_M × in0_block_w double-buffered CB,
            # collapsing the static-CB L1 footprint (per PERF_PLAYBOOKS/05 §4).
            # The escape hatch when 2D-mcast can't fit chunk=1024 even with
            # PI0_VLM_MLP_BF8_OUT=1 + PI0_VLM_MLP_IBW=2. NOTE: minimal_matmul
            # requires input.dtype == weight.dtype, so x_chunk must be cast
            # to bf8_b (weights are bf8_b). Only kicks in for VLM-sized M
            # (m_tiles ≥ 8); expert (m_tiles=1-2) stays on the 1D-width path.
            _mlp_minimal = _os_mlp.environ.get("PI0_VLM_MLP_MINIMAL", "").lower() in ("1", "true", "yes", "on")
            use_minimal = _mlp_minimal and m_tiles >= 8

            if use_minimal:
                x_chunk_bf8 = ttnn.typecast(x_chunk, ttnn.bfloat8_b, memory_config=ttnn.L1_MEMORY_CONFIG)
                ttnn.deallocate(x_chunk)
                # Playbook 05 §5: LoFi + fp32_dest=False unlocks subblock h*w cap 4 → 8.
                # bf8b weights tolerate LoFi (PCC validated layer-by-layer in BGE-M3).
                minimal_compute = ttnn.init_device_compute_kernel_config(
                    self.device.arch(),
                    math_fidelity=ttnn.MathFidelity.LoFi,
                    math_approx_mode=False,
                    fp32_dest_acc_en=False,
                    packer_l1_acc=True,
                )
                # PI0_VLM_MINIMAL_CFG="M,K,N,sh,sw" overrides block/subblock sizes
                # for the gate/up minimal_matmul. Default (8,8,8,4,2) matches the
                # Llama prefill ≤4096 config. Try (8,8,8,1,8) for the playbook 05
                # §5 wide-N subblock orientation.
                _cfg_env = _os_mlp.environ.get("PI0_VLM_MINIMAL_CFG", "").strip()
                _bs = [8, 8, 8, 4, 2]
                if _cfg_env:
                    try:
                        _parts = [int(x) for x in _cfg_env.split(",")]
                        if len(_parts) == 5 and all(p > 0 for p in _parts):
                            _bs = _parts
                    except ValueError:
                        pass
                minimal_cfg = ttnn.MinimalMatmulConfig(
                    M_block_size=_bs[0],
                    K_block_size=_bs[1],
                    N_block_size=_bs[2],
                    subblock_h=_bs[3],
                    subblock_w=_bs[4],
                    compute_with_storage_grid_size=ttnn.CoreCoord(self._pcfg_grid[0], self._pcfg_grid[1]),
                )
                gate_activated = ttnn.experimental.minimal_matmul(
                    x_chunk_bf8,
                    self.gate_proj,
                    fused_activation=(ttnn.UnaryOpType.GELU, True),
                    config=minimal_cfg,
                    compute_kernel_config=minimal_compute,
                    memory_config=ttnn.L1_MEMORY_CONFIG,
                    dtype=ttnn.bfloat8_b if _mlp_bf8_out else ttnn.bfloat16,
                )
                up = ttnn.experimental.minimal_matmul(
                    x_chunk_bf8,
                    self.up_proj,
                    config=minimal_cfg,
                    compute_kernel_config=minimal_compute,
                    memory_config=ttnn.L1_MEMORY_CONFIG,
                    dtype=ttnn.bfloat8_b if _mlp_bf8_out else ttnn.bfloat16,
                )
                ttnn.deallocate(x_chunk_bf8)
            else:
                if gate_pcfg is not None:
                    gate_activated = ttnn.linear(x_chunk, self.gate_proj, program_config=gate_pcfg, **common_kwargs)
                else:
                    gate_activated = ttnn.linear(
                        x_chunk, self.gate_proj, core_grid=self.core_grid, activation="gelu", **common_kwargs
                    )
                if up_pcfg is not None:
                    up = ttnn.linear(x_chunk, self.up_proj, program_config=up_pcfg, **common_kwargs)
                else:
                    up = ttnn.linear(x_chunk, self.up_proj, core_grid=self.core_grid, **common_kwargs)
                ttnn.deallocate(x_chunk)

            # Element-wise multiply (keep on L1 — feeds the down-proj matmul next)
            hidden_out = ttnn.multiply(gate_activated, up, memory_config=ttnn.L1_MEMORY_CONFIG)
            ttnn.deallocate(gate_activated)
            ttnn.deallocate(up)

            # Down projection — PI0_DRAM_SHARDED_MLP_DOWN=1 takes the
            # DRAM-width-sharded weight + L1-width-sharded activation path
            # (PERF_PLAYBOOKS/05 §3b decode recipe). Only fires for tiny M
            # (denoise expert m_tiles=1), VLM stays on existing path.
            if self.down_proj_dram_sharded is not None and m_tiles == 1:
                # Activation: L1 interleaved → L1 width-sharded across compute cores.
                # Compute cores are INDEPENDENT of DRAM banks (BH: 120 cores, 8 banks).
                # PI0_DRAM_SHARDED_MLP_COMPUTE_CORES overrides default (8). Must divide
                # both K_tiles (activation shard) and N_tiles_padded (output shard).
                # For our shape: K_t=128, N_t=32 → clean divisors {1,2,4,8,16,32}.
                # 32 cores gives K=4/core, N=1/core, 8×4 grid (uses 27% of BH grid).
                k_tiles_local = k_to_hidden  # mlp_dim // 32
                padded_n_tiles = self.down_proj_dram_sharded_padded_n // 32
                import os as _os_inner

                _cores_env = _os_inner.environ.get("PI0_DRAM_SHARDED_MLP_COMPUTE_CORES", "")
                if _cores_env in ("16", "32"):
                    num_compute_cores = int(_cores_env)
                elif self.down_proj_dram_sharded_dram_cores == 12:
                    num_compute_cores = 4
                else:
                    num_compute_cores = 8
                act_shard_k = (k_tiles_local // num_compute_cores) * 32  # K-cols per core
                # Construct grid: for ≤12 cores use 1-row strip; for 16/32 use a 2D grid
                if num_compute_cores <= 12:
                    act_grid = ttnn.CoreRangeSet(
                        {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(num_compute_cores - 1, 0))}
                    )
                elif num_compute_cores == 16:
                    # 8 × 2 grid
                    act_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 1))})
                elif num_compute_cores == 32:
                    # 8 × 4 grid
                    act_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))})
                else:
                    raise ValueError(f"unsupported num_compute_cores={num_compute_cores}")
                act_memcfg = ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                    ttnn.BufferType.L1,
                    ttnn.ShardSpec(
                        act_grid,
                        (padded_chunk_size, act_shard_k),
                        ttnn.ShardOrientation.ROW_MAJOR,
                    ),
                )
                hidden_out_sh = ttnn.to_memory_config(hidden_out, act_memcfg)
                ttnn.deallocate(hidden_out)

                # DRAM-sharded matmul program config.
                # in0_block_w capped at 8 (the helper's `_find_largest_divisor(N, max=8)`
                # convention); larger ibw would inflate L1 CB pressure.
                _kpc = k_tiles_local // num_compute_cores
                _ibw = min(_kpc, 8)
                # Walk down to a divisor of K-per-core if needed
                while _kpc % _ibw != 0 and _ibw > 1:
                    _ibw -= 1
                ds_pcfg = ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
                    in0_block_w=_ibw,
                    per_core_M=padded_chunk_size // 32,
                    per_core_N=padded_n_tiles // num_compute_cores,
                    fused_activation=None,
                )
                # Output stays L1 width-sharded; convert back to interleaved + slice padding.
                # No compute_kernel_config — match existing MLP matmuls (which use ttnn default).
                output_sh = ttnn.linear(
                    hidden_out_sh,
                    self.down_proj_dram_sharded,
                    program_config=ds_pcfg,
                    dtype=ttnn.bfloat16,
                    memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
                )
                ttnn.deallocate(hidden_out_sh)
                output_chunk = ttnn.sharded_to_interleaved(output_sh, memory_config=ttnn.L1_MEMORY_CONFIG)
                ttnn.deallocate(output_sh)
                # Slice off N-padding if applied (e.g. N=1024 → padded 1152 → slice back)
                if self.down_proj_dram_sharded_padded_n > hidden:
                    output_chunk = ttnn.slice(
                        output_chunk,
                        [0, 0, 0, 0],
                        [batch_size, 1, padded_chunk_size, hidden],
                    )
            elif down_pcfg is not None:
                output_chunk = ttnn.linear(hidden_out, self.down_proj, program_config=down_pcfg, **common_kwargs)
                ttnn.deallocate(hidden_out)
            else:
                output_chunk = ttnn.linear(hidden_out, self.down_proj, core_grid=self.core_grid, **common_kwargs)
                ttnn.deallocate(hidden_out)

            # Slice back to actual size if padded
            if needs_chunk_padding:
                output_chunk = ttnn.slice(output_chunk, [0, 0, 0, 0], [batch_size, 1, actual_chunk_size, hidden])

            output_chunks.append(output_chunk)

        # Concatenate all chunks along sequence dimension (always 4D now)
        if len(output_chunks) == 1:
            output = output_chunks[0]
        else:
            output = output_chunks[0]
            for i in range(1, len(output_chunks)):
                output = ttnn.concat([output, output_chunks[i]], dim=2, memory_config=ttnn.L1_MEMORY_CONFIG)
                ttnn.deallocate(output_chunks[i])

        # Reshape back to 3D if input was 3D
        if was_3d:
            output = ttnn.reshape(output, [batch_size, seq_len, hidden])

        # Convert back to PyTorch if input was PyTorch
        if was_torch:
            output = ttnn.to_torch(output)

        return output


# ============================================================================
# Full Transformer Block (TTNN)
# ============================================================================


class GemmaBlockTTNN:
    """
    Complete Gemma transformer block using TTNN.

    Architecture: Pre-LN with residual connections
        x -> RMSNorm -> Attention -> + -> RMSNorm -> MLP -> +
        |______________________________|___________________|
    """

    def __init__(
        self,
        config: GemmaConfig,
        weights: Dict[str, ttnn.Tensor],
        layer_idx: int,
        device: ttnn.Device,
        cos_meta: Optional[ttnn.Tensor] = None,
        sin_meta: Optional[ttnn.Tensor] = None,
    ):
        """
        Initialize transformer block with TTNN weights.

        Args:
            config: Gemma configuration
            weights: TTNN weight tensors
            layer_idx: Layer index
            device: TTNN device
            cos_meta: Precomputed cos for native TTNN RoPE [1, 1, max_seq, head_dim]
            sin_meta: Precomputed sin for native TTNN RoPE [1, 1, max_seq, head_dim]
        """
        self.config = config
        self.layer_idx = layer_idx
        self.device = device

        self.input_layernorm_weight = weights["input_layernorm.weight"]
        self.post_attention_layernorm_weight = weights["post_attention_layernorm.weight"]

        self.attention = GemmaAttentionTTNN(config, weights, layer_idx, device, cos_meta, sin_meta)
        self.mlp = GemmaMLPTTNN(config, weights, device)

        # Lazy sharded RMSNorm config (ViT-BH pattern). VLM hidden=2048
        # (hidden_tiles=64), M_padded is set by caller (prefix tile-aligned).
        self._rms_norm_sharded_pcfg = None
        self._rms_norm_sharded_memcfg = None
        self._rms_norm_sharded_m_padded = 0

    def _get_sharded_norm(self, m_padded: int):
        if self._rms_norm_sharded_pcfg is None or self._rms_norm_sharded_m_padded != m_padded:
            m_tiles = m_padded // 32
            hidden_tiles = self.config.width // 32
            # PI0_LN_INTERLEAVED_SMALL_M=1 forces interleaved LN at M_tiles==1 (the
            # denoise expert case). The sharded path at M_tiles=1 runs on 8 cores
            # but adds an I2S+S2I round-trip (~1.15 µs/LN); the interleaved op may
            # win net on this shape. Toggle for perf A/B testing.
            import os as _os

            disable_small_m_sharded = (
                _os.environ.get("PI0_LN_INTERLEAVED_SMALL_M", "").lower() in ("1", "true", "yes", "on") and m_tiles == 1
            )
            if disable_small_m_sharded:
                norm_cfg = None
            else:
                norm_cfg = build_sharded_norm_pcfg(
                    m_tiles, hidden_tiles, max_grid_x=8, max_grid_y=min(8, max(1, m_tiles))
                )
            if norm_cfg is not None:
                pc, memcfg_factory, _grid = norm_cfg
                self._rms_norm_sharded_pcfg = pc
                self._rms_norm_sharded_memcfg = memcfg_factory(1, m_padded, m_padded, self.config.width)
                self._rms_norm_sharded_m_padded = m_padded
            else:
                self._rms_norm_sharded_pcfg = None
                self._rms_norm_sharded_memcfg = None
        return self._rms_norm_sharded_pcfg, self._rms_norm_sharded_memcfg

    def forward(
        self,
        hidden_states: ttnn.Tensor,
        cos: ttnn.Tensor,
        sin: ttnn.Tensor,
        attention_mask: Optional[ttnn.Tensor] = None,
        position_ids: Optional[ttnn.Tensor] = None,
        past_key_value: Optional[Tuple[ttnn.Tensor, ttnn.Tensor]] = None,
        use_cache: bool = False,
        keep_padded: bool = False,
    ) -> Tuple[ttnn.Tensor, Optional[Tuple[ttnn.Tensor, ttnn.Tensor]]]:
        """
        Forward pass using TTNN operations.

        Args:
            hidden_states: TTNN input tensor
            cos, sin: Unused (kept for API compatibility, passed through to attention)
            attention_mask: Attention mask
            position_ids: Position indices
            past_key_value: Cached KV
            use_cache: Whether to return cache

        Returns:
            Tuple of (output, optional_cache)
        """
        m_padded = hidden_states.shape[1] if len(hidden_states.shape) == 3 else hidden_states.shape[2]
        sh_pc, sh_mc = self._get_sharded_norm(m_padded)

        # Pre-attention norm (sharded if available)
        normed = rms_norm_ttnn(
            hidden_states,
            self.input_layernorm_weight,
            self.config.rms_norm_eps,
            sharded_pcfg=sh_pc,
            sharded_memcfg=sh_mc,
        )
        if sh_pc is not None:
            normed = ttnn.sharded_to_interleaved(normed, memory_config=ttnn.L1_MEMORY_CONFIG)

        # Attention with residual
        attn_output, new_cache = self.attention.forward(
            normed,
            cos,
            sin,
            attention_mask,
            position_ids,
            past_key_value,
            use_cache,
            keep_padded=keep_padded,
        )
        ttnn.deallocate(normed)
        hidden_states = ttnn.add(hidden_states, attn_output, memory_config=ttnn.L1_MEMORY_CONFIG)
        ttnn.deallocate(attn_output)

        # Pre-MLP norm (sharded if available)
        normed = rms_norm_ttnn(
            hidden_states,
            self.post_attention_layernorm_weight,
            self.config.rms_norm_eps,
            sharded_pcfg=sh_pc,
            sharded_memcfg=sh_mc,
        )
        if sh_pc is not None:
            normed = ttnn.sharded_to_interleaved(normed, memory_config=ttnn.L1_MEMORY_CONFIG)

        # MLP with residual
        mlp_output = self.mlp.forward(normed)
        ttnn.deallocate(normed)
        hidden_states = ttnn.add(hidden_states, mlp_output, memory_config=ttnn.L1_MEMORY_CONFIG)
        ttnn.deallocate(mlp_output)

        return hidden_states, new_cache


# Default exports
GemmaAttention = GemmaAttentionTTNN
GemmaMLP = GemmaMLPTTNN
GemmaBlock = GemmaBlockTTNN

# ============================================================================
# pi0.5-specific additions (subclasses, overrides)
# ============================================================================

from typing import Dict, List, Optional, Tuple

import ttnn

from models.experimental.pi0_5.common.configs import GemmaConfig


_sharded_norm_cache: Dict[tuple, tuple] = {}


def build_sharded_norm_pcfg(
    m_tiles: int,
    hidden_tiles: int,
    *,
    max_grid_x: int = 8,
    max_grid_y: int = 8,
) -> Optional[tuple]:
    """Build (program_config, sharded_memory_config_factory, grid) for sharded
    RMS/LayerNorm using the LayerNormShardedMultiCoreProgramConfig pattern.

    Returns None if no clean grid divides both dims.

    For the pi0.5 expert (M_tiles=2, hidden_tiles=32) this picks (8, 2) →
    16 cores instead of the default 2-core interleaved path. For VLM
    (M_tiles=16, hidden_tiles=64) this picks (8, 8) → 64 cores.
    """
    key = (m_tiles, hidden_tiles, max_grid_x, max_grid_y)
    if key in _sharded_norm_cache:
        return _sharded_norm_cache[key]

    # grid_y must divide M_tiles; grid_x must divide hidden_tiles.
    cand_y = [y for y in range(min(max_grid_y, m_tiles), 0, -1) if m_tiles % y == 0]
    cand_x = [x for x in range(min(max_grid_x, hidden_tiles), 0, -1) if hidden_tiles % x == 0]
    if not cand_y or not cand_x:
        _sharded_norm_cache[key] = None
        return None

    best = None
    best_cores = 0
    for gy in cand_y:
        for gx in cand_x:
            cores = gx * gy
            # Prefer high core count; tie-break on grid_x (sharded LN parallelizes
            # along W primarily — wider grid helps small-M shapes).
            if cores > best_cores or (cores == best_cores and gx > best[0]):
                best = (gx, gy)
                best_cores = cores
    if best is None:
        _sharded_norm_cache[key] = None
        return None

    gx, gy = best
    block_h = m_tiles // gy
    block_w = hidden_tiles // gx
    subblock_w = block_w  # full-width per-core subblock (typical for sharded LN)

    pc = ttnn.LayerNormShardedMultiCoreProgramConfig(
        compute_with_storage_grid_size=(gx, gy),
        subblock_w=subblock_w,
        block_h=block_h,
        block_w=block_w,
        inplace=False,
    )

    def make_memcfg(b: int, m_logical: int, m_physical: int, hidden: int):
        # Block-sharded across the chosen grid.
        return ttnn.create_sharded_memory_config(
            (b, 1, m_physical, hidden),
            core_grid=ttnn.CoreGrid(y=gy, x=gx),
            strategy=ttnn.ShardStrategy.BLOCK,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
        )

    grid = ttnn.CoreGrid(y=gy, x=gx)
    result = (pc, make_memcfg, grid)
    _sharded_norm_cache[key] = result
    return result


def make_bs_memcfg(batch: int, m_padded: int, width: int, grid_x: int, grid_y: int) -> ttnn.MemoryConfig:
    """Block-sharded L1 memcfg on the same grid as sharded RMSNorm."""
    return ttnn.create_sharded_memory_config(
        (batch, 1, m_padded, width),
        core_grid=ttnn.CoreGrid(y=grid_y, x=grid_x),
        strategy=ttnn.ShardStrategy.BLOCK,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
    )


def bs_matmul_divisible(m_tiles: int, k_tiles: int, n_tiles: int, grid_x: int, grid_y: int) -> bool:
    return m_tiles % grid_y == 0 and k_tiles % grid_x == 0 and n_tiles % grid_x == 0


def build_bs_matmul_pcfg(
    m_tiles: int,
    k_tiles: int,
    n_tiles: int,
    grid_x: int,
    grid_y: int,
    *,
    in0_block_w: Optional[int] = None,
    activation=None,
    dst_budget: int = 4,
) -> ttnn.MatmulMultiCoreReuseMultiCastProgramConfig:
    """Block-sharded matmul program config pinned to the LN grid (SigLIP BS pattern)."""
    assert bs_matmul_divisible(m_tiles, k_tiles, n_tiles, grid_x, grid_y)
    per_core_M = m_tiles // grid_y
    per_core_N = n_tiles // grid_x
    per_core_K = k_tiles // grid_x
    if in0_block_w is None:
        in0_block_w = min(per_core_K, 4)
    while in0_block_w > 1 and per_core_K % in0_block_w != 0:
        in0_block_w -= 1
    in0_block_w = max(1, in0_block_w)
    out_subblock_w = min(per_core_N, dst_budget)
    while out_subblock_w > 1 and per_core_N % out_subblock_w != 0:
        out_subblock_w -= 1
    out_subblock_h = max(1, dst_budget // out_subblock_w)
    out_subblock_h = min(per_core_M, out_subblock_h)
    while out_subblock_h > 1 and per_core_M % out_subblock_h != 0:
        out_subblock_h -= 1
    return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=(grid_x, grid_y),
        in0_block_w=in0_block_w,
        out_subblock_h=out_subblock_h,
        out_subblock_w=out_subblock_w,
        per_core_M=per_core_M,
        per_core_N=per_core_N,
        transpose_mcast=False,
        fused_activation=activation,
    )


# Shared compute-kernel config for sharded RMS/LayerNorm. packer_l1_acc=True is
# a Blackhole feature (matches ViT-BH ln_compute_config from
# tech_reports/ViT-TTNN/vit_bh.md §2.2) — reduces CB pressure by letting the
# packer accumulate into L1 directly.
_RMS_NORM_COMPUTE_CONFIG = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi2,
    math_approx_mode=False,  # was True — precision A/B: keep fp32_dest_acc_en=False
    # to avoid halving the dst-register budget (which trips
    # subblock_wt <= 4 in the sharded LN kernel).
    fp32_dest_acc_en=False,
    packer_l1_acc=True,
)


def _modulated_rms_norm(
    x: "ttnn.Tensor",
    scale: "ttnn.Tensor",
    shift: "ttnn.Tensor",
    eps: float,
    pre_added: bool = False,
    sharded_pcfg: Optional["ttnn.LayerNormShardedMultiCoreProgramConfig"] = None,
    sharded_memcfg: Optional["ttnn.MemoryConfig"] = None,
) -> "ttnn.Tensor":
    """Fused: ((x · rsqrt(mean(x²)+ε)) · (1+scale)) + shift in one kernel.

    If pre_added=True, `scale` is already (1+scale) — skip the add. Used by
    the bs1 fast path where modulations are precomputed at init.

    If sharded_pcfg + sharded_memcfg are provided, runs the sharded multi-core
    path: convert input to block-sharded layout, run rms_norm with the sharded
    program config + packer_l1_acc compute kernel, return the block-sharded
    result (caller is responsible for converting back if downstream consumer
    needs interleaved). For pi0.5 expert (M_tiles=2) this lifts LN from 2-core
    to 16-core execution.
    """
    if sharded_pcfg is not None and sharded_memcfg is not None:
        # Move input into the matching block-sharded layout.
        x_sh = ttnn.to_memory_config(x, sharded_memcfg)
        if not pre_added:
            scale_plus_one = ttnn.add(scale, 1.0)
            weight = scale_plus_one
        else:
            scale_plus_one = None
            weight = scale
        out = ttnn.rms_norm(
            x_sh,
            weight=weight,
            bias=shift,
            epsilon=eps,
            program_config=sharded_pcfg,
            memory_config=sharded_memcfg,
            compute_kernel_config=_RMS_NORM_COMPUTE_CONFIG,
        )
        if scale_plus_one is not None:
            ttnn.deallocate(scale_plus_one)
        if x_sh is not x:
            ttnn.deallocate(x_sh)
        return out

    if pre_added:
        return ttnn.rms_norm(
            x,
            weight=scale,
            bias=shift,
            epsilon=eps,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
    scale_plus_one = ttnn.add(scale, 1.0)
    out = ttnn.rms_norm(
        x,
        weight=scale_plus_one,
        bias=shift,
        epsilon=eps,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    ttnn.deallocate(scale_plus_one)
    return out


def _split_modulation_6(mod: "ttnn.Tensor") -> List["ttnn.Tensor"]:
    """mod: (B, 6*W) -> 6 tensors of shape (B, 1, W)."""
    B = mod.shape[0]
    total = mod.shape[-1]
    width = total // 6
    mod3 = ttnn.reshape(mod, (B, 1, total))
    return [mod3[:, :, i * width : (i + 1) * width] for i in range(6)]


def ada_rms_norm_ttnn(
    x: "ttnn.Tensor",
    cond: "ttnn.Tensor",
    mod_weight: "ttnn.Tensor",
    mod_bias: Optional["ttnn.Tensor"],
    eps: float,
    core_grid: "ttnn.CoreGrid",
) -> Tuple["ttnn.Tensor", "ttnn.Tensor"]:
    """Standalone adaRMS (single (scale, shift, gate) from a 3*W Dense).

    Kept for the final stack norm path, which uses a separate modulation
    Dense (`model.norm.dense.*`) rather than the block-level fused Dense.
    """
    mod = ttnn.linear(
        cond,
        mod_weight,
        bias=mod_bias,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        core_grid=core_grid,
    )
    B = mod.shape[0]
    width3 = mod.shape[-1]
    width = width3 // 3
    mod3 = ttnn.reshape(mod, (B, 1, width3))
    ttnn.deallocate(mod)

    scale = mod3[:, :, 0:width]
    shift = mod3[:, :, width : 2 * width]
    gate = mod3[:, :, 2 * width : 3 * width]

    out = _modulated_rms_norm(x, scale, shift, eps)
    ttnn.deallocate(scale)
    ttnn.deallocate(shift)
    ttnn.deallocate(mod3)
    return out, gate


def ada_rms_norm_no_gate_ttnn(
    x: "ttnn.Tensor",
    cond: "ttnn.Tensor",
    mod_weight: "ttnn.Tensor",
    mod_bias: Optional["ttnn.Tensor"],
    eps: float,
    core_grid: "ttnn.CoreGrid",
) -> "ttnn.Tensor":
    """Adaptive RMSNorm for the final stack norm — gate is discarded."""
    out, gate = ada_rms_norm_ttnn(x, cond, mod_weight, mod_bias, eps, core_grid)
    ttnn.deallocate(gate)
    return out


def ada_rms_norm_no_gate_precomputed_ttnn(
    x: "ttnn.Tensor",
    scale_plus_one: "ttnn.Tensor",
    shift: "ttnn.Tensor",
    eps: float,
) -> "ttnn.Tensor":
    """Final-norm adaRMS using precomputed (1+scale, shift) — gate is discarded."""
    return _modulated_rms_norm(x, scale_plus_one, shift, eps, pre_added=True)


class AdaRMSGemmaBlockTTNN:
    """PI0.5 action-expert block (TTNN): one fused 6*W modulation Dense + adaRMS + gated residuals."""

    def __init__(
        self,
        config: GemmaConfig,
        weights: Dict[str, "ttnn.Tensor"],
        layer_idx: int,
        device: "ttnn.Device",
        cos_meta: Optional["ttnn.Tensor"] = None,
        sin_meta: Optional["ttnn.Tensor"] = None,
    ):
        self.config = config
        self.layer_idx = layer_idx
        self.device = device

        self.mod_weight = weights["adarms_mod.weight"]
        self.mod_bias = weights.get("adarms_mod.bias")

        # Action-expert (denoise) path: fused-only attention + bf16 MLP output (feeds the fused
        # addcmul gated residual). The shared VLM-prefill block keeps the unfused GemmaAttentionTTNN.
        self.attention = AdaRMSExpertAttentionTTNN(config, weights, layer_idx, device, cos_meta, sin_meta)
        self.mlp = GemmaMLPTTNN(config, weights, device, force_bf16_out=True)

        device_grid = device.compute_with_storage_grid_size()
        self.core_grid = ttnn.CoreGrid(y=device_grid.y, x=device_grid.x)

        # HiFi2 for the per-block modulation Dense — matches the rest of the
        # expert's projection linears and shaves real cycles since this runs
        # 18 × 10 steps = 180 times per chunk.
        self.mod_compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=True,
        )

        # Sharded RMSNorm config (ViT-BH pattern, tech_reports/ViT-TTNN/vit_bh.md §5.3).
        # Expert suffix M_padded = 64 (action_horizon=50 padded to 64), hidden = 1024.
        # → m_tiles=2, hidden_tiles=32 → grid (8,2)=16 cores instead of the
        # default 2-core (M_tiles=2) interleaved path. Built lazily on first
        # forward() call since the M dimension is set by the caller (suffix
        # embedding) and may vary with action_horizon.
        self._rms_norm_sharded_pcfg = None
        self._rms_norm_sharded_memcfg = None
        self._rms_norm_sharded_m_padded = 0

    def forward(
        self,
        hidden_states: "ttnn.Tensor",
        cos: "ttnn.Tensor",
        sin: "ttnn.Tensor",
        adarms_cond: "ttnn.Tensor",
        attention_mask: Optional["ttnn.Tensor"] = None,
        position_ids: Optional["ttnn.Tensor"] = None,
        past_key_value: Optional[Tuple["ttnn.Tensor", "ttnn.Tensor"]] = None,
        use_cache: bool = False,
        precomputed_mod: Optional[Tuple["ttnn.Tensor", ...]] = None,
        keep_padded: bool = False,
    ) -> Tuple["ttnn.Tensor", Optional[Tuple["ttnn.Tensor", "ttnn.Tensor"]]]:
        # Fast path: 6 modulation tensors precomputed at init (sa already = 1+scale_a).
        if precomputed_mod is not None:
            sa1, ta, ga, sf1, tf, gf = precomputed_mod
            mod_owned = False
        else:
            mod = ttnn.linear(
                adarms_cond,
                self.mod_weight,
                bias=self.mod_bias,
                memory_config=ttnn.L1_MEMORY_CONFIG,
                core_grid=self.core_grid,
                compute_kernel_config=self.mod_compute_kernel_config,
            )
            sa1, ta, ga, sf1, tf, gf = _split_modulation_6(mod)
            ttnn.deallocate(mod)
            mod_owned = True

        # Lazy build of the sharded RMSNorm config keyed on actual M.
        # PI0_LN_INTERLEAVED_SMALL_M=1 disables the sharded path at M_tiles=1
        # (denoise suffix case): the sharded path runs on only 8 cores and adds
        # an I2S input convert + S2I output convert (~1.15 µs/LN). At this size
        # interleaved LN may net out faster — toggle for perf A/B testing.
        m_padded = hidden_states.shape[1] if len(hidden_states.shape) == 3 else hidden_states.shape[2]
        if self._rms_norm_sharded_pcfg is None or self._rms_norm_sharded_m_padded != m_padded:
            m_tiles = m_padded // 32
            hidden_tiles = self.config.width // 32
            import os as _os

            disable_small_m_sharded = (
                _os.environ.get("PI0_LN_INTERLEAVED_SMALL_M", "").lower() in ("1", "true", "yes", "on") and m_tiles == 1
            )
            if disable_small_m_sharded:
                norm_cfg = None
            else:
                norm_cfg = build_sharded_norm_pcfg(m_tiles, hidden_tiles, max_grid_x=8, max_grid_y=max(1, m_tiles))
            if norm_cfg is not None:
                pc, memcfg_factory, _grid = norm_cfg
                self._rms_norm_sharded_pcfg = pc
                self._rms_norm_sharded_memcfg = memcfg_factory(1, m_padded, m_padded, self.config.width)
                self._rms_norm_sharded_m_padded = m_padded
            else:
                self._rms_norm_sharded_pcfg = None
                self._rms_norm_sharded_memcfg = None

        sh_pc = self._rms_norm_sharded_pcfg
        sh_mc = self._rms_norm_sharded_memcfg

        # ---- Attention sublayer ----
        normed = _modulated_rms_norm(
            hidden_states,
            sa1,
            ta,
            self.config.rms_norm_eps,
            pre_added=not mod_owned,
            sharded_pcfg=sh_pc,
            sharded_memcfg=sh_mc,
        )
        if mod_owned:
            ttnn.deallocate(sa1)
            ttnn.deallocate(ta)
        # Convert back to interleaved L1 for the existing attention path
        # (QKV matmul uses 1D width-shard with mcast_in0 over interleaved input).
        if sh_pc is not None:
            normed = ttnn.sharded_to_interleaved(normed, memory_config=ttnn.L1_MEMORY_CONFIG)
        attn_output, new_cache = self.attention.forward(
            normed, cos, sin, attention_mask, position_ids, past_key_value, use_cache, keep_padded=keep_padded
        )
        ttnn.deallocate(normed)
        # Fused gated residual: hidden + ga*attn_output in one addcmul. All operands are bf16
        # (AdaRMSExpertAttentionTTNN emits bf16; the residual stream + gate are bf16), satisfying
        # the ternary addcmul same-dtype requirement.
        hidden_states = ttnn.addcmul(hidden_states, ga, attn_output, memory_config=ttnn.L1_MEMORY_CONFIG)
        ttnn.deallocate(attn_output)
        if mod_owned:
            ttnn.deallocate(ga)

        # ---- FFW sublayer ----
        normed = _modulated_rms_norm(
            hidden_states,
            sf1,
            tf,
            self.config.rms_norm_eps,
            pre_added=not mod_owned,
            sharded_pcfg=sh_pc,
            sharded_memcfg=sh_mc,
        )
        if sh_pc is not None:
            normed = ttnn.sharded_to_interleaved(normed, memory_config=ttnn.L1_MEMORY_CONFIG)
        if mod_owned:
            ttnn.deallocate(sf1)
            ttnn.deallocate(tf)
        mlp_output = self.mlp.forward(normed)
        ttnn.deallocate(normed)
        # Fused gated residual: hidden + gf*mlp_output in one addcmul. The expert MLP runs with
        # force_bf16_out so mlp_output is bf16, matching the bf16 residual stream + gate.
        hidden_states = ttnn.addcmul(hidden_states, gf, mlp_output, memory_config=ttnn.L1_MEMORY_CONFIG)
        ttnn.deallocate(mlp_output)
        if mod_owned:
            ttnn.deallocate(gf)

        return hidden_states, new_cache
