# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Voxtral TT shared constants and presets.


"""

from __future__ import annotations

import math

import ttnn

from models.tt_transformers.tt.common import get_out_subblock_w

# Acoustic flow-matching trunk: HiFi4 + FP32 dest accumulation for closer CPU parity.
COMPUTE_KERNEL_CONFIG_VOXTRAL_ACOUSTIC = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi4,
    math_approx_mode=False,
    fp32_dest_acc_en=True,
    packer_l1_acc=True,
)

# BFP8_B transformer matmuls (wqkv, wo, ff1/ff2/ff3): HiFi2 is sufficient because
# BFP8_B weights have 7-bit mantissa (same as BF16); HiFi4's extra mantissa bits are
# wasted on 8-bit data and cost ~2x compute time for no accuracy benefit.
COMPUTE_KERNEL_CONFIG_VOXTRAL_ACOUSTIC_BFP8 = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi2,
    math_approx_mode=False,
    fp32_dest_acc_en=True,
    packer_l1_acc=True,
)

# Semantic head only: HiFi4_FP32 (dst_full_sync_en=False) for near-tie argmax logits.
COMPUTE_KERNEL_CONFIG_VOXTRAL_SEMANTIC = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi4,
    math_approx_mode=False,
    fp32_dest_acc_en=True,
    packer_l1_acc=True,
    dst_full_sync_en=False,
)

# Audio tokenizer transformer matmuls/SDPA: HiFi2 (matches conv/RMSNorm; weights stay BFP8 in default preset).
COMPUTE_KERNEL_CONFIG_VOXTRAL_AUDIO_TOKENIZER = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi2,
    math_approx_mode=False,
    fp32_dest_acc_en=False,
    packer_l1_acc=True,
)

# Matmul activation staging: weights stay in DRAM; in0 uses L1 when T fits one tile row (32).
VOXTRAL_MATMUL_L1_MAX_SEQ_LEN = 32
# Tier-1 2D matmul program configs OOM on large M (e.g. block 7 at T=12800, per_core_M=50).
VOXTRAL_MATMUL_PROGCFG_MAX_SEQ_LEN = 6400


def voxtral_matmul_activation_mem_config(seq_len: int, *, max_l1_seq_len: int = VOXTRAL_MATMUL_L1_MAX_SEQ_LEN):
    """Pick L1 vs DRAM for matmul/elementwise activations from sequence length (tile M dimension)."""
    if int(seq_len) <= int(max_l1_seq_len):
        return ttnn.L1_MEMORY_CONFIG
    return ttnn.DRAM_MEMORY_CONFIG


def _find_largest_divisor(n: int, max_divisor: int = 8) -> int:
    for i in range(max_divisor, 0, -1):
        if n % i == 0:
            return i
    return 1


def _pick_audio_tokenizer_matmul_grid(k: int, max_x: int, max_y: int) -> tuple[int, int]:
    """Pick (grid_x, grid_y) for 2D mcast matmul; ``k % (TILE * grid_y) == 0``."""
    tile = ttnn.TILE_SIZE
    max_cores = max_x * max_y
    for y in range(max_y, 0, -1):
        if k % (tile * y) != 0:
            continue
        x = min(max_x, max_cores // y)
        if x >= 1:
            return (x, y)
    return (1, 1)


def voxtral_audio_tokenizer_matmul_program_config(
    device,
    m: int,
    k: int,
    n: int,
) -> ttnn.MatmulMultiCoreReuseMultiCastProgramConfig:
    """2D multicast matmul program config for audio tokenizer decode (M = seq_len)."""
    tile = ttnn.TILE_SIZE
    m, k, n = int(m), int(k), int(n)
    max_g = device.compute_with_storage_grid_size()
    max_x, max_y = int(max_g.x), int(max_g.y)
    grid_x, grid_y = _pick_audio_tokenizer_matmul_grid(k, max_x, max_y)
    per_core_m = math.ceil(m / (tile * grid_y))
    per_core_n = math.ceil(n / (tile * grid_x))
    k_per_core_tiles = k // (tile * grid_y)
    in0_block_w = _find_largest_divisor(k_per_core_tiles, max_divisor=8)
    out_subblock_h = 1
    out_subblock_w = get_out_subblock_w(per_core_n, out_subblock_h)
    return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=(grid_x, grid_y),
        in0_block_w=in0_block_w,
        out_subblock_h=out_subblock_h,
        out_subblock_w=out_subblock_w,
        per_core_M=per_core_m,
        per_core_N=per_core_n,
        transpose_mcast=False,
        fuse_batch=False,
        fused_activation=None,
    )


def voxtral_audio_tokenizer_ttnn_sliding_window_size(config_window: int) -> int:
    """Map vLLM ``attn_sliding_window_size`` to TTNN ``sliding_window_size`` (inclusive token count)."""
    return int(config_window) + 1


def _pick_audio_tokenizer_sdpa_chunk_size(seq_len: int) -> int:
    tile = ttnn.TILE_SIZE
    for candidate in (256, 128, 64, 32):
        if seq_len % candidate == 0 and candidate % tile == 0:
            return candidate
    return tile


def voxtral_audio_tokenizer_sdpa_program_config(
    device,
    seq_len: int,
) -> ttnn.SDPAProgramConfig:
    """Chunked SDPA program config for audio tokenizer decode (full-sequence prefill-style SDPA)."""
    chunk = _pick_audio_tokenizer_sdpa_chunk_size(int(seq_len))
    # Match tt_transformers prefill: 8x8 sub-grid avoids L1 OOM seen with full-grid configs at large T.
    return ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=(8, 8),
        q_chunk_size=chunk,
        k_chunk_size=chunk,
        exp_approx_mode=True,
    )


def voxtral_audio_tokenizer_matmul_program_configs(
    device,
    seq_len: int,
    *,
    dim: int,
    hidden_dim: int,
    n_heads: int,
    n_kv_heads: int,
    head_dim: int,
) -> dict[str, ttnn.MatmulMultiCoreReuseMultiCastProgramConfig] | None:
    """Per-forward matmul program configs for one audio tokenizer decoder layer."""
    if int(seq_len) > VOXTRAL_MATMUL_PROGCFG_MAX_SEQ_LEN:
        return None
    qkv_n = n_heads * head_dim + 2 * n_kv_heads * head_dim
    wo_k = n_heads * head_dim
    return {
        "wqkv": voxtral_audio_tokenizer_matmul_program_config(device, seq_len, dim, qkv_n),
        "wo": voxtral_audio_tokenizer_matmul_program_config(device, seq_len, wo_k, dim),
        "ff1_3": voxtral_audio_tokenizer_matmul_program_config(device, seq_len, dim, hidden_dim),
        "ff2": voxtral_audio_tokenizer_matmul_program_config(device, seq_len, hidden_dim, dim),
    }
