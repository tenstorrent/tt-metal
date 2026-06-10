# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Voxtral TT shared constants and presets.


"""

import ttnn

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


def voxtral_matmul_activation_mem_config(seq_len: int, *, max_l1_seq_len: int = VOXTRAL_MATMUL_L1_MAX_SEQ_LEN):
    """Pick L1 vs DRAM for matmul/elementwise activations from sequence length (tile M dimension)."""
    if int(seq_len) <= int(max_l1_seq_len):
        return ttnn.L1_MEMORY_CONFIG
    return ttnn.DRAM_MEMORY_CONFIG
