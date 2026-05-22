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

# Audio tokenizer transformer matmuls: HiFi2 decode (see audio_tokenizer_optimizations.py for full preset).
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
