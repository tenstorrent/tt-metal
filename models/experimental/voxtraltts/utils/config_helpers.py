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

# Audio tokenizer (1024-dim conv stacks): same preset until tokenizer-specific tuning is needed.
COMPUTE_KERNEL_CONFIG_VOXTRAL_AUDIO_TOKENIZER = COMPUTE_KERNEL_CONFIG_VOXTRAL_ACOUSTIC
