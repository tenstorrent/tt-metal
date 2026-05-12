# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Voxtral TT shared constants and presets.

Mirrors ``models/demos/deepseek_v3/utils/config_helpers.py``: centralize
``ttnn.WormholeComputeKernelConfig`` values for matmuls, SDPA, MLP, and norm wrappers.
"""

import ttnn

# Acoustic flow-matching trunk: HiFi4 + FP32 dest accumulation for closer CPU parity.
COMPUTE_KERNEL_CONFIG_VOXTRAL_ACOUSTIC = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi4,
    math_approx_mode=False,
    fp32_dest_acc_en=True,
    packer_l1_acc=True,
)
