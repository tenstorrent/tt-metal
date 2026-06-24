# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Zero-centered RMSNorm for Qwen3.5 (weights pre-offset by +1; fused ttnn.rms_norm)."""
import ttnn


def rms_norm_ttnn(x, weight, eps=1e-6, memory_config=None):
    """Zero-centered RMSNorm using fused ttnn.rms_norm.

    Qwen3.5 uses zero-centered RMSNorm for ALL layer norms:
      output = x * rsqrt(mean(x^2) + eps) * (1 + weight)

    The weight should be pre-offset by +1 so we can use the standard fused op.
    Verified: fused ttnn.rms_norm works for all decode shapes on Blackhole P150:
      [1, 1, 4096], [1, 1, 16, 256], [1, 1, 32, 128]
    """
    return ttnn.rms_norm(x, weight=weight, epsilon=eps, memory_config=memory_config)
