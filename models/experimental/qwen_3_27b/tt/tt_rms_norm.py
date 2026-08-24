# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
RMSNorm over the last dimension.

CAREFUL: this model has TWO different RMSNorms, and they differ in a way that is
easy to miss and silently wrong.

1. Qwen3_5RMSNorm -- ZERO-CENTERED (modeling_qwen3_5.py:736)

       output = x * rsqrt(mean(x^2) + eps) * (1 + weight)

   The stored weight is centred on zero (initialized to zeros), hence the `1 +`.
   Used by: every layer's input/post-attention norm, the final norm, and the
   q_norm / k_norm inside gated attention.

   The cheap way to get the `1 +` is to add 1.0 to the weight ONCE at load time;
   then a plain ttnn.rms_norm is exact, with no extra runtime op.

2. Qwen3_5RMSNormGated -- PLAIN, then gated (modeling_qwen3_5.py:187)

       output = (x * rsqrt(mean(x^2) + eps) * weight) * silu(gate)

   No `1 +` here (weight is initialized to ones), and the norm happens BEFORE the
   gate is applied. Used only inside Gated DeltaNet, over head_v_dim = 128.
   That one lives in tt_gated_deltanet.py since it is fused with the gate.

This class implements form 1.
"""

import ttnn
from models.common.lightweightmodule import LightweightModule

EPS = 1e-6  # rms_norm_eps


class TtQwen36RmsNorm(LightweightModule):
    """
    Zero-centered RMSNorm.

    Used over two different widths:
        * per-layer input / post-attention norms   -> dim = 5120
        * final norm before the lm_head            -> dim = 5120
        * q_norm / k_norm inside gated attention   -> dim = 256 (per head)

    Shapes:
        x        [..., dim]
        weight   [dim]
        output   [..., dim]     same shape as input

    To implement:
        1. fold the +1 into the weight on host: weight + 1.0
        2. ttnn.rms_norm(x, weight=self.weight, epsilon=EPS)
    """

    def __init__(self, device, dim: int = 5120, eps: float = EPS):
        self.device = device
        self.dim = dim
        self.eps = eps

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        """[..., dim] -> [..., dim]."""
        return x
