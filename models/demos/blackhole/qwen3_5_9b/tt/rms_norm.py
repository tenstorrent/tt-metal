# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Plain (non-distributed) RMSNorm — ttnn port of transformers' Qwen3_5RMSNorm.

This is the attention block's internal q/k-norm primitive: a local RMS over the
last (head) dim, never sharded across devices. The decoder's input/post-attention
norms use the framework RMSNorm + DistributedNorm instead (see tt/layer.py).
"""
import ttnn
from models.common.lightweightmodule import LightweightModule


class Qwen35RMSNorm(LightweightModule):
    """ttnn port of transformers' Qwen3_5RMSNorm (modeling_qwen3_5.py).

    ``scale=True`` bakes the Qwen3_5RMSNorm ``(1 + weight)`` gain into the stored
    weight once at construction, so the forward stays a plain ``rms_norm * weight``.
    """

    def __init__(self, weight: ttnn.Tensor, eps: float = 1e-6, scale: bool = False):
        super().__init__()
        self.eps = eps
        # Fold the (1 + weight) gain in up front when requested; otherwise use the raw weight.
        self.weight = ttnn.add(weight, 1.0) if scale else weight

    def forward(self, x: ttnn.Tensor):
        """Normalize over the last dim, then scale by the (pre-baked) weight."""
        output = ttnn.rms_norm(x, epsilon=self.eps)
        output = output * self.weight
        return output

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.eps}"
