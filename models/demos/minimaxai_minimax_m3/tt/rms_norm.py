# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Gemma-style RMSNorm for MiniMax-M3 (TP=32, replicated weight).

Matches ``reference.functional.rms_norm_forward``: normalize in fp32 over the
last dim, then scale by ``(1.0 + weight)`` (the gemma ``+1`` on gamma). The
``+1`` is folded into the gamma HOST-SIDE by the weight loader
(``add_gemma_one=True``), so the on-device forward is a single fused
``ttnn.rms_norm`` with a plain multiplicative weight — no add in the forward
path, no torch fallback.

The norm weight is REPLICATED on every device (TP recipe: norms replicate).
"""

from __future__ import annotations

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.demos.minimaxai_minimax_m3.tt import model_config as mc


class RMSNorm(LightweightModule):
    def __init__(self, mesh_device, weight: ttnn.Tensor, eps: float = 1e-6):
        """
        Args:
            mesh_device: the open bh_galaxy mesh.
            weight: REPLICATED ttnn gamma tensor with the gemma ``+1`` already
                folded in (loaded via weight_loader add_gemma_one=True).
            eps: RMSNorm variance epsilon (1e-6 for M3 text).
        """
        super().__init__()
        self.mesh_device = mesh_device
        self.weight = weight
        self.eps = eps
        self.compute_kernel_config = mc.default_compute_kernel_config()

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        return ttnn.rms_norm(
            x,
            epsilon=self.eps,
            weight=self.weight,
            compute_kernel_config=self.compute_kernel_config,
        )
