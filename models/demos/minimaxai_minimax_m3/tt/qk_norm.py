# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Per-head QK-norm for MiniMax-M3 attention (TP=32, replicated weights).

Matches ``reference.functional.qk_norm_forward``: a gemma-style RMSNorm
(``(1+w)`` scale, eps 1e-6) applied per-head over the last dim ``head_dim``
(128) of the query and key states, BEFORE rope. Because RMSNorm reduces only
the last dim, the per-head layout ``[B, H, S, head_dim]`` is handled directly
by ``ttnn.rms_norm`` with the ``[head_dim]`` gamma broadcast across all heads.

The gemma ``+1`` is folded into the gammas host-side by the weight loader
(``add_gemma_one=True``); the forward is two fused ``ttnn.rms_norm`` calls with
no add / no torch fallback. Weights REPLICATED per the TP recipe (norms).
"""

from __future__ import annotations

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.demos.minimaxai_minimax_m3.tt import model_config as mc


class QKNorm(LightweightModule):
    def __init__(self, mesh_device, q_weight: ttnn.Tensor, k_weight: ttnn.Tensor, eps: float = 1e-6):
        """
        Args:
            mesh_device: open bh_galaxy mesh.
            q_weight, k_weight: REPLICATED ttnn gammas (shape [head_dim]) with
                the gemma ``+1`` already folded in.
            eps: RMSNorm eps (1e-6).
        """
        super().__init__()
        self.mesh_device = mesh_device
        self.q_weight = q_weight
        self.k_weight = k_weight
        self.eps = eps
        self.compute_kernel_config = mc.default_compute_kernel_config()

    def _norm(self, x: ttnn.Tensor, w: ttnn.Tensor) -> ttnn.Tensor:
        return ttnn.rms_norm(
            x,
            epsilon=self.eps,
            weight=w,
            compute_kernel_config=self.compute_kernel_config,
        )

    def forward(self, q: ttnn.Tensor, k: ttnn.Tensor):
        """Normalize q and k over head_dim. Returns ``(q_normed, k_normed)``."""
        return self._norm(q, self.q_weight), self._norm(k, self.k_weight)
