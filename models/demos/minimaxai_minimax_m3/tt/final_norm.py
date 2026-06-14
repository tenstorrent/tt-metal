# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Final RMSNorm (``language_model.model.norm``) for MiniMax-M3.

Identical math to the per-layer :class:`RMSNorm` (gemma ``+1``, eps 1e-6,
replicated gamma). Kept as a distinct block so the bring-up state tracks the
final-norm golden separately from the per-layer norm golden.
"""

from __future__ import annotations

import ttnn
from models.demos.minimaxai_minimax_m3.tt.rms_norm import RMSNorm


class FinalNorm(RMSNorm):
    """Alias of RMSNorm for the model's final norm before lm_head."""

    def __init__(self, mesh_device, weight: ttnn.Tensor, eps: float = 1e-6):
        super().__init__(mesh_device, weight, eps=eps)
