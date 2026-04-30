# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Thin eager executor for ``Qwen25_7BTTT``: token ids on device → hidden / logits.

This is intentionally small; deep Tracy integration can extend later (playbook §F).
"""

from __future__ import annotations

import ttnn
from models.common.models.qwen25_7b.model import Qwen25_7BTTT


def run_prefill(model: Qwen25_7BTTT, token_ids_tt: ttnn.Tensor, *, start_pos: int = 0) -> ttnn.Tensor:
    """Prefill chunk; ``token_ids_tt`` shape ``[1,1,1,S]``, ``S % 128 == 0``."""
    return model.prefill_forward(token_ids_tt, start_pos=start_pos)


def run_decode(model: Qwen25_7BTTT, token_id_tt: ttnn.Tensor, *, current_pos: int) -> ttnn.Tensor:
    """Single-token decode; ``token_id_tt`` shape ``[1,1,1,1]``."""
    return model.decode_forward(token_id_tt, current_pos=current_pos)


def run_lm_head(model: Qwen25_7BTTT, hidden_tt: ttnn.Tensor) -> ttnn.Tensor:
    return model.lm_logits(hidden_tt)
