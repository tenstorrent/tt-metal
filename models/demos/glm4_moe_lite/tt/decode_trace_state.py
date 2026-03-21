# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Decode trace state dataclass for GLM-4.7-Flash.

Extracted from model_tt.py. This dataclass holds persistent tensor buffers
used for batch-bucketed traced decode (vLLM trace_mode=all).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import ttnn


@dataclass
class DecodeTraceSamplingState:
    """Per-bucket state for batch-bucketed decode traces."""

    trace_id: Any | None = None
    batch: int = 0
    page_table_width: int = 0

    # Persistent inputs
    tokens_tt: ttnn.Tensor | None = None
    positions_tt: ttnn.Tensor | None = None
    rot_idxs_tt: ttnn.Tensor | None = None
    cos_batch_tt: ttnn.Tensor | None = None
    sin_batch_tt: ttnn.Tensor | None = None
    trans_matrix_tt: ttnn.Tensor | None = None
    page_table_tt: ttnn.Tensor | None = None
    rope_sharded_mem_config: Any | None = None
    rot_idxs_padded_batch: int = 0

    # Batch-expansion serial cache update
    positions_main_tt: ttnn.Tensor | None = None
    positions_draft_tt: ttnn.Tensor | None = None

    # Persistent outputs
    logits_tt: ttnn.Tensor | None = None
    top1_values_tt: ttnn.Tensor | None = None
    top1_indices_tt: ttnn.Tensor | None = None

    # MTP hidden state (preserved from before final_norm)
    mtp_hidden_tt: ttnn.Tensor | None = None

    # MTP trace state (Phase B2)
    mtp_tokens_tt: ttnn.Tensor | None = None
    mtp_positions_tt: ttnn.Tensor | None = None
    mtp_rot_idxs_tt: ttnn.Tensor | None = None
    mtp_rot_idxs_padded_batch: int = 0
    mtp_cos_batch_tt: ttnn.Tensor | None = None
    mtp_sin_batch_tt: ttnn.Tensor | None = None
    mtp_trans_matrix_tt: ttnn.Tensor | None = None
    mtp_rope_sharded_mem_config: Any | None = None
    mtp_trace_id: int | None = None
    mtp_top1_values_tt: ttnn.Tensor | None = None
    mtp_top1_indices_tt: ttnn.Tensor | None = None
