# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Qwen long-prefill Q prep (typecast, q_norm, RoPE, typecast) as one generic_op program per device."""

from models.demos.llama3_70b_galaxy.tt.custom_ops.qwen_q_prep.op import (
    QPrepPlan,
    column_slices,
    pair_cores,
    qwen_q_prep,
)

__all__ = ["QPrepPlan", "column_slices", "pair_cores", "qwen_q_prep"]
