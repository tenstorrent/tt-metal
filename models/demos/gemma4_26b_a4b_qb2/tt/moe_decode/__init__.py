# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Batch-1 routed-expert MoE decode (WS4): ProgramDescriptor ops in op.py, kernels in kernels/, see README.md."""

from models.demos.gemma4_26b_a4b_qb2.tt.moe_decode.op import (
    MoeDecodeB1,
    down_layout,
    gate_up_bias_layout,
    gate_up_local_layout,
    router_topk_b1,
    shared_virtual_experts,
    tp_expert_layouts,
    upload_per_chip,
    upload_rows_per_chip,
)

__all__ = [
    "MoeDecodeB1",
    "down_layout",
    "gate_up_bias_layout",
    "gate_up_local_layout",
    "router_topk_b1",
    "shared_virtual_experts",
    "tp_expert_layouts",
    "upload_per_chip",
    "upload_rows_per_chip",
]
