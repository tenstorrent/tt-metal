# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Decode trace state for batch-bucketed traced decode.

Re-exports the agentic DecodeTraceSamplingState which holds persistent
tensor buffers for vLLM trace_mode=all decode. Each batch bucket gets its
own state with pre-allocated input/output tensors that survive across
traced decode steps.
"""

from models.demos.glm4_moe_lite.tt.decode_trace_state import DecodeTraceSamplingState

__all__ = ["DecodeTraceSamplingState"]
