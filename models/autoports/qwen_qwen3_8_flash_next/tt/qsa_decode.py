# SPDX-License-Identifier: Apache-2.0
"""Decode-step state shared by every sparse-attention (QSA) layer.

All QSA layers of one decode step see the same positions, so the packed
positions, RoPE position column and both RoPE table pairs (token and pooled
block) are computed once per step by the first QSA layer and reused by the
rest. The ops and their inputs are unchanged, so results are bit-identical;
only the repeated position work disappears (~20 ops per later QSA layer).

model.forward hands every QSA layer the same dict for a decode step (and None
otherwise). QWEN_FLASH_QSA_STEP_SHARE=0 restores per-layer recomputation.
"""

import os

import ttnn

from .fused_decoder import _shared_rope_tables


def enabled():
    return os.environ.get("QWEN_FLASH_QSA_STEP_SHARE", "1") != "0"


def step_positions(attention, current_pos):
    """Return the step dict for this decode position tensor, filling it on first use."""
    step = attention._decode_step
    if step.get("source") is current_pos:
        return step
    step.clear()
    step["source"] = current_pos
    batch = attention.batch_size
    step["packed"] = attention._packed_position(current_pos)
    start = ttnn.permute(attention._position_float(attention._canonical_position(current_pos)), (1, 0, 2, 3))
    positions = ttnn.add(start, attention.chunk_offsets[:, :, :1, :])
    step["rope"] = _shared_rope_tables(attention, positions, batch, 1)
    block_start = ttnn.multiply(ttnn.floor(ttnn.multiply(positions, 0.25)), 4.0)
    # The pooled-key RoPE (FusedQSA._chunk_rope) gathers its tables from the block start.
    step["block_rope"] = _shared_rope_tables(attention, block_start, batch, 1)
    return step
