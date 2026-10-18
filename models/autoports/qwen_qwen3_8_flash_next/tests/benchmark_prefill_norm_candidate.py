# SPDX-License-Identifier: Apache-2.0
"""Isolated prefill HC norm row compaction; precision/stream weights unchanged."""

import ttnn
from models.autoports.qwen_qwen3_8_flash_next.tt.prefill_view import _PrefillOps


def compact_norm(ops, value, weight, eps, *, reuse_output=False):
    shape = tuple(value.shape)
    if len(shape) != 4 or shape[0] != 1 or shape[1] <= 1 or shape[2:] != (4, 640):
        return _PrefillOps.norm(ops, value, weight, eps, reuse_output=reuse_output)
    # Only layout/row geometry changes. RMS reduction still spans each640-wide
    # hidden shard; TP statistics remain FP32 with the original HiFi4 config.
    flat = ttnn.to_layout(
        ttnn.reshape(ttnn.to_layout(value, ttnn.ROW_MAJOR_LAYOUT), (1, 1, shape[1] * 4, 640)),
        ttnn.TILE_LAYOUT,
    )
    stats = ttnn.rms_norm_pre_all_gather(flat, dtype=ttnn.float32, compute_kernel_config=ops.compute)
    gathered = ops.gather(stats)
    if reuse_output:
        del stats
    normalized = ttnn.rms_norm_post_all_gather(flat, gathered, epsilon=eps, compute_kernel_config=ops.compute)
    restored = ttnn.to_layout(ttnn.reshape(ttnn.to_layout(normalized, ttnn.ROW_MAJOR_LAYOUT), shape), ttnn.TILE_LAYOUT)
    # Restore stream dimension before multiplying: its four distinct weight
    # rows must broadcast across tokens, never across a flattened row index.
    return ttnn.multiply(restored, weight, output_tensor=restored if reuse_output else None)
