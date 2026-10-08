# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Compact predecessor and replacement carry exchange, with fixed collective participation."""

import ttnn


def exchange_convolution_carry(
    projected_qkv: ttnn.Tensor,
    *,
    sequence_parallel_axis: int,
    actual_start: ttnn.Tensor,
    actual_end: ttnn.Tensor | None,
    local_rows: int,
    width: int,
) -> tuple[ttnn.Tensor, ttnn.Tensor]:
    """Return predecessor history and the replacement logical stream carry.

    Both outputs are BF16 row-major DRAM tensors shaped ``[1, 3, local_channels]``. Predecessor history varies by
    SP rank; the final carry is replicated across each SP line. Channels remain sharded across TP. The native
    convolution selects the caller's initial history at the logical sequence start. ``projected_qkv`` is the tiled
    projection whose leading ``width`` columns are the channels. Every selection derives its rows on device from
    the bounds.
    """
    # One fabric exchange gathers the outgoing and local final history from the projection, sends the outgoing
    # rows to the next physical rank and the last valid token's owner's final rows to every rank.
    predecessor, final_carry = ttnn.experimental.kda.exchange_histories(
        projected_qkv,
        width=width,
        actual_start=actual_start,
        local_rows=local_rows,
        actual_end=actual_end,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        sequence_parallel_axis=sequence_parallel_axis,
    )
    return predecessor, final_carry
