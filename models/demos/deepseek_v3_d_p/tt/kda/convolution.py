# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Compact predecessor and replacement carry exchange, with fixed collective participation."""

import ttnn
from models.demos.deepseek_v3_d_p.tt.kda.device_chronology import DeviceChronology


def exchange_convolution_carry(
    projected_qkv: ttnn.Tensor,
    *,
    sequence_parallel_axis: int,
    chronology: DeviceChronology,
) -> tuple[ttnn.Tensor, ttnn.Tensor]:
    outgoing = chronology.select_rows(projected_qkv, 1)
    gathered = ttnn.all_gather(
        outgoing, dim=1, cluster_axis=sequence_parallel_axis, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    predecessor = chronology.select_rows(gathered, 2)
    batch, rows, width = projected_qkv.shape
    physical_end = ttnn.slice(projected_qkv, (0, rows - 3, 0), (batch, rows, width))
    finals = ttnn.all_broadcast(physical_end, cluster_axis=sequence_parallel_axis)
    candidates = ttnn.concat(finals, dim=1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    final_carry = chronology.select_rows(candidates, 3)
    return predecessor, final_carry
