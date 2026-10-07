# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Sequence-parallel all-gather used by every KDA collective exchange."""

from __future__ import annotations

import ttnn


def sp_all_gather(
    tensor: ttnn.Tensor, *, outputs: dict[tuple, ttnn.Tensor], name: str, dim: int, cluster_axis: int
) -> ttnn.Tensor:
    """All-gather ``tensor`` over ``cluster_axis`` into a persistent DRAM output.

    ``outputs`` holds the persistent outputs keyed by name, shape and format; its owner (one per mesh) allocates them
    on the first (untraced) call and reuses them in every later call and trace capture. The result aliases that
    output: consume it before the next gather with the same ``name`` and never deallocate it. ``name`` keeps
    same-shaped gathers that are live together apart.
    """
    device = tensor.device()
    shape = list(tensor.shape)
    shape[dim] *= tuple(device.shape)[cluster_axis]
    key = (name, tuple(shape), str(tensor.dtype), str(tensor.layout), dim, cluster_axis)
    output = outputs.get(key)
    if output is None:
        output = ttnn.zeros(
            shape, dtype=tensor.dtype, layout=tensor.layout, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        outputs[key] = output
    if tensor.memory_config().buffer_type != ttnn.BufferType.DRAM:
        tensor = ttnn.to_memory_config(tensor, ttnn.DRAM_MEMORY_CONFIG)
    return ttnn.experimental.high_bw_all_gather(tensor, dim, output, cluster_axis=cluster_axis)
