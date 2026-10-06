# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Sequence-parallel all-gather used by every KDA collective exchange."""

from __future__ import annotations

import ttnn

# Persistent all-gather outputs keyed by name, mesh, shape and format. They are allocated on the
# first (untraced) call and reused by every later call and trace capture.
_GATHER_OUTPUTS: dict[tuple, ttnn.Tensor] = {}


def sp_all_gather(tensor: ttnn.Tensor, *, name: str, dim: int, cluster_axis: int) -> ttnn.Tensor:
    """All-gather ``tensor`` over ``cluster_axis`` into a persistent DRAM output.

    The result aliases that output: consume it before the next gather with the same ``name`` and
    never deallocate it. ``name`` keeps same-shaped gathers that are live together apart.
    """
    device = tensor.device()
    shape = list(tensor.shape)
    shape[dim] *= tuple(device.shape)[cluster_axis]
    key = (name, id(device), tuple(shape), str(tensor.dtype), str(tensor.layout), dim, cluster_axis)
    output = _GATHER_OUTPUTS.get(key)
    if output is None:
        output = ttnn.zeros(
            shape, dtype=tensor.dtype, layout=tensor.layout, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        _GATHER_OUTPUTS[key] = output
    if tensor.memory_config().buffer_type != ttnn.BufferType.DRAM:
        tensor = ttnn.to_memory_config(tensor, ttnn.DRAM_MEMORY_CONFIG)
    return ttnn.experimental.high_bw_all_gather(tensor, dim, output, cluster_axis=cluster_axis)
