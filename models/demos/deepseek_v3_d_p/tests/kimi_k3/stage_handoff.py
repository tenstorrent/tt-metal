# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The pipeline-stage handoff saved to a file, for running Kimi-K3 / K3.1 stage by stage on one galaxy.

What crosses a rank boundary is exactly what `prefill_runner`'s D2D socket carries: the upstream rank's
output activation (the live stream plus every AttnRes snapshot sealed upstream) as the socket's global tensor
`activation_global_spec(rows, hidden, planes)` -- bf16 `[1, planes, chunk, hidden]` -- with rows sharded over the
SP mesh axis and the hidden width over the TP axis (the runner's `D2D_MAPPER_CONFIG`), so chip (r, c) of one
stage feeds chip (r, c) of the next. The file is that global tensor's raw bf16 bytes and nothing else: the
receiver knows the shape the way it sizes its socket, from the boundary, so a file of the wrong size fails
to load instead of being misread. The per-chunk metadata the socket forwards beside it is not stored; the
receiving stage builds the same (slot, start, end) record itself.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

import ttnn

# Mesh axis -> tensor dim of the D2D payload: rows on the SP axis (0), hidden width on the TP axis (1).
_SHARD_DIMS = (2, 3)


def compose_activation(mesh_device, activation: ttnn.Tensor) -> torch.Tensor:
    """The rank's outbound activation on host as the socket's global tensor."""
    composer = ttnn.ConcatMesh2dToTensor(mesh_device, dims=_SHARD_DIMS, mesh_shape=tuple(mesh_device.shape))
    host = ttnn.to_torch(activation, mesh_composer=composer)
    assert host.dtype == torch.bfloat16, f"the D2D payload is bf16, got {host.dtype}"
    return host


def save_activation(mesh_device, activation: ttnn.Tensor, path: Path) -> torch.Tensor:
    """Write the rank's outbound activation as the socket's global tensor; returns it on host."""
    host = compose_activation(mesh_device, activation)
    host.contiguous().view(torch.int16).numpy().tofile(path)
    return host


def load_activation(mesh_device, path: Path, *, planes: int, rows: int, hidden: int) -> ttnn.Tensor:
    """The inbound activation for a rank whose socket is `[1, planes, rows, hidden]`, placed as it arrives."""
    raw = np.fromfile(path, dtype=np.int16)
    shape = (1, planes, rows, hidden)
    assert raw.size == int(np.prod(shape)), f"{path} holds {raw.size} bf16 values; the socket carries {shape}"
    host = torch.from_numpy(raw).view(torch.bfloat16).reshape(shape)
    return ttnn.from_torch(
        host,
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=_SHARD_DIMS, mesh_shape=tuple(mesh_device.shape)),
    )
