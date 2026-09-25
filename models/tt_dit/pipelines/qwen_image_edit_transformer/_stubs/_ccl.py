# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Whole-mesh collectives for the TP=N transformer ports.

The ports shard with ShardTensorToMesh over ALL devices of the mesh (row-major device order), so a
TP collective must span every device. On a 1xN mesh that is one collective on axis 1; on an RxC mesh
it is axis 1 then axis 0. all_gather in that order concatenates the shards back in row-major order,
and a reduce over both axes sums all R*C partials.

Reduction precision: ttnn.all_reduce / ttnn.reduce_scatter round float32 partials. Measured on this
T3K 2x4 mesh: 7e-3..1.1e-2 max abs error on O(3) float32 sums of [2,64,256]..[4,128,3072] per chip.
That is bf16-level rounding, on EVERY residual update of every block. The ports keep their residual
streams in float32 for a reason, so the reduce here gathers the partials (all_gather moves bits
exactly) and adds them in float32. Measured error of that path: 0.0 on the same tensors. The gather
is chunked along the token dim so the N-fold temporary stays bounded.
"""

from __future__ import annotations

import ttnn

_CHUNK_BYTES = 48 * 1024 * 1024  # per-partial bytes gathered at once


def mesh_axes(device):
    try:
        shape = tuple(device.shape)
    except (AttributeError, TypeError):
        return []
    if len(shape) != 2:
        return [1] if device.get_num_devices() > 1 else []
    return [ax for ax in (1, 0) if shape[ax] > 1]


def _gather_sum_once(y, ax, n):
    shape = list(y.shape)
    g = ttnn.all_gather(
        ttnn.reshape(y, [1] + shape), dim=0, cluster_axis=ax, num_links=1, topology=ttnn.Topology.Linear
    )
    out = None
    for i in range(n):
        part = ttnn.slice(g, [i] + [0] * len(shape), [i + 1] + shape)
        out = part if out is None else ttnn.add(out, part)
    ttnn.deallocate(g)
    return ttnn.reshape(out, shape)


def _gather_sum(y, device, ax):
    """Exact float32 reduce over mesh axis `ax`: gather every partial on a new leading dim, then add."""
    n = tuple(device.shape)[ax]
    shape = list(y.shape)
    nbytes = 4 if y.dtype == ttnn.float32 else 2
    for s in shape:
        nbytes *= s
    if nbytes <= _CHUNK_BYTES or len(shape) < 2 or shape[-2] <= 32:
        return _gather_sum_once(y, ax, n)
    # chunk along the token (second-to-last) dim in whole tiles
    rows = shape[-2]
    per_row = nbytes // rows
    step = max(32, (_CHUNK_BYTES // per_row) // 32 * 32)
    outs = []
    for lo in range(0, rows, step):
        hi = min(rows, lo + step)
        start = [0] * len(shape)
        end = list(shape)
        start[-2], end[-2] = lo, hi
        outs.append(_gather_sum_once(ttnn.slice(y, start, end), ax, n))
    return ttnn.concat(outs, dim=len(shape) - 2)


def all_reduce(y, device):
    """Sum of the partials over every device of the mesh (exact float32, see module docstring)."""
    for ax in mesh_axes(device):
        y = _gather_sum(y, device, ax)
    return y


def all_gather(y, device, dim=-1):
    dim = dim % len(y.shape)
    for ax in mesh_axes(device):
        y = ttnn.all_gather(y, dim=dim, cluster_axis=ax, num_links=1, topology=ttnn.Topology.Linear)
    return y
