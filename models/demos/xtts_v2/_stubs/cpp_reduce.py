# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""C++ Metalium reduce-over-last-axis kernel (cpp rung, GUIDELINES/12).

cpp rung for the memory-bound `ReduceDeviceOperation` (the conditioning
group_norm32 mean/mean_sq). A last-axis reduction is `X[R,K] @ ones[K,1]`, so
this reuses the PROVEN multi-core output-tile-partitioned C++ Metalium matmul
(`cpp_matmul._run_mm`, HiFi4 + fp32 dest acc, run via ttnn.generic_op /
ProgramDescriptor) with an all-ones second operand to sum each row; `mean_last`
scales by 1/W. I/O contract matches the stock reduce: bf16 TILE in/out.
"""

from __future__ import annotations

import ttnn

from models.demos.xtts_v2._stubs.cpp_matmul import _run_mm, TILE


def build_mean_last(device):
    """Return fn(x3d[1,G,W]) -> mean over last axis [1,G,1] via the C++ matmul kernel."""
    _ones_cache = {}

    def _ones(w):
        o = _ones_cache.get(w)
        if o is None:
            import torch
            o = ttnn.from_torch(
                torch.ones(w, TILE, dtype=torch.bfloat16),
                dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            _ones_cache[w] = o
        return o

    def mean_last(x3d):
        g = int(x3d.shape[1])
        w = int(x3d.shape[2])
        x2d = ttnn.reshape(x3d, (g, w))
        if x2d.get_dtype() != ttnn.bfloat16:
            x2d = ttnn.typecast(x2d, ttnn.bfloat16)
        y = ttnn.empty([g, TILE], ttnn.bfloat16, ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG)
        _run_mm(x2d, _ones(w), y, device)
        col = ttnn.slice(y, [0, 0], [g, 1])            # [G,1] rowsum
        mean = ttnn.multiply(col, 1.0 / float(w))
        return ttnn.reshape(mean, (1, g, 1))

    return mean_last
