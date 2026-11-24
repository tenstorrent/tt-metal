# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""``mm_stream``: the decode DRAM-sharded linear's entry and the placement helpers and kernels that ``mm_stream2`` shares.

The decode linears of the model call :func:`linear`: with ``QWEN38_MM_STREAM2`` on (default) the re-laid streamed program
of ``mm_stream2``, else the stock ``ttnn.linear``.  ``kernels/shard_mm_stream.cpp``, ``writer_mm_stream.cpp`` and
``compute_mm_stream.cpp`` are the streamed program's shard, writer and compute kernels (``mm_stream2`` builds the program
around them); the column grouping and the greedy bank -> core assignment below are its placement.
"""

from __future__ import annotations

import math

import ttnn

from .. import program as fp

NAME = "mm_stream"
CB_X, CB_W, CB_OUT, CB_STAGE = 0, 1, 2, 3
SEM_X, SEM_HUB = 0, 1
FID = {
    ttnn.bfloat8_b: ttnn.MathFidelity.HiFi2,
    ttnn.bfloat16: ttnn.MathFidelity.HiFi4,
    ttnn.bfloat4_b: ttnn.MathFidelity.LoFi,
}
_WORKERS: dict = {}


def column_groups(n: int, nb: int, banks: int, cols: int):
    """``(bank, first column in the bank, ncols, first global tile)`` of every core: the real tile columns of each bank's
    shard (the padding tiles of the last banks are not computed) in groups of ``cols``."""
    total = math.ceil(n / fp.TILE)
    groups = []
    for b in range(banks):
        real = max(0, min(nb, total - b * nb))
        for c0 in range(0, real, cols):
            groups.append((b, c0, min(cols, real - c0), b * nb + c0))
    return groups


def bank_workers(mesh) -> dict:
    """DRAM bank -> (x, y) logical worker of the reference die."""
    key = id(mesh)
    if key not in _WORKERS:
        a = mesh.get_optimal_dram_bank_to_logical_worker_assignment(ttnn.NOC.NOC_0, ttnn.MeshCoordinate(0, 0))
        _WORKERS[key] = {b: (a[b].x, a[b].y) for b in a}
    return _WORKERS[key]


def _assign(rect_cores, groups, banks, workers):
    """Greedy group -> core assignment (groups in bank order, each the nearest free core to its bank's reader worker) and its
    total Manhattan distance."""
    free, assign, cost = list(rect_cores), {}, 0
    for i in sorted(range(len(groups)), key=lambda i: groups[i][0]):
        w = workers[banks[groups[i][0]]]
        best = min(free, key=lambda c: abs(c[0] - w[0]) + abs(c[1] - w[1]))
        cost += abs(best[0] - w[0]) + abs(best[1] - w[1])
        free.remove(best)
        assign[i] = best
    return assign, cost


def linear(act, weight, *, memory_config, program_config, compute_kernel_config, f32_dram=False, **rest):
    """``ttnn.linear`` (same arguments) for the decode linears; with ``QWEN38_MM_STREAM2`` on (default) the re-laid streamed
    program of ``mm_stream2`` (which falls back to the stock op outside its contract)."""
    from .. import mm_stream2

    if mm_stream2.enabled():
        return mm_stream2.linear(
            act,
            weight,
            memory_config=memory_config,
            program_config=program_config,
            compute_kernel_config=compute_kernel_config,
            f32_dram=f32_dram,
            **rest,
        )
    return ttnn.linear(
        act,
        weight,
        memory_config=memory_config,
        program_config=program_config,
        compute_kernel_config=compute_kernel_config,
        **rest,
    )
