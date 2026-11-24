# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""``row_gather``: the per-block hidden all-gather of a decode row, moving only the valid rows; the program is ``row_gather2``'s.

Stands in for ``ttnn.all_gather(x, dim=3, cluster_axis=<line>, memory_config=<caller's>)`` of a local ``[1, 1, rows,
W]`` bf16 TILE tensor (``rows`` 1..32 valid rows of one padded tile row) -> ``[1, 1, rows, W * TP_SIZE]``, device d's
shard at columns ``[W * d, W * (d + 1))`` (page ``t + (W / 32) * d`` of the tile layout).  Tolerance class BITWISE.  ``QWEN38_ROW_GATHER=0`` keeps ``ttnn.all_gather`` everywhere;
a gather whose packed rows do not fit one fabric packet payload (``ttnn.get_tt_fabric_max_payload_size_bytes``) or whose shapes are outside the
contract takes ``ttnn.all_gather`` too (``admits``).
"""

from __future__ import annotations

import os

import ttnn

from ..registry import BITWISE, FusedKernel, register

NAME = "row_gather"
ENV = "QWEN38_ROW_GATHER"


def enabled_by_env(environ=None) -> bool:
    """QWEN38_ROW_GATHER: 1 (default) = the ``row_gather2`` program, 0 = the stock all_gather."""

    return (os.environ if environ is None else environ).get(ENV, "1") == "1"


def prepare(mesh, tiles: int) -> None:
    """Create the program's persistent landing buffer and semaphores (before any trace capture)."""

    from .. import row_gather2

    row_gather2.prepare(mesh, tiles)


def admits(hidden, *, cluster_axis=1, memory_config=None, live_rows=None) -> bool:
    """``row_gather2``'s contract (``[1, 1, rows, W]`` bf16 TILE on the TP line, packed rows that fit one packet per
    link); a full tile row takes the all-gather: nothing to trim."""

    if not enabled_by_env():
        return False
    from .. import row_gather2

    return row_gather2.admits(hidden, cluster_axis=cluster_axis, memory_config=memory_config, live_rows=live_rows)


def composed(hidden, *, cluster_axis=1, memory_config=None, live_rows=None):
    return ttnn.all_gather(hidden, dim=3, cluster_axis=cluster_axis, memory_config=memory_config)


def row_gather(hidden, *, cluster_axis=1, memory_config=None, live_rows=None):
    shape = tuple(int(v) for v in hidden.shape)
    if not admits(hidden, cluster_axis=cluster_axis, memory_config=memory_config, live_rows=live_rows):
        raise ValueError(f"row_gather does not admit {shape} {hidden.dtype} {hidden.layout}")
    from .. import row_gather2

    return row_gather2.row_gather(hidden, cluster_axis=cluster_axis, memory_config=memory_config, live_rows=live_rows)


def row_gather_or_all_gather(hidden, *, cluster_axis=1, memory_config=None, live_rows=None):
    """The call sites' entry: the packed program within its contract, ``ttnn.all_gather`` otherwise."""

    if admits(hidden, cluster_axis=cluster_axis, memory_config=memory_config, live_rows=live_rows):
        return row_gather(hidden, cluster_axis=cluster_axis, memory_config=memory_config, live_rows=live_rows)
    return composed(hidden, cluster_axis=cluster_axis, memory_config=memory_config)


register(
    FusedKernel(
        name=NAME,
        replaces="ttnn.all_gather(dim=3) of the per-block decode hidden (gdn / moe / qsa _all_gather_hidden), one "
        "program with only the valid rows crossing the fabric",
        tolerance=BITWISE,
        fused=row_gather,
        composed=composed,
        gate=None,
        admits=admits,
    )
)
