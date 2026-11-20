# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""FFN_FEED (P.36): the attn_residual pass writes the MoE head's input on every chip of the TP group.

Under OWN_ROWS each chip holds S/(SP TP) whole rows; the MoE block's fused head takes the column split [S/SP, H/TP].
With NORM_IN_MHC the attn_residual's mhc_post (collapse_pre + norm_weight) already writes ffn_norm's output; with
FFN_FEED its feed_output option also sends each 1280-column block of it into the TP peer's head input (a persistent
buffer at the same address on every chip, tt/moe_block_main_adapter.py feed_buffer) over the fabric, so the MoE layer
runs no all_to_all_async_generic. This module makes the op's two global semaphores (ready, arrival) once a mesh, at
load, before the MoE blocks place their persistent L1 (they would move under a later allocation)."""

from __future__ import annotations

import ttnn

from .cores import worker_cores
from .settings import settings

_SEMS: dict = {}  # id(mesh) -> (mesh, [handles], [addresses])


def links() -> int:
    return int(settings.get("FFN_FEED_LINKS"))


def prepare(mesh) -> list[int] | None:
    """The feed's [ready, arrival] global-semaphore addresses for ``mesh`` (made on the first call: over the capped
    worker grid, at 0, a device sync so every chip's zero has landed before any chip's first call). None when
    FFN_FEED is off."""
    if settings.get("FFN_FEED") != "on":
        return None
    hit = _SEMS.get(id(mesh))
    if hit is None or hit[0] is not mesh:
        crs = worker_cores(mesh)
        handles = [ttnn.create_global_semaphore(mesh, crs, 0) for _ in range(2)]
        ttnn.synchronize_device(mesh)  # (load time, once a mesh)
        hit = _SEMS[id(mesh)] = (mesh, handles, [int(ttnn.get_global_semaphore_address(h)) for h in handles])
    return hit[2]
