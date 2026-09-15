# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Device-memory probes gated by ``DIFFVAE_MEM_LOG``.

Each probe is a no-op unless the variable is set, so a model can leave calls at its allocation
milestones and a normal run pays nothing. Set it to find where DRAM is held between stages and
which collectives' persistent buffers hold it.
"""

from __future__ import annotations

import os

from loguru import logger

import ttnn

ENABLED = bool(os.environ.get("DIFFVAE_MEM_LOG"))


def log_dram(mesh_device, label: str) -> None:
    """Log allocated, total and largest contiguous free DRAM across the mesh device's banks."""
    if not ENABLED:
        return
    view = ttnn.get_memory_view(mesh_device, ttnn.BufferType.DRAM)
    banks = view.num_banks
    logger.info(
        f"[dram] {label}: allocated {view.total_bytes_allocated_per_bank * banks / 2**30:6.2f} GiB"
        f" of {view.total_bytes_per_bank * banks / 2**30:.2f} GiB,"
        f" largest contiguous free {view.largest_contiguous_bytes_free_per_bank * banks / 2**30:5.2f} GiB"
    )


def log_ccl_cache(ccl_manager, label: str) -> None:
    """Itemise the CCL manager's persistent ping-pong buffers by kind and shape."""
    if not ENABLED or ccl_manager is None:
        return
    rows = ccl_manager.ping_pong_buffer_report()
    total = sum(nbytes for nbytes, _ in rows)
    lines = [f"  {nbytes / 2**20:8.1f} MiB  {desc}" for nbytes, desc in rows]
    logger.info(f"[ccl-cache] {label}: {total / 2**30:.2f} GiB per chip in {len(rows)} shape(s)\n" + "\n".join(lines))
