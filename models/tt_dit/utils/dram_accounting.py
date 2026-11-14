# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Per-chip device DRAM by category, callable from Python at any point (no env flag).

`dram_by_category` splits one chip's allocated DRAM into weights, CCL buffers, tracer I/O and
other (the remainder: module-held tensors, live activations and the program cache's kernel
binaries, which Python cannot reach), and reports the trace region, a separate carve-out, beside
it. The mesh allocator is lockstep, so every chip holds the same: the numbers are one chip's.
Each tensor is counted once, by its buffer address, in the first category that reaches it.
"""

from __future__ import annotations

import gc
from typing import TYPE_CHECKING, Any

import ttnn

from .dram_probe import _bytes_addr, _device_dram_tensor, _walk
from .tracing import StateTensor, live_tracers, tracer_io_tensors

if TYPE_CHECKING:
    from collections.abc import Iterable

CATEGORIES = ("weights", "ccl", "tracer_io", "other")


def _view(mesh_device: Any, buffer_type: Any) -> dict[str, int]:
    v = ttnn.get_memory_view(mesh_device, buffer_type)
    n = int(v.num_banks)
    return {
        "allocated": int(v.total_bytes_allocated_per_bank) * n,
        "free": int(v.total_bytes_free_per_bank) * n,
        "largest_free": int(v.largest_contiguous_bytes_free_per_bank) * n,
        "size": int(v.total_bytes_per_bank) * n,
    }


def dram_by_category(
    mesh_device: Any,
    *,
    weight_roots: Iterable[Any] = (),
    ccl_managers: Iterable[Any] = (),
    state_tensors: Iterable[Any] = (),
) -> dict[str, Any]:
    """One chip's DRAM in bytes: {"weights", "ccl", "tracer_io", "other", "allocated", "free",
    "largest_free", "trace_used", "trace_size", "trace_largest_free", "program_cache_entries"}.

    weight_roots: modules whose ``Parameter`` tensors count as weights.
    ccl_managers: CCLManagers whose shape-keyed buffer caches count as CCL.
    state_tensors: StateTensors (or tensors) that are trace I/O, beside every live Tracer's I/O.
    A gc runs first, so tensors nobody references are freed before the reading.
    """
    gc.collect()
    seen: set[int] = set()
    out = dict.fromkeys(CATEGORIES, 0)

    def take(category: str, t: Any) -> None:
        if not _device_dram_tensor(t):
            return
        nbytes, addr = _bytes_addr(t)
        key = addr if addr is not None else id(t)
        if nbytes and key not in seen:
            seen.add(key)
            out[category] += nbytes

    found: list[tuple[str, Any, bool]] = []
    walked: set[int] = set()
    for i, root in enumerate(weight_roots):
        _walk(root, f"root{i}", walked, found, False, 0)
    for _path, t, is_weight in found:
        if is_weight:
            take("weights", t)
    for m in {id(m): m for m in ccl_managers}.values():
        for t in m._cached_buffer_entries():
            take("ccl", t)
    for tracer in live_tracers():
        for t in tracer_io_tensors(tracer):
            take("tracer_io", t)
    for st in state_tensors:
        take("tracer_io", st.data if isinstance(st, StateTensor) else st)

    dram = _view(mesh_device, ttnn.BufferType.DRAM)
    out["other"] = max(0, dram["allocated"] - out["weights"] - out["ccl"] - out["tracer_io"])
    out.update(allocated=dram["allocated"], free=dram["free"], largest_free=dram["largest_free"])
    try:
        trace = _view(mesh_device, ttnn.BufferType.TRACE)
        out.update(
            trace_used=trace["allocated"], trace_size=trace["size"], trace_largest_free=trace["largest_free"]
        )
    except Exception:  # noqa: BLE001 - no trace region on this device
        out.update(trace_used=0, trace_size=0, trace_largest_free=0)
    try:
        out["program_cache_entries"] = int(mesh_device.num_program_cache_entries())
    except Exception:  # noqa: BLE001 - an older binding
        out["program_cache_entries"] = -1
    return out


def format_row(label: str, row: dict[str, Any]) -> str:
    """One line of the table in MB, for logs."""
    mb = lambda k: f"{row[k] / 2**20:9.1f}"  # noqa: E731
    cols = " ".join(f"{k}={mb(k)}" for k in (*CATEGORIES, "allocated", "largest_free", "trace_used"))
    return f"DRAM/chip [{label}] {cols} MB programs={row['program_cache_entries']}"
