# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""Helpers shared by the Chronos-2 ``ttnn.generic_op`` wrappers."""

from __future__ import annotations

import math

import ttnn

TILE = 32
KERNEL_ROOT = "models/experimental/chronos_forecast/ops"

_TILE_BYTES = {
    ttnn.bfloat16: 2048,
    ttnn.bfloat8_b: 1088,
    ttnn.bfloat4_b: 576,
    ttnn.float32: 4096,
}


def kernel_path(op: str, name: str) -> str:
    """Kernel source path relative to TT_METAL_HOME."""
    return f"{KERNEL_ROOT}/{op}/kernels/{name}"


def tile_bytes(dtype) -> int:
    return _TILE_BYTES[dtype]


def padded_volume(t: ttnn.Tensor) -> int:
    return math.prod(t.padded_shape)


def num_tiles(t: ttnn.Tensor) -> int:
    return padded_volume(t) // (TILE * TILE)


def split_rows(device, num_units: int):
    """Row-major split of ``num_units`` over the worker grid, as tt::tt_metal::split_work_to_cores.

    Returns ``(all_cores, [(core, count, first_unit), ...])``.
    """
    grid = device.compute_with_storage_grid_size()
    _, all_cores, group_1, _, per_core_1, per_core_2 = ttnn.split_work_to_cores(grid, num_units, True)
    n_group_1 = group_1.num_cores()
    work, start = [], 0
    for i, core in enumerate(ttnn.corerange_to_cores(all_cores, row_wise=True)):
        n = per_core_1 if i < n_group_1 else per_core_2
        work.append((core, n, start))
        start += n
    return all_cores, work


_BANK_CORES: dict[int, list] = {}


def l1_bank_cores(device) -> list:
    """Logical worker core of each L1 bank; interleaved L1 page p lives in bank p % len(result).

    The allocator shuffles banks over cores, so the map is read back from the page report of a probe
    buffer the first time it is needed on a device.
    """
    key = device.id()
    if key not in _BANK_CORES:
        grid = device.compute_with_storage_grid_size()
        shape = ttnn.Shape([1, 1, TILE, TILE * grid.x * grid.y])
        probe = ttnn.allocate_tensor_on_device(shape, ttnn.bfloat16, ttnn.TILE_LAYOUT, device, ttnn.L1_MEMORY_CONFIG)
        addr = probe.buffer_address()
        pages = {}
        for p in ttnn._ttnn.reports.get_buffer_pages(device):
            if p.address == addr and p.buffer_type == ttnn.BufferType.L1:
                pages.setdefault(p.page_index, ttnn.CoreCoord(p.core_x, p.core_y))
        ttnn.deallocate(probe)
        num_banks = len({(c.x, c.y) for c in pages.values()})
        _BANK_CORES[key] = [pages[b] for b in range(num_banks)]
    return _BANK_CORES[key]


def split_banks(device, num_pages: int):
    """Bank-local split of interleaved L1 pages: bank j's core takes pages j, j + num_banks, ...

    Returns ``(all_cores, [(core, count, first_page), ...], num_banks)``.
    """
    cores = l1_bank_cores(device)
    num_banks = len(cores)
    work = [(core, len(range(j, num_pages, num_banks)), j) for j, core in enumerate(cores) if j < num_pages]
    return core_rects([c for c, _, _ in work]), work, num_banks


def core_rects(cores) -> ttnn.CoreRangeSet:
    """Cover a set of cores with rectangles (x runs per row, stacked over rows) so dispatch can multicast."""
    rows: dict[int, list[int]] = {}
    for c in cores:
        rows.setdefault(c.y, []).append(c.x)
    runs = []
    for y in sorted(rows):
        xs = sorted(rows[y])
        start = xs[0]
        for prev, x in zip(xs, xs[1:] + [None]):
            if x != prev + 1:
                runs.append((start, prev, y))
                start = x
    rects: list[list[int]] = []
    for x0, x1, y in runs:
        top = next((r for r in rects if r[0] == x0 and r[1] == x1 and r[3] == y - 1), None)
        if top is not None:
            top[3] = y
        else:
            rects.append([x0, x1, y, y])
    return ttnn.CoreRangeSet(
        [ttnn.CoreRange(ttnn.CoreCoord(x0, y0), ttnn.CoreCoord(x1, y1)) for x0, x1, y0, y1 in rects]
    )


def is_l1_interleaved(t: ttnn.Tensor) -> bool:
    mem = t.memory_config()
    return mem.buffer_type == ttnn.BufferType.L1 and mem.memory_layout == ttnn.TensorMemoryLayout.INTERLEAVED


def cb(index: int, num_pages: int, dtype, cores: ttnn.CoreRangeSet) -> ttnn.CBDescriptor:
    page = tile_bytes(dtype)
    return ttnn.CBDescriptor(
        total_size=num_pages * page,
        core_ranges=cores,
        format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=index, data_format=dtype, page_size=page)],
    )


def accessor_args(t: ttnn.Tensor) -> list[int]:
    return list(ttnn.TensorAccessorArgs(t).get_compile_time_args())


def require_interleaved_tile(name: str, t: ttnn.Tensor) -> None:
    if t.layout != ttnn.TILE_LAYOUT:
        raise ValueError(f"{name} must be TILE layout, got {t.layout}")
    if t.is_sharded():
        raise ValueError(f"{name} must be interleaved, got {t.memory_config()}")
