# SPDX-License-Identifier: Apache-2.0
"""Thin, teaching-oriented wrappers around ttnn.generic_op.

The whole point of the dojo is that the *learner writes kernels*, not host
boilerplate. This module owns the boilerplate: circular buffers, kernel
descriptors, core-grid maths and work splitting. Exercises describe what they
want in a few lines; this file turns it into a ttnn.ProgramDescriptor.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Iterable, Sequence

import ttnn

TILE_HW = 32
TILE_ELEMS = TILE_HW * TILE_HW

#: Bytes per tile, per data format. Only the formats the dojo uses.
TILE_BYTES = {
    ttnn.bfloat16: 2 * TILE_ELEMS,
    ttnn.float32: 4 * TILE_ELEMS,
}


def tile_size(dtype) -> int:
    try:
        return TILE_BYTES[dtype]
    except KeyError as exc:  # pragma: no cover - guard for future dtypes
        raise ValueError(f"dojo does not know the tile size for {dtype}") from exc


# --------------------------------------------------------------------------
# Core grids and work splitting
# --------------------------------------------------------------------------


def core_range_set(x0: int, y0: int, x1: int, y1: int) -> "ttnn.CoreRangeSet":
    """Inclusive rectangular core range set."""
    return ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(x0, y0), ttnn.CoreCoord(x1, y1))])


def single_core() -> "ttnn.CoreRangeSet":
    return core_range_set(0, 0, 0, 0)


def device_grid(device) -> tuple[int, int]:
    """(width, height) of the device's usable compute grid."""
    g = device.compute_with_storage_grid_size()
    return (g.x, g.y)


def first_n_cores(device, n: int) -> "ttnn.CoreRangeSet":
    """A core range set covering the first `n` cores of the compute grid.

    Cores are taken row-major, so `first_n_cores(dev, 8)` is one full row on a
    grid at least 8 wide. Used by the scaling exercises to vary the core count
    without changing anything else.
    """
    w, h = device_grid(device)
    total = w * h
    if n < 1 or n > total:
        raise ValueError(f"requested {n} cores, device grid has {total}")

    ranges = []
    full_rows, leftover = divmod(n, w)
    if full_rows:
        ranges.append(ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(w - 1, full_rows - 1)))
    if leftover:
        ranges.append(
            ttnn.CoreRange(ttnn.CoreCoord(0, full_rows), ttnn.CoreCoord(leftover - 1, full_rows))
        )
    return ttnn.CoreRangeSet(ranges)


def iter_cores(crs: "ttnn.CoreRangeSet") -> Iterable[tuple[int, int]]:
    """Yield (x, y) for every core in the set, walking x then y per range.

    Only stability matters here, not the specific order: each core is told its
    own tile range explicitly, so any consistent enumeration is correct.
    """
    for rng in crs.ranges():
        for x in range(rng.start.x, rng.end.x + 1):
            for y in range(rng.start.y, rng.end.y + 1):
                yield (x, y)


def num_cores(crs: "ttnn.CoreRangeSet") -> int:
    return sum(1 for _ in iter_cores(crs))


@dataclass(frozen=True)
class CoreWork:
    """How many tiles a single core handles, and where its slice starts."""

    core: tuple[int, int]
    start_tile: int
    n_tiles: int


def split_tiles(crs: "ttnn.CoreRangeSet", total_tiles: int) -> list[CoreWork]:
    """Spread `total_tiles` over the cores in `crs` as evenly as possible.

    Cores that would get zero tiles are dropped from the result, so callers can
    use `len(...)` to find how many cores actually participate. The first
    `total_tiles % n` cores each take one extra tile.
    """
    cores = list(iter_cores(crs))
    if not cores:
        raise ValueError("empty core range set")
    n = len(cores)
    base, extra = divmod(total_tiles, n)

    work: list[CoreWork] = []
    cursor = 0
    for i, core in enumerate(cores):
        count = base + (1 if i < extra else 0)
        if count == 0:
            continue
        work.append(CoreWork(core=core, start_tile=cursor, n_tiles=count))
        cursor += count
    assert cursor == total_tiles, (cursor, total_tiles)
    return work


def cores_used(crs: "ttnn.CoreRangeSet", total_tiles: int) -> "ttnn.CoreRangeSet":
    """The subset of `crs` that `split_tiles` would actually give work to.

    Kernels must only be placed on cores that receive runtime args, otherwise
    the idle cores run with garbage args.
    """
    used = [w.core for w in split_tiles(crs, total_tiles)]
    return ttnn.CoreRangeSet(
        [ttnn.CoreRange(ttnn.CoreCoord(x, y), ttnn.CoreCoord(x, y)) for (x, y) in used]
    )


# --------------------------------------------------------------------------
# Circular buffers
# --------------------------------------------------------------------------


def cb(
    index: int,
    core_ranges: "ttnn.CoreRangeSet",
    dtype=ttnn.bfloat16,
    n_pages: int = 2,
) -> "ttnn.CBDescriptor":
    """A tile-granular circular buffer holding `n_pages` tiles.

    `n_pages` is the depth knob the perf lessons care about: 1 page means the
    producer and consumer cannot overlap at all, 2 pages is classic double
    buffering, more is deeper pipelining at the cost of L1.
    """
    page = tile_size(dtype)
    fmt = ttnn.CBFormatDescriptor(buffer_index=index, data_format=dtype, page_size=page)
    return ttnn.CBDescriptor(
        total_size=page * n_pages,
        core_ranges=core_ranges,
        format_descriptors=[fmt],
    )


# --------------------------------------------------------------------------
# Kernels
# --------------------------------------------------------------------------

#: Set by the runner so exercises can refer to kernels by bare filename.
_KERNEL_DIR: str | None = None


def set_kernel_dir(path: str) -> None:
    global _KERNEL_DIR
    _KERNEL_DIR = os.path.abspath(path)


def kernel_path(name: str) -> str:
    """Resolve a kernel filename against the directory being graded.

    Absolute paths are what tt-metal's resolver prefers (it checks CWD first,
    which would otherwise silently pick up a same-named file elsewhere).
    """
    if os.path.isabs(name):
        return name
    if _KERNEL_DIR is None:
        raise RuntimeError("kernel dir not set; the runner should call set_kernel_dir()")
    return os.path.join(_KERNEL_DIR, name)


@dataclass
class RtArgs:
    """Per-core runtime args, built incrementally then converted for ttnn."""

    _rows: list[tuple[tuple[int, int], list[int]]] = field(default_factory=list)

    def set(self, core: tuple[int, int], args: Sequence[int]) -> None:
        self._rows.append((core, [int(a) for a in args]))

    def build(self) -> "ttnn.RuntimeArgs":
        out = ttnn.RuntimeArgs()
        for (x, y), args in self._rows:
            out[x][y] = args
        return out


def reader_kernel(source, core_ranges, ct_args=(), rt_args=None, defines=()):
    return _kernel(source, core_ranges, ct_args, rt_args, defines, ttnn.ReaderConfigDescriptor())


def writer_kernel(source, core_ranges, ct_args=(), rt_args=None, defines=()):
    return _kernel(source, core_ranges, ct_args, rt_args, defines, ttnn.WriterConfigDescriptor())


def compute_kernel(
    source,
    core_ranges,
    ct_args=(),
    rt_args=None,
    defines=(),
    math_fidelity=ttnn.MathFidelity.HiFi4,
    fp32_dest_acc_en: bool = False,
    math_approx_mode: bool = False,
):
    cfg = ttnn.ComputeConfigDescriptor()
    cfg.math_fidelity = math_fidelity
    cfg.fp32_dest_acc_en = fp32_dest_acc_en
    cfg.math_approx_mode = math_approx_mode
    return _kernel(source, core_ranges, ct_args, rt_args, defines, cfg)


def _kernel(source, core_ranges, ct_args, rt_args, defines, config):
    return ttnn.KernelDescriptor(
        kernel_source=kernel_path(source),
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=core_ranges,
        compile_time_args=[int(a) for a in ct_args],
        defines=list(defines),
        runtime_args=(rt_args.build() if isinstance(rt_args, RtArgs) else (rt_args or [])),
        config=config,
    )


def program(kernels, cbs, semaphores=()) -> "ttnn.ProgramDescriptor":
    return ttnn.ProgramDescriptor(
        kernels=list(kernels), semaphores=list(semaphores), cbs=list(cbs)
    )


def accessor_args(tensor) -> list[int]:
    """Compile-time args describing how a tensor is laid out in memory.

    Kernels decode these with `TensorAccessorArgs<N>()`; they carry the sharding
    / interleaving scheme so one kernel body works for several memory configs.
    """
    return list(ttnn.TensorAccessorArgs(tensor).get_compile_time_args())
