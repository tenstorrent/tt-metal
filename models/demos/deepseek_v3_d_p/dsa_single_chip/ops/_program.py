# SPDX-License-Identifier: Apache-2.0
"""Thin helpers over ``ttnn.generic_op`` for the DSA single-shape ops (patterned on
models/demos/blackhole/qwen38_flash_next/ttnn/fused/program.py)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import ttnn

TILE = ttnn.TILE_SIZE
CB_COUNT = 64
REPO_ROOT = Path(__file__).resolve().parents[5]
KERNEL_DIR = "models/demos/deepseek_v3_d_p/dsa_single_chip/ops/kernels"

TILE_BYTES = {
    ttnn.bfloat16: 2048,
    ttnn.float32: 4096,
    ttnn.uint32: 4096,
    ttnn.bfloat8_b: 1088,
    ttnn.bfloat4_b: 576,
}


def kernel_path(file: str) -> str:
    """Kernel source relative to the repo root (what generic_op's FILE_PATH expects); must exist."""
    rel = f"{KERNEL_DIR}/{file}"
    if not (REPO_ROOT / rel).is_file():
        raise FileNotFoundError(f"{rel} not found under {REPO_ROOT}")
    return rel


@dataclass(frozen=True)
class CoreWork:
    core: ttnn.CoreCoord
    start: int
    count: int


def split_work(units: int, device, cores: int | None = None) -> list[CoreWork]:
    """``units`` items over the compute grid in column-major linear order, evenly; only cores with work."""
    grid = device.compute_with_storage_grid_size()
    most = min(units, grid.x * grid.y)
    cores = most if cores is None else cores
    assert 1 <= cores <= most, (units, cores, grid)
    base, extra = divmod(units, cores)
    work, start = [], 0
    for i in range(cores):
        count = base + (1 if i < extra else 0)
        work.append(CoreWork(ttnn.CoreCoord(i // grid.y, i % grid.y), start, count))
        start += count
    return work


def core_set(work: Iterable[CoreWork]) -> ttnn.CoreRangeSet:
    return ttnn.CoreRangeSet([ttnn.CoreRange(w.core, w.core) for w in work])


def core_rect(work: Sequence[CoreWork], device) -> ttnn.CoreRangeSet:
    """Cores of ``work`` (column-major prefix of the grid, as split_work produces) as the fewest CoreRanges
    (full columns + one partial column). Dispatch cost grows with the range count."""
    grid = device.compute_with_storage_grid_size()
    n = len(work)
    assert [(w.core.x, w.core.y) for w in work] == [(i // grid.y, i % grid.y) for i in range(n)]
    cols, rest = divmod(n, grid.y)
    ranges = []
    if cols:
        ranges.append(ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(cols - 1, grid.y - 1)))
    if rest:
        ranges.append(ttnn.CoreRange(ttnn.CoreCoord(cols, 0), ttnn.CoreCoord(cols, rest - 1)))
    return ttnn.CoreRangeSet(ranges)


def accessor_args(tensor) -> list[int]:
    return list(ttnn.TensorAccessorArgs(tensor).get_compile_time_args())


def cb(index: int, dtype, pages: int, cores: ttnn.CoreRangeSet, page_bytes: int | None = None) -> ttnn.CBDescriptor:
    pb = TILE_BYTES[dtype] if page_bytes is None else page_bytes
    return ttnn.CBDescriptor(
        total_size=pages * pb,
        core_ranges=cores,
        format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=index, data_format=dtype, page_size=pb)],
    )


def _kernel(source, cores, ct_args, rt_args, defines, config, named=(), common=()):
    named = named.items() if isinstance(named, dict) else named
    return ttnn.KernelDescriptor(
        kernel_source=source,
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=cores,
        compile_time_args=[int(a) for a in ct_args],
        named_compile_time_args=[(str(k), int(v)) for k, v in named],
        defines=list(defines),
        runtime_args=[(core, [int(a) for a in args]) for core, args in rt_args],
        common_runtime_args=[int(a) for a in common],
        config=config,
    )


def reader(source, cores, ct_args, rt_args, defines=(), named=(), common=()):
    return _kernel(source, cores, ct_args, rt_args, defines, ttnn.ReaderConfigDescriptor(), named, common)


def writer(source, cores, ct_args, rt_args, defines=(), named=(), common=()):
    return _kernel(source, cores, ct_args, rt_args, defines, ttnn.WriterConfigDescriptor(), named, common)


def compute(
    source,
    cores,
    ct_args,
    rt_args=(),
    defines=(),
    named=(),
    *,
    fidelity=ttnn.MathFidelity.HiFi2,
    fp32_dest: bool = False,
    approx: bool = False,
    dst_full_sync: bool = False,
    bfp8_pack_precise: bool = False,
    common=(),
):
    cfg = ttnn.ComputeConfigDescriptor(
        math_fidelity=fidelity,
        math_approx_mode=approx,
        fp32_dest_acc_en=fp32_dest,
        dst_full_sync_en=dst_full_sync,
        bfp8_pack_precise=bfp8_pack_precise,
    )
    return _kernel(source, cores, ct_args, rt_args, defines, cfg, named, common)


def program(kernels: Sequence, cbs: Sequence = (), semaphores: Sequence = ()) -> ttnn.ProgramDescriptor:
    return ttnn.ProgramDescriptor(kernels=list(kernels), semaphores=list(semaphores), cbs=list(cbs))


def run(io_tensors: Sequence, descriptor):
    """Inputs first, pre-allocated output(s) last; returns the last tensor."""
    return ttnn.generic_op(list(io_tensors), descriptor)


def alloc(shape: Sequence[int], dtype, layout, device, mem=ttnn.DRAM_MEMORY_CONFIG):
    return ttnn.allocate_tensor_on_device(ttnn.Shape(list(shape)), dtype, layout, device, mem)
