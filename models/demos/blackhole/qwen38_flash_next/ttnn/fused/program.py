# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Thin layer over ``ttnn.generic_op`` for this model's fused decode kernels.

A fused kernel is one program: kernel sources under ``ttnn/fused/<name>/kernels/*.cpp`` (paths relative to the
repository root), circular buffers, semaphores and per-core runtime args described from Python, run with
``run_program``.

Rows contract: every decode activation is one 32-row tile with ``rows`` valid rows (1 for decode, up to 32 for lanes
and MTP verify); ``rows_of`` reads and checks it.  A kernel takes the padded tile and produces only the valid rows.

generic_op leaves each output with its allocation's default placement, so a builder declares its outputs' placements
with ``program_meta`` and ``run_program`` stamps them after the launch.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import ttnn

TILE = ttnn.TILE_SIZE
FACE = 16
ROWS_MAX = TILE
CB_COUNT = (
    64  # NUM_CIRCULAR_BUFFERS, the maximum over architectures; the JIT wants unpack_to_dest_mode at least that long
)
REPO_ROOT = Path(__file__).resolve().parents[6]
KERNEL_ROOT = "models/demos/blackhole/qwen38_flash_next/ttnn/fused"
ELEMENT_BYTES = {ttnn.bfloat16: 2, ttnn.uint16: 2, ttnn.float32: 4, ttnn.uint32: 4, ttnn.int32: 4}
TILE_BYTES = {
    ttnn.bfloat16: 2048,
    ttnn.uint16: 2048,
    ttnn.float32: 4096,
    ttnn.uint32: 4096,
    ttnn.int32: 4096,
    ttnn.bfloat8_b: 1088,
    ttnn.bfloat4_b: 576,
}


def kernel_source(name: str, file: str) -> str:
    """``ttnn/fused/<name>/kernels/<file>`` relative to the repository root; the file must exist."""

    relative = f"{KERNEL_ROOT}/{name}/kernels/{file}"
    if not (REPO_ROOT / relative).is_file():
        raise FileNotFoundError(f"fused kernel source {relative} is absent under {REPO_ROOT}")
    return relative


def _padded_shape(tensor) -> tuple[int, ...]:
    return tuple(getattr(tensor, "padded_shape", tensor.shape))


def is_row_tile(tensor) -> bool:
    """Whether ``tensor`` is one row tile ``[1, 1, rows, W]`` with ``1 <= rows <= ROWS_MAX`` padded to ``TILE`` rows.

    The fused programs' input contract as a predicate, so a dispatcher can take the composed chain instead of
    raising."""

    shape, padded = tuple(tensor.shape), _padded_shape(tensor)
    return len(shape) == 4 and shape[:2] == (1, 1) and 1 <= shape[-2] <= ROWS_MAX and padded[-2] == TILE


def is_tile_width(tensor, width: int | None = None) -> bool:
    """Whether the last dimension is whole tiles, unpadded, and (when given) exactly ``width``."""

    actual, padded = tensor.shape[-1], _padded_shape(tensor)[-1]
    return actual % TILE == 0 and padded == actual and (width is None or actual == width)


def rows_of(tensor) -> int:
    """The valid rows of one row tile ``[1, 1, rows, W]`` (padded to 32 rows); rejects any other shape."""

    if not is_row_tile(tensor):
        raise ValueError(
            f"expected one row tile [1, 1, 1..{ROWS_MAX}, W] padded to {TILE} rows, got {tuple(tensor.shape)} padded "
            f"{_padded_shape(tensor)}"
        )
    return tensor.shape[-2]


def tile_width_of(tensor) -> int:
    """The width of a tensor whose last dimension is whole tiles."""

    if not is_tile_width(tensor):
        raise ValueError(f"expected a width of whole tiles, got {tensor.shape[-1]} padded {_padded_shape(tensor)[-1]}")
    return tensor.shape[-1]


@dataclass(frozen=True)
class CoreWork:
    core: ttnn.CoreCoord
    start: int
    count: int


def split_work(units: int, mesh, cores: int | None = None) -> list[CoreWork]:
    """``units`` work items over the compute grid in linear core order, as evenly as possible; cores with work only.

    ``cores`` caps the core count below one unit per core."""

    grid = mesh.compute_with_storage_grid_size()
    if units <= 0:
        raise ValueError(f"nothing to split: {units} units")
    most = min(units, grid.x * grid.y)
    cores = most if cores is None else cores
    if not 1 <= cores <= most:
        raise ValueError(f"cannot split {units} units over {cores} cores (grid {grid.x}x{grid.y})")
    base, extra = divmod(units, cores)
    work, start = [], 0
    for i in range(cores):
        count = base + (1 if i < extra else 0)
        work.append(CoreWork(ttnn.CoreCoord(i // grid.y, i % grid.y), start, count))
        start += count
    return work


def core_set(work: Iterable[CoreWork]) -> ttnn.CoreRangeSet:
    return ttnn.CoreRangeSet([ttnn.CoreRange(w.core, w.core) for w in work])


def core_rectangle(work: Sequence[CoreWork], mesh=None) -> ttnn.CoreRangeSet:
    """The cores of ``work`` as few CoreRanges: one range when they fill a rectangle; with ``mesh``, the whole columns
    plus the partial one; otherwise the per-core set.

    Dispatch multicasts the binaries once per range, so fewer ranges launch faster."""

    cores = {(w.core.x, w.core.y) for w in work}
    xs, ys = sorted({x for x, _ in cores}), sorted({y for _, y in cores})
    if cores and len(cores) == len(xs) * len(ys) and all((x, y) in cores for x in xs for y in ys):
        return ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(xs[0], ys[0]), ttnn.CoreCoord(xs[-1], ys[-1]))])
    if mesh is not None and work:
        grid = mesh.compute_with_storage_grid_size()
        if [(w.core.x, w.core.y) for w in work] == [(i // grid.y, i % grid.y) for i in range(len(work))]:
            columns, rest = divmod(len(work), grid.y)
            ranges = []
            if columns:
                ranges.append(ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(columns - 1, grid.y - 1)))
            if rest:
                ranges.append(ttnn.CoreRange(ttnn.CoreCoord(columns, 0), ttnn.CoreCoord(columns, rest - 1)))
            return ttnn.CoreRangeSet(ranges)
    return core_set(work)


def accessor_args(tensor) -> list[int]:
    return list(ttnn.TensorAccessorArgs(tensor).get_compile_time_args())


def cb_descriptor(index: int, dtype, page_bytes: int, pages: int, cores: ttnn.CoreRangeSet) -> ttnn.CBDescriptor:
    return ttnn.CBDescriptor(
        total_size=pages * page_bytes,
        core_ranges=cores,
        format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=index, data_format=dtype, page_size=page_bytes)],
    )


def _named(named_compile_time_args) -> list[tuple[str, int]]:
    items = named_compile_time_args.items() if isinstance(named_compile_time_args, dict) else named_compile_time_args
    return [(str(name), int(value)) for name, value in items]


def _kernel(source, cores, compile_time_args, runtime_args, defines, config, named=()) -> ttnn.KernelDescriptor:
    """``named`` = named compile-time args (``get_named_compile_time_arg_val`` in the kernel), a dict or pairs."""

    return ttnn.KernelDescriptor(
        kernel_source=source,
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=cores,
        compile_time_args=[int(a) for a in compile_time_args],
        named_compile_time_args=_named(named),
        defines=list(defines),
        runtime_args=[(core, [int(a) for a in args]) for core, args in runtime_args],
        config=config,
    )


def reader_kernel(source, cores, compile_time_args, runtime_args, defines=(), named=()) -> ttnn.KernelDescriptor:
    return _kernel(source, cores, compile_time_args, runtime_args, defines, ttnn.ReaderConfigDescriptor(), named)


def writer_kernel(source, cores, compile_time_args, runtime_args, defines=(), named=()) -> ttnn.KernelDescriptor:
    return _kernel(source, cores, compile_time_args, runtime_args, defines, ttnn.WriterConfigDescriptor(), named)


def compute_kernel(
    source,
    cores,
    compile_time_args,
    runtime_args=(),
    defines=(),
    named=(),
    *,
    fidelity=ttnn.MathFidelity.HiFi4,
    fp32_dest: bool = False,
    approx: bool = False,
    dst_full_sync: bool = False,
    bfp8_pack_precise: bool = False,
    unpack_to_dest_fp32: Sequence[int] = (),
) -> ttnn.KernelDescriptor:
    """``unpack_to_dest_fp32`` = the CB indices unpacked to DST as exact fp32; the other CBs keep the default."""

    config = ttnn.ComputeConfigDescriptor(
        math_fidelity=fidelity,
        math_approx_mode=approx,
        fp32_dest_acc_en=fp32_dest,
        dst_full_sync_en=dst_full_sync,
        bfp8_pack_precise=bfp8_pack_precise,
    )
    if unpack_to_dest_fp32:
        modes = [ttnn.UnpackToDestMode.Default] * CB_COUNT
        for cb in unpack_to_dest_fp32:
            modes[int(cb)] = ttnn.UnpackToDestMode.UnpackToDestFp32
        vector = getattr(ttnn, "VectorUnpackToDestMode", None) or getattr(
            ttnn._ttnn.program_descriptor, "VectorUnpackToDestMode", None
        )
        config.unpack_to_dest_mode = vector(modes) if vector else modes
    return _kernel(source, cores, compile_time_args, runtime_args, defines, config, named)


def semaphore_descriptor(id: int, cores: ttnn.CoreRangeSet, initial: int = 0) -> ttnn.SemaphoreDescriptor:
    return ttnn.SemaphoreDescriptor(id=id, core_ranges=cores, initial_value=initial)


def program_descriptor(kernels: Sequence, cbs: Sequence = (), semaphores: Sequence = ()) -> ttnn.ProgramDescriptor:
    return ttnn.ProgramDescriptor(kernels=list(kernels), semaphores=list(semaphores), cbs=list(cbs))


def allocate(shape: Sequence[int], dtype, layout, mesh, memory_config=ttnn.DRAM_MEMORY_CONFIG):
    return ttnn.allocate_tensor_on_device(ttnn.Shape(list(shape)), dtype, layout, mesh, memory_config)


def stamp_topology(tensor, reference, shard_dim: int | None = None):
    """Give a program's output the topology the chain's op would have produced; returns ``tensor``.

    ``reference``'s mesh and coordinates, replicated or sharded on ``shard_dim``; downstream collectives read it."""

    topology = reference.tensor_topology()
    placements = [
        ttnn.PlacementReplicate(),
        ttnn.PlacementReplicate() if shard_dim is None else ttnn.PlacementShard(shard_dim),
    ]
    tensor.update_tensor_topology(
        ttnn.TensorTopology(topology.distribution_shape(), placements, topology.mesh_coords())
    )
    return tensor


# ---------------------------------------------------------------- program metadata


@dataclass(frozen=True)
class FusedProgramMeta:
    """The written buffers of one fused program as its builder declares them.

    ``kernel`` is the registry name, ``variant`` the program form."""

    kernel: str
    variant: str
    # ``(tensor, placement)`` pairs for the written buffers: ``None`` replicated, an int the sharded dim, a tensor "as
    # that tensor's". ``run_program`` stamps them; generic_op leaves the allocation's default placement.
    outputs: tuple


def program_meta(kernel: str, variant: str, *, outputs: Iterable) -> FusedProgramMeta:
    declared = []
    for entry in outputs:
        tensor, placement = entry
        if not (placement is None or isinstance(placement, int) or hasattr(placement, "tensor_topology")):
            raise ValueError(
                f"program meta of {kernel}/{variant}: an output placement is None, a shard dim or a tensor"
            )
        declared.append((tensor, placement))
    return FusedProgramMeta(str(kernel), str(variant), tuple(declared))


def restamp_outputs(outputs, io_tensors: Sequence, result=None) -> None:
    """Give the launch's written buffers their declared placements on the first io tensor's mesh.

    ``result`` (the launch's returned handle) is stamped with the output it shares a buffer with."""

    if not outputs:
        return
    reference = next(t for t in io_tensors if hasattr(t, "tensor_topology"))
    for tensor, placement in outputs:
        targets = [tensor]
        if result is not None and result is not tensor and hasattr(result, "buffer_address"):
            if result.buffer_address() == tensor.buffer_address():
                targets.append(result)
        for target in targets:
            if placement is None or isinstance(placement, int):
                stamp_topology(target, reference, placement)
            else:
                target.update_tensor_topology(placement.tensor_topology())


def run_program(io_tensors: Sequence, descriptor, *, meta: FusedProgramMeta | None = None):
    """Inputs first, pre-allocated outputs last; returns the last tensor.

    ``descriptor`` is a ProgramDescriptor or a MeshProgramDescriptor; ``meta``'s ``outputs`` are stamped with their
    declared placements after the launch."""

    result = ttnn.generic_op(list(io_tensors), descriptor)
    if meta is not None:
        restamp_outputs(meta.outputs, io_tensors, result)
    return result
