# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Thin layer over ``ttnn.generic_op`` for this model's fused decode kernels.

A fused kernel is one program: kernel sources under ``ttnn/fused/<name>/kernels/*.cpp`` (paths relative to the
repository root, which the launchers make ``TT_METAL_HOME`` / ``TT_METAL_KERNEL_PATH``), circular buffers, semaphores
and per-core runtime args described from Python, run with ``run_program``.  No runtime rebuild: a change to a kernel
or a descriptor is picked up by the JIT.

Rows contract: every decode activation is one 32-row tile with ``rows`` valid rows (1 for decode, up to 32 for lanes
and MTP verify); ``rows_of`` reads and checks it.  A kernel takes the padded tile and produces only the valid rows.

Legibility: the device profiler and tt-perf-report see every fused program as ``GenericOp`` with ``runtime_id=N`` for
its attributes (generic_op's operation attributes are the descriptor, which the profiler cannot reflect), so each
builder states what its program is with ``run_program(io, descriptor, meta=program_meta(...))``: the kernel, the
program form, the rows, the DRAM and L1 bytes the descriptor addresses and the arithmetic it issues, by construction
from the builder's shapes.  With recording on (``record_program_meta``; the census switches it on around its
captures, the served process never pays for it) every launch is recorded in call order with the device operation id it
took -- the program's runtime id, the census's ``base`` -- and the census joins its per-program rows to the records.
"""

from __future__ import annotations

import math
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
# The data-movement descriptors' processors and NoCs (tt_metal/impl/kernels/kernel_types.cpp: Reader = RISCV_1 +
# preferred_noc_for_dram_read = NOC_0, Writer = RISCV_0 + preferred_noc_for_dram_write = NOC_1, on every arch per
# tt_metal/api/tt-metalium/kernel_types.hpp).  A NOC_1 multicast takes the bottom-right corner as its start.
READER_NOC = 0
WRITER_NOC = 1
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


def multicast_corners(x0: int, y0: int, x1: int, y1: int, *, noc: int) -> tuple[int, int, int, int]:
    """The (start_x, start_y, end_x, end_y) a multicast on ``noc`` needs for the NoC rectangle whose top-left is
    (x0, y0) and bottom-right (x1, y1) in NoC-0 coordinates: NOC_0 starts top-left, NOC_1 bottom-right (the
    DRAM-sharded matmul factory swaps the corners the same way).  A kernel that swaps on ``noc_index`` itself
    (gr_read/kernels/mcast_writer.cpp) must be given the top-left-first rectangle instead, never both."""

    if noc not in (READER_NOC, WRITER_NOC):
        raise ValueError(f"noc must be 0 or 1, got {noc}")
    if x0 > x1 or y0 > y1:
        raise ValueError(f"rectangle corners out of order: ({x0}, {y0}) .. ({x1}, {y1})")
    return (x0, y0, x1, y1) if noc == READER_NOC else (x1, y1, x0, y0)


def kernel_source(name: str, file: str) -> str:
    """``ttnn/fused/<name>/kernels/<file>`` relative to the repository root; the file must exist."""

    relative = f"{KERNEL_ROOT}/{name}/kernels/{file}"
    if not (REPO_ROOT / relative).is_file():
        raise FileNotFoundError(f"fused kernel source {relative} is absent under {REPO_ROOT}")
    return relative


def _padded_shape(tensor) -> tuple[int, ...]:
    return tuple(getattr(tensor, "padded_shape", tensor.shape))


def is_row_tile(tensor) -> bool:
    """Whether ``tensor`` is one row tile ``[1, 1, rows, W]`` with ``1 <= rows <= ROWS_MAX`` padded to ``TILE`` rows:
    the fused programs' input contract as a predicate, so a dispatcher can take the composed chain instead of raising
    (a host fake whose padded shape is its logical shape is outside it)."""

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
    """``units`` work items over the compute grid in linear core order, as evenly as possible; cores with work only
    (``cores`` caps the count below one unit per core)."""

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
    """The cores of ``work`` as few CoreRanges: one range when they fill a rectangle; with ``mesh``, the
    ``split_work`` order (grid columns filled top to bottom) as the whole columns plus the partial one (two ranges);
    otherwise the per-core set.  One range = one kernel group: the dispatcher multicasts the binaries once per range
    instead of once per core (80 single-core ranges cost 52 us per launch against 24 for one rectangle; 15-20 us of
    dispatch per program on 80 cores)."""

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


# The study build's per-phase device profiler zones: every fused kernel marks its phases with FUSED_ZONE(name)
# (``kernels/zones.h``), which is DeviceZoneScopedN under the define QWEN38_FUSED_ZONES and nothing otherwise.  The
# define goes on every kernel descriptor built here when the environment says QWEN38_FUSED_ZONES=1 (the census's zones
# arm); with it unset the descriptors are the served build's, byte for byte.
ZONES_ENV = "QWEN38_FUSED_ZONES"
ZONES_DEFINE = "QWEN38_FUSED_ZONES"


def zone_defines(environ=None) -> list[tuple[str, str]]:
    """``[(QWEN38_FUSED_ZONES, 1)]`` when the environment (default ``os.environ``) says ``QWEN38_FUSED_ZONES=1``, else ``[]``."""

    import os

    return [(ZONES_DEFINE, "1")] if (os.environ if environ is None else environ).get(ZONES_ENV) == "1" else []


def _kernel(source, cores, compile_time_args, runtime_args, defines, config, named=()) -> ttnn.KernelDescriptor:
    """``named`` = named compile-time args (``get_named_compile_time_arg_val`` in the kernel), a dict or pairs; the
    zone define is appended when the study build asks for it (``zone_defines``)."""

    return ttnn.KernelDescriptor(
        kernel_source=source,
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=cores,
        compile_time_args=[int(a) for a in compile_time_args],
        named_compile_time_args=_named(named),
        defines=[*defines, *zone_defines()],
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
    """``unpack_to_dest_fp32`` = the CB indices unpacked to DST as exact fp32 (fp32 intermediates; the other CBs keep
    the default), set on the config before it is copied into the descriptor."""

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
    """Give a program's output (``allocate`` leaves the allocation's topology) the distributed topology the chain's op
    would have produced: ``reference``'s mesh and coordinates, replicated (or sharded on ``shard_dim``); a collective
    or a mesh contract check downstream reads it.  Returns ``tensor``."""

    topology = reference.tensor_topology()
    placements = [
        ttnn.PlacementReplicate(),
        ttnn.PlacementReplicate() if shard_dim is None else ttnn.PlacementShard(shard_dim),
    ]
    tensor.update_tensor_topology(
        ttnn.TensorTopology(topology.distribution_shape(), placements, topology.mesh_coords())
    )
    return tensor


# ---------------------------------------------------------------- what a program is and what it moves (the census's legibility)


@dataclass(frozen=True)
class FusedProgramMeta:
    """One fused program as its builder states it, by construction from the shapes the descriptor was built from.

    ``kernel`` is the registry name the program serves (the module's ``NAME``; the QSA block's helpers that are not a
    registered kernel of their own say ``qsa_block``), ``variant`` the program form (one name per builder), ``rows``
    the valid rows served (1 = the decode step).  ``dram_bytes`` = the DRAM bytes the descriptor addresses, reads and
    writes (an operand every core streams counts once per core; a data-dependent partial read counts at its bound);
    ``l1_bytes`` = the bytes moved through L1 besides that: L1-resident operands read or written, multicasts and
    core-to-core hand-offs; ``flops`` = the arithmetic issued (a matmul 2MNK, a reduction or an elementwise pass one
    operation per element; 0 for data movement).  OUR model of the program, not the profiler's: the census divides
    these by the measured kernel time for achieved GB/s and TFLOPs."""

    kernel: str
    variant: str
    rows: int
    dram_bytes: int
    l1_bytes: int
    flops: int
    cores: int = 0  # the cores the program runs on (0 = not stated)
    # the buffers the launch writes with their declared mesh placement, ``(tensor, placement)`` pairs: ``None`` replicated,
    # an int the sharded dim (the second axis of the 1x4 mesh), a tensor "as that tensor's".  ``run_program`` stamps them
    # after the launch (``generic_op`` leaves the allocation's default placement: PlacementShard(0) on the line, which
    # a collective or a mesh-contract check downstream misreads; a 1x1 device test cannot see it -- 2026-09-26).
    outputs: tuple = ()


@dataclass(frozen=True)
class FusedProgramRecord:
    """One recorded launch: the device operation ids ``[first_id, end_id)`` the launch spanned (``first_id`` is the
    program's runtime id, the census's ``base``) and its meta."""

    first_id: int
    end_id: int
    meta: FusedProgramMeta

    def covers(self, base: int) -> bool:
        return self.first_id <= base < self.end_id


_META_RECORDS: list[FusedProgramRecord] = []
_META_RECORDING = False


def tensor_bytes(tensor) -> int:
    """The bytes of ``tensor``'s padded volume in its dtype (block floats by whole tiles)."""

    volume = math.prod(int(v) for v in _padded_shape(tensor))
    if tensor.dtype in ELEMENT_BYTES:
        return volume * ELEMENT_BYTES[tensor.dtype]
    if tensor.dtype in TILE_BYTES:
        return volume // (TILE * TILE) * TILE_BYTES[tensor.dtype]
    raise ValueError(f"no byte size for dtype {tensor.dtype}")


def in_l1(tensor) -> bool:
    """Whether ``tensor``'s buffer lives in L1 (a host fake without a memory config counts as DRAM)."""

    try:
        return tensor.memory_config().buffer_type == ttnn.BufferType.L1
    except (AttributeError, RuntimeError, TypeError):
        return False


def program_meta(
    kernel: str,
    variant: str,
    rows: int,
    *,
    reads: Iterable = (),
    writes: Iterable = (),
    partial: Iterable = (),
    flops: int = 0,
    dram_bytes: int = 0,
    l1_bytes: int = 0,
    cores: int = 0,
    outputs: Iterable = (),
) -> FusedProgramMeta:
    """The meta of one program: every tensor of ``reads`` and ``writes`` once and every ``(tensor, bytes)`` of
    ``partial`` (a window of a tensor) at those bytes, to ``dram_bytes`` or ``l1_bytes`` by where the tensor's buffer
    lives, plus the explicit bytes (per-core re-reads of an operand, multicasts) and the FLOPs the builder states;
    ``outputs`` the written buffers' declared placements (``FusedProgramMeta.outputs``), stamped after the launch."""

    dram, l1 = int(dram_bytes), int(l1_bytes)
    declared = []
    for entry in outputs:
        tensor, placement = entry
        if not (placement is None or isinstance(placement, int) or hasattr(placement, "tensor_topology")):
            raise ValueError(
                f"program meta of {kernel}/{variant}: an output placement is None, a shard dim or a tensor"
            )
        declared.append((tensor, placement))
    for tensor, count in [*((t, None) for t in (*reads, *writes)), *partial]:
        size = tensor_bytes(tensor) if count is None else int(count)
        if in_l1(tensor):
            l1 += size
        else:
            dram += size
    if dram < 0 or l1 < 0 or int(flops) < 0 or int(rows) < 1:
        raise ValueError(f"program meta of {kernel}/{variant}: bytes and flops must be >= 0 and rows >= 1")
    return FusedProgramMeta(str(kernel), str(variant), int(rows), dram, l1, int(flops), int(cores), tuple(declared))


def record_program_meta(enabled: bool = True) -> None:
    """Switch the recording of launches on (the census, around its captures) or off (the default: no host cost)."""

    global _META_RECORDING
    _META_RECORDING = bool(enabled)


def program_meta_recording() -> bool:
    return _META_RECORDING


def reset_program_meta() -> None:
    _META_RECORDS.clear()


def program_meta_records() -> tuple[FusedProgramRecord, ...]:
    """Every launch recorded since the last reset, in call order."""

    return tuple(_META_RECORDS)


def _device_operation_id() -> int:
    return int(ttnn._ttnn.get_device_operation_id())


def restamp_outputs(outputs, io_tensors: Sequence, result=None) -> None:
    """Give the launch's written buffers their declared placements (``FusedProgramMeta.outputs``): the mesh (its
    distribution shape and coordinates) is the first io tensor's; ``result`` (the launch's returned handle) is stamped
    with the output it shares a buffer with."""

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
    """Inputs first, pre-allocated outputs last; returns the last tensor.  ``descriptor`` is a ProgramDescriptor or a
    MeshProgramDescriptor; ``meta`` (``program_meta``) is recorded with the launch's device operation ids while
    recording is on, and its ``outputs`` are stamped with their declared placements after the launch."""

    if not _META_RECORDING or meta is None:
        result = ttnn.generic_op(list(io_tensors), descriptor)
    else:
        first = _device_operation_id()
        result = ttnn.generic_op(list(io_tensors), descriptor)
        _META_RECORDS.append(FusedProgramRecord(first, max(first + 1, _device_operation_id()), meta))
    if meta is not None:
        restamp_outputs(meta.outputs, io_tensors, result)
    return result


def traced_us_per_call(mesh, call, *, calls: int = 20, replays: int = 10, release=None) -> float:
    """Host wall per call of ``call()`` captured ``calls`` times in one trace and replayed ``replays`` times (the
    first replay warms and is not timed).  ``call`` runs once eagerly BEFORE the capture: a program that is not in
    the program cache cannot be loaded during trace capture (mesh_workload.cpp: "Warm up before capturing a trace").
    ``release(result)`` frees what ``call`` returns (default: deallocate a tensor or every tensor of a tuple)."""

    import time

    def free(result):
        for tensor in result if isinstance(result, (tuple, list)) else (result,):
            if hasattr(tensor, "injection"):  # a GR state
                ttnn.deallocate(tensor.injection)
            elif tensor is not None:
                ttnn.deallocate(tensor)

    release = free if release is None else release
    release(call())
    trace = ttnn.begin_trace_capture(mesh, cq_id=0)
    kept = [call() for _ in range(calls)]
    ttnn.end_trace_capture(mesh, trace, cq_id=0)
    ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
    started = time.perf_counter()
    for _ in range(replays):
        ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
    elapsed = time.perf_counter() - started
    ttnn.release_trace(mesh, trace)
    for result in kept:
        release(result)
    return round(elapsed / (replays * calls) * 1e6, 2)
