# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Delay and work injection after any of the KDA layer's programs (a measurement tool).

``inject(layer, point, program)`` makes ``point``'s program run longer on chosen chips without touching any of the
layer's kernels: right after the layer dispatches that program, it dispatches one more program with a per-chip amount
of work (none on the chips not selected). A chip runs its programs one after another, so every later program on a
selected chip starts that much later, exactly as if ``point`` had run that much longer there; data and results are
unchanged. Two kinds of injected program:

- ``WallClockSpin`` ("spin"): one core spins on the chip's wall clock. It adds time and no energy (a waiting core
  costs no measurable power over an idle one), so the wall's response is the value of critical-path time. How much of
  the delay reaches the traced wall is ``point``'s slack on those chips: the full delay when the program is on the
  critical chain, nothing when another chip's waits absorb it.
- ``MatmulWork`` ("matmul"): every worker core runs 1x4 HiFi2 FP32-DEST tile-matmul subblocks (k_block 8) on whatever
  its circular buffers hold, for the same nominal time. It adds the same time plus a known FPU energy, so its response
  minus the spin's at equal time is the value of that energy.

Outside the context manager nothing is dispatched, so the layer is unchanged when nothing is injected. Inside it, the
injected program also runs, with zero work, on the chips that are not selected, so every chip dispatches the same
program sequence. A zero-cycle injection moves the traced wall by under 2 us (within noise).

Durations are given in profiler-us, cycles of the chip's AICLK at the nominal 1.35 GHz (``NOMINAL_CYCLES_PER_US``), so
an injection maps 1:1 onto per-op profile times; under a lowered clock it lasts proportionally longer in real time,
like the layer's own work.
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable, Collection, Iterator
from dataclasses import dataclass

import torch

import ttnn
from models.demos.deepseek_v3_d_p.tt.kda import kda as kda_module
from models.demos.deepseek_v3_d_p.tt.kda.kda import ttKDA

NOMINAL_CYCLES_PER_US = 1350

# A point names one program of the layer and where its dispatch is intercepted: (owner, attribute), the owner being
# the layer itself for its own methods. Each point's callable runs exactly one program per layer call.
_POINTS: dict[str, Callable[[ttKDA], tuple[object, str]]] = {
    "input_projection": lambda layer: (layer, "_project_inputs"),
    "conv": lambda layer: (ttnn.experimental.kda, "qkv_causal_conv1d_silu_with_carry"),
    "decay": lambda layer: (layer, "_compute_decay"),
    "prep": lambda layer: (ttnn.experimental.kda, "prepare_chunk_recurrence"),
    "summary": lambda layer: (ttnn.experimental.kda, "summarize_chunk_recurrence"),
    # The SP relay with the output gate (one program).
    "relay": lambda layer: (ttnn.experimental.kda, "relay_affine_prefix"),
    "correction": lambda layer: (ttnn.experimental.kda, "entry_correction"),
    "norm": lambda layer: (layer, "_kda_rms_norm"),
    # matmul_reduce_scatter with the final carry, the layer's last program.
    "output": lambda layer: (kda_module, "matmul_reduce_scatter"),
}
POINTS: tuple[str, ...] = tuple(_POINTS)

_SPIN_KERNEL = """
#include <cstdint>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t cycles = get_compile_time_arg_val(0);
    if constexpr (cycles != 0) {
        riscv_wait(cycles);
    }
}
"""
_SPIN_CORE = ttnn.CoreCoord(0, 0)
_SHAPE = [1, 1, 32, 32]

# The matmul work: pack_sfpu_overlap.cpp's subblock loop without SFPU work, 1 x 4 output tiles, k_block 8, HiFi2,
# FP32 DEST: 32 tile-matmuls per subblock, 1112 nominal cycles per subblock (compute-only microbenchmark).
_MATMUL_KERNEL = "models/demos/deepseek_v3_d_p/tests/kda/perf/kernels/pack_sfpu_overlap.cpp"
MATMUL_K_BLOCK = 8
MATMUL_SUB_W = 4
MATMUL_TILES_PER_SUBBLOCK = MATMUL_K_BLOCK * MATMUL_SUB_W
MATMUL_CYCLES_PER_SUBBLOCK = 1112


def injection_io(mesh_device: ttnn.MeshDevice) -> list[ttnn.Tensor]:
    """The input and output tensors generic_op needs for an injected program (which reads neither). Allocate them
    once, before the layer, and pass them to every injection so they never move the layer's own allocations."""
    return [
        ttnn.from_torch(
            torch.zeros(_SHAPE, dtype=torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )
        for _ in range(2)
    ]


def release_io(io: list[ttnn.Tensor]) -> None:
    for tensor in io:
        ttnn.deallocate(tensor)


class InjectedProgram:
    """One program per chip, built by ``program(amount)`` from that chip's amount of work (0 for no work)."""

    def __init__(
        self,
        mesh_device: ttnn.MeshDevice,
        amounts: Callable[[ttnn.MeshCoordinate], int],
        io: list[ttnn.Tensor] | None = None,
    ):
        rows, cols = tuple(mesh_device.shape)
        self.amounts = {(r, c): int(amounts(ttnn.MeshCoordinate(r, c))) for r in range(rows) for c in range(cols)}
        self._owned_io = io is None
        self._io = injection_io(mesh_device) if io is None else io
        self._program = ttnn.MeshProgramDescriptor(
            {
                ttnn.MeshCoordinateRange(ttnn.MeshCoordinate(r, c), ttnn.MeshCoordinate(r, c)): self.program(
                    mesh_device, amount
                )
                for (r, c), amount in self.amounts.items()
            }
        )
        self.dispatches = 0

    def program(self, mesh_device: ttnn.MeshDevice, amount: int) -> ttnn.ProgramDescriptor:
        raise NotImplementedError

    def __call__(self) -> None:
        ttnn.generic_op(self._io, self._program)
        self.dispatches += 1

    def release(self) -> None:
        if self._owned_io:
            release_io(self._io)


class WallClockSpin(InjectedProgram):
    """A one-core program per chip that spins for that chip's number of wall-clock cycles."""

    @staticmethod
    def amount(delay_us: float) -> int:
        return round(delay_us * NOMINAL_CYCLES_PER_US)

    def program(self, mesh_device: ttnn.MeshDevice, amount: int) -> ttnn.ProgramDescriptor:
        return ttnn.ProgramDescriptor(
            kernels=[
                ttnn.KernelDescriptor(
                    kernel_source=_SPIN_KERNEL,
                    source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
                    core_ranges=ttnn.CoreRangeSet([ttnn.CoreRange(_SPIN_CORE, _SPIN_CORE)]),
                    compile_time_args=[amount],
                    config=ttnn.WriterConfigDescriptor(),
                )
            ],
            semaphores=[],
            cbs=[],
        )


def worker_cores(mesh_device: ttnn.MeshDevice) -> ttnn.CoreRangeSet:
    grid = mesh_device.compute_with_storage_grid_size()
    return ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))])


def matmul_program(cores: ttnn.CoreRangeSet, subblocks: int) -> ttnn.ProgramDescriptor:
    """``subblocks`` 1x4 HiFi2 FP32-DEST subblocks at k_block 8 on every core of ``cores``, compute only: the operands
    are whatever the circular buffers hold (the previous program's data)."""

    def cb(index: int, fmt: ttnn.DataType, page: int, pages: int) -> ttnn.CBDescriptor:
        return ttnn.CBDescriptor(
            total_size=page * pages,
            core_ranges=cores,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=index, data_format=fmt, page_size=page)],
        )

    return ttnn.ProgramDescriptor(
        kernels=[
            ttnn.KernelDescriptor(
                kernel_source=_MATMUL_KERNEL,
                core_ranges=cores,
                compile_time_args=[MATMUL_K_BLOCK, subblocks, 0, 1],
                runtime_args=[],
                config=ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi2, fp32_dest_acc_en=True),
            )
        ],
        semaphores=[],
        cbs=[
            cb(0, ttnn.bfloat16, 2048, MATMUL_K_BLOCK),
            cb(1, ttnn.bfloat8_b, 1088, MATMUL_K_BLOCK * MATMUL_SUB_W),
            cb(16, ttnn.bfloat16, 2048, MATMUL_SUB_W),
        ],
    )


class MatmulWork(InjectedProgram):
    """Every worker core of each chip runs that chip's number of matmul subblocks (``matmul_program``)."""

    @staticmethod
    def amount(delay_us: float) -> int:
        return round(delay_us * NOMINAL_CYCLES_PER_US / MATMUL_CYCLES_PER_SUBBLOCK)

    def program(self, mesh_device: ttnn.MeshDevice, amount: int) -> ttnn.ProgramDescriptor:
        return matmul_program(worker_cores(mesh_device), amount)

    def tile_matmuls_per_chip(self, mesh_device: ttnn.MeshDevice) -> dict[tuple[int, int], int]:
        cores = worker_cores(mesh_device).num_cores()
        return {coord: amount * MATMUL_TILES_PER_SUBBLOCK * cores for coord, amount in self.amounts.items()}


KINDS: dict[str, type[InjectedProgram]] = {"spin": WallClockSpin, "matmul": MatmulWork}

# A chip selection: (SP rank, TP rank), the TP rank None for every chip of that SP rank.
Chip = tuple[int, int | None]


def chip_amounts(layer: ttKDA, amount: int, chips: Collection[Chip] | None) -> Callable[[ttnn.MeshCoordinate], int]:
    """Per-chip amounts: ``amount`` on the selected chips (every chip when None), zero elsewhere. Every selected chip
    must exist, so a selection never silently injects nothing."""
    sp_axis = layer.sequence_parallel_axis
    shape = tuple(layer.device.shape)
    sp_size, tp_size = shape[sp_axis], shape[1 - sp_axis]
    for sp, tp in chips or ():
        if not 0 <= sp < sp_size or (tp is not None and not 0 <= tp < tp_size):
            raise ValueError(f"chip {sp}.{tp} is outside the SP{sp_size}xTP{tp_size} mesh")

    def selected(coord: ttnn.MeshCoordinate) -> bool:
        sp, tp = coord[sp_axis], coord[1 - sp_axis]
        return chips is None or (sp, None) in chips or (sp, tp) in chips

    return lambda coord: amount if selected(coord) else 0


@contextlib.contextmanager
def inject(layer: ttKDA, point: str, make_program: Callable[[], InjectedProgram]) -> Iterator[InjectedProgram]:
    """Within the context, every program of the layer named ``point`` is followed by the program ``make_program()``
    builds. The yielded program's ``dispatches`` counts the injections."""
    if point not in _POINTS:
        raise ValueError(f"unknown injection point {point!r}; one of {POINTS}")
    owner, name = _POINTS[point](layer)
    injected = make_program()
    original = getattr(owner, name)
    owned = owner is layer and name in vars(owner)

    def delayed(*args, **kwargs):
        result = original(*args, **kwargs)
        injected()
        return result

    setattr(owner, name, delayed)
    try:
        yield injected
    finally:
        if owner is layer and not owned:
            delattr(owner, name)
        else:
            setattr(owner, name, original)
        injected.release()


def inject_delay(
    layer: ttKDA, point: str, cycles: Callable[[ttnn.MeshCoordinate], int], io: list[ttnn.Tensor] | None = None
) -> contextlib.AbstractContextManager[InjectedProgram]:
    """A ``WallClockSpin`` of ``cycles(coord)`` wall-clock cycles on each chip after ``point``."""
    return inject(layer, point, lambda: WallClockSpin(layer.device, cycles, io))


@dataclass(frozen=True)
class DelayConfig:
    """An injection of ``delay_us`` profiler-us after ``point`` on the chips ``chips`` (every chip when None), written
    ``point:delay_us:chips[:kind]``. chips is "all", or "+"-joined SP ranks ("4") and single chips "sp.tp" ("4.2").
    kind is "spin" (the default: time only) or "matmul" (the same time of FPU work on every worker core), e.g.
    "conv:30:4", "relay:10:3+4", "output:30:4.0+4.1" or "output:20:all:matmul"."""

    point: str
    delay_us: float
    chips: frozenset[Chip] | None
    kind: str = "spin"

    @classmethod
    def parse(cls, text: str) -> DelayConfig:
        fields = text.split(":")
        if len(fields) not in (3, 4):
            raise ValueError(f"injection {text!r} is not point:delay_us:chips[:kind]")
        point, delay_us, chips = fields[:3]
        kind = fields[3] if len(fields) == 4 else "spin"
        if point not in POINTS:
            raise ValueError(f"unknown injection point {point!r} in {text!r}; one of {POINTS}")
        if kind not in KINDS:
            raise ValueError(f"unknown injection kind {kind!r} in {text!r}; one of {tuple(KINDS)}")

        def chip(item: str) -> Chip:
            sp, _, tp = item.partition(".")
            return int(sp), int(tp) if tp else None

        selection = None if chips == "all" else frozenset(chip(c) for c in chips.split("+") if c)
        if selection is not None and not selection:
            raise ValueError(f"injection {text!r} selects no chip")
        return cls(point, float(delay_us), selection, kind)

    def __str__(self) -> str:
        chips = (
            "all"
            if self.chips is None
            else "+".join(f"{sp}" if tp is None else f"{sp}.{tp}" for sp, tp in sorted(self.chips, key=str))
        )
        suffix = "" if self.kind == "spin" else f":{self.kind}"
        return f"{self.point}:{self.delay_us:g}:{chips}{suffix}"

    def inject(
        self, layer: ttKDA, io: list[ttnn.Tensor] | None = None
    ) -> contextlib.AbstractContextManager[InjectedProgram]:
        """Inject into ``layer``; ``io`` (from ``injection_io``) avoids allocating per injection."""
        kind = KINDS[self.kind]
        amounts = chip_amounts(layer, kind.amount(self.delay_us), self.chips)
        return inject(layer, self.point, lambda: kind(layer.device, amounts, io))


def parse_configs(texts: Collection[str]) -> list[DelayConfig]:
    """Parse configurations, dropping repeats (keeping the first) so each one is measured and reported once."""
    configs: dict[str, DelayConfig] = {}
    for text in texts:
        config = DelayConfig.parse(text)
        configs.setdefault(str(config), config)
    return list(configs.values())
