# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Delay injection after any of the KDA layer's programs (a measurement tool).

``inject_delay(layer, point, cycles)`` makes ``point``'s program run longer on chosen chips without touching any of
the layer's kernels: right after the layer dispatches that program, it dispatches a one-core program that spins on
the chip's wall clock for a per-chip number of cycles (zero on the others). A chip runs its programs one after
another, so every later program on a delayed chip starts that much later, exactly as if ``point`` had run that much
longer there; data and results are unchanged. How much of the delay reaches the layer's traced wall is ``point``'s
slack on those chips: the full delay when the program is on the critical chain, nothing when another chip's waits
absorb it.

Outside the context manager nothing is dispatched, so the layer is unchanged when no delay is injected. Inside it,
the spin program also runs, with zero cycles, on the chips that are not delayed, so every chip dispatches the same
program sequence. A zero-cycle injection moves the traced wall by under 2 us (within noise); a zero-cycle injection at
the same point cancels even that.

Cycles are wall-clock cycles of the chip's AICLK. ``NOMINAL_CYCLES_PER_US`` converts microseconds the way the
real-time profiler reports them (nominal 1.35 GHz), so a delay in profiler-us maps 1:1 onto per-op profile times.
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


class WallClockSpin:
    """A one-core program per chip that spins for that chip's number of wall-clock cycles."""

    def __init__(self, mesh_device: ttnn.MeshDevice, cycles: Callable[[ttnn.MeshCoordinate], int]):
        rows, cols = tuple(mesh_device.shape)
        self.cycles = {(r, c): int(cycles(ttnn.MeshCoordinate(r, c))) for r in range(rows) for c in range(cols)}
        # generic_op reads no tensor here, but takes an input and its returned output; both stay allocated, so they
        # never move the layer's own allocations.
        self._io = [
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
        core = ttnn.CoreRangeSet([ttnn.CoreRange(_SPIN_CORE, _SPIN_CORE)])
        programs = {}
        for (r, c), chip_cycles in self.cycles.items():
            coord = ttnn.MeshCoordinate(r, c)
            programs[ttnn.MeshCoordinateRange(coord, coord)] = ttnn.ProgramDescriptor(
                kernels=[
                    ttnn.KernelDescriptor(
                        kernel_source=_SPIN_KERNEL,
                        source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
                        core_ranges=core,
                        compile_time_args=[chip_cycles],
                        config=ttnn.WriterConfigDescriptor(),
                    )
                ],
                semaphores=[],
                cbs=[],
            )
        self._program = ttnn.MeshProgramDescriptor(programs)
        self.dispatches = 0

    def __call__(self) -> None:
        ttnn.generic_op(self._io, self._program)
        self.dispatches += 1

    def release(self) -> None:
        for tensor in self._io:
            ttnn.deallocate(tensor)


# A chip selection: (SP rank, TP rank), the TP rank None for every chip of that SP rank.
Chip = tuple[int, int | None]


def chip_cycles(layer: ttKDA, delay_us: float, chips: Collection[Chip] | None) -> Callable[[ttnn.MeshCoordinate], int]:
    """Per-chip cycles: ``delay_us`` on the selected chips (every chip when None), zero elsewhere."""
    cycles = round(delay_us * NOMINAL_CYCLES_PER_US)
    sp_axis = layer.sequence_parallel_axis

    def selected(coord: ttnn.MeshCoordinate) -> bool:
        sp, tp = coord[sp_axis], coord[1 - sp_axis]
        return chips is None or (sp, None) in chips or (sp, tp) in chips

    return lambda coord: cycles if selected(coord) else 0


@contextlib.contextmanager
def inject_delay(layer: ttKDA, point: str, cycles: Callable[[ttnn.MeshCoordinate], int]) -> Iterator[WallClockSpin]:
    """Within the context, every program of the layer named ``point`` is followed by a ``WallClockSpin`` of
    ``cycles(coord)`` on each chip. The yielded spin's ``dispatches`` counts the injections."""
    if point not in _POINTS:
        raise ValueError(f"unknown delay point {point!r}; one of {POINTS}")
    owner, name = _POINTS[point](layer)
    spin = WallClockSpin(layer.device, cycles)
    original = getattr(owner, name)
    owned = owner is layer and name in vars(owner)

    def delayed(*args, **kwargs):
        result = original(*args, **kwargs)
        spin()
        return result

    setattr(owner, name, delayed)
    try:
        yield spin
    finally:
        if owner is layer and not owned:
            delattr(owner, name)
        else:
            setattr(owner, name, original)
        spin.release()


@dataclass(frozen=True)
class DelayConfig:
    """A delay of ``delay_us`` profiler-us after ``point`` on the chips ``chips`` (every chip when None), written
    ``point:delay_us:chips``: chips is "all", or "+"-joined SP ranks ("4") and single chips "sp.tp" ("4.2"), e.g.
    "conv:30:4", "relay:10:3+4" or "output:30:4.0+4.1"."""

    point: str
    delay_us: float
    chips: frozenset[Chip] | None

    @classmethod
    def parse(cls, text: str) -> DelayConfig:
        point, delay_us, chips = text.split(":")
        if point not in POINTS:
            raise ValueError(f"unknown delay point {point!r} in {text!r}; one of {POINTS}")

        def chip(item: str) -> Chip:
            sp, _, tp = item.partition(".")
            return int(sp), int(tp) if tp else None

        return cls(point, float(delay_us), None if chips == "all" else frozenset(chip(c) for c in chips.split("+")))

    def __str__(self) -> str:
        chips = (
            "all"
            if self.chips is None
            else "+".join(f"{sp}" if tp is None else f"{sp}.{tp}" for sp, tp in sorted(self.chips, key=str))
        )
        return f"{self.point}:{self.delay_us:g}:{chips}"

    def inject(self, layer: ttKDA) -> contextlib.AbstractContextManager[WallClockSpin]:
        return inject_delay(layer, self.point, chip_cycles(layer, self.delay_us, self.chips))
