# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Energy per call of each of the Galaxy KDA layer's programs, and the power of a waiting core (measurement tools).

Sustained prefill is power-limited: every chip runs into its 130 W limit and the firmware lowers the AI clock, so the
energy a call takes sets the sustained throughput. Telemetry (sysfs AI clock, hwmon power) updates only every
25-100 ms, so one program's energy needs a sustained loop of that program alone.

test_layer_energy: for each KDA_ENERGY_POINTS program (default: the layer's 9 programs, then the whole layer), runs
the layer once on its real inputs up to that program, traces KDA_ENERGY_COPIES[point] back-to-back copies of the
program on the inputs the layer produced (kept alive), and replays that trace for KDA_ENERGY_SECONDS. Before each
point it records each chip's idle power. Energy per call per chip = (busy mean power - idle power) x window / calls,
the dynamic energy above idle. KDA_ENERGY_DUTY < 1 idles between blocks of replays so the average power stays under
the limit and the clock stays at its nominal 1350 MHz (a fixed-clock energy); 1 replays back to back. Results go to
KDA_ENERGY= lines and to KDA_ENERGY_OUT as JSON.

test_wait_power: what a core costs while it waits, against an idle core (firmware polling its go message) and two
active references: every worker core of every chip runs kernels/wait_power_dm.cpp on both data movement RISCs and
optionally kernels/wait_power_compute.cpp on the TRISCs, for KDA_WAIT_SECONDS per variant. KDA_WAIT_VARIANTS selects
variants by name. The matmul reference reuses kernels/pack_sfpu_overlap.cpp (1x4 HiFi2 FP32-DEST subblocks).
"""

from __future__ import annotations

import dataclasses
import gc
import json
import os
import statistics
import time
from collections.abc import Callable
from pathlib import Path

import pytest
import torch

import ttnn
from models.demos.deepseek_v3_d_p.tests.fabric_profiles import torus_xy_device_params
from models.demos.deepseek_v3_d_p.tests.kda.perf.telemetry import Telemetry as _Telemetry
from models.demos.deepseek_v3_d_p.tests.kda.perf.test_layer_perf import (
    _GALAXY_ACTUAL_START,
    _L1_SMALL_SIZE,
    _capture_layer_trace,
    _release_layer_trace,
)
from models.demos.deepseek_v3_d_p.tests.kda.utils import (
    deallocate_state,
    make_kimi_k3_device_case,
    make_synthetic_kimi_k3_test_case,
)
from models.demos.deepseek_v3_d_p.tt.kda import kda as kda_module
from models.demos.deepseek_v3_d_p.tt.kda.config import KDA_WORKER_L1_SIZE
from models.demos.deepseek_v3_d_p.tt.kda.kda import ttKDA
from models.demos.deepseek_v3_d_p.utils.chunk_config import PREFILL_CHUNK_TOKENS
from tests.ttnn.unit_tests.operations.experimental.kda.kda_test_utils import make_actual_start

# Each point names one program of the layer and where its dispatch is intercepted: (owner, attribute).
_POINTS: dict[str, Callable[[ttKDA], tuple[object, str]]] = {
    "input_projection": lambda layer: (layer, "_project_inputs"),
    "conv": lambda layer: (ttnn.experimental.kda, "qkv_causal_conv1d_silu_with_carry"),
    "decay": lambda layer: (layer, "_compute_decay"),
    "prep": lambda layer: (ttnn.experimental.kda, "prepare_chunk_recurrence"),
    "summary": lambda layer: (ttnn.experimental.kda, "summarize_chunk_recurrence"),
    "relay": lambda layer: (ttnn.experimental.kda, "relay_affine_prefix"),
    "correction": lambda layer: (ttnn.experimental.kda, "entry_correction"),
    "norm": lambda layer: (layer, "_kda_rms_norm"),
    "output": lambda layer: (kda_module, "matmul_reduce_scatter"),
}
# Copies per trace: about 1 ms of device time each, so a replay's own overhead is amortized.
_DEFAULT_COPIES = {
    "input_projection": 3,
    "conv": 16,
    "decay": 40,
    "prep": 14,
    "summary": 14,
    "relay": 11,
    "correction": 48,
    "norm": 28,
    "output": 6,
    "layer": 1,
}
_POINTS_ENV = os.environ.get("KDA_ENERGY_POINTS")
_SELECTED = _POINTS_ENV.split(",") if _POINTS_ENV else [*_POINTS, "layer"]
_COPIES = {**_DEFAULT_COPIES, **json.loads(os.environ.get("KDA_ENERGY_COPIES", "{}"))}
_SECONDS = float(os.environ.get("KDA_ENERGY_SECONDS", "4"))
_DUTY = float(os.environ.get("KDA_ENERGY_DUTY", "1"))
_BLOCK = int(os.environ.get("KDA_ENERGY_BLOCK", "10"))
_IDLE_S = float(os.environ.get("KDA_ENERGY_IDLE_S", "1.5"))
_OUT = Path(os.environ.get("KDA_ENERGY_OUT", "generated/kda_energy.json"))
_TRACE_REGION = 64 * 1024 * 1024


class _Captured(Exception):
    """Ends the layer's forward right after the traced program."""


def _tensors(value) -> list[ttnn.Tensor]:
    if isinstance(value, ttnn.Tensor):
        return [value]
    if isinstance(value, (list, tuple)):
        return [t for item in value for t in _tensors(item)]
    if isinstance(value, dict):
        return [t for item in value.values() for t in _tensors(item)]
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return [t for field in dataclasses.fields(value) for t in _tensors(getattr(value, field.name))]
    return []


def _address(tensor: ttnn.Tensor) -> int | None:
    try:
        return tensor.buffer_address() if tensor.is_allocated() else None
    except Exception:
        return None


@dataclasses.dataclass
class _PointTrace:
    trace_id: int
    copies: int
    keep: list  # the program's inputs and last outputs, alive until released


def _capture_point_trace(mesh_device, layer, point, copies, forward) -> _PointTrace:
    """Run ``forward`` (the layer's call) up to ``point``'s program and trace ``copies`` of that program there."""
    owner, name = _POINTS[point](layer)
    original = getattr(owner, name)
    owned = owner is layer and name in vars(owner)
    captured: dict = {}

    def capture(*args, **kwargs):
        inputs = {_address(t) for t in _tensors((args, kwargs))}
        ttnn.synchronize_device(mesh_device)
        trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
        try:
            for copy in range(copies):
                result = original(*args, **kwargs)
                if copy + 1 < copies:
                    # Free this copy's outputs (never an input it returns) so the next copy reuses their memory.
                    for tensor in _tensors(result):
                        if _address(tensor) not in inputs:
                            ttnn.deallocate(tensor)
        finally:
            ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
        captured.update(trace_id=trace_id, keep=[args, kwargs, result])
        raise _Captured

    setattr(owner, name, capture)
    try:
        forward()
        raise AssertionError(f"the layer never dispatched {point}")
    except _Captured:
        pass
    finally:
        if owner is layer and not owned:
            delattr(owner, name)
        else:
            setattr(owner, name, original)
    gc.collect()  # the aborted forward's intermediates
    return _PointTrace(captured["trace_id"], copies, captured["keep"])


def _chip_power(telemetry: _Telemetry, start: float, end: float) -> list[float]:
    rows = [row for row in telemetry.samples if start <= row[0] <= end]
    chips = len(telemetry.hwmons)
    return [statistics.fmean(row[2][chip] for row in rows if row[2][chip] is not None) for chip in range(chips)]


def _chip_clock(telemetry: _Telemetry, start: float, end: float) -> list[float]:
    rows = [row for row in telemetry.samples if start <= row[0] <= end]
    return [
        statistics.fmean(row[1][chip] for row in rows if row[1][chip] is not None)
        for chip in range(len(telemetry.clocks))
    ]


def _measure(
    mesh_device, telemetry: _Telemetry, step: Callable[[], None], calls_per_step: int, seconds: float, duty: float
) -> dict:
    """Idle, then run ``step`` for ``seconds`` at ``duty``; energy and clock over the run's second half."""
    ttnn.synchronize_device(mesh_device)
    idle_start = time.perf_counter()
    time.sleep(_IDLE_S)
    idle_end = time.perf_counter()
    blocks = []
    run_start = time.perf_counter()
    while time.perf_counter() - run_start < seconds:
        start = time.perf_counter()
        step()
        ttnn.synchronize_device(mesh_device)
        end = time.perf_counter()
        blocks.append((start, end))
        if duty < 1:
            time.sleep((end - start) * (1 - duty) / duty)
    run_end = time.perf_counter()
    half = run_start + seconds / 2
    window = [(s, e) for s, e in blocks if s >= half] or blocks[len(blocks) // 2 :]
    calls = len(window) * calls_per_step
    span = run_end - window[0][0]
    idle = _chip_power(telemetry, idle_start + 0.4 * _IDLE_S, idle_end)
    busy = _chip_power(telemetry, window[0][0], run_end)
    clocks = _chip_clock(telemetry, window[0][0], run_end)
    energy = [(b - i) * span / calls * 1e6 for b, i in zip(busy, idle)]  # uJ per call per chip
    ms_per_call = statistics.median((e - s) * 1e3 / calls_per_step for s, e in window)
    return {
        "calls": calls,
        "busy_ms_per_call": ms_per_call,
        "duty_measured": sum(e - s for s, e in window) / span,
        "aiclk_mean_mhz": statistics.fmean(clocks),
        "aiclk_min_chip_mhz": min(clocks),
        "idle_w_per_chip": statistics.fmean(idle),
        "busy_w_per_chip": statistics.fmean(busy),
        "busy_w_chip_max": max(busy),
        "energy_uj_per_chip_mean": statistics.fmean(energy),
        "energy_uj_per_chip_max": max(energy),
        "energy_uj_per_chip": [round(value, 2) for value in energy],
    }


def _device_params() -> dict:
    return torus_xy_device_params(
        l1_small_size=_L1_SMALL_SIZE, worker_l1_size=KDA_WORKER_L1_SIZE, trace_region_size=_TRACE_REGION
    )


_MESH = pytest.param(
    (8, 4),
    _device_params(),
    marks=pytest.mark.requires_mesh_topology(mesh_shape=(8, 4), topology="mesh-8x4"),
    id="SP8xTP4-torus-xy",
)


@pytest.mark.parametrize("mesh_device,device_params", [_MESH], indirect=True)
def test_layer_energy(mesh_device: ttnn.MeshDevice, device_params: dict) -> None:
    """Energy per call of each selected program of the layer, and of the layer."""
    telemetry = _Telemetry()
    if not telemetry.clocks or not telemetry.hwmons:
        pytest.skip("no Blackhole sysfs telemetry on this host")
    case = make_synthetic_kimi_k3_test_case(sequence=PREFILL_CHUNK_TOKENS)
    layer, hidden = make_kimi_k3_device_case(mesh_device, case, tensor_parallel_axis=1, cache_weights=False)
    actual_start, actual_end = _GALAXY_ACTUAL_START, _GALAXY_ACTUAL_START + PREFILL_CHUNK_TOKENS
    start_tt = make_actual_start(mesh_device, actual_start)
    end_tt = make_actual_start(mesh_device, actual_end)
    state = layer.allocate_state(batch_size=1)
    # Warm-up: compiles every program.
    output, next_state = layer.forward(hidden, state, start_tt, end_tt)
    ttnn.synchronize_device(mesh_device)
    ttnn.deallocate(output)
    deallocate_state(next_state)

    results = {}
    telemetry.start()
    try:
        for point in _SELECTED:
            if point == "layer":
                trace = _capture_layer_trace(mesh_device, layer, hidden, actual_start, actual_end)
                trace_id, copies = trace.trace_id, 1
            else:
                trace = _capture_point_trace(
                    mesh_device, layer, point, _COPIES[point], lambda: layer.forward(hidden, state, start_tt, end_tt)
                )
                trace_id, copies = trace.trace_id, trace.copies
            try:
                for _ in range(3):
                    ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
                ttnn.synchronize_device(mesh_device)

                def step():
                    for _ in range(_BLOCK):
                        ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)

                result = _measure(mesh_device, telemetry, step, _BLOCK * copies, _SECONDS, _DUTY)
            finally:
                if point == "layer":
                    _release_layer_trace(mesh_device, trace)
                else:
                    ttnn.release_trace(mesh_device, trace_id)
                    for tensor in _tensors(trace.keep[2]):
                        if tensor.is_allocated():
                            ttnn.deallocate(tensor)
                    trace.keep.clear()
                    gc.collect()
            result.update(point=point, copies=copies, duty=_DUTY)
            results[point] = result
            print("KDA_ENERGY=" + json.dumps({k: v for k, v in result.items() if k != "energy_uj_per_chip"}))
    finally:
        telemetry.stop()
    ops = [p for p in results if p != "layer"]
    if ops and "layer" in results:
        summed = sum(results[p]["energy_uj_per_chip_mean"] for p in ops)
        print(f"KDA_ENERGY_SUM ops={summed:.1f} uJ/chip layer={results['layer']['energy_uj_per_chip_mean']:.1f}")
    _OUT.parent.mkdir(parents=True, exist_ok=True)
    _OUT.write_text(json.dumps(results))


_DM_KERNEL = "models/demos/deepseek_v3_d_p/tests/kda/perf/kernels/wait_power_dm.cpp"
_COMPUTE_KERNEL = "models/demos/deepseek_v3_d_p/tests/kda/perf/kernels/wait_power_compute.cpp"
_MATMUL_KERNEL = "models/demos/deepseek_v3_d_p/tests/kda/perf/kernels/pack_sfpu_overlap.cpp"
_WAIT_SECONDS = float(os.environ.get("KDA_WAIT_SECONDS", "4"))
_WAIT_PROGRAM_CYCLES = 1350 * 4000  # 4 ms per program
# name: (data movement mode, pause cycles, compute mode); compute mode None runs no compute kernel.
_WAIT_VARIANTS = {
    "dm_clock_spin": (1, 0, None),
    "dm_l1_poll": (2, 0, None),
    "all5_poll": (2, 0, 2),
    "all5_clock_spin": (1, 0, 1),
    "dm_poll_pause64": (3, 64, None),
    "dm_poll_pause1024": (3, 1024, None),
    "dm_noc_read": (4, 0, None),
    "matmul_hifi2": None,  # k_block 8
}
# Also accepted: matmul_kb<N> (the matmul reference at k_block N, the same tile-matmul count).
_WAIT_SELECTED = os.environ.get("KDA_WAIT_VARIANTS", ",".join(_WAIT_VARIANTS)).split(",")
_MATMUL_K_BLOCK, _MATMUL_SUBBLOCKS = 8, 4096  # 4096 x 1112 cycles: about 3.4 ms per program
_WAIT_DUTY = float(os.environ.get("KDA_WAIT_DUTY", "1"))


def _wait_program(cores: ttnn.CoreRangeSet, dm_mode: int, pause: int, compute_mode: int | None):
    kernels = [
        ttnn.KernelDescriptor(
            kernel_source=_DM_KERNEL,
            core_ranges=cores,
            compile_time_args=[dm_mode, _WAIT_PROGRAM_CYCLES, pause, offset],
            runtime_args=[],
            config=config,
        )
        for config, offset in ((ttnn.ReaderConfigDescriptor(), 0), (ttnn.WriterConfigDescriptor(), 8192))
    ]
    if compute_mode is not None:
        kernels.append(
            ttnn.KernelDescriptor(
                kernel_source=_COMPUTE_KERNEL,
                core_ranges=cores,
                compile_time_args=[compute_mode, _WAIT_PROGRAM_CYCLES],
                runtime_args=[],
                config=ttnn.ComputeConfigDescriptor(),
            )
        )
    return ttnn.ProgramDescriptor(
        kernels=kernels,
        semaphores=[ttnn.SemaphoreDescriptor(id=0, core_ranges=cores, initial_value=0)],
        cbs=[
            ttnn.CBDescriptor(
                total_size=16384,
                core_ranges=cores,
                format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=0, data_format=ttnn.bfloat16, page_size=2048)],
            )
        ],
    )


def _matmul_program(cores: ttnn.CoreRangeSet, kt: int):
    sub_w = 4

    def cb(index, fmt, page, pages):
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
                compile_time_args=[kt, _MATMUL_SUBBLOCKS * _MATMUL_K_BLOCK // kt, 0, 1],
                runtime_args=[],
                config=ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi2, fp32_dest_acc_en=True),
            )
        ],
        semaphores=[],
        cbs=[
            cb(0, ttnn.bfloat16, 2048, kt),
            cb(1, ttnn.bfloat8_b, 1088, kt * sub_w),
            cb(16, ttnn.bfloat16, 2048, sub_w),
        ],
    )


@pytest.mark.parametrize("mesh_device,device_params", [_MESH], indirect=True)
def test_wait_power(mesh_device: ttnn.MeshDevice, device_params: dict) -> None:
    """Power above idle of every worker core waiting in each KDA_WAIT_VARIANTS way."""
    telemetry = _Telemetry()
    if not telemetry.clocks or not telemetry.hwmons:
        pytest.skip("no Blackhole sysfs telemetry on this host")
    grid = mesh_device.compute_with_storage_grid_size()
    cores = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))])
    io = [
        ttnn.from_torch(
            torch.zeros(1, 1, 32, 32),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )
        for _ in range(2)
    ]
    results = {}
    telemetry.start()
    try:
        for name in _WAIT_SELECTED:
            k_block = int(name.removeprefix("matmul_kb")) if name.startswith("matmul_kb") else _MATMUL_K_BLOCK
            variant = None if name.startswith("matmul_kb") else _WAIT_VARIANTS[name]
            program = _matmul_program(cores, k_block) if variant is None else _wait_program(cores, *variant)
            ttnn.generic_op(io, program)  # compile
            ttnn.synchronize_device(mesh_device)

            def step():
                for _ in range(4):
                    ttnn.generic_op(io, program)

            result = _measure(mesh_device, telemetry, step, 4, _WAIT_SECONDS, _WAIT_DUTY)
            result.update(variant=name, cores_per_chip=grid.x * grid.y)
            if variant is None:
                tile_ops = _MATMUL_SUBBLOCKS * _MATMUL_K_BLOCK * 4 * grid.x * grid.y
                result["tile_matmuls_per_call_per_chip"] = tile_ops
                result["nj_per_tile_matmul"] = result["energy_uj_per_chip_mean"] * 1e3 / tile_ops
            results[name] = result
            print("KDA_WAIT=" + json.dumps({k: v for k, v in result.items() if k != "energy_uj_per_chip"}))
    finally:
        telemetry.stop()
        for tensor in io:
            ttnn.deallocate(tensor)
    out = _OUT.with_name(_OUT.stem + "_wait.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results))
