# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Device time of ttnn.linear program configs on the Galaxy mesh (a measurement tool, not a gate).

KDA_MM_CASES is a JSON list of cases: {"name", "M", "K", "N", "kind": "2d" | "1d" | "mmrs", "cols", "rows", "k_block",
"sub_h", "sub_w", "per_core_M"?, "per_core_N"?, "fidelity": "HiFi2", "wdtype": "bfloat8_b", "out": "L1" | "DRAM",
"in0": "DRAM" | "L1", "out_block_w"?, "out_block_h"?, "activation"?}. "activation" fuses a unary op into the matmul's
last K block: an op name ("sigmoid", the FP32-accurate RC sigmoid) or [name, params...] (e.g. ["sigmoid", 1, 0] for
the row-vector mode). out_block_w / out_block_h below per_core_N / per_core_M shrink the in1 / in0 circular buffers,
which lets larger k_blocks fit L1 (in1 holds 2 x k_block x out_block_w tiles). "fp32_acc" and "packer_l1_acc"
(default true) set the compute config. Each case runs replicated on all 32 chips; the slowest chip's realtime-profiler
duration is reported (median of KDA_MM_RUNS runs).

"kind": "mmrs" runs the output projection fused with its TP reduce-scatter (ttnn.operations.matmul_reduce_scatter with the
layer's settings: scatter -1, Ring, balanced lines, ephemeral staging; without the final-carry program the layer merges
onto the idle cores, and on a near-empty L1, so its plan and power can differ from the layer's) on [M, K] @ [K, N], K split over the TP axis.
Its "mrs" dict overrides the op's planning knobs (module constants of its program descriptor, e.g. {"KB_MAX": 32,
"SUPER_BLOCK": 3}) for that case only; the plan it took is reported as "plan".

"compare": the name of an earlier case. Reports whether this case's output (chip 0) is bit-identical to that case's,
and the largest absolute difference.

Every case also reports the per-chip spread of its eager runs: "chips_us" = [min, median, max] over the chips of each
chip's median duration ("us" is the slowest chip), and "chips_mesh_us" the same per chip as mesh rows. An eager mesh
dispatch reaches the chips one after another, so in a cross-chip op (mmrs) a chip launched early waits for its late
peers: its duration holds that launch skew. KDA_MM_TRACE_CALLS = n > 0 also profiles the traced op replayed n times back
to back (3 replays after a warm-up one): "trace_first_chips_us" (each replay's first call), "trace_steady_chips_us"
(calls 2..n), "trace_steady_mesh_us" (per chip) and "trace_period_us" (median over chips of the start-to-start period
of calls 2..n, which adds the trace's per-replay launch gap): the op's cost once the chips run in step.

KDA_MM_SUSTAIN_S > 0 also replays each case's traced op back to back for that many seconds (after
KDA_MM_SUSTAIN_IDLE_S of idle, in blocks of KDA_MM_SUSTAIN_BLOCK calls with a host sync after each block; 1 syncs
after every call) and reports over the second half the wall per call ("sustain_ms_per_call"), the chips'
AI clock and total power (telemetry.py), and the energy per call of all chips ("sustain_mj_per_call"). Back-to-back
load runs into each chip's power limit, so the sustained wall ranks configs by energy per call as much as by time.
Results go to stdout as KDA_MM= lines and to KDA_MM_OUT as JSON.
"""

from __future__ import annotations

import contextlib
import json
import os
import statistics
import sys
import time
from pathlib import Path

import pytest
import torch

import ttnn
from models.demos.deepseek_v3_d_p.tests.fabric_profiles import torus_xy_device_params
from models.demos.deepseek_v3_d_p.tests.kda.perf.telemetry import Telemetry
from models.demos.deepseek_v3_d_p.tests.kda.perf.test_layer_perf import _L1_SMALL_SIZE
from models.demos.deepseek_v3_d_p.tt.kda.config import KDA_WORKER_L1_SIZE
from tests.ttnn.profiling.realtime_profiler_utils import require_realtime_profiler

_CASES = json.loads(os.environ.get("KDA_MM_CASES", "[]"))
_RUNS = int(os.environ.get("KDA_MM_RUNS", "5"))
_OUT = Path(os.environ.get("KDA_MM_OUT", "generated/kda_mm.json"))
_SUSTAIN_S = float(os.environ.get("KDA_MM_SUSTAIN_S", "0"))
_SUSTAIN_IDLE_S = float(os.environ.get("KDA_MM_SUSTAIN_IDLE_S", "1"))
_SUSTAIN_BLOCK = int(os.environ.get("KDA_MM_SUSTAIN_BLOCK", "20"))
_TRACE_CALLS = int(os.environ.get("KDA_MM_TRACE_CALLS", "0"))
_TRACE_REPLAYS = 3
_RECORD_SETTLE_S = 0.3
_RECORD_TIMEOUT_S = 10.0
_T = 32
_TP_AXIS = 1


def _activation(case: dict) -> ttnn.UnaryWithParam | None:
    spec = case.get("activation")
    if spec is None:
        return None
    name, *params = [spec] if isinstance(spec, str) else spec
    return ttnn.UnaryWithParam(getattr(ttnn.UnaryOpType, name.upper()), *[float(p) for p in params])


def _program_config(case: dict):
    m_tiles, n_tiles = case["M"] // _T, case["N"] // _T
    grid = ttnn.CoreCoord(case["cols"], case["rows"])
    if case.get("kind", "2d") == "1d":
        cores = case["cols"] * case["rows"]
        per_core_m = case.get("per_core_M", m_tiles)
        per_core_n = case.get("per_core_N", -(-n_tiles // cores))
        return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=grid,
            in0_block_w=case["k_block"],
            out_subblock_h=case["sub_h"],
            out_subblock_w=case["sub_w"],
            out_block_h=case.get("out_block_h", per_core_m),
            out_block_w=case.get("out_block_w", per_core_n),
            per_core_M=per_core_m,
            per_core_N=per_core_n,
            fuse_batch=True,
            fused_activation=_activation(case),
            mcast_in0=True,
        )
    per_core_m = case.get("per_core_M", -(-m_tiles // case["rows"]))
    per_core_n = case.get("per_core_N", -(-n_tiles // case["cols"]))
    return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=grid,
        in0_block_w=case["k_block"],
        out_subblock_h=case["sub_h"],
        out_subblock_w=case["sub_w"],
        out_block_h=case.get("out_block_h", per_core_m),
        out_block_w=case.get("out_block_w", per_core_n),
        per_core_M=per_core_m,
        per_core_N=per_core_n,
        transpose_mcast=case.get("transpose_mcast", False),
        fused_activation=_activation(case),
        fuse_batch=True,
    )


def _inputs(mesh_device: ttnn.MeshDevice, case: dict) -> tuple[ttnn.Tensor, ttnn.Tensor]:
    """The case's activation and weight: replicated, or (mmrs) with K split over the TP axis."""
    mmrs = case.get("kind") == "mmrs"
    shape = tuple(mesh_device.shape)

    def mapper(dim: int):
        if mmrs:
            return ttnn.ShardTensor2dMesh(mesh_device, dims=(None, dim), mesh_shape=shape)
        return ttnn.ReplicateTensorToMesh(mesh_device)

    torch.manual_seed(0)
    a = ttnn.from_torch(
        torch.randn(1, 1, case["M"], case["K"]) * 0.1,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.L1_MEMORY_CONFIG if case.get("in0") == "L1" else ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=mapper(3),
    )
    w = ttnn.from_torch(
        torch.randn(1, 1, case["K"], case["N"]) * 0.02,
        dtype=getattr(ttnn, case.get("wdtype", "bfloat8_b")),
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=mapper(2),
    )
    return a, w


def _linear(mesh_device: ttnn.MeshDevice, case: dict, a: ttnn.Tensor, w: ttnn.Tensor):
    compute = ttnn.init_device_compute_kernel_config(
        mesh_device.arch(),
        math_fidelity=getattr(ttnn.MathFidelity, case.get("fidelity", "HiFi2")),
        fp32_dest_acc_en=case.get("fp32_acc", True),
        packer_l1_acc=case.get("packer_l1_acc", True),
    )
    memory = ttnn.L1_MEMORY_CONFIG if case.get("out", "L1") == "L1" else ttnn.DRAM_MEMORY_CONFIG
    config = _program_config(case)
    return lambda: ttnn.linear(a, w, program_config=config, memory_config=memory, compute_kernel_config=compute)


def _mmrs(case: dict, a: ttnn.Tensor, w: ttnn.Tensor, num_links: int):
    from ttnn.operations.matmul_reduce_scatter import matmul_reduce_scatter

    compute = ttnn.ComputeConfigDescriptor(
        math_fidelity=getattr(ttnn.MathFidelity, case.get("fidelity", "HiFi2")), fp32_dest_acc_en=True
    )
    return lambda: matmul_reduce_scatter(
        a,
        w,
        cluster_axis=_TP_AXIS,
        scatter_dim=-1,
        topology=ttnn.Topology.Ring,
        num_links=num_links,
        compute_kernel_config=compute,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        staging="ephemeral",
        balance_lines=True,
    )


@contextlib.contextmanager
def _overrides(module, values: dict):
    """Set module constants for the duration of the block."""
    saved = {}
    try:
        for name, value in values.items():
            if not hasattr(module, name):
                raise AttributeError(f"{module.__name__} has no knob {name}")
            saved[name] = getattr(module, name)
            setattr(module, name, value)
        yield
    finally:
        for name, value in saved.items():
            setattr(module, name, value)


def _chip0(tensor: ttnn.Tensor) -> torch.Tensor:
    return ttnn.to_torch(ttnn.get_device_tensors(tensor)[0]).float()


def _profile(mesh_device: ttnn.MeshDevice, run) -> tuple[object, dict]:
    """Run under the realtime profiler: (run's result, {chip: [(start us, end us)] in start order})."""
    records, dropped = [], [0]

    def collect(batch):
        dropped[0] += int(batch.dropped)
        for record in batch.records:
            frequency = float(record.frequency)
            start, end = int(record.start_timestamp), int(record.end_timestamp)
            if record.runtime_id and frequency > 0 and end > start:
                records.append((int(record.chip_id), start / frequency / 1e3, end / frequency / 1e3))

    handle = ttnn.device.RegisterProgramRealtimeProfilerCallback(collect)
    try:
        result = run()
        ttnn.synchronize_device(mesh_device)
        deadline, last, changed = time.monotonic() + _RECORD_TIMEOUT_S, -1, time.monotonic()
        while time.monotonic() < deadline:
            if len(records) != last:
                last, changed = len(records), time.monotonic()
            elif records and time.monotonic() - changed >= _RECORD_SETTLE_S:
                break
            time.sleep(0.01)
    finally:
        ttnn.device.UnregisterProgramRealtimeProfilerCallback(handle)
    assert records and not dropped[0], f"realtime profiler: {len(records)} records, {dropped[0]} dropped"
    per_chip: dict = {}
    for chip, start, end in sorted(records, key=lambda r: r[1]):
        per_chip.setdefault(chip, []).append((start, end))
    return result, per_chip


def _spread(per_chip: dict) -> list[float]:
    """[min, median, max] over the chips of each chip's median value."""
    medians = [statistics.median(v) for v in per_chip.values() if v]
    return [min(medians), statistics.median(medians), max(medians)] if medians else []


def _by_mesh(mesh_device: ttnn.MeshDevice, per_chip: dict) -> list[list[float]]:
    """Each chip's median value laid out as the mesh: rows x columns (SP x TP), rounded to 0.1 us."""
    ids = [int(i) for i in mesh_device.get_device_ids()]
    cols = tuple(mesh_device.shape)[1]
    values = [round(statistics.median(per_chip[c]), 1) if per_chip.get(c) else None for c in ids]
    return [values[r : r + cols] for r in range(0, len(values), cols)]


def _trace_profile(mesh_device: ttnn.MeshDevice, run) -> dict:
    """Profile _TRACE_CALLS back-to-back replays of the traced op: per-chip first / steady durations and period."""
    trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    out = run()
    ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)

    def replays():
        for _ in range(_TRACE_CALLS):
            ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)

    first, steady, periods = {}, {}, {}
    try:
        for replay in range(_TRACE_REPLAYS + 1):
            _, per_chip = _profile(mesh_device, replays)
            if replay == 0:
                continue  # warm-up
            for chip, spans in per_chip.items():
                durations = [end - start for start, end in spans]
                first.setdefault(chip, []).append(durations[0])
                steady.setdefault(chip, []).extend(durations[1:])
                starts = [start for start, _ in spans[1:]]
                periods.setdefault(chip, []).extend(b - a for a, b in zip(starts, starts[1:]))
    finally:
        ttnn.release_trace(mesh_device, trace_id)
        ttnn.deallocate(out)
    result = {
        "trace_first_chips_us": _spread(first),
        "trace_steady_chips_us": _spread(steady),
        "trace_steady_mesh_us": _by_mesh(mesh_device, steady),
    }
    if any(periods.values()):
        result["trace_period_us"] = _spread(periods)[1]
    return result


def _sustain(mesh_device: ttnn.MeshDevice, run, telemetry: Telemetry) -> dict:
    """Replay the traced op back to back for _SUSTAIN_S seconds: wall, clock and power over the second half."""
    trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    out = run()
    ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
    try:
        ttnn.synchronize_device(mesh_device)
        time.sleep(_SUSTAIN_IDLE_S)
        blocks = []
        run_start = time.perf_counter()
        while time.perf_counter() - run_start < _SUSTAIN_S:
            start = time.perf_counter()
            for _ in range(_SUSTAIN_BLOCK):
                ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
            ttnn.synchronize_device(mesh_device)
            end = time.perf_counter()
            blocks.append((start, end, (end - start) * 1e3 / _SUSTAIN_BLOCK))
    finally:
        ttnn.release_trace(mesh_device, trace_id)
        ttnn.deallocate(out)
    half = run_start + _SUSTAIN_S / 2
    ms = statistics.median(ms for start, _, ms in blocks if start >= half)
    window = telemetry.window(half, blocks[-1][1])
    return {
        "sustain_ms_per_call": ms,
        "sustain_first_ms_per_call": blocks[0][2],
        "sustain_aiclk_mean_mhz": window.get("aiclk_mean_mhz"),
        "sustain_aiclk_min_mhz": window.get("aiclk_min_mhz"),
        "sustain_power_total_w": window.get("power_total_mean_w"),
        "sustain_mj_per_call": window["power_total_mean_w"] * ms if window else None,
        "sustain_temp_max_c": window.get("temp_max_c"),
    }


@pytest.mark.parametrize(
    "mesh_device,device_params",
    [
        pytest.param(
            (8, 4),
            torus_xy_device_params(l1_small_size=_L1_SMALL_SIZE, worker_l1_size=KDA_WORKER_L1_SIZE),
            marks=pytest.mark.requires_mesh_topology(mesh_shape=(8, 4), topology="mesh-8x4"),
            id="SP8xTP4-torus-xy",
        ),
    ],
    indirect=True,
)
def test_projection_microbench(mesh_device: ttnn.MeshDevice, device_params: dict) -> None:
    """Time each KDA_MM_CASES config of ttnn.linear or matmul_reduce_scatter."""
    if not _CASES:
        pytest.skip("KDA_MM_CASES is empty")
    require_realtime_profiler("the projection microbenchmark")
    telemetry = Telemetry()
    if _SUSTAIN_S > 0 and not telemetry.available:
        pytest.skip("KDA_MM_SUSTAIN_S needs the Blackhole sysfs telemetry")
    mrs_descriptor, num_links = None, None
    if any(case.get("kind") == "mmrs" for case in _CASES):
        from ttnn.operations.matmul_reduce_scatter import matmul_reduce_scatter_program_descriptor as mrs_descriptor

        from models.tt_transformers.tt.ccl import TT_CCL

        mrs_descriptor.prepare_mesh(mesh_device)
        num_links = TT_CCL(mesh_device).get_num_links(_TP_AXIS)
    if _SUSTAIN_S > 0:
        telemetry.start()
    results, outputs, tensors = [], {}, {}
    try:
        for case in _CASES:
            mmrs = case.get("kind") == "mmrs"
            key = (mmrs, case["M"], case["K"], case["N"], case.get("wdtype", "bfloat8_b"), case.get("in0", "DRAM"))
            if key not in tensors:
                for old in tensors.values():
                    for t in old:
                        ttnn.deallocate(t)
                tensors.clear()
                tensors[key] = _inputs(mesh_device, case)
            a, w = tensors[key]
            entry = dict(case)
            try:
                with _overrides(mrs_descriptor, case.get("mrs", {})) if mmrs else contextlib.nullcontext():
                    run = _mmrs(case, a, w, num_links) if mmrs else _linear(mesh_device, case, a, w)
                    out = run()  # compile
                    if mmrs:
                        plan = getattr(sys.modules[mrs_descriptor.__package__], "_LAST_PLAN", None) or {}
                        entry["plan"] = {k: v for k, v in plan.items() if isinstance(v, (int, str, bool, tuple, list))}
                    outputs[case["name"]] = _chip0(out)
                    ttnn.deallocate(out)
                    if case.get("compare") in outputs:
                        ref, got = outputs[case["compare"]], outputs[case["name"]]
                        entry["bit_identical"] = bool(torch.equal(ref, got))
                        entry["max_abs_diff"] = float((ref - got).abs().max())
                    durations, chip_us = [], {}
                    for _ in range(_RUNS):
                        out, per_chip = _profile(mesh_device, run)
                        ttnn.deallocate(out)
                        run_us = {chip: sum(end - start for start, end in spans) for chip, spans in per_chip.items()}
                        durations.append(max(run_us.values()))
                        for chip, us in run_us.items():
                            chip_us.setdefault(chip, []).append(us)
                    entry.update(
                        us=statistics.median(durations),
                        runs_us=durations,
                        chips_us=_spread(chip_us),
                        chips_mesh_us=_by_mesh(mesh_device, chip_us),
                    )
                    if _TRACE_CALLS > 0:
                        entry.update(_trace_profile(mesh_device, run))
                    if _SUSTAIN_S > 0:
                        entry.update(_sustain(mesh_device, run, telemetry))
            except Exception as error:  # a config the op rejects, or one that does not fit L1
                entry["error"] = str(error).splitlines()[0][:300]
            results.append(entry)
            print("KDA_MM=" + json.dumps(entry, default=str))
    finally:
        if _SUSTAIN_S > 0:
            telemetry.stop()
    _OUT.parent.mkdir(parents=True, exist_ok=True)
    _OUT.write_text(json.dumps(results, default=str))
