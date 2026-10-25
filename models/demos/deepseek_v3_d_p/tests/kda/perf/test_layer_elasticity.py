# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Elasticity of the Galaxy KDA layer's walls to added time and added energy (a measurement tool, not a gate).

Under back-to-back load every chip runs at its power limit, so the sustained wall depends on the energy a call takes
as well as on its critical path, and the gate wall (short bursts from a cool chip) on the critical path and on how
long the clock holds. This injects known amounts of both (see delay_injection.DelayConfig) and reads both walls:

- "spin" injections add critical-path time and no energy: the response per us is the value of a us of critical path.
- "matmul" injections add the same time of HiFi2 FPU work on every worker core: their response minus the spin's at
  equal time is the value of that FPU energy. The energy of the injected work is measured on its own (below).

For each configuration (KDA_ELAST_CONFIGS, comma separated; repeats dropped) it captures the layer with that injection
in a trace, idles KDA_ELAST_GATE_IDLE_S, times the gate wall as test_layer_perf does (5 samples of 10 back-to-back
calls, median), then the sustained wall as test_layer_power does (blocks of 10 calls for KDA_ELAST_SECONDS, median of
the second half, with each chip's clock and power). Configurations run in KDA_ELAST_ROUNDS interleaved rounds, the
order rotating each round, with an uninjected baseline between every two of them; each delta is taken against the
baselines around it, so drift cancels.

Before the rounds, with KDA_ELAST_WORK_ENERGY=1 (the default), the injected matmul runs alone in a loop at duty 0.4
(about the nominal clock) and duty 1 (power-limited), as test_layer_energy's test_wait_power does, and reports its
energy per tile-matmul per chip; its circular buffers hold what the layer's last program left, as they do when it is
injected after "output". Results go to KDA_ELAST_OUT as JSON and to KDA_ELAST= lines.
"""

from __future__ import annotations

import contextlib
import json
import os
import statistics
import time
from pathlib import Path

import pytest

import ttnn
from models.demos.deepseek_v3_d_p.tests.kda.perf.delay_injection import (
    MATMUL_CYCLES_PER_SUBBLOCK,
    MATMUL_TILES_PER_SUBBLOCK,
    DelayConfig,
    injection_io,
    matmul_program,
    parse_configs,
    release_io,
    worker_cores,
)
from models.demos.deepseek_v3_d_p.tests.kda.perf.telemetry import Telemetry
from models.demos.deepseek_v3_d_p.tests.kda.perf.test_layer_energy import _MESH, _measure
from models.demos.deepseek_v3_d_p.tests.kda.perf.test_layer_perf import (
    _GALAXY_ACTUAL_START,
    _REPETITIONS,
    _TIMING_SAMPLES,
    _capture_layer_trace,
    _release_layer_trace,
)
from models.demos.deepseek_v3_d_p.tests.kda.perf.test_layer_power import sustained_run
from models.demos.deepseek_v3_d_p.tests.kda.utils import make_kimi_k3_device_case, make_synthetic_kimi_k3_test_case
from models.demos.deepseek_v3_d_p.utils.chunk_config import PREFILL_CHUNK_TOKENS

_DEFAULT_CONFIGS = (
    "output:10:all,output:20:all,output:40:all,"
    "output:10:all:matmul,output:20:all:matmul,output:40:all:matmul,"
    "input_projection:20:all"
)
_CONFIGS = os.environ.get("KDA_ELAST_CONFIGS", _DEFAULT_CONFIGS).split(",")
_ROUNDS = int(os.environ.get("KDA_ELAST_ROUNDS", "6"))
_SECONDS = float(os.environ.get("KDA_ELAST_SECONDS", "4"))
_GATE_IDLE_S = float(os.environ.get("KDA_ELAST_GATE_IDLE_S", "2"))
_WORK_ENERGY = os.environ.get("KDA_ELAST_WORK_ENERGY", "1") == "1"
_OUT = Path(os.environ.get("KDA_ELAST_OUT", "generated/kda_elasticity.json"))
_BLOCK = 10
# Configurations measured between two baselines.
_BASELINE_EVERY = 2
# The standalone matmul: 4096 subblocks (about 3.4 ms) per program, 4 programs per step.
_WORK_SUBBLOCKS, _WORK_PROGRAMS = 4096, 4
pytestmark = pytest.mark.timeout(7200)


def _per_chip(window: dict, key: str, chips: int) -> float | None:
    return window[key] / chips if key in window else None


@pytest.mark.parametrize("mesh_device,device_params", [_MESH], indirect=True)
def test_layer_elasticity(mesh_device: ttnn.MeshDevice, device_params: dict) -> None:
    """Gate and sustained wall responses of the SP8xTP4 layer to injected time and FPU work."""
    telemetry = Telemetry()
    if not telemetry.available:
        pytest.skip("no Blackhole sysfs telemetry on this host")
    configs = parse_configs(_CONFIGS)
    io = injection_io(mesh_device)  # shared by every injection, allocated before the layer
    case = make_synthetic_kimi_k3_test_case(sequence=PREFILL_CHUNK_TOKENS)
    layer, hidden = make_kimi_k3_device_case(mesh_device, case, tensor_parallel_axis=1, cache_weights=False)
    actual_start, actual_end = _GALAXY_ACTUAL_START, _GALAXY_ACTUAL_START + PREFILL_CHUNK_TOKENS
    chips = mesh_device.get_num_devices()

    def measure(config: DelayConfig | None) -> dict:
        with config.inject(layer, io) if config is not None else contextlib.nullcontext() as injected:
            trace = _capture_layer_trace(mesh_device, layer, hidden, actual_start, actual_end)
            try:
                time.sleep(_GATE_IDLE_S)
                gate = []
                for _ in range(_TIMING_SAMPLES):
                    start = time.perf_counter()
                    for _ in range(_REPETITIONS):
                        ttnn.execute_trace(mesh_device, trace.trace_id, cq_id=0, blocking=False)
                    ttnn.synchronize_device(mesh_device)
                    gate.append((time.perf_counter() - start) * 1e3 / _REPETITIONS)
                sustained = sustained_run(mesh_device, trace.trace_id, telemetry, _SECONDS, _BLOCK)
            finally:
                _release_layer_trace(mesh_device, trace)
        if injected is not None:
            # The warm-up call and the traced call each ran the point's program once.
            assert injected.dispatches == 2, f"{config}: {injected.dispatches} injections in two layer calls"
        steady, idle = sustained["steady_telemetry"], sustained["idle_telemetry"]
        busy_w = _per_chip(steady, "power_total_mean_w", chips)
        idle_w = _per_chip(idle, "power_total_mean_w", chips)
        ms = sustained["steady_ms_per_call"]
        return {
            "gate_ms": statistics.median(gate),
            "gate_samples_ms": gate,
            "sustained_ms": ms,
            "aiclk_mean_mhz": steady.get("aiclk_mean_mhz"),
            "aiclk_min_mhz": steady.get("aiclk_min_mhz"),
            "busy_w_per_chip": busy_w,
            "idle_w_per_chip": idle_w,
            # Dynamic energy per call per chip at the sustained operating point (mJ).
            "energy_mj_per_chip": None if busy_w is None or idle_w is None else (busy_w - idle_w) * ms,
            "temp_max_c": steady.get("temp_max_c"),
        }

    telemetry.start()
    try:
        measure(None)  # compile and first-capture jitter
        work_energy = {}
        if _WORK_ENERGY:
            program = matmul_program(worker_cores(mesh_device), _WORK_SUBBLOCKS)
            ttnn.generic_op(io, program)  # compile
            ttnn.synchronize_device(mesh_device)

            def step() -> None:
                for _ in range(_WORK_PROGRAMS):
                    ttnn.generic_op(io, program)

            tiles = _WORK_SUBBLOCKS * MATMUL_TILES_PER_SUBBLOCK * worker_cores(mesh_device).num_cores()
            for duty in (0.4, 1.0):
                result = _measure(mesh_device, telemetry, step, _WORK_PROGRAMS, _SECONDS, duty)
                result.pop("energy_uj_per_chip")
                result["cores_per_chip"] = worker_cores(mesh_device).num_cores()
                result["nj_per_tile_matmul"] = result["energy_uj_per_chip_mean"] * 1e3 / tiles
                result["nominal_us_per_subblock"] = MATMUL_CYCLES_PER_SUBBLOCK / 1350
                result["us_per_subblock_measured"] = result["busy_ms_per_call"] * 1e3 / _WORK_SUBBLOCKS
                work_energy[f"duty{duty:g}"] = result
                print("KDA_ELAST_WORK=" + json.dumps({"duty": duty, **result}))

        rounds = []
        for round_index in range(_ROUNDS):
            shift = round_index * len(configs) // max(_ROUNDS, 1)
            order = configs[shift:] + configs[:shift]
            sequence: list[DelayConfig | None] = [None]
            for start in range(0, len(order), _BASELINE_EVERY):
                sequence += order[start : start + _BASELINE_EVERY] + [None]
            measured = [(config, measure(config)) for config in sequence]
            slots = [i for i, (config, _) in enumerate(measured) if config is None]
            entries = {}
            for i, (config, values) in enumerate(measured):
                if config is None:
                    continue
                before = max(b for b in slots if b < i)
                after = min(b for b in slots if b > i)
                weight = (i - before) / (after - before)
                entry = dict(values)
                for key in ("gate_ms", "sustained_ms", "energy_mj_per_chip", "aiclk_mean_mhz"):
                    lo, hi = measured[before][1][key], measured[after][1][key]
                    if None in (lo, hi, values[key]):
                        continue
                    baseline = lo + (hi - lo) * weight
                    entry[f"baseline_{key}"] = baseline
                    entry[f"delta_{key}"] = values[key] - baseline
                entries[str(config)] = entry
            baselines = [measured[b][1] for b in slots]
            rounds.append({"baselines": baselines, "configs": entries})
            print(
                f"KDA_ELAST round {round_index}: baselines gate "
                + " ".join(f"{b['gate_ms']:.4f}" for b in baselines)
                + " sustained "
                + " ".join(f"{b['sustained_ms']:.4f}" for b in baselines)
            )
            for name, entry in entries.items():
                print(
                    f"KDA_ELAST {round_index} {name:<28} gate {entry['delta_gate_ms'] * 1e3:+7.1f} us  sustained "
                    f"{entry['delta_sustained_ms'] * 1e3:+7.1f} us  E {entry.get('delta_energy_mj_per_chip', 0):+6.2f} "
                    f"mJ  clk {entry.get('delta_aiclk_mean_mhz', 0):+6.1f} MHz"
                )
            _OUT.parent.mkdir(parents=True, exist_ok=True)
            _OUT.write_text(
                json.dumps({"configs": [str(c) for c in configs], "work_energy": work_energy, "rounds": rounds})
            )
    finally:
        telemetry.stop()
        release_io(io)

    print(f"{'config':<28} {'gate us':>8} {'sust us':>8}  (median delta over {_ROUNDS} rounds) -> {_OUT}")
    for config in configs:
        name = str(config)
        gate = statistics.median(r["configs"][name]["delta_gate_ms"] * 1e3 for r in rounds)
        sustained = statistics.median(r["configs"][name]["delta_sustained_ms"] * 1e3 for r in rounds)
        print(f"KDA_ELAST_SUMMARY {name:<28} {gate:8.1f} {sustained:8.1f}")
