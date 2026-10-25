# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Slack map of the Kimi-K3 KDA layer on Galaxy SP8xTP4 (a measurement tool, not a gate).

For each configuration ``point:delay_us:chips`` it delays the program ``point`` (see delay_injection.POINTS) by
``delay_us`` profiler-us on the chips ``chips`` ("all", or SP ranks and single "sp.tp" chips joined by "+", e.g. "4",
"3+4" or "4.2"; see delay_injection.DelayConfig), captures the layer call in a trace and measures its traced wall the way test_layer_perf does. Configurations run in
interleaved rounds with an undelayed baseline between every few of them, and each one's wall delta is taken against
the baselines around it in the same round, so run-to-run drift cancels. A delta close to the delay means the program
is on the layer's critical chain on those chips; a delta near zero means other chips' waits absorb it.

KDA_SLACK_CONFIGS lists the configurations (comma separated); without it they are every KDA_SLACK_POINTS (default all
points) x KDA_SLACK_DELAYS_US (default 10,30) x KDA_SLACK_CHIPS (default all). KDA_SLACK_ROUNDS sets the rounds
(default 3), KDA_SLACK_REPETITIONS the back-to-back calls per timing sample (default 10, as test_layer_perf),
KDA_SLACK_SETTLE_S an idle pause before each sample (default 0, as test_layer_perf) and KDA_SLACK_OUT the JSON written
(default generated/kda_slack.json).

The response depends on the load regime. test_layer_perf's samples follow one another, and under that sustained load
the layer runs about 2% slower than from idle (and slower still over longer runs, as with a power- or
thermal-limited clock), and the wall absorbs about 18 us of a delay on every chip, whichever program it follows. With
KDA_SLACK_SETTLE_S=0.5 each sample starts from idle and delays reach the wall in full, which maps the dependency
chain itself.
"""

from __future__ import annotations

import itertools
import json
import os
import statistics
import time
from pathlib import Path

import pytest

import ttnn
from models.demos.deepseek_v3_d_p.tests.fabric_profiles import torus_xy_device_params
from models.demos.deepseek_v3_d_p.tests.kda.perf.delay_injection import (
    POINTS,
    DelayConfig,
    injection_io,
    parse_configs,
    release_io,
)
from models.demos.deepseek_v3_d_p.tests.kda.perf.test_layer_perf import (
    _GALAXY_ACTUAL_START,
    _L1_SMALL_SIZE,
    _TIMING_SAMPLES,
    _capture_layer_trace,
    _release_layer_trace,
)
from models.demos.deepseek_v3_d_p.tests.kda.utils import make_kimi_k3_device_case, make_synthetic_kimi_k3_test_case
from models.demos.deepseek_v3_d_p.tt.kda.config import KDA_WORKER_L1_SIZE
from models.demos.deepseek_v3_d_p.utils.chunk_config import PREFILL_CHUNK_TOKENS

_ROUNDS = int(os.environ.get("KDA_SLACK_ROUNDS", "3"))
# Back-to-back calls per timing sample; test_layer_perf's wall uses 10.
_REPETITIONS = int(os.environ.get("KDA_SLACK_REPETITIONS", "10"))
# Idle seconds before each timing sample (0, as test_layer_perf: its samples follow one another).
_SETTLE_SECONDS = float(os.environ.get("KDA_SLACK_SETTLE_S", "0"))
_OUT = Path(os.environ.get("KDA_SLACK_OUT", "generated/kda_slack.json"))
# A full map runs hundreds of trace captures.
pytestmark = pytest.mark.timeout(7200)
# Configurations measured between two baselines.
_BASELINE_EVERY = 4


def _configs() -> list[DelayConfig]:
    listed = os.environ.get("KDA_SLACK_CONFIGS")
    if listed:
        return parse_configs(listed.split(","))
    points = os.environ.get("KDA_SLACK_POINTS", ",".join(POINTS)).split(",")
    delays = os.environ.get("KDA_SLACK_DELAYS_US", "10,30").split(",")
    chips = os.environ.get("KDA_SLACK_CHIPS", "all").split(",")
    return parse_configs(f"{p}:{d}:{r}" for p, d, r in itertools.product(points, delays, chips))


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
def test_kimi_k3_layer_slack(mesh_device: ttnn.MeshDevice, device_params: dict) -> None:
    """Write the traced-wall response of the SP8xTP4 layer to a delay after each chosen program."""
    configs = _configs()
    io = injection_io(mesh_device)  # shared by every injection, allocated before the layer
    actual_start = _GALAXY_ACTUAL_START
    actual_end = actual_start + PREFILL_CHUNK_TOKENS
    case = make_synthetic_kimi_k3_test_case(sequence=PREFILL_CHUNK_TOKENS)
    layer, hidden_tt = make_kimi_k3_device_case(mesh_device, case, tensor_parallel_axis=1, cache_weights=False)

    def wall_ms(config: DelayConfig | None) -> float:
        def measure() -> float:
            trace = _capture_layer_trace(mesh_device, layer, hidden_tt, actual_start, actual_end)
            try:
                samples, enqueue = [], []
                for _ in range(_TIMING_SAMPLES):
                    if _SETTLE_SECONDS:
                        time.sleep(_SETTLE_SECONDS)
                    start = time.perf_counter()
                    for _ in range(_REPETITIONS):
                        ttnn.execute_trace(mesh_device, trace.trace_id, cq_id=0, blocking=False)
                    enqueued = time.perf_counter()
                    ttnn.synchronize_device(mesh_device)
                    samples.append((time.perf_counter() - start) * 1e3 / _REPETITIONS)
                    enqueue.append((enqueued - start) * 1e3 / _REPETITIONS)
            finally:
                _release_layer_trace(mesh_device, trace)
            enqueue_ms.append(statistics.median(enqueue))
            return statistics.median(samples)

        if config is None:
            return measure()
        with config.inject(layer, io) as spin:
            wall = measure()
        # The warm-up call and the traced call each ran the point's program once.
        assert spin.dispatches == 2, f"{config}: {spin.dispatches} injections in two layer calls"
        return wall

    enqueue_ms: list[float] = []  # host time per call to enqueue the replays, per measurement
    wall_ms(None)  # compile and first-capture jitter
    rounds = []
    for round_index in range(_ROUNDS):
        # Baselines bracket every _BASELINE_EVERY configurations; the order rotates from round to round.
        shift = round_index * len(configs) // max(_ROUNDS, 1)
        order = configs[shift:] + configs[:shift]
        sequence: list[DelayConfig | None] = [None]
        for start in range(0, len(order), _BASELINE_EVERY):
            sequence += order[start : start + _BASELINE_EVERY] + [None]
        measured = [(config, wall_ms(config)) for config in sequence]
        baseline_slots = [i for i, (config, _) in enumerate(measured) if config is None]
        entries = {}
        for i, (config, wall) in enumerate(measured):
            if config is None:
                continue
            before = max(b for b in baseline_slots if b < i)
            after = min(b for b in baseline_slots if b > i)
            span = after - before
            baseline = measured[before][1] + (measured[after][1] - measured[before][1]) * (i - before) / span
            entries[str(config)] = {"wall_ms": wall, "baseline_ms": baseline, "delta_us": (wall - baseline) * 1e3}
        rounds.append({"baselines_ms": [measured[b][1] for b in baseline_slots], "configs": entries})
        print(f"KDA_SLACK round {round_index}: baselines " + " ".join(f"{measured[b][1]:.4f}" for b in baseline_slots))
    release_io(io)

    summary = {}
    for config in configs:
        deltas = [r["configs"][str(config)]["delta_us"] for r in rounds]
        summary[str(config)] = {
            "delay_us": config.delay_us,
            "median_delta_us": statistics.median(deltas),
            "deltas_us": deltas,
        }
    baselines = [b for r in rounds for b in r["baselines_ms"]]
    _OUT.parent.mkdir(parents=True, exist_ok=True)
    _OUT.write_text(
        json.dumps(
            {
                "actual_start": actual_start,
                "sequence_parallel_axis": layer.sequence_parallel_axis,
                "repetitions": _REPETITIONS,
                "settle_seconds": _SETTLE_SECONDS,
                "median_baseline_ms": statistics.median(baselines),
                "median_enqueue_ms_per_call": statistics.median(enqueue_ms),
                "rounds": rounds,
                "summary": summary,
            },
            indent=1,
        )
    )
    print(
        f"KDA_SLACK baseline median {statistics.median(baselines):.4f} ms over {len(baselines)}, host enqueue "
        f"{statistics.median(enqueue_ms):.4f} ms per call ({_REPETITIONS} calls per sample) -> {_OUT}"
    )
    print(f"{'config':<28} {'delta us':>9}  per round")
    for name, entry in summary.items():
        rounds_text = " ".join(f"{d:6.1f}" for d in entry["deltas_us"])
        print(f"KDA_SLACK {name:<28} {entry['median_delta_us']:7.1f}  {rounds_text}")
