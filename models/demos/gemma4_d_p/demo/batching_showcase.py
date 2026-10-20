# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Continuous-batching showcase: prompt scenarios through batching_scheduler.py, batched against serial baselines.

    pytest models/demos/gemma4_d_p/demo/batching_showcase.py -s

Plays each scenario in every mode, SHOWCASE_REPEATS times (default 3; the median-wall run is reported). Prints every
run, then wall time per mode with batching's speedup over each serial baseline, and mean latency of short (<= 4k) and
long prompts.

Env: DEMO_SLOTS (requests in flight, default 4), DEMO_CAPACITY (max prompt tokens, default 106496).
"""

import os
import random
from concurrent.futures import ThreadPoolExecutor

import pytest
import torch
from rich.console import Console
from rich.table import Table

from models.demos.gemma4_d_p.demo.batched_steps import TRACE_REGION_SIZE
from models.demos.gemma4_d_p.demo.batching_scheduler import BATCHED, MODES, SERIAL_CHUNKS, start_scheduler
from models.demos.gemma4_d_p.tests.test_factory import parametrize_mesh_with_fabric

console = Console(width=200)
SHORT_MAX = 4096  # prompts up to this many tokens count as short in the summaries

SIZES = {"2k": 2048, "4k": 4096, "6k": 6144, "8k": 8192, "32k": 32768, "64k": 65536, "106k": 106496}


def _burst(*labels):
    return [(0.0, label) for label in labels]


def _staggered(mean_gap_s, n=16, seed=0):
    """n prompts, mostly short, arriving with exponential gaps (a fixed seed: both modes, and every gap, see the same
    prompts in the same order)."""
    rng = random.Random(seed)
    at, arrivals = 0.0, []
    for _ in range(n):
        arrivals.append((at, rng.choices(["2k", "4k", "8k", "32k"], weights=[8, 4, 2, 1])[0]))
        at += rng.expovariate(1 / mean_gap_s)
    return arrivals


# name -> [(seconds after the scenario starts, prompt size)]
SCENARIOS = {
    "solo 2k": _burst("2k"),
    "solo 4k": _burst("4k"),
    "solo 6k": _burst("6k"),
    "solo 8k": _burst("8k"),
    "solo 106k": _burst("106k"),
    "burst 16 x 2k": _burst(*["2k"] * 16),
    "burst 64k + 8 x 2k": _burst("64k", *["2k"] * 8),
    "burst mixed": _burst("8k", "2k", "4k", "2k", "32k", "2k", "2k", "4k", "2k", "8k"),
    # About 1.4 s of serial work arriving over ~1.5 s (a lightly loaded server) and over ~0.5 s (a busy one).
    "staggered, trickle": _staggered(0.08),
    "staggered, busy": _staggered(0.025),
}


def show(title, results, wall):
    if not results:
        return
    longest = max(r["latency_ms"] for r in results)
    table = Table(title=title)
    for col in ("request", "tokens", "queue", "prefill", "latency", ""):
        table.add_column(col)
    for r in results:
        table.add_row(
            r["label"],
            f"{r['tokens']:,}",
            f"{r['queue_ms'] / 1000:.2f}s",
            f"{r['prefill_ms'] / 1000:.2f}s",
            f"{r['latency_ms'] / 1000:.2f}s",
            "█" * max(1, round(40 * r["latency_ms"] / longest)),
        )
    console.print(table)
    tokens = sum(r["tokens"] for r in results)
    console.print(f"{tokens:,} tokens in {wall:.1f}s wall ({tokens / wall:,.0f} tok/s)")


def mean_latency_s(results, short):
    """Mean latency in seconds of the short (<= SHORT_MAX tokens) or long prompts; None if there are none."""
    picked = [r["latency_ms"] for r in results if (r["tokens"] <= SHORT_MAX) == short]
    return sum(picked) / len(picked) / 1000 if picked else None


def _play(scheduler, arrivals):
    """Play the scenario; returns (results, wall seconds from first arrival to last done)."""
    _, requests = scheduler.play([(at, SIZES[label], f"#{i} {label}", 0) for i, (at, label) in enumerate(arrivals)])
    return [r.result() for r in requests], max(r.finished for r in requests) - requests[0].arrived


def _drive(scheduler, repeats):
    """Play every scenario in every mode; returns {scenario: {mode: (results, wall)}}, the median-wall run of each."""
    summary = {}
    try:
        for name, arrivals in SCENARIOS.items():
            summary[name] = {}
            for mode in MODES:
                scheduler.mode = mode
                played = sorted((_play(scheduler, arrivals) for _ in range(repeats)), key=lambda run: run[1])
                summary[name][mode] = played[len(played) // 2]
                walls = ", ".join(f"{wall:.3f}" for _, wall in played)
                show(f"{name}: {mode} (median of {repeats} runs; walls {walls} s)", *summary[name][mode])
    finally:
        scheduler.stop.set()
    return summary


def _seconds(value):
    return "-" if value is None else f"{value:.3f}"


def _summary_tables(summary):
    """Wall time per mode with batching's speedup over each serial baseline, and mean latency per mode."""
    walls = Table(title="wall time, first arrival to last completion (median-wall run); batching's speedup")
    latencies = Table(title="mean latency (queue + prefill), short (<= 4k) / long prompts")
    for table, cols in (
        (
            walls,
            (
                "scenario",
                "requests",
                *MODES,
                *(f"vs {m}" for m in SERIAL_CHUNKS),
                "vs best serial",
            ),
        ),
        (latencies, ("scenario", *MODES)),
    ):
        for col in cols:
            table.add_column(col)
    for name, runs in summary.items():
        wall = {mode: runs[mode][1] for mode in MODES}
        best = min(SERIAL_CHUNKS, key=wall.get)
        speedup = {mode: wall[mode] / wall[BATCHED] for mode in SERIAL_CHUNKS}
        walls.add_row(
            name,
            str(len(runs[BATCHED][0])),
            *(f"{wall[mode]:.3f}s" for mode in MODES),
            *(f"{speedup[mode]:.2f}x" for mode in SERIAL_CHUNKS),
            f"{speedup[best]:.2f}x ({best})",
        )
        latency = {
            mode: " / ".join(_seconds(mean_latency_s(runs[mode][0], s)) for s in (True, False)) for mode in MODES
        }
        latencies.add_row(name, *(latency[mode] for mode in MODES))
    return walls, latencies


@torch.no_grad()
@pytest.mark.timeout(0)
@parametrize_mesh_with_fabric([(8, 4)], device_params_extra={"trace_region_size": TRACE_REGION_SIZE})
def test_batching_showcase(mesh_device, reset_seeds):
    num_slots = int(os.environ.get("DEMO_SLOTS", "4"))
    capacity = int(os.environ.get("DEMO_CAPACITY", str(SIZES["106k"])))
    repeats = int(os.environ.get("SHOWCASE_REPEATS", "3"))
    scheduler, runner = start_scheduler(mesh_device, num_slots, capacity)
    try:
        # The scheduler owns the device on this thread; the scenarios are submitted from another, like the server's.
        with ThreadPoolExecutor(max_workers=1) as pool:
            driver = pool.submit(_drive, scheduler, repeats)
            scheduler.run()
            summary = driver.result()
    finally:
        runner.release()
    for table in _summary_tables(summary):
        console.print(table)
