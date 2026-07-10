# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Correctness + single-core device profiling for compute fusion vs L1 round-trips.

Correctness is the only pass/fail: every variant of every scenario must match torch. Perf is
measured (DEVICE KERNEL DURATION [ns], in-process via ReadDeviceProfiler) and reported, never
asserted.
"""

import os

os.environ.setdefault("TT_METAL_DEVICE_PROFILER", "1")
os.environ.setdefault("TT_METAL_PROFILER_MID_RUN_DUMP", "1")
os.environ.setdefault("TT_METAL_PROFILER_CPP_POST_PROCESS", "1")

import socket
import statistics
from pathlib import Path

import torch
import ttnn
from loguru import logger

from ttnn.operations.examples.compute_fusion import (
    PHASE_ZONES,
    SCENARIOS,
    create_sharded_memory_config,
    run_fusion,
    variants_for,
)

TILE = 32
_DURATION_KEY = "DEVICE KERNEL DURATION [ns]"
_BASELINE = {s: f"{s}.unfused" for s in SCENARIOS}


# =============================================================================
# Inputs + golden (quantize to bf16 first, so tolerance covers only op-internal error)
# =============================================================================
def _quant(t):
    return t.to(torch.bfloat16).to(torch.float32)


def _to_device(t, device, num_tiles):
    return ttnn.from_torch(
        t.to(torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=create_sharded_memory_config(num_tiles),
    )


def _make_case(device, scenario, num_tiles, seed=7):
    torch.manual_seed(seed)
    w = num_tiles * TILE
    if scenario == "sfpu_chain":
        x = torch.rand(TILE, w) * 1.9 + 0.1  # [0.1, 2.0], >=0 for sqrt
        y = torch.rand(TILE, w) - 0.5  # [-0.5, 0.5]
        expected = torch.exp(torch.sqrt(_quant(x)) + _quant(y))
        return [_to_device(x, device, num_tiles), _to_device(y, device, num_tiles)], expected, "full"
    if scenario == "fpu_sfpu":
        x = torch.rand(TILE, w) * 1.9 + 0.1
        b = torch.rand(TILE, w) + 0.5  # [0.5, 1.5]
        expected = torch.sqrt(_quant(x)) * _quant(b)
        return [_to_device(x, device, num_tiles), _to_device(b, device, num_tiles)], expected, "full"
    if scenario == "reduce_recip":
        x = torch.rand(TILE, w) * 0.3 + 0.2  # [0.2, 0.5], positive
        expected = 1.0 / _quant(x).sum(dim=-1, keepdim=True)  # [32, 1]
        return [_to_device(x, device, num_tiles)], expected, "reduce_col0"
    raise ValueError(scenario)


def _check(output, expected, kind, label):
    actual = ttnn.to_torch(output).to(torch.float32)
    if kind == "reduce_col0":
        actual = actual[..., :TILE, :1]
        rtol, atol = 0.06, 1e-4
    else:
        rtol, atol = 0.06, 0.5
    torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol, msg=f"{label} mismatch")


# =============================================================================
# In-process device-kernel timing
# =============================================================================
def _read_kernel_ns(device):
    ttnn.ReadDeviceProfiler(device)
    total, found = 0.0, False
    for programs in (ttnn.get_latest_programs_perf_data() or {}).values():
        for program in programs:
            entry = (getattr(program, "program_analyses_results", None) or {}).get(_DURATION_KEY)
            if entry is not None:
                total += float(entry.duration)
                found = True
    return total if found else None


def _measure(device, runners, trials, kernel_iters):
    for run in runners.values():
        run()
    ttnn.synchronize_device(device)
    _read_kernel_ns(device)  # discard warmup window
    samples = {name: [] for name in runners}
    for trial in range(trials + 1):
        for name, run in runners.items():
            run()
            duration = _read_kernel_ns(device)
            assert duration is not None, f"no profiler data for {name}"
            if trial:  # discard first timed pass
                samples[name].append(duration / kernel_iters)
    return samples


# =============================================================================
# Config knobs (env-driven so __main__ can set them)
# =============================================================================
def _selected_scenarios():
    sel = tuple(os.environ.get("CF_SCENARIOS", ",".join(SCENARIOS)).split(","))
    unknown = set(sel) - set(SCENARIOS)
    if unknown:
        raise ValueError(f"unknown CF_SCENARIOS: {sorted(unknown)}")
    return sel


def _int_list(name, default):
    return tuple(int(v) for v in os.environ.get(name, default).split(","))


def _blocks_for(scenario, blocks):
    return blocks if scenario in ("sfpu_chain", "fpu_sfpu") else (1,)


# =============================================================================
# Tests
# =============================================================================
def test_compute_fusion_correctness(device):
    for scenario in _selected_scenarios():
        for num_tiles in (2, 8):
            inputs, expected, kind = _make_case(device, scenario, num_tiles)
            for block_size in _blocks_for(scenario, (1, 4)):
                for variant in variants_for(scenario):
                    out = run_fusion(
                        inputs, variant=variant, num_tiles=num_tiles, block_size=block_size, kernel_iters=2
                    )
                    _check(out, expected, kind, f"{variant} n={num_tiles} blk={block_size}")


def test_compute_fusion_device_perf(device):
    scenarios = _selected_scenarios()
    tiles = _int_list("CF_TILES", "4,16,64")
    blocks = _int_list("CF_BLOCKS", "1,4")
    trials = int(os.environ.get("CF_TRIALS", "5"))
    kernel_iters = int(os.environ.get("CF_KERNEL_ITERS", "100"))

    results = []  # (scenario, num_tiles, block_size, {variant: [samples]})
    for scenario in scenarios:
        for num_tiles in tiles:
            inputs, expected, kind = _make_case(device, scenario, num_tiles)
            for block_size in _blocks_for(scenario, blocks):
                for variant in variants_for(scenario):  # correctness gate before timing
                    out = run_fusion(
                        inputs, variant=variant, num_tiles=num_tiles, block_size=block_size, kernel_iters=1
                    )
                    _check(out, expected, kind, f"{variant} n={num_tiles} blk={block_size}")
                runners = {
                    variant: (
                        lambda variant=variant, block_size=block_size: run_fusion(
                            inputs,
                            variant=variant,
                            num_tiles=num_tiles,
                            block_size=block_size,
                            kernel_iters=kernel_iters,
                        )
                    )
                    for variant in variants_for(scenario)
                }
                results.append((scenario, num_tiles, block_size, _measure(device, runners, trials, kernel_iters)))

    report = _format_report(
        results,
        box=socket.gethostname(),
        arch=os.environ.get("ARCH_NAME", str(device.arch())),
        trials=trials,
        kernel_iters=kernel_iters,
    )
    logger.info("\n" + report)
    if report_path := os.environ.get("CF_REPORT"):
        Path(report_path).write_text(report)


def _format_report(results, *, box, arch, trials, kernel_iters):
    lines = [
        "# Compute fusion vs L1 round-trips — single-core report",
        "",
        f"box={box}  arch={arch}  cores=1  placement=single-core sharded-L1  "
        f"N={trials} (median)  kernel-iters={kernel_iters} (steady-state)",
        "",
        "Metric: DEVICE KERNEL DURATION [ns] per expression evaluation. Speedup = unfused / variant.",
        "",
        "| Scenario | Tiles | Block | Variant | Median ns | Std/med | Speedup vs unfused |",
        "|---|---:|---:|---|---:|---:|---:|",
    ]
    for scenario, num_tiles, block_size, samples in results:
        base = statistics.median(samples[_BASELINE[scenario]])
        for variant, values in samples.items():
            median = statistics.median(values)
            std = statistics.pstdev(values) if len(values) > 1 else 0.0
            method = variant.split(".", 1)[1]
            lines.append(
                f"| {scenario} | {num_tiles} | {block_size} | {method} | {median:.1f} | "
                f"{std / median * 100:.1f}% | {base / median:.2f}x |"
            )
    return "\n".join(lines) + "\n"


# =============================================================================
# Micro-benchmark: per-phase device zone timings (DeviceZoneScopedN)
#
# Built with the CF_MICROBENCH define, each phase is wrapped in a device zone (CF_*). A
# compute-kernel zone records on all three TRISCs (unpack/math/pack), so per phase we see both
# the wall time (max of the three) and which engine dominates. The device profiler writes every
# marker to profile_log_device.csv; we isolate one launch by its (new) run-host-id and average
# each zone over the in-kernel loop.
# =============================================================================
_DEVICE_CSV = os.path.join(os.environ.get("TT_METAL_HOME", "."), "generated/profiler/.logs/profile_log_device.csv")
_RISC_LABEL = {"TRISC_0": "unpack", "TRISC_1": "math", "TRISC_2": "pack"}
_RISC_ORDER = ["unpack", "math", "pack"]


def _read_csv_rows(path):
    with open(path) as f:
        lines = f.read().splitlines()
    freq_mhz = 1000.0
    for part in lines[0].split(","):
        if "CHIP_FREQ" in part:
            freq_mhz = float(part.split(":")[1])
    rows = [[x.strip() for x in ln.split(",")] for ln in lines[2:] if ln.strip()]
    return [r for r in rows if len(r) >= 12], 1000.0 / freq_mhz


def _all_run_ids(path):
    if not os.path.exists(path):
        return set()
    rows, _ = _read_csv_rows(path)
    return {r[7] for r in rows}


def _zone_durations_for_new_run(path, seen_ids, zone_names, drop_first=True):
    """{zone: {engine: median_ns}} for rows whose run-host-id is new (this launch only)."""
    rows, ns_per_cycle = _read_csv_rows(path)
    starts, ends = {}, {}
    for r in rows:
        risc, cyc, run_id, zone, typ = r[3], r[5], r[7], r[10], r[11]
        if run_id in seen_ids or zone not in zone_names:
            continue
        (starts if typ == "ZONE_START" else ends).setdefault((risc, zone), []).append(int(cyc))
    out = {}
    for (risc, zone), s in starts.items():
        s.sort()
        e = sorted(ends.get((risc, zone), []))
        durs = [(ee - ss) * ns_per_cycle for ss, ee in zip(s, e)]
        if drop_first and len(durs) > 1:
            durs = durs[1:]
        if durs:
            out.setdefault(zone, {})[_RISC_LABEL.get(risc, risc)] = statistics.median(durs)
    return out


def _measure_zones(device, inputs, variant, num_tiles, kernel_iters):
    kw = dict(variant=variant, num_tiles=num_tiles, kernel_iters=kernel_iters, microbench=True)
    run_fusion(inputs, **kw)  # warmup + flush so its markers land under an id we mark 'seen'
    ttnn.ReadDeviceProfiler(device)
    seen = _all_run_ids(_DEVICE_CSV)
    run_fusion(inputs, **kw)  # measured
    ttnn.synchronize_device(device)
    ttnn.ReadDeviceProfiler(device)
    return _zone_durations_for_new_run(_DEVICE_CSV, seen, set(PHASE_ZONES[variant]))


def test_compute_fusion_microbench(device):
    scenarios = _selected_scenarios()
    num_tiles = int(os.environ.get("CF_MB_TILES", "32"))
    kernel_iters = int(os.environ.get("CF_MB_KERNEL_ITERS", "16"))

    results = []  # (scenario, variant, {zone: {engine: ns}})
    for scenario in scenarios:
        inputs, expected, kind = _make_case(device, scenario, num_tiles)
        for variant in variants_for(scenario):
            out = run_fusion(inputs, variant=variant, num_tiles=num_tiles, kernel_iters=2, microbench=True)
            _check(out, expected, kind, f"{variant} microbench")
            zones = _measure_zones(device, inputs, variant, num_tiles, kernel_iters)
            results.append((scenario, variant, zones))

    report = _format_microbench(
        results,
        box=socket.gethostname(),
        arch=os.environ.get("ARCH_NAME", str(device.arch())),
        num_tiles=num_tiles,
        kernel_iters=kernel_iters,
    )
    logger.info("\n" + report)
    if report_path := os.environ.get("CF_MB_REPORT"):
        Path(report_path).write_text(report)


def _phase_wall(engines):
    # phase wall time = the slowest engine (unpack/math/pack run concurrently, pipelined)
    return max(engines.values()) if engines else 0.0


def _format_microbench(results, *, box, arch, num_tiles, kernel_iters):
    lines = [
        "# Compute fusion — per-phase device-zone micro-benchmark",
        "",
        f"box={box}  arch={arch}  cores=1  placement=single-core sharded-L1  "
        f"num-tiles={num_tiles}  kernel-iters={kernel_iters}  metric=median ns per phase (per launch)",
        "",
        "Each phase is one `eltwise_chain` / `reduce` call. A compute zone records on all three "
        "TRISCs; `wall` is the slowest engine (they pipeline). `Σ wall` sums the variant's phases "
        "= the serial cost the whole-kernel number reflects.",
        "",
        "| Scenario | Variant | Phase | unpack ns | math ns | pack ns | wall ns |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for scenario, variant, zones in results:
        method = variant.split(".", 1)[1]
        total = 0.0
        for zone in PHASE_ZONES[variant]:
            eng = zones.get(zone, {})
            wall = _phase_wall(eng)
            total += wall
            cells = [f"{eng.get(e, 0.0):.0f}" for e in _RISC_ORDER]
            lines.append(f"| {scenario} | {method} | {zone} | {cells[0]} | {cells[1]} | {cells[2]} | {wall:.0f} |")
        if len(PHASE_ZONES[variant]) > 1:
            lines.append(f"| {scenario} | {method} | **Σ wall** | | | | **{total:.0f}** |")
    return "\n".join(lines) + "\n"
