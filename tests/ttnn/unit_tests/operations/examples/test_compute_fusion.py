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
os.environ.setdefault("TT_METAL_LOGGER_LEVEL", "error")  # silence per-read profiler histograms

import socket
import statistics
from pathlib import Path

import pytest
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
    if scenario == "fpu_chain":
        a = torch.rand(TILE, w) * 2 - 1  # [-1, 1]
        b = torch.rand(TILE, w) * 2 - 1  # [-1, 1]
        c = torch.rand(TILE, w) + 0.5  # [0.5, 1.5]
        # Both variants hold the intermediate a+b at bf16 precision: unfused packs it to a bf16 L1 CB;
        # dstreuse keeps it in a 16-bit DEST (fp32_dest_acc_en=False) and moves it to srcA, which is
        # no narrower. So the golden rounds the intermediate to bf16 too.
        expected = _quant(_quant(a) + _quant(b)) * _quant(c)
        inputs = [_to_device(t, device, num_tiles) for t in (a, b, c)]
        return inputs, expected, "fpu_chain"
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
    elif kind == "fpu_chain":
        # |a+b| <= 2 and c <= 1.5: one bf16 ulp of the intermediate (2^-7 at |a+b| in [1, 2)) times c is
        # <= 0.012, plus one ulp of the output's rounding (<= 0.0117); 1/64 covers a rounding-mode
        # difference (truncate vs round) at either step but not a wrong operand.
        rtol, atol = 1 / 64, 1 / 64
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
    return blocks if scenario in ("sfpu_chain", "fpu_sfpu", "fpu_chain") else (1,)


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


# =============================================================================
# Work per call: the same total tiles per launch, split into calls of n tiles
#
# A "call" is one pass of the scenario's expression over n tiles: for `dstreuse` one eltwise_chain
# call, for `unfused` the two chain calls (pass 1 -> L1 scratch, pass 2) that make the same result.
# Every call re-runs the per-op init, as a helper call does. The launch makes total/n calls, so every
# tile count does the same total work and the ns-per-tile column compares directly across rows.
# =============================================================================
_PC_SCENARIOS = ("fpu_chain", "fpu_sfpu")


def _pc_lever(scenario):
    """The single-pass variant compared against `unfused` (dstreuse for FPU consumers, else fused)."""
    names = variants_for(scenario)
    return f"{scenario}.dstreuse" if f"{scenario}.dstreuse" in names else f"{scenario}.fused"


def _chip_freq_mhz():
    """AICLK the profiler converted cycles with (CSV header, rewritten per device open)."""
    try:
        with open(_DEVICE_CSV) as f:
            header = f.readline()
    except OSError:
        return None
    for part in header.split(","):
        if "CHIP_FREQ" in part:
            return int(float(part.split(":")[1]))
    return None


def _max_abs_diff(a, b):
    return (ttnn.to_torch(a).float() - ttnn.to_torch(b).float()).abs().max().item()


def test_compute_fusion_per_call(device):
    scenarios = tuple(os.environ.get("CF_PC_SCENARIOS", ",".join(_PC_SCENARIOS)).split(","))
    unknown = set(scenarios) - set(SCENARIOS)
    if unknown:
        raise ValueError(f"unknown CF_PC_SCENARIOS: {sorted(unknown)}")
    tiles = _int_list("CF_PC_TILES", "1,2,4,6,8,16,32,64")
    blocks = _int_list("CF_PC_BLOCKS", "1,4")
    total = int(os.environ.get("CF_PC_TOTAL_TILES", "768"))
    trials = int(os.environ.get("CF_PC_TRIALS", "5"))

    results = []  # (scenario, n, blk, calls, {variant: [ns per call]})
    for scenario in scenarios:
        for n in tiles:
            calls = max(1, total // n)
            inputs, expected, kind = _make_case(device, scenario, n)
            for blk in sorted({min(b, n) for b in blocks}):
                outs = {}
                for variant in variants_for(scenario):  # correctness gate on the measured build
                    outs[variant] = run_fusion(inputs, variant=variant, num_tiles=n, block_size=blk, kernel_iters=calls)
                    _check(outs[variant], expected, kind, f"{variant} n={n} blk={blk} calls={calls}")
                lever = _pc_lever(scenario)
                diff = _max_abs_diff(outs[lever], outs[_BASELINE[scenario]])
                runners = {
                    variant: (
                        lambda variant=variant, n=n, blk=blk, calls=calls: run_fusion(
                            inputs, variant=variant, num_tiles=n, block_size=blk, kernel_iters=calls
                        )
                    )
                    for variant in variants_for(scenario)
                }
                results.append((scenario, n, blk, calls, diff, _measure(device, runners, trials, calls)))

    report = _format_per_call(
        results,
        box=socket.gethostname(),
        arch=os.environ.get("ARCH_NAME", str(device.arch())),
        freq=_chip_freq_mhz(),
        trials=trials,
        total=total,
    )
    logger.info("\n" + report)
    if report_path := os.environ.get("CF_PC_REPORT"):
        Path(report_path).write_text(report)


def _format_per_call(results, *, box, arch, freq, trials, total):
    lines = [
        "# Compute fusion — work per call (DEST reuse vs a second pass through L1)",
        "",
        f"box={box}  arch={arch}  CHIP_FREQ={freq} MHz  cores=1  placement=single-core sharded-L1  "
        f"N={trials} interleaved trials (median)  total={total} tiles per launch, split into calls of n tiles",
        "",
        "Metric: DEVICE KERNEL DURATION [ns] / calls per launch = ns per call; ns/tile = ns per call / n. "
        "Speedup = unfused / variant. max|lever-unfused| = largest elementwise output difference.",
        "",
        "| Scenario | Tiles per call | Block | Calls per launch | Variant | ns per call | ns per tile | "
        "Std/med | Speedup vs unfused | max abs diff vs unfused |",
        "|---|---:|---:|---:|---|---:|---:|---:|---:|---:|",
    ]
    for scenario, n, blk, calls, diff, samples in results:
        base = statistics.median(samples[_BASELINE[scenario]])
        lever = _pc_lever(scenario)
        for variant, values in samples.items():
            median = statistics.median(values)
            std = statistics.pstdev(values) if len(values) > 1 else 0.0
            method = variant.split(".", 1)[1]
            d = f"{diff:.4g}" if variant == lever else ""
            lines.append(
                f"| {scenario} | {n} | {blk} | {calls} | {method} | {median:.1f} | {median / n:.1f} | "
                f"{std / median * 100:.1f}% | {base / median:.3f}x | {d} |"
            )
    return "\n".join(lines) + "\n"


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
# marker to profile_log_device.csv; we isolate one launch by the rows it appends and average
# each zone over the in-kernel loop.
# =============================================================================
_DEVICE_CSV = os.path.join(os.environ.get("TT_METAL_HOME", "."), "generated/profiler/.logs/profile_log_device.csv")
_RISC_LABEL = {"TRISC_0": "unpack", "TRISC_1": "math", "TRISC_2": "pack"}
_RISC_ORDER = ["unpack", "math", "pack"]


def _csv_size(path):
    return os.path.getsize(path) if os.path.exists(path) else 0


def _read_csv_rows(path, offset=0):
    """Rows appended after byte `offset` (all rows when 0). The CSV grows for the whole pytest
    session, so re-reading it in full per launch gets slow once earlier perf tests have filled it."""
    size = os.path.getsize(path)
    with open(path, "rb") as f:
        header = f.readline().decode()
        f.readline()  # column names
        if f.tell() < offset <= size:
            f.seek(offset)
        lines = f.read().decode().splitlines()
    freq_mhz = 1000.0
    for part in header.split(","):
        if "CHIP_FREQ" in part:
            freq_mhz = float(part.split(":")[1])
    rows = [[x.strip() for x in ln.split(",")] for ln in lines if ln.strip()]
    return [r for r in rows if len(r) >= 12], 1000.0 / freq_mhz


def _zone_durations_for_new_run(path, offset, zone_names, drop_first=True):
    """{zone: {engine: median_ns}} for rows appended after `offset` (this launch only)."""
    rows, ns_per_cycle = _read_csv_rows(path, offset)
    starts, ends = {}, {}
    for r in rows:
        risc, cyc, zone, typ = r[3], r[5], r[10], r[11]
        if zone not in zone_names:
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
    run_fusion(inputs, **kw)  # warmup + flush so its markers land before the measured launch
    ttnn.ReadDeviceProfiler(device)
    offset = _csv_size(_DEVICE_CSV)
    run_fusion(inputs, **kw)  # measured
    ttnn.synchronize_device(device)
    ttnn.ReadDeviceProfiler(device)
    return _zone_durations_for_new_run(_DEVICE_CSV, offset, set(PHASE_ZONES[variant]))


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


# =============================================================================
# Regimes: where fusing through DEST helps, and where it does not (Wormhole B0 n150)
# =============================================================================
_REGIME_TRIALS = 11
_REGIME_ITERS = 100

# (scenario, num_tiles, block_size, lever method, verdict, measured speedup)
#   helps:          sfpu_chain n=4 blk=4 -- every intermediate feeds an SFPU op, so the L1 round trip is pure
#                   overhead; fused is ~1.1x at 4 tiles (the win shrinks to ~1.05x by 64 tiles).
#   does_not_help:  fpu_sfpu n=64 blk=1 dstreuse -- the consumer is an FPU multiply, so "fusing" costs a
#                   DEST->source transfer per tile that exceeds the pack+unpack it saves (~0.93x, a loss).
#   fpu_chain:      (a + b) * c, two FPU ops. DEST reuse saves the second pass's fixed cost per call (pipeline
#                   drain + re-init) but pays a DEST->srcA move and a DEST face clear per face of every tile, so it
#                   wins only at 1-2 tiles per call and loses from 4 tiles per call up (crossover ~2-3 tiles per call).
# Thresholds hold on every measured board (WH B0 n150, BH p150b); `measure` regimes flip between boards and only log
# -- numbers per board in README.md. The fpu_chain rows were measured on BH p150b only.
_REGIMES = [
    pytest.param("sfpu_chain", 4, 4, "fused", "helps", 1.11, id="helps_sfpu_chain_n4_blk4"),  # WH 1.12x, BH 1.11x
    pytest.param(
        "fpu_sfpu", 64, 1, "dstreuse", "loses", 0.93, id="loses_fpu_consumer_dstreuse_n64"
    ),  # WH 0.92-0.94x, BH 0.93x
    pytest.param("fpu_chain", 1, 1, "dstreuse", "helps", 1.11, id="helps_fpu_chain_dstreuse_n1"),  # BH 1.11x
    pytest.param("fpu_chain", 4, 4, "dstreuse", "loses", 0.96, id="loses_fpu_chain_dstreuse_n4_blk4"),  # BH 0.96x
    pytest.param("fpu_chain", 64, 4, "dstreuse", "loses", 0.72, id="loses_fpu_chain_dstreuse_n64_blk4"),  # BH 0.72x
]


@pytest.mark.parametrize("scenario,num_tiles,block_size,lever,verdict,measured", _REGIMES)
def test_compute_fusion_regimes(device, scenario, num_tiles, block_size, lever, verdict, measured):
    inputs, expected, kind = _make_case(device, scenario, num_tiles)
    base_variant, lever_variant = f"{scenario}.unfused", f"{scenario}.{lever}"
    for variant in (base_variant, lever_variant):  # correctness gate
        out = run_fusion(inputs, variant=variant, num_tiles=num_tiles, block_size=block_size, kernel_iters=1)
        _check(out, expected, kind, f"{variant} n={num_tiles} blk={block_size}")

    runners = {
        name: (
            lambda variant=variant: run_fusion(
                inputs, variant=variant, num_tiles=num_tiles, block_size=block_size, kernel_iters=_REGIME_ITERS
            )
        )
        for name, variant in (("base", base_variant), ("lever", lever_variant))
    }
    samples = _measure(device, runners, _REGIME_TRIALS, _REGIME_ITERS)  # interleaved A,B,A,B...
    base, lev = samples["base"], samples["lever"]
    b, l = statistics.median(base), statistics.median(lev)
    speedup = b / l
    arch = {"WORMHOLE_B0": "WH_B0", "BLACKHOLE": "BH"}.get(str(device.arch()).rsplit(".", 1)[-1], str(device.arch()))
    logger.info(
        f"[{arch}] {verdict} {scenario} n={num_tiles} tiles blk={block_size}: unfused {b:.0f} ns "
        f"[{min(base):.0f}..{max(base):.0f}] -> {lever} {l:.0f} ns [{min(lev):.0f}..{max(lev):.0f}] "
        f"= {speedup:.2f}x (trials={len(base)}, ns per expression evaluation)"
    )
    if verdict == "helps":
        assert speedup >= max(1.05, 1 + 0.5 * (measured - 1)), f"expected fusion to help, got {speedup:.2f}x"
    elif verdict == "loses":
        assert speedup <= 1.0, f"expected fusion to lose (<= 1.00x), got {speedup:.2f}x"
    else:
        assert speedup <= 1.05, f"expected no meaningful gain (<= 1.05x), got {speedup:.2f}x"
