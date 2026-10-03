# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""The streaming profiler's clock sync gates.

test_streaming_profiler_sync_check is the accuracy gate. Each workload runs as a subprocess with the streaming profiler
and its sync check on (TT_METAL_STREAMING_PROFILER_SYNC_CHECK=1), which puts a ruler on one more idle eth core per chip
and logs, at each capture's end, the chip-to-chip error of the global timeline measured against held-out link rounds.
The test asserts that every chip pair was measured, that the error's p50, p99.9 and max (every reading counted), the link
sync's precision and parallel links' agreement are within the bounds below, that the log has no streaming profiler
warnings other than the per-core stall counts, and, for the di/dt workload, that AICLK swung enough
to exercise DVFS. The workloads: an idle mesh; host round trips that bound where the host timeline puts a device zone;
the di/dt workload, the FF1 matmul and SDPA di/dt ops on 2D fabric with a multicast core-to-core check and a fabric
ping-pong check before, between and after them; and two ttnn CCL ops. Needs a multi-chip Blackhole system (an 8-chip
LoudBox in CI); about 4 minutes.

test_streaming_profiler_fabric_overhead bounds the profiler's cost to the fabric. It runs test_tt_fabric's unicast
microbench (latency and bandwidth over linear, mesh and ring routes) with the profiler off as the baseline, then on,
and bounds the latency and bandwidth the profiler adds per test. test_streaming_profiler_fabric_sync_check runs the
microbench with the sync check on and gates only the sync's accuracy. test_streaming_profiler_fabric_eth_zones turns on
Ethernet-core zones (TT_METAL_STREAMING_PROFILER_ETH=1) and checks that the linear, ring and mesh routers still build and
run and that eth zones reach the zone CSV. All three need four or more Blackhole chips; about 2 minutes warm, 5 cold.

Run as a script, this file is the sync check's CCL workload (ccl_workload).
"""

from __future__ import annotations

import argparse
import csv
import functools
import os
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path

import pytest
import yaml

from tools.tracy.common import TT_METAL_HOME

# ~1.25x and ~1.4x the worst an 8-chip LoudBox has shown over three runs of the suite: p99.9 2.78 ns, max 4.38 ns with
# every reading counted.
P999_BOUND_NS = 3.5
MAX_BOUND_NS = 6.0
# The typical error: p50 ran 0.41-0.72 ns over 163 sync-check captures (09-29 and 09-30), so 1.0 ns is ~40% above it.
P50_BOUND_NS = 1.0
# Link sync precision over 22 captures (sync check, fabric overhead, torus): the worst link's round scatter ran
# 0.51-0.63 ns and the worst window's fit at its centre 0.13-0.17 ns, so the bounds sit ~25% and ~50% above them.
ROUND_SCATTER_BOUND_NS = 0.8
WINDOW_FIT_BOUND_NS = 0.25
# The worst pair of parallel links put their chips' offset 1.26-2.68 ns apart over 31 captures across 8 link-ups (mean
# 1.84, sd 0.41); a link-up redraws it, so 4 ns is the one-sided 99.9% prediction bound with the 8 link-ups as samples.
PARALLEL_LINKS_BOUND_NS = 4.0

HEADLINE = re.compile(
    r"sync check: chip-to-chip error of the global timeline, a bound, over (\d+) of \d+ chip pairs and \d+ samples: "
    r"\|err\| p50 ([0-9.]+), p99 [0-9.]+, p99\.9 ([0-9.]+), max ([0-9.]+) ns \(([^)]*)\)"
)
CHIP_LINE = re.compile(r"sync check chip (\d+): \d+ readings against the reference, .*?max ([0-9.]+) ns")
AICLK_LINE = re.compile(
    r"sync check chip (\d+) AICLK: (mean \d+ MHz, sd \d+, (\d+)-(\d+) MHz.*?)(?: \(\w+\.cpp:\d+\))?$", re.MULTILINE
)
PRECISION = re.compile(
    r"link sync precision over (\d+) of (\d+) links: round scatter ([0-9.]+) ns \(links' mean ([0-9.]+) ns\), "
    r"fit at the window centre ([0-9.]+) ns \((chip \d+ -> chip \d+)\), worst window ([0-9.]+) ns"
)
PARALLEL = re.compile(r"parallel link agreement over \d+ link pairs: rms [0-9.]+ ns, worst ([0-9.]+) ns \([^)]*\)")
FORBIDDEN = [
    "overflows region",  # tt_elffile.cpp
    "TT_FATAL",  # tt_stl/assert.hpp
]
# tt-logger's plain line: "<date> <time> | <level, padded to 8> | <logger name> | <message> (<file>:<line>)".
PROFILER_WARNING = re.compile(r"\S+ \S+ \| warning  \| [^|]*\| \[streaming profiler\] ")
# DevicePrograms::verify_completeness's per-core stall counts, which a busy workload may log.
ALLOWED_WARNING = re.compile(r" on \d+ of \d+ cores; \(virt x,y\)#index=count: ")
PROBE_TIMEOUT_S = 300
# DevicePrograms::boot logs it once per capture; test_tt_fabric opens a new capture for each fabric config it runs.
CAPTURE_START = "[streaming profiler] active on"
# CI's logger colors every message, which puts an escape code right before a line's first word.
ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;]*m")


def run(args: list[str | Path], env_extra: dict[str, str], timeout: float) -> str:
    """Run `args` with `env_extra` over the caller's environment minus its TT_METAL_STREAMING_PROFILER* variables, and
    return stdout then stderr. Skips the test if the profiler declined this system, and otherwise fails it on a timeout,
    a nonzero exit, or if the profiler was asked for but never started."""
    env = {name: value for name, value in os.environ.items() if not name.startswith("TT_METAL_STREAMING_PROFILER")}
    env.update(env_extra)
    args = [str(arg) for arg in args]
    try:
        proc = subprocess.run(args, env=env, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired as timeout_error:
        # On POSIX the output captured before the timeout comes back as bytes even with text=True.
        out = "".join(
            stream.decode(errors="replace") if isinstance(stream, bytes) else stream or ""
            for stream in (timeout_error.stdout, timeout_error.stderr)
        )
        out = ANSI_ESCAPE.sub("", out)
        pytest.fail(f"{args[0]} timed out after {timeout} s:\n{out[-4000:]}")
    out = ANSI_ESCAPE.sub("", proc.stdout + proc.stderr)
    if "[streaming profiler] not capturing" in out:
        pytest.skip("the streaming profiler does not capture here (not Blackhole / no DRAM programmable cores)")
    assert proc.returncode == 0, f"{args[0]} exited {proc.returncode}:\n{out[-4000:]}"
    if "TT_METAL_STREAMING_PROFILER" in env_extra:
        assert CAPTURE_START in out, f"the streaming profiler did not start:\n{out[-4000:]}"
    return out


@functools.cache
def system() -> tuple[str, int]:
    """Read in a subprocess so the pytest parent never takes the PCIe lock."""
    code = "import ttnn; print('SYSTEM', ttnn.get_arch_name(), ttnn.get_num_devices())"
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=PROBE_TIMEOUT_S)
    match = re.search(r"^SYSTEM (\S+) (\d+)$", proc.stdout, re.MULTILINE)
    assert match, f"no system probe output:\n{proc.stdout[-2000:]}\n{proc.stderr[-2000:]}"
    return match.group(1).lower(), int(match.group(2))


def skip_unless_blackhole(min_chips: int) -> None:
    arch, chips = system()
    if arch != "blackhole":
        pytest.skip(f"the streaming profiler runs on Blackhole, not {arch}")
    if chips < min_chips:
        pytest.skip(f"needs at least {min_chips} chips, this system has {chips}")


def aiclk_swing_mhz(out: str) -> int:
    """The widest range, in MHz, that any chip's AICLK covered in any of the run's captures."""
    return max((int(hi) - int(lo) for _, _, lo, hi in AICLK_LINE.findall(out)), default=0)


def check_link_precision(out: str) -> None:
    """Assert every capture in `out` measured all its links, within the round scatter and window fit bounds."""
    reports = list(PRECISION.finditer(out))
    captures = out.count(CAPTURE_START)
    assert len(reports) == captures, f"{len(reports)} link precision reports for {captures} captures:\n{out[-4000:]}"
    for report in reports:
        print(f"\n[link-precision] {report.group(0)}")
        measured, links = int(report.group(1)), int(report.group(2))
        assert measured == links, f"precision measured on {measured} of {links} links"
        scatter, window = float(report.group(3)), float(report.group(7))
        assert scatter <= ROUND_SCATTER_BOUND_NS, f"round scatter {scatter} ns > {ROUND_SCATTER_BOUND_NS} ns"
        assert window <= WINDOW_FIT_BOUND_NS, f"worst window fit {window} ns > {WINDOW_FIT_BOUND_NS} ns"


def check_parallel_links(out: str) -> None:
    """Assert some capture in `out` reports parallel link agreement, and every report is within the bound."""
    reports = list(PARALLEL.finditer(out))
    assert reports, f"no parallel link agreement report:\n{out[-4000:]}"
    for report in reports:
        print(f"\n[parallel-links] {report.group(0)}")
        worst = float(report.group(1))
        assert worst <= PARALLEL_LINKS_BOUND_NS, f"parallel links {worst} ns apart > {PARALLEL_LINKS_BOUND_NS} ns"


def check_sync_accuracy(out: str) -> None:
    """Assert every capture in `out` has a sync check report within the error bounds, then the link precision, parallel
    link and clean-log checks."""
    headlines = list(HEADLINE.finditer(out))
    captures = out.count(CAPTURE_START)
    assert len(headlines) == captures, f"{len(headlines)} sync check reports for {captures} captures:\n{out[-4000:]}"
    for k, headline in enumerate(headlines):
        block = out[headline.start() : headlines[k + 1].start() if k + 1 < len(headlines) else len(out)]
        chips = {int(chip): float(max_ns) for chip, max_ns in CHIP_LINE.findall(block)}
        assert len(chips) == system()[1], f"the report covers chips {sorted(chips)} of {system()[1]}"
        measured, p50, p999, worst, where = headline.groups()
        print(f"\n[sync-check] {headline.group(0)}")
        per_chip = " ".join(f"c{chip} {max_ns:.2f}" for chip, max_ns in sorted(chips.items()))
        print(f"[sync-check] per chip max ns: {per_chip}")
        for chip, clock, _, _ in AICLK_LINE.findall(block):
            print(f"[sync-check] chip {chip} AICLK: {clock}")
        num_chips = len(chips)
        all_pairs = num_chips * (num_chips - 1) // 2
        assert int(measured) == all_pairs, f"{measured} chip pairs measured for {num_chips} chips"
        assert float(p50) <= P50_BOUND_NS, f"p50 {p50} ns > {P50_BOUND_NS} ns ({headline.group(0)})"
        assert float(p999) <= P999_BOUND_NS, f"p99.9 {p999} ns > {P999_BOUND_NS} ns ({headline.group(0)})"
        assert float(worst) <= MAX_BOUND_NS, f"max {worst} ns > {MAX_BOUND_NS} ns ({where})"
    check_link_precision(out)
    check_parallel_links(out)
    check_log_clean(out)


def check_log_clean(out: str) -> None:
    """Fail if `out` has a streaming profiler warning other than the per-core stall counts, or a line
    logged only when a kernel or the host went wrong."""
    warnings = [line for line in out.splitlines() if PROFILER_WARNING.match(line) and not ALLOWED_WARNING.search(line)]
    assert not warnings, "streaming profiler warnings in the log:\n" + "\n".join(warnings)[:4000]
    for bad in FORBIDDEN:
        offending = [line for line in out.splitlines() if bad in line]
        assert not offending, f"'{bad}' in the log:\n" + "\n".join(offending)[:4000]


SYNC_WORKLOADS = Path(TT_METAL_HOME) / "build/test/ttnn/tracy/test_streaming_profiler_sync_workloads"
CCL = [Path(sys.executable), Path(__file__)]
SECONDS = 20
WORKLOADS = {
    "idle": [SYNC_WORKLOADS, "idle", "--seconds", str(SECONDS)],
    "host_sync": [SYNC_WORKLOADS, "host_sync"],
    "didt": [SYNC_WORKLOADS, "didt"],
    "ccl_all_gather_ring": [*CCL, "--op", "all_gather", "--fabric", "ring", "--seconds", str(SECONDS)],
    "ccl_all_reduce_2d_load": [*CCL, "--op", "all_reduce", "--fabric", "2d", "--load", "4", "--seconds", str(SECONDS)],
}
SYNC_TIMEOUT_S = 300
# The di/dt ops are here for the DVFS their throttling causes: over the run some chip's AICLK swings over 100 MHz,
# against the 6 MHz an idle chip jitters by.
DVFS_WORKLOADS = {"didt"}
MIN_DVFS_SWING_MHZ = 100
SYNC_CHECK_ENV = {"TT_METAL_STREAMING_PROFILER": "1", "TT_METAL_STREAMING_PROFILER_SYNC_CHECK": "1"}


@pytest.mark.timeout(PROBE_TIMEOUT_S + SYNC_TIMEOUT_S + 60)
@pytest.mark.parametrize("workload", list(WORKLOADS))
def test_streaming_profiler_sync_check(workload):
    skip_unless_blackhole(min_chips=2)
    out = run(WORKLOADS[workload], SYNC_CHECK_ENV, SYNC_TIMEOUT_S)
    check_sync_accuracy(out)
    if workload in DVFS_WORKLOADS:
        swing = aiclk_swing_mhz(out)
        assert swing >= MIN_DVFS_SWING_MHZ, f"AICLK moved at most {swing} MHz on any chip, so no DVFS was exercised"


FABRIC_BIN = Path(TT_METAL_HOME) / "build/test/tt_metal/tt_fabric/test_infra/test_tt_fabric"
CONFIG = (
    Path(TT_METAL_HOME) / "tests/tt_metal/tt_fabric/test_infra/test_yamls/test_fabric_ubench_at_least_2x2_mesh.yaml"
)
FABRIC_TIMEOUT_S = 180
# test_tt_fabric keeps a latency test's samples in L1 and holds at most 1024 (MAX_LATENCY_SAMPLES), so the gate runs
# each latency test LATENCY_REPEATS times in one process and pools their samples from the per-sample CSV.
LATENCY_SAMPLES = 1024
LATENCY_REPEATS = 40
LATENCY_CSV = Path(TT_METAL_HOME) / "generated/fabric/latency_results_blackhole.csv"
# Latency added, in ns: the median over tests of p50 and p99, and the worst test's p99.9, about its 41st-slowest packet.
# Over 20 runs on an 8-chip LoudBox the profiler added, mean +- sd per run, -1.2 +- 0.8 ns to the median p50, +35 +- 5
# to the median p99 and +316 +- 7 to the worst test's p99.9; each bound is mean + 3.67 sd rounded up, the one-sided
# 99.9% prediction bound for one more run.
MEDIAN_P50_ADDED_BOUND_NS = 2.0
MEDIAN_P99_ADDED_BOUND_NS = 55.0
WORST_P999_ADDED_BOUND_NS = 345.0
# Over 20 runs on an 8-chip LoudBox the bandwidth ratio with the profiler on was, mean +- sd per run, 1.0241 +- 0.0004
# for the median test and 0.99769 +- 0.0001 for the worst. The profiler-on routers run ~2.4% faster because their link
# sync start and stop code is laid out cold, and a gate shouldn't depend on that speedup, so instead of the prediction
# bound the bounds deliberately allow a 0.4% loss on the median test and 1.1% on the worst.
MEDIAN_BANDWIDTH_RATIO_BOUND = 0.996
WORST_BANDWIDTH_RATIO_BOUND = 0.989

RUNNING = re.compile(r"Running Test: (\S+)")
BANDWIDTH = re.compile(r"BW \(GB/s\)=([0-9.]+)")
SAMPLE_SUFFIX = re.compile(r"_sample\d+$")


@pytest.fixture(scope="module")
def ubench_config(tmp_path_factory) -> Path:
    skip_unless_blackhole(min_chips=4)
    assert FABRIC_BIN.exists(), f"{FABRIC_BIN} not built (build with --build-tests)"
    config = yaml.safe_load(CONFIG.read_text())
    tests = []
    for test in config["Tests"]:
        if test.get("latency_test_mode"):
            test["senders"][0]["patterns"][0]["num_packets"] = LATENCY_SAMPLES
            tests += [test] * LATENCY_REPEATS
        else:
            tests.append(test)
    config["Tests"] = tests
    path = tmp_path_factory.mktemp("fabric") / "ubench.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False))
    return path


def run_fabric_ubench(config: Path, env_extra: dict, name_glob: str = "*Unicast*") -> tuple[dict, dict, str]:
    """Run the microbench and return each latency test's p50, p99 and p99.9 net latency in ns, each bandwidth test's
    mean bandwidth, and the log."""
    out = run([FABRIC_BIN, "--test_config", config, "--filter", f"name.{name_glob}"], env_extra, FABRIC_TIMEOUT_S)
    net_ns = {}
    with LATENCY_CSV.open() as latency_file:
        for row in csv.DictReader(latency_file):
            net_ns.setdefault(SAMPLE_SUFFIX.sub("", row["test_name"]), []).append(float(row["net_avg_ns"]))
    counts = {test: len(samples) for test, samples in net_ns.items()}
    assert all(count == LATENCY_SAMPLES * LATENCY_REPEATS for count in counts.values()), f"latency samples: {counts}"
    latency = {}
    for test, samples in net_ns.items():
        permille = statistics.quantiles(samples, n=1000)
        latency[test] = {"p50": permille[499], "p99": permille[989], "p99.9": permille[998]}
    bandwidth, running = {}, None
    for line in out.splitlines():
        if match := RUNNING.search(line):
            running = match.group(1)
        elif running and not running.startswith("Latency") and (match := BANDWIDTH.search(line)):
            bandwidth.setdefault(running, []).append(float(match.group(1)))
    bandwidth = {test: statistics.mean(samples) for test, samples in bandwidth.items()}
    assert latency or bandwidth, f"no latency or bandwidth results:\n{out[-4000:]}"
    return latency, bandwidth, out


def assert_same_tests(kind: str, results: dict, base: dict) -> None:
    assert set(results) == set(base), f"{kind} tests differ: {sorted(set(results) ^ set(base))}"


@pytest.fixture(scope="module")
def baseline(ubench_config):
    return run_fabric_ubench(ubench_config, {})


@pytest.mark.timeout(PROBE_TIMEOUT_S + 2 * FABRIC_TIMEOUT_S + 60)
def test_streaming_profiler_fabric_overhead(ubench_config, baseline):
    base_latency, base_bandwidth, _ = baseline
    latency, bandwidth, out = run_fabric_ubench(ubench_config, {"TT_METAL_STREAMING_PROFILER": "1"})
    assert_same_tests("latency", latency, base_latency)
    assert_same_tests("bandwidth", bandwidth, base_bandwidth)
    added = {
        stat: {test: latency[test][stat] - base_latency[test][stat] for test in base_latency}
        for stat in ("p50", "p99", "p99.9")
    }
    ratio = {test: bandwidth[test] / base_bandwidth[test] for test in base_bandwidth}
    table = "\n".join(
        f"  {test}: " + ", ".join(f"{stat} {added[stat][test]:+.0f} ns" for stat in added)
        for test in sorted(base_latency)
    )
    worst = "\n".join(
        f"  {test}: {ratio[test]:.3f}" for test in sorted(ratio, key=lambda test: (ratio[test], test))[:8]
    )
    median_p50, median_p99 = statistics.median(added["p50"].values()), statistics.median(added["p99"].values())
    worst_p999 = max(added["p99.9"].values())
    median_ratio, worst_ratio = statistics.median(ratio.values()), min(ratio.values())
    print(
        f"\n[fabric-overhead] latency added: median p50 {median_p50:+.1f} ns, median p99 {median_p99:+.1f} ns, "
        f"worst p99.9 {worst_p999:+.1f} ns; per test:\n{table}"
    )
    print(
        f"[fabric-overhead] bandwidth ratio: median {median_ratio:.4f}, worst {worst_ratio:.4f}; worst tests:\n{worst}"
    )
    assert median_p50 <= MEDIAN_P50_ADDED_BOUND_NS, f"median p50 added > {MEDIAN_P50_ADDED_BOUND_NS} ns:\n{table}"
    assert median_p99 <= MEDIAN_P99_ADDED_BOUND_NS, f"median p99 added > {MEDIAN_P99_ADDED_BOUND_NS} ns:\n{table}"
    assert worst_p999 <= WORST_P999_ADDED_BOUND_NS, f"a test's p99.9 added > {WORST_P999_ADDED_BOUND_NS} ns:\n{table}"
    assert (
        median_ratio >= MEDIAN_BANDWIDTH_RATIO_BOUND
    ), f"median bandwidth ratio < {MEDIAN_BANDWIDTH_RATIO_BOUND}:\n{worst}"
    assert (
        worst_ratio >= WORST_BANDWIDTH_RATIO_BOUND
    ), f"a test's bandwidth ratio < {WORST_BANDWIDTH_RATIO_BOUND}:\n{worst}"
    check_link_precision(out)
    check_log_clean(out)


@pytest.mark.timeout(PROBE_TIMEOUT_S + FABRIC_TIMEOUT_S + 60)
def test_streaming_profiler_fabric_sync_check(ubench_config):
    _, _, out = run_fabric_ubench(ubench_config, SYNC_CHECK_ENV)
    check_sync_accuracy(out)


@pytest.mark.timeout(PROBE_TIMEOUT_S + 2 * FABRIC_TIMEOUT_S + 60)
def test_streaming_profiler_fabric_eth_zones(ubench_config, baseline, tmp_path):
    """With Ethernet-core zones on, the latency tests' routers on linear, ring and mesh fabric each fit their kernel
    config buffer and run, Ethernet zones reach the zone CSV, and the log is clean. No other gate turns eth zones on."""
    base_latency, _, _ = baseline
    zone_csv = tmp_path / "zones.csv"
    latency, _, out = run_fabric_ubench(
        ubench_config,
        {
            "TT_METAL_STREAMING_PROFILER": "1",
            "TT_METAL_STREAMING_PROFILER_ETH": "1",
            "TT_METAL_STREAMING_PROFILER_ZONE_CSV": str(zone_csv),
        },
        name_glob="Latency*Unicast",
    )
    assert_same_tests("latency", latency, base_latency)
    with zone_csv.open() as zone_file:
        zone_file.readline()
        riscs = {row["RISC processor type"] for row in csv.DictReader(zone_file, skipinitialspace=True)}
    assert "ERISC" in riscs, f"no ERISC zone in the zone CSV, only {sorted(riscs)}"
    check_log_clean(out)


CCL_OPS_PER_SYNC = 20


def ccl_workload() -> None:
    """A ttnn CCL op in a loop on every chip of the box for a fixed time, optionally with matmuls between the ops."""
    import torch
    import ttnn

    fabrics = {
        "ring": (ttnn.FabricConfig.FABRIC_1D_RING, ttnn.Topology.Ring),
        "2d": (ttnn.FabricConfig.FABRIC_2D, ttnn.Topology.Linear),
    }
    parser = argparse.ArgumentParser(description=ccl_workload.__doc__)
    parser.add_argument("--op", choices=["all_gather", "all_reduce"], required=True)
    parser.add_argument("--fabric", choices=list(fabrics), required=True)
    parser.add_argument("--seconds", type=float, default=20.0)
    parser.add_argument("--load", type=int, default=0, help="matmuls before each CCL op")
    args = parser.parse_args()

    fabric, topology = fabrics[args.fabric]
    rows, cols = sorted(tuple(ttnn._ttnn.multi_device.SystemMeshDescriptor().shape()), reverse=True)
    ttnn.set_fabric_config(fabric)
    mesh = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(rows, cols))

    def to_mesh(host_tensor: torch.Tensor, mapper) -> ttnn.Tensor:
        return ttnn.from_torch(
            host_tensor,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
            mesh_mapper=mapper,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    ccl_input = to_mesh(
        torch.randn([rows, cols, 512, 2048]).bfloat16(),
        ttnn.ShardTensor2dMesh(mesh, dims=(0, 1), mesh_shape=(rows, cols)),
    )
    tensors = [ccl_input]
    if args.load:
        matmul_a = to_mesh(torch.randn([1, 1, 2048, 4096]).bfloat16(), ttnn.ReplicateTensorToMesh(mesh))
        matmul_b = to_mesh(torch.randn([1, 1, 4096, 4096]).bfloat16(), ttnn.ReplicateTensorToMesh(mesh))
        tensors += [matmul_a, matmul_b]

    ops = 0
    end = time.monotonic() + args.seconds
    while time.monotonic() < end:
        for _ in range(CCL_OPS_PER_SYNC):
            for _ in range(args.load):
                ttnn.matmul(matmul_a, matmul_b)
            if args.op == "all_gather":
                ttnn.all_gather(ccl_input, dim=3, cluster_axis=0, topology=topology)
            else:
                ttnn.all_reduce(ccl_input, cluster_axis=0, topology=topology)
        ttnn.synchronize_device(mesh)
        ops += CCL_OPS_PER_SYNC
    print(f"{args.op} over {args.fabric} fabric, load {args.load}: {ops} ops in {args.seconds:.0f} s")
    for tensor in tensors:
        ttnn.deallocate(tensor)
    ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    ccl_workload()
