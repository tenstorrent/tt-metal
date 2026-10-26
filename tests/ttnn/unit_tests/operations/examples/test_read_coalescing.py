# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Four-arm correctness and device timing for bank affinity × read coalescing."""

import os
import socket
import statistics
from collections import defaultdict

# Enable the on-device profiler IN-PROCESS (all three, before the device opens).
os.environ.setdefault("TT_METAL_DEVICE_PROFILER", "1")
os.environ.setdefault("TT_METAL_PROFILER_MID_RUN_DUMP", "1")
os.environ.setdefault("TT_METAL_PROFILER_CPP_POST_PROCESS", "1")
os.environ.setdefault("TT_METAL_LOGGER_LEVEL", "error")
# Raw timestamps allow a board with mismatched core timer bases to be measured explicitly.
os.environ.pop("TT_METAL_PROFILER_DISABLE_DUMP_TO_FILES", None)

from pathlib import Path

import pytest
import torch

import ttnn
from ttnn.operations.examples.read_coalescing import (
    read_coalescing,
    work_geometry,
    num_dram_banks,
    VARIANTS,
)

from loguru import logger

BLOCK_ROWS = 32

# HxWxSHARD_WIDTH cases, one width shard per DRAM bank. Auto sizes H to equal work per core.
CASES_SPEC = os.environ.get("RC_CASES", "auto")


def _parse_case(spec):
    h, w, chunk = (int(v) for v in spec.split("x"))
    return h, w, chunk, "ws"


def _cases(grid_cores, num_banks):
    if CASES_SPEC != "auto":
        return [_parse_case(c) for c in CASES_SPEC.split(",")]
    n = (grid_cores // num_banks) * num_banks
    return [(BLOCK_ROWS * rows * n, num_banks * chunk, chunk, "ws") for chunk in (64, 128, 512) for rows in (1, 2)]


VARIANT_SEL = os.environ.get("RC_VARIANT", "all")
KERNEL_ITERS = int(os.environ.get("RC_ITERS", "1"))
N_TRIALS = int(os.environ.get("RC_TRIALS", "5"))
N_LAUNCHES = int(os.environ.get("RC_LAUNCHES", "10"))
N_WARMUP = 3
REPORT_PATH = os.environ.get("RC_REPORT")


def _memory_config(device, h, w, src):
    nb = num_dram_banks(device)  # one shard per DRAM bank: shard s lives in bank s
    grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(nb - 1, 0))})
    spec = ttnn.ShardSpec(grid, (h, w // nb), ttnn.ShardOrientation.ROW_MAJOR)
    return ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.DRAM, spec)


def _make_input(device, h, w, src="ws"):
    torch.manual_seed(0)
    torch_input = torch.randn((h, w), dtype=torch.float32).to(torch.bfloat16)
    tt = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=_memory_config(device, h, w, src),
    )
    return torch_input, tt


_RAW_OFFSET = 0
_FREQ_MHZ = None
_MULTI_BASE_PROGRAMS = 0
_LAST_PROGRAMS = 0


def _read_kernel_ns(device):
    """Maximum within-clock-cluster span per launch; exact stock span on one-clock boards.

    Different timer bases cannot be subtracted. Keep every core, take the maximum
    cluster span, and report that limitation explicitly when multiple bases occur.
    The 1e8-cycle gap is far above these sub-ms kernels' dispatch skew.
    """
    global _RAW_OFFSET, _FREQ_MHZ, _MULTI_BASE_PROGRAMS, _LAST_PROGRAMS
    ttnn.ReadDeviceProfiler(device)
    root = Path(
        os.environ.get("TT_METAL_PROFILER_DIR", str(Path(os.environ.get("TT_METAL_HOME", ".")) / "generated/profiler"))
    )
    path = root / ".logs/profile_log_device.csv"
    assert path.exists(), "raw profiler output is required"
    if path.stat().st_size < _RAW_OFFSET:
        _RAW_OFFSET = 0
    with path.open() as f:
        f.seek(_RAW_OFFSET)
        lines = f.readlines()
        _RAW_OFFSET = f.tell()
    programs = defaultdict(lambda: defaultdict(dict))
    for line in lines:
        if "CHIP_FREQ[MHz]:" in line:
            _FREQ_MHZ = float(line.split("CHIP_FREQ[MHz]:")[1].split(",")[0])
        r = [v.strip() for v in line.split(",")]
        if len(r) < 12 or not r[10].endswith("-KERNEL") or r[11] not in ("ZONE_START", "ZONE_END"):
            continue
        programs[(r[0], r[7])][(r[1], r[2], r[3])][r[11]] = int(r[5])
    _LAST_PROGRAMS = len(programs)
    if not programs:
        return None
    assert _FREQ_MHZ and _FREQ_MHZ > 0
    total = 0
    for entries in programs.values():
        assert all("ZONE_START" in v and "ZONE_END" in v for v in entries.values()), "incomplete kernel zones"
        spans = sorted((v["ZONE_START"], v["ZONE_END"]) for v in entries.values())
        clusters = [[spans[0]]]
        for span in spans[1:]:
            if span[0] - clusters[-1][-1][0] > 100_000_000:
                clusters.append([])
            clusters[-1].append(span)
        _MULTI_BASE_PROGRAMS += len(clusters) > 1
        total += max(max(e for _, e in cluster) - min(b for b, _ in cluster) for cluster in clusters)
    return total * 1000 / _FREQ_MHZ


def _measure_ns(device, run_fn):
    """One trial = N_LAUNCHES launches, averaged."""
    ttnn.synchronize_device(device)
    _read_kernel_ns(device)  # flush
    for _ in range(N_LAUNCHES):
        run_fn()
    total = _read_kernel_ns(device)
    assert _LAST_PROGRAMS == N_LAUNCHES, f"expected {N_LAUNCHES} programs, got {_LAST_PROGRAMS}"
    return total / N_LAUNCHES if total is not None else None


@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("rows,chunk,iters", [(1, 32, 1), (3, 64, 1), (137, 128, 1), (33, 256, 1), (17, 512, 2)])
def test_read_coalescing_correctness(device, variant, rows, chunk, iters):
    h, w = rows * BLOCK_ROWS, num_dram_banks(device) * chunk
    torch_input, tt_input = _make_input(device, h, w)
    result = read_coalescing(tt_input, **VARIANTS[variant], shard_width=chunk, kernel_iters=iters)
    assert result.layout == ttnn.ROW_MAJOR_LAYOUT
    out = ttnn.to_torch(result)
    assert torch.equal(out, torch_input), f"{variant}: mismatch"


def test_read_coalescing_device_perf(device):
    """Device kernel duration per variant per case. Perf is reported, never asserted;
    each measured variant is also checked bit-exact once."""
    variants = VARIANTS if VARIANT_SEL == "all" else tuple(VARIANT_SEL.split(","))
    assert N_TRIALS > 0 and N_LAUNCHES > 0
    assert variants and all(v in VARIANTS for v in variants)
    grid = device.compute_with_storage_grid_size()
    grid_cores = grid.x * grid.y
    num_banks = num_dram_banks(device)
    results = {}
    for h, w, chunk, src in _cases(grid_cores, num_banks):
        torch_input, tt_input = _make_input(device, h, w, src)
        for v in variants:
            out = ttnn.to_torch(read_coalescing(tt_input, **VARIANTS[v], shard_width=chunk, kernel_iters=KERNEL_ITERS))
            assert torch.equal(out, torch_input), f"{v} {(h, w, chunk)}: mismatch"
        runs = {
            v: (lambda v=v: read_coalescing(tt_input, **VARIANTS[v], shard_width=chunk, kernel_iters=KERNEL_ITERS))
            for v in variants
        }
        for v in variants:
            for _ in range(N_WARMUP):
                runs[v]()
        samples = {v: [] for v in variants}
        for trial in range(N_TRIALS):  # rotate first arm to avoid fixed launch-order bias
            order = list(variants)
            order = order[trial % len(order) :] + order[: trial % len(order)]
            for v in order:
                ns = _measure_ns(device, runs[v])
                assert ns is not None, "profiler produced no data (profiler-enabled build?)"
                samples[v].append(ns)
        results[(h, w, chunk, src)] = samples

    arch = str(device.arch()).split(".")[-1]
    host = socket.gethostname()
    lines = [
        f"read_coalescing   box={host}  arch={arch}  grid={grid.x}x{grid.y}  dram_banks={num_banks}  "
        f"N={N_TRIALS} trials x {N_LAUNCHES} launches (median)  kernel-iters={KERNEL_ITERS}",
        f"  timing = max within-timer-cluster kernel span; multi-base programs observed={_MULTI_BASE_PROGRAMS}",
        "  op = copy bf16, ROW_MAJOR DRAM (ws = width-sharded, one shard per bank)"
        " -> ROW_MAJOR interleaved DRAM; placement = row-major grid; blk = blocks per core",
        "",
        f"  {'shape':>12} {'src':>3} {'chunk':>5} {'read B':>6} {'nt_h':>4} {'n_w':>4} {'cores':>5} {'blk':>3}  "
        f"{'variant':<7} {'ns':>9} {'spread':>7}  vs none",
    ]
    for (h, w, chunk, src), samples in results.items():
        nt_h, n_w, units, ncores = work_geometry((h, w), chunk, grid_cores)
        base = statistics.median(samples["none"]) if "none" in samples else None
        for v in variants:
            med = statistics.median(samples[v])
            spread = (max(samples[v]) - min(samples[v])) / med * 100
            ratio = f"{base / med:5.3f}x" if base and v != "none" else "  base" if v == "none" else ""
            lines.append(
                f"  {h:>5}x{w:<6} {src:>3} {chunk:>5} {chunk * 2:>6} {nt_h:>4} {n_w:>4} {ncores:>5} "
                f"{-(-units // ncores):>3}  "
                f"{v:<7} {med:>9.0f} {spread:>6.1f}%  {ratio}"
            )
        if "both" in samples:
            both_ns = statistics.median(samples["both"])
            if "affinity" in samples:
                lines.append(
                    f"    coalescing added to affinity: {statistics.median(samples['affinity']) / both_ns:.3f}x"
                )
            if "coalesce" in samples:
                lines.append(
                    f"    affinity added to coalescing: {statistics.median(samples['coalesce']) / both_ns:.3f}x"
                )
        lines.append("")
    text = "\n".join(lines)
    logger.info("\n" + text)
    if REPORT_PATH:
        Path(REPORT_PATH).write_text(text + "\n")
