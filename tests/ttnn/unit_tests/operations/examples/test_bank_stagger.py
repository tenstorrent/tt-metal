# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the `bank_stagger` (bank-de-clustered issue order) example.

When every core of a width-split grid reads the SAME interleaved DRAM pages in the
same order, each issue step lands on one bank. Rotating each core's issue order
(reads and/or writes) spreads every step over the banks.
See ttnn/ttnn/operations/examples/bank_stagger/README.md.

    # every variant produces the exact tilized input
    scripts/run_safe_pytest.sh --run-all \\
        tests/ttnn/unit_tests/operations/examples/test_bank_stagger.py::test_bank_stagger_correctness

    # device kernel duration, none / read / write / both, across shapes
    scripts/run_safe_pytest.sh --run-all \\
        tests/ttnn/unit_tests/operations/examples/test_bank_stagger.py::test_bank_stagger_device_perf
"""

import os
import socket
import statistics

# Enable the on-device profiler IN-PROCESS (all three, before the device opens).
os.environ.setdefault("TT_METAL_DEVICE_PROFILER", "1")
os.environ.setdefault("TT_METAL_PROFILER_MID_RUN_DUMP", "1")
os.environ.setdefault("TT_METAL_PROFILER_CPP_POST_PROCESS", "1")
os.environ.setdefault("TT_METAL_LOGGER_LEVEL", "error")

from pathlib import Path

import pytest
import torch

import ttnn
from ttnn.operations.examples.bank_stagger import (
    bank_stagger,
    work_geometry,
    num_dram_banks,
    VARIANTS,
)

from loguru import logger

TILE = 32

# Cases are (H, W, chunk_wt, src) with src "il" (DRAM interleaved) or "ws" (DRAM width-sharded,
# one shard per bank); on the CLI "HxWxCHUNK" or "HxWxCHUNKws". "auto" (default) sizes them to
# the grid: width and height blocking with one block per core (interleaved), and a width-sharded
# source with one tile-row -- so one block per shard -- per core.
CASES_SPEC = os.environ.get("BS_CASES", "auto")


def _parse_case(spec):
    src = "ws" if spec.endswith("ws") else "il"
    h, w, chunk = (int(v) for v in spec.removesuffix("ws").split("x"))
    return h, w, chunk, src


def _cases(grid_cores, num_banks):
    if CASES_SPEC != "auto":
        return [_parse_case(c) for c in CASES_SPEC.split(",")]
    n = grid_cores
    return [
        (TILE, TILE * 2 * n, 2, "il"),
        (TILE, TILE * 16 * n, 16, "il"),
        (TILE * n, 512, 16, "il"),
        (TILE * n, num_banks * TILE * 4, 4, "ws"),
        (TILE * n, num_banks * TILE * 16, 16, "ws"),
    ]


VARIANT_SEL = os.environ.get("BS_VARIANT", "all")
KERNEL_ITERS = int(os.environ.get("BS_ITERS", "1"))
N_TRIALS = int(os.environ.get("BS_TRIALS", "5"))
N_LAUNCHES = int(os.environ.get("BS_LAUNCHES", "10"))
N_WARMUP = 3
REPORT_PATH = os.environ.get("BS_REPORT")

_DURATION_KEY = "DEVICE KERNEL DURATION [ns]"


def _memory_config(device, h, w, src):
    if src == "il":
        return ttnn.DRAM_MEMORY_CONFIG
    nb = num_dram_banks(device)  # one shard per DRAM bank: shard s lives in bank s
    grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(nb - 1, 0))})
    spec = ttnn.ShardSpec(grid, (h, w // nb), ttnn.ShardOrientation.ROW_MAJOR)
    return ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.DRAM, spec)


def _make_input(device, h, w, src="il"):
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


def _read_kernel_ns(device):
    ttnn.ReadDeviceProfiler(device)
    per_chip = ttnn.get_latest_programs_perf_data()
    total, found = 0.0, False
    for programs in (per_chip or {}).values():
        for program in programs:
            entry = (getattr(program, "program_analyses_results", None) or {}).get(_DURATION_KEY)
            if entry is None:
                continue
            total += float(entry.duration)
            found = True
    return total if found else None


def _measure_ns(device, run_fn):
    """One trial = N_LAUNCHES launches, averaged."""
    ttnn.synchronize_device(device)
    _read_kernel_ns(device)  # flush
    for _ in range(N_LAUNCHES):
        run_fn()
    total = _read_kernel_ns(device)
    return total / N_LAUNCHES if total is not None else None


_CORRECTNESS_CASES = [
    (v, c)
    for v in VARIANTS
    for c in (
        (32, 8192, 4, "il"),
        (64, 16384, 16, "il"),
        (2048, 512, 16, "il"),
        (96, 1024, 1, "il"),
        (256, 4096, 16, "ws"),
        (96, 1024, 2, "ws"),  # 2 chunks per shard, unit counts not a multiple of the shard count
    )
]


@pytest.mark.parametrize(
    "variant,case", _CORRECTNESS_CASES, ids=lambda x: "x".join(map(str, x[:3])) + x[3] if isinstance(x, tuple) else x
)
def test_bank_stagger_correctness(device, variant, case):
    """Every variant tilizes exactly: out == input (bf16, bit-exact)."""
    h, w, chunk, src = case
    torch_input, tt_input = _make_input(device, h, w, src)
    out = ttnn.to_torch(bank_stagger(tt_input, **VARIANTS[variant], chunk_wt=chunk, kernel_iters=KERNEL_ITERS))
    assert list(out.shape) == [h, w]
    assert torch.equal(out, torch_input), f"{variant} {case}: mismatch"


def test_bank_stagger_device_perf(device):
    """Device kernel duration per variant per case. Perf is reported, never asserted;
    each measured variant is also checked bit-exact once."""
    variants = VARIANTS if VARIANT_SEL == "all" else tuple(VARIANT_SEL.split(","))
    grid = device.compute_with_storage_grid_size()
    grid_cores = grid.x * grid.y
    num_banks = num_dram_banks(device)
    results = {}
    for h, w, chunk, src in _cases(grid_cores, num_banks):
        torch_input, tt_input = _make_input(device, h, w, src)
        for v in variants:
            out = ttnn.to_torch(bank_stagger(tt_input, **VARIANTS[v], chunk_wt=chunk, kernel_iters=KERNEL_ITERS))
            assert torch.equal(out, torch_input), f"{v} {(h, w, chunk)}: mismatch"
        runs = {
            v: (lambda v=v: bank_stagger(tt_input, **VARIANTS[v], chunk_wt=chunk, kernel_iters=KERNEL_ITERS))
            for v in variants
        }
        for v in variants:
            for _ in range(N_WARMUP):
                runs[v]()
        samples = {v: [] for v in variants}
        for _ in range(N_TRIALS):  # interleave variants per trial so drift hits all equally
            for v in variants:
                ns = _measure_ns(device, runs[v])
                assert ns is not None, "profiler produced no data (profiler-enabled build?)"
                samples[v].append(ns)
        results[(h, w, chunk, src)] = samples

    arch = str(device.arch()).split(".")[-1]
    host = socket.gethostname()
    lines = [
        f"bank_stagger   box={host}  arch={arch}  grid={grid.x}x{grid.y}  dram_banks={num_banks}  "
        f"N={N_TRIALS} trials x {N_LAUNCHES} launches (median)  kernel-iters={KERNEL_ITERS}",
        "  op = tilize bf16, ROW_MAJOR DRAM (il = interleaved, ws = width-sharded, one shard per bank)"
        " -> TILE interleaved DRAM; placement = row-major grid; blk = blocks per core",
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
                f"  {h:>5}x{w:<6} {src:>3} {chunk:>5} {chunk * 64:>6} {nt_h:>4} {n_w:>4} {ncores:>5} "
                f"{-(-units // ncores):>3}  "
                f"{v:<7} {med:>9.0f} {spread:>6.1f}%  {ratio}"
            )
        lines.append("")
    text = "\n".join(lines)
    logger.info("\n" + text)
    if REPORT_PATH:
        Path(REPORT_PATH).write_text(text + "\n")
