# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the `bank_placement` example.

One core per DRAM bank copies an interleaved DRAM tensor. When each core's pages all
live in one bank, placing that core next to its bank shortens every transfer.
See ttnn/ttnn/operations/examples/bank_placement/README.md.

    scripts/run_safe_pytest.sh --run-all \\
        tests/ttnn/unit_tests/operations/examples/test_bank_placement.py::test_bank_placement_correctness
    scripts/run_safe_pytest.sh --run-all \\
        tests/ttnn/unit_tests/operations/examples/test_bank_placement.py::test_bank_placement_device_perf
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
from ttnn.operations.examples.bank_placement import bank_placement, placement_cores, VARIANTS, PATTERNS

from loguru import logger


# (rows, width) of a bf16 ROW_MAJOR tensor: one row == one DRAM page of width*2 bytes.
_DEFAULT_CASES = "3072x256,3072x1024,3072x4096"
CASES = [tuple(int(v) for v in c.split("x")) for c in os.environ.get("BP_CASES", _DEFAULT_CASES).split(",")]
PATTERN_SEL = os.environ.get("BP_PATTERN", "all")
VARIANT_SEL = os.environ.get("BP_VARIANT", "all")
BLOCK = int(os.environ.get("BP_BLOCK", "8"))
KERNEL_ITERS = int(os.environ.get("BP_ITERS", "1"))
N_TRIALS = int(os.environ.get("BP_TRIALS", "5"))
N_LAUNCHES = int(os.environ.get("BP_LAUNCHES", "10"))
N_WARMUP = 3
REPORT_PATH = os.environ.get("BP_REPORT")

_DURATION_KEY = "DEVICE KERNEL DURATION [ns]"


def _make_input(device, h, w):
    torch.manual_seed(0)
    torch_input = torch.randn((h, w), dtype=torch.float32).to(torch.bfloat16)
    tt = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
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
    ttnn.synchronize_device(device)
    _read_kernel_ns(device)  # flush
    for _ in range(N_LAUNCHES):
        run_fn()
    total = _read_kernel_ns(device)
    return total / N_LAUNCHES if total is not None else None


_CORRECTNESS_CASES = [(v, p) for v in VARIANTS for p in PATTERNS]


@pytest.mark.parametrize("variant,pattern", _CORRECTNESS_CASES)
def test_bank_placement_correctness(device, variant, pattern):
    """Every placement x pattern copies exactly: out == input."""
    torch_input, tt_input = _make_input(device, 1200, 512)
    out = ttnn.to_torch(bank_placement(tt_input, variant=variant, pattern=pattern, block=BLOCK))
    assert torch.equal(out, torch_input), f"{variant}/{pattern}: mismatch"


def test_bank_placement_device_perf(device):
    """Device kernel duration per placement, pattern and page size. Reported, never asserted."""
    variants = VARIANTS if VARIANT_SEL == "all" else tuple(VARIANT_SEL.split(","))
    patterns = PATTERNS if PATTERN_SEL == "all" else tuple(PATTERN_SEL.split(","))
    results = {}
    for h, w in CASES:
        torch_input, tt_input = _make_input(device, h, w)
        for p in patterns:
            runs = {
                v: (
                    lambda v=v, p=p: bank_placement(
                        tt_input, variant=v, pattern=p, block=BLOCK, kernel_iters=KERNEL_ITERS
                    )
                )
                for v in variants
            }
            for v in variants:
                assert torch.equal(ttnn.to_torch(runs[v]()), torch_input), f"{v}/{p}: mismatch"
                for _ in range(N_WARMUP):
                    runs[v]()
            samples = {v: [] for v in variants}
            for _ in range(N_TRIALS):  # interleave variants per trial so drift hits all equally
                for v in variants:
                    ns = _measure_ns(device, runs[v])
                    assert ns is not None, "profiler produced no data (profiler-enabled build?)"
                    samples[v].append(ns)
            results[(h, w, p)] = samples

    grid = device.compute_with_storage_grid_size()
    arch = str(device.arch()).split(".")[-1]
    placements = {v: " ".join(f"{c.x},{c.y}" for c in placement_cores(device, v)) for v in variants}
    lines = [
        f"bank_placement   box={socket.gethostname()}  arch={arch}  grid={grid.x}x{grid.y}  "
        f"N={N_TRIALS} trials x {N_LAUNCHES} launches (median)  kernel-iters={KERNEL_ITERS}  block={BLOCK}",
        "  op = DRAM interleaved -> DRAM interleaved copy, one core per DRAM bank (bank id = list index)",
    ]
    for v in variants:
        lines.append(f"    {v:<14} cores: {placements[v]}")
    lines += [
        "",
        f"  {'pattern':<7} {'pages':>6} {'page B':>6} {'MB':>6}  {'placement':<14} {'ns':>9} {'GB/s':>6} {'spread':>7}  vs row_major",
    ]
    for (h, w, p), samples in results.items():
        base = statistics.median(samples["row_major"]) if "row_major" in samples else None
        mb = h * w * 2 / 1e6
        for v in variants:
            med = statistics.median(samples[v])
            spread = (max(samples[v]) - min(samples[v])) / med * 100
            ratio = "  base" if v == "row_major" else (f"{base / med:5.3f}x" if base else "")
            lines.append(
                f"  {p:<7} {h:>6} {w * 2:>6} {mb:>6.1f}  {v:<14} {med:>9.0f} {2 * mb * 1e6 / med:>6.1f} {spread:>6.1f}%  {ratio}"
            )
        lines.append("")
    text = "\n".join(lines)
    logger.info("\n" + text)
    if REPORT_PATH:
        Path(REPORT_PATH).write_text(text + "\n")
