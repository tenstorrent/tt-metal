# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the `reader_placement` data-movement example (where you place the readers).

See ttnn/ttnn/operations/examples/reader_placement/README.md for what this
illustrates and why. Tests:

    # every placement produces the identical, correct identity copy
    scripts/run_safe_pytest.sh --run-all \\
        tests/ttnn/unit_tests/operations/examples/test_reader_placement.py::test_reader_placement_correctness

    # measured device kernel duration, column vs row vs diagonal (in-process profiler)
    scripts/run_safe_pytest.sh --run-all \\
        tests/ttnn/unit_tests/operations/examples/test_reader_placement.py::test_reader_placement_device_perf
"""

import os

# Enable the on-device profiler IN-PROCESS (needs all three, set before the device
# opens). Scoped to this module (not a dir conftest) so it doesn't perturb other
# examples' measurement. setdefault -> respects an outer tracy run if present.
os.environ.setdefault("TT_METAL_DEVICE_PROFILER", "1")
os.environ.setdefault("TT_METAL_PROFILER_MID_RUN_DUMP", "1")
os.environ.setdefault("TT_METAL_PROFILER_CPP_POST_PROCESS", "1")

import pytest
import torch

import ttnn
from ttnn.operations.examples.reader_placement import reader_placement, PLACEMENTS

from loguru import logger

from tests.ttnn.utils_for_testing import assert_with_pcc


TILE = 32
# Defaults are overridable via env so the CLI (python -m ...reader_placement) can
# measure the caller's own shapes/params through the same in-process path.
SHAPE = tuple(int(x) for x in os.environ.get("RP_SHAPE", "1024,2048").split(","))  # 2048 tiles (4 MB bf16)
NUM_CORES_SWEEP = tuple(int(x) for x in os.environ.get("RP_CORES", "4,8").split(","))  # 8x8 grid -> diagonal <= 8
BLOCK = int(os.environ.get("RP_BLOCK", "16"))  # pages in flight per barrier (NoC pressure)
N_WARMUP = 5
N_PROFILE_ITERS = int(os.environ.get("RP_ITERS", "20"))

WORKLOAD_CASES = [(p, nc) for nc in NUM_CORES_SWEEP for p in PLACEMENTS]
WORKLOAD_IDS = [f"{p}_{nc}c" for (p, nc) in WORKLOAD_CASES]

_DURATION_KEY = "DEVICE KERNEL DURATION [ns]"


def _make_input(device):
    torch.manual_seed(0)
    torch_input = torch.rand(SHAPE, dtype=torch.float32)
    tt_input = ttnn.from_torch(
        torch_input, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    return tt_input


def _read_kernel_ns(device):
    """Sum of on-device kernel duration over programs dispatched since the last read.

    ReadDeviceProfiler finishes the queue before reading and *consumes* the window,
    so a flush-read then a work-read brackets exactly the ops run in between.
    """
    ttnn.ReadDeviceProfiler(device)
    per_chip = ttnn.get_latest_programs_perf_data()
    total, found = 0.0, False
    for programs in (per_chip or {}).values():
        for program in programs:
            results = getattr(program, "program_analyses_results", None) or {}
            entry = results.get(_DURATION_KEY)
            if entry is None:
                continue
            total += float(entry.duration)
            found = True
    return total if found else None


def _measure_ns(device, run_fn):
    """Median-free average ns/op: warm up, flush, run N, read, divide by N."""
    for _ in range(N_WARMUP):
        run_fn()
    ttnn.synchronize_device(device)
    _read_kernel_ns(device)  # flush the warmup window
    for _ in range(N_PROFILE_ITERS):
        run_fn()
    total_ns = _read_kernel_ns(device)
    return total_ns / N_PROFILE_ITERS if total_ns is not None else None


@pytest.mark.parametrize("placement,num_cores", WORKLOAD_CASES, ids=WORKLOAD_IDS)
def test_reader_placement_correctness(device, placement, num_cores):
    """Every placement is a pure identity copy: output must equal input."""
    tt_input = _make_input(device)
    expected = ttnn.to_torch(tt_input).to(torch.float32)

    out = ttnn.to_torch(reader_placement(tt_input, placement=placement, num_cores=num_cores, block=BLOCK)).to(
        torch.float32
    )
    assert list(out.shape) == list(expected.shape), f"{out.shape} != {expected.shape}"
    assert_with_pcc(expected, out, 0.9999)


def test_reader_placement_device_perf(device):
    """Measure device kernel duration for column vs row vs diagonal, in one session.

    Correctness lives in test_reader_placement_correctness; this test only measures
    and reports (perf is evidence, never a pass/fail — the only assertion here is
    that the profiler produced a number).
    """
    tt_input = _make_input(device)

    ns = {}
    for placement, num_cores in WORKLOAD_CASES:
        run_fn = lambda p=placement, nc=num_cores: reader_placement(tt_input, placement=p, num_cores=nc, block=BLOCK)
        value = _measure_ns(device, run_fn)
        assert (
            value is not None
        ), f"profiler produced no data for {placement}_{num_cores}c (is the build profiler-enabled?)"
        ns[(placement, num_cores)] = value

    # Front-and-center table: num_cores, placement, ns/op, ratio vs the column baseline.
    arch = os.environ.get("ARCH_NAME", "unknown")
    lines = [
        "",
        "=== reader_placement device perf (interleaved DRAM copy; placement is the only variable) ===",
        f"    shape={SHAPE}  pages={(SHAPE[0] // TILE) * (SHAPE[1] // TILE)}  arch={arch}  block={BLOCK}  iters={N_PROFILE_ITERS}",
        f"    {'cores':>5}  {'placement':<10}  {'ns/op':>12}  {'vs column':>10}",
    ]
    for num_cores in NUM_CORES_SWEEP:
        base = ns[("column", num_cores)]
        for placement in PLACEMENTS:
            v = ns[(placement, num_cores)]
            ratio = base / v if v else float("nan")
            tag = "  (baseline)" if placement == "column" else f"  -> {ratio:.2f}x"
            lines.append(f"    {num_cores:>5}  {placement:<10}  {v:>12.1f}{tag}")
    logger.info("\n".join(lines))
