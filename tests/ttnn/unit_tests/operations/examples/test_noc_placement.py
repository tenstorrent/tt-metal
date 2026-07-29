# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the `noc_placement` data-movement example (placement × NoC × operation).

See ttnn/ttnn/operations/examples/noc_placement/README.md for what this illustrates.

    # every copy placement/NoC produces the identical, correct identity copy
    scripts/run_safe_pytest.sh --run-all \\
        tests/ttnn/unit_tests/operations/examples/test_noc_placement.py::test_noc_placement_correctness

    # measured device kernel duration for the 12-cell read/write × NoC × placement matrix
    scripts/run_safe_pytest.sh --run-all \\
        tests/ttnn/unit_tests/operations/examples/test_noc_placement.py::test_noc_placement_device_perf

    # one op launch per matrix cell (NoC-trace driver for tt-npe; used by noc_report.py)
    scripts/run_safe_pytest.sh --run-all \\
        tests/ttnn/unit_tests/operations/examples/test_noc_placement.py::test_noc_placement_matrix
"""

import json
import os

# Enable the on-device profiler IN-PROCESS (needs all three, set before the device opens).
os.environ.setdefault("TT_METAL_DEVICE_PROFILER", "1")
os.environ.setdefault("TT_METAL_PROFILER_MID_RUN_DUMP", "1")
os.environ.setdefault("TT_METAL_PROFILER_CPP_POST_PROCESS", "1")

import pytest
import torch

import ttnn
from ttnn.operations.examples.noc_placement import noc_placement, PLACEMENTS, NOCS

from loguru import logger

from tests.ttnn.utils_for_testing import assert_with_pcc


TILE = 32
# Overridable via env so the CLI (python -m ...noc_placement) measures the caller's params.
SHAPE = tuple(int(x) for x in os.environ.get("NP_SHAPE", "1024,2048").split(","))
NUM_CORES = int(os.environ.get("NP_CORES", "8"))
BLOCK = int(os.environ.get("NP_BLOCK", "16"))
KERNEL_ITERS = int(os.environ.get("NP_ITERS", "8"))  # in-kernel repeat for steady-state throughput
N_WARMUP = 5
N_PROFILE_ITERS = int(os.environ.get("NP_PROFILE_ITERS", "20"))
_NS_OUT = os.environ.get("NP_NS_OUT")  # if set, device_perf dumps {cell: ns} JSON here (for noc_report.py)

# The read/write matrix, in a FIXED order so the NoC-trace op-id sequence maps to cells.
BENCH_MATRIX = [(op, noc, pl) for noc in NOCS for op in ("read", "write") for pl in PLACEMENTS]
BENCH_IDS = [f"{op}_{noc}_{pl}" for (op, noc, pl) in BENCH_MATRIX]
# The identity copy, canonical pairing (reads NoC0 / writes NoC1), for the placement story.
COPY_MATRIX = [("copy", "noc0", pl) for pl in PLACEMENTS]

_DURATION_KEY = "DEVICE KERNEL DURATION [ns]"


def _make_input(device):
    torch.manual_seed(0)
    return ttnn.from_torch(
        torch.rand(SHAPE, dtype=torch.float32),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def _read_kernel_ns(device):
    """Sum of on-device kernel duration over programs since the last read (flush-consumes)."""
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
    """Average ns/op: warm up, flush, run N, read, divide by N."""
    for _ in range(N_WARMUP):
        run_fn()
    ttnn.synchronize_device(device)
    _read_kernel_ns(device)  # flush the warmup window
    for _ in range(N_PROFILE_ITERS):
        run_fn()
    total_ns = _read_kernel_ns(device)
    return total_ns / N_PROFILE_ITERS if total_ns is not None else None


@pytest.mark.parametrize("placement", PLACEMENTS)
@pytest.mark.parametrize("noc", NOCS)
def test_noc_placement_correctness(device, noc, placement):
    """op="copy" is a pure identity copy: output must equal input for every placement/NoC."""
    tt_input = _make_input(device)
    expected = ttnn.to_torch(tt_input).to(torch.float32)
    out = ttnn.to_torch(noc_placement(tt_input, op="copy", noc=noc, placement=placement, num_cores=NUM_CORES)).to(
        torch.float32
    )
    assert list(out.shape) == list(expected.shape), f"{out.shape} != {expected.shape}"
    assert_with_pcc(expected, out, 0.9999)


def test_noc_placement_device_perf(device):
    """Measure device kernel duration for the 12-cell read/write × NoC × placement matrix."""
    tt_input = _make_input(device)
    ns = {}
    for op, noc, pl in BENCH_MATRIX + COPY_MATRIX:
        run_fn = lambda o=op, n=noc, p=pl: noc_placement(
            tt_input, op=o, noc=n, placement=p, num_cores=NUM_CORES, kernel_iters=KERNEL_ITERS, block=BLOCK
        )
        value = _measure_ns(device, run_fn)
        assert value is not None, f"profiler produced no data for {op}_{noc}_{pl} (is the build profiler-enabled?)"
        ns[f"{op}_{noc}_{pl}"] = value

    arch = os.environ.get("ARCH_NAME", "unknown")
    lines = [
        "",
        "=== noc_placement device perf (isolated read/write + copy; NoC × placement) ===",
        f"    shape={SHAPE}  cores={NUM_CORES}  arch={arch}  block={BLOCK}  kernel_iters={KERNEL_ITERS}",
        f"    {'op':<6} {'noc':<5} {'placement':<10} {'ns/op':>12}",
    ]
    for op, noc, pl in BENCH_MATRIX + COPY_MATRIX:
        lines.append(f"    {op:<6} {noc:<5} {pl:<10} {ns[f'{op}_{noc}_{pl}']:>12.1f}")
    logger.info("\n".join(lines))

    if _NS_OUT:
        with open(_NS_OUT, "w") as f:
            json.dump(ns, f)
        logger.info(f"noc_placement: wrote device-ns matrix to {_NS_OUT}")


@pytest.mark.parametrize("op,noc,placement", BENCH_MATRIX, ids=BENCH_IDS)
def test_noc_placement_matrix(device, op, noc, placement):
    """One op launch per matrix cell (kernel_iters=1) -> a small, comparable NoC trace per cell.
    Run under tools/tracy/profile_this.py --collect-noc-traces; consumed by noc_report.py."""
    tt_input = _make_input(device)
    noc_placement(tt_input, op=op, noc=noc, placement=placement, num_cores=NUM_CORES, kernel_iters=1, block=BLOCK)
    ttnn.synchronize_device(device)
