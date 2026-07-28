# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the `constant_synthesis` data-movement example.

One kernel-level decision drives a constant-valued output: stream the constant
from a DRAM-resident tensor (stream_from_dram, baseline — real reads), or invent
one page on-core and replicate it (synthesize, candidate — zero reads). A second
measurement axis, the core count, exposes when the read side actually matters:
the win appears once the move is DRAM-bandwidth-bound.
See ttnn/ttnn/operations/examples/constant_synthesis/README.md.

    # every variant produces the identical, correct constant-valued output
    scripts/run_safe_pytest.sh --run-all \\
        tests/ttnn/unit_tests/operations/examples/test_constant_synthesis.py::test_constant_synthesis_correctness

    # device kernel duration, variant x cores (in-process profiler)
    scripts/run_safe_pytest.sh --run-all \\
        tests/ttnn/unit_tests/operations/examples/test_constant_synthesis.py::test_constant_synthesis_device_perf
"""

import os

# Enable the on-device profiler IN-PROCESS (needs all three, set before the device
# opens). Scoped to this module (not a dir conftest) so it doesn't perturb other
# examples' measurement. setdefault -> respects an outer tracy run if present.
os.environ.setdefault("TT_METAL_DEVICE_PROFILER", "1")
os.environ.setdefault("TT_METAL_PROFILER_MID_RUN_DUMP", "1")
os.environ.setdefault("TT_METAL_PROFILER_CPP_POST_PROCESS", "1")
os.environ.setdefault("TT_METAL_LOGGER_LEVEL", "error")  # silence loud C++ profiler histograms

import pytest
import torch

import ttnn
from ttnn.operations.examples.constant_synthesis import constant_synthesis, VARIANTS, ELEM_BYTES

from loguru import logger

from tests.ttnn.utils_for_testing import assert_with_pcc


# Defaults overridable via env so the CLI (python -m ...constant_synthesis) can
# measure the caller's own shape/params through the same in-process path.
SHAPE = tuple(int(x) for x in os.environ.get("CS_SHAPE", "4096,1024").split(","))  # (rows, W)
VALUE = float(os.environ.get("CS_VALUE", "1.0"))
ITERS = int(os.environ.get("CS_ITERS", "1"))
BLOCK = int(os.environ.get("CS_BLOCK", "8"))  # reads/writes in flight per barrier
N_WARMUP = 5
N_PROFILE_ITERS = int(os.environ.get("CS_TRIALS", "20"))

# Which variants the perf sweep runs (correctness always runs all).
_VARIANT_ENV = os.environ.get("CS_VARIANT", "all")
PERF_VARIANTS = VARIANTS if _VARIANT_ENV == "all" else tuple(v for v in VARIANTS if v == _VARIANT_ENV)

_DURATION_KEY = "DEVICE KERNEL DURATION [ns]"


def _make_source(device, rows, w):
    """DRAM-resident constant tensor [rows, w] bf16 == VALUE (the baseline's source)."""
    torch_src = torch.full((rows, w), VALUE, dtype=torch.bfloat16)
    tt_src = ttnn.from_torch(
        torch_src,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    return tt_src, torch_src


def _core_counts(device):
    """1 (latency, read hidden by overlap) and the full grid (bandwidth-bound, read shows)."""
    env = os.environ.get("CS_CORES")
    if env:
        return tuple(int(x) for x in env.split(","))
    grid = device.compute_with_storage_grid_size()
    return (1, grid.x * grid.y)


def _read_kernel_ns(device):
    """Sum of on-device kernel duration over programs dispatched since the last read."""
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


@pytest.mark.parametrize("variant", VARIANTS)
def test_constant_synthesis_correctness(device, variant):
    """Every variant produces the same output: VALUE in every element."""
    rows, w = SHAPE
    tt_src, torch_src = _make_source(device, rows, w)
    expected = torch_src.to(torch.float32)

    out = ttnn.to_torch(constant_synthesis(tt_src, variant=variant, value=VALUE, kernel_iters=ITERS, block=BLOCK)).to(
        torch.float32
    )
    assert list(out.shape) == list(expected.shape), f"{out.shape} != {expected.shape}"
    # Bit-exact: every element must equal the bf16-quantized constant.
    assert torch.equal(out, expected), f"output is not constant: unique values {torch.unique(out)[:8]}"
    assert_with_pcc(expected, out, 0.9999)


def test_constant_synthesis_device_perf(device):
    """Measure device kernel duration over variant x core count.

    Correctness lives in test_constant_synthesis_correctness; this test only
    measures and reports (perf is evidence, never a pass/fail — the only
    assertion here is that the profiler produced a number).
    """
    rows, w = SHAPE
    tt_src, _ = _make_source(device, rows, w)
    core_counts = _core_counts(device)
    page_bytes = w * ELEM_BYTES
    write_bytes = rows * page_bytes

    ns = {}
    for cores in core_counts:
        for variant in PERF_VARIANTS:
            run_fn = lambda c=cores, v=variant: constant_synthesis(
                tt_src, variant=v, value=VALUE, num_cores=c, kernel_iters=ITERS, block=BLOCK
            )
            value = _measure_ns(device, run_fn)
            assert (
                value is not None
            ), f"profiler produced no data for {variant} @ {cores} cores (profiler-enabled build?)"
            ns[(cores, variant)] = value

    arch = os.environ.get("ARCH_NAME", "unknown")
    lines = [
        "",
        "=== constant_synthesis device perf (constant-valued DRAM output) — source bytes read vs invented ===",
        f"    rows={rows}  W={w}  page_bytes={page_bytes}  write_bytes={write_bytes}  value={VALUE}  arch={arch}  block={BLOCK}  iters={ITERS}  trials={N_PROFILE_ITERS}",
        f"    {'cores':>6}  {'variant':<18}  {'ns/op':>11}  {'GB/s':>8}  {'vs baseline':>12}",
    ]
    for cores in core_counts:
        base = ns.get((cores, "stream_from_dram"))
        for variant in PERF_VARIANTS:
            v = ns[(cores, variant)]
            # candidate moves only writes; baseline moves reads + writes.
            moved = write_bytes if variant == "synthesize" else 2 * write_bytes
            gbps = (moved / v) if v else float("nan")  # bytes/ns == GB/s
            if variant == "stream_from_dram":
                tag = "  (baseline)"
            elif base:
                tag = f"  {base / v:6.2f}x"
            else:
                tag = ""
            lines.append(f"    {cores:>6}  {variant:<18}  {v:>11.1f}  {gbps:>8.1f}{tag}")
    logger.info("\n".join(lines))
