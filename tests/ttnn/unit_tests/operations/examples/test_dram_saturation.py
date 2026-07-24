# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the `dram_saturation` work-distribution example.

A DRAM-bound copy does not get faster forever as you add cores: achieved bandwidth
rises with core count, then PLATEAUS once the DRAM interface saturates. Past that
knee, extra cores add no bandwidth (wasted), and if stacked onto shared NoC links
they congest and the copy gets slower. The sweet spot is the minimum well-placed
cores that saturate the bus. This sweeps core count for two placements (spread vs
stacked) and reports achieved GB/s. See
ttnn/ttnn/operations/examples/dram_saturation/README.md.

    # every variant/core-count produces the identical copy: out == input
    scripts/run_safe_pytest.sh --run-all \\
        tests/ttnn/unit_tests/operations/examples/test_dram_saturation.py::test_dram_saturation_correctness

    # achieved GB/s vs core count, spread vs stacked
    scripts/run_safe_pytest.sh --run-all \\
        tests/ttnn/unit_tests/operations/examples/test_dram_saturation.py::test_dram_saturation_device_perf
"""

import os

# Enable the on-device profiler IN-PROCESS (all three, before the device opens).
os.environ.setdefault("TT_METAL_DEVICE_PROFILER", "1")
os.environ.setdefault("TT_METAL_PROFILER_MID_RUN_DUMP", "1")
os.environ.setdefault("TT_METAL_PROFILER_CPP_POST_PROCESS", "1")

import pytest
import torch

import ttnn
from ttnn.operations.examples.dram_saturation import dram_saturation, VARIANTS, num_active_cores

from loguru import logger

from tests.ttnn.utils_for_testing import assert_with_pcc


TILE = 32
_DTYPES = {"bfloat8_b": (ttnn.bfloat8_b, 0.99), "bfloat16": (ttnn.bfloat16, 0.9999), "float32": (ttnn.float32, 0.9999)}


def _parse_shape(spec):
    h, w = spec.lower().split("x")
    return int(h), int(w)


# Large, DRAM-bound tensor so the copy is bandwidth-bound (where the core-count knee lives).
PERF_SHAPE = _parse_shape(os.environ.get("DS_SHAPE", "2048x2048"))
CORES = tuple(int(x) for x in os.environ.get("DS_CORES", "1,2,4,8,16,32,48,64").split(","))
VARIANT_SEL = os.environ.get("DS_VARIANT", "all")
DTYPE_NAME = os.environ.get("DS_DTYPE", "bfloat16")
DTYPE, DTYPE_PCC = _DTYPES[DTYPE_NAME]
KERNEL_ITERS = int(os.environ.get("DS_ITERS", "1"))
N_WARMUP = 5
N_PROFILE_ITERS = int(os.environ.get("DS_TRIALS", "20"))

_DURATION_KEY = "DEVICE KERNEL DURATION [ns]"


def _make_input(device, h, w, dtype=DTYPE):
    torch.manual_seed(0)
    torch_input = torch.rand((h, w), dtype=torch.float32) * 2.0 - 1.0
    return ttnn.from_torch(
        torch_input, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )


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
    for _ in range(N_WARMUP):
        run_fn()
    ttnn.synchronize_device(device)
    _read_kernel_ns(device)  # flush warmup window
    for _ in range(N_PROFILE_ITERS):
        run_fn()
    total = _read_kernel_ns(device)
    return total / N_PROFILE_ITERS if total is not None else None


_CORRECTNESS_CASES = [
    (name, variant, n)
    for name in _DTYPES
    for variant in VARIANTS
    for n in (1, 7, 8, 64)  # incl. a remainder case (7) and full grid
]


@pytest.mark.parametrize("dtype_name,variant,n", _CORRECTNESS_CASES)
def test_dram_saturation_correctness(device, dtype_name, variant, n):
    """Every variant at every core count produces the identical copy: out == input."""
    dtype, pcc = _DTYPES[dtype_name]
    tt_input = _make_input(device, 256, 256, dtype=dtype)
    expected = ttnn.to_torch(tt_input).to(torch.float32)
    out = ttnn.to_torch(dram_saturation(tt_input, variant=variant, num_cores=n, kernel_iters=KERNEL_ITERS)).to(
        torch.float32
    )
    assert list(out.shape) == list(expected.shape), f"{out.shape} != {expected.shape}"
    assert_with_pcc(expected, out, pcc)


def test_dram_saturation_device_perf(device):
    """Achieved GB/s vs core count, spread vs stacked, on a DRAM-bound copy.

    Correctness lives in test_dram_saturation_correctness; this only measures/reports
    (perf is evidence, never pass/fail — the only assertion is that the profiler produced a number)."""
    variants = VARIANTS if VARIANT_SEL == "all" else (VARIANT_SEL,)
    h, w = PERF_SHAPE
    tt_input = _make_input(device, h, w)
    page_bytes = tt_input.buffer_aligned_page_size()
    num_pages = tt_input.buffer_num_pages()
    # read + write => 2x the tensor bytes moved. GB/s = 2*bytes / ns (the 1e-9/1e9 cancel).
    bytes_moved = 2.0 * num_pages * page_bytes

    ns = {}
    for variant in variants:
        for n in CORES:
            run_fn = lambda v=variant, c=n, t=tt_input: dram_saturation(
                t, variant=v, num_cores=c, kernel_iters=KERNEL_ITERS
            )
            value = _measure_ns(device, run_fn)
            assert value is not None, f"profiler produced no data for {variant} cores={n} (profiler-enabled build?)"
            ns[(variant, n)] = value

    arch = os.environ.get("ARCH_NAME", "unknown")
    grid = device.compute_with_storage_grid_size()
    lines = [
        "",
        "=== dram_saturation device perf (pure DRAM->DRAM copy) — achieved GB/s vs core count ===",
        f"    shape={h}x{w} ({num_pages} tiles, {num_pages * page_bytes / 1e6:.1f} MB)  dtype={DTYPE_NAME}"
        f"  grid={grid.x}x{grid.y}={grid.x * grid.y}  arch={arch}  iters={KERNEL_ITERS}  trials={N_PROFILE_ITERS}",
        f"    {'variant':<9}  {'cores':>5}  {'ns/op':>11}  {'GB/s':>8}  {'GB/s/core':>9}",
    ]
    for variant in variants:
        prev = None
        for n in CORES:
            v = ns[(variant, n)]
            cores = num_active_cores(device, n)
            gbps = bytes_moved / v  # = 2*bytes / ns
            marker = ""
            if prev is not None and v > prev * 1.02:
                marker = "  <- slower (rollover)"
            lines.append(f"    {variant:<9}  {cores:>5}  {v:>11.1f}  {gbps:>8.1f}  {gbps / cores:>9.1f}{marker}")
            prev = v
    logger.info("\n".join(lines))
