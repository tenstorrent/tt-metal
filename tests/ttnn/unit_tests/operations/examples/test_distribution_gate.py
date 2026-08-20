# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the `distribution_gate` work-distribution example.

Splitting a tile op across the grid along a FIXED axis fills the grid for one
aspect ratio and strands the other on ~1 core: a height (tile-row) split strands
WIDE-SHORT tensors; a width (tile-column) split strands TALL-NARROW tensors. The
"gated" strategy keeps the height split as the default and diverts to a width
split only when the height split under-fills the grid — filling the grid on both
regimes without regressing either. See
ttnn/ttnn/operations/examples/distribution_gate/README.md.

    # all three variants produce the identical, correct relu output at every shape
    scripts/run_safe_pytest.sh --run-all \\
        tests/ttnn/unit_tests/operations/examples/test_distribution_gate.py::test_distribution_gate_correctness

    # device kernel duration + active cores, height_split vs width_split vs gated
    scripts/run_safe_pytest.sh --run-all \\
        tests/ttnn/unit_tests/operations/examples/test_distribution_gate.py::test_distribution_gate_device_perf
"""

import os

# Enable the on-device profiler IN-PROCESS (all three, before the device opens).
os.environ.setdefault("TT_METAL_DEVICE_PROFILER", "1")
os.environ.setdefault("TT_METAL_PROFILER_MID_RUN_DUMP", "1")
os.environ.setdefault("TT_METAL_PROFILER_CPP_POST_PROCESS", "1")

import pytest
import torch

import ttnn
from ttnn.operations.examples.distribution_gate import distribution_gate, VARIANTS, num_active_cores

from loguru import logger

from tests.ttnn.utils_for_testing import assert_with_pcc


TILE = 32
_DTYPES = {"bfloat8_b": (ttnn.bfloat8_b, 0.99), "bfloat16": (ttnn.bfloat16, 0.9999), "float32": (ttnn.float32, 0.9999)}


def _parse_shapes(spec):
    """'HxW,HxW,...' -> [(H, W), ...]."""
    out = []
    for tok in spec.split(","):
        h, w = tok.lower().split("x")
        out.append((int(h), int(w)))
    return out


# Aspect-ratio sweep: wide-short (height split strands it), tall-narrow (width split
# strands it), and two grid-filling controls where both axes are fine.
DEFAULT_SHAPES = "32x4096,2048x32,2048x2048,1024x1024"
SHAPES = _parse_shapes(os.environ.get("DG_SHAPES", DEFAULT_SHAPES))
VARIANT_SEL = os.environ.get("DG_VARIANT", "all")
DTYPE_NAME = os.environ.get("DG_DTYPE", "bfloat16")
DTYPE, DTYPE_PCC = _DTYPES[DTYPE_NAME]
KERNEL_ITERS = int(os.environ.get("DG_ITERS", "1"))
N_WARMUP = 5
N_PROFILE_ITERS = int(os.environ.get("DG_TRIALS", "20"))

_DURATION_KEY = "DEVICE KERNEL DURATION [ns]"


def _make_input(device, h, w, dtype=DTYPE):
    torch.manual_seed(0)
    torch_input = torch.rand((h, w), dtype=torch.float32) * 2.0 - 1.0  # signed -> relu is non-trivial
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
    (name, variant, h, w)
    for name in _DTYPES
    for variant in VARIANTS
    for (h, w) in ((32, 4096), (2048, 32), (1024, 1024))
]


@pytest.mark.parametrize("dtype_name,variant,h,w", _CORRECTNESS_CASES)
def test_distribution_gate_correctness(device, dtype_name, variant, h, w):
    """Every variant at every aspect ratio produces the same output: out == relu(input)."""
    dtype, pcc = _DTYPES[dtype_name]
    tt_input = _make_input(device, h, w, dtype=dtype)
    expected = torch.relu(ttnn.to_torch(tt_input).to(torch.float32))
    out = ttnn.to_torch(distribution_gate(tt_input, variant=variant, kernel_iters=KERNEL_ITERS)).to(torch.float32)
    assert list(out.shape) == list(expected.shape), f"{out.shape} != {expected.shape}"
    assert_with_pcc(expected, out, pcc)


def test_distribution_gate_device_perf(device):
    """Measure device kernel duration + active cores: height_split vs width_split vs gated,
    across aspect ratios.

    Correctness lives in test_distribution_gate_correctness; this only measures/reports
    (perf is evidence, never pass/fail — the only assertion is that the profiler produced a number)."""
    variants = VARIANTS if VARIANT_SEL == "all" else (VARIANT_SEL,)
    ns = {}
    for h, w in SHAPES:
        tt_input = _make_input(device, h, w)
        for variant in variants:
            run_fn = lambda v=variant, t=tt_input: distribution_gate(t, variant=v, kernel_iters=KERNEL_ITERS)
            value = _measure_ns(device, run_fn)
            assert value is not None, f"profiler produced no data for {variant} {h}x{w} (profiler-enabled build?)"
            ns[(h, w, variant)] = value

    arch = os.environ.get("ARCH_NAME", "unknown")
    grid = device.compute_with_storage_grid_size()
    lines = [
        "",
        "=== distribution_gate device perf (relu) — height_split vs width_split vs gated ===",
        f"    dtype={DTYPE_NAME}  grid={grid.x}x{grid.y}={grid.x*grid.y}  arch={arch}  iters={KERNEL_ITERS}  trials={N_PROFILE_ITERS}",
        f"    {'H':>5}x{'W':<5}  {'Ht':>3}x{'Wt':<3}  {'variant':<13}  {'cores':>5}  {'ns/op':>11}  {'vs gated':>9}",
    ]
    for h, w in SHAPES:
        ht, wt = h // TILE, w // TILE
        gated = ns.get((h, w, "gated"))
        for variant in variants:
            v = ns[(h, w, variant)]
            cores = num_active_cores(variant, device, ht, wt)
            ratio = (v / gated) if (gated and variant != "gated") else None
            tag = "  (ref)" if variant == "gated" else (f"  {ratio:5.2f}x" if ratio else "")
            lines.append(f"    {h:>5}x{w:<5}  {ht:>3}x{wt:<3}  {variant:<13}  {cores:>5}  {v:>11.1f}{tag}")
    logger.info("\n".join(lines))
