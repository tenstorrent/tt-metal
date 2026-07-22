# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the `index_staging` data-movement example.

One kernel-level decision drives an index-driven access (out[w] = src[idx[w]]): service
each random per-index access with a SEPARATE remote NoC read (remote_per_index,
baseline), or bulk-read the indexable dimension into L1 once and index locally
(l1_staged, candidate). A second axis, the index distribution (sorted vs.
shuffled), exposes the baseline's DRAM-locality sensitivity.
See ttnn/ttnn/operations/examples/index_staging/README.md.

    # every variant x distribution produces the identical, correct indexed select
    scripts/run_safe_pytest.sh --run-all \\
        tests/ttnn/unit_tests/operations/examples/test_index_staging.py::test_index_staging_correctness

    # device kernel duration, variant x distribution (in-process profiler)
    scripts/run_safe_pytest.sh --run-all \\
        tests/ttnn/unit_tests/operations/examples/test_index_staging.py::test_index_staging_device_perf
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
from ttnn.operations.examples.index_staging import index_staging, VARIANTS, DISTRIBUTIONS, ELEMS_PER_LINE

from loguru import logger

from tests.ttnn.utils_for_testing import assert_with_pcc


# Defaults overridable via env so the CLI (python -m ...index_staging) can measure
# the caller's own shape/params through the same in-process path.
# Shape is (rows, W): W = indices per row = source/output columns. Each selected
# element is one bf16 (2 B); a per-index remote read pulls a whole 32-byte line.
SHAPE = tuple(int(x) for x in os.environ.get("IS_SHAPE", "8,512").split(","))  # (rows, W)
NUM_CORES = int(os.environ.get("IS_CORES", "1"))
KERNEL_ITERS = int(os.environ.get("IS_ITERS", "1"))
DISTS = tuple(os.environ.get("IS_DISTS", "sorted,shuffled").split(","))
N_WARMUP = 5
N_PROFILE_ITERS = int(os.environ.get("IS_TRIALS", "20"))

_DURATION_KEY = "DEVICE KERNEL DURATION [ns]"


def _make_inputs(device, rows, w, dist):
    """Source [rows, w] bf16 + index [rows, w] uint32 (values in [0, w)).

    `sorted` and `shuffled` share the SAME index multiset (per row) — only the
    order differs — so the distribution axis isolates access ordering, nothing else.
    """
    torch.manual_seed(0)
    torch_src = (torch.rand(rows, w, dtype=torch.float32) * 2.0 - 1.0).to(torch.bfloat16)
    base = torch.randint(0, w, (rows, w), dtype=torch.int64)
    if dist == "sorted":
        idx = base.sort(dim=1).values
    elif dist == "shuffled":
        idx = base
    else:
        raise ValueError(f"unknown dist {dist!r}")

    tt_src = ttnn.from_torch(
        torch_src,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    tt_idx = ttnn.from_torch(
        idx.to(torch.int32),
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    return tt_src, tt_idx, torch_src, idx


def _torch_select(torch_src, idx, rows, w):
    return torch.take_along_dim(torch_src, idx, dim=1)


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


# (variant, dist) — every access strategy x both index distributions.
_CORRECTNESS_CASES = [(variant, dist) for variant in VARIANTS for dist in DISTRIBUTIONS]


@pytest.mark.parametrize("variant,dist", _CORRECTNESS_CASES)
def test_index_staging_correctness(device, variant, dist):
    """Every (variant, dist) is the same indexed select: output == source[:, index]."""
    rows, w = SHAPE
    tt_src, tt_idx, torch_src, idx = _make_inputs(device, rows, w, dist)
    expected = _torch_select(torch_src, idx, rows, w).to(torch.float32)

    out = ttnn.to_torch(
        index_staging(tt_src, tt_idx, variant=variant, num_cores=NUM_CORES, kernel_iters=KERNEL_ITERS)
    ).to(torch.float32)
    assert list(out.shape) == list(expected.shape), f"{out.shape} != {expected.shape}"
    assert_with_pcc(expected, out, 0.9999)  # bit-exact line copy; PCC is a formality


def test_index_staging_device_perf(device):
    """Measure device kernel duration over variant x distribution.

    Correctness lives in test_index_staging_correctness; this test only measures
    and reports (perf is evidence, never a pass/fail — the only assertion here is
    that the profiler produced a number).
    """
    rows, w = SHAPE

    ns = {}
    for dist in DISTS:
        tt_src, tt_idx, _, _ = _make_inputs(device, rows, w, dist)
        for variant in VARIANTS:
            run_fn = lambda s=tt_src, i=tt_idx, v=variant: index_staging(
                s, i, variant=v, num_cores=NUM_CORES, kernel_iters=KERNEL_ITERS
            )
            value = _measure_ns(device, run_fn)
            assert value is not None, f"profiler produced no data for {variant} {dist} (profiler-enabled build?)"
            ns[(dist, variant)] = value

    arch = os.environ.get("ARCH_NAME", "unknown")
    line_bytes = ELEMS_PER_LINE * 2
    lines = [
        "",
        "=== index_staging device perf (indexed select, DRAM row) — access strategy x index distribution ===",
        f"    rows={rows}  W={w}  elem_bytes=2  line_bytes={line_bytes}  src_row_bytes={w * 2}  baseline_read_bytes={w * line_bytes}  cores={NUM_CORES}  arch={arch}  iters={KERNEL_ITERS}  trials={N_PROFILE_ITERS}",
        f"    {'dist':<10}  {'variant':<18}  {'ns/op':>11}  {'vs baseline':>12}",
    ]
    for dist in DISTS:
        base = ns[(dist, "remote_per_index")]
        for variant in VARIANTS:
            v = ns[(dist, variant)]
            ratio = base / v if v else float("nan")
            tag = "  (baseline)" if variant == "remote_per_index" else f"  {ratio:6.2f}x"
            lines.append(f"    {dist:<10}  {variant:<18}  {v:>11.1f}{tag}")
    logger.info("\n".join(lines))
