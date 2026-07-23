# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the `transfer_alignment` data-movement example.

One kernel-level decision drives a sub-page span read: service a non-congruent span
start with an over-read into an aligned scratch plus a local L1 realign (misaligned,
baseline), or arrange the span start congruent with the destination residue so one
direct read moves exactly the span bytes (aligned, candidate).
See ttnn/ttnn/operations/examples/transfer_alignment/README.md.

    # both variants extract a correct sub-page span; constant-source control proves byte-identity
    scripts/run_safe_pytest.sh --run-all \\
        tests/ttnn/unit_tests/operations/examples/test_transfer_alignment.py::test_transfer_alignment_correctness

    # device kernel duration, variant x span-width x span-count (in-process profiler)
    scripts/run_safe_pytest.sh --run-all \\
        tests/ttnn/unit_tests/operations/examples/test_transfer_alignment.py::test_transfer_alignment_device_perf
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
from ttnn.operations.examples.transfer_alignment import transfer_alignment, span_geometry, VARIANTS, ELEM_BYTES

from loguru import logger

from tests.ttnn.utils_for_testing import assert_with_pcc


# Defaults overridable via env so the CLI (python -m ...transfer_alignment) can measure
# the caller's own params through the same in-process path.
WIDTHS = tuple(int(x) for x in os.environ.get("TA_WIDTHS", "64,256,1024,4096").split(","))  # span bytes
NSPANS = tuple(int(x) for x in os.environ.get("TA_SPANS", "16,64,256").split(","))  # rows == spans
NUM_CORES = int(os.environ.get("TA_CORES", "1"))
KERNEL_ITERS = int(os.environ.get("TA_ITERS", "1"))
PERF_VARIANTS = tuple(os.environ.get("TA_VARIANTS", ",".join(VARIANTS)).split(","))
N_WARMUP = 3
N_PROFILE_ITERS = int(os.environ.get("TA_TRIALS", "10"))

_DURATION_KEY = "DEVICE KERNEL DURATION [ns]"


def _make_source(device, num_rows, span_bytes, *, constant=None):
    """Row-major bf16 source [num_rows, row_elems] wide enough for the padded geometry."""
    align = ttnn.get_dram_alignment()
    geom = span_geometry(align, span_bytes)
    if constant is not None:
        torch_src = torch.full((num_rows, geom.row_elems), float(constant), dtype=torch.float32).to(torch.bfloat16)
    else:
        torch.manual_seed(0)
        torch_src = (torch.rand(num_rows, geom.row_elems, dtype=torch.float32) * 2.0 - 1.0).to(torch.bfloat16)
    tt_src = ttnn.from_torch(
        torch_src,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    return tt_src, torch_src, geom


def _expected_span(torch_src, geom, variant):
    """The correct sub-page span each variant extracts (plain torch slicing, per variant)."""
    off = geom.off_aligned_elems if variant == "aligned" else geom.off_misaligned_elems
    return torch_src[:, off : off + geom.width_elems].to(torch.float32)


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


_CORRECTNESS_CASES = [(variant, w) for variant in VARIANTS for w in (64, 1024)]


@pytest.mark.parametrize("variant,span_bytes", _CORRECTNESS_CASES)
def test_transfer_alignment_correctness(device, variant, span_bytes):
    """Each variant extracts its correct sub-page span (bit-exact vs a torch span)."""
    num_rows = 32
    tt_src, torch_src, geom = _make_source(device, num_rows, span_bytes)
    expected = _expected_span(torch_src, geom, variant)

    out = ttnn.to_torch(transfer_alignment(tt_src, variant=variant, span_bytes=span_bytes, num_cores=NUM_CORES)).to(
        torch.float32
    )
    assert list(out.shape) == list(expected.shape), f"{out.shape} != {expected.shape}"
    assert torch.equal(out, expected), f"max diff {(out - expected).abs().max()}"  # bit-exact span copy
    assert_with_pcc(expected, out, 0.9999)


@pytest.mark.parametrize("span_bytes", [64, 1024])
def test_transfer_alignment_control_byte_identical(device, span_bytes):
    """Same-value control: on a constant source both variants produce byte-identical output.

    The aligned and misaligned variants read spans at different offsets, but a constant
    source makes every span equal — so this isolates that the misaligned over-read + L1
    realign path corrupts nothing relative to the direct read.
    """
    num_rows = 32
    tt_src, _, _ = _make_source(device, num_rows, span_bytes, constant=0.5)
    out_mis = ttnn.to_torch(
        transfer_alignment(tt_src, variant="misaligned", span_bytes=span_bytes, num_cores=NUM_CORES)
    )
    out_ali = ttnn.to_torch(transfer_alignment(tt_src, variant="aligned", span_bytes=span_bytes, num_cores=NUM_CORES))
    assert torch.equal(out_mis, out_ali), "misaligned and aligned outputs differ on a constant source"


def test_transfer_alignment_device_perf(device):
    """Measure device kernel duration over variant x span-width x span-count.

    Correctness lives in test_transfer_alignment_correctness; this test only measures
    and reports (perf is evidence, never a pass/fail — the only assertion here is that
    the profiler produced a number).
    """
    align = ttnn.get_dram_alignment()

    ns = {}
    for num_rows in NSPANS:
        for span_bytes in WIDTHS:
            tt_src, _, _ = _make_source(device, num_rows, span_bytes)
            for variant in PERF_VARIANTS:
                run_fn = lambda s=tt_src, v=variant, w=span_bytes: transfer_alignment(
                    s, variant=v, span_bytes=w, num_cores=NUM_CORES, kernel_iters=KERNEL_ITERS
                )
                value = _measure_ns(device, run_fn)
                assert value is not None, f"profiler produced no data for {variant} w={span_bytes} N={num_rows}"
                ns[(num_rows, span_bytes, variant)] = value

    arch = os.environ.get("ARCH_NAME", "unknown")
    residue = align // 2
    lines = [
        "",
        "=== transfer_alignment device perf (sub-page span read, DRAM row) — read strategy x width x count ===",
        f"    align_window={align}B  misalign_residue={residue}B  elem_bytes={ELEM_BYTES}  cores={NUM_CORES}  arch={arch}  iters={KERNEL_ITERS}  trials={N_PROFILE_ITERS}",
        f"    {'N (spans)':>10}  {'width (B)':>9}  {'misaligned ns':>14}  {'aligned ns':>12}  {'over-read B':>11}  {'speedup':>8}",
    ]
    for num_rows in NSPANS:
        for span_bytes in WIDTHS:
            mis = ns.get((num_rows, span_bytes, "misaligned"))
            ali = ns.get((num_rows, span_bytes, "aligned"))
            over = span_bytes + residue
            if mis is not None and ali is not None:
                speed = mis / ali if ali else float("nan")
                lines.append(
                    f"    {num_rows:>10}  {span_bytes:>9}  {mis:>14.1f}  {ali:>12.1f}  {over:>11}  {speed:>7.2f}x"
                )
            else:
                only = "misaligned" if mis is not None else "aligned"
                val = mis if mis is not None else ali
                lines.append(f"    {num_rows:>10}  {span_bytes:>9}  {only}={val:.1f}")
    logger.info("\n".join(lines))
