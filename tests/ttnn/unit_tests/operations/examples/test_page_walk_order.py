# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the `page_walk_order` data-movement example.

One kernel-level decision drives a single reader over N interleaved-DRAM pages: the
ORDER in which it walks the page indices. Interleaved DRAM places page p in bank
(p % num_banks), so the walk stride decides whether consecutive in-flight reads hit
different banks (bank parallelism) or the same bank (serialized). Every walk order
reads the exact same set of pages, so the checksum is identical — only the temporal
order, and thus the achieved read bandwidth, changes.
See ttnn/ttnn/operations/examples/page_walk_order/README.md.

    # every walk order produces the identical, correct checksum
    scripts/run_safe_pytest.sh --run-all \\
        tests/ttnn/unit_tests/operations/examples/test_page_walk_order.py::test_page_walk_order_correctness

    # device kernel duration + read GB/s per walk order (in-process profiler)
    scripts/run_safe_pytest.sh --run-all \\
        tests/ttnn/unit_tests/operations/examples/test_page_walk_order.py::test_page_walk_order_device_perf
"""

import os

# Enable the on-device profiler IN-PROCESS (needs all three, set before the device
# opens). Scoped to this module so it doesn't perturb other examples' measurement.
# setdefault -> respects an outer tracy run if present.
os.environ.setdefault("TT_METAL_DEVICE_PROFILER", "1")
os.environ.setdefault("TT_METAL_PROFILER_MID_RUN_DUMP", "1")
os.environ.setdefault("TT_METAL_PROFILER_CPP_POST_PROCESS", "1")
os.environ.setdefault("TT_METAL_LOGGER_LEVEL", "error")  # silence loud C++ profiler histograms

import pytest
import torch

import ttnn
from ttnn.operations.examples.page_walk_order import page_walk_order, VARIANTS, stride_for, num_dram_banks

from loguru import logger

from tests.ttnn.utils_for_testing import assert_with_pcc  # noqa: F401 (kept for parity; checksum is exact)


# Defaults overridable via env so the CLI (python -m ...page_walk_order) can measure
# the caller's own params through the same in-process path.
PAGES = int(os.environ.get("PWO_PAGES", "1536"))  # requested; rounded up to a multiple of banks
PAGE_SIZE = int(os.environ.get("PWO_PAGE_SIZE", "1024"))  # bf16 elements per page (page bytes = 2*this)
STRIDES = os.environ.get("PWO_STRIDES", "auto")
VARIANT_SEL = os.environ.get("PWO_VARIANT", "all")
BLOCK = int(os.environ.get("PWO_BLOCK", "0"))  # 0 -> auto (2*banks)
KERNEL_ITERS = int(os.environ.get("PWO_ITERS", "1"))
N_WARMUP = 5
N_PROFILE_ITERS = int(os.environ.get("PWO_TRIALS", "20"))

_DURATION_KEY = "DEVICE KERNEL DURATION [ns]"


def _round_pages(requested, banks):
    """Round the requested page count up to a multiple of the bank count so a
    bank-strided walk enumerates cleanly (no wrap-induced bank changes)."""
    return ((requested + banks - 1) // banks) * banks


def _make_source(device, num_pages, width):
    """Source [num_pages, width] bf16 in interleaved DRAM (one row = one page)."""
    torch.manual_seed(0)
    torch_src = (torch.rand(num_pages, width, dtype=torch.float32) * 2.0 - 1.0).to(torch.bfloat16)
    tt_src = ttnn.from_torch(
        torch_src,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    return tt_src, torch_src


def _expected_checksum(torch_src):
    """Order-independent checksum: sum of word 0 (first bf16 halfword) of every page,
    mod 2**32 — matches the kernel's negligible per-page checksum."""
    word0 = torch_src.view(torch.int16)[:, 0].to(torch.int64) & 0xFFFF
    return int(word0.sum().item()) & 0xFFFFFFFF


def _actual_checksum(out_tensor):
    word0 = int(ttnn.to_torch(out_tensor).flatten()[0].item())
    return word0 & 0xFFFFFFFF


def _read_kernel_ns(device):
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
    for _ in range(N_WARMUP):
        run_fn()
    ttnn.synchronize_device(device)
    _read_kernel_ns(device)  # flush the warmup window
    for _ in range(N_PROFILE_ITERS):
        run_fn()
    total_ns = _read_kernel_ns(device)
    return total_ns / N_PROFILE_ITERS if total_ns is not None else None


@pytest.mark.parametrize("variant", VARIANTS)
def test_page_walk_order_correctness(device, variant):
    """Every walk order reads the same page set -> the same checksum as the host reference."""
    banks = num_dram_banks(device)
    num_pages = _round_pages(PAGES, banks)
    block = BLOCK or None
    tt_src, torch_src = _make_source(device, num_pages, PAGE_SIZE)
    expected = _expected_checksum(torch_src)

    out = page_walk_order(tt_src, variant=variant, block=block, kernel_iters=KERNEL_ITERS)
    actual = _actual_checksum(out)
    assert actual == expected, f"{variant}: checksum {actual:#x} != expected {expected:#x}"


def _selected_cases(banks):
    """Return list of (label, stride) to measure, from the env selection."""
    if STRIDES != "auto":
        return [(f"stride_{int(s)}", int(s)) for s in STRIDES.split(",") if s.strip()]
    if VARIANT_SEL == "all":
        names = VARIANTS
    else:
        names = tuple(v.strip() for v in VARIANT_SEL.split(",") if v.strip())
    return [(name, stride_for(name, banks)) for name in names]


def test_page_walk_order_device_perf(device):
    """Measure device kernel duration + achieved read GB/s per walk order.

    Correctness lives in test_page_walk_order_correctness; this test only measures
    and reports (perf is evidence, never a pass/fail — the only assertion here is
    that the profiler produced a number).
    """
    banks = num_dram_banks(device)
    num_pages = _round_pages(PAGES, banks)
    block = BLOCK or None
    page_bytes = PAGE_SIZE * 2
    bytes_read = num_pages * page_bytes * KERNEL_ITERS

    tt_src, _ = _make_source(device, num_pages, PAGE_SIZE)
    cases = _selected_cases(banks)

    ns = {}
    for label, stride in cases:
        run_fn = lambda s=stride: page_walk_order(tt_src, stride=s, block=block, kernel_iters=KERNEL_ITERS)
        value = _measure_ns(device, run_fn)
        assert value is not None, f"profiler produced no data for {label} (profiler-enabled build?)"
        ns[label] = value

    arch = os.environ.get("ARCH_NAME", ttnn.get_arch_name())
    effective_block = block if block is not None else 2 * banks
    # baseline for the ratio column: bank_stride if present, else the slowest.
    base_label = "bank_stride" if "bank_stride" in ns else max(ns, key=ns.get)
    base = ns[base_label]

    lines = [
        "",
        "=== page_walk_order device perf (single-core interleaved-DRAM read; page-index walk order) ===",
        f"    num_banks={banks}  pages={num_pages}  page_bytes={page_bytes}  bytes_read={bytes_read}  "
        f"block={effective_block}  cores=1  arch={arch}  iters={KERNEL_ITERS}  trials={N_PROFILE_ITERS}",
        f"    {'walk order':<16}  {'stride':>7}  {'ns/op':>12}  {'read GB/s':>10}  {'vs baseline':>12}",
    ]
    for label, stride in cases:
        v = ns[label]
        gbps = bytes_read / v if v else float("nan")  # bytes/ns == GB/s
        ratio = base / v if v else float("nan")
        tag = "  (baseline)" if label == base_label else f"  {ratio:6.2f}x"
        lines.append(f"    {label:<16}  {stride:>7}  {v:>12.1f}  {gbps:>10.2f}{tag}")
    logger.info("\n".join(lines))
