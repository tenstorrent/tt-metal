# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the `noc_one_packet` data-movement example.

`noc_async_write`'s `max_page_size` template parameter defaults to
`NOC_MAX_BURST_SIZE + 1`, so a plain call compiles to the GENERIC multi-burst path
even for a page that is provably one packet. Naming the size as the template
argument selects the one-packet path. Both move the same bytes over the same NoC
transaction; only the per-call software cost on the issuing RISC-V differs.

The example is a pure L1->L1 ring shift with no compute, so the issuing core IS the
kernel and the per-call cost shows up directly in the device kernel duration.

    # both variants produce the identical, correct ring-shifted output
    scripts/run_safe_pytest.sh --run-all \\
        tests/ttnn/unit_tests/operations/examples/test_noc_one_packet.py::test_noc_one_packet_correctness

    # device kernel duration, generic vs one_packet, across page sizes / page counts
    scripts/run_safe_pytest.sh --run-all \\
        tests/ttnn/unit_tests/operations/examples/test_noc_one_packet.py::test_noc_one_packet_device_perf
"""

import os

# Enable the on-device profiler IN-PROCESS (all three, before the device opens).
os.environ.setdefault("TT_METAL_DEVICE_PROFILER", "1")
os.environ.setdefault("TT_METAL_PROFILER_MID_RUN_DUMP", "1")
os.environ.setdefault("TT_METAL_PROFILER_CPP_POST_PROCESS", "1")
# The profiler logs a per-read duration histogram at C++ INFO; a variant x params
# sweep would bury the report. This gags only the C++ logger -- the report below is
# emitted through Python loguru, and durations come from program_analyses_results.
os.environ.setdefault("TT_METAL_LOGGER_LEVEL", "error")

import pytest
import torch

import ttnn
from ttnn.operations.examples.noc_one_packet import (
    VARIANTS,
    create_sharded_memory_config,
    noc_one_packet,
)

from loguru import logger

_DURATION_KEY = "DEVICE KERNEL DURATION [ns]"

# page_bytes = page_elems * 2 (bfloat16). All are multiples of 64 B.
PAGE_ELEMS = tuple(int(x) for x in os.environ.get("N1P_PAGE_ELEMS", "32,128,512,1024,2048").split(","))
PAGES_PER_CORE = tuple(int(x) for x in os.environ.get("N1P_PAGES", "32").split(","))
NUM_CORES = int(os.environ.get("N1P_CORES", "8"))
KERNEL_ITERS = int(os.environ.get("N1P_KERNEL_ITERS", "1"))
TRIALS = int(os.environ.get("N1P_TRIALS", "3"))
VARIANT_SEL = os.environ.get("N1P_VARIANT", "all")

_RESULTS = []


def _variants():
    if VARIANT_SEL == "all":
        return list(VARIANTS)
    names = [v for v in VARIANT_SEL.split(",") if v]
    unknown = set(names) - set(VARIANTS)
    if unknown:
        raise ValueError(f"unknown variant(s): {sorted(unknown)}")
    return names


def _make_tensors(device, page_elems, pages_per_core):
    rows = NUM_CORES * pages_per_core
    torch.manual_seed(0)
    torch_input = torch.randn((rows, page_elems), dtype=torch.float32)
    mem_cfg = create_sharded_memory_config(device, NUM_CORES, pages_per_core, page_elems, ttnn.bfloat16)
    tt_in = ttnn.from_torch(
        torch_input, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=mem_cfg
    )
    tt_out = ttnn.from_torch(
        torch.zeros_like(torch_input),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=mem_cfg,
    )
    return tt_in, tt_out, ttnn.to_torch(tt_in)


def _golden(src, pages_per_core, dest_spread):
    """Where each page lands.

    dest_spread=False -- ring shift by one shard: output shard k+1 gets input shard k.
    dest_spread=True  -- page p of core k goes to core (k+1+p) % C, same page slot, so
                         output shard d page p comes from shard (d-1-p) % C.
    """
    if not dest_spread:
        return torch.roll(src, shifts=pages_per_core, dims=0)
    out = torch.empty_like(src)
    for d in range(NUM_CORES):
        for p in range(pages_per_core):
            s_core = (d - 1 - (p % NUM_CORES)) % NUM_CORES
            out[d * pages_per_core + p] = src[s_core * pages_per_core + p]
    return out


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
    run_fn()  # discard the first launch (JIT + first-touch)
    ttnn.synchronize_device(device)
    samples = []
    for _ in range(TRIALS):
        _read_kernel_ns(device)  # flush the window
        run_fn()
        ttnn.synchronize_device(device)
        ns = _read_kernel_ns(device)
        if ns:
            samples.append(ns)
    if not samples:
        return None, None
    samples.sort()
    median = samples[len(samples) // 2]
    spread = (samples[-1] - samples[0]) / median * 100 if median else 0.0
    return median, spread


@pytest.mark.parametrize("page_elems", PAGE_ELEMS)
@pytest.mark.parametrize("pages_per_core", PAGES_PER_CORE)
@pytest.mark.parametrize("dest_spread", [False, True], ids=["dest=single", "dest=spread"])
@pytest.mark.parametrize("variant", VARIANTS)
def test_noc_one_packet_correctness(device, variant, dest_spread, pages_per_core, page_elems):
    """Both variants must produce the identical permuted copy. Bit-exact: it is a copy."""
    tt_in, tt_out, src = _make_tensors(device, page_elems, pages_per_core)
    golden = _golden(src, pages_per_core, dest_spread)
    noc_one_packet(
        tt_in,
        tt_out,
        variant=variant,
        num_cores=NUM_CORES,
        pages_per_core=pages_per_core,
        kernel_iters=1,
        dest_spread=dest_spread,
    )
    got = ttnn.to_torch(tt_out)
    assert torch.equal(got, golden), (
        f"{variant} page_elems={page_elems} pages={pages_per_core} spread={dest_spread}: copy mismatch "
        f"(max abs diff {(got - golden).abs().max().item()})"
    )


@pytest.mark.parametrize("page_elems", PAGE_ELEMS)
@pytest.mark.parametrize("pages_per_core", PAGES_PER_CORE)
def test_noc_one_packet_device_perf(device, pages_per_core, page_elems):
    """Device kernel duration per variant. Correctness is asserted; perf is only reported."""
    page_bytes = page_elems * 2
    for dest_spread in (False, True):
        for variant in _variants():
            tt_in, tt_out, src = _make_tensors(device, page_elems, pages_per_core)
            golden = _golden(src, pages_per_core, dest_spread)

            def run():
                noc_one_packet(
                    tt_in,
                    tt_out,
                    variant=variant,
                    num_cores=NUM_CORES,
                    pages_per_core=pages_per_core,
                    kernel_iters=KERNEL_ITERS,
                    dest_spread=dest_spread,
                )

            median, spread = _measure_ns(device, run)
            assert torch.equal(ttnn.to_torch(tt_out), golden), f"{variant}: incorrect output"
            writes = pages_per_core * KERNEL_ITERS
            _RESULTS.append(
                {
                    "dest": "spread" if dest_spread else "single",
                    "variant": variant,
                    "page_bytes": page_bytes,
                    "pages_per_core": pages_per_core,
                    "ns": median,
                    "spread": spread,
                    "ns_per_write": (median / writes) if median else None,
                }
            )


@pytest.fixture(scope="module", autouse=True)
def _report():
    yield
    if not _RESULTS:
        return
    arch = os.environ.get("ARCH_NAME", "unknown")
    box = os.uname().nodename
    logger.info("")
    logger.info(
        f"noc_one_packet   box={box}  arch={arch}   trials={TRIALS} (median)   "
        f"kernel-iters={KERNEL_ITERS}   cores={NUM_CORES}  placement=row-major grid  "
        f"traffic=L1->L1 ring shift (no compute)"
    )
    header = (
        f"  {'dest':<7} {'page_bytes':>10} {'pages/core':>10} {'variant':<11} {'kernel ns':>10} "
        f"{'spread':>7} {'ns/write':>9} {'speedup':>8}"
    )
    logger.info(header)
    by_case = {}
    for row in _RESULTS:
        by_case.setdefault((row["dest"], row["page_bytes"], row["pages_per_core"]), []).append(row)
    for (dest, page_bytes, pages), rows in sorted(by_case.items()):
        base = next((r for r in rows if r["variant"] == "generic"), None)
        for row in rows:
            speed = ""
            if base and base["ns"] and row["ns"] and row["variant"] != "generic":
                speed = f"{base['ns'] / row['ns']:.2f}x"
            logger.info(
                f"  {dest:<7} {page_bytes:>10} {pages:>10} {row['variant']:<11} {row['ns']:>10.0f} "
                f"{row['spread']:>6.1f}% {row['ns_per_write']:>9.1f} {speed:>8}"
            )
