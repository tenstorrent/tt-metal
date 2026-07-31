# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the `dual_noc_read` example: two DRAM operand streams, one or two read engines.

An op with two DRAM inputs usually has the reader (NCRISC, NoC 0) fetch both, leaving BRISC's NoC-1
port idle and pinning the core's input bandwidth to one NoC port. Giving one operand to each
data-movement RISC fetches them concurrently on the two NoCs. See
ttnn/ttnn/operations/examples/dual_noc_read/README.md.

    # every variant at every block produces the identical, correct A*B
    scripts/run_safe_pytest.sh --run-all \\
        tests/ttnn/unit_tests/operations/examples/test_dual_noc_read.py::test_dual_noc_read_correctness

    # device kernel duration + achieved read GB/s, block x variant (in-process profiler)
    scripts/run_safe_pytest.sh --run-all \\
        tests/ttnn/unit_tests/operations/examples/test_dual_noc_read.py::test_dual_noc_read_device_perf
"""

import os

# Enable the on-device profiler IN-PROCESS (needs all three, set before the device opens). Scoped to
# this module so it doesn't perturb other examples. setdefault -> an outer tracy run still wins.
os.environ.setdefault("TT_METAL_DEVICE_PROFILER", "1")
os.environ.setdefault("TT_METAL_PROFILER_MID_RUN_DUMP", "1")
os.environ.setdefault("TT_METAL_PROFILER_CPP_POST_PROCESS", "1")
# The profiler logs a per-read duration histogram at C++ INFO; a variant x block sweep buries the
# report in them. Gags only the C++ logger — loguru (the report) and the measurement are unaffected.
os.environ.setdefault("TT_METAL_LOGGER_LEVEL", "error")

import socket
import statistics

import pytest
import torch
from loguru import logger

import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc
from ttnn.operations.examples.dual_noc_read import (
    BASELINE,
    LABEL,
    NOCS,
    RISCS,
    VARIANTS,
    dual_noc_read,
)

TILE = 32

# Defaults overridable via env so the CLI measures the caller's own params through this same path.
SHAPE = tuple(int(x) for x in os.environ.get("DNR_SHAPE", "1024,128").split(","))  # 128 tiles/operand
BLOCK_SWEEP = tuple(int(x) for x in os.environ.get("DNR_BLOCKS", "1,2,4,8,16,32").split(","))
# Transaction-size sweep for the MECHANISM table: same total bytes, proportionally more NoC
# commands as txn_bytes shrinks. 2048 = one read per bf16 tile page.
TXN_SWEEP = tuple(int(x) for x in os.environ.get("DNR_TXNS", "2048,1024,512,256").split(","))


def _selected_variants():
    """DNR_VARIANTS=a,b restricts the sweep; the baseline is always kept so speedups have a
    reference. Order always follows VARIANTS (baseline first)."""
    sel = os.environ.get("DNR_VARIANTS")
    if not sel:
        return VARIANTS
    chosen = set(sel.split(",")) | {BASELINE}
    if unknown := chosen - set(VARIANTS):
        raise ValueError(f"unknown DNR_VARIANTS: {sorted(unknown)}; valid: {VARIANTS}")
    return tuple(v for v in VARIANTS if v in chosen)


N_WARMUP = 5
N_PROFILE_ITERS = int(os.environ.get("DNR_ITERS", "10"))  # launches averaged inside one profiler window
N_TRIALS = int(os.environ.get("DNR_TRIALS", "5"))  # independent windows -> median + spread

_DURATION_KEY = "DEVICE KERNEL DURATION [ns]"
_PCC = 0.9999  # bf16 multiply; exact up to tile-format rounding


def _make_inputs(device, shape=SHAPE, seed=0):
    """Two DRAM-interleaved bf16 tiled operands. Values in [0.5, 1.5] so the product is well
    conditioned in bf16 (no cancellation, no denormals) and PCC reflects wiring, not rounding."""
    torch.manual_seed(seed)
    ta = 0.5 + torch.rand(shape, dtype=torch.float32)
    tb = 0.5 + torch.rand(shape, dtype=torch.float32)
    to_dev = lambda t: ttnn.from_torch(
        t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    return to_dev(ta), to_dev(tb)


def _read_kernel_ns(device):
    """Sum of on-device kernel duration over programs dispatched since the last read.

    ReadDeviceProfiler finishes the queue before reading and *consumes* the window, so a flush-read
    then a work-read brackets exactly the ops run in between.
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
    """(median ns/op, spread %) over N_TRIALS independent windows of N_PROFILE_ITERS launches each.

    Each window averages N_PROFILE_ITERS launches; the median across windows is the reported number
    and the population stdev across them is the noise estimate.
    """
    for _ in range(N_WARMUP):
        run_fn()
    ttnn.synchronize_device(device)
    _read_kernel_ns(device)  # discard the warmup window
    samples = []
    for _ in range(N_TRIALS):
        for _ in range(N_PROFILE_ITERS):
            run_fn()
        total_ns = _read_kernel_ns(device)
        if total_ns is None:
            return None, None
        samples.append(total_ns / N_PROFILE_ITERS)
    med = statistics.median(samples)
    spread = (statistics.pstdev(samples) / med * 100) if (len(samples) > 1 and med) else 0.0
    return med, spread


def _arch_label(device):
    if name := os.environ.get("ARCH_NAME"):
        return name
    a = str(device.arch()).rsplit(".", 1)[-1]
    return {"WORMHOLE_B0": "WH_B0", "BLACKHOLE": "BH", "GRAYSKULL": "GS"}.get(a, a)


# =============================================================================
# Correctness — the only pass/fail. Every variant x block must produce A*B.
# =============================================================================
_CORRECTNESS_CASES = [(variant, block) for variant in VARIANTS for block in (1, 2, 8, 32)]


@pytest.mark.parametrize("variant,block", _CORRECTNESS_CASES)
def test_dual_noc_read_correctness(device, variant, block):
    """Same multiply however the operands are fetched: output == A*B."""
    a, b = _make_inputs(device)
    expected = ttnn.to_torch(a).to(torch.float32) * ttnn.to_torch(b).to(torch.float32)
    out = ttnn.to_torch(dual_noc_read(a, b, variant=variant, block=block)).to(torch.float32)
    assert list(out.shape) == list(expected.shape), f"{out.shape} != {expected.shape}"
    assert_with_pcc(expected, out, _PCC)


def test_dual_noc_read_block_must_divide_tiles(device, expect_error):
    """The semaphore variant's cb_b slot arithmetic assumes full blocks, so a tail is rejected
    up front rather than silently reading into the wrong slot."""
    a, b = _make_inputs(device)
    with expect_error(ValueError, "divisible by block"):
        dual_noc_read(a, b, variant="two_riscv", block=48)  # 48 does not divide 128


# =============================================================================
# Perf — measured and reported, never asserted.
# =============================================================================
def test_dual_noc_read_device_perf(device):
    """Device kernel duration + achieved DRAM READ bandwidth over block x variant.

    Correctness lives in test_dual_noc_read_correctness; this test only measures and reports (the
    only assertion is that the profiler produced a number).
    """
    variants = _selected_variants()
    a, b = _make_inputs(device)
    tiles_per_operand = (SHAPE[0] // TILE) * (SHAPE[1] // TILE)
    page_bytes = a.buffer_aligned_page_size()
    read_bytes = 2 * tiles_per_operand * page_bytes  # both operands, one pass, no write traffic

    # do_math=True is the real op; do_math=False is the payload ablation (mul_tiles removed, every
    # CB handshake and the pack cycle kept) that exposes the pure read ceiling — it tells us whether
    # the FPU is masking the read-side win at large blocks.
    ns, sd = {}, {}
    for block in BLOCK_SWEEP:
        if tiles_per_operand % block:
            continue  # not a legal block for this shape (see validate)
        for variant in variants:
            for math in (True, False):
                run_fn = lambda v=variant, bl=block, m=math: dual_noc_read(a, b, variant=v, block=bl, do_math=m)
                value, spread = _measure_ns(device, run_fn)
                assert value is not None, f"profiler produced no data for {variant} block={block} math={math}"
                ns[(block, variant, math)] = value
                sd[(block, variant, math)] = spread

    blocks = sorted({k[0] for k in ns})
    max_spread = max(sd.values())
    lines = [
        "",
        "=== dual_noc_read device perf — two DRAM operands, one core, 1 vs 2 read engines ===",
        f"    box={socket.gethostname()}  arch={_arch_label(device)}  cores=1  placement=single core (0,0)",
        f"    shape={SHAPE}  tiles/operand={tiles_per_operand}  dtype=bfloat16  tile_bytes={page_bytes}"
        f"  N={N_TRIALS} windows x {N_PROFILE_ITERS} launches (median +- pstdev)",
        f"    DRAM READ traffic = {read_bytes / 1e6:.3f} MB/launch (both operands; output stays in L1,"
        " so there is no write traffic)",
        f"    read GB/s = {read_bytes} B / kernel_ns;  speedup is vs {BASELINE} at the SAME block",
        "",
        f"    max spread across all cells: {max_spread:.1f}%"
        + ("  (noisy — treat sub-5% deltas as ties)" if max_spread > 5 else ""),
        "",
        "  [1] FULL OP (C = A*B)",
        f"    {'block':>5}  {'variant':<18} {'riscs':>5} {'nocs':>4}  {'ns/op':>10} {'+-%':>5}  {'read GB/s':>9}  {'vs base':>8}",
    ]
    for block in blocks:
        base = ns[(block, BASELINE, True)]
        for variant in variants:
            v = ns[(block, variant, True)]
            tag = "  (base)" if variant == BASELINE else f"  {base / v:5.2f}x"
            lines.append(
                f"    {block:>5}  {variant:<18} {RISCS[variant]:>5} {NOCS[variant]:>4}"
                f"  {v:>10.1f} {sd[(block, variant, True)]:>5.1f}  {read_bytes / v:>9.1f}{tag}"
            )

    lines += [
        "",
        "  [2] PAYLOAD-ABLATED (mul_tiles removed; every CB handshake + the pack cycle kept) —",
        "      the pure read ceiling. 'math cost' = full - ablated, i.e. how much the FPU adds on top",
        "      of the read. Where math cost is large the FPU, not the NoC, sets the pace.",
        f"    {'block':>5}  {'variant':<18} {'riscs':>5} {'nocs':>4}  {'ablated ns':>10} {'+-%':>5}  {'read GB/s':>9}  {'vs base':>8}",
    ]
    for block in blocks:
        base = ns[(block, BASELINE, False)]
        for variant in variants:
            v = ns[(block, variant, False)]
            tag = "  (base)" if variant == BASELINE else f"  {base / v:5.2f}x"
            lines.append(
                f"    {block:>5}  {variant:<18} {RISCS[variant]:>5} {NOCS[variant]:>4}"
                f"  {v:>10.1f} {sd[(block, variant, False)]:>5.1f}  {read_bytes / v:>9.1f}{tag}"
            )

    # ---- mechanism probe: cost per COMMAND vs cost per BYTE -------------------------------------
    # RISC count and NoC-port count cannot be varied independently (firmware binds one NoC per
    # data-movement RISC), so instead hold total bytes fixed and scale the NUMBER of transactions.
    # If the read is limited by how fast a RISC can ISSUE commands, time tracks command count and
    # the 2-engine win grows as transactions shrink. If it is limited by port/wire BYTES, time is
    # roughly flat in txn_bytes and the win does not change.
    fixed_block = 8 if 8 in blocks else blocks[-1]
    mech = {}
    for txn in TXN_SWEEP:
        if page_bytes % txn:
            continue
        for variant in variants:
            run_fn = lambda v=variant, t=txn: dual_noc_read(
                a, b, variant=v, block=fixed_block, do_math=False, txn_bytes=t
            )
            value, spread = _measure_ns(device, run_fn)
            assert value is not None, f"profiler produced no data for {variant} txn={txn}"
            mech[(txn, variant)] = (value, spread)

    lines += [
        "",
        f"  [3] MECHANISM PROBE — commands vs bytes (payload-ablated, block={fixed_block}, total bytes FIXED)",
        "      txn_bytes shrinking => same bytes in proportionally MORE NoC commands.",
        "      issue-bound  => ns scales with commands, and 'vs base' GROWS as txn_bytes shrinks.",
        "      byte/port-bound => ns roughly flat, and 'vs base' stays put.",
        f"    {'txn B':>6} {'cmds':>6}  {'variant':<18}  {'ns/op':>10} {'+-%':>5}  {'read GB/s':>9}  {'vs base':>8}",
    ]
    for txn in sorted({k[0] for k in mech}, reverse=True):
        cmds = (read_bytes // txn) if txn else 0
        base = mech[(txn, BASELINE)][0]
        for variant in variants:
            v, sp = mech[(txn, variant)]
            tag = "  (base)" if variant == BASELINE else f"  {base / v:5.2f}x"
            lines.append(f"    {txn:>6} {cmds:>6}  {variant:<18}  {v:>10.1f} {sp:>5.1f}  {read_bytes / v:>9.1f}{tag}")

    lines += ["", "    variants:"] + [f"      {k:<18} {LABEL[k]}" for k in variants]
    logger.info("\n".join(lines))
