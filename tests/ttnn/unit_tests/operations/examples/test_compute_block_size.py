# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Correctness + single-core device profiling for the compute-block-size example.

The op is `out = (A + B) @ C` built from five row-parallel compute-helper phases (tilize A,
tilize B, add, matmul, untilize). Every variant does the identical math and identical total work;
they differ only in how many tile-rows each helper call processes per pass (block_rows). Correctness
is the only pass/fail — every variant must match torch. Perf is measured (DEVICE KERNEL DURATION
[ns], in-process via ReadDeviceProfiler) and reported, never asserted.
"""

import os

os.environ.setdefault("TT_METAL_DEVICE_PROFILER", "1")
os.environ.setdefault("TT_METAL_PROFILER_MID_RUN_DUMP", "1")
os.environ.setdefault("TT_METAL_PROFILER_CPP_POST_PROCESS", "1")
# Keep the console signal-to-noise high: the profiler logs a per-read duration histogram at INFO,
# which — over trials x variants x reconfig-modes reads — buries the report. Silence C++ INFO/WARN
# (device boot, JIT stats, per-read histograms); the report is Python loguru, so it survives. The
# measurement is unaffected (durations come from program_analyses_results, not the log). Override
# with TT_METAL_LOGGER_LEVEL=info to see everything.
os.environ.setdefault("TT_METAL_LOGGER_LEVEL", "error")

import socket
import statistics
from pathlib import Path

import torch
import ttnn
from loguru import logger

from ttnn.operations.examples.compute_block_size import (
    BASELINE,
    VARIANTS,
    block_rows_for,
    create_sharded_memory_config,
    run_op,
    variant_is_valid,
)

TILE = 32
_DURATION_KEY = "DEVICE KERNEL DURATION [ns]"


# =============================================================================
# Inputs + golden (quantize to bf16 first, so tolerance covers only op-internal error)
# =============================================================================
def _quant(t):
    return t.to(torch.bfloat16).to(torch.float32)


def _make_case(device, m_tiles, k_tiles, n_tiles, seed=7):
    torch.manual_seed(seed)
    m, k, n = m_tiles * TILE, k_tiles * TILE, n_tiles * TILE
    a = torch.rand(m, k) * 2 - 1  # [-1, 1]
    b = torch.rand(m, k) * 2 - 1
    c = torch.rand(k, n) * 2 - 1
    expected = (_quant(a) + _quant(b)) @ _quant(c)

    a_dev = ttnn.from_torch(
        a.to(torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=create_sharded_memory_config((m, k)),
    )
    b_dev = ttnn.from_torch(
        b.to(torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=create_sharded_memory_config((m, k)),
    )
    c_dev = ttnn.from_torch(
        c.to(torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=create_sharded_memory_config((k, n)),
    )
    return [a_dev, b_dev, c_dev], expected


def _pcc(actual, expected):
    a = actual.flatten().to(torch.float64)
    e = expected.flatten().to(torch.float64)
    return torch.corrcoef(torch.stack([a, e]))[0, 1].item()


def _check(output, expected, label, min_pcc=0.99):
    actual = ttnn.to_torch(output).to(torch.float32)
    pcc = _pcc(actual, expected)
    assert pcc >= min_pcc, f"{label}: PCC {pcc:.5f} < {min_pcc}"
    return pcc


# =============================================================================
# In-process device-kernel timing (validated pattern — do not reinvent)
# =============================================================================
def _read_kernel_ns(device):
    ttnn.ReadDeviceProfiler(device)
    total, found = 0.0, False
    for programs in (ttnn.get_latest_programs_perf_data() or {}).values():
        for program in programs:
            entry = (getattr(program, "program_analyses_results", None) or {}).get(_DURATION_KEY)
            if entry is not None:
                total += float(entry.duration)
                found = True
    return total if found else None


def _measure(device, runners, trials, kernel_iters):
    for run in runners.values():
        run()
    ttnn.synchronize_device(device)
    _read_kernel_ns(device)  # discard warmup window
    samples = {name: [] for name in runners}
    for trial in range(trials + 1):
        for name, run in runners.items():
            run()
            duration = _read_kernel_ns(device)
            assert duration is not None, f"no profiler data for {name}"
            if trial:  # discard first timed pass
                samples[name].append(duration / kernel_iters)
    return samples


# =============================================================================
# Config knobs (env-driven so __main__ can set them)
# =============================================================================
def _int(name, default):
    return int(os.environ.get(name, default))


def _arch_label(device):
    if name := os.environ.get("ARCH_NAME"):
        return name
    a = str(device.arch()).rsplit(".", 1)[-1]  # "Arch.WORMHOLE_B0" -> "WORMHOLE_B0"
    return {"WORMHOLE_B0": "WH_B0", "BLACKHOLE": "BH", "GRAYSKULL": "GS"}.get(a, a)


def _selected_variants(m_tiles):
    sel = os.environ.get("CBS_VARIANTS")
    chosen = tuple(sel.split(",")) if sel else VARIANTS
    unknown = set(chosen) - set(VARIANTS)
    if unknown:
        raise ValueError(f"unknown CBS_VARIANTS: {sorted(unknown)}; valid: {VARIANTS}")
    valid = tuple(v for v in chosen if variant_is_valid(v, m_tiles))
    skipped = tuple(v for v in chosen if not variant_is_valid(v, m_tiles))
    if skipped:
        logger.info(f"skipping variants not dividing M_tiles={m_tiles}: {skipped}")
    return valid


# =============================================================================
# Tests
# =============================================================================
def test_compute_block_size_correctness(device):
    for m_tiles, k_tiles, n_tiles in ((8, 4, 4), (4, 2, 8)):
        inputs, expected = _make_case(device, m_tiles, k_tiles, n_tiles)
        for variant in _selected_variants(m_tiles):
            br = block_rows_for(variant, m_tiles)
            out = run_op(inputs, block_rows=br, kernel_iters=2)
            pcc = _check(out, expected, f"{variant} (block_rows={br}) M={m_tiles} K={k_tiles} N={n_tiles}")
            logger.info(f"{variant:12s} block_rows={br} shape=({m_tiles},{k_tiles},{n_tiles})t  PCC={pcc:.5f}")


def test_compute_block_size_device_perf(device):
    m_tiles = _int("CBS_M_TILES", "8")
    k_tiles = _int("CBS_K_TILES", "4")
    n_tiles = _int("CBS_N_TILES", "4")
    trials = _int("CBS_TRIALS", "5")
    kernel_iters = _int("CBS_KERNEL_ITERS", "100")

    variants = _selected_variants(m_tiles)
    inputs, expected = _make_case(device, m_tiles, k_tiles, n_tiles)

    # correctness gate before timing
    pccs = {}
    for variant in variants:
        br = block_rows_for(variant, m_tiles)
        out = run_op(inputs, block_rows=br, kernel_iters=1)
        pccs[variant] = _check(out, expected, f"{variant} (block_rows={br})")

    runners = {
        variant: (lambda br=block_rows_for(variant, m_tiles): run_op(inputs, block_rows=br, kernel_iters=kernel_iters))
        for variant in variants
    }
    samples = _measure(device, runners, trials, kernel_iters)

    report = _format_report(
        samples,
        pccs,
        m_tiles,
        k_tiles,
        n_tiles,
        box=socket.gethostname(),
        arch=_arch_label(device),
        trials=trials,
        kernel_iters=kernel_iters,
    )
    logger.info("\n" + report)
    if report_path := os.environ.get("CBS_REPORT"):
        Path(report_path).write_text(report)


# =============================================================================
# Reconfig ablation (experiment): per-phase data-format reconfig ON (helper default) vs OFF.
# Every CB is bf16, so the format never changes through the op and the reconfig is wasted MMIO.
# The inits stay on in both modes (each phase is still a different op). Measures the reconfig cost
# and how it interacts with block size. Correctness gated for BOTH modes.
# =============================================================================
def test_compute_block_size_reconfig_ablation(device):
    m_tiles = _int("CBS_M_TILES", "8")
    k_tiles = _int("CBS_K_TILES", "4")
    n_tiles = _int("CBS_N_TILES", "4")
    trials = _int("CBS_TRIALS", "5")
    kernel_iters = _int("CBS_KERNEL_ITERS", "100")

    variants = _selected_variants(m_tiles)
    inputs, expected = _make_case(device, m_tiles, k_tiles, n_tiles)

    # correctness gate: both reconfig modes must match torch
    pccs = {}  # (variant, reconfig) -> pcc
    for variant in variants:
        br = block_rows_for(variant, m_tiles)
        for reconfig in (True, False):
            out = run_op(inputs, block_rows=br, kernel_iters=1, reconfig=reconfig)
            pccs[(variant, reconfig)] = _check(out, expected, f"{variant} (block_rows={br}, reconfig={reconfig})")

    runners = {}
    for variant in variants:
        br = block_rows_for(variant, m_tiles)
        for reconfig in (True, False):
            runners[(variant, reconfig)] = lambda br=br, reconfig=reconfig: run_op(
                inputs, block_rows=br, kernel_iters=kernel_iters, reconfig=reconfig
            )
    samples = _measure(device, runners, trials, kernel_iters)

    report = _format_ablation(
        samples,
        pccs,
        variants,
        m_tiles,
        k_tiles,
        n_tiles,
        box=socket.gethostname(),
        arch=_arch_label(device),
        trials=trials,
        kernel_iters=kernel_iters,
    )
    logger.info("\n" + report)
    if report_path := os.environ.get("CBS_ABLATION_REPORT"):
        Path(report_path).write_text(report)


def _format_ablation(samples, pccs, variants, m_tiles, k_tiles, n_tiles, *, box, arch, trials, kernel_iters):
    def med(variant, reconfig):
        return statistics.median(samples[(variant, reconfig)])

    lines = [
        "# Compute block size — reconfig ON vs OFF ablation (single core)",
        "",
        f"box={box}  arch={arch}  cores=1  placement=single-core sharded-L1  "
        f"N={trials} (median)  kernel-iters={kernel_iters} (steady-state)",
        f"problem: M={m_tiles * TILE} K={k_tiles * TILE} N={n_tiles * TILE}  "
        f"(M_tiles={m_tiles}, K_tiles={k_tiles}, N_tiles={n_tiles})  dtype=bf16  fidelity=HiFi2 fp32_dest_acc",
        "",
        "Experiment: helpers always init (each phase is a different op) but the per-phase data-format "
        "reconfig is turned OFF, since every CB is bf16 and the format never changes through the op. "
        "`reconfig off` = the same run with all helper reconfigs disabled. Correctness gated for both.",
        "",
        "| Variant | block_rows | num_blocks | reconfig ON ns | reconfig OFF ns | OFF speedup | PCC on/off |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for variant in variants:
        br = block_rows_for(variant, m_tiles)
        on, off = med(variant, True), med(variant, False)
        lines.append(
            f"| {variant} | {br} | {m_tiles // br} | {on:.1f} | {off:.1f} | {on / off:.2f}x | "
            f"{pccs[(variant, True)]:.5f} / {pccs[(variant, False)]:.5f} |"
        )
    # Headline: the two extremes.
    base_on = med(BASELINE, True)
    if "one_block" in variants:
        best_off = med("one_block", False)
        lines += [
            "",
            f"Span: slowest (per_tile_row, reconfig ON) {base_on:.0f} ns  →  "
            f"fastest (one_block, reconfig OFF) {best_off:.0f} ns  =  {base_on / best_off:.2f}x combined.",
        ]
    return "\n".join(lines) + "\n"


def _format_report(samples, pccs, m_tiles, k_tiles, n_tiles, *, box, arch, trials, kernel_iters):
    base = statistics.median(samples[BASELINE]) if BASELINE in samples else None
    lines = [
        "# Compute block size — (A + B) @ C, single-core report",
        "",
        f"box={box}  arch={arch}  cores=1  placement=single-core sharded-L1  "
        f"N={trials} (median)  kernel-iters={kernel_iters} (steady-state)",
        f"problem: M={m_tiles * TILE} K={k_tiles * TILE} N={n_tiles * TILE}  "
        f"(M_tiles={m_tiles}, K_tiles={k_tiles}, N_tiles={n_tiles})  dtype=bf16  fidelity=HiFi2 fp32_dest_acc",
        "",
        "Metric: DEVICE KERNEL DURATION [ns] per (A+B)@C evaluation. "
        f"Speedup = {BASELINE} / variant. Correctness gate: PCC vs torch.",
        "",
        "| Variant | block_rows | num_blocks | Median ns | Std/med | Speedup | PCC |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for variant, values in samples.items():
        br = block_rows_for(variant, m_tiles)
        median = statistics.median(values)
        std = statistics.pstdev(values) if len(values) > 1 else 0.0
        speedup = f"{base / median:.2f}x" if base else "—"
        lines.append(
            f"| {variant} | {br} | {m_tiles // br} | {median:.1f} | "
            f"{std / median * 100:.1f}% | {speedup} | {pccs.get(variant, float('nan')):.5f} |"
        )
    return "\n".join(lines) + "\n"
