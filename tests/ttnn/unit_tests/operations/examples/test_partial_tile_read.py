# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Correctness and in-process device profiling: whole-tile vs row-0-only reads of a row-broadcast operand.

    scripts/run_safe_pytest.sh --run-all tests/ttnn/unit_tests/operations/examples/test_partial_tile_read.py

See ttnn/ttnn/operations/examples/partial_tile_read/README.md.
"""

import os

os.environ.setdefault("TT_METAL_DEVICE_PROFILER", "1")
os.environ.setdefault("TT_METAL_PROFILER_MID_RUN_DUMP", "1")
os.environ.setdefault("TT_METAL_PROFILER_CPP_POST_PROCESS", "1")
os.environ.setdefault("TT_METAL_LOGGER_LEVEL", "error")

import socket
import statistics
from pathlib import Path

import pytest
import torch
import ttnn
from loguru import logger

from ttnn.operations.examples.partial_tile_read import (
    ABLATION,
    BASELINE,
    DEFAULT_BLOCK,
    LEVER,
    VARIANTS,
    bytes_per_tile,
    create_sharded_memory_config,
    effective_block,
    partial_tile_read,
    segment_bytes,
)

TILE = 32
_DURATION_KEY = "DEVICE KERNEL DURATION [ns]"
_DEVICE_CSV = Path(os.environ.get("TT_METAL_HOME", ".")) / "generated/profiler/.logs/profile_log_device.csv"


def _grid_cores(device):
    grid = device.compute_with_storage_grid_size()
    return grid.x * grid.y


def _cores(device, spec):
    return _grid_cores(device) if spec == "full" else min(int(spec), _grid_cores(device))


def _make_inputs(device, num_cores, rows, width, seed):
    """x and v hold k/8 with |k| <= 16, so x + v is exact in bf16 and the check can be exact.
    Rows 1-31 of v are large, distinct values: if anything but row 0 were used the check would fail."""
    g = torch.Generator().manual_seed(seed)
    x = torch.randint(-16, 17, (num_cores * rows * TILE, width), generator=g).float() / 8
    v = 1000 + torch.randint(0, 64, (TILE, width), generator=g).float()
    v[0] = torch.randint(-16, 17, (width,), generator=g).float() / 8
    expected = x + v[0:1]
    tt_x = ttnn.from_torch(
        x,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=create_sharded_memory_config(device, num_cores, rows, width),
    )
    tt_v = ttnn.from_torch(
        v, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    return tt_x, tt_v, expected


def _run(tt_x, tt_v, variant, num_cores, rows, block=DEFAULT_BLOCK, kernel_iters=1):
    return partial_tile_read(
        tt_x, tt_v, variant=variant, num_cores=num_cores, rows=rows, block=block, kernel_iters=kernel_iters
    )


def _check(device, num_cores, rows, width, block=DEFAULT_BLOCK, kernel_iters=1, seed=0):
    """Every variant, each on its own fresh inputs, baseline first: a lever that skipped its reads
    would find the baseline's stale whole tiles (a different vector) in the CB slots and fail."""
    for i, variant in enumerate(VARIANTS):
        tt_x, tt_v, expected = _make_inputs(device, num_cores, rows, width, seed + 101 * i)
        tt_out = _run(tt_x, tt_v, variant, num_cores, rows, block, kernel_iters)
        out = ttnn.to_torch(tt_out).float()
        _free(tt_out, tt_x, tt_v)  # a 1 MB-per-Tensix-core case leaves no room for two live copies
        torch.testing.assert_close(out, expected, rtol=0, atol=0, msg=lambda m: f"{variant}: {m}")


def _free(*tensors):
    for t in tensors:
        ttnn.deallocate(t)


# (cores, rows, W, block, kernel_iters): one tile; a block that does not divide Wt (3 tiles -> block 1);
# several tile-rows per Tensix core; the full grid at the longest vector; in-kernel repeats.
_CORRECTNESS_CASES = [
    ("1", 1, 32, 8, 1),
    ("3", 2, 96, 8, 1),
    ("8", 4, 1024, 8, 1),
    ("full", 1, 8192, 8, 1),
    ("full", 1, 1024, 4, 3),
]


@pytest.mark.parametrize("cores, rows, width, block, kernel_iters", _CORRECTNESS_CASES)
def test_partial_tile_read_correctness(device, cores, rows, width, block, kernel_iters):
    num_cores = _cores(device, cores)
    _check(device, num_cores, rows, width, block, kernel_iters)
    # The check is sensitive to stale CB contents: the no-read ablation, run on fresh inputs right
    # after the lever, finds the previous vector in its CB slots and must NOT match.
    tt_x, tt_v, expected = _make_inputs(device, num_cores, rows, width, seed=999)
    tt_out = _run(tt_x, tt_v, ABLATION, num_cores, rows, block, kernel_iters)
    out = ttnn.to_torch(tt_out).float()
    _free(tt_out, tt_x, tt_v)
    assert not torch.equal(out, expected), "stale CB contents reproduced the expected output"


# ---------------------------------------------------------------------------
# Measurement
# ---------------------------------------------------------------------------
def _read_kernel_ns(device):
    ttnn.ReadDeviceProfiler(device)
    per_chip = ttnn.get_latest_programs_perf_data() or {}
    total, found = 0.0, False
    for programs in per_chip.values():
        for program in programs:
            entry = (getattr(program, "program_analyses_results", None) or {}).get(_DURATION_KEY)
            if entry is not None:
                total += float(entry.duration)
                found = True
    return total if found else None


def _measure(device, runners, trials, kernel_iters):
    """{label: [ns per stream, ...]}; methods interleaved within each trial so a clock change mid-run
    moves every method together. The first (JIT) launch of each runner is discarded."""
    for run in runners.values():
        run()
    ttnn.synchronize_device(device)
    _read_kernel_ns(device)
    samples = {label: [] for label in runners}
    for _ in range(trials):
        for label, run in runners.items():
            run()
            duration = _read_kernel_ns(device)
            assert duration is not None, f"device profiler produced no duration for {label}"
            samples[label].append(duration / kernel_iters)
    return samples


def _chip_freq_mhz():
    try:
        with open(_DEVICE_CSV) as f:
            header = f.readline()
        for part in header.split(","):
            if "CHIP_FREQ" in part:
                return float(part.split(":")[1])
    except (OSError, ValueError, IndexError):
        pass
    return None


def _arch_label(device):
    arch = str(device.arch()).rsplit(".", 1)[-1]
    return {"WORMHOLE_B0": "WH_B0", "BLACKHOLE": "BH"}.get(arch, arch)


def _env_list(name, default):
    return [s for s in os.environ.get(name, default).split(",") if s]


def _perf_cases():
    """(regime label, cores spec, rows, W). Override with PTR_CORES / PTR_WIDTHS / PTR_ROWS."""
    if any(k in os.environ for k in ("PTR_CORES", "PTR_WIDTHS", "PTR_ROWS")):
        return [
            ("custom", c, int(r), int(w))
            for c in _env_list("PTR_CORES", "full")
            for r in _env_list("PTR_ROWS", "1")
            for w in _env_list("PTR_WIDTHS", "1024,8192")
        ]
    cases = [("one vector per core", c, 1, w) for c in ("1", "8", "32", "64", "full") for w in (1024, 2048, 4096, 8192)]
    cases += [("one tile", c, 1, 32) for c in ("1", "8", "full")]
    cases += [("reused over tile-rows", "32", r, 512) for r in (1, 4, 8, 16)]
    return cases


def test_partial_tile_read_device_perf(device):
    """Correctness-gate both methods, then measure; perf never fails a test."""
    trials = int(os.environ.get("PTR_TRIALS", "3"))
    kernel_iters = int(os.environ.get("PTR_KERNEL_ITERS", "1"))
    block = int(os.environ.get("PTR_BLOCK", str(DEFAULT_BLOCK)))
    selected = _env_list("PTR_VARIANTS", ",".join(VARIANTS + (ABLATION,)))
    labels = [v for v in VARIANTS + (ABLATION,) if v in selected]
    seg = segment_bytes(device)
    rows_out = []
    for regime, cores_spec, rows, width in _perf_cases():
        cores = _cores(device, cores_spec)
        _check(device, cores, rows, width, block, kernel_iters)
        tt_x, tt_v, _ = _make_inputs(device, cores, rows, width, seed=7)
        tile_bytes = tt_v.buffer_aligned_page_size()
        runners = {v: (lambda v=v: _run(tt_x, tt_v, v, cores, rows, block, kernel_iters)) for v in labels}
        samples = _measure(device, runners, trials, kernel_iters)
        _free(tt_x, tt_v)
        med = {v: statistics.median(s) for v, s in samples.items()}
        wt = width // TILE
        for v in labels:
            per_core = wt * bytes_per_tile(v, device, tile_bytes)
            rows_out.append(
                dict(
                    regime=regime,
                    cores=cores,
                    rows=rows,
                    width=width,
                    variant=v,
                    bytes_per_core=per_core,
                    ns=med[v],
                    spread=(min(samples[v]), max(samples[v])),
                    speedup=(med[BASELINE] / med[v]) if BASELINE in med else float("nan"),
                    gbps=cores * per_core / med[v],
                )
            )

    freq = _chip_freq_mhz()
    lines = [
        "# partial_tile_read — whole-tile vs row-0-only reads of a row-broadcast operand",
        "",
        f"box={socket.gethostname()}  arch={_arch_label(device)}  CHIP_FREQ={freq:.0f} MHz  "
        f"N={trials} (median, methods interleaved)  kernel-iters={kernel_iters}  block={block} tiles/barrier  "
        f"dtype=bf16  tile=2048 B  row-0 segment={seg} B x 2 per tile  "
        f"grid={_grid_cores(device)} Tensix cores (row-major placement)",
        "",
        "Bytes = DRAM bytes each Tensix core reads per stream; GB/s = all Tensix cores' bytes / kernel ns.",
        "`no_read` is a stage ablation (reader pushes pages without reading): everything but the vector read.",
        "",
        "| Regime | Tensix cores | Tile-rows per core | W (elements) | Method | Bytes read per core | ns | min..max ns | vs whole_tile | GB/s |",
        "|---|---:|---:|---:|---|---:|---:|---:|---:|---:|",
    ]
    for r in rows_out:
        lines.append(
            f"| {r['regime']} | {r['cores']} | {r['rows']} | {r['width']} | {r['variant']} | {r['bytes_per_core']} | "
            f"{r['ns']:.0f} | {r['spread'][0]:.0f}..{r['spread'][1]:.0f} | {r['speedup']:.2f}x | {r['gbps']:.1f} |"
        )
    report = "\n".join(lines) + "\n"
    logger.info("\n" + report)
    if path := os.environ.get("PTR_REPORT"):
        Path(path).write_text(report)


# ---------------------------------------------------------------------------
# When it helps / when it does not (BH p150b, 130 Tensix cores, block = 8 tiles per barrier):
#   helps          the whole-tile reads of all Tensix cores together hit the DRAM limit: >= 32 Tensix
#                  cores reading the vector. Full grid W=8192 2.03x; 32 Tensix cores W=2048 2.40x.
#   does not help  few Tensix cores: each reader is latency-bound (8 tiles in flight per barrier), so
#                  bytes per tile do not set the time. 1 Tensix core W=8192 0.97x; 8 Tensix cores 1.00x.
# helps -> speedup >= max(1.05, 1 + half the measured gain); does_not_help -> speedup <= 1.05.
# Interleaved A/B, median of 5 trials. row0_span (one read per tile) is measured and logged too.
# WH B0: not measured. (id, cores, rows, W, kind, threshold)
# ---------------------------------------------------------------------------
_REGIMES = [
    ("helps_full_grid_W8192", "full", 1, 8192, "helps", 1.5),  # BH p150b 2.03x
    ("helps_32cores_W2048", "32", 1, 2048, "helps", 1.7),  # BH p150b 2.40x
    ("does_not_help_1core_W8192", "1", 1, 8192, "does_not_help", 1.05),  # BH p150b 0.97x
    ("does_not_help_8cores_W8192", "8", 1, 8192, "does_not_help", 1.05),  # BH p150b 1.00x
]


@pytest.mark.parametrize("cores, rows, width, kind, threshold", [r[1:] for r in _REGIMES], ids=[r[0] for r in _REGIMES])
def test_partial_tile_read_regimes(device, cores, rows, width, kind, threshold):
    num_cores = _cores(device, cores)
    _check(device, num_cores, rows, width)
    tt_x, tt_v, _ = _make_inputs(device, num_cores, rows, width, seed=7)
    runners = {v: (lambda v=v: _run(tt_x, tt_v, v, num_cores, rows)) for v in VARIANTS}
    samples = _measure(device, runners, 5, 1)
    _free(tt_x, tt_v)
    med = {v: statistics.median(s) for v, s in samples.items()}
    speedup = med[BASELINE] / med[LEVER]
    logger.info(
        f"REGIME {kind} cores={num_cores} rows={rows} W={width} "
        + " ".join(f"{v}={med[v]:.0f}ns" for v in VARIANTS)
        + f" speedup={speedup:.3f}x span_speedup={med[BASELINE] / med['row0_span']:.3f}x "
        f"arch={_arch_label(device)} CHIP_FREQ={_chip_freq_mhz()}"
    )
    if kind == "helps":
        assert speedup >= threshold, f"expected >= {threshold}x, got {speedup:.3f}x"
    else:
        assert speedup <= threshold, f"expected <= {threshold}x, got {speedup:.3f}x"
