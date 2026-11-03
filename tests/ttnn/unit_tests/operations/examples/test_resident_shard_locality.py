# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Data already resident in L1: split the work by the shard spec (`local`) or ignore it (`resplit`).

See ttnn/ttnn/operations/examples/resident_shard_locality/README.md.

    # every variant on every case is a bit-exact copy of its input
    scripts/run_safe_pytest.sh --run-all \\
        tests/ttnn/unit_tests/operations/examples/test_resident_shard_locality.py::test_resident_shard_locality_correctness

    # predefined sweep -> report.md (device kernel duration per variant x case)
    scripts/run_safe_pytest.sh --run-all \\
        tests/ttnn/unit_tests/operations/examples/test_resident_shard_locality.py::test_resident_shard_locality_device_perf

    # only the wide-row / partial-width cases (RSL_SUITE=base: only the original cases), printed
    RSL_SUITE=wide RSL_REPORT= scripts/run_safe_pytest.sh --run-all \\
        tests/ttnn/unit_tests/operations/examples/test_resident_shard_locality.py::test_resident_shard_locality_device_perf

    # helps / does-not-help boundaries (asserted)
    scripts/run_safe_pytest.sh --run-all \\
        tests/ttnn/unit_tests/operations/examples/test_resident_shard_locality.py::test_resident_shard_locality_regimes
"""

import os

# In-process device profiler (all three, before the device opens) + quiet C++ logger.
os.environ.setdefault("TT_METAL_DEVICE_PROFILER", "1")
os.environ.setdefault("TT_METAL_PROFILER_MID_RUN_DUMP", "1")
os.environ.setdefault("TT_METAL_PROFILER_CPP_POST_PROCESS", "1")
os.environ.setdefault("TT_METAL_LOGGER_LEVEL", "error")

import datetime
import socket
import statistics
import subprocess
from pathlib import Path

import pytest
import torch
import ttnn
from loguru import logger

from ttnn.operations.examples.resident_shard_locality import (
    VARIANTS,
    alias_eligible,
    resident_shard_copy,
    sharded_memory_config,
    tensor_shape,
)
from ttnn.operations.examples.resident_shard_locality.program_descriptor_with_inline_kernels import (
    chunk_plan,
    elem_bytes,
    local_l1_bytes,
)

_DURATION_KEY = "DEVICE KERNEL DURATION [ns]"
_DEVICE_CSV = os.path.join(os.environ.get("TT_METAL_HOME", "."), "generated/profiler/.logs/profile_log_device.csv")
_EXAMPLE_DIR = Path(__file__).resolve().parents[5] / "ttnn/ttnn/operations/examples/resident_shard_locality"

# Cases: (id, layout, cores, shard_rows, shard_width). cores: "grid" = every Tensix core of the
# compute grid (for "block": the whole grid as rows x cols); an int = that many Tensix cores
# row-major, capped at the grid. shard_rows = sticks per Tensix core; shard_width = elements per
# stick (per stick piece for BLOCK_SHARDED); bf16, so bytes per stick = 2 x shard_width.
# An optional 6th field is the dtype ("bf16" default, or "fp32": 4 bytes per element).
_DEFAULT_CASES = [
    # HEIGHT_SHARDED, shard height not a multiple of 32: the re-split crosses shard boundaries
    ("h12x128", "height", "grid", 12, 128),
    ("h40x320", "height", "grid", 40, 320),
    ("h72x544", "height", "grid", 72, 544),
    ("h78x544", "height", "grid", 78, 544),
    # whole tile-rows per Tensix core: the re-split lands on the owners by accident; local_alias legal
    ("h32x128", "height", "grid", 32, 128),
    ("h32x1024", "height", "grid", 32, 1024),
    ("h64x544", "height", "grid", 64, 544),
    ("h96x544", "height", "grid", 96, 544),
    # BLOCK_SHARDED over the whole grid
    ("b311x320", "block", "grid", 311, 320),
    ("b23x848", "block", "grid", 23, 848),
    # sub-tile-row shards: local tilizes a whole tile-row for 1-8 valid sticks
    ("h1x1024", "height", 128, 1, 1024),
    ("h2x1024", "height", 128, 2, 1024),
    ("h4x1024", "height", 128, 4, 1024),
    ("h1x4096", "height", 128, 1, 4096),
    ("h2x4096", "height", 128, 2, 4096),
    ("h3x4096", "height", 128, 3, 4096),
    ("h1x8192", "height", 128, 1, 8192),
    ("h2x8192", "height", 128, 2, 8192),
    ("h3x8192", "height", 128, 3, 8192),
    ("h4x8192", "height", 128, 4, 8192),
    ("h8x8192", "height", 128, 8, 8192),
]

# Wide rows and partial-width BLOCK shards (field shapes): HEIGHT 127 / 160 tiles per stick on 130
# Tensix cores, BLOCK shard widths that are not whole tiles on the whole grid (K = grid columns).
# local stages one <= 32-tile column chunk at a time (`chunk_plan`), so L1 is bounded by the chunk.
_WIDE_CASES = [
    ("h24x4064", "height", "grid", 24, 4064),  # 127 tiles: 4 chunks of 32 (last 31 valid)
    ("h17x5120", "height", "grid", 17, 5120),  # 160 tiles: 5 chunks of 32
    ("b13x632f32", "block", "grid", 13, 632, "fp32"),  # 19.75 tiles fp32
    ("b221x400", "block", "grid", 221, 400),  # 12.5 tiles
    ("b23x848", "block", "grid", 23, 848),  # 26.5 tiles
    ("b16x848", "block", "grid", 16, 848),  # 26.5 tiles
]
_DTYPES = {"bf16": ttnn.bfloat16, "fp32": ttnn.float32}
_TORCH_DTYPES = {"bf16": torch.bfloat16, "fp32": torch.float32}


def _case(case):
    """(id, layout, cores, shard_rows, shard_width, dtype) with the dtype defaulted to bf16."""
    return tuple(case) + ("bf16",) * (6 - len(case))


def _custom_case():
    """CLI override (RSL_LAYOUT / RSL_CORES / RSL_SHARD_ROWS / RSL_SHARD_WIDTH), else None."""
    if "RSL_SHARD_ROWS" not in os.environ:
        return None
    layout = os.environ.get("RSL_LAYOUT", "height")
    cores = os.environ.get("RSL_CORES", "grid")
    if cores != "grid":
        cores = tuple(int(c) for c in cores.split(",")) if layout == "block" else int(cores)
    rows, width = int(os.environ["RSL_SHARD_ROWS"]), int(os.environ["RSL_SHARD_WIDTH"])
    dtype = os.environ.get("RSL_DTYPE", "bf16")
    return (f"{layout[0]}{rows}x{width}{'' if dtype == 'bf16' else dtype}", layout, cores, rows, width, dtype)


def _resolve_cores(device, layout, cores):
    grid = device.compute_with_storage_grid_size()
    if cores == "grid":
        return grid.x * grid.y if layout == "height" else (grid.y, grid.x)
    if layout == "height":
        return min(cores, grid.x * grid.y)
    return cores


def _n_cores(layout, cores):
    return cores if layout == "height" else cores[0] * cores[1]


def _make_input(device, layout, cores, shard_rows, shard_width, seed=0, dtype="bf16"):
    torch.manual_seed(seed)
    shape = tensor_shape(layout, cores, shard_rows, shard_width)
    torch_input = torch.randn(shape, dtype=torch.float32).to(_TORCH_DTYPES[dtype])
    tt_input = ttnn.from_torch(
        torch_input,
        dtype=_DTYPES[dtype],
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=sharded_memory_config(device, layout, cores, shard_rows, shard_width),
    )
    return tt_input, torch_input


def _variants_for(shard_rows, shard_width, selected=VARIANTS):
    return [v for v in selected if v != "local_alias" or alias_eligible(shard_rows, shard_width)]


def _check(tt_input, torch_input, variant, kernel_iters=1, swap_nocs=False):
    out = ttnn.to_torch(resident_shard_copy(tt_input, variant=variant, kernel_iters=kernel_iters, swap_nocs=swap_nocs))
    assert torch.equal(out, torch_input), (
        f"{variant}: output differs from input "
        f"({(out != torch_input).sum().item()} of {out.numel()} elements wrong)"
    )


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
    """{label: [ns per pass, ...]}; one untimed launch each first (JIT), then `trials` rotated rounds."""
    for run in runners.values():
        run()
    ttnn.synchronize_device(device)
    _read_kernel_ns(device)
    samples = {label: [] for label in runners}
    for _ in range(trials):
        for label, run in runners.items():
            run()
            ns = _read_kernel_ns(device)
            assert ns is not None, f"device profiler produced no duration for {label} (profiler build?)"
            samples[label].append(ns / kernel_iters)
    return samples


def _arch_label(device):
    arch = str(device.arch()).rsplit(".", 1)[-1]
    return {"WORMHOLE_B0": "WH_B0", "BLACKHOLE": "BH"}.get(arch, arch)


def _chip_freq_mhz():
    try:
        with open(_DEVICE_CSV) as f:
            header = f.readline()
    except OSError:
        return None
    for part in header.split(","):
        if "CHIP_FREQ" in part:
            return int(float(part.split(":")[1]))
    return None


def _git_sha():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=_EXAMPLE_DIR, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _units(layout, cores, shard_rows, shard_width):
    """Work units the resplit deals out: tile-rows x page columns x column chunks."""
    total_rows, width = tensor_shape(layout, cores, shard_rows, shard_width)
    chunks = chunk_plan(shard_width)[1]
    return -(-total_rows // 32) * (width // shard_width) * chunks


# ---------------------------------------------------------------------------------------
# Correctness (the only pass/fail)
# ---------------------------------------------------------------------------------------

_CORRECTNESS_IDS = ("h72x544", "h64x544", "b311x320", "b23x848", "h3x8192", "h24x4064", "b13x632f32", "b221x400")
_CORRECTNESS_CASES = [c for c in _DEFAULT_CASES if c[0] in _CORRECTNESS_IDS] + [
    c for c in _WIDE_CASES if c[0] in _CORRECTNESS_IDS and c[0] not in {d[0] for d in _DEFAULT_CASES}
]


@pytest.mark.parametrize("case", _CORRECTNESS_CASES, ids=[c[0] for c in _CORRECTNESS_CASES])
def test_resident_shard_locality_correctness(device, case):
    """Every variant is a bit-exact copy (tilize -> untilize of bf16), also with kernel_iters=2."""
    _, layout, cores, shard_rows, shard_width, dtype = _case(case)
    cores = _resolve_cores(device, layout, cores)
    tt_input, torch_input = _make_input(device, layout, cores, shard_rows, shard_width, dtype=dtype)
    for variant in _variants_for(shard_rows, shard_width):
        _check(tt_input, torch_input, variant)
        _check(tt_input, torch_input, variant, kernel_iters=2)
        _check(tt_input, torch_input, variant, swap_nocs=True)


# ---------------------------------------------------------------------------------------
# Sweep -> report.md
# ---------------------------------------------------------------------------------------


def test_resident_shard_locality_device_perf(device):
    """Correctness-gate every variant, then measure device kernel duration; perf never fails a test."""
    custom = _custom_case()
    suite = os.environ.get("RSL_SUITE", "all")
    wide_only = [c for c in _WIDE_CASES if c[0] not in {d[0] for d in _DEFAULT_CASES}]
    suites = {"all": _DEFAULT_CASES + wide_only, "base": _DEFAULT_CASES, "wide": _WIDE_CASES}
    cases = [_case(c) for c in ([custom] if custom else suites[suite])]
    selected = tuple(os.environ.get("RSL_VARIANTS", ",".join(VARIANTS)).split(","))
    trials = int(os.environ.get("RSL_TRIALS", "1"))
    kernel_iters = int(os.environ.get("RSL_KERNEL_ITERS", "1"))
    swap_nocs = os.environ.get("RSL_SWAP_NOCS", "0") == "1"

    rows_out = []
    for case_id, layout, cores, shard_rows, shard_width, dtype in cases:
        cores = _resolve_cores(device, layout, cores)
        tt_input, torch_input = _make_input(device, layout, cores, shard_rows, shard_width, dtype=dtype)
        variants = _variants_for(shard_rows, shard_width, selected)
        for variant in variants:
            _check(tt_input, torch_input, variant, kernel_iters, swap_nocs)
        runners = {
            v: (lambda v=v: resident_shard_copy(tt_input, variant=v, kernel_iters=kernel_iters, swap_nocs=swap_nocs))
            for v in variants
        }
        samples = _measure(device, runners, trials, kernel_iters)
        rows_out.append((case_id, layout, cores, shard_rows, shard_width, dtype, {v: samples[v] for v in variants}))

    grid = device.compute_with_storage_grid_size()
    freq = _chip_freq_mhz()
    header = (
        f"box={socket.gethostname()}  arch={_arch_label(device)}  grid={grid.x}x{grid.y}  "
        f"clock={freq} MHz  N={trials} ({'median' if trials > 1 else 'single pass'})  "
        f"kernel-iters={kernel_iters} ({'per-launch latency' if kernel_iters == 1 else 'steady-state'})  "
        f"date={datetime.date.today().isoformat()}  sha={_git_sha()}  "
        f"NoCs: reader {'NoC1' if swap_nocs else 'NoC0'} / writer {'NoC0' if swap_nocs else 'NoC1'}"
    )
    lines = [
        "# resident_shard_locality — resplit vs local vs local_alias",
        "",
        header,
        "",
        "Tilize -> untilize of an L1-sharded ROW_MAJOR bf16 (fp32 where marked) tensor; output sharded the same way. "
        "Sticks per Tensix core = shard height; bytes per stick = shard width x bytes per element. "
        "`resplit` deals out work units (32 sticks x one column chunk) over the same Tensix cores.",
        "",
        "| Case | Memory layout | Tensix cores | Shard [sticks, elements] | B per stick | resplit units | Method | ns | vs resplit | Correct |",
        "|---|---|---:|---|---:|---:|---|---:|---:|---|",
    ]
    for case_id, layout, cores, shard_rows, shard_width, dtype, results in rows_out:
        base = statistics.median(results["resplit"]) if "resplit" in results else None
        n = _n_cores(layout, cores)
        stick_bytes = shard_width * elem_bytes(_DTYPES[dtype])
        memory_layout = "HEIGHT_SHARDED" if layout == "height" else f"BLOCK_SHARDED {cores[0]}x{cores[1]}"
        for variant in VARIANTS:
            if variant not in results:
                if variant in selected and variant == "local_alias":
                    lines.append(
                        f"| {case_id} | {memory_layout} | {n} | [{shard_rows}, {shard_width}] | {stick_bytes} | "
                        f"{_units(layout, cores, shard_rows, shard_width)} | local_alias | n/a (shard not whole tiles) | | |"
                    )
                continue
            med = statistics.median(results[variant])
            vs = f"{base / med:.2f}x" if base else ""
            lines.append(
                f"| {case_id} | {memory_layout} | {n} | [{shard_rows}, {shard_width}] | {stick_bytes} | "
                f"{_units(layout, cores, shard_rows, shard_width)} | {variant} | {med:.0f} | {vs} | yes |"
            )
    lines += _l1_table(rows_out)
    plain = selected == VARIANTS and trials == 1 and kernel_iters == 1 and not swap_nocs
    default_run = custom is None and suite == "all" and plain
    if default_run:
        lines += _mechanism_table(device)
    report = "\n".join(lines) + "\n"
    logger.info("\n" + report)
    path = os.environ.get("RSL_REPORT")
    if path is None and default_run:
        path = str(_EXAMPLE_DIR / "report.md")
    if path:
        Path(path).write_text(report)
        logger.info(f"report written to {path}")


def _l1_table(rows_out):
    """Per-Tensix-core CB bytes of `local` for the cases wider than one chunk or not whole tiles wide."""
    lines = [
        "",
        "## L1 per Tensix core: `local` staging one column chunk vs staging the whole row",
        "",
        "CBs only (input staging, tiled, output staging; 2 chunks deep each, tile-sized pages); the input "
        "and output shards come on top. Whole-row staging = the same 3 CBs, 2 deep, one chunk = the whole padded width.",
        "",
        "| Case | dtype | Width tiles | Chunks x tiles | local CB KB | whole-row CB KB | Shards in+out KB |",
        "|---|---|---:|---|---:|---:|---:|",
    ]
    shown = False
    for case_id, _, _, shard_rows, shard_width, dtype, _ in rows_out:
        ct, chunks = chunk_plan(shard_width)
        if chunks == 1 and shard_width % 32 == 0:
            continue
        shown = True
        tt_dtype = _DTYPES[dtype]
        shards_kb = 2 * shard_rows * shard_width * elem_bytes(tt_dtype) / 1024
        lines.append(
            f"| {case_id} | {dtype} | {-(-shard_width // 32)} | {chunks} x {ct} | "
            f"{local_l1_bytes(shard_width, tt_dtype) // 1024} | "
            f"{local_l1_bytes(shard_width, tt_dtype, chunked=False) // 1024} | {shards_kb:.0f} |"
        )
    return lines if shown else []


# Mechanism: the same [72, 544] HEIGHT_SHARDED shards on a growing number of Tensix cores (row-major),
# with the default NoCs and with the reader/writer NoCs swapped. local never leaves its Tensix core.
_MECHANISM_CORES = (1, 2, 13, 26, 65, 130)


def _mechanism_table(device):
    grid = device.compute_with_storage_grid_size()
    lines = [
        "",
        "## Mechanism: resplit cost vs Tensix cores and NoC direction ([72, 544] HEIGHT_SHARDED shards)",
        "",
        "Same shards, more Tensix cores in row-major order. `swapped` = reader on NoC1, writer on NoC0.",
        "",
        "| Tensix cores | resplit ns | resplit swapped ns | local ns | resplit / local | resplit swapped / local |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for n in (c for c in _MECHANISM_CORES if c <= grid.x * grid.y):
        tt_input, torch_input = _make_input(device, "height", n, 72, 544)
        runners = {
            "resplit": lambda: resident_shard_copy(tt_input, variant="resplit"),
            "swapped": lambda: resident_shard_copy(tt_input, variant="resplit", swap_nocs=True),
            "local": lambda: resident_shard_copy(tt_input, variant="local"),
        }
        _check(tt_input, torch_input, "resplit", swap_nocs=True)
        med = {k: statistics.median(v) for k, v in _measure(device, runners, 3, 1).items()}
        lines.append(
            f"| {n} | {med['resplit']:.0f} | {med['swapped']:.0f} | {med['local']:.0f} | "
            f"{med['resplit'] / med['local']:.2f}x | {med['swapped'] / med['local']:.2f}x |"
        )
    return lines


# ---------------------------------------------------------------------------------------
# Regimes (asserted): helps -> speedup >= max(1.05, 1 + half the measured gain);
# does_not_help -> speedup <= 1.05. Speedup = resplit ns / local ns (or local / local_alias).
# ---------------------------------------------------------------------------------------

_REGIMES = [
    # (id, layout, cores, shard_rows, shard_width, dtype, baseline, lever, kind, threshold)  BH p150b, 13x10
    ("helps_height_72x544", "height", "grid", 72, 544, "bf16", "resplit", "local", "helps", 5.9),  # 10.9x
    ("helps_block_311x320", "block", "grid", 311, 320, "bf16", "resplit", "local", "helps", 3.1),  # 5.3x
    # 0.6-0.75x: <= 2 sticks of >= 8 KB per Tensix core
    ("does_not_help_height_1x8192", "height", 128, 1, 8192, "bf16", "resplit", "local", "does_not_help", 1.05),
    ("helps_alias_32x1024", "height", "grid", 32, 1024, "bf16", "local", "local_alias", "helps", 1.27),  # 1.55x
    # wide rows (127 / 160 tiles per stick), W chunked into <= 32-tile column chunks
    ("helps_height_wide_24x4064", "height", "grid", 24, 4064, "bf16", "resplit", "local", "helps", 4.4),  # 7.8x
    ("helps_height_wide_17x5120", "height", "grid", 17, 5120, "bf16", "resplit", "local", "helps", 6.1),  # 11.2x
    # partial-width BLOCK shards (shard width not a multiple of 32), K = grid columns
    ("helps_block_13x632_fp32", "block", "grid", 13, 632, "fp32", "resplit", "local", "helps", 1.56),  # 2.1x
    ("helps_block_221x400", "block", "grid", 221, 400, "bf16", "resplit", "local", "helps", 3.35),  # 5.7x
    ("helps_block_23x848", "block", "grid", 23, 848, "bf16", "resplit", "local", "helps", 3.6),  # 6.3x
    ("helps_block_16x848", "block", "grid", 16, 848, "bf16", "resplit", "local", "helps", 2.6),  # 4.2x
]


@pytest.mark.parametrize("regime", _REGIMES, ids=[r[0] for r in _REGIMES])
def test_resident_shard_locality_regimes(device, regime):
    """Interleaved A/B of baseline vs lever, median of 5 rounds; both correctness-checked first."""
    rid, layout, cores, shard_rows, shard_width, dtype, baseline, lever, kind, threshold = regime
    cores = _resolve_cores(device, layout, cores)
    tt_input, torch_input = _make_input(device, layout, cores, shard_rows, shard_width, dtype=dtype)
    for variant in (baseline, lever):
        _check(tt_input, torch_input, variant)
    runners = {v: (lambda v=v: resident_shard_copy(tt_input, variant=v)) for v in (baseline, lever)}
    samples = _measure(device, runners, 5, 1)
    b, l = statistics.median(samples[baseline]), statistics.median(samples[lever])
    speedup = b / l
    logger.info(
        f"REGIME {rid} [{_arch_label(device)}] {baseline}={b:.0f}ns {lever}={l:.0f}ns speedup={speedup:.2f}x "
        f"(threshold {'>=' if kind == 'helps' else '<='} {threshold})"
    )
    if kind == "helps":
        assert speedup >= threshold, f"{rid}: expected >= {threshold}x, got {speedup:.2f}x"
    else:
        assert speedup <= threshold, f"{rid}: expected <= {threshold}x, got {speedup:.2f}x"
