# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Data already resident in L1: split the work by the shard spec (`local`) or ignore it (`resplit`).

See ttnn/ttnn/operations/examples/resident_shard_locality/README.md.

    # every variant on every case is a bit-exact copy of its input
    scripts/run_safe_pytest.sh --run-all \\
        tests/ttnn/unit_tests/operations/examples/test_resident_shard_locality.py::test_resident_shard_locality_correctness

    # predefined sweep (device kernel duration per variant x case): a measurement sweep, skipped unless
    # RSL_SWEEPS=1 (or EXAMPLES_SWEEPS=1); printed, and written to a file only when RSL_REPORT names one
    RSL_SWEEPS=1 RSL_REPORT=ttnn/ttnn/operations/examples/resident_shard_locality/report.md \\
        scripts/run_safe_pytest.sh --run-all \\
        tests/ttnn/unit_tests/operations/examples/test_resident_shard_locality.py::test_resident_shard_locality_device_perf

    # only the wide-row / partial-width cases (RSL_SUITE=base: only the original cases), printed
    RSL_SWEEPS=1 RSL_SUITE=wide scripts/run_safe_pytest.sh --run-all \\
        tests/ttnn/unit_tests/operations/examples/test_resident_shard_locality.py::test_resident_shard_locality_device_perf

    # helps / does-not-help boundaries (asserted on Blackhole, logged elsewhere)
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

from tests.ttnn.unit_tests.operations.examples.perf_test_utils import (
    arch_label,
    check_regime,
    chip_freq_mhz,
    grid_xy,
    interleaved_ns,
    require_cores,
    require_grid,
    sweep_gate,
)
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

# Cases: (id, memory_layout, cores, shard_rows, shard_width). memory_layout: "height" =
# TensorMemoryLayout::HEIGHT_SHARDED, "block" = TensorMemoryLayout::BLOCK_SHARDED. cores: "grid" = every
# Tensix core of the compute grid (for "block": the whole grid as rows x cols, so K = grid columns); an
# int = exactly that many Tensix cores row-major (a grid with fewer skips the case). shard_rows = sticks
# per Tensix core; shard_width = elements per stick (per stick piece for BLOCK_SHARDED); bf16, so bytes
# per stick = 2 x shard_width. An optional 6th field is the dtype ("bf16" default, or "fp32": 4 bytes
# per element).
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
    """(id, memory_layout, cores, shard_rows, shard_width, dtype) with the dtype defaulted to bf16."""
    return tuple(case) + ("bf16",) * (6 - len(case))


def _custom_case():
    """CLI override (RSL_MEMORY_LAYOUT / RSL_CORES / RSL_SHARD_ROWS / RSL_SHARD_WIDTH), else None."""
    if "RSL_SHARD_ROWS" not in os.environ:
        return None
    memory_layout = os.environ.get("RSL_MEMORY_LAYOUT", "height")
    cores = os.environ.get("RSL_CORES", "grid")
    if cores != "grid":
        cores = tuple(int(c) for c in cores.split(",")) if memory_layout == "block" else int(cores)
    rows, width = int(os.environ["RSL_SHARD_ROWS"]), int(os.environ["RSL_SHARD_WIDTH"])
    dtype = os.environ.get("RSL_DTYPE", "bf16")
    return (
        f"{memory_layout[0]}{rows}x{width}{'' if dtype == 'bf16' else dtype}",
        memory_layout,
        cores,
        rows,
        width,
        dtype,
    )


def _fits(device, memory_layout, cores):
    """True when the case's shard grid fits this compute grid ("grid" always does)."""
    if cores == "grid":
        return True
    x, y = grid_xy(device)
    if memory_layout == "height":
        return cores <= x * y
    rows, cols = cores
    return rows <= y and cols <= x


def _require_fits(device, memory_layout, cores, what):
    """Skip (never shrink) a case whose fixed Tensix-core count or shard grid exceeds this compute grid."""
    if cores == "grid":
        return
    if memory_layout == "height":
        require_cores(device, cores, what=what)
    else:
        rows, cols = cores
        require_grid(device, min_x=cols, min_y=rows, what=what)


def _resolve_cores(device, memory_layout, cores):
    """'grid' -> the whole compute grid (a count for height, (rows, cols) for block); else unchanged.
    Callers check `_fits` / `_require_fits` first: a fixed count is never capped to the grid."""
    if cores == "grid":
        x, y = grid_xy(device)
        return x * y if memory_layout == "height" else (y, x)
    return cores


def _n_cores(memory_layout, cores):
    return cores if memory_layout == "height" else cores[0] * cores[1]


def _make_input(device, memory_layout, cores, shard_rows, shard_width, seed=0, dtype="bf16"):
    torch.manual_seed(seed)
    shape = tensor_shape(memory_layout, cores, shard_rows, shard_width)
    torch_input = torch.randn(shape, dtype=torch.float32).to(_TORCH_DTYPES[dtype])
    tt_input = ttnn.from_torch(
        torch_input,
        dtype=_DTYPES[dtype],
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=sharded_memory_config(device, memory_layout, cores, shard_rows, shard_width),
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


def _git_sha():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=Path(__file__).resolve().parent, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _units(memory_layout, cores, shard_rows, shard_width):
    """Work units the resplit deals out: tile-rows x page columns x column chunks (a page here is one
    stick, or one shard-width piece of a stick for BLOCK_SHARDED; it is not a tile)."""
    total_rows, width = tensor_shape(memory_layout, cores, shard_rows, shard_width)
    chunks = chunk_plan(shard_width)[1]
    return -(-total_rows // 32) * (width // shard_width) * chunks


# ---------------------------------------------------------------------------------------
# Correctness (the only pass/fail)
# ---------------------------------------------------------------------------------------

# Every case of the measurement sweep is correctness-checked here, because the sweep itself is opt-in.
# The cases in _FULL_CHECK_IDS also run with kernel_iters=2 and with the NoCs swapped; the rest get the
# one plain launch per variant that the sweep's correctness gate gave them.
_FULL_CHECK_IDS = ("h72x544", "h64x544", "b311x320", "b23x848", "h3x8192", "h24x4064", "b13x632f32", "b221x400")
_CORRECTNESS_CASES = _DEFAULT_CASES + [c for c in _WIDE_CASES if c[0] not in {d[0] for d in _DEFAULT_CASES}]


@pytest.mark.parametrize("case", _CORRECTNESS_CASES, ids=[c[0] for c in _CORRECTNESS_CASES])
def test_resident_shard_locality_correctness(device, case):
    """Every variant is a bit-exact copy (tilize -> untilize); _FULL_CHECK_IDS also with kernel_iters=2
    and with the reader / writer NoCs swapped."""
    case_id, memory_layout, cores, shard_rows, shard_width, dtype = _case(case)
    _require_fits(device, memory_layout, cores, what=case_id)
    cores = _resolve_cores(device, memory_layout, cores)
    tt_input, torch_input = _make_input(device, memory_layout, cores, shard_rows, shard_width, dtype=dtype)
    for variant in _variants_for(shard_rows, shard_width):
        _check(tt_input, torch_input, variant)
        if case_id in _FULL_CHECK_IDS:
            _check(tt_input, torch_input, variant, kernel_iters=2)
            _check(tt_input, torch_input, variant, swap_nocs=True)


# ---------------------------------------------------------------------------------------
# Measurement sweep (opt-in) -> printed report, written to RSL_REPORT when set
# ---------------------------------------------------------------------------------------


@sweep_gate("RSL_SWEEPS")
def test_resident_shard_locality_device_perf(device):
    """Correctness-gate every variant, then measure device kernel duration; perf never fails a test.

    A measurement sweep: skipped unless RSL_SWEEPS=1 (the CLI sets it) or EXAMPLES_SWEEPS=1. The report
    is logged; it is written to a file only when RSL_REPORT is a non-empty path. Cases whose fixed
    Tensix-core count or shard grid does not fit this compute grid are skipped and listed in the report."""
    custom = _custom_case()
    suite = os.environ.get("RSL_SUITE", "all")
    wide_only = [c for c in _WIDE_CASES if c[0] not in {d[0] for d in _DEFAULT_CASES}]
    suites = {"all": _DEFAULT_CASES + wide_only, "base": _DEFAULT_CASES, "wide": _WIDE_CASES}
    cases = [_case(c) for c in ([custom] if custom else suites[suite])]
    selected = tuple(os.environ.get("RSL_VARIANTS", ",".join(VARIANTS)).split(","))
    trials = int(os.environ.get("RSL_TRIALS", "1"))
    kernel_iters = int(os.environ.get("RSL_KERNEL_ITERS", "1"))
    swap_nocs = os.environ.get("RSL_SWAP_NOCS", "0") == "1"

    rows_out, skipped = [], []
    for case_id, memory_layout, cores, shard_rows, shard_width, dtype in cases:
        if not _fits(device, memory_layout, cores):
            skipped.append(f"{case_id} ({cores} Tensix cores)")
            continue
        cores = _resolve_cores(device, memory_layout, cores)
        tt_input, torch_input = _make_input(device, memory_layout, cores, shard_rows, shard_width, dtype=dtype)
        variants = _variants_for(shard_rows, shard_width, selected)
        for variant in variants:
            _check(tt_input, torch_input, variant, kernel_iters, swap_nocs)
        runners = {
            v: (lambda v=v: resident_shard_copy(tt_input, variant=v, kernel_iters=kernel_iters, swap_nocs=swap_nocs))
            for v in variants
        }
        samples = interleaved_ns(device, runners, trials=trials, warmup=1, per_launch_divisor=kernel_iters)
        rows_out.append(
            (case_id, memory_layout, cores, shard_rows, shard_width, dtype, {v: samples[v] for v in variants})
        )
    if not rows_out:
        x, y = grid_xy(device)
        pytest.skip(f"no case fits the {x}x{y} grid: {', '.join(skipped)}")

    x, y = grid_xy(device)
    freq = chip_freq_mhz()
    header = (
        f"box={socket.gethostname()}  arch={arch_label(device)}  grid={x}x{y}  "
        f"clock={f'{freq:.0f}' if freq else 'unknown'} MHz  N={trials} ({'median' if trials > 1 else 'single pass'})  "
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
        "| Case | TensorMemoryLayout | Tensix cores | Shard [sticks, elements] | B per stick | resplit units | Method | ns | vs resplit | Correct |",
        "|---|---|---:|---|---:|---:|---|---:|---:|---|",
    ]
    for case_id, memory_layout, cores, shard_rows, shard_width, dtype, results in rows_out:
        base = statistics.median(results["resplit"]) if "resplit" in results else None
        n = _n_cores(memory_layout, cores)
        stick_bytes = shard_width * elem_bytes(_DTYPES[dtype])
        tml = "HEIGHT_SHARDED" if memory_layout == "height" else f"BLOCK_SHARDED {cores[0]}x{cores[1]}"
        units = _units(memory_layout, cores, shard_rows, shard_width)
        for variant in VARIANTS:
            if variant not in results:
                if variant in selected and variant == "local_alias":
                    lines.append(
                        f"| {case_id} | {tml} | {n} | [{shard_rows}, {shard_width}] | {stick_bytes} | "
                        f"{units} | local_alias | n/a (shard not whole tiles) | | |"
                    )
                continue
            med = statistics.median(results[variant])
            vs = f"{base / med:.2f}x" if base else ""
            lines.append(
                f"| {case_id} | {tml} | {n} | [{shard_rows}, {shard_width}] | {stick_bytes} | "
                f"{units} | {variant} | {med:.0f} | {vs} | yes |"
            )
    if skipped:
        lines += ["", f"Skipped (shard grid larger than this {x}x{y} grid): {', '.join(skipped)}"]
    lines += _l1_table(rows_out)
    plain = selected == VARIANTS and trials == 1 and kernel_iters == 1 and not swap_nocs
    default_run = custom is None and suite == "all" and plain
    if default_run:
        lines += _mechanism_table(device)
    report = "\n".join(lines) + "\n"
    logger.info("\n" + report)
    path = os.environ.get("RSL_REPORT", "")
    if path:
        Path(path).write_text(report)
        logger.info(f"report written to {path}")


def _l1_table(rows_out):
    """Per-Tensix-core circular-buffer (CB) bytes of `local` for the cases wider than one chunk or not
    whole tiles wide."""
    lines = [
        "",
        "## L1 per Tensix core: `local` staging one column chunk vs staging the whole row",
        "",
        "Circular buffers (CBs) only (input staging, tiled, output staging; 2 chunks deep each; one page = one "
        "tile); the input and output shards come on top. Whole-row staging = the same 3 CBs, 2 deep, one chunk = "
        "the whole padded width.",
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
# Counts larger than the compute grid are left out.
_MECHANISM_CORES = (1, 2, 13, 26, 65, 130)


def _mechanism_table(device):
    x, y = grid_xy(device)
    lines = [
        "",
        "## Mechanism: resplit cost vs Tensix cores and NoC direction ([72, 544] HEIGHT_SHARDED shards)",
        "",
        "Same shards, more Tensix cores in row-major order. `swapped` = reader on NoC1, writer on NoC0.",
        "",
        "| Tensix cores | resplit ns | resplit swapped ns | local ns | resplit / local | resplit swapped / local |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for n in (c for c in _MECHANISM_CORES if c <= x * y):
        tt_input, torch_input = _make_input(device, "height", n, 72, 544)
        runners = {
            "resplit": lambda: resident_shard_copy(tt_input, variant="resplit"),
            "swapped": lambda: resident_shard_copy(tt_input, variant="resplit", swap_nocs=True),
            "local": lambda: resident_shard_copy(tt_input, variant="local"),
        }
        _check(tt_input, torch_input, "resplit", swap_nocs=True)
        med = {k: statistics.median(v) for k, v in interleaved_ns(device, runners, trials=3, warmup=1).items()}
        lines.append(
            f"| {n} | {med['resplit']:.0f} | {med['swapped']:.0f} | {med['local']:.0f} | "
            f"{med['resplit'] / med['local']:.2f}x | {med['swapped'] / med['local']:.2f}x |"
        )
    return lines


# ---------------------------------------------------------------------------------------
# Regimes (asserted): speedup = baseline ns / lever ns (resplit / local, or local / local_alias),
# interleaved A,B,A,B... trials per variant, verdict from `perf_test_utils.check_regime`.
# The bound each regime is asserted against is derived from the speedup measured on the architecture it
# runs on (`perf_test_utils.regime_bound`): "helps" asserts half the measured gain but never closer than
# 10% to the measurement; "does_not_help" asserts <= max(1.0, 1.10 x the highest measured value).
# "bh": Blackhole p150b, 13x10 = 130 Tensix cores (ranges: the regime test's runs in the comment /
# README.md / report.md). "wh": Wormhole B0 n150, 8x8 = 64 Tensix cores, 1000 MHz, min-max of the regime
# test's median over 10 runs (2026-11-13); [1, 8192] needs 128 Tensix cores and has no WH number.
# A missed bound is re-measured once and re-checked on the pooled samples.
# "grid" cases use the whole compute grid (BLOCK_SHARDED: rows x cols = the grid, K = grid columns), so
# they build on any grid; a fixed Tensix-core count skips on a grid that is too small.
# ---------------------------------------------------------------------------------------

_REGIME_TRIALS = 7

_REGIMES = [
    # (id, memory_layout, cores, shard_rows, shard_width, dtype, baseline, lever, kind, measured speedup)
    (
        "helps_height_72x544",
        "height",
        "grid",
        72,
        544,
        "bf16",
        "resplit",
        "local",
        "helps",
        {"wh": (4.07, 4.14), "bh": (10.71, 10.9)},
    ),
    (
        "helps_block_311x320",
        "block",
        "grid",
        311,
        320,
        "bf16",
        "resplit",
        "local",
        "helps",
        {"wh": (3.64, 3.80), "bh": (5.29, 5.47)},
    ),
    # <= 2 sticks of >= 8 KB per Tensix core; needs 128 Tensix cores
    (
        "does_not_help_height_1x8192",
        "height",
        128,
        1,
        8192,
        "bf16",
        "resplit",
        "local",
        "does_not_help",
        {"bh": (0.6, 0.75)},
    ),
    (
        "helps_alias_32x1024",
        "height",
        "grid",
        32,
        1024,
        "bf16",
        "local",
        "local_alias",
        "helps",
        {"wh": (1.73, 1.74), "bh": (1.55, 1.56)},
    ),
    # wide rows (127 / 160 tiles per stick), W chunked into <= 32-tile column chunks
    (
        "helps_height_wide_24x4064",
        "height",
        "grid",
        24,
        4064,
        "bf16",
        "resplit",
        "local",
        "helps",
        {"wh": (1.29, 1.56), "bh": (7.8, 7.94)},
    ),
    (
        "helps_height_wide_17x5120",
        "height",
        "grid",
        17,
        5120,
        "bf16",
        "resplit",
        "local",
        "helps",
        {"wh": (4.45, 4.70), "bh": (11.1, 11.2)},
    ),
    # partial-width BLOCK shards (shard width not a multiple of 32), K = grid columns
    (
        "helps_block_13x632_fp32",
        "block",
        "grid",
        13,
        632,
        "fp32",
        "resplit",
        "local",
        "helps",
        {"wh": (2.31, 2.34), "bh": (2.1, 2.13)},
    ),
    (
        "helps_block_221x400",
        "block",
        "grid",
        221,
        400,
        "bf16",
        "resplit",
        "local",
        "helps",
        {"wh": (1.23, 1.27), "bh": (5.66, 5.7)},
    ),
    (
        "helps_block_23x848",
        "block",
        "grid",
        23,
        848,
        "bf16",
        "resplit",
        "local",
        "helps",
        {"wh": (4.33, 4.41), "bh": (6.28, 6.3)},
    ),
    (
        "helps_block_16x848",
        "block",
        "grid",
        16,
        848,
        "bf16",
        "resplit",
        "local",
        "helps",
        {"wh": (2.41, 2.45), "bh": (4.2, 4.25)},
    ),
]


@pytest.mark.parametrize("regime", _REGIMES, ids=[r[0] for r in _REGIMES])
def test_resident_shard_locality_regimes(device, regime):
    """Interleaved A/B of baseline vs lever, both correctness-checked first; verdict from `check_regime`."""
    rid, memory_layout, cores, shard_rows, shard_width, dtype, baseline, lever, kind, measured = regime
    _require_fits(device, memory_layout, cores, what=rid)
    cores = _resolve_cores(device, memory_layout, cores)
    tt_input, torch_input = _make_input(device, memory_layout, cores, shard_rows, shard_width, dtype=dtype)
    for variant in (baseline, lever):
        _check(tt_input, torch_input, variant)
    runners = {v: (lambda v=v: resident_shard_copy(tt_input, variant=v)) for v in (baseline, lever)}

    def measure():
        samples = interleaved_ns(device, runners, trials=_REGIME_TRIALS)
        return samples[baseline], samples[lever]

    tml = "HEIGHT_SHARDED" if memory_layout == "height" else f"BLOCK_SHARDED {cores[0]}x{cores[1]}"
    check_regime(
        device,
        rid,
        kind,
        measured,
        measure,
        detail=(
            f"{tml} [{shard_rows}, {shard_width}] {dtype}, {_n_cores(memory_layout, cores)} Tensix cores, "
            f"{baseline} vs {lever}"
        ),
    )
