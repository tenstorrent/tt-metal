# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Layout::ROW_MAJOR data as flat tiles (`flat_tiles`) vs tilize -> compute -> untilize (`tilize_untilize`).

See ttnn/ttnn/operations/examples/row_major_flat_tiles/README.md.

    # both variants of both cases match torch; flat_tiles refuses a reduce whose W is not a multiple of 1024
    scripts/run_safe_pytest.sh --run-all \\
        tests/ttnn/unit_tests/operations/examples/test_row_major_flat_tiles.py -k "correctness or refuses"

    # the reduce datapath / read-order knobs, every combination, against torch
    scripts/run_safe_pytest.sh --run-all \\
        tests/ttnn/unit_tests/operations/examples/test_row_major_flat_tiles.py::test_row_major_flat_tiles_reduce_knobs

    # predefined sweep (device kernel duration per case x op x shape x variant + a reduce ablation): a measurement sweep,
    # skipped unless RMFT_SWEEPS=1 (or EXAMPLES_SWEEPS=1); printed, written to a file only when RMFT_REPORT names one
    RMFT_SWEEPS=1 RMFT_REPORT=ttnn/ttnn/operations/examples/row_major_flat_tiles/report_wh_n150.md \\
        scripts/run_safe_pytest.sh --run-all \\
        tests/ttnn/unit_tests/operations/examples/test_row_major_flat_tiles.py::test_row_major_flat_tiles_device_perf

    # helps / does-not-help regimes (asserted on the architectures they were measured on, logged elsewhere)
    scripts/run_safe_pytest.sh --run-all \\
        tests/ttnn/unit_tests/operations/examples/test_row_major_flat_tiles.py::test_row_major_flat_tiles_regimes
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
    grid_cores,
    grid_xy,
    interleaved_ns,
    sweep_gate,
)
from ttnn.operations.examples.row_major_flat_tiles import (
    PIECE_ORDERS,
    VARIANTS,
    cb_bytes,
    flat_tiles_applicable,
    resolve_reduce_algo,
    row_major_flat_tiles,
    work_split,
)

BASELINE, LEVER = VARIANTS


def _make_input(device, h, w, seed=0):
    torch.manual_seed(seed)
    torch_input = torch.randn((h, w), dtype=torch.float32).to(torch.bfloat16)
    tt_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    return tt_input, torch_input


def _reference(case, torch_input, op="relu", reps=1, **_):
    x = torch_input.float()
    if case == "reduce":
        return x - x.mean(dim=1, keepdim=True)
    for _ in range(reps):
        x = torch.relu(x) if op == "relu" else torch.nn.functional.gelu(x)
    return x


def _tolerance(case, op="relu", reps=1, **_):
    """(rtol, atol) vs the fp32 torch reference, identical for both variants. relu: bit-exact (relu applied
    any number of times is relu); gelu: the SFPU approximation with a bf16 round per pass, so the error
    grows with reps; reduce: bf16 out (8-bit mantissa) minus a bf16 mean."""
    if case == "reduce":
        return 0.01, 0.02
    return (0.0, 0.0) if op == "relu" else (0.02, 0.03 * reps)


def _run(tt_input, case, variant, **kw):
    return ttnn.to_torch(row_major_flat_tiles(tt_input, case=case, variant=variant, **kw))


def _check(tt_input, torch_input, case, variant, **kw):
    out = _run(tt_input, case, variant, **kw)
    ref = _reference(case, torch_input, **kw)
    rtol, atol = _tolerance(case, **kw)
    if rtol == 0.0 and atol == 0.0:
        ok = torch.equal(out.float(), ref.to(torch.bfloat16).float())
    else:
        ok = torch.allclose(out.float(), ref, rtol=rtol, atol=atol)
    diff = (out.float() - ref).abs()
    assert ok, (
        f"{case}/{variant} {kw} {list(torch_input.shape)}: max |diff| {diff.max().item():.4f} "
        f"at {divmod(int(diff.argmax()), torch_input.shape[1])} (rtol {rtol}, atol {atol})"
    )
    return out


def _label(case, kw):
    if case == "reduce":
        return "x - mean"
    reps = kw.get("reps", 1)
    return kw.get("op", "relu") + (f" x{reps}" if reps > 1 else "")


# ---------------------------------------------------------------------------------------
# Correctness (the only pass/fail)
# ---------------------------------------------------------------------------------------

# (case, H, W, kwargs)
_CORRECTNESS = [
    ("eltwise", 32, 64, {"op": "relu"}),
    ("eltwise", 32, 1024, {"op": "relu"}),
    ("eltwise", 96, 1056, {"op": "relu"}),
    ("eltwise", 2048, 4096, {"op": "relu"}),
    ("eltwise", 64, 1024, {"op": "relu", "reps": 4}),
    ("eltwise", 64, 1024, {"op": "gelu"}),
    ("eltwise", 32, 96, {"op": "gelu"}),
    ("reduce", 32, 1024, {}),
    ("reduce", 64, 2048, {}),
    ("reduce", 96, 4096, {}),
    ("reduce", 32, 8192, {}),
    ("reduce", 2048, 1024, {}),
    ("reduce", 64, 1536, {}),  # baseline only: flat_tiles refuses W % 1024 != 0
]


def _cid(c):
    case, h, w, kw = c
    return f"{case}{'_' + _label(case, kw).replace(' ', '') if case == 'eltwise' else ''}_{h}x{w}"


@pytest.mark.parametrize("cfg", _CORRECTNESS, ids=[_cid(c) for c in _CORRECTNESS])
def test_row_major_flat_tiles_correctness(device, cfg):
    """Every applicable variant matches torch; eltwise flat_tiles is bit-identical to the baseline; a
    second in-kernel repeat gives the same answer."""
    case, h, w, kw = cfg
    tt_input, torch_input = _make_input(device, h, w)
    outs = {}
    for variant in VARIANTS:
        if variant == LEVER and not flat_tiles_applicable(case, w)[0]:
            continue
        outs[variant] = _check(tt_input, torch_input, case, variant, **kw)
    if case == "eltwise":
        assert torch.equal(outs[BASELINE], outs[LEVER]), "eltwise: flat_tiles differs from tilize_untilize"
    if h <= 96:
        for variant in outs:
            _check(tt_input, torch_input, case, variant, kernel_iters=2, **kw)


@pytest.mark.parametrize("w", [1024, 4096])
def test_row_major_flat_tiles_reduce_knobs(device, w):
    """Both reduce datapaths on both variants, both read orders and a non-default block on flat_tiles."""
    tt_input, torch_input = _make_input(device, 64, w)
    for algo in ("accumulate", "reduce_tile"):
        _check(tt_input, torch_input, "reduce", BASELINE, reduce_algo=algo)
        for order in PIECE_ORDERS:
            _check(tt_input, torch_input, "reduce", LEVER, reduce_algo=algo, piece_order=order)
    _check(tt_input, torch_input, "reduce", LEVER, rows_per_block_override=2)


@pytest.mark.parametrize("w", [1056, 1536, 3072 + 32])
def test_row_major_flat_tiles_refuses_partial_flat_rows(device, w, expect_error):
    """flat_tiles refuses a per-row reduction whose W is not a multiple of 1024, and says why."""
    tt_input, _ = _make_input(device, 32, w)
    with expect_error(ValueError, "not a multiple of 1024"):
        row_major_flat_tiles(tt_input, case="reduce", variant=LEVER)


# ---------------------------------------------------------------------------------------
# Measurement sweep (opt-in) -> printed report, written to RMFT_REPORT when set
# ---------------------------------------------------------------------------------------

_SWEEP_ROWS = (32, 512, 2048, 8192)  # 1 tile-row (1 Tensix core) .. 256 tile-rows (4 per Tensix core on 8x8)
# (case, kwargs, widths): relu x16 = sixteen relu passes in DEST per tile (compute per element knob);
# W = 1536 is the reduce width flat_tiles refuses
_SWEEP = [
    ("eltwise", {"op": "relu"}, (64, 1024, 4096)),
    ("eltwise", {"op": "gelu"}, (64, 1024, 4096)),
    ("eltwise", {"op": "relu", "reps": 16}, (1024, 4096)),
    ("reduce", {}, (1024, 2048, 4096, 8192, 1536)),
]
# Reduce ablation: (H, W) cells x (label, variant, kwargs)
_ABLATION_CELLS = ((32, 1024), (32, 2048), (32, 4096), (32, 8192), (2048, 4096), (8192, 8192))
_ABLATION = [
    ("tilize_untilize accumulate", BASELINE, {"reduce_algo": "accumulate"}),
    ("tilize_untilize reduce_tile", BASELINE, {"reduce_algo": "reduce_tile"}),
    ("flat_tiles accumulate row", LEVER, {"reduce_algo": "accumulate", "piece_order": "row"}),
    ("flat_tiles accumulate interleaved", LEVER, {"reduce_algo": "accumulate", "piece_order": "interleaved"}),
    ("flat_tiles reduce_tile interleaved", LEVER, {"reduce_algo": "reduce_tile", "piece_order": "interleaved"}),
]


def _env_ints(name, default):
    value = os.environ.get(name)
    return tuple(int(v) for v in value.split(",")) if value else default


def _sweep_plan():
    """(plan, custom): the predefined sweep, or one case from the CLI overrides (RMFT_CASE + RMFT_OP / RMFT_REPS /
    RMFT_REDUCE_ALGO / RMFT_PIECE_ORDER), over RMFT_ROWS x RMFT_WIDTHS when given."""
    rows = _env_ints("RMFT_ROWS", _SWEEP_ROWS)
    case = os.environ.get("RMFT_CASE", "")
    if not case:
        return [(c, kw, rows, _env_ints("RMFT_WIDTHS", widths)) for c, kw, widths in _SWEEP], False
    kw = {}
    if case == "eltwise":
        kw = {"op": os.environ.get("RMFT_OP", "relu"), "reps": int(os.environ.get("RMFT_REPS", "1"))}
    else:
        kw = {
            "reduce_algo": os.environ.get("RMFT_REDUCE_ALGO", "auto"),
            "piece_order": os.environ.get("RMFT_PIECE_ORDER", PIECE_ORDERS[0]),
        }
    widths = _env_ints("RMFT_WIDTHS", (1024, 4096))
    return [(case, kw, rows, widths)], True


def _git_sha():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=Path(__file__).resolve().parent, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _measure_cell(device, case, h, w, entries, trials, kernel_iters):
    """entries: [(label, variant, kwargs)]; correctness-gate each, then interleaved device kernel ns (median)."""
    tt_input, torch_input = _make_input(device, h, w)
    for _, variant, kw in entries:
        _check(tt_input, torch_input, case, variant, kernel_iters=kernel_iters, **kw)
    runners = {
        label: (
            lambda v=variant, kw=kw: row_major_flat_tiles(
                tt_input, case=case, variant=v, kernel_iters=kernel_iters, **kw
            )
        )
        for label, variant, kw in entries
    }
    samples = interleaved_ns(device, runners, trials=trials, warmup=1, per_launch_divisor=kernel_iters)
    return {label: statistics.median(v) for label, v in samples.items()}


@sweep_gate("RMFT_SWEEPS")
def test_row_major_flat_tiles_device_perf(device):
    """Correctness-gate every variant of every cell, then measure device kernel duration; perf never fails a test.

    A measurement sweep: skipped unless RMFT_SWEEPS=1 (the CLI sets it) or EXAMPLES_SWEEPS=1. Logged; written to
    a file only when RMFT_REPORT is a non-empty path."""
    selected = tuple(os.environ.get("RMFT_VARIANTS", ",".join(VARIANTS)).split(","))
    trials = int(os.environ.get("RMFT_TRIALS", "1"))
    kernel_iters = int(os.environ.get("RMFT_KERNEL_ITERS", "1"))
    plan, custom = _sweep_plan()

    rows_out = []
    for case, kw, rows, widths in plan:
        for w in widths:
            for h in rows:
                entries = [(v, v, kw) for v in selected if v == BASELINE or flat_tiles_applicable(case, w)[0]]
                med = _measure_cell(device, case, h, w, entries, trials, kernel_iters)
                rows_out.append((case, kw, h, w, len(work_split(device, h // 32)), med))

    x, y = grid_xy(device)
    freq = chip_freq_mhz()
    header = (
        f"box={socket.gethostname()}  arch={arch_label(device)}  grid={x}x{y}  "
        f"clock={f'{freq:.0f}' if freq else 'unknown'} MHz  N={trials} ({'median' if trials > 1 else 'single pass'})  "
        f"kernel-iters={kernel_iters} ({'per-launch latency' if kernel_iters == 1 else 'steady-state'})  "
        f"date={datetime.date.today().isoformat()}  sha={_git_sha()}"
    )
    lines = [
        "# row_major_flat_tiles — tilize_untilize vs flat_tiles",
        "",
        header,
        "",
        "Interleaved DRAM ROW_MAJOR bf16 [H, W] in and out. Tile-rows (32 sticks) split in contiguous ranges over "
        "min(H / 32, grid) Tensix cores, identical for both variants. GB/s = (bytes read + bytes written) / time "
        "(baseline / lever). CB KB = circular buffers per Tensix core. Reduce datapath = the one each variant "
        "runs (`resolve_reduce_algo`).",
        "",
        "| Case | Op | H | W | Tensix cores | tilize_untilize ns | flat_tiles ns | speedup | GB/s | CB KB base / lever | reduce datapath base / lever |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|---|---|",
    ]
    for case, kw, h, w, cores, med in rows_out:
        base, lever = med.get(BASELINE), med.get(LEVER)
        applicable = flat_tiles_applicable(case, w)[0]
        nbytes = 2 * h * w * 2
        gbs = " / ".join(f"{nbytes / v:.1f}" for v in (base, lever) if v)
        if not applicable:
            lever_cell, speed = "n/a (W % 1024 != 0: refused)", ""
        else:
            lever_cell = f"{lever:.0f}" if lever else ""
            speed = f"{base / lever:.2f}x" if base and lever else ""
        lever_kb = f"{cb_bytes(case, LEVER, w) // 1024}" if applicable else "-"
        algo = (
            f"{resolve_reduce_algo(BASELINE, w, kw.get('reduce_algo', 'auto'))} / "
            f"{resolve_reduce_algo(LEVER, w, kw.get('reduce_algo', 'auto')) if applicable else '-'}"
            if case == "reduce"
            else ""
        )
        lines.append(
            f"| {case} | {_label(case, kw)} | {h} | {w} | {cores} | {f'{base:.0f}' if base else ''} | {lever_cell} | "
            f"{speed} | {gbs} | {cb_bytes(case, BASELINE, w) // 1024} / {lever_kb} | {algo} |"
        )
    if not custom and selected == VARIANTS and kernel_iters == 1:
        lines += _ablation_table(device, trials)
    report = "\n".join(lines) + "\n"
    logger.info("\n" + report)
    path = os.environ.get("RMFT_REPORT", "")
    if path:
        Path(path).write_text(report)
        logger.info(f"report written to {path}")


def _ablation_table(device, trials):
    """Reduce case: each variant on each datapath, flat_tiles with both read orders."""
    lines = [
        "",
        "## Reduce ablation: datapath and read order",
        "",
        "`accumulate` = pairwise add_tiles in DEST + SFPU within-tile collapse + x 1/W (one collapse per output); "
        "`reduce_tile` = FPU reduce_tile per tile with a scaler tile. flat_tiles produces one mean per row (32 per "
        "tile-row), the baseline one per tile-row. Read order (flat_tiles): `row` = a stick's Wf flat tiles back to "
        "back (Wf consecutive 2 KB transfers to one DRAM bank), `interleaved` = flat-tile column by column across "
        "the block's sticks. ns, median.",
        "",
        "| H | W | Wf | Tensix cores | " + " | ".join(label for label, _, _ in _ABLATION) + " |",
        "|---:|---:|---:|---:|" + "---:|" * len(_ABLATION),
    ]
    for h, w in _ABLATION_CELLS:
        med = _measure_cell(device, "reduce", h, w, _ABLATION, max(trials, 3), 1)
        lines.append(
            f"| {h} | {w} | {w // 1024} | {len(work_split(device, h // 32))} | "
            + " | ".join(f"{med[label]:.0f}" for label, _, _ in _ABLATION)
            + " |"
        )
    return lines


# ---------------------------------------------------------------------------------------
# Regimes (asserted): speedup = tilize_untilize ns / flat_tiles ns, interleaved A,B,A,B... trials,
# verdict from `perf_test_utils.check_regime`. The bound is derived from the speedup measured on the
# architecture the test runs on ("helps": half the measured gain but never closer than 10% to it;
# "does_not_help": <= max(1.0, 1.10 x the highest measured value); "measure": logged only). An
# architecture with no measured value is logged, not asserted. "wh": Wormhole B0 n150, 8x8 grid,
# 1000 MHz - min-max of the regime test's median over 3 runs (2026-11-19, bgd-lab-22). No "bh": not measured on Blackhole.
# ---------------------------------------------------------------------------------------

_REGIME_TRIALS = 7

# (id, case, H, W, kwargs, kind, measured speedup)
_REGIMES = [
    # one tile-row on one Tensix core: tilize + untilize (~170 ns per tile) are a large share of the core's time
    ("helps_eltwise_relu_32x4096", "eltwise", 32, 4096, {"op": "relu"}, "helps", {"wh": (1.595, 1.597)}),
    ("helps_eltwise_relu_512x4096", "eltwise", 512, 4096, {"op": "relu"}, "helps", {"wh": (1.40, 1.41)}),
    # sixteen relu passes per tile: the saved tilize / untilize is a small share of the compute
    (
        "does_not_help_eltwise_relu_x16_32x4096",
        "eltwise",
        32,
        4096,
        {"op": "relu", "reps": 16},
        "does_not_help",
        {"wh": 1.05},
    ),
    # full grid, DRAM-bound (~190 GB/s read + write): compute was never the limit
    (
        "does_not_help_eltwise_relu_2048x4096",
        "eltwise",
        2048,
        4096,
        {"op": "relu"},
        "does_not_help",
        {"wh": (1.01, 1.02)},
    ),
    ("helps_reduce_32x8192", "reduce", 32, 8192, {}, "helps", {"wh": (1.85, 1.85)}),
    ("helps_reduce_512x4096", "reduce", 512, 4096, {}, "helps", {"wh": (1.26, 1.29)}),
    # one flat tile per row: 32 outputs per tile-row instead of 1 eat most of the saving
    ("measure_reduce_32x1024", "reduce", 32, 1024, {}, "measure", {"wh": (1.08, 1.08)}),
    ("does_not_help_reduce_8192x8192", "reduce", 8192, 8192, {}, "does_not_help", {"wh": (1.00, 1.01)}),
]


@pytest.mark.parametrize("regime", _REGIMES, ids=[r[0] for r in _REGIMES])
def test_row_major_flat_tiles_regimes(device, regime):
    """Interleaved A/B of tilize_untilize vs flat_tiles, both correctness-checked first; verdict from check_regime."""
    rid, case, h, w, kw, kind, measured = regime
    tt_input, torch_input = _make_input(device, h, w)
    for variant in VARIANTS:
        _check(tt_input, torch_input, case, variant, **kw)
    runners = {v: (lambda v=v: row_major_flat_tiles(tt_input, case=case, variant=v, **kw)) for v in VARIANTS}

    def measure():
        samples = interleaved_ns(device, runners, trials=_REGIME_TRIALS)
        return samples[BASELINE], samples[LEVER]

    check_regime(
        device,
        rid,
        kind,
        measured,
        measure,
        detail=f"{case} {_label(case, kw)} [{h}, {w}] on {len(work_split(device, h // 32))} Tensix cores",
    )
