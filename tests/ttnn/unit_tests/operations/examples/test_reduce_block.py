# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""The accumulate + SFPU-finalize fast path on a 2-D (Ht, Wt, NC) tile block reduced along one dim.

The committed reduce_accumulate example only reduces blocks whose non-reduced dim is a single tile (one
output tile). Here the fast path is driven over general blocks — Ht>1 for row, Wt>1 for col, NC>1 batches
— where the reduce emits MULTIPLE output tiles, and is compared against the reduce library over the same
block. The fast path handles this as a loop over output tiles (each an independent accumulate-subset +
SFPU-finalize); this test proves that multi-tile-output fast path is correct and measures it vs the library.

Correctness (vs the fp64 mean) is the only pass/fail; perf (DEVICE KERNEL DURATION [ns] per block) and
accuracy (max/mean abs + bf16-ULP) are measured and reported.
"""

import os

os.environ.setdefault("TT_METAL_DEVICE_PROFILER", "1")
os.environ.setdefault("TT_METAL_PROFILER_MID_RUN_DUMP", "1")
os.environ.setdefault("TT_METAL_PROFILER_CPP_POST_PROCESS", "1")
os.environ.setdefault("TT_METAL_LOGGER_LEVEL", "error")

import socket
import statistics
from pathlib import Path

import torch
import ttnn
from loguru import logger

from ttnn.operations.examples.reduce_block import (
    BASELINE,
    DIMS,
    VARIANTS,
    create_sharded_memory_config,
    input_shape,
    out_tile_count,
    output_shape,
    run_accumulate,
    run_op,
)

TILE = 32
_DURATION_KEY = "DEVICE KERNEL DURATION [ns]"

# Per-accum correctness tolerance (max abs err vs fp64 mean) — catches wiring/scale/tile-order bugs while
# allowing the real bf16 quantization + accumulation error.
_MAX_ABS_TOL = {"fp32": 0.05, "bf16": 1.00}

# 2-D block shapes (Ht, Wt, NC) per reduce dim. The non-reduced dim is > 1 tile (so the output is MANY
# tiles — the surface the 1-D example never hits); a couple of NC>1 batched cases; one degenerate
# single-output-tile shape per dim as a sanity anchor.
_SHAPES = {
    "row": [(2, 3, 1), (4, 2, 1), (3, 5, 1), (2, 3, 2), (1, 4, 1)],  # reduce width -> Ht*NC output tiles
    "col": [(3, 2, 1), (2, 4, 1), (5, 3, 1), (3, 2, 2), (4, 1, 1)],  # reduce height -> Wt*NC output tiles
    "scalar": [(2, 3, 1), (3, 2, 1), (4, 4, 1), (2, 2, 2)],  # reduce both -> NC output tiles
}
# Perf sweep: a few blocks per dim that produce multiple output tiles.
_PERF_SHAPES = {
    "row": [(4, 2, 1), (4, 4, 1), (2, 8, 1)],
    "col": [(2, 4, 1), (4, 4, 1), (8, 2, 1)],
    "scalar": [(2, 2, 1), (4, 4, 1), (2, 8, 1)],
}


# =============================================================================
# Inputs + golden (positive [0,1): all-positive, nonzero mean -> ULP meaningful)
# =============================================================================
def _make_input(device, dim, Ht, Wt, NC, seed=13):
    torch.manual_seed(seed)
    h, w = input_shape(Ht, Wt, NC)
    data = torch.rand(h, w)  # fp32 "true" data
    b = data.view(NC, Ht * TILE, Wt * TILE).to(torch.float64)  # batch nc = tile-rows [nc*Ht, (nc+1)*Ht)
    if dim == "row":
        golden = b.mean(dim=2).reshape(-1)  # (NC, Ht*32) per-row means -> flat
    elif dim == "col":
        golden = b.mean(dim=1).reshape(-1)  # (NC, Wt*32) per-col means -> flat
    else:
        golden = b.mean(dim=(1, 2)).reshape(-1)  # (NC,) one mean per batch
    x_dev = ttnn.from_torch(
        data.to(torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=create_sharded_memory_config((h, w)),
    )
    return x_dev, golden


def _readout(output, dim):
    """The meaningful values of the multi-tile output, flattened to match the golden."""
    t = ttnn.to_torch(output).to(torch.float64)
    if dim == "row":
        return t[:, 0]  # [NC*Ht*32, 32] -> per-row means live in column 0 of every output tile
    if dim == "col":
        return t[0, :]  # [32, NC*Wt*32] -> per-col means live in row 0
    return t[::TILE, 0]  # [NC*32, 32] -> each batch's scalar mean at tile-local [0, 0]


def _ulp_bf16(x):
    x = x.abs().to(torch.float64).clamp_min(2.0**-14)
    e = torch.floor(torch.log2(x))
    return torch.pow(torch.tensor(2.0, dtype=torch.float64), e - 7)


def _accuracy(output, golden, dim):
    diff = (_readout(output, dim) - golden).abs()
    return diff.max().item(), diff.mean().item(), (diff / _ulp_bf16(golden)).max().item()


def _check(output, golden, dim, accum, label):
    got = _readout(output, dim)
    assert got.numel() == golden.numel(), f"{label}: output has {got.numel()} values, expected {golden.numel()}"
    max_abs, mean_abs, max_ulp = _accuracy(output, golden, dim)
    assert max_abs < _MAX_ABS_TOL[accum], f"{label}: max-abs {max_abs:.4f} >= {_MAX_ABS_TOL[accum]}"
    return max_abs, mean_abs, max_ulp


# =============================================================================
# In-process device-kernel timing (validated pattern)
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
    _read_kernel_ns(device)
    samples = {key: [] for key in runners}
    for trial in range(trials + 1):
        for key, run in runners.items():
            run()
            duration = _read_kernel_ns(device)
            assert duration is not None, f"no profiler data for {key}"
            if trial:
                samples[key].append(duration / kernel_iters)
    return samples


def _arch_label(device):
    if name := os.environ.get("ARCH_NAME"):
        return name
    a = str(device.arch()).rsplit(".", 1)[-1]
    return {"WORMHOLE_B0": "WH_B0", "BLACKHOLE": "BH", "GRAYSKULL": "GS"}.get(a, a)


def _int(name, default):
    return int(os.environ.get(name, default))


def _csv(name, default):
    """Comma-separated env override -> list, else `default`."""
    val = os.environ.get(name)
    return [s for s in val.split(",") if s] if val else list(default)


def _shapes(name):
    """Semicolon-separated `Ht,Wt,NC` env override -> list of tuples, else None (use built-in)."""
    val = os.environ.get(name)
    if not val:
        return None
    return [tuple(int(x) for x in tok.split(",")) for tok in val.split(";") if tok]


# =============================================================================
# Tests
# =============================================================================
def test_reduce_block_correctness(device):
    """Every variant (reduce_tile / accumulate_via_add / _inline / dispatch) -> correct MULTI-tile output."""
    for dim in DIMS:
        for Ht, Wt, NC in _SHAPES[dim]:
            x_dev, golden = _make_input(device, dim, Ht, Wt, NC)
            n_out = out_tile_count(dim, Ht, Wt, NC)
            for accum in ("fp32", "bf16"):
                for variant in VARIANTS:
                    out = run_op(x_dev, variant=variant, dim=dim, Ht=Ht, Wt=Wt, NC=NC, accum=accum, kernel_iters=2)
                    assert list(out.shape) == list(output_shape(dim, Ht, Wt, NC))
                    label = f"{variant}/{dim}/{accum} Ht={Ht} Wt={Wt} NC={NC}"
                    ma, me, ul = _check(out, golden, dim, accum, label)
                    logger.info(f"{label:44s} out_tiles={n_out:2d}  max_abs={ma:.5f} mean_abs={me:.5f} ulp={ul:.2f}")


def test_reduce_block_device_perf(device):
    """fast vs the reduce library over 2-D blocks: perf (ns/block) + accuracy, per dim.

    CLI-driveable: REDBLK_DIMS (row,col,scalar), REDBLK_VARIANTS, REDBLK_SHAPES (Ht,Wt,NC;...) override
    the built-in sweep; REDBLK_TRIALS / REDBLK_KERNEL_ITERS / REDBLK_REPORT tune the measurement.
    """
    trials = _int("REDBLK_TRIALS", "5")
    kernel_iters = _int("REDBLK_KERNEL_ITERS", "200")
    # reduce_tile = library default (matmul-reduce); accumulate_via_add = library opt-in AccumulateViaAdd;
    # accumulate_via_add_inline = the hand-written standalone kernel (init hoisted out of the perf loop).
    _PERF_VARIANTS = ("reduce_tile", "accumulate_via_add", "accumulate_via_add_inline")
    variants = [v for v in _csv("REDBLK_VARIANTS", _PERF_VARIANTS) if v in _PERF_VARIANTS]
    sel_dims = [d for d in _csv("REDBLK_DIMS", DIMS) if d in DIMS]
    shape_override = _shapes("REDBLK_SHAPES")
    perf_shapes = {dim: (shape_override or _PERF_SHAPES[dim]) for dim in sel_dims}

    inputs, goldens = {}, {}
    for dim in sel_dims:
        for shape in perf_shapes[dim]:
            inputs[(dim, shape)], goldens[(dim, shape)] = _make_input(device, dim, *shape)

    # accuracy (both accum) + correctness gate at kernel_iters=1
    acc = {}  # (variant, dim, shape, accum) -> (max_abs, mean_abs, max_ulp)
    for dim in sel_dims:
        for shape in perf_shapes[dim]:
            for accum in ("fp32", "bf16"):
                for variant in variants:
                    out = run_op(
                        inputs[(dim, shape)],
                        variant=variant,
                        dim=dim,
                        Ht=shape[0],
                        Wt=shape[1],
                        NC=shape[2],
                        accum=accum,
                        kernel_iters=1,
                    )
                    acc[(variant, dim, shape, accum)] = _check(
                        out, goldens[(dim, shape)], dim, accum, f"{variant}/{dim}/{accum} {shape}"
                    )

    # perf (fp32 accum, steady-state)
    runners = {
        (variant, dim, shape): (
            lambda v=variant, d=dim, s=shape: run_op(
                inputs[(d, s)], variant=v, dim=d, Ht=s[0], Wt=s[1], NC=s[2], accum="fp32", kernel_iters=kernel_iters
            )
        )
        for dim in sel_dims
        for shape in perf_shapes[dim]
        for variant in variants
    }
    samples = _measure(device, runners, trials, kernel_iters)

    def med(v, d, s):
        return statistics.median(samples[(v, d, s)]) if (v, d, s) in samples else None

    def cell(ns, base):
        """A perf cell: `ns (×vs base)`, or `—` if the variant wasn't run."""
        if ns is None:
            return "—"
        return f"{ns:.0f} ({base / ns:.2f}×)" if base else f"{ns:.0f}"

    def acc_cell(v, d, s):
        entry = acc.get((v, d, s, "bf16"))
        return f"{entry[0]:.1e} \\| {entry[2]:.1f}u" if entry else "—"

    lines = [
        "# Reduce over a 2-D (Ht, Wt, NC) block — fast (per-output accumulate+SFPU) vs the reduce library (single core)",
        "",
        f"box={socket.gethostname()}  arch={_arch_label(device)}  cores=1  placement=single-core sharded-L1  "
        f"N={trials} (median)  kernel-iters={kernel_iters}",
        "problem: reduce a 2-D tile block along ONE dim -> MANY output tiles. Input bf16, output fp32, HiFi4.",
        "perf = median ns per whole-block reduce (fp32 accum). accuracy = bf16 accum, max_abs | max ULP_bf16.",
        "",
        "| dim | Ht×Wt×NC | out | reduce_tile ns | acc_via_add ns (×) | inline ns (×) | reduce_tile acc | acc_via_add acc |",
        "|---|---|---:|---:|---:|---:|---|---|",
    ]
    for dim in sel_dims:
        for shape in perf_shapes[dim]:
            n_out = out_tile_count(dim, *shape)
            rt_ns = med("reduce_tile", dim, shape)
            rt_cell = f"{rt_ns:.0f}" if rt_ns is not None else "—"
            lines.append(
                f"| {dim} | {shape[0]}×{shape[1]}×{shape[2]} | {n_out} | {rt_cell} | "
                f"{cell(med('accumulate_via_add', dim, shape), rt_ns)} | "
                f"{cell(med('accumulate_via_add_inline', dim, shape), rt_ns)} | "
                f"{acc_cell('reduce_tile', dim, shape)} | {acc_cell('accumulate_via_add', dim, shape)} |"
            )
    lines += [
        "",
        "Variants: reduce_tile = library default (ReduceAlgorithm::Auto -> ReduceTile, FPU matmul-with-ones); "
        "acc_via_add = library with the opt-in ReduceAlgorithm::AccumulateViaAdd; inline = the same algorithm "
        "as a hand-written standalone kernel with the one-time init hoisted OUT of the kernel_iters loop. "
        "acc_via_add runs its init per reduce() call (like the library's own reduce_init), so it trails inline "
        "by that fixed per-call cost but is the apples-to-apples library-vs-library number. Accuracy of "
        "acc_via_add matches inline (same algorithm). AccumulateViaAdd uses one DST register per output tile, "
        "so it reduces an arbitrary block without the REDUCE_COL DST/chunk limit the library default chunks around.",
    ]
    report = "\n".join(lines) + "\n"
    logger.info("\n" + report)
    if report_path := os.environ.get("REDBLK_REPORT"):
        Path(report_path).write_text(report)


# 1-D reference (single-output) medians on this box's arch, reduced tiles = 1,2,4,8,16,32, fp32 accum,
# from the reduce_accumulate report (Blackhole p150b). Used to confirm the 2-D block per-output cost is
# the SAME as the committed 1-D example at the same reduced tile-count.
_REDUCED_SWEEP = (1, 2, 4, 8, 16, 32)
_REF_1D_HELPER = {
    "row": (296, 417, 647, 1109, 2033, 3884),
    "col": (234, 298, 421, 665, 1155, 2140),
    "scalar": (290, 414, 648, 1115, 2050, 3924),
}
_REF_1D_FAST = {
    "row": (538, 536, 552, 595, 665, 823),
    "col": (444, 443, 459, 506, 576, 730),
    "scalar": (614, 616, 632, 674, 745, 899),
}
# scalar: a 2-D arrangement (Ht, Wt) of R tiles into ONE output — squareish, to show cost tracks the total
# reduced count R = Ht*Wt regardless of layout (the 1-D scalar was 1xR).
_SCALAR_2D = {1: (1, 1), 2: (1, 2), 4: (2, 2), 8: (2, 4), 16: (4, 4), 32: (4, 8)}


def _one_output_cfg(dim, R):
    """(Ht, Wt, NC) that reduces R tiles into exactly ONE output tile, per dim."""
    if dim == "row":
        return (1, R, 1)  # width R -> 1 row of output
    if dim == "col":
        return (R, 1, 1)  # height R -> 1 col of output
    return (*_SCALAR_2D[R], 1)  # scalar: 2-D block of R tiles -> 1 output


def test_reduce_block_reduced_sweep(device):
    """accumulate_via_add / reduce_tile ratio vs REDUCED tiles-per-output up to 32 (one output) — vs 1-D.

    Also a linearity check: a MULTI-output block's total ns ≈ out_tiles × the single-output cost, i.e. the
    fast path's per-output-tile loop is linear and each output behaves like the 1-D single-output reduce.
    """
    trials = _int("REDBLK_TRIALS", "5")
    kernel_iters = _int("REDBLK_KERNEL_ITERS", "200")
    variants = ("reduce_tile", "accumulate_via_add_inline")

    # ---- single-output sweep over reduced-count R ----
    inputs = {}
    for dim in DIMS:
        for R in _REDUCED_SWEEP:
            Ht, Wt, NC = _one_output_cfg(dim, R)
            inputs[(dim, R)], _ = _make_input(device, dim, Ht, Wt, NC)
    single = {
        (v, dim, R): (
            lambda vv=v, d=dim, RR=R, cfg=_one_output_cfg(dim, R): run_op(
                inputs[(d, RR)],
                variant=vv,
                dim=d,
                Ht=cfg[0],
                Wt=cfg[1],
                NC=cfg[2],
                accum="fp32",
                kernel_iters=kernel_iters,
            )
        )
        for dim in DIMS
        for R in _REDUCED_SWEEP
        for v in variants
    }

    # ---- linearity check: one multi-output block per dim (out_tiles=4, reduced=8) ----
    multi_cfgs = {"row": (4, 8, 1), "col": (8, 4, 1), "scalar": (2, 4, 4)}  # out_tiles = 4 each
    multi_inputs = {dim: _make_input(device, dim, *multi_cfgs[dim])[0] for dim in DIMS}
    multi = {
        (v, dim): (
            lambda vv=v, d=dim: run_op(
                multi_inputs[d],
                variant=vv,
                dim=d,
                Ht=multi_cfgs[d][0],
                Wt=multi_cfgs[d][1],
                NC=multi_cfgs[d][2],
                accum="fp32",
                kernel_iters=kernel_iters,
            )
        )
        for dim in DIMS
        for v in variants
    }

    runners = {**{("S", *k): f for k, f in single.items()}, **{("M", *k): f for k, f in multi.items()}}
    samples = _measure(device, runners, trials, kernel_iters)

    def med(*key):
        return statistics.median(samples[key])

    lines = [
        "# Reduce-block: accumulate_via_add/reduce_tile ratio vs REDUCED tiles-per-output (one output) — vs 1-D",
        "",
        f"box={socket.gethostname()}  arch={_arch_label(device)}  cores=1  N={trials} (median)  "
        f"kernel-iters={kernel_iters}  fp32 accum",
        "Each config reduces R tiles into ONE output tile. row=(1,R), col=(R,1), scalar=2-D (Ht,Wt) with "
        "Ht*Wt=R. '1-D ref' is the reduce_accumulate single-output number at the same R (this box).",
        "",
    ]
    for dim in DIMS:
        lines += [
            f"### dim = {dim}",
            "",
            "| reduced R | reduce_tile ns | acc_via_add ns | ratio (2-D) | ratio (1-D ref) | reduce_tile 1-D ref |",
            "|---:|---:|---:|---:|---:|---:|",
        ]
        for i, R in enumerate(_REDUCED_SWEEP):
            h, f = med("S", "reduce_tile", dim, R), med("S", "accumulate_via_add_inline", dim, R)
            r2d = h / f if f else 0.0
            r1d = _REF_1D_HELPER[dim][i] / _REF_1D_FAST[dim][i]
            cfg = _one_output_cfg(dim, R)
            note = f"  ({cfg[0]}×{cfg[1]})" if dim == "scalar" else ""
            lines.append(f"| {R}{note} | {h:.0f} | {f:.0f} | {r2d:.2f}× | {r1d:.2f}× | {_REF_1D_HELPER[dim][i]} |")
        lines.append("")

    lines += [
        "## Linearity — a MULTI-output block ≈ out_tiles × the single-output cost (reduced=8, out_tiles=4)",
        "",
        "| dim | block (Ht×Wt×NC) | total ns | ÷ out_tiles | single-output ns (R=8) |",
        "|---|---|---:|---:|---:|",
    ]
    for dim in DIMS:
        cfg = multi_cfgs[dim]
        n_out = out_tile_count(dim, *cfg)
        total_f = med("M", "accumulate_via_add_inline", dim)
        per_out = total_f / n_out
        single_f = med("S", "accumulate_via_add_inline", dim, 8)
        lines.append(f"| {dim} | {cfg[0]}×{cfg[1]}×{cfg[2]} | {total_f:.0f} | {per_out:.0f} | {single_f:.0f} |")
    lines += [
        "",
        "Consistency: row/col one-output configs are byte-identical to the 1-D example (same tensor + kernel), "
        "so fast×(2-D) should match fast×(1-D ref); scalar uses a 2-D arrangement, so a match shows the cost "
        "tracks the TOTAL reduced count, not the layout. Linearity: total ≈ out_tiles × single-output confirms "
        "the fast path's per-output loop is linear and each output behaves like the 1-D reduce.",
    ]
    report = "\n".join(lines) + "\n"
    logger.info("\n" + report)
    if report_path := os.environ.get("REDBLK_SWEEP_REPORT"):
        Path(report_path).write_text(report)


def test_reduce_block_partial_row(device):
    """Non-tile-aligned ROW reduce via accumulate_via_add_inline: the last tile's invalid columns hold
    GARBAGE, and the masked accumulate must exclude them (result = mean over the L valid columns only).
    Proves the DEST-accumulating masked broadcast-mul folds the partial tile in correctly."""
    GARBAGE = 999.0
    for Wt, L in [(2, 48), (3, 80), (4, 100), (2, 40), (3, 65)]:
        P = L - (Wt - 1) * TILE  # valid columns in the last tile
        assert 1 <= P < TILE, (Wt, L, P)
        torch.manual_seed(13)
        data = torch.full((TILE, Wt * TILE), GARBAGE, dtype=torch.float32)
        data[:, :L] = torch.rand(TILE, L)
        golden = data[:, :L].to(torch.float64).mean(dim=1)  # per-row mean over the L valid columns
        x = ttnn.from_torch(
            data.to(torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=create_sharded_memory_config((TILE, Wt * TILE)),
        )
        for variant in ("accumulate_via_add_inline", "accumulate_via_add"):  # inline reference + library
            out = run_op(
                x,
                variant=variant,
                dim="row",
                Ht=1,
                Wt=Wt,
                NC=1,
                accum="fp32",
                kernel_iters=2,
                partial_elems=P,
            )
            got = ttnn.to_torch(out).to(torch.float64)[:, 0]
            ma = (got - golden).abs().max().item()
            logger.info(f"partial row {variant:26s} Wt={Wt} L={L} P={P}  max_abs={ma:.5f}")
            assert ma < 0.05, f"{variant} row L={L} P={P}: max_abs {ma:.4f} — garbage leaked"


def test_reduce_block_partial_col(device):
    """Non-tile-aligned COL reduce via accumulate_via_add_inline: the last row-tile's invalid rows hold
    GARBAGE; the col-0 masked accumulate (mul_tiles_bcast_cols) must exclude them."""
    GARBAGE = 999.0
    for Ht, L in [(2, 48), (3, 80), (4, 100), (2, 40), (3, 65)]:
        P = L - (Ht - 1) * TILE  # valid rows in the last row-tile
        assert 1 <= P < TILE, (Ht, L, P)
        torch.manual_seed(13)
        data = torch.full((Ht * TILE, TILE), GARBAGE, dtype=torch.float32)
        data[:L, :] = torch.rand(L, TILE)
        golden = data[:L, :].to(torch.float64).mean(dim=0)  # per-column mean over the L valid rows
        x = ttnn.from_torch(
            data.to(torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=create_sharded_memory_config((Ht * TILE, TILE)),
        )
        for variant in ("accumulate_via_add_inline", "accumulate_via_add"):  # inline reference + library
            out = run_op(
                x,
                variant=variant,
                dim="col",
                Ht=Ht,
                Wt=1,
                NC=1,
                accum="fp32",
                kernel_iters=2,
                partial_elems=P,
            )
            got = ttnn.to_torch(out).to(torch.float64)[0, :]  # row 0 holds per-column means
            ma = (got - golden).abs().max().item()
            logger.info(f"partial col {variant:26s} Ht={Ht} L={L} P={P}  max_abs={ma:.5f}")
            assert ma < 0.05, f"{variant} col L={L} P={P}: max_abs {ma:.4f} — garbage leaked"


def test_reduce_block_partial_perf(device):
    """Compute-side ns: accumulate_via_add ALIGNED vs PARTIAL (masked last tile) vs the reduce_tile library
    default, for a wide row (reduce width Wt=8) and a wide col (reduce height Ht=8). Shows the on-device cost
    of the masked broadcast-mul the partial path folds the last reduce-dim tile in with. P=26 -> the last of
    the 8 tiles carries 26 valid elements, so partial does the SAME 8-tile work as aligned, differing only in
    that one tile (add -> masked mul). Writes REDBLK_PARTIAL_REPORT."""
    trials = _int("REDBLK_TRIALS", "5")
    ki = _int("REDBLK_KERNEL_ITERS", "200")
    P = 26
    # (dim, Ht, Wt): a wide reduce along each partial-capable dim (8 reduced tiles).
    cases = [("row", 1, 8), ("col", 8, 1)]
    runners = {}
    for dim, Ht, Wt in cases:
        x, _ = _make_input(device, dim, Ht, Wt, 1)  # perf is data-independent
        runners[("reduce_tile", dim)] = lambda xx=x, d=dim, h=Ht, w=Wt: run_op(
            xx, variant="reduce_tile", dim=d, Ht=h, Wt=w, NC=1, accum="fp32", kernel_iters=ki
        )
        runners[("aligned", dim)] = lambda xx=x, d=dim, h=Ht, w=Wt: run_op(
            xx, variant="accumulate_via_add", dim=d, Ht=h, Wt=w, NC=1, accum="fp32", kernel_iters=ki
        )
        runners[("partial", dim)] = lambda xx=x, d=dim, h=Ht, w=Wt: run_op(
            xx, variant="accumulate_via_add", dim=d, Ht=h, Wt=w, NC=1, accum="fp32", kernel_iters=ki, partial_elems=P
        )
    samples = _measure(device, runners, trials, ki)

    def med(kind, dim):
        return statistics.median(samples[(kind, dim)])

    lines = [
        "# Reduce-block: PARTIAL (non-tile-aligned) reduce cost — masked last tile vs aligned (single core)",
        "",
        f"box={socket.gethostname()}  arch={_arch_label(device)}  cores=1  N={trials} (median)  "
        f"kernel-iters={ki}  fp32 accum",
        "8 reduced tiles per output; partial P=26 -> the last tile carries 26 valid elements and is folded in "
        "with a DEST-accumulating masked broadcast-mul (row-0 mask for ROW, col-0 for COL). Same 8-tile work "
        "as aligned; the delta is that one tile (add -> masked mul). reduce_tile handles the aligned reduce "
        "only here (the library default's partial path uses a scaler, not the mask).",
        "",
        "| dim | reduce_tile ns | acc_via_add aligned ns | acc_via_add partial ns | partial vs aligned |",
        "|---|---:|---:|---:|---:|",
    ]
    for dim, _, _ in cases:
        rt, al, pa = med("reduce_tile", dim), med("aligned", dim), med("partial", dim)
        lines.append(f"| {dim} | {rt:.0f} | {al:.0f} | {pa:.0f} | {pa / al:.2f}× |")
    lines += [
        "",
        "The masked-mul on the single partial tile adds a small fixed cost over the pure-add aligned path "
        "(one FPU broadcast-mul at MATH_FIDELITY vs one add), independent of the reduce width — so the "
        "relative overhead shrinks as the reduce widens. Both partial and aligned stay well under the "
        "reduce_tile library default for a wide reduce.",
    ]
    report = "\n".join(lines) + "\n"
    logger.info("\n" + report)
    for dim, _, _ in cases:
        logger.info(
            f"PARTIAL-PERF {dim:6s}  reduce_tile={med('reduce_tile', dim):.0f}  "
            f"aligned={med('aligned', dim):.0f}  partial={med('partial', dim):.0f} ns"
        )
    if report_path := os.environ.get("REDBLK_PARTIAL_REPORT"):
        Path(report_path).write_text(report)


def test_reduce_block_streaming(device):
    """WaitAndPopPerTile: AccumulateViaAdd STREAMS the reduce dim through DST (DST is the accumulator, only
    2 input tiles resident at a time — no CB accumulator, no reload). Row (incl. multi-output) + scalar
    (contiguous); result must equal the bulk mean."""
    for dim, Ht, Wt, NC in [
        ("row", 1, 8, 1),
        ("row", 1, 5, 1),
        ("row", 2, 4, 1),
        ("row", 3, 5, 1),  # single + multi-output
        ("scalar", 4, 4, 1),
        ("scalar", 1, 7, 1),
    ]:
        x, golden = _make_input(device, dim, Ht, Wt, NC)
        out = run_op(
            x,
            variant="accumulate_via_add",
            dim=dim,
            Ht=Ht,
            Wt=Wt,
            NC=NC,
            accum="fp32",
            kernel_iters=2,
            stream=True,
        )
        got = _readout(out, dim)
        assert got.numel() == golden.numel()
        ma = (got - golden).abs().max().item()
        logger.info(f"streaming {dim:6s} Ht={Ht} Wt={Wt} NC={NC}  max_abs={ma:.5f}")
        assert ma < 0.05, f"streaming {dim} Ht={Ht} Wt={Wt}: max_abs {ma:.4f}"


def _make_input_sum(device, dim, Ht, Wt, NC, num_chunks, seed=13):
    """Input tensor + the SUM golden for the accumulate path: `num_chunks` chunks each re-reduce the SAME
    block, so the result is num_chunks * sum(block, reduce_dim). Golden is taken from the bf16-rounded data
    (fp64) so the only residual error is the on-device fp32 accumulation, isolating the accumulator fold."""
    torch.manual_seed(seed)
    h, w = input_shape(Ht, Wt, NC)
    data_bf16 = torch.rand(h, w).to(torch.bfloat16)
    b = data_bf16.to(torch.float64).view(NC, Ht * TILE, Wt * TILE)
    if dim == "row":
        s = b.sum(dim=2).reshape(-1)  # per-row sums -> flat (NC*Ht*32)
    elif dim == "col":
        s = b.sum(dim=1).reshape(-1)  # per-col sums -> flat (NC*Wt*32)
    else:
        s = b.sum(dim=(1, 2)).reshape(-1)  # one sum per batch (NC)
    golden = s * num_chunks
    x_dev = ttnn.from_torch(
        data_bf16,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=create_sharded_memory_config((h, w)),
    )
    return x_dev, golden


def test_reduce_block_accumulate(device):
    """Cross-call Accumulate over AccumulateViaAdd: the reduce dim is split into `num_chunks` chunks, each
    folding the running RAW partial-sum tile (cb_acc) into the pairwise add NATIVELY (no binary_dest_reuse)
    and finalizing (sfpu_reduce) only on the last chunk. Every chunk re-reduces the same block, so the SUM
    result must equal num_chunks * sum(block, reduce_dim). Cases cover even/odd reduce-tile counts (the parity
    rule: even -> copy-reload seed + add pairs; odd -> pairs with the accumulator as the last add operand),
    N==1, strided col, multi-output, batched, and a single-chunk (is_first && is_last) anchor."""
    cases = [
        # (dim, Ht, Wt, NC, num_chunks)   cnt = reduce-tiles per output = Wt(row)/Ht(col)/Ht*Wt(scalar)
        ("row", 2, 3, 1, 3),  # cnt=3 odd, 2 outputs, 3 chunks
        ("row", 1, 4, 1, 2),  # cnt=4 even
        ("row", 2, 1, 1, 2),  # cnt=1 (odd, N==1 fold: new[0] + accumulator)
        ("row", 2, 3, 2, 2),  # batched, 4 outputs, cnt=3 odd
        ("col", 3, 2, 1, 2),  # cnt=3 odd, strided, 2 outputs
        ("col", 4, 2, 1, 3),  # cnt=4 even, strided
        ("scalar", 2, 2, 1, 2),  # cnt=4 even
        ("scalar", 1, 3, 1, 2),  # cnt=3 odd
        ("row", 1, 5, 1, 1),  # single chunk: is_first && is_last (plain sum)
    ]
    failures = []
    for dim, Ht, Wt, NC, num_chunks in cases:
        x, golden = _make_input_sum(device, dim, Ht, Wt, NC, num_chunks)
        out = run_accumulate(x, dim=dim, Ht=Ht, Wt=Wt, NC=NC, accum="fp32", kernel_iters=2, num_chunks=num_chunks)
        assert list(out.shape) == list(output_shape(dim, Ht, Wt, NC))
        got = _readout(out, dim)
        assert got.numel() == golden.numel(), f"{dim}: {got.numel()} values, expected {golden.numel()}"
        scale = golden.abs().max().clamp_min(1.0).item()
        ma = (got - golden).abs().max().item()
        rel = ma / scale
        cnt = Wt if dim == "row" else (Ht if dim == "col" else Ht * Wt)
        logger.info(
            f"accumulate {dim:6s} Ht={Ht} Wt={Wt} NC={NC} chunks={num_chunks} cnt={cnt} "
            f"{'odd ' if cnt & 1 else 'even'}  max_abs={ma:.4f} rel={rel:.2e}"
        )
        if not (rel < 0.02):
            failures.append(f"{dim} Ht={Ht} Wt={Wt} NC={NC} chunks={num_chunks} cnt={cnt}: rel {rel:.3e}")
    assert not failures, "accumulate mismatches:\n" + "\n".join(failures)
