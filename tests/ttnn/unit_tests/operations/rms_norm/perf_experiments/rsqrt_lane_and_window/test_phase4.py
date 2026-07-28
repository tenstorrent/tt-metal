# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Correctness gate + on-device bake-off for rms_norm PHASE 4 (`AddUnary(eps)` -> `Rsqrt`).

Correctness is the ONLY pass/fail. Perf is measured (`DEVICE KERNEL DURATION [ns]`) and reported,
never asserted.

The correctness gate has three parts, all on the focus config:
  1. PCC >= 0.9995 of output COLUMN 0 against an fp64 torch reference (the focus case's soft gate).
     Column 0 is the whole contract: `cb_rms_sum` is a REDUCE_ROW result whose only meaningful lane
     is column 0, and its consumer is a `BroadcastDim::Col` multiply.
  2. An ABSOLUTE all-ones check with a LARGE eps: input column 0 == 1.0 exactly (what an all-ones
     rms_norm input produces for mean(x^2)) must come back as 1/sqrt(1 + eps) with eps = 0.25, i.e.
     0.8944272. This is scale-SENSITIVE by construction — dropping eps yields 1.0 (11.8% off) and
     double-applying it yields 0.8165 (8.7% off), both far outside bf16 rounding. PCC cannot see
     either (it is scale-invariant), which is exactly why this op has shipped accumulation bugs at
     PCC >= 0.9998.
  3. A POISON check: columns 1..31 of the input are filled with garbage, so any variant that reads
     the wrong lanes cannot pass part 1 by luck.

Plus a separate BCAST PROBE test that runs the op's real consumer primitive
(`mul_tiles_bcast<BroadcastType::COL>`) against a poisoned 1/rms tile to prove the broadcast reads
column 0 ONLY — the precondition that makes the lane-scoped variants safe to graduate.
"""

import os

os.environ.setdefault("TT_METAL_DEVICE_PROFILER", "1")
os.environ.setdefault("TT_METAL_PROFILER_MID_RUN_DUMP", "1")
os.environ.setdefault("TT_METAL_PROFILER_CPP_POST_PROCESS", "1")
os.environ.setdefault("TT_METAL_LOGGER_LEVEL", "error")

import socket
import struct

import pytest
import torch
import ttnn
from loguru import logger

# perf_experiments now lives under tests/ (repo policy: no global torch imports
# under ttnn/, and ttnn/ttnn/operations/__init__.py exec_module()s everything it
# walks at `import ttnn`). Load the sibling bench by file path.
import importlib.util as _ilu
from pathlib import Path as _P

_spec = _ilu.spec_from_file_location("phase4_bench", _P(__file__).resolve().parent / "phase4_bench.py")
phase4_bench = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(phase4_bench)

from phase4_bench import (  # noqa: E402
    ABLATIONS,
    BASELINE,
    LABEL,
    TILE,
    VALID_COLS,
    VARIANTS,
    VECTORS_PER_TILE,
    dtype_of,
    run_bcast_probe,
    run_op,
    sharded_memory_config,
    tensor_height,
)

_DURATION_KEY = "DEVICE KERNEL DURATION [ns]"

# ---- focus geometry: (1,1,8192,1024) BLOCK_SHARDED shard [1024,128] grid (8,8) ----
# derived per core: 32 tile-rows, ht_block = 8, nh_core = 4  =>  4 groups x 8 fp32 tiles.
FOCUS_HT = 8
FOCUS_TILES_PER_CORE = 32
FOCUS_GRID = (8, 8)

EPS_SMALL = 1e-5  # the realistic rms_norm epsilon (perf runs use this)
EPS_BIG = 0.25  # the absolute all-ones probe (bf16-visible, so a dropped eps is unmissable)

PCC_GATE = 0.9995  # the focus case's soft pcc_threshold
_ONES_REL_TOL = 0.01  # bf16 quantum near 0.894 is ~0.4%; 1% catches any real bug, no false alarms


def _eps_bits(eps):
    return struct.unpack("<I", struct.pack("<f", float(eps)))[0]


def _pcc(a, b):
    a = a.flatten().to(torch.float64)
    b = b.flatten().to(torch.float64)
    if torch.allclose(a, b):
        return 1.0
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


# =============================================================================
# Inputs
# =============================================================================
def _make_pair(device, tiles_per_core, grid, in_fmt, out_fmt, *, col0, poison, seed=17):
    """Input/output sharded L1 tensor pair. `col0` fills column 0 of every row; `poison` the rest."""
    gx, gy = grid
    h = tensor_height(tiles_per_core, gx, gy)
    mem = sharded_memory_config(tiles_per_core, gx, gy)

    torch.manual_seed(seed)
    data = poison(h)
    data[:, 0] = col0(h)
    # Quantize to bf16 up front: DEST is bf16 under fp32_dest_acc_en=False, so the device sees the
    # bf16 value regardless of the CB's container format. This makes the golden exact.
    data = data.to(torch.bfloat16).to(torch.float32)

    x = ttnn.from_torch(
        data,
        dtype=dtype_of(in_fmt),
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=mem,
    )
    out = ttnn.allocate_tensor_on_device(ttnn.Shape([h, TILE]), dtype_of(out_fmt), ttnn.TILE_LAYOUT, device, mem)
    return x, out, data


def _random_col0(h):
    # mean(x^2) of a unit-normal row lands near 1; span two decades either way so rsqrt is exercised.
    return torch.exp(torch.randn(h) * 1.5).clamp(1e-3, 1e3)


def _poison(h):
    """Columns 1..31 = garbage a scoped variant must not depend on (all still rsqrt-safe: > 0)."""
    t = torch.full((h, TILE), 7.5e3)
    t[:, 1::2] = 3.0e-4
    return t


def _ones_col0(h):
    return torch.ones(h)


# =============================================================================
# Measurement — one fresh-cache dispatch per variant (device kernel time has no warm-up transient)
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


def _measure(device, run):
    run()
    ttnn.synchronize_device(device)
    ns = _read_kernel_ns(device)
    assert ns is not None, "no profiler data"
    return ns


def _arch_label(device):
    a = str(device.arch()).rsplit(".", 1)[-1]
    return {"WORMHOLE_B0": "WH_B0", "BLACKHOLE": "BH", "GRAYSKULL": "GS"}.get(a, a)


# =============================================================================
# Correctness
# =============================================================================
def _check(variant, out_dev, in_data, eps, *, label):
    out = ttnn.to_torch(out_dev).to(torch.float64)
    gold_col0 = torch.rsqrt(in_data[:, 0].to(torch.float64) + eps)
    pcc = _pcc(out[:, 0], gold_col0)
    assert pcc >= PCC_GATE, f"{label}: column-0 PCC {pcc:.6f} < {PCC_GATE}"
    # Also confirm the variant's DOCUMENTED valid width really is valid (guards the vector-count
    # model in VECTORS_PER_TILE against a silent off-by-one in the scope).
    cols = VALID_COLS[variant]
    if cols > 1:
        gold = torch.rsqrt(in_data[:, :cols].to(torch.float64) + eps)
        pcc_wide = _pcc(out[:, :cols], gold)
        assert pcc_wide >= PCC_GATE, f"{label}: cols[0:{cols}] PCC {pcc_wide:.6f} < {PCC_GATE}"
    return pcc


def _check_ones(variant, out_dev, eps, *, label):
    out = ttnn.to_torch(out_dev).to(torch.float64)[:, 0]
    expected = 1.0 / (1.0 + eps) ** 0.5
    rel = ((out - expected).abs() / expected).max().item()
    assert rel < _ONES_REL_TOL, (
        f"{label}: all-ones absolute check failed — got {out.min().item():.6f}..{out.max().item():.6f}, "
        f"expected {expected:.6f} (rel {rel:.4f} >= {_ONES_REL_TOL}). "
        f"eps dropped => 1.0, eps doubled => {1.0 / (1.0 + 2 * eps) ** 0.5:.6f}."
    )
    return rel


def _variants_from_env():
    sel = os.environ.get("P4_VARIANTS")
    names = list(VARIANTS)
    if not sel:
        return names
    chosen = sel.split(",")
    unknown = set(chosen) - set(names)
    if unknown:
        raise ValueError(f"unknown variants {sorted(unknown)}; valid: {names}")
    return [n for n in names if n in chosen]


def _fmts_from_env():
    return os.environ.get("P4_FMTS", "fp32/fp32").split(",")


# =============================================================================
# Test 1 — correctness of every variant on the focus geometry
# =============================================================================
@pytest.mark.parametrize("fmt", ["fp32/fp32", "fp32/bf16", "bf16/bf16"])
@pytest.mark.parametrize("fp32_dest", [False, True], ids=["dest_bf16", "dest_fp32"])
def test_phase4_correctness(device, fmt, fp32_dest):
    """`fp32_dest` is a CORRECTNESS axis only: the op's SUPPORTED matrix allows fp32_dest_acc_en=True,
    so an unguarded lane-scoped element has to be right there too. Every perf number is taken at the
    focus case's fp32_dest_acc_en=False."""
    in_fmt, out_fmt = fmt.split("/")
    gx, gy = FOCUS_GRID
    x, out, data = _make_pair(
        device, FOCUS_TILES_PER_CORE, FOCUS_GRID, in_fmt, out_fmt, col0=_random_col0, poison=_poison
    )
    x1, out1, _ = _make_pair(device, FOCUS_TILES_PER_CORE, FOCUS_GRID, in_fmt, out_fmt, col0=_ones_col0, poison=_poison)
    for variant in _variants_from_env():
        if variant in ABLATIONS:
            continue  # ablations deliberately produce no valid output
        run_op(
            x,
            out,
            variant=variant,
            ht=FOCUS_HT,
            n_groups=FOCUS_TILES_PER_CORE // FOCUS_HT,
            eps_bits=_eps_bits(EPS_SMALL),
            grid_x=gx,
            grid_y=gy,
            fp32_dest_acc_en=fp32_dest,
        )
        pcc = _check(variant, out, data, EPS_SMALL, label=f"{fmt}/{variant}")

        run_op(
            x1,
            out1,
            variant=variant,
            ht=FOCUS_HT,
            n_groups=FOCUS_TILES_PER_CORE // FOCUS_HT,
            eps_bits=_eps_bits(EPS_BIG),
            grid_x=gx,
            grid_y=gy,
            fp32_dest_acc_en=fp32_dest,
        )
        rel = _check_ones(variant, out1, EPS_BIG, label=f"{fmt}/{variant}")
        logger.info(
            f"fp32_dest={int(fp32_dest)} {fmt:11s} {variant:22s} col0 PCC={pcc:.7f}  " f"all-ones rel-err={rel:.5f}"
        )


# =============================================================================
# Test 2 — the bcast-COL safety probe
# =============================================================================
def test_bcast_col_reads_column_zero_only(device):
    """The op's phase-5 consumer, fed a tile whose columns 1..31 are POISON.

    If `mul_tiles_bcast<COL>` reads any lane other than column 0, this fails — and lane-scoping
    phase 4 would be unsafe. If it passes, scoping to column 0 provably cannot change the op's
    output.
    """
    gx, gy = FOCUS_GRID
    tpc = 8
    h = tensor_height(tpc, gx, gy)
    mem = sharded_memory_config(tpc, gx, gy)

    torch.manual_seed(5)
    recip = _poison(h)
    recip[:, 0] = (0.25 + torch.rand(h)).to(torch.bfloat16).to(torch.float32)
    recip_dev = ttnn.from_torch(recip, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mem)
    ones = torch.ones(h, TILE)
    ones_dev = ttnn.from_torch(ones, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mem)
    out_dev = ttnn.allocate_tensor_on_device(ttnn.Shape([h, TILE]), ttnn.bfloat16, ttnn.TILE_LAYOUT, device, mem)
    run_bcast_probe(recip_dev, ones_dev, out_dev, tiles_per_core=tpc, grid_x=gx, grid_y=gy)

    got = ttnn.to_torch(out_dev).to(torch.float64)
    expected = recip[:, 0].to(torch.float64).unsqueeze(1).expand(-1, TILE)
    rel = ((got - expected).abs() / expected.abs()).max().item()
    logger.info(f"bcast<COL> probe: max rel-err over the WHOLE tile = {rel:.6f} (poison ignored)")
    assert rel < 0.01, (
        f"mul_tiles_bcast<COL> did NOT read column 0 only: max rel-err {rel:.4f}. "
        "Lane-scoping phase 4 to column 0 would be UNSAFE."
    )


# =============================================================================
# Test 3 — the bake-off (correctness-gated, then one measured dispatch per variant)
# =============================================================================
@pytest.mark.parametrize("ht", [int(v) for v in os.environ.get("P4_HTS", "8").split(",")])
def test_phase4_bakeoff(device, ht):
    gx, gy = FOCUS_GRID
    tpc = FOCUS_TILES_PER_CORE
    assert tpc % ht == 0
    n_groups = tpc // ht
    variants = _variants_from_env()
    fmts = _fmts_from_env()
    # Perf is taken at the focus case's fp32_dest_acc_en=False by default. The True setting is
    # measured only to characterise the predicate across the op's other supported DEST mode — it is
    # NEVER varied to gain speed (baseline and candidate always share whatever value is set).
    fp32_dest = os.environ.get("P4_FP32_DEST", "0") == "1"

    rows = []
    for fmt in fmts:
        in_fmt, out_fmt = fmt.split("/")
        x, out, data = _make_pair(device, tpc, FOCUS_GRID, in_fmt, out_fmt, col0=_random_col0, poison=_poison)
        for variant in variants:
            # correctness gate first (a faster wrong answer is disqualified)
            if variant in ABLATIONS:
                pcc = float("nan")  # ablation: scaffolding only, payload removed
            else:
                run_op(
                    x,
                    out,
                    variant=variant,
                    ht=ht,
                    n_groups=n_groups,
                    eps_bits=_eps_bits(EPS_SMALL),
                    grid_x=gx,
                    grid_y=gy,
                    fp32_dest_acc_en=fp32_dest,
                )
                pcc = _check(variant, out, data, EPS_SMALL, label=f"{fmt}/{variant}/ht{ht}")
            ttnn.synchronize_device(device)
            _read_kernel_ns(device)  # drop the gate dispatch's profiler window
            ns = _measure(
                device,
                lambda v=variant: run_op(
                    x,
                    out,
                    variant=v,
                    ht=ht,
                    n_groups=n_groups,
                    eps_bits=_eps_bits(EPS_SMALL),
                    grid_x=gx,
                    grid_y=gy,
                    fp32_dest_acc_en=fp32_dest,
                ),
            )
            rows.append((fmt, variant, ns, pcc))

    base = {fmt: ns for fmt, v, ns, _ in rows if v == BASELINE}
    logger.info(
        f"\n=== phase-4 bake-off: box={socket.gethostname()} arch={_arch_label(device)} "
        f"grid={gx}x{gy} ({gx * gy} cores) tiles/core={tpc} ht={ht} groups={n_groups} "
        f"HiFi2/fp32_dest_acc={fp32_dest}/approx=False ==="
    )
    logger.info(
        f"{'fmt':11s} {'variant':22s} {'vec/tile':>8s} {'ns':>9s} {'ns/tile':>8s} {'vs base':>8s} {'col0 PCC':>10s}  how"
    )
    for fmt, variant, ns, pcc in rows:
        b = base.get(fmt)
        spd = f"{b / ns:.2f}x" if b else "-"
        logger.info(
            f"{fmt:11s} {variant:22s} {VECTORS_PER_TILE[variant]:8d} {ns:9.0f} {ns / tpc:8.1f} "
            f"{spd:>8s} {pcc:10.7f}  {LABEL[variant]}"
        )
