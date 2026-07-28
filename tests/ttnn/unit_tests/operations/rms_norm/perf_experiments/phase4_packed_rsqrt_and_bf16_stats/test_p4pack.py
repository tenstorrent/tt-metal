# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Correctness gate + on-device bake-off for rms_norm PHASE 4's packed-rsqrt / bf16-stats
follow-ups (Perf 2). See `bench.py` for the full mechanism writeup.

Correctness is the ONLY pass/fail. Perf is measured (`DEVICE KERNEL DURATION [ns]`) and
reported, never asserted.

Gate, all on the focus geometry:
  1. FULL-TILE check (not a column slice, per the task's explicit trap): for every h in
     0..ht-1, output tile h is compared, over its WHOLE 32x32 extent, against the golden
     tile (column 0 = rsqrt(mean_h + eps), columns 1..31 = 0). This is what catches a wrong
     extract scaler landing the right answer in the wrong TILE-ROW -- a column-0-only check
     would happily miss that (the task's own words).
  2. ABSOLUTE all-ones check, large eps: input column 0 == 1.0 exactly must come back as
     1/sqrt(1 + eps).
  3. POISON check: columns 1..31 of the input carry garbage the packed path must not read
     (PICK0's scaler structurally selects only column 0, but this is the empirical check).
"""

import os
import struct
import socket

os.environ.setdefault("TT_METAL_DEVICE_PROFILER", "1")
os.environ.setdefault("TT_METAL_PROFILER_MID_RUN_DUMP", "1")
os.environ.setdefault("TT_METAL_PROFILER_CPP_POST_PROCESS", "1")
os.environ.setdefault("TT_METAL_LOGGER_LEVEL", "error")

import pytest
import torch
import ttnn
from loguru import logger

import importlib.util as _ilu
from pathlib import Path as _P
import sys as _sys

_spec = _ilu.spec_from_file_location("p4pack_bench", _P(__file__).resolve().parent / "bench.py")
p4pack_bench = _ilu.module_from_spec(_spec)
_sys.modules["p4pack_bench"] = p4pack_bench
_spec.loader.exec_module(p4pack_bench)

from p4pack_bench import (  # noqa: E402
    CSKIP_VARIANTS,
    LABEL,
    TILE,
    VARIANTS,
    _SCOPE_C,
    _SCOPE_CSKIP,
    build_colsel_bank,
    build_pick0_bank,
    core_range_set,
    dtype_of,
    pack_columns,
    run_op,
    sharded_memory_config,
    tensor_height,
)

_DURATION_KEY = "DEVICE KERNEL DURATION [ns]"

FOCUS_GRID = (8, 8)
FOCUS_TILES_PER_CORE = 32
EPS_SMALL = 1e-5
EPS_BIG = 0.25
PCC_GATE = 0.9995
_ONES_REL_TOL = 0.01


def _eps_bits(eps):
    return struct.unpack("<I", struct.pack("<f", float(eps)))[0]


def _pcc(a, b):
    a = a.flatten().to(torch.float64)
    b = b.flatten().to(torch.float64)
    if torch.allclose(a, b):
        return 1.0
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


# =============================================================================
# Tensor construction. `num_cores` copies of the same per-core content ride the
# HEIGHT-sharded grid; PICK0/COLSEL are the SAME constant tile bank on every core.
# =============================================================================
def _random_col0(n):
    return torch.exp(torch.randn(n) * 1.5).clamp(1e-3, 1e3)


def _poison(n_rows):
    t = torch.full((n_rows, TILE), 7.5e3)
    t[:, 1::2] = 3.0e-4
    return t


def _make_bundle(device, ht, n_groups, grid, in_bf16, out_bf16, scope, *, col0_fn=_random_col0, seed=17):
    gx, gy = grid
    num_cores = gx * gy
    tiles_per_core = ht * n_groups
    h_in = tensor_height(tiles_per_core, gx, gy)

    torch.manual_seed(seed)
    col0_full = col0_fn(h_in)
    data = _poison(h_in)
    data[:, 0] = col0_full
    data = data.to(torch.bfloat16).to(torch.float32)  # DEST is bf16 either way at fp32_dest_acc_en=False

    in_mem = sharded_memory_config(tiles_per_core, gx, gy)
    in_tensor = ttnn.from_torch(
        data, dtype=dtype_of(in_bf16), layout=ttnn.TILE_LAYOUT, device=device, memory_config=in_mem
    )

    # packed_in: derived PER CORE from the SAME col0 values `in` carries, so pack_given's
    # correctness does not depend on pack_here's own PICK0 reduce.
    per_core_col0 = data[:, 0].reshape(num_cores, tiles_per_core * TILE)
    packed_rows = []
    for c in range(num_cores):
        packed_rows.append(pack_columns(per_core_col0[c], ht, n_groups, scope))
    packed_data = torch.cat(packed_rows, dim=0)  # [num_cores*n_groups*32, 32]
    packed_mem = sharded_memory_config(n_groups, gx, gy)
    packed_in_tensor = ttnn.from_torch(
        packed_data, dtype=dtype_of(in_bf16), layout=ttnn.TILE_LAYOUT, device=device, memory_config=packed_mem
    )

    pick0_1core = build_pick0_bank(ht, scope)
    colsel_1core = build_colsel_bank(ht, scope)
    pick0_full = pick0_1core.repeat(num_cores, 1)
    colsel_full = colsel_1core.repeat(num_cores, 1)
    sel_mem = sharded_memory_config(ht, gx, gy)
    pick0_tensor = ttnn.from_torch(
        pick0_full, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device, memory_config=sel_mem
    )
    colsel_tensor = ttnn.from_torch(
        colsel_full, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device, memory_config=sel_mem
    )

    out_tensor = ttnn.allocate_tensor_on_device(
        ttnn.Shape([h_in, TILE]), dtype_of(out_bf16), ttnn.TILE_LAYOUT, device, in_mem
    )

    return {
        "data": data,  # [h_in, 32] torch, column 0 = ground truth stat, rest = poison
        "in": in_tensor,
        "packed_in": packed_in_tensor,
        "pick0": pick0_tensor,
        "colsel": colsel_tensor,
        "out": out_tensor,
        "num_cores": num_cores,
        "tiles_per_core": tiles_per_core,
    }


def _dealloc_bundle(bundle):
    """Every bundle tensor is L1-resident (zero-copy sharded); a correctness sweep over many
    variants without freeing them exhausts L1 and TT_FATALs in the allocator."""
    for key in ("in", "packed_in", "pick0", "colsel", "out"):
        t = bundle.get(key)
        if t is not None:
            ttnn.deallocate(t)


def _tensors_for_variant(bundle, variant):
    mode = VARIANTS[variant][0]
    out = {"out": bundle["out"]}
    if mode == p4pack_bench._MODE_PACK_GIVEN:
        out["packed_in"] = bundle["packed_in"]
        out["colsel"] = bundle["colsel"]
    else:
        out["in"] = bundle["in"]
        out["colsel"] = bundle["colsel"]
        if mode == p4pack_bench._MODE_PACK_HERE:
            out["pick0"] = bundle["pick0"]
    return out


# =============================================================================
# Correctness
# =============================================================================
def _check_full_tile(variant, out_dev, data, eps, tiles_per_core, num_cores, *, label):
    """Per the task: assert over the WHOLE output tile, not a column slice -- for the
    PACKED variants, whose COLSEL extract runs under the reduce packer's default edge mask
    and therefore legitimately zeros columns 1..31 (a wrong extract scaler would instead
    put the right rsqrt value in the WRONG tile-row, which this full-tile-per-h check
    catches and a column-0-only check would not).

    `baseline` is NOT put through the full-tile check: its `RsqrtAddUnaryColZero` element
    (round 1, unchanged here) never claimed columns 1..31 -- they hold whatever `CopyTile`
    copied in (poison, by construction of this test) -- so column 0 is its whole contract,
    exactly as round 1's own gate checks it.
    """
    mode = VARIANTS[variant][0]
    out = ttnn.to_torch(out_dev).to(torch.float64)  # [num_cores*tiles_per_core*32, 32]
    data64 = data.to(torch.float64)
    worst_pcc = 1.0
    for c in range(num_cores):
        for h in range(tiles_per_core):
            row0 = (c * tiles_per_core + h) * TILE
            out_tile = out[row0 : row0 + TILE, :]
            col0_in = data64[row0 : row0 + TILE, 0]
            gold_col0 = torch.rsqrt(col0_in + eps)
            if mode == p4pack_bench._MODE_BASELINE:
                pcc = _pcc(out_tile[:, 0], gold_col0)
            else:
                gold_tile = torch.zeros(TILE, TILE, dtype=torch.float64)
                gold_tile[:, 0] = gold_col0
                pcc = _pcc(out_tile, gold_tile)
            worst_pcc = min(worst_pcc, pcc)
            assert pcc >= PCC_GATE, (
                f"{label}: core {c} tile-row {h} PCC {pcc:.6f} < {PCC_GATE} "
                f"(checks the extract landed in the RIGHT tile-row, not just SOME column-0 value)"
            )
    return worst_pcc


def _check_ones(variant, out_dev, eps, tiles_per_core, num_cores, *, label):
    out = ttnn.to_torch(out_dev).to(torch.float64)[:, 0]
    expected = 1.0 / (1.0 + eps) ** 0.5
    rel = ((out - expected).abs() / expected).max().item()
    assert rel < _ONES_REL_TOL, f"{label}: all-ones absolute check failed, rel={rel:.4f}"
    return rel


# =============================================================================
# Test 1 — correctness, focus geometry, every variant, both DEST modes
# =============================================================================
@pytest.mark.parametrize("fp32_dest", [False, True], ids=["dest_bf16", "dest_fp32"])
def test_p4pack_correctness(device, fp32_dest):
    gx, gy = FOCUS_GRID
    ht, n_groups = 8, FOCUS_TILES_PER_CORE // 8
    for variant in VARIANTS:
        scope = VARIANTS[variant][1]
        if variant in CSKIP_VARIANTS and ht > 8:
            continue
        in_bf16 = out_bf16 = bool(VARIANTS[variant][2])
        bundle = _make_bundle(device, ht, n_groups, FOCUS_GRID, in_bf16=in_bf16, out_bf16=out_bf16, scope=scope)
        run_op(
            _tensors_for_variant(bundle, variant),
            variant=variant,
            ht=ht,
            n_groups=n_groups,
            eps_bits=_eps_bits(EPS_SMALL),
            grid_x=gx,
            grid_y=gy,
            fp32_dest_acc_en=fp32_dest,
        )
        pcc = _check_full_tile(variant, bundle["out"], bundle["data"], EPS_SMALL, n_groups * ht, gx * gy, label=variant)

        bundle1 = _make_bundle(
            device, ht, n_groups, FOCUS_GRID, in_bf16=in_bf16, out_bf16=out_bf16, scope=scope, col0_fn=torch.ones
        )
        run_op(
            _tensors_for_variant(bundle1, variant),
            variant=variant,
            ht=ht,
            n_groups=n_groups,
            eps_bits=_eps_bits(EPS_BIG),
            grid_x=gx,
            grid_y=gy,
            fp32_dest_acc_en=fp32_dest,
        )
        rel = _check_ones(variant, bundle1["out"], EPS_BIG, n_groups * ht, gx * gy, label=variant)
        logger.info(f"fp32_dest={int(fp32_dest)} {variant:20s} full-tile PCC={pcc:.7f} all-ones rel={rel:.5f}")
        _dealloc_bundle(bundle)
        _dealloc_bundle(bundle1)


@pytest.mark.parametrize("ht", [1, 2, 4, 16])
def test_p4pack_correctness_predicate_sweep(device, ht):
    gx, gy = FOCUS_GRID
    n_groups = FOCUS_TILES_PER_CORE // ht
    for variant in ("baseline", "pack_here_c", "pack_given_c"):
        scope = VARIANTS[variant][1]
        bundle = _make_bundle(device, ht, n_groups, FOCUS_GRID, in_bf16=False, out_bf16=False, scope=scope)
        run_op(
            _tensors_for_variant(bundle, variant),
            variant=variant,
            ht=ht,
            n_groups=n_groups,
            eps_bits=_eps_bits(EPS_SMALL),
            grid_x=gx,
            grid_y=gy,
        )
        pcc = _check_full_tile(variant, bundle["out"], bundle["data"], EPS_SMALL, n_groups * ht, gx * gy, label=variant)
        logger.info(f"ht={ht} {variant:20s} full-tile PCC={pcc:.7f}")
        _dealloc_bundle(bundle)


# =============================================================================
# Test 2 — bit-exactness of sub-lever (a) at fp32_dest_acc_en=False, and the guard at True
# =============================================================================
def test_bf16_stat_bit_exactness_and_guard(device):
    gx, gy = FOCUS_GRID
    ht, n_groups = 8, 4
    for fp32_dest, expect_exact in ((False, True), (True, False)):
        bundle_fp32 = _make_bundle(device, ht, n_groups, FOCUS_GRID, in_bf16=False, out_bf16=False, scope=_SCOPE_C)
        bundle_bf16 = _make_bundle(device, ht, n_groups, FOCUS_GRID, in_bf16=True, out_bf16=True, scope=_SCOPE_C)
        # Same underlying data (same seed) -> same column-0 statistics.
        for variant, bundle in (("baseline", bundle_fp32), ("baseline", bundle_bf16)):
            run_op(
                _tensors_for_variant(bundle, variant),
                variant=variant,
                ht=ht,
                n_groups=n_groups,
                eps_bits=_eps_bits(EPS_SMALL),
                grid_x=gx,
                grid_y=gy,
                fp32_dest_acc_en=fp32_dest,
            )
        out_fp32 = ttnn.to_torch(bundle_fp32["out"]).to(torch.float64)
        out_bf16 = ttnn.to_torch(bundle_bf16["out"]).to(torch.float64)
        max_abs_diff = (out_fp32 - out_bf16).abs().max().item()
        logger.info(
            f"fp32_dest_acc_en={fp32_dest} fp32-CB vs bf16-CB max|diff|={max_abs_diff:.3e} "
            f"(expect {'0 (bit-exact)' if expect_exact else '> 0 (load-bearing)'})"
        )
        _dealloc_bundle(bundle_fp32)
        _dealloc_bundle(bundle_bf16)
        if expect_exact:
            assert max_abs_diff == 0.0, (
                f"sub-lever (a): bf16 stat CB is NOT bit-exact at fp32_dest_acc_en=False "
                f"(max|diff|={max_abs_diff:.3e}) -- narrowing is unsafe here too, contrary to the argument."
            )
        else:
            assert max_abs_diff > 0.0, (
                "sub-lever (a) GUARD CHECK FAILED: bf16 stat CB is bit-exact even at "
                "fp32_dest_acc_en=True -- the guard may be unnecessary (re-examine before trusting "
                "the narrowing is unconditionally safe)."
            )


# =============================================================================
# Test 3 — the bake-off (correctness-gated already by test 1; measured dispatch here)
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


def _variants_from_env():
    sel = os.environ.get("P4PACK_VARIANTS")
    names = list(VARIANTS)
    if not sel:
        return names
    chosen = sel.split(",")
    return [n for n in names if n in chosen]


@pytest.mark.parametrize("ht", [int(v) for v in os.environ.get("P4PACK_HTS", "8").split(",")])
def test_p4pack_bakeoff(device, ht):
    gx, gy = FOCUS_GRID
    tpc = FOCUS_TILES_PER_CORE
    assert tpc % ht == 0
    n_groups = tpc // ht
    variants = [v for v in _variants_from_env() if not (v in CSKIP_VARIANTS and ht > 8)]

    rows = []
    for variant in variants:
        scope = VARIANTS[variant][1]
        in_bf16 = out_bf16 = bool(VARIANTS[variant][2])
        bundle = _make_bundle(device, ht, n_groups, FOCUS_GRID, in_bf16=in_bf16, out_bf16=out_bf16, scope=scope)
        tensors = _tensors_for_variant(bundle, variant)
        # correctness gate first (a faster wrong answer is disqualified)
        run_op(tensors, variant=variant, ht=ht, n_groups=n_groups, eps_bits=_eps_bits(EPS_SMALL), grid_x=gx, grid_y=gy)
        pcc = _check_full_tile(variant, bundle["out"], bundle["data"], EPS_SMALL, n_groups * ht, gx * gy, label=variant)
        ttnn.synchronize_device(device)
        _read_kernel_ns(device)  # drop the gate dispatch's profiler window
        ns = _measure(
            device,
            lambda v=variant, t=tensors: run_op(
                t, variant=v, ht=ht, n_groups=n_groups, eps_bits=_eps_bits(EPS_SMALL), grid_x=gx, grid_y=gy
            ),
        )
        rows.append((variant, ns, pcc))
        _dealloc_bundle(bundle)

    base = {v: ns for v, ns, _ in rows if v == "baseline"}.get("baseline")
    logger.info(f"\n=== phase4-packed bake-off: box={socket.gethostname()} grid={gx}x{gy} tpc={tpc} ht={ht} ===")
    logger.info(f"{'variant':20s} {'ns':>9s} {'ns/tile':>8s} {'vs base':>8s} {'PCC':>10s}  how")
    for variant, ns, pcc in rows:
        spd = f"{base / ns:.3f}x" if base else "-"
        logger.info(f"{variant:20s} {ns:9.0f} {ns / tpc:8.1f} {spd:>8s} {pcc:10.7f}  {LABEL[variant]}")
