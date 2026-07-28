# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Isolated bake-off: fusing rms_norm's phase 5 (`x * 1/rms`) and phase 6 (`* gamma`).

Correctness is the ONLY pass/fail. Perf is measured (DEVICE KERNEL DURATION [ns],
read in-process via `ttnn.ReadDeviceProfiler`) and reported, never asserted.

Three gates per variant, because every accumulation/expansion bug this op has
shipped scored PCC >= 0.9998 (an error that only rescales rows or columns is
invisible to a scale-invariant metric):

  1. PCC        >= 0.9995 vs a torch reference on random data
  2. ABSOLUTE   x = 1, gamma = 1, 1/rms = 1/sqrt(1+eps)  =>  out == that, EXACTLY
  3. RAMP GAMMA a per-column-distinct gamma, so a pre-expansion that replicates
                the wrong axis (which passes gate 2 perfectly) fails here

Run:
    scripts/run_safe_pytest.sh --run-all \
        tests/ttnn/unit_tests/operations/rms_norm/perf_experiments/scale_gamma_dest_fusion/test_scale_gamma_fusion.py
"""

from __future__ import annotations

import os

# Device profiler on, in-process (no tracy wrapper needed). Must precede `import ttnn`.
os.environ.setdefault("TT_METAL_DEVICE_PROFILER", "1")
os.environ.setdefault("TT_METAL_PROFILER_MID_RUN_DUMP", "1")
os.environ.setdefault("TT_METAL_PROFILER_CPP_POST_PROCESS", "1")

import importlib.util
import math
from pathlib import Path

import pytest
import torch
import ttnn
from loguru import logger

from eval.sharding import shard_config

# Loaded by path (not as a package) so this experiment dir needs no __init__.py
# anywhere outside itself — parallel sibling experiments share `perf_experiments/`.
_spec = importlib.util.spec_from_file_location("sgdf_bench", Path(__file__).parent / "bench.py")
bench = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bench)

TILE = bench.TILE
_DURATION_KEY = "DEVICE KERNEL DURATION [ns]"
PCC_THRESHOLD = 0.9995

# The absolute gates use `1/rms = 0.5` (the exact analogue of `1/sqrt(1+eps)` for
# eps = 3) and gamma values on the exact bf16 grid `1 + k/64`, so `x * (1/rms) *
# gamma` is representable EXACTLY at every step of the datapath — DEST (bf16 here,
# since fp32_dest_acc_en=False), the intermediate CB and the output. That is what
# lets the gate be an exact equality instead of a tolerance: a dropped, duplicated
# or wrong-axis factor cannot hide inside a rounding allowance. (An eps whose
# 1/sqrt is NOT bf16-exact makes the two datapaths disagree by one bf16 step,
# which is a precision artifact, not the class of bug this gate exists for.)
ABS_RCP = 0.5


# =============================================================================
# tensors
# =============================================================================


def _mc(shard_shape, *, dtype, layout, device):
    return shard_config(
        shard_shape,
        (bench.GRID_X, bench.GRID_Y),
        ttnn.TensorMemoryLayout.BLOCK_SHARDED,
        layout=layout,
        dtype=dtype,
        device=device,
    )


def _to_dev(t, shape, shard_shape, *, dtype, device, layout=ttnn.TILE_LAYOUT):
    return ttnn.from_torch(
        t.reshape(shape).contiguous(),
        dtype=dtype,
        layout=layout,
        device=device,
        memory_config=_mc(shard_shape, dtype=dtype, layout=layout, device=device),
    )


class Case:
    """One (regime, data-kind) case: the device tensors plus the torch golden."""

    def __init__(self, device, regime, *, kind, recip_dtype, seed=7):
        shapes = bench.regime_shapes(regime)
        _hs, _wt, _ht_block, has_gamma, _xr, is_rm = bench.regime_geometry(regime)
        (_, _, H, W) = shapes["x"][0]
        self.regime = regime
        self.kind = kind
        self.has_gamma = has_gamma

        torch.manual_seed(seed)
        x = torch.randn(H, W) * 1.5 if kind == "pcc" else torch.ones(H, W)
        x_q = x.to(torch.bfloat16).to(torch.float32)

        # ---- 1/rms (a per-row column) --------------------------------------
        if kind == "pcc":
            rcp = 1.0 / torch.sqrt(x_q.pow(2).mean(dim=-1, keepdim=True) + 1e-6)
        elif kind == "ones":
            rcp = torch.full((H, 1), ABS_RCP)
        else:  # "ramp": 1/rms varies PER ROW, so a row/column mix-up on the
            # broadcast operand is caught as well as one on gamma.
            rcp = (ABS_RCP * 0.5 ** (torch.arange(H, dtype=torch.float32) % 4)).reshape(H, 1)
        # The recip tensor is one W-tile per core, so the column is replicated across
        # the tile's 32 columns AND across the 8 column bands; only column 0 of each
        # tile is read (BroadcastDim::Col).
        recip_shape, recip_shard = shapes["recip"]
        recip_full = rcp.expand(H, recip_shape[-1])
        # bfloat16 recip is the "shared srcA/srcB format" option: rounding here IS the
        # precision cost being priced, so the golden keeps the value the kernel sees.
        rcp_used = rcp if recip_dtype == ttnn.float32 else rcp.to(torch.bfloat16).to(torch.float32)

        # ---- gamma (a per-column row) --------------------------------------
        if not has_gamma or kind == "ones":
            g = torch.ones(W)
        elif kind == "pcc":
            g = torch.randn(W) * 0.5 + 1.0
        else:  # "ramp": distinct per column and per W-tile, exactly bf16-representable
            g = 1.0 + (torch.arange(W, dtype=torch.float32) % 64) / 64.0
        g_q = g.to(torch.bfloat16).to(torch.float32)

        gamma_shape, gamma_shard = shapes["gamma"]
        gH = gamma_shape[-2]
        # Row 0 of every shard band carries gamma; every other row is POISON, so a
        # pre-expansion that reads the wrong row (or forgets to expand) is caught.
        g_rows = torch.full((gH, W), -7.0)
        g_rows[0::TILE, :] = g_q
        g_rows_full = g_q.expand(gH, W)  # what `fused_pre` is handed

        # ---- device tensors -------------------------------------------------
        self.x = _to_dev(x, shapes["x"][0], shapes["x"][1], dtype=ttnn.bfloat16, device=device)
        self.recip = _to_dev(recip_full, recip_shape, recip_shard, dtype=recip_dtype, device=device)
        self.gamma = None
        self.gamma_full = None
        if has_gamma:
            self.gamma = _to_dev(g_rows, gamma_shape, gamma_shard, dtype=ttnn.bfloat16, device=device)
            self.gamma_full = _to_dev(g_rows_full, gamma_shape, gamma_shard, dtype=ttnn.bfloat16, device=device)

        # The output shard is TILE-layout in EVERY regime, including `rm`: there the
        # untilize is an extra downstream consumer of the same (aliased) cb_out, so
        # the tiled result is still written and the numeric gate is unchanged.
        self.out = _to_dev(
            torch.zeros(H, W),
            shapes["out"][0],
            shapes["out"][1],
            dtype=ttnn.bfloat16,
            device=device,
            layout=ttnn.TILE_LAYOUT,
        )

        self.golden = (x_q * rcp_used * g_q).reshape(shapes["out"][0])

    def gamma_for(self, variant):
        if not self.has_gamma:
            return None
        return self.gamma_full if variant == "fused_pre" else self.gamma

    def dealloc(self):
        for t in (self.x, self.recip, self.gamma, self.gamma_full, self.out):
            if t is not None:
                ttnn.deallocate(t)


def _pcc(a, b):
    a = a.flatten().to(torch.float32)
    b = b.flatten().to(torch.float32)
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


# =============================================================================
# measurement — one dispatch per number, no trial loop
# =============================================================================


def _kernel_ns(device):
    ttnn.ReadDeviceProfiler(device)
    total, found = 0.0, False
    for programs in (ttnn.get_latest_programs_perf_data() or {}).values():
        for program in programs:
            entry = (getattr(program, "program_analyses_results", None) or {}).get(_DURATION_KEY)
            if entry is not None:
                total += float(entry.duration)
                found = True
    return total if found else None


def _run(device, case, variant):
    out = bench.run(case.x, case.recip, case.gamma_for(variant), case.out, variant=variant, regime=case.regime)
    ttnn.synchronize_device(device)
    return out, _kernel_ns(device)


def _gate(case, out, label):
    actual = ttnn.to_torch(out).to(torch.float32).reshape(case.golden.shape)
    expected = case.golden
    pcc = _pcc(actual, expected)
    if case.kind == "pcc":
        assert pcc >= PCC_THRESHOLD, f"{label}: PCC {pcc:.6f} < {PCC_THRESHOLD}"
        return pcc
    # ABSOLUTE gate: the reference rounded to the output dtype, compared exactly.
    ref = expected.to(torch.bfloat16).to(torch.float32)
    bad = actual != ref
    assert not bool(bad.any()), (
        f"{label}: absolute check failed on {int(bad.sum())}/{bad.numel()} elements; "
        f"actual={actual[bad].flatten()[:4].tolist()} expected={ref[bad].flatten()[:4].tolist()}"
    )
    return pcc


def _filter(variants):
    """`SGDF_VARIANTS=fused,fused_pre` narrows a run (debugging / re-measuring one option)."""
    sel = os.environ.get("SGDF_VARIANTS")
    if not sel:
        return list(variants)
    want = sel.split(",")
    return [v for v in variants if v in want]


def _variants_for(regime):
    _hs, _wt, _hb, has_gamma, _xr, _rm = bench.regime_geometry(regime)
    return _filter(("baseline",) if not has_gamma else ("baseline", "fused", "bcast_free"))


# =============================================================================
# tests
# =============================================================================

# (variant, recip CB dtype). recip in bfloat16 shares srcA's format and halves the
# broadcast operand's bytes — a PRECISION option (1/rms carried in bf16), reported
# with its cost, never silently chosen.
_MENU = (
    ("baseline", ttnn.float32),
    ("fused", ttnn.float32),
    ("fused_pre", ttnn.float32),
    ("fused_srcb", ttnn.float32),
    ("bcast_free", ttnn.float32),
    ("fused_sfpu", ttnn.float32),
    ("baseline_blk1", ttnn.float32),
    ("baseline", ttnn.bfloat16),
    ("fused", ttnn.bfloat16),
    # `fused_norc` turns every dtype reconfig off, which is only legal when srcA
    # and srcB share one format — i.e. only with a bfloat16 cb_rms_recip.
    ("fused_norc", ttnn.bfloat16),
    ("bcast_free", ttnn.bfloat16),
)


@pytest.mark.parametrize("recip_dtype", [ttnn.float32, ttnn.bfloat16], ids=["recip_fp32", "recip_bf16"])
def test_option_menu(device, recip_dtype):
    """The option menu on the FOCUS shape: one dispatch per option, all three gates."""
    variants = [v for (v, d) in _MENU if d == recip_dtype]
    variants = _filter(variants)
    rows = []
    for kind in ("pcc", "ones", "ramp"):
        case = Case(device, "focus", kind=kind, recip_dtype=recip_dtype)
        for variant in variants:
            out, ns = _run(device, case, variant)
            pcc = _gate(case, out, f"focus/{variant}/{recip_dtype}/{kind}")
            rows.append((kind, variant, ns, pcc))
        case.dealloc()
    for kind, variant, ns, pcc in rows:
        if kind != "pcc":
            continue
        logger.info(
            f"OPTION focus {variant:10s} recip={str(recip_dtype):22s} {ns:9.0f} ns  pcc={pcc:.6f}  "
            f"L1={bench.l1_bytes('focus', variant, recip_dtype=recip_dtype)} B"
        )


def test_debug_expand(device):
    """Diagnostic: with x == 1 and 1/rms == 0.5, out[r, c] == 0.5 * gamma_full[r % 32, c],
    so the output IS a readout of whatever the gamma pre-expansion produced."""
    regime = os.environ.get("SGDF_REGIME", "focus")
    _hs, wt, _hb, _g, _xr, _rm = bench.regime_geometry(regime)
    case = Case(device, regime, kind="ramp", recip_dtype=ttnn.float32)
    for variant in _filter(os.environ.get("SGDF_VARIANTS", "fused_pre,fused").split(",")):
        out, _ns = _run(device, case, variant)
        a = ttnn.to_torch(out).to(torch.float32).reshape(case.golden.shape)[0, 0]
        logger.info(f"DEBUG {variant}: out[0, :6]={a[0, :6].tolist()}")
        logger.info(f"DEBUG {variant}: out[1, :6]={a[1, :6].tolist()}")
        logger.info(f"DEBUG {variant}: out[2, :6]={a[2, :6].tolist()}")
        logger.info(f"DEBUG {variant}: out[31,:6]={a[31, :6].tolist()}")
        logger.info(f"DEBUG {variant}: out[:6, 0]={a[:6, 0].tolist()}")
        logger.info(f"DEBUG {variant}: golden[0,:6]={case.golden[0, 0, 0, :6].tolist()}")
        logger.info(f"DEBUG {variant}: golden[1,:6]={case.golden[0, 0, 1, :6].tolist()}")
        g = case.golden[0, 0]
        bad = (a != g).float()
        logger.info(f"DEBUG {variant}: exact-mismatch frac={bad.mean().item():.6f}")
        if bad.any():
            rows = bad.sum(dim=1)
            cols = bad.sum(dim=0)
            logger.info(f"DEBUG {variant}: bad-per-row[0:40]={rows[:40].int().tolist()}")
            logger.info(f"DEBUG {variant}: bad-per-col[0:40]={cols[:40].int().tolist()}")
    case.dealloc()

    case = Case(device, regime, kind="pcc", recip_dtype=ttnn.float32)
    for variant in _filter(os.environ.get("SGDF_VARIANTS", "fused_pre,fused").split(",")):
        out, _ns = _run(device, case, variant)
        a = ttnn.to_torch(out).to(torch.float32).reshape(case.golden.shape)[0, 0]
        g = case.golden[0, 0]
        logger.info(f"DEBUG-PCC {variant}: overall pcc={_pcc(a, g):.6f}")
        for rb in range(min(4, a.shape[0] // 32)):
            sl = slice(rb * 32, (rb + 1) * 32)
            logger.info(f"DEBUG-PCC {variant}: row-block {rb} pcc={_pcc(a[sl], g[sl]):.6f}")
        for c in range(0, wt):
            sl = slice(c * 32, (c + 1) * 32)
            logger.info(f"DEBUG-PCC {variant}: w-tile {c} pcc={_pcc(a[:, sl], g[:, sl]):.6f}")
        logger.info(f"DEBUG-PCC {variant}: a[0,:6]={a[0, :6].tolist()}")
        logger.info(f"DEBUG-PCC {variant}: g[0,:6]={g[0, :6].tolist()}")
        logger.info(f"DEBUG-PCC {variant}: ratio[0,:6]={(a[0, :6] / g[0, :6]).tolist()}")
        logger.info(f"DEBUG-PCC {variant}: ratio[1,:6]={(a[1, :6] / g[1, :6]).tolist()}")
        logger.info(f"DEBUG-PCC {variant}: ratio[:,0][:6]={(a[:6, 0] / g[:6, 0]).tolist()}")
    case.dealloc()


def test_op_reference_zones(device):
    """Run the REAL op once on the focus config (read-only) so its per-stage zones can be
    compared with this bench's isolated ones. Attributes `cmp_scale`'s 17.6 us: if the
    rsqrt (`cmp_rsqrt`) is MATH-bound and its UNPACK thread is idle, then `cmp_scale`'s
    UNPACK number is that thread parked in `cb_wait_front(cb_rms_recip, ht)` — a
    pipeline-skew stall, not phase-5 unpack work."""
    from ttnn.operations.rms_norm import rms_norm

    shape = (1, 1, 8192, 1024)
    mc = _mc((1024, 128), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    torch.manual_seed(7)
    x = torch.randn(*shape) * 1.5
    g = torch.randn(1, 1, 1, shape[-1]) * 0.5 + 1.0
    tt_x = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mc)
    tt_g = ttnn.from_torch(g, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    out = rms_norm(tt_x, gamma=tt_g, compute_kernel_config=bench.compute_config(), memory_config=mc)
    ttnn.synchronize_device(device)
    ns = _kernel_ns(device)
    ref = (
        (x.to(torch.bfloat16).to(torch.float32))
        / torch.sqrt(x.to(torch.bfloat16).to(torch.float32).pow(2).mean(-1, keepdim=True) + 1e-6)
        * g.to(torch.bfloat16).to(torch.float32)
    )
    pcc = _pcc(ttnn.to_torch(out).to(torch.float32), ref)
    logger.info(f"OP-REF whole rms_norm focus config: {ns:.0f} ns  pcc={pcc:.6f}")
    assert pcc >= PCC_THRESHOLD, f"op reference PCC {pcc:.6f}"


@pytest.mark.parametrize("regime", list(bench.REGIMES))
def test_regime_sweep(device, regime):
    """The predicate sweep: baseline vs candidates on each regime boundary."""
    rows = []
    for kind in ("pcc", "ramp"):
        case = Case(device, regime, kind=kind, recip_dtype=ttnn.float32)
        for variant in _variants_for(regime):
            out, ns = _run(device, case, variant)
            pcc = _gate(case, out, f"{regime}/{variant}/{kind}")
            rows.append((kind, variant, ns, pcc))
        case.dealloc()
    for kind, variant, ns, pcc in rows:
        if kind != "pcc":
            continue
        logger.info(
            f"SWEEP {regime:10s} {variant:10s} {ns:9.0f} ns  pcc={pcc:.6f}  L1={bench.l1_bytes(regime, variant)} B"
        )
