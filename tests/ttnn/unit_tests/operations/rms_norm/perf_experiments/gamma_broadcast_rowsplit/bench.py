# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Isolated A/B bench for the `gamma_broadcast_rowsplit` idea (op_design.md Lamp L2).

THE IDEA. In the row-split regime (`cw == 1`) the tile-row axis is what gets
distributed across cores, and `gamma` does not vary along it — so every one of up
to 110 cores fetches the SAME `Wt` gamma pages from DRAM. Read it once per
virtually-contiguous column run and multicast it to that run.

THE VARIANTS (all run at the identical pinned perf config — bf16 / TILE / HiFi2 /
fp32_dest_acc_en=False / bf16 TILE gamma — so no precision knob is ever a lever):

    baseline   the op's current per-core TensorAccessor read of gamma.
    ablate     the gamma NoC reads deleted, reserve/push kept. WRONG output by
               design; it is the UPPER BOUND on any gamma-traffic optimization
               (removes the bytes, adds no synchronization). Measure this FIRST:
               if the bytes are absorbed by an already-saturated DRAM, the
               broadcast cannot win and the investigation is a measured NULL.
    mcast      the candidate broadcast.
    baseline_res / ablate_res
               the same two with gamma residency FORCED, for the geometries where
               the op deliberately streams gamma (NH_core == 1) — so the
               "make it resident" half of the change is separable from the
               "broadcast it" half.

CORRECTNESS (pass/fail; perf is measured, never asserted):
    mode="random"  PCC >= 0.9995 vs torch, random gamma.
    mode="ones"    ABSOLUTE check with gamma == 1 (catches a dropped/duplicated
                   element the way PCC cannot).
    mode="ramp"    gamma[w] = 1 + w/W — NON-UNIFORM. This is the one that matters
                   here: a broadcast that delivers the wrong Wt slice, or core 0's
                   slice to everyone, passes an all-ones test perfectly.
    mode="bitexact" run baseline AND the variant in one process and require the
                   two outputs to be BIT-IDENTICAL. A broadcast moves the same
                   bytes to the same consumers, so anything else is a bug.

MEASURED — blackhole_p150b, 11x10 = 110-core grid, AICLK 1349.99 MHz, DEVICE
KERNEL DURATION [ns], one fresh dispatch per variant (the focus shapes repeated
3-5x; medians below, per-run spread 2.6% baseline / 0.9% mcast).

  INTERLEAVED PREFILL (the focus; row-only split, cw == 1, gamma resident):

    case                   baseline      mcast   speedup   ablate (bound)  capture
    (1,1,8192,7168)         664_433    555_770    1.196x   539_248 1.232x     89%
    (1,1,8192,5120)         478_206    399_119    1.198x         —              —
    (1,1,8192,2304)         213_098    181_280    1.176x         —              —
    (1,1,8192,1024)          97_979     84_244    1.163x    82_617 1.186x     87%
    (1,1,8192,1000)*         99_322     83_705    1.187x         —              —
    * W % 32 != 0 (partial last gamma tile) — same win, bit-exact.

  The `ablate` column is the UPPER BOUND (gamma reads deleted outright). The
  broadcast captures ~88% of it, so the gamma bytes were NOT absorbed by the
  saturated DRAM — the byte share (17.7%) converts to time almost exactly.

  CORE-COUNT PREDICATE (W = 7168, interleaved, row split):

    cores   baseline      mcast   speedup
       11     41_782     49_658    0.841x   REGRESSION
       22     74_654     74_344    1.004x   NULL
       33    104_810     99_423    1.054x
       44    137_969    121_122    1.139x
       55    171_124    146_270    1.170x
      110    354_010    253_671    1.386x   (3520 rows, NH_core == 1)
      110    664_433    555_770    1.196x   (8192 rows, NH_core == 3)

  SHARDED:
    (1,1,256,512) HEIGHT (8,1)   4_425 -> 6_022   0.735x REGRESSION
        gamma-read ablation there is 4_307 -> 4_319, i.e. ZERO headroom: both
        tensors are zero-copy L1 shards, so gamma is the only DRAM traffic and it
        is uncontended. 1_053 ns of the loss is forcing gamma residency (which
        the op deliberately avoids at NH_core == 1), 587 ns the broadcast itself.
    (1,1,32,7168) WIDTH (7,4)    cw = 28 -> the predicate REFUSES (disjoint gamma
        slices, nothing to share). baseline 6_554, forced-residency 7_230,
        "mcast" 7_331 — the whole delta is the residency force, not a broadcast.

  SCHEDULE / INJECTOR-COUNT options, same focus shape (1,1,8192,7168):
    mcast (prologue, 1 injector per virtual-x run)          555_770   BEST
    mcast_1inj (prologue, ONE injector for the whole grid)  557_443   = (noise)
    mcast_late (delivery moved after the first pass-A read)  604_072   WORSE
    baseline_late (the op's own read moved late, control)     669_459   WORSE
  The prologue position is not an accident: it is the ONE window where the
  injector's own DRAM read is uncontended. Moved late, that read competes with
  109 other cores for the same DRAM and takes ~1/110th of the bandwidth.
  On (1,1,8192,1024) and the 8-core HEIGHT case one injector per run also beats a
  single grid-wide injector (85_284 vs 87_558; 6_022 vs 6_819), because the two
  broadcasts then overlap on two cores instead of serializing on one.

  COST OF THE GLOBAL SYNC (permanent zones, prefill7168, 110 cores, ns per core):
    baseline  rdr_gamma_resident  avg 243_525  max 542_121
    mcast     rdr_gamma_resident  avg  14_877  max  15_236
  i.e. the broadcast + handshake is 14.9 us, uniform across all 110 cores = 2.7%
  of the 555 us kernel, replacing a per-core prologue read that cost 243 us of
  wall clock on average. NCRISC-KERNEL avg 342_175 -> 288_275.

  vs the DM ceiling: at 555_770 ns the mcast variant moves its 235.8 MB at
  424 GB/s and is 1.056x FASTER than `ttnn.clone` on the same tensor
  (586_752 ns) — the reference that Refinement 5 set as the column's floor.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import torch

import ttnn

_HERE = Path(__file__).resolve().parent


def _pd():
    """Load the forked program descriptor by PATH (this dir is deliberately not a
    package, so `import ttnn` never exec_module()s the fork)."""
    spec = importlib.util.spec_from_file_location("_gbr_pd", _HERE / "gbr_program_descriptor.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_PD = None


def pd():
    global _PD
    if _PD is None:
        _PD = _pd()
    return _PD


# ---------------------------------------------------------------------------
# cases
# ---------------------------------------------------------------------------
#
# kind: "interleaved" | ("HEIGHT"|"WIDTH"|"BLOCK", shard_shape, core_grid)
# Every case is bf16 / TILE unless `gamma_dtype` says otherwise.

CASES = {
    # --- FOCUS: the perf-flagged interleaved prefill entries. Row-only split,
    # 110 cores, every core reads all Wt gamma tiles.
    "prefill7168": dict(shape=(1, 1, 8192, 7168), kind="interleaved"),
    "prefill1024": dict(shape=(1, 1, 8192, 1024), kind="interleaved"),
    "prefill2304": dict(shape=(1, 1, 8192, 2304), kind="interleaved"),
    "prefill5120": dict(shape=(1, 1, 8192, 5120), kind="interleaved"),
    # --- predicate sweep: the same W, fewer cores => less reuse to remove.
    # Row counts chosen so the active core set is whole grid ROWS (a dense
    # rectangle per virtual-x run); 11 tile-rows = 1 grid row = 11 cores.
    "rows11x7168": dict(shape=(1, 1, 352, 7168), kind="interleaved"),
    "rows22x7168": dict(shape=(1, 1, 704, 7168), kind="interleaved"),
    "rows33x7168": dict(shape=(1, 1, 1056, 7168), kind="interleaved"),
    "rows44x7168": dict(shape=(1, 1, 1408, 7168), kind="interleaved"),
    "rows55x7168": dict(shape=(1, 1, 1760, 7168), kind="interleaved"),
    "rows110x7168": dict(shape=(1, 1, 3520, 7168), kind="interleaved"),
    # --- generality gates on the focus regime.
    "prefill1000_partialw": dict(shape=(1, 1, 8192, 1000), kind="interleaved"),
    "prefill7168_fp32gamma": dict(shape=(1, 1, 8192, 7168), kind="interleaved", gamma_dtype=ttnn.float32),
    "prefill1024_rmgamma": dict(shape=(1, 1, 8192, 1024), kind="interleaved", gamma_layout="rm"),
    # --- HEIGHT_SHARDED: cw == 1, so the reuse is real at 8 cores.
    "height256x512": dict(shape=(1, 1, 256, 512), kind=("HEIGHT", (32, 512), (8, 1))),
    # --- NULL control: the cross-core W-split hands every core a DISJOINT gamma
    # slice, so gamma is already read exactly once in total.
    "width32x7168": dict(shape=(1, 1, 32, 7168), kind=("WIDTH", (32, 256), (7, 4))),
}

# variant -> (GAMMA_MODE, FORCE_GAMMA_RESIDENT, GAMMA_LATE[, GAMMA_ONE_INJECTOR])
VARIANTS = {
    "baseline": ("dram", None, False),
    "ablate": ("ablate", None, False),
    "mcast": ("mcast", True, False),
    "baseline_res": ("dram", True, False),
    "ablate_res": ("ablate", True, False),
    # schedule A/B: the same delivery moved off the prologue. `baseline_late` is
    # the CONTROL — it isolates how much of `mcast_late`'s delta is the schedule
    # rather than the broadcast.
    "mcast_late": ("mcast", True, True),
    "baseline_late": ("dram", None, True),
    "baseline_res_late": ("dram", True, True),
    # injector-count A/B: ONE injector for the whole grid (it broadcasts to both
    # virtual-x runs in turn) vs one per run (the default).
    "mcast_1inj": ("mcast", True, False, True),
}


def perf_compute_kernel_config():
    """feature_spec._PERF_BASE's FIXED precision contract: HiFi2 + bf16 DEST.

    Identical for every variant — the precision knobs are user-provided inputs
    here, never a speed lever.
    """
    cfg = ttnn.ComputeConfigDescriptor()
    cfg.math_fidelity = ttnn.MathFidelity.HiFi2
    cfg.fp32_dest_acc_en = False
    cfg.math_approx_mode = False
    return cfg


def _gamma_tensor(device, W, mode, gamma_dtype, gamma_layout):
    if mode == "ones":
        g = torch.ones(W, dtype=torch.float32)
    elif mode == "ramp":
        # NON-UNIFORM along W: a per-tile-index signature, so delivering the
        # wrong Wt slice (or core 0's slice to everyone) shows up immediately.
        g = 1.0 + torch.arange(W, dtype=torch.float32) / float(W)
    else:
        torch.manual_seed(7)
        g = torch.randn(W, dtype=torch.float32)
    tdt = torch.float32 if gamma_dtype == ttnn.float32 else torch.bfloat16
    gt = g.to(tdt)
    layout = ttnn.ROW_MAJOR_LAYOUT if gamma_layout == "rm" else ttnn.TILE_LAYOUT
    tt_g = ttnn.from_torch(gt.reshape(1, 1, 1, W), dtype=gamma_dtype, layout=layout, device=device)
    return gt.to(torch.float32), tt_g


def _make_inputs(device, case, mode):
    spec = CASES[case]
    shape = spec["shape"]
    gamma_dtype = spec.get("gamma_dtype", ttnn.bfloat16)
    gamma_layout = spec.get("gamma_layout", "tile")

    mc = None
    if spec["kind"] != "interleaved":
        from eval.sharding import shard_config

        kind, shard_shape, core_grid = spec["kind"]
        mc = shard_config(
            list(shard_shape),
            core_grid,
            getattr(ttnn.TensorMemoryLayout, f"{kind}_SHARDED"),
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
            device=device,
        )

    torch.manual_seed(42)
    x = torch.randn(shape, dtype=torch.bfloat16)
    kw = dict(dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    if mc is not None:
        kw["memory_config"] = mc
    tt_x = ttnn.from_torch(x, **kw)
    g, tt_g = _gamma_tensor(device, shape[-1], mode, gamma_dtype, gamma_layout)
    return x, g, tt_x, tt_g, mc


def _dispatch(device, tt_x, tt_g, mc, variant):
    """One fresh dispatch of the forked op with `variant`'s gamma delivery."""
    m = pd()
    spec = VARIANTS[variant]
    gamma_mode, force_res, late = spec[0], spec[1], spec[2]
    one_inj = spec[3] if len(spec) > 3 else False
    saved = (m.GAMMA_MODE, m.FORCE_GAMMA_RESIDENT, m.GAMMA_LATE, m.GAMMA_ONE_INJECTOR)
    m.GAMMA_MODE, m.FORCE_GAMMA_RESIDENT, m.GAMMA_LATE = gamma_mode, force_res, late
    m.GAMMA_ONE_INJECTOR = one_inj
    try:
        out_mc = mc if mc is not None else ttnn.DRAM_MEMORY_CONFIG
        tt_out = ttnn.allocate_tensor_on_device(ttnn.Shape(list(tt_x.shape)), tt_x.dtype, tt_x.layout, device, out_mc)
        desc = m.create_program_descriptor(
            tt_x,
            tt_out,
            gamma=tt_g,
            epsilon=1e-6,
            compute_kernel_config=perf_compute_kernel_config(),
            device=device,
        )
        io = [tt_x, tt_g, tt_out] if tt_g is not None else [tt_x, tt_out]
        print(f"    plan[{variant}] {dict(m.LAST_PLAN)}")
        return ttnn.generic_op(io, desc)
    finally:
        m.GAMMA_MODE, m.FORCE_GAMMA_RESIDENT, m.GAMMA_LATE, m.GAMMA_ONE_INJECTOR = saved


def describe(device, case):
    """The derived blocking / placement / broadcast plan for one case (no dispatch)."""
    m = pd()
    x, g, tt_x, tt_g, mc = _make_inputs(device, case, "ones")
    grid = device.compute_with_storage_grid_size()
    ht_total, wt_global = m._tile_geometry(tt_x)
    in_sharded = tt_x.layout == ttnn.TILE_LAYOUT and (
        tt_x.memory_config().memory_layout != ttnn.TensorMemoryLayout.INTERLEAVED
    )
    placement = m._select_placement(device, grid, tt_x, ht_total, wt_global, in_sharded)
    blk = m._derive_blocking(
        tt_x,
        tt_g,
        grid.x * grid.y,
        placement,
        sharded_in=in_sharded,
        sharded_out=in_sharded,
        l1_total_budget=m._l1_total_budget(device),
    )
    runs = m._virtual_x_runs(device, grid)
    plan = m._gamma_mcast_plan(placement, runs)
    active = sum(1 for w in placement.works if w.num_rows > 0)
    info = dict(
        case=case,
        shape=tuple(tt_x.shape),
        cores=len(placement.works),
        active_cores=active,
        cw=placement.cw,
        Wt=blk.Wt,
        wt_chunk=blk.wt_chunk,
        nw=blk.nw,
        ht_block=blk.ht_block,
        nh_core=blk.nh_core_max,
        gamma_resident=blk.gamma_resident,
        x_res_depth=blk.x_res_depth,
        runs=runs,
        mcast_families=None if plan is None else [f[0] for f in plan[0]],
        injectors=None if plan is None else [f[1] for f in plan[0]],
        acks=None if plan is None else [f[2] for f in plan[0]],
        gamma_bytes_per_core=blk.Wt * ttnn.tile_size(tt_g.dtype) * (1 if blk.gamma_resident else blk.nh_core_max),
    )
    ttnn.deallocate(tt_x)
    ttnn.deallocate(tt_g)
    return info


def _reference(x, g):
    xf = x.to(torch.float32)
    out = xf / torch.sqrt(torch.mean(xf**2, dim=-1, keepdim=True) + 1e-6)
    return out * g.reshape(-1)


def _pcc(a, b):
    a = a.flatten().to(torch.float32)
    b = b.flatten().to(torch.float32)
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


def run_case(device, case, variant, mode="random"):
    """ONE fresh dispatch. This is the TIMED unit (one --profile CSV row).

    Correctness is gated for every variant except `ablate*` (wrong by design).
    """
    x, g, tt_x, tt_g, mc = _make_inputs(device, case, mode)
    tt_out = _dispatch(device, tt_x, tt_g, mc, variant)

    if variant.startswith("ablate"):
        print(f"[{case}/{variant}] ABLATION — timing only, output not checked")
        return None

    actual = ttnn.to_torch(tt_out).to(torch.float32)
    expected = _reference(x, g)
    pcc = _pcc(expected, actual)
    print(f"[{case}/{variant}/{mode}] pcc={pcc:.6f}")
    assert pcc >= 0.9995, f"{case}/{variant}/{mode}: pcc {pcc} < 0.9995"
    if mode in ("ones", "ramp"):
        # ABSOLUTE gate: PCC is scale-invariant and has hidden four bugs in this
        # op that only rescaled rows/columns. Normalized by the tensor's own scale
        # (an ELEMENTWISE relative error is meaningless where randn is ~0), so
        # this catches a wrong gamma slice, which shifts whole COLUMNS by O(1).
        err = (actual - expected).abs().max().item() / expected.abs().max().item()
        # Per-column mean ratio: the signature a mis-delivered gamma slice breaks
        # (all-ones and per-row rescaling both leave it at 1.0).
        col = (actual.reshape(-1, actual.shape[-1]).abs().mean(0) + 1e-9) / (
            expected.reshape(-1, expected.shape[-1]).abs().mean(0) + 1e-9
        )
        print(
            f"[{case}/{variant}/{mode}] max err/scale={err:.5f} "
            f"col-ratio min={col.min().item():.5f} max={col.max().item():.5f}"
        )
        # 3% of the tensor's own scale. Measured: the op's BASELINE datapath is
        # already at 1.1% (ones) / 1.5% (ramp) here — bf16 activations reduced at
        # HiFi2 into a non-fp32 DEST (the FIXED user config), i.e. a per-ROW rms
        # error, not a per-column one. So this is a loose sanity band; the TIGHT
        # gate for this idea is `run_bitexact` (the broadcast must reproduce the
        # baseline's bytes exactly), plus the per-column ratio below, which is the
        # thing a mis-delivered gamma slice actually breaks.
        assert err < 0.03, f"{case}/{variant}/{mode}: max error / scale = {err}"
        assert col.min() > 0.97 and col.max() < 1.03, (
            f"{case}/{variant}/{mode}: per-column mean ratio out of band "
            f"[{col.min().item()}, {col.max().item()}] — a gamma slice is mis-delivered"
        )
    return pcc


def run_bitexact(device, case, variant, mode="ramp"):
    """Baseline vs variant in ONE process — the outputs must be bit-identical.

    The broadcast delivers the SAME bytes to the same consumers, so this, not PCC,
    is the real correctness statement for this idea.
    """
    x, g, tt_x, tt_g, mc = _make_inputs(device, case, mode)
    a = ttnn.to_torch(_dispatch(device, tt_x, tt_g, mc, "baseline"))
    b = ttnn.to_torch(_dispatch(device, tt_x, tt_g, mc, variant))
    same = torch.equal(a, b)
    nbad = int((a != b).sum())
    print(f"[{case}/{variant}/{mode}] bit-exact vs baseline: {same} ({nbad} differing elements)")
    assert same, f"{case}/{variant}: {nbad} elements differ from the baseline output"
