# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""`gather_payload_shrink` — isolated A/B bench for ONE idea.

IDEA. Under the cross-core W-split each worker ships `ht` full 4 KB Float32 tiles
to its group root per row-block, in which all 32 columns are still live x^2
partial sums. The only information in each tile is its 32 row-sums — 128 B of
4096. On the focus shape that is 256 KB per row-block converging on one core's L1
(~17 GB/s inbound, the single-core wall) to carry 8 KB. **Shrink the payload.**

VARIANTS (`GPS_VARIANT`), all under the IDENTICAL precision contract
(bf16 / HiFi2 / fp32_dest_acc_en=False / math_approx_mode=False / bf16 TILE gamma)
and the identical placement, core count and per-core slice:

  baseline      the op as it stands: `ht` RAW fp32 tiles per core per row-block.
  colpack       ONE fp32 tile per core per row-block — the `ht` row-sums folded
                within-tile and landed in `ht` DISTINCT COLUMNS of one tile
                (`ht`x fewer bytes). Mechanism measured in probe_mechanism.py.
  bf16          `ht` RAW bf16 tiles (2x fewer bytes). CHANGES NUMERICS.
  colpack_bf16  ONE bf16 tile (2*`ht`x fewer bytes). CHANGES NUMERICS.

`GPS_SHRINK=k` is an ABLATION (output WRONG by design): the writer puts only
tile_bytes/k of each gather tile on the wire. It measures the TRANSPORT ceiling
of any byte reduction before the math that makes it correct exists.

`GPS_HT_BLOCK=n` overrides HT_BLOCK, which IS the column-pack factor — so
sweeping it sweeps the idea's own axis directly.

CORRECTNESS GATE (pass/fail; perf is measured, never asserted):
  * PCC >= 0.9995 vs torch fp32, and
  * an ABSOLUTE all-ones check: mean(x^2) must come back EXACTLY 1.0, so
    out == gamma / sqrt(1 + eps). A dropped W-tile, a double-counted lane, a
    wrong n_reduced or a wrong row->core map only RESCALES rows, and PCC scores
    that >= 0.9998 — this op has shipped four such bugs. This gate EARNED its
    keep here: the first cut of `colpack` scored PCC 0.9998 (pass) while silently
    corrupting 12.5% of tile-rows, and only the all-ones check (max err 0.0625)
    caught it. Root cause was a CB whose per-row-block push count (9) did not
    divide its page count (16), so a multi-page cb_reserve_back straddled
    fifo_limit and the non-wrapping pack_tile wrote past it.

-------------------------------------------------------------------------------
MEASURED (blackhole_p150b, 11x10 grid, AICLK 1349.99 MHz, one fresh-cache
dispatch per (variant, case), raw rows in measurements/results.tsv)
-------------------------------------------------------------------------------
Reproducibility: baseline focus measured 3x -> 75490 / 75472 / 75430 ns (0.08%).

FOCUS shape (1,1,8192,1024) BLOCK_SHARDED (1024,128) grid (8,8), HT_BLOCK 8:

    variant          ns       vs baseline   ship B/core/row-block
    FLOOR no-combine 56 070      1.346x     0        (ABLATION, wrong output)
    colpack_bf16     57 181      1.320x     2 048
    colpack          57 551      1.312x     4 096
    ABL byte-only    66 384      1.137x     128      (ABLATION, wrong output)
    bf16             67 720      1.115x     16 384
    baseline         75 490      1.000x     32 768

The combine's WHOLE critical-path cost is 75 490 - 56 070 = 19 420 ns, so
`colpack` recovers 92.4% of it and `colpack_bf16` 94.3%. Both BEAT the
byte-only ablation ceiling (66 384) because the column-pack does not just
shrink the wire — it moves the fold OFF the root: the root's fold drops from
ht*CW1 = 64 tile-reduces to CW1 + ht = 16 ops (per-stage zones,
measurements/zones_focus_*.csv):

    stage (focus)                     baseline    colpack
    cmp_combine  TRISC_1 max (root)     25 479      6 248
    wtr_gather_hop BRISC avg            59 076     40 096
    rdr_gather_wait NCRISC max          43 512     36 185
    rdr_mcast NCRISC avg                56 486     39 722
    cmp_colpack (NEW, all 64 cores)          -    496..1 731
    wtr_selectors (NEW, once, BRISC)         -      4 398

PREDICATE SWEEP — the column-pack factor IS HT_BLOCK, so HT_BLOCK is swept
directly (GPS_HT_BLOCK) rather than hunting shapes:

    case            HT_BLOCK  baseline  colpack  colpack_bf16   bf16
    focus                  8    75 490   57 551        57 181  67 720
    focus_hb4              4    79 856   65 475        64 800  72 307
    focus_hb2              2    90 216   82 631        81 347  83 707
    focus_hb1              1   108 725  113 156       111 096 106 592
    w32x1024               1     5 189    5 422         5 012   4 865
    w32x7168               1     7 420    7 551         6 836   6 395
    block8192x2304         1   126 696  131 100       128 877 124 233

    -> colpack: 1.312x / 1.220x / 1.092x / 0.961x for HT_BLOCK 8 / 4 / 2 / 1.
       Monotone in HT_BLOCK; a small REGRESSION at HT_BLOCK == 1, where there is
       nothing to pack and only the fold + selector bank are paid.
    -> bf16: a win on EVERY case (1.020x .. 1.160x), independent of HT_BLOCK.

PRECISION — no option trades precision beyond the contract. Focus shape:

    variant        PCC          implied 1/rms scale: mean / max|rel|
    baseline       0.99998402   0.996620 / 0.02403
    bf16           0.99998402   0.996620 / 0.02403   <- BIT-IDENTICAL
    colpack        0.99998256   0.998767 / 0.02385
    colpack_bf16   0.99998256   0.998767 / 0.02385

    A bf16 gather payload is numerically FREE: fp32_dest_acc_en=False already
    makes DEST 16-bit, so the 4 KB Float32 tile's extra mantissa was discarded on
    unpack. Re-checked at fp32_dest_acc_en=True (predicate characterisation only,
    never a timing claim): still equal to 8 digits of PCC (0.99998475 both),
    implied-scale mean 0.992760 vs 0.992912. colpack's row scale is slightly
    CLOSER to 1.0 than the baseline's on every case measured.

NOT WORTH PURSUING (measured): shrinking the MULTICAST leg too (root packs ->
mcast 1 tile -> every core column-selects). colpack already sits 1 481 ns above
the fully-ablated no-combine floor, so the entire remaining prize is < 2.6%.
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import torch

import ttnn

_HERE = Path(__file__).resolve().parent


def _load_pd():
    spec = importlib.util.spec_from_file_location("_gps_pd", _HERE / "gps_program_descriptor.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


pd = _load_pd()


# --------------------------------------------------------------------------
# geometry cases
# --------------------------------------------------------------------------
# (shape, kind, shard_shape, core_grid, ht_block_override)
CASES = {
    # THE FOCUS SHAPE — feature_spec's perf-flagged BLOCK_SHARDED prefill cell.
    # 64 cores, per core 32 tile-rows x Wt=4, nw=1, ht_block=8, cw=cw1=8 (flat).
    "focus": ((1, 1, 8192, 1024), "BLOCK", (1024, 128), (8, 8), None),
    # Same geometry, column-pack factor swept directly.
    "focus_hb4": ((1, 1, 8192, 1024), "BLOCK", (1024, 128), (8, 8), 4),
    "focus_hb2": ((1, 1, 8192, 1024), "BLOCK", (1024, 128), (8, 8), 2),
    "focus_hb1": ((1, 1, 8192, 1024), "BLOCK", (1024, 128), (8, 8), 1),
    # ht == 1 decode geometries: the payload is ONE tile per core already, so the
    # column-pack has nothing to pack and can only cost.
    "w32x1024": ((1, 1, 32, 1024), "WIDTH", (32, 128), (8, 1), None),
    "w32x7168": ((1, 1, 32, 7168), "WIDTH", (32, 256), (7, 4), None),
    # A second, wider BLOCK_SHARDED prefill cell (Wt/core = 9).
    "block8192x2304": ((1, 1, 8192, 2304), "BLOCK", (1024, 288), (8, 8), None),
}


def perf_compute_kernel_config():
    """The PINNED precision contract. Identical for every variant — never a lever.

    `GPS_FP32DEST=1` is NOT a perf knob and is never used for a timing claim: it
    exists only to CHARACTERIZE the predicate of the `bf16` option. A bf16 gather
    payload is measured numerically FREE at the pinned contract because
    fp32_dest_acc_en=False already makes DEST 16-bit, so the fp32 tile's extra
    mantissa was discarded on unpack anyway. That equivalence must not be assumed
    to survive fp32_dest_acc_en=True — this switch is how that is checked.
    """
    cfg = ttnn.ComputeConfigDescriptor()
    cfg.math_fidelity = ttnn.MathFidelity.HiFi2
    cfg.fp32_dest_acc_en = os.environ.get("GPS_FP32DEST") == "1"
    cfg.math_approx_mode = False
    return cfg


def _dispatch(device, shape, mc, torch_x, torch_gamma):
    tt_x = ttnn.from_torch(torch_x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mc)
    tt_gamma = ttnn.from_torch(
        torch_gamma.reshape(1, 1, 1, shape[-1]), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
    )
    out = ttnn.allocate_tensor_on_device(ttnn.Shape(list(shape)), ttnn.bfloat16, ttnn.TILE_LAYOUT, device, mc)
    prog = pd.create_program_descriptor(
        tt_x,
        out,
        gamma=tt_gamma,
        epsilon=1e-6,
        compute_kernel_config=perf_compute_kernel_config(),
        device=device,
    )
    return ttnn.generic_op([tt_x, tt_gamma, out], prog)


def run_case(device, case, variant, mode="random"):
    """One fresh dispatch of one variant on one geometry. Returns nothing; the
    ns comes from --profile's ops_perf_results CSV."""
    from eval.sharding import shard_config

    shape, kind, shard_shape, core_grid, hb = CASES[case]
    os.environ["GPS_VARIANT"] = variant
    if hb is not None:
        os.environ["GPS_HT_BLOCK"] = str(hb)
    else:
        os.environ.pop("GPS_HT_BLOCK", None)

    # PIN THE TOPOLOGY FLAT for every variant and every case. The staged (two-
    # stage) gather is a DIFFERENT idea (combine topology); pinning it flat keeps
    # this bench's delta attributable to the payload alone.
    pd.COMBINE_MAX_FLAT_FANIN = 10**6

    # GPS_ABLATE=combine — the FLOOR. Keeps the placement, the core count and the
    # per-core slice byte-for-byte and removes only the gather + root fold +
    # multicast, so `baseline - floor` is the combine's whole critical-path cost
    # and `baseline - candidate` can be quoted as a fraction of it. Output is
    # WRONG by design (each core normalizes by its own slice).
    if os.environ.get("GPS_ABLATE") == "combine":
        _real_select = pd._select_placement

        def _no_combine(*args, **kwargs):
            pl = _real_select(*args, **kwargs)
            pl.cw = pl.cw1 = pl.cw2 = 1
            pl.groups = []
            return pl

        pd._select_placement = _no_combine

    memory_layout = getattr(ttnn.TensorMemoryLayout, f"{kind}_SHARDED")
    mc = shard_config(
        list(shard_shape), core_grid, memory_layout, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, device=device
    )

    torch.manual_seed(42)
    if mode == "ones":
        torch_x = torch.ones(shape, dtype=torch.bfloat16)
        torch_gamma = torch.ones(shape[-1], dtype=torch.bfloat16)
    else:
        torch_x = torch.randn(shape, dtype=torch.bfloat16)
        torch_gamma = torch.randn(shape[-1], dtype=torch.bfloat16)

    if os.environ.get("GPS_REPORT"):
        blk = _report_blocking(device, shape, mc, torch_x, torch_gamma)
        print(f"[gps] {case:16s} {blk}")
        return

    tt_out = _dispatch(device, shape, mc, torch_x, torch_gamma)
    actual = ttnn.to_torch(tt_out).to(torch.float32)

    if os.environ.get("GPS_SHRINK", "1") != "1" or os.environ.get("GPS_ABLATE"):
        print(f"\n[gps] {case} {variant} ABLATION (GPS_SHRINK / GPS_ABLATE) — output WRONG by design, no gate\n")
        return

    xf = torch_x.to(torch.float32)
    if mode == "ones":
        # ABSOLUTE gate. mean(x^2) == 1.0 exactly for an all-ones input, so every
        # output element is 1/sqrt(1+eps). Any dropped/duplicated/mismapped lane
        # shows up as a scale error here even though PCC would be ~1.0.
        want = 1.0 / (1.0 + 1e-6) ** 0.5
        err = (actual - want).abs().max().item()
        rec = (actual.to(torch.float32) ** 2).mean().item() * (1.0 + 1e-6)
        print(
            f"\n[gps] {case} {variant} ALL-ONES: max|out - 1/sqrt(1+eps)| = {err:.6f}  (recovered mean(x^2) = {rec:.6f})"
        )
        if err >= 5e-3:
            # Diagnose WHICH rows/columns are wrong: a payload bug shows up as a
            # per-tile-row scale error, so recover mean(x^2) per row and print the
            # pattern modulo HT_BLOCK.
            a = actual.reshape(-1, actual.shape[-1])
            m = (1.0 / a[:, 0] ** 2) - 1e-6
            uniq = sorted(set(round(float(v), 5) for v in m))
            print(f"[gps]   per-row recovered mean(x^2): {len(uniq)} distinct -> {uniq[:8]}")
            print(f"[gps]   rows 0..39 (x1024): {[round(float(v) * 1024, 1) for v in m[:40]]}")
            bad = (m - 1.0).abs() > 1e-3
            idx = bad.nonzero().flatten()[:24].tolist()
            print(f"[gps]   first bad rows: {idx}   (bad fraction {bad.float().mean().item():.4f})")
            col_bad = (actual.reshape(-1, actual.shape[-1]) - want).abs().amax(dim=0)
            print(f"[gps]   max err by output column (first 40): {[round(float(v), 4) for v in col_bad[:40]]}")
        # bf16 output quantum is 2^-8 relative => 0.004 absolute at ~1.0.
        assert err < 5e-3, f"{case}/{variant}: all-ones absolute check FAILED, max err {err}"
    else:
        expected = xf / torch.sqrt(torch.mean(xf**2, dim=-1, keepdim=True) + 1e-6)
        expected = expected * torch_gamma.to(torch.float32).reshape(-1)
        pcc = _pcc(expected, actual)
        # PCC is scale-invariant per-tensor, so also report the per-ROW statistic
        # the payload can actually damage: the implied 1/rms scale.
        e = expected.reshape(-1, expected.shape[-1])
        a = actual.reshape(-1, actual.shape[-1])
        big = e.abs() > 0.2
        scale = a[big] / e[big]
        rel = (scale - 1.0).abs()
        print(
            f"\n[gps] {case} {variant} PCC = {pcc:.8f}  "
            f"implied-scale mean={scale.mean().item():.6f} max|rel|={rel.max().item():.5f} "
            f"p99.9|rel|={rel.quantile(0.999).item():.5f}"
        )
        assert pcc >= 0.9995, f"{case}/{variant}: PCC {pcc} < 0.9995"


def _pcc(a, b):
    a = a.flatten().double()
    b = b.flatten().double()
    a = a - a.mean()
    b = b - b.mean()
    return float((a * b).sum() / (a.norm() * b.norm()))


def _report_blocking(device, shape, mc, torch_x, torch_gamma):
    """Derived blocking knobs for one case (no perf claim; predicate bookkeeping)."""
    tt_x = ttnn.from_torch(torch_x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mc)
    tt_g = ttnn.from_torch(
        torch_gamma.reshape(1, 1, 1, shape[-1]), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
    )
    grid = device.compute_with_storage_grid_size()
    ht_total, wt_global = pd._tile_geometry(tt_x)
    placement = pd._select_placement(device, grid, tt_x, ht_total, wt_global, True)
    blk = pd._derive_blocking(
        tt_x,
        tt_g,
        grid.x * grid.y,
        placement,
        sharded_in=True,
        sharded_out=True,
        l1_total_budget=pd._l1_total_budget(device),
    )
    return (
        f"cores={placement.num_cores:3d} cw={placement.cw} cw1={placement.cw1} cw2={placement.cw2} "
        f"Wt/core={blk.Wt} nw={blk.nw} HT_BLOCK={blk.ht_block} rows/core={blk._rows_core_max} "
        f"nh_core={-(-blk._rows_core_max // blk.ht_block)} fuse_sq={int(blk.fuse_sq)} "
        f"colpack={int(blk.colpack)} gather_ht={blk.gather_ht} gather_tile_B={blk.gather_tile_bytes} "
        f"ship_B/rowblock={blk.gather_ht * blk.gather_tile_bytes}"
    )
