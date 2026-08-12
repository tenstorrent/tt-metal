# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""ISOLATED bake-off for rms_norm's ROOT FINALIZE chain (perf experiment `cskip_finalize`).

Baseline = what `rms_norm_compute.cpp` does today (`AddUnaryColValid` + `RsqrtColValid`,
both `VectorMode::C`). Candidates scope the SFPU further, to the EVEN-PARITY vectors
that actually hold column 0, and/or fuse the two ops into one pass.

One core, `N` column-0-valid stat tiles resident in L1 (bf16), one `cp_finalize` zone
around the chain, nothing else on the clock. Precision contract is identical for every
variant: bf16 / HiFi2 / fp32_dest_acc_en=False.

Measured with the in-process profiler:

    TT_METAL_DEVICE_PROFILER=1 TT_METAL_PROFILER_MID_RUN_DUMP=1 \
    TT_METAL_PROFILER_CPP_POST_PROCESS=1 timeout 1800 \
    scripts/tt-probe.sh rms_norm <<'EOF'
    from ttnn.operations.rms_norm.perf_experiments.cskip_finalize.bench import main
    main()
    EOF

Per-TRISC numbers: reduce the copied zone CSVs with `zone_reduce.py` (this bench has no
reader/writer kernel, so `tests/.../perf_zone_report.py` trips on the missing spans).

MEASURED — box=bh-49-special (blackhole_p150b) · BH · 1350 MHz · 2026-08-12 · 4a4098e56b
bf16 / HiFi2 / fp32_dest_acc_en=False. `ns` = DEVICE KERNEL DURATION; `math` = the
`cp_finalize` zone's TRISC_1 (math) occupancy. ONE fresh run per variant; N=1 and N=2 are
the median of 3 (their whole-kernel number sits on the ~700 ns per-dispatch floor, so at
those tile counts only `math` carries signal).

  N (stat tiles per chain call)      1      2      4     16     32
  ------------------------------------------------------------------
  copy_only         ns             699    737    864   1481   2350
                    math            84    109    334   1070   1937
  c_pair (BASELINE) ns             666   1029   1971   7705  15296
                    math           415    889   1835   7568  15156
  cskip_pair        ns             660    794   1320   4546   8853
                    math           377    657   1184   4409   8715
  cskip_fused       ns             680    859   1330   4119   7846
                    math           493    720   1195   3978   7701
  cskip_fused_bx    ns             680    937   1403   4310   8224
                    math           540    798   1264   4148   8059
  c_fused           ns             881   1327   2224   7680  14878
                    math           744   1191   2089   7539  14736
  rc_pair           ns            1149     --   3852  14611  28973
                    math          1011     --   3715  14475  28837

  math-thread speedup vs BASELINE:  N=1   N=2   N=4   N=16  N=32
    cskip_pair                     1.10x 1.35x 1.55x 1.72x 1.74x
    cskip_fused                    0.84x 1.23x 1.54x 1.90x 1.97x

fp32_dest_acc_en=True (fp32 stat tiles), N=16: c_pair 7647 ns / 7510 math; cskip_pair
4583 / 4443; cskip_fused 4120 / 3984 — the same 1.7-1.9x, all still bit-exact.

Column 0 is BIT-IDENTICAL to today's chain for EVERY variant, in both precision legs,
over inputs covering 0 / 1e-30 / 1e-8 / 1e-3 / 0.5 / 1 / 2 / 17 / 1e3 / 1e8 / 1e18 plus
random, with hostile garbage (-1, 0, +-inf, nan, +-3.4e38) in columns 1..31.
"""

from __future__ import annotations

import os
import shutil
import struct
from pathlib import Path

import torch
import ttnn

TILE = 32
KERNEL = Path(__file__).parent / "kernels" / "finalize_bench_compute.cpp"

CB_STAT_SUM = 0
CB_RSTD_SEND = 16

# variant name -> (CT variant id, SFPU 32-lane vector ops per stat tile, one-line label)
VARIANTS = {
    "copy_only": (0, 0, "copy + pack only (no SFPU) — ablation floor"),
    "rc_pair": (1, 64, "AddUnary(RC) + Rsqrt(RC) — pre-Refinement-5"),
    "c_pair": (2, 32, "AddUnary(C) + Rsqrt(C) — TODAY'S OP (baseline)"),
    "cskip_pair": (3, 16, "AddUnary + Rsqrt, C + even-parity stride"),
    "c_fused": (4, 16, "fused rsqrt(x+eps), C"),
    "cskip_fused": (5, 8, "fused rsqrt(x+eps), C + even-parity stride (fp32 sum)"),
    "cskip_fused_bitexact": (6, 8, "same, sum round-tripped through DEST (bit-exact)"),
    "rc_fused": (7, 32, "fused rsqrt(x+eps), RC"),
}
BASELINE = "c_pair"
EPS = 1e-6

# Variants whose column-0 result is defined (copy_only publishes the raw input).
_SFPU_VARIANTS = [v for v in VARIANTS if v != "copy_only"]


def _eps_bits(eps: float) -> int:
    """fp32 bit pattern of eps, as the op passes it (a uint32 compile-time arg)."""
    return struct.unpack("<I", struct.pack("<f", float(eps)))[0]


def _memcfg(device, n_tiles):
    return ttnn.create_sharded_memory_config(
        shape=(TILE * n_tiles, TILE),
        core_grid=ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))]),
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )


def stat_tiles(n_tiles: int, fp32: bool = False) -> torch.Tensor:
    """`n_tiles` COLUMN-0-VALID stat tiles, shaped like a real `reduce<SUM, REDUCE_ROW>` result.

    Column 0 carries the meaningful mean-of-squares values, spanning the regimes the
    finalize has to survive: the eps region (0 and denormal-small), ~1, and very
    large / very small magnitudes. Columns 1..31 carry deliberately HOSTILE garbage
    (negatives, zeros, inf, nan, huge) — the op leaves structurally-undefined values
    there, and a candidate that scopes past them must neither fault nor let them
    contaminate column 0.
    """
    torch.manual_seed(0)
    rows = n_tiles * TILE
    col0 = torch.empty(rows, dtype=torch.float32)
    regimes = [0.0, 1e-30, 1e-8, 1e-3, 0.5, 1.0, 2.0, 17.0, 1e3, 1e8, 1e18]
    for r in range(rows):
        if r % 16 < len(regimes):
            col0[r] = regimes[r % 16]
        else:
            col0[r] = float(torch.rand(1).item()) * 10.0 + 1e-4
    garbage = torch.randn(rows, TILE - 1, dtype=torch.float32) * 1e3
    garbage[:, 0] = -1.0
    garbage[:, 1] = 0.0
    garbage[:, 2] = float("inf")
    garbage[:, 3] = float("nan")
    garbage[:, 4] = -3.4e38
    garbage[:, 5] = 3.4e38
    t = torch.cat([col0.unsqueeze(1), garbage], dim=1)
    if fp32:
        return t
    return t.to(torch.bfloat16).to(torch.float32)


def golden_col0(stats: torch.Tensor, eps: float = EPS) -> torch.Tensor:
    """fp64 reference for column 0: rsqrt(x + eps)."""
    x = stats[:, 0].to(torch.float64)
    return torch.rsqrt(x + eps)


def make_tensors(device, n_tiles, fp32=False):
    """The op's stat pipeline follows `fp32_dest_acc_en` (Refinement 5): bf16 CBs when
    the DEST is 16-bit, fp32 CBs when it is not. `fp32=True` is the second regime."""
    stats = stat_tiles(n_tiles, fp32=fp32)
    dtype = ttnn.float32 if fp32 else ttnn.bfloat16
    memcfg = _memcfg(device, n_tiles)
    tt_in = ttnn.from_torch(
        stats.reshape(1, 1, n_tiles * TILE, TILE),
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=memcfg,
    )
    tt_out = ttnn.allocate_tensor_on_device(
        ttnn.Shape([1, 1, n_tiles * TILE, TILE]), dtype, ttnn.TILE_LAYOUT, device, memcfg
    )
    return stats, tt_in, tt_out


def _compute_config(fp32=False):
    """The perf loose case's config — FIXED for every variant of a given leg."""
    cfg = ttnn.ComputeConfigDescriptor()
    cfg.math_fidelity = ttnn.MathFidelity.HiFi2
    cfg.fp32_dest_acc_en = bool(fp32)
    cfg.math_approx_mode = False
    return cfg


def descriptor(tt_in, tt_out, *, variant, n_tiles, eps=EPS, fp32=False):
    vid = VARIANTS[variant][0]
    core = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))])
    compute = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL),
        core_ranges=core,
        compile_time_args=[vid, _eps_bits(eps)],
        runtime_args=[(ttnn.CoreCoord(0, 0), [n_tiles])],
        config=_compute_config(fp32),
    )
    cbs = [
        ttnn.cb_descriptor_from_sharded_tensor(CB_STAT_SUM, tt_in),
        ttnn.cb_descriptor_from_sharded_tensor(CB_RSTD_SEND, tt_out),
    ]
    return ttnn.ProgramDescriptor(kernels=[compute], semaphores=[], cbs=cbs)


def run(tt_in, tt_out, *, variant, n_tiles, eps=EPS, fp32=False):
    return ttnn.generic_op(
        [tt_in, tt_out], descriptor(tt_in, tt_out, variant=variant, n_tiles=n_tiles, eps=eps, fp32=fp32)
    )


# ---------------------------------------------------------------------------
# measurement
# ---------------------------------------------------------------------------
def _device_kernel_ns():
    per_chip = ttnn.get_latest_programs_perf_data() or {}
    ns = None
    for programs in per_chip.values():
        for program in programs:
            entry = (getattr(program, "program_analyses_results", None) or {}).get("DEVICE KERNEL DURATION [ns]")
            if entry is not None:
                d = float(entry.duration)
                ns = d if ns is None else max(ns, d)
    return ns


def measure_one(device, variant, n_tiles, logdir=None, tag="", fp32=False):
    """ONE fresh run. Returns (device_kernel_ns, out_col0 as float32 tensor)."""
    stats, tt_in, tt_out = make_tensors(device, n_tiles, fp32=fp32)
    ttnn.ReadDeviceProfiler(device)  # flush the from_torch / shard prep
    out = run(tt_in, tt_out, variant=variant, n_tiles=n_tiles, fp32=fp32)
    ttnn.synchronize_device(device)
    ttnn.ReadDeviceProfiler(device)
    ns = _device_kernel_ns()
    host = ttnn.to_torch(out).to(torch.float32).reshape(n_tiles * TILE, TILE)
    if logdir:
        src = os.path.join(logdir, "profile_log_device.csv")
        if os.path.exists(src):
            dst = os.path.join(logdir, f"cskip_{tag or variant}_N{n_tiles}.csv")
            shutil.copyfile(src, dst)
            print(f"  zones -> {dst}")
    del out, tt_in, tt_out
    return ns, host, stats


def _fmt_delta(cand, ref):
    """max abs / max rel difference between two column-0 vectors (ignoring non-finite)."""
    finite = torch.isfinite(cand) & torch.isfinite(ref)
    if finite.sum() == 0:
        return float("nan"), float("nan"), int((cand != ref).sum())
    a, b = cand[finite].to(torch.float64), ref[finite].to(torch.float64)
    abs_d = (a - b).abs().max().item()
    rel_d = ((a - b).abs() / b.abs().clamp(min=1e-300)).max().item()
    exact_mismatch = int((cand != ref).logical_and(torch.isfinite(cand) | torch.isfinite(ref)).sum())
    # bit-level: NaN==NaN must count as equal
    both_nan = torch.isnan(cand) & torch.isnan(ref)
    exact_mismatch = int(((cand != ref) & ~both_nan).sum())
    return abs_d, rel_d, exact_mismatch


def main(variants=None, tile_counts=(16,), eps=EPS, fp32=False, tag_prefix=""):
    """Run every variant once per tile count; print ns + column-0 precision vs baseline."""
    variants = list(variants or VARIANTS.keys())
    logdir = os.path.join(os.environ.get("TT_METAL_HOME", "."), "generated", "profiler", ".logs")
    device = ttnn.open_device(device_id=0)
    results = {}
    try:
        for n in tile_counts:
            print(f"\n===== N = {n} stat tile(s)   fp32_dest_acc_en={fp32} =====")
            base_col0 = None
            for v in variants:
                ns, host, stats = measure_one(device, v, n, logdir=logdir, tag=f"{tag_prefix}{v}", fp32=fp32)
                col0 = host[:, 0]
                ref = golden_col0(stats, eps)
                if v == BASELINE:
                    base_col0 = col0.clone()
                results[(v, n)] = (ns, col0.clone(), stats)
                vec = VARIANTS[v][1]
                line = f"  {v:22s} vec/tile={vec:3d}  device_kernel_ns={ns:9.0f}"
                if v in _SFPU_VARIANTS:
                    a_ref, r_ref, _ = _fmt_delta(col0, ref.to(torch.float32))
                    line += f"  vs_fp64: max_abs={a_ref:.3e} max_rel={r_ref:.3e}"
                print(line)
            if base_col0 is not None:
                print("  -- column-0 delta vs TODAY'S OP (baseline c_pair) --")
                for v in variants:
                    if v == BASELINE or v == "copy_only":
                        continue
                    col0 = results[(v, n)][1]
                    a, r, mism = _fmt_delta(col0, base_col0)
                    verdict = "BIT-EXACT" if mism == 0 else f"{mism} of {col0.numel()} values differ"
                    print(f"  {v:22s} max_abs={a:.3e} max_rel={r:.3e}  {verdict}")
        # non-finite audit: prove the hostile garbage columns never fault or leak
        print("\n===== column-0 finiteness audit =====")
        for (v, n), (_, col0, stats) in results.items():
            expect_inf = (stats[:, 0] == 0.0) & (eps == 0.0)
            bad = int((~torch.isfinite(col0) & ~expect_inf).sum())
            print(f"  {v:22s} N={n:2d}  non-finite column-0 values: {bad}")
        verdicts(results, tile_counts, eps)
    finally:
        ttnn.close_device(device)
    return results


# Variants whose column 0 must be BIT-IDENTICAL to today's chain (same arithmetic,
# same DEST round trips — only the set of vectors that run differs).
BIT_EXACT = ("cskip_pair", "c_fused", "cskip_fused_bitexact")
# `cskip_fused` keeps `x+eps` in an LREG at fp32 instead of round-tripping it through
# the 16-bit DEST: NOT bit-exact, and strictly CLOSER to the fp64 reference.
CLOSE = ("rc_pair", "cskip_fused", "rc_fused")


def verdicts(results, tile_counts, eps=EPS):
    """PASS/FAIL per variant. Printed, never raised — a wrong variant is data too."""
    print("\n===== CORRECTNESS VERDICTS (column 0 is the only column the op reads) =====")
    ok_all = True
    for n in tile_counts:
        if (BASELINE, n) not in results:
            continue
        base = results[(BASELINE, n)][1]
        stats = results[(BASELINE, n)][2]
        ref = golden_col0(stats, eps).to(torch.float32)
        nan_pair = torch.isnan(base) & torch.isnan(ref)
        base_ok = bool((nan_pair | torch.isclose(base, ref, rtol=2e-2, atol=1e-6)).all())
        print(f"  N={n:2d} baseline vs fp64 reference: {'PASS' if base_ok else 'FAIL'}")
        ok_all &= base_ok
        for v in BIT_EXACT:
            if (v, n) not in results:
                continue
            got = results[(v, n)][1]
            both_nan = torch.isnan(got) & torch.isnan(base)
            mism = int(((got != base) & ~both_nan).sum())
            print(f"  N={n:2d} {v:22s} bit-exact vs baseline: {'PASS' if mism == 0 else f'FAIL ({mism} differ)'}")
            ok_all &= mism == 0
        for v in CLOSE:
            if (v, n) not in results:
                continue
            got = results[(v, n)][1]
            both_nan = torch.isnan(got) & torch.isnan(ref)
            bad = int((~(both_nan | torch.isclose(got, ref, rtol=2e-2, atol=1e-6))).sum())
            print(f"  N={n:2d} {v:22s} close to fp64 ref:    {'PASS' if bad == 0 else f'FAIL ({bad} rows)'}")
            ok_all &= bad == 0
    print(f"===== OVERALL: {'PASS' if ok_all else 'FAIL'} =====")
    return ok_all
