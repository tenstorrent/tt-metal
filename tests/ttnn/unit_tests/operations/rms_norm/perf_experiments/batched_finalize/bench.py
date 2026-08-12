# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""ISOLATED bake-off for BATCHING rms_norm's ROOT FINALIZE across tile-rows (Perf-2 idea I12).

Baseline = what `rms_norm_compute.cpp` does today: `eltwise_chain(EltwiseShape::tiles(rows_t),
CopyTile<input(cb_stat_sum)>, AddUnaryColValid, RsqrtColValid, PackTile<output(cb_rstd_send)>)`
with DEFAULT (per-tile / Streaming) CB policies, which force `block_size` to 1 — one DEST-sync
window and two SFPU inits PER STAT TILE.

Candidates give the chain block-capable CB policies (PerChunk or Upfront) and a `block_size`
B > 1, so one acquire/commit/wait/release and one init cover B stat tiles; plus the round-1
fusion of `+eps` and `rsqrt` into one SFPU element (which also makes the chain SFPU-init-uniform
so the init is boot-hoisted).

One core, `rows_t` column-0-valid stat tiles resident in L1, one `cp_finalize` zone around the
chain, nothing else on the clock. The precision contract is identical for every variant:
bf16 / TILE / HiFi2 / fp32_dest_acc_en=False.

Run:

    TT_METAL_DEVICE_PROFILER=1 TT_METAL_PROFILER_MID_RUN_DUMP=1 \
    TT_METAL_PROFILER_CPP_POST_PROCESS=1 timeout 1800 \
    scripts/tt-probe.sh rms_norm <<'EOF'
    import sys; sys.path.insert(0, "tests/ttnn/unit_tests/operations/rms_norm/perf_experiments")
    from batched_finalize import bench; bench.main()
    EOF

Per-TRISC numbers: reduce the copied zone CSVs with `../cskip_finalize/zone_reduce.py`.

MEASURED — box=bh-49-special (blackhole_p150b) · BH · 1350 MHz · 2026-08-12
bf16 / TILE / HiFi2 / fp32_dest_acc_en=False. `ns` = DEVICE KERNEL DURATION (median of 3 at
rows_t 1 and 32, spread < 0.5%; one fresh run elsewhere). `math` = the `cp_finalize` zone's
TRISC_1 occupancy, which is the thread on the critical path here.

  rows_t                        1      4      8     16     32     64
  ---------------------------------------------------------------------
  baseline_stream (TODAY)     699   1333   2401   4547   8807  17410
  chunk_pair    B=2             -   1373   2363   4418   8541  16779
  chunk_pair    B=4             -   1562   2595   4614   8591  16659
  chunk_pair    B=8             -      -   2714   4714   8764  16822
  chunk_fused   B=1           690   1347   2324   4193   7792  15269
  chunk_fused   B=4             -   1434   2384   4181   7753  14976
  upfront_*     (same +-1%, but publishes rstd only after the LAST tile)

  cp_finalize MATH occupancy (ns)      rows_t=1   rows_t=32   rows_t=64
    baseline_stream                        408        8684       17251
    chunk_pair  B=1 (policy only)          375        8691           -
    chunk_pair  B=4 (BATCHED)                -        8226       16301
    chunk_fused B=1 (fusion only)          484        7653       15117
    chunk_fused B=4 (both)                   -        7507       14739

VERDICT for the batching idea itself: NULL-to-marginal. One DEST window + one init per B
tiles cuts UNPACK a lot (7710 -> 6374 ns at rows_t=32) but unpack is NOT the critical path;
the finalize is SFPU-THROUGHPUT bound (16 even-parity vector ops per tile), so math only
moves 8684 -> 8226 (1.056x) and the whole kernel 8807 -> 8591 (1.025x), on the edge of the
noise band. B=8 is a MEASURED REGRESSION at rows_t 8/16 (one oversized block loses the
math/pack overlap): 2401 -> 2714 and 4547 -> 4714.

The FUSION, on the other hand, is a real 1.13x at rows_t=32 with no batching at all — and it
is the SAME number round 1 (`cskip_finalize`) measured and deliberately did NOT graduate,
because at rows_t=1 (the decode geometry) it is a math REGRESSION: 408 -> 484 ns (0.84x), 8
long dependent vectors pipelining worse than 16 short independent ones. Batching does not
change that: it is inert at rows_t=1 (B clamps to 1) and adds only ~2% on top of the fusion.
"""

from __future__ import annotations

import os
import shutil
import struct
from pathlib import Path

import torch
import ttnn

TILE = 32
KERNEL = Path(__file__).parent / "kernels" / "batched_finalize_compute.cpp"

CB_STAT_SUM = 0
CB_RSTD_SEND = 16

# name -> (CT variant id, blocked?, one-line label)
VARIANTS = {
    "baseline_stream": (0, False, "per-tile policies, block=1 — TODAY'S OP"),
    "chunk_pair": (1, True, "PerChunk policies + block=B, add+rsqrt pair"),
    "upfront_pair": (2, True, "Upfront/AtEnd + block=B, add+rsqrt pair"),
    "chunk_fused": (3, True, "PerChunk + block=B, fused rsqrt(x+eps)"),
    "upfront_fused": (4, True, "Upfront/AtEnd + block=B, fused rsqrt(x+eps)"),
    "upfront_fused_bitexact": (5, True, "same, sum round-tripped through DEST"),
    "stream_fused": (6, False, "fusion only, no batching (round-1 cskip_fused)"),
}
BASELINE = "baseline_stream"
# Column 0 must be BIT-IDENTICAL to today's chain (same arithmetic, same DEST round trips).
BIT_EXACT = ("chunk_pair", "upfront_pair", "upfront_fused_bitexact")
# Keeps `x+eps` in an LREG at fp32: NOT bit-exact, strictly closer to the fp64 reference.
CLOSE = ("chunk_fused", "upfront_fused", "stream_fused")

EPS = 1e-6


def _eps_bits(eps: float) -> int:
    return struct.unpack("<I", struct.pack("<f", float(eps)))[0]


def _memcfg(n_tiles):
    return ttnn.create_sharded_memory_config(
        shape=(TILE * n_tiles, TILE),
        core_grid=ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))]),
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )


def stat_tiles(n_tiles: int) -> torch.Tensor:
    """`n_tiles` COLUMN-0-VALID stat tiles, shaped like a real `reduce<SUM, REDUCE_ROW>` result.

    Column 0 spans the regimes the finalize must survive (the eps region, ~1, extremes);
    columns 1..31 carry deliberately HOSTILE garbage, since the op leaves them structurally
    undefined and a candidate that scopes past them must neither fault nor contaminate column 0.
    """
    torch.manual_seed(0)
    rows = n_tiles * TILE
    col0 = torch.empty(rows, dtype=torch.float32)
    regimes = [0.0, 1e-30, 1e-8, 1e-3, 0.5, 1.0, 2.0, 17.0, 1e3, 1e8, 1e18]
    for r in range(rows):
        col0[r] = regimes[r % 16] if r % 16 < len(regimes) else float(torch.rand(1).item()) * 10.0 + 1e-4
    garbage = torch.randn(rows, TILE - 1, dtype=torch.float32) * 1e3
    garbage[:, 0] = -1.0
    garbage[:, 1] = 0.0
    garbage[:, 2] = float("inf")
    garbage[:, 3] = float("nan")
    garbage[:, 4] = -3.4e38
    garbage[:, 5] = 3.4e38
    t = torch.cat([col0.unsqueeze(1), garbage], dim=1)
    return t.to(torch.bfloat16).to(torch.float32)


def golden_col0(stats: torch.Tensor, eps: float = EPS) -> torch.Tensor:
    """fp64 reference for column 0: rsqrt(x + eps)."""
    return torch.rsqrt(stats[:, 0].to(torch.float64) + eps)


def make_tensors(device, n_tiles):
    stats = stat_tiles(n_tiles)
    memcfg = _memcfg(n_tiles)
    tt_in = ttnn.from_torch(
        stats.reshape(1, 1, n_tiles * TILE, TILE),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=memcfg,
    )
    tt_out = ttnn.allocate_tensor_on_device(
        ttnn.Shape([1, 1, n_tiles * TILE, TILE]), ttnn.bfloat16, ttnn.TILE_LAYOUT, device, memcfg
    )
    return stats, tt_in, tt_out


def _compute_config():
    """The perf loose case's config — FIXED for every variant, never a perf lever."""
    cfg = ttnn.ComputeConfigDescriptor()
    cfg.math_fidelity = ttnn.MathFidelity.HiFi2
    cfg.fp32_dest_acc_en = False
    cfg.math_approx_mode = False
    return cfg


def descriptor(tt_in, tt_out, *, variant, n_tiles, blk, eps=EPS):
    vid = VARIANTS[variant][0]
    core = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))])
    compute = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL),
        core_ranges=core,
        compile_time_args=[vid, _eps_bits(eps)],
        runtime_args=[(ttnn.CoreCoord(0, 0), [n_tiles, blk])],
        config=_compute_config(),
    )
    cbs = [
        ttnn.cb_descriptor_from_sharded_tensor(CB_STAT_SUM, tt_in),
        ttnn.cb_descriptor_from_sharded_tensor(CB_RSTD_SEND, tt_out),
    ]
    return ttnn.ProgramDescriptor(kernels=[compute], semaphores=[], cbs=cbs)


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


def measure_one(device, variant, n_tiles, blk, logdir=None):
    """ONE fresh run. Returns (device_kernel_ns, column-0 result, the stat tiles)."""
    stats, tt_in, tt_out = make_tensors(device, n_tiles)
    ttnn.ReadDeviceProfiler(device)  # flush the from_torch / shard prep
    out = ttnn.generic_op([tt_in, tt_out], descriptor(tt_in, tt_out, variant=variant, n_tiles=n_tiles, blk=blk))
    ttnn.synchronize_device(device)
    ttnn.ReadDeviceProfiler(device)
    ns = _device_kernel_ns()
    host = ttnn.to_torch(out).to(torch.float32).reshape(n_tiles * TILE, TILE)
    if logdir:
        src = os.path.join(logdir, "profile_log_device.csv")
        if os.path.exists(src):
            dst = os.path.join(logdir, f"batched_{variant}_N{n_tiles}_B{blk}.csv")
            shutil.copyfile(src, dst)
    del out, tt_in, tt_out
    return ns, host[:, 0].clone(), stats


def _delta(cand, ref):
    finite = torch.isfinite(cand) & torch.isfinite(ref)
    a, b = cand[finite].to(torch.float64), ref[finite].to(torch.float64)
    abs_d = (a - b).abs().max().item() if a.numel() else float("nan")
    rel_d = ((a - b).abs() / b.abs().clamp(min=1e-300)).max().item() if a.numel() else float("nan")
    both_nan = torch.isnan(cand) & torch.isnan(ref)
    mism = int(((cand != ref) & ~both_nan).sum())
    return abs_d, rel_d, mism


def _pcc(cand, ref):
    """PCC on the finite column-0 values (the only column the op ever reads)."""
    finite = torch.isfinite(cand) & torch.isfinite(ref)
    a, b = cand[finite].to(torch.float64), ref[finite].to(torch.float64)
    if a.numel() < 2:
        return float("nan")
    a = a - a.mean()
    b = b - b.mean()
    denom = (a.norm() * b.norm()).item()
    return float((a @ b).item() / denom) if denom else 1.0


def main(tile_counts=(1, 4, 8, 16, 32, 64), blocks=(2, 4, 8), variants=None, eps=EPS):
    variants = list(variants or VARIANTS.keys())
    logdir = os.path.join(os.environ.get("TT_METAL_HOME", "."), "generated", "profiler", ".logs")
    device = ttnn.open_device(device_id=0)
    results = {}
    try:
        for n in tile_counts:
            print(f"\n===== rows_t = {n} stat tile(s) =====")
            for v in variants:
                blocked = VARIANTS[v][1]
                # the pure policy-change control also gets B=1
                bs = ([1] + [b for b in blocks if b <= max(n, 1)]) if blocked else [1]
                if blocked:
                    bs = sorted(set(bs))
                for b in bs:
                    ns, col0, stats = measure_one(device, v, n, b, logdir=logdir)
                    ref = golden_col0(stats, eps).to(torch.float32)
                    a, r, _ = _delta(col0, ref)
                    results[(v, n, b)] = (ns, col0, stats)
                    # ns is None when the run has no device profiler (TT_METAL_DEVICE_PROFILER=1);
                    # correctness still gates, perf is simply unmeasured in that mode.
                    ns_s = f"{ns:9.0f}" if ns is not None else "  (no prof)"
                    print(
                        f"  {v:24s} B={b:<2d} device_kernel_ns={ns_s}"
                        f"  pcc={_pcc(col0, ref):.6f} max_abs={a:.3e} max_rel={r:.3e}"
                    )
            base = results[(BASELINE, n, 1)][1]
            print("  -- column-0 delta vs TODAY'S OP --")
            for key, (_, col0, _) in results.items():
                if key[1] != n or key[0] == BASELINE:
                    continue
                a, r, mism = _delta(col0, base)
                print(
                    f"  {key[0]:24s} B={key[2]:<2d} max_abs={a:.3e} max_rel={r:.3e}"
                    f"  {'BIT-EXACT' if mism == 0 else f'{mism} differ'}"
                )
        verdicts(results, tile_counts, eps)
    finally:
        ttnn.close_device(device)
    return results


def verdicts(results, tile_counts, eps=EPS):
    """PASS/FAIL per variant. Printed, never raised for perf — correctness is the only gate."""
    print("\n===== CORRECTNESS VERDICTS (column 0 is the only column the op reads) =====")
    ok_all = True
    for n in tile_counts:
        if (BASELINE, n, 1) not in results:
            continue
        base = results[(BASELINE, n, 1)][1]
        stats = results[(BASELINE, n, 1)][2]
        ref = golden_col0(stats, eps).to(torch.float32)
        nan_pair = torch.isnan(base) & torch.isnan(ref)
        base_ok = bool((nan_pair | torch.isclose(base, ref, rtol=2e-2, atol=1e-6)).all())
        print(f"  rows_t={n:2d} baseline vs fp64 reference: {'PASS' if base_ok else 'FAIL'}")
        ok_all &= base_ok
        for (v, nn, b), (_, col0, _) in results.items():
            if nn != n or v == BASELINE:
                continue
            if v in BIT_EXACT:
                both_nan = torch.isnan(col0) & torch.isnan(base)
                mism = int(((col0 != base) & ~both_nan).sum())
                print(f"  rows_t={n:2d} {v:24s} B={b:<2d} bit-exact vs op: {'PASS' if mism == 0 else f'FAIL ({mism})'}")
                ok_all &= mism == 0
            else:
                both_nan = torch.isnan(col0) & torch.isnan(ref)
                bad = int((~(both_nan | torch.isclose(col0, ref, rtol=2e-2, atol=1e-6))).sum())
                print(f"  rows_t={n:2d} {v:24s} B={b:<2d} close to fp64 ref: {'PASS' if bad == 0 else f'FAIL ({bad})'}")
                ok_all &= bad == 0
    print(f"===== OVERALL: {'PASS' if ok_all else 'FAIL'} =====")
    return ok_all
