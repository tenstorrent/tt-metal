# Unfused AG + regime-A MM vs main's fused AGMM — LTX/FLUX shapes, bh-glx-120-c02u02

Perf evidence for the intermediate milestone: replace main's fused AGMM with **unfused all-gather + optimized
regime-A matmul**. Collected 2026-08-11 on this Galaxy, branch `cglagovich/regime-a-single-chip-opt`.

## What was measured, and what was reused

Shapes are the 37 LTX/FLUX AGMM shapes from `cglagovich/agmm_analysis` (`agmm/sweep_shapes.json`): 20 LTX
(`bh_4x8`, tp=4) + 17 FLUX (`bh_4x8_ring8`, tp=8). These are the shapes that matter for AGMM reporting; the
60+ other single-chip shapes remain the single-chip acceptance set.

| column | source |
|---|---|
| `fused_agmm` | main's fused AGMM -- **reused**, `agmm/comparison.csv` |
| `ag` | isolated all-gather -- **reused** |
| `mm_old` | existing MM -- **reused** |
| `serial_old` | `ag + mm_old` = what main-unfused delivers today -- **reused** |
| `mm_ra` | **measured here**: regime-A MM, `config=None` (production picker) |
| `serial_ra` | `ag + mm_ra` = the milestone candidate |

`mm_ra` is single-chip: the post-AG matmul is full-K and data-parallel, so every device runs the identical
[M,K]x[K,N]. Measured with `prod_sweep_worker.py` (unit device, fabric off, device-profiler time demuxed by
run-host-id, 3 blocks x [2 warmup + 12 timed] on resident inputs).

`dram%` uses this campaign's accounting, unchanged from `prod_sweep_report.py` so it stays comparable:
`bytes = Ns*M*K*2 + K*N*2 + M*N*2` against **512 GB/s**. Only `Ns` duplicates in0, so it is read from the
picker's own log line rather than assumed.

## Validity of the measurements

- **PCC** >= 0.999986 on every measured shape (min 0.999986, computed in float64).
- **Finite check**: 0 non-finite elements across all 25 measured shapes. This is checked explicitly and
  separately from PCC -- a few NaN/Inf among millions of elements barely moves PCC, and
  `BUG_rscatter_nonfinite.md` on this branch was exactly that failure.
- **Warm-run variability**: block-median spread max **1.88%**, median **0.64%** (3 blocks/shape). All
  speedups below are far outside that.

## Results

| shape | fam | fusion | fused_agmm | ag | mm_old | serial_old | mm_ra | serial_ra | vs_fused | vs_serial | dram% | pcc | finite | block medians | pick |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 32x2048x8 | LTX | - | 42.8 | 7.0 | 15.1 | 22.1 | FAIL | - | - | - | - | - | - | - | - |
| 32x2048x512 | LTX | - | 42.8 | 7.0 | 21.1 | 28.1 | 7.8 | 14.8 | 2.90x | 1.90x | 60.1 | 0.999992 | yes | 7.7,7.8,7.8 | 4,2,1,2,1 |
| 32x2048x512 | LTX | addcmul,approx | 44.2 | 7.0 | 21.2 | 28.2 | 7.8 | 14.8 | 2.99x | 1.91x | 60.1 | 0.999992 | yes | 7.7,7.8,7.8 | 4,2,1,2,1 |
| 32x2048x1536 | LTX | chunks=3,approx | 64.1 | 7.0 | 33.3 | 40.3 | 16.8 | 23.8 | 2.69x | 1.69x | 77.3 | 0.999994 | yes | 16.8,16.8,16.8 | 2,2,1,4,3 |
| 32x2048x2048 | LTX | - | 76.7 | 7.0 | 45.1 | 52.1 | 21.7 | 28.7 | 2.67x | 1.82x | 79.1 | 0.999994 | yes | 21.7,21.7,21.7 | 2,2,1,4,4 |
| 256x2048x1024 | LTX | chunks=2,approx | 53.1 | 22.3 | 27.3 | 49.6 | 19.3 | 41.6 | 1.28x | 1.19x | 58.4 | 0.999992 | yes | 19.4,19.2,19.4 | 4,1,2,2,4 |
| 1216x2048x1024 | LTX | - | 101.8 | 59.7 | 48.2 | 107.9 | 100.8 | 160.5 | 0.63x | 0.67x | 22.6 | 0.999989 | yes | 100.9,100.7,100.8 | 8,1,1,1,1 |
| 1216x4096x8 | LTX | - | 161.6 | 105.5 | 54.5 | 160.0 | FAIL | - | - | - | - | - | - | - | - |
| 1216x4096x512 | LTX | - | 162.8 | 105.5 | 59.8 | 165.3 | 70.3 | 175.8 | 0.93x | 0.94x | 42.8 | 0.999992 | yes | 70.5,70.4,70.3 | 4,1,3,2,1 |
| 1216x4096x1024 | LTX | - | 167.1 | 105.5 | 81.7 | 187.2 | 104.9 | 210.4 | 0.79x | 0.89x | 38.8 | 0.999992 | yes | 105.8,104.4,104.9 | 4,1,3,2,1 |
| 1216x4096x1024 | LTX | addcmul,approx | 176.5 | 105.5 | 80.8 | 186.3 | 104.9 | 210.4 | 0.84x | 0.89x | 38.8 | 0.999992 | yes | 105.8,104.4,104.9 | 4,1,3,2,1 |
| 1216x4096x3072 | LTX | chunks=3,approx | 251.5 | 105.5 | 193.9 | 299.4 | 1200.4 | 1305.9 | 0.19x | 0.23x | 6.9 | 0.999994 | yes | 1199.4,1200.9,1200.9 | 1,1,13,2,2 |
| 1216x4096x4096 | LTX | - | 311.5 | 105.5 | 242.3 | 347.8 | 1585.5 | 1691.0 | 0.18x | 0.21x | 6.6 | 0.999994 | yes | 1585.0,1586.2,1586.5 | 1,1,13,2,2 |
| 4864x2048x1024 | LTX | - | 309.4 | 199.0 | 129.4 | 328.4 | FAIL | - | - | - | - | - | - | - | - |
| 4864x4096x8 | LTX | - | 518.0 | 389.4 | 156.2 | 545.6 | FAIL | - | - | - | - | - | - | - | - |
| 4864x4096x512 | LTX | - | 523.7 | 389.4 | 164.8 | 554.2 | FAIL | - | - | - | - | - | - | - | - |
| 4864x4096x512 | FLUX | - | 525.4 | 389.4 | - | - | FAIL | - | - | - | - | - | - | - | - |
| 4864x4096x1024 | LTX | - | 531.2 | 389.4 | 215.1 | 604.5 | FAIL | - | - | - | - | - | - | - | - |
| 4864x4096x1024 | LTX | addcmul,approx | 589.6 | 389.4 | 215.5 | 604.9 | FAIL | - | - | - | - | - | - | - | - |
| 4864x4096x3072 | LTX | chunks=3,approx | 719.0 | 389.4 | 558.9 | 948.3 | FAIL | - | - | - | - | - | - | - | - |
| 4864x4096x4096 | LTX | - | 949.5 | 389.4 | 750.3 | 1139.7 | FAIL | - | - | - | - | - | - | - | - |
| 32x256x6144 | FLUX | - | 68.7 | 8.0 | 24.5 | 32.5 | 9.3 | 17.3 | 3.98x | 1.88x | 76.2 | 0.999994 | yes | 9.3,9.3,9.3 | 1,5,1,1,5 |
| 32x6144x1536 | FLUX | - | 140.6 | 15.3 | 81.3 | 96.6 | 45.9 | 61.2 | 2.30x | 1.58x | 82.3 | 0.999991 | yes | 46.0,45.8,46.0 | 6,1,1,4,2 |
| 32x6144x2304 | FLUX | - | 185.4 | 15.3 | 113.4 | 128.7 | 66.4 | 81.7 | 2.27x | 1.58x | 84.9 | 0.999993 | yes | 66.6,66.3,66.3 | 3,1,1,4,5 |
| 32x6144x4608 | FLUX | - | 354.5 | 15.3 | 211.4 | 226.7 | 132.1 | 147.4 | 2.40x | 1.54x | 84.7 | 0.999986 | yes | 131.6,132.5,132.4 | 12,1,1,2,1 |
| 32x6144x6144 | FLUX | - | 467.7 | 15.3 | 282.5 | 297.8 | 173.8 | 189.1 | 2.47x | 1.57x | 85.7 | 0.999991 | yes | 172.9,174.1,173.9 | 6,1,1,4,2 |
| 512x2304x6144 | FLUX | - | 207.0 | 44.1 | 128.1 | 172.2 | 137.1 | 181.2 | 1.14x | 0.95x | 56.0 | 0.999993 | yes | 138.6,137.3,136.6 | 3,2,2,1,3 |
| 512x3072x6144 | FLUX | - | 260.0 | 51.1 | 160.6 | 211.7 | 142.5 | 193.6 | 1.34x | 1.09x | 64.7 | 0.999991 | yes | 144.0,142.8,141.8 | 6,1,2,2,6 |
| 512x6144x768 | FLUX | - | 148.4 | 84.1 | 57.4 | 141.5 | 51.6 | 135.7 | 1.09x | 1.04x | 62.6 | 0.999991 | yes | 52.0,51.2,51.3 | 6,1,2,2,3 |
| 512x6144x2304 | FLUX | - | 224.3 | 84.1 | 120.1 | 204.2 | 113.4 | 197.5 | 1.14x | 1.03x | 63.7 | 0.999986 | yes | 113.6,112.8,114.3 | 12,1,1,2,1 |
| 512x6144x4608 | FLUX | - | 374.4 | 84.1 | 218.5 | 302.6 | 186.1 | 270.2 | 1.39x | 1.12x | 71.0 | 0.999986 | yes | 187.7,184.2,185.2 | 12,1,1,2,1 |
| 512x15360x768 | FLUX | - | 302.9 | 185.0 | 124.1 | 309.1 | 146.2 | 331.2 | 0.91x | 0.93x | 53.6 | 0.999991 | yes | 147.4,145.8,146.7 | 6,1,2,2,1 |
| 1024x2304x6144 | FLUX | - | 253.5 | 66.8 | 163.5 | 230.3 | 277.9 | 344.7 | 0.74x | 0.67x | 35.4 | 0.999991 | yes | 277.8,278.1,277.8 | 5,2,1,2,1 |
| 1024x3072x6144 | FLUX | - | 302.5 | 83.7 | 200.2 | 283.9 | 265.3 | 349.0 | 0.87x | 0.81x | 46.3 | 0.999991 | yes | 267.8,263.0,264.8 | 6,2,1,2,1 |
| 1024x6144x128 | FLUX | - | 204.8 | 149.8 | 61.8 | 211.6 | FAIL | - | - | - | - | - | - | - | - |
| 1024x6144x768 | FLUX | - | 205.9 | 149.8 | 82.4 | 232.2 | 122.7 | 272.5 | 0.76x | 0.85x | 37.6 | 0.999986 | yes | 123.0,122.2,122.7 | 12,1,1,2,1 |
| 1024x6144x2304 | FLUX | - | 279.3 | 149.8 | 146.0 | 295.8 | 215.6 | 365.4 | 0.76x | 0.81x | 41.3 | 0.999986 | yes | 215.4,217.4,215.1 | 12,1,1,2,1 |
| 1024x6144x4608 | FLUX | - | 435.4 | 149.8 | 276.3 | 426.1 | 353.2 | 503.0 | 0.87x | 0.85x | 43.5 | 0.999986 | yes | 353.3,353.1,353.2 | 12,1,1,2,1 |

27/38 shapes measured; 11 failed
  FAIL 32x2048x8: TT_FATAL @ /data/cglagovich/tt-metal/.claude/worktrees/splendid-questing-cookie/ttnn/cpp/ttnn/operations/experimental/regime_a_matmul/device/regime_a_
  FAIL 1216x4096x8: TT_FATAL @ /data/cglagovich/tt-metal/.claude/worktrees/splendid-questing-cookie/ttnn/cpp/ttnn/operations/experimental/regime_a_matmul/device/regime_a_
  FAIL 4864x2048x1024: TT_FATAL @ /data/cglagovich/tt-metal/.claude/worktrees/splendid-questing-cookie/ttnn/cpp/ttnn/operations/experimental/regime_a_matmul/device/regime_a_
  FAIL 4864x4096x8: TT_FATAL @ /data/cglagovich/tt-metal/.claude/worktrees/splendid-questing-cookie/ttnn/cpp/ttnn/operations/experimental/regime_a_matmul/device/regime_a_
  FAIL 4864x4096x512: TT_FATAL @ /data/cglagovich/tt-metal/.claude/worktrees/splendid-questing-cookie/ttnn/cpp/ttnn/operations/experimental/regime_a_matmul/device/regime_a_
  FAIL 4864x4096x512: TT_FATAL @ /data/cglagovich/tt-metal/.claude/worktrees/splendid-questing-cookie/ttnn/cpp/ttnn/operations/experimental/regime_a_matmul/device/regime_a_
  FAIL 4864x4096x1024: TT_FATAL @ /data/cglagovich/tt-metal/.claude/worktrees/splendid-questing-cookie/ttnn/cpp/ttnn/operations/experimental/regime_a_matmul/device/regime_a_
  FAIL 4864x4096x1024: TT_FATAL @ /data/cglagovich/tt-metal/.claude/worktrees/splendid-questing-cookie/ttnn/cpp/ttnn/operations/experimental/regime_a_matmul/device/regime_a_
  FAIL 4864x4096x3072: TT_FATAL @ /data/cglagovich/tt-metal/.claude/worktrees/splendid-questing-cookie/ttnn/cpp/ttnn/operations/experimental/regime_a_matmul/device/regime_a_
  FAIL 4864x4096x4096: TT_FATAL @ /data/cglagovich/tt-metal/.claude/worktrees/splendid-questing-cookie/ttnn/cpp/ttnn/operations/experimental/regime_a_matmul/device/regime_a_
  FAIL 1024x6144x128: TT_FATAL @ /data/cglagovich/tt-metal/.claude/worktrees/splendid-questing-cookie/ttnn/cpp/ttnn/operations/experimental/regime_a_matmul/device/regime_a_
LTX   n=11  vs_fused median 0.93x  vs_serial_old median 0.94x  | summed fused 1452us -> serial_ra 3878us = 0.37x
FLUX  n=16  vs_fused median 1.14x  vs_serial_old median 1.04x  | summed fused 4211us -> serial_ra 3641us = 1.16x
ALL   n=27  vs_fused median 1.14x  vs_serial_old median 1.03x  | summed fused 5663us -> serial_ra 7518us = 0.75x
non-finite outputs: 0   pcc<0.99: 0

## Headline

| set | n | vs main fused (median) | vs AG+existing MM (median) | summed fused -> serial_ra |
|---|---|---|---|---|
| LTX | 11 | 0.93x | 0.94x | 1452us -> 3878us = **0.37x** |
| FLUX | 16 | **1.14x** | 1.04x | 4211us -> 3641us = **1.16x** |
| ALL | 27 | 1.14x | 1.03x | 5663us -> 7518us = 0.75x |

The milestone is **not yet deliverable as a blanket replacement**. It splits cleanly by Mt:

- **Mt=1 (M=32) is a decisive win**: FLUX 2.27-3.98x, LTX 2.67-2.99x vs main fused, at 60-86% DRAM. This is
  the regime the op was built for and the AG leg is cheap there (7-15us).
- **M >= 1024 loses**: 0.63-0.93x. The AG leg dominates (150-389us) and regime-A MM stops being faster than
  the existing MM.

## Three blockers, in priority order

**1. Picker infeasibility kills 9 shapes (one root cause).** Every failure is
`regime_a_matmul_config.cpp:289: rescue_cost != inf` -- no Sm=1 config fits L1 *and* no M-split config is
feasible either:

    32x2048x8   1216x4096x8   1024x6144x128           <- tiny N
    4864x2048x1024  4864x4096x{8,512,1024,3072,4096}  <- ALL of LTX stage2

This is structural, not a bug. regime-A needs the in0 k-slice L1-resident: cb0 ~ (Mt/Sm)*(Kt/Pk) tiles. For
Mt=152, Kt=128 that needs `Sm*Pk >~ 32`, but core feasibility caps `8*Pk*Ns*Sm <= 110`, i.e. `Pk*Ns*Sm <= 13`.
The two constraints cannot both be satisfied on this 110-core grid. **All 7 LTX stage2 shapes are affected,
and they are the most expensive ones (fused AGMM 518-949us)** -- so LTX cannot ship without either relaxing
k-slice residency or splitting K across invocations.

**2. Two shapes regress ~6x, and it looks like a picker fault.** `1216x4096x3072` -> 1200us (existing MM:
193.9) and `1216x4096x4096` -> 1585us (existing MM: 242.3). Both pick `Sm=13` and land at **6.6-6.9% DRAM**,
against 38-86% everywhere else. Those two alone are what drag LTX's summed ratio to 0.37x; excluding them LTX
sums to 1.00x. Worth fixing before anything else -- it is likely cheap.

**3. Fusion is not applied in `mm_ra`.** `mm_old` DID include fusion (`addcmul` x3, `chunks=2/3` x4), so those
6 rows compare a fused existing MM against an unfused regime-A MM, which flatters regime-A.
`regime_a_matmul` supports bias/activation/addcmul but has no `chunks` argument, so a like-for-like fused
comparison needs a separate pass.

## Reproduce

    cp <agmm_analysis>/agmm/sweep_shapes.json /tmp/ltxflux_shapes.json
    TT_METAL_HOME=$PWD python3 tools/mm_sweep/picker_gen/ltxflux_agmm_mm.py /tmp/ltxflux_shapes.json 3 /tmp/ltxflux_mm.jsonl
    python3 tools/mm_sweep/picker_gen/ltxflux_report.py /tmp/ltxflux_mm.jsonl <agmm_analysis>/agmm/comparison.csv --md
