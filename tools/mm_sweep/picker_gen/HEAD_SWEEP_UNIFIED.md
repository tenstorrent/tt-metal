# Regime-A matmul perf — all shapes, measured on HEAD

The single perf report for this op. Measured at commit `fc71aea404c` (post-rebase onto main, post the
AGMM-prep fixes). Supersedes and replaces PROD_SWEEP_MT16.md (66-shape LTX u FLUX corpus) and
HEYGEN_SWEEP.md (11 filtered HeyGen shapes), both deleted; their numbers and rationale are folded in
below and their full text remains in git history.

`src` = which corpus a shape came from. `was us` / `Δ%` compare against those superseded reports
(measured on the pre-rebase build); a `*` on the config would mark a pick that moved. Sorted ascending
by effective DRAM bandwidth.

| shape | src | Mt | Pk,Ns,Sm,kb,nsb | core | reduction | placement | dev us | was us | Δ% | eff GB/s | %pk | TFLOP/s | FPU% | PCC | blk% |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 32x256x512 | HeyGen | 1 | 1,2,1,1,1 | 16 | chain | bank-local | 4.29 | 4.29 | +0.1% | 76.3 | 15% | 2.0 | 0.6% | 0.99999 | 0.9 |
| 256x2048x512 | corpus | 8 | 4,1,3,2,2 | 96 | reduce-scatter | mesh | 13.70 | 13.76 | -0.5% | 248.8 | 49% | 39.2 | 12.9% | 0.99999 | 1.0 |
| 128x2048x512 | corpus | 4 | 4,1,2,2,2 | 64 | reduce-scatter | in1-near | 10.29 | 10.22 | +0.7% | 267.6 | 52% | 26.1 | 8.6% | 0.99999 | 0.5 |
| 512x6144x768 | corpus | 16 | 12,1,1,2,1 | 96 | chain | mesh | 61.57 | 61.79 | -0.4% | 268.2 | 52% | 78.5 | 25.8% | 0.99999 | 0.1 |
| 512x15360x768 | new | 16 | 6,1,2,2,1 | 96 | chain | mesh | 144.33 | — | new | 277.9 | 54% | 83.7 | 27.5% | 0.99999 | 0.0 |
| 256x2048x1536 | corpus | 8 | 4,1,3,2,3 | 96 | reduce-scatter | in1-near | 27.73 | 27.48 | +0.9% | 293.1 | 57% | 58.1 | 19.1% | 1.00000 | 0.1 |
| 256x2048x1024 | corpus | 8 | 4,1,2,2,4 | 64 | reduce-scatter | in1-near | 19.47 | 19.54 | -0.4% | 296.2 | 58% | 55.2 | 18.1% | 0.99999 | 1.8 |
| 64x2048x512 | corpus | 2 | 4,2,1,2,1 | 64 | chain | bank-local | 8.85 | 8.83 | +0.3% | 303.5 | 59% | 15.2 | 5.0% | 0.99999 | 0.7 |
| 256x2048x2048 | corpus | 8 | 4,1,3,2,4 | 96 | reduce-scatter | in1-near | 34.45 | 34.38 | +0.2% | 304.3 | 59% | 62.3 | 20.5% | 1.00000 | 1.1 |
| 512x6144x1536 | corpus | 16 | 12,1,1,2,1 | 96 | chain | mesh | 87.17 | 86.66 | +0.6% | 306.7 | 60% | 110.9 | 36.5% | 0.99999 | 0.3 |
| 512x5120x2560 | HeyGen | 16 | 10,1,1,2,1 | 80 | chain | mesh | 109.79 | 109.54 | +0.2% | 310.4 | 61% | 122.2 | 40.2% | 1.00005 | 0.1 |
| 512x2304x6144 | corpus | 16 | 3,4,1,1,1 | 96 | chain | mesh | 141.59 | 141.28 | +0.2% | 311.0 | 61% | 102.4 | 33.7% | 1.00009 | 0.1 |
| 512x3072x6144 | corpus | 16 | 6,2,1,2,1 | 96 | chain | mesh | 158.68 | 158.31 | +0.2% | 317.2 | 62% | 121.8 | 40.0% | 1.00001 | 0.4 |
| 32x2048x512 | corpus | 1 | 4,2,1,2,1 | 64 | chain | bank-local | 7.34 | 7.32 | +0.3% | 325.7 | 64% | 9.1 | 3.0% | 0.99999 | 0.5 |
| 512x5120x5120 | HeyGen | 16 | 10,1,1,2,1 | 80 | chain | mesh | 187.63 | 187.17 | +0.2% | 335.3 | 65% | 143.1 | 47.0% | 1.00009 | 0.0 |
| 512x6144x2304 | corpus | 16 | 12,1,1,2,1 | 96 | chain | mesh | 109.98 | 109.95 | +0.0% | 336.1 | 66% | 131.8 | 43.3% | 1.00002 | 0.0 |
| 128x2048x1024 | corpus | 4 | 4,1,2,2,4 | 64 | reduce-scatter | in1-near | 14.50 | 14.51 | -0.1% | 343.5 | 67% | 37.0 | 12.2% | 0.99999 | 0.4 |
| 512x4096x5120 | HeyGen | 16 | 4,3,1,2,1 | 96 | chain | mesh | 173.44 | 173.27 | +0.1% | 344.6 | 67% | 123.8 | 40.7% | 1.00009 | 0.4 |
| 256x6144x768 | corpus | 8 | 12,1,1,2,1 | 96 | chain | mesh | 36.11 | 35.72 | +1.1% | 359.3 | 70% | 66.9 | 22.0% | 0.99999 | 0.5 |
| 256x15360x768 | corpus | 8 | 6,1,2,2,3 | 96 | reduce-scatter | mesh | 87.02 | 86.86 | +0.2% | 366.0 | 71% | 69.4 | 22.8% | 0.99999 | 0.1 |
| 128x6144x768 | corpus | 4 | 12,1,1,2,1 | 96 | chain | bank-local | 30.48 | 30.37 | +0.4% | 367.7 | 72% | 39.6 | 13.0% | 0.99999 | 0.6 |
| 256x6144x1536 | corpus | 8 | 6,1,2,4,2 | 96 | reduce-scatter | in1-near | 61.00 | 60.74 | +0.4% | 373.9 | 73% | 79.2 | 26.0% | 0.99999 | 0.4 |
| 512x6144x4608 | corpus | 16 | 12,1,1,2,1 | 96 | chain | mesh | 180.31 | 180.16 | +0.1% | 375.1 | 73% | 160.8 | 52.9% | 1.00009 | 0.1 |
| 128x2048x1536 | corpus | 4 | 4,3,1,2,1 | 96 | chain | bank-local | 21.95 | 21.93 | +0.1% | 376.2 | 73% | 36.7 | 12.1% | 0.99999 | 1.0 |
| 64x2048x1024 | corpus | 2 | 4,2,1,2,2 | 64 | reduce-scatter | bank-local | 12.20 | 12.16 | +0.3% | 397.5 | 78% | 22.0 | 7.2% | 0.99999 | 0.4 |
| 64x6144x768 | corpus | 2 | 12,1,1,2,1 | 96 | chain | bank-local | 25.63 | 25.56 | +0.3% | 402.7 | 79% | 23.6 | 7.7% | 0.99999 | 0.3 |
| 256x6144x2304 | corpus | 8 | 12,1,1,2,1 | 96 | chain | mesh | 79.98 | 79.79 | +0.2% | 408.1 | 80% | 90.6 | 29.8% | 0.99999 | 0.1 |
| 256x15360x1536 | corpus | 8 | 6,1,2,2,6 | 96 | reduce-scatter | in1-near | 136.04 | 135.53 | +0.4% | 410.4 | 80% | 88.8 | 29.2% | 1.00002 | 1.1 |
| 32x256x6144 | corpus | 1 | 1,3,1,1,8 | 24 | chain | bank-local | 8.65 | 8.62 | +0.4% | 414.7 | 81% | 11.6 | 3.8% | 0.99999 | 0.1 |
| 256x6080x4640 | corpus | 8 | 12,1,1,2,1 | 96 | chain | mesh | 148.48 | 148.77 | -0.2% | 417.0 | 81% | 97.3 | 32.0% | 1.00002 | 0.0 |
| 64x2048x1536 | corpus | 2 | 4,3,1,2,1 | 96 | chain | bank-local | 17.44 | 17.35 | +0.5% | 417.0 | 81% | 23.1 | 7.6% | 0.99999 | 0.2 |
| 32x256x5120 | HeyGen | 1 | 1,5,1,1,4 | 40 | chain | bank-local | 7.23 | 7.23 | +0.0% | 419.1 | 82% | 11.6 | 3.8% | 0.99999 | 0.3 |
| 256x2304x6144 | corpus | 8 | 3,4,1,1,3 | 96 | chain | bank-local | 85.63 | 85.29 | +0.4% | 422.5 | 83% | 84.6 | 27.8% | 1.00001 | 0.9 |
| 32x6144x768 | corpus | 1 | 6,1,1,2,3 | 48 | chain | bank-local | 23.20 | 23.15 | +0.2% | 425.8 | 83% | 13.0 | 4.3% | 0.99999 | 0.2 |
| 128x2048x2048 | corpus | 4 | 4,3,1,2,3 | 96 | reduce-scatter | bank-local | 24.49 | 24.44 | +0.2% | 428.2 | 84% | 43.8 | 14.4% | 0.99999 | 1.2 |
| 128x6144x1536 | corpus | 4 | 12,1,1,2,2 | 96 | chain | bank-local | 48.42 | 48.42 | -0.0% | 430.4 | 84% | 49.9 | 16.4% | 0.99999 | 0.8 |
| 128x15360x768 | corpus | 4 | 6,1,1,2,3 | 48 | reduce-scatter | bank-local | 63.89 | 63.84 | +0.1% | 433.9 | 85% | 47.3 | 15.5% | 0.99999 | 0.3 |
| 32x2048x1024 | corpus | 1 | 2,4,1,4,1 | 64 | chain | bank-local | 10.94 | 10.95 | -0.1% | 437.3 | 85% | 12.3 | 4.0% | 0.99999 | 0.2 |
| 32x5120x1280 | HeyGen | 1 | 10,1,1,2,1 | 80 | chain | bank-local | 30.86 | 30.93 | -0.2% | 438.0 | 86% | 13.6 | 4.5% | 0.99999 | 0.1 |
| 256x6144x6144 | corpus | 8 | 6,1,2,4,2 | 96 | reduce-scatter | in1-near | 186.68 | 187.48 | -0.4% | 438.1 | 86% | 103.5 | 34.0% | 1.00003 | 0.4 |
| 256x6144x4608 | corpus | 8 | 6,1,2,4,2 | 96 | reduce-scatter | in1-near | 141.57 | 141.37 | +0.1% | 438.9 | 86% | 102.4 | 33.7% | 1.00002 | 0.2 |
| 64x2048x2048 | corpus | 2 | 2,3,1,2,3 | 48 | chain | bank-local | 21.49 | 21.63 | -0.6% | 439.1 | 86% | 25.0 | 8.2% | 0.99999 | 0.1 |
| 32x2048x1536 | corpus | 1 | 2,2,1,4,3 | 32 | chain | bank-local | 15.04 | 15.02 | +0.1% | 442.4 | 86% | 13.4 | 4.4% | 0.99999 | 0.0 |
| 128x6080x4640 | corpus | 4 | 12,1,1,2,2 | 96 | chain | bank-local | 133.26 | 133.29 | -0.0% | 444.0 | 87% | 54.2 | 17.8% | 1.00001 | 0.3 |
| 64x15360x768 | corpus | 2 | 10,1,1,2,3 | 80 | chain | bank-local | 57.69 | 57.82 | -0.2% | 444.7 | 87% | 26.2 | 8.6% | 0.99999 | 0.7 |
| 128x6144x2304 | corpus | 4 | 12,1,1,2,1 | 96 | chain | bank-local | 67.66 | 67.84 | -0.3% | 450.4 | 88% | 53.6 | 17.6% | 0.99999 | 0.6 |
| 128x2304x6144 | corpus | 4 | 3,2,1,1,6 | 48 | chain | bank-local | 68.54 | 68.40 | +0.2% | 453.2 | 89% | 52.9 | 17.4% | 1.00001 | 0.5 |
| 96x5120x2560 | HeyGen | 3 | 10,1,1,2,2 | 80 | chain | bank-local | 61.05 | 60.92 | +0.2% | 453.6 | 89% | 41.2 | 13.6% | 0.99999 | 0.4 |
| 32x2048x2048 | corpus | 1 | 2,2,1,4,4 | 32 | chain | bank-local | 19.34 | 19.34 | +0.0% | 454.0 | 89% | 13.9 | 4.6% | 0.99999 | 0.2 |
| 64x6080x4640 | corpus | 2 | 12,1,1,2,2 | 96 | chain | bank-local | 126.93 | 127.02 | -0.1% | 455.3 | 89% | 28.4 | 9.4% | 0.99999 | 0.1 |
| 64x6144x2304 | corpus | 2 | 12,1,1,2,2 | 96 | chain | bank-local | 64.04 | 63.80 | +0.4% | 459.0 | 90% | 28.3 | 9.3% | 0.99999 | 0.4 |
| 128x15360x1536 | corpus | 4 | 12,1,1,1,3 | 96 | chain | bank-local | 111.83 | 111.67 | +0.1% | 460.6 | 90% | 54.0 | 17.8% | 0.99999 | 0.1 |
| 64x6144x1536 | corpus | 2 | 3,1,1,8,2 | 24 | chain | bank-local | 42.82 | 42.79 | +0.1% | 463.8 | 91% | 28.2 | 9.3% | 0.99999 | 0.9 |
| 96x2048x5120 | HeyGen | 3 | 4,3,1,2,2 | 96 | reduce-scatter | bank-local | 49.78 | 49.76 | +0.0% | 464.7 | 91% | 40.4 | 13.3% | 0.99999 | 0.2 |
| 32x6080x4640 | corpus | 1 | 12,1,1,2,1 | 96 | chain | bank-local | 122.75 | 122.73 | +0.0% | 465.2 | 91% | 14.7 | 4.8% | 0.99999 | 0.3 |
| 128x6144x6144 | corpus | 4 | 12,1,1,2,2 | 96 | chain | bank-local | 167.94 | 167.63 | +0.2% | 468.3 | 91% | 57.5 | 18.9% | 1.00001 | 0.2 |
| 32x6144x1536 | corpus | 1 | 6,1,1,4,2 | 48 | chain | bank-local | 41.24 | 40.56 | +1.7% | 469.6 | 92% | 14.6 | 4.8% | 0.99999 | 0.0 |
| 128x6144x4608 | corpus | 4 | 12,1,1,2,1 | 96 | chain | bank-local | 125.52 | 125.27 | +0.2% | 473.1 | 92% | 57.7 | 19.0% | 1.00001 | 0.0 |
| 96x5120x5120 | HeyGen | 3 | 10,1,1,2,2 | 80 | chain | bank-local | 114.12 | 113.91 | +0.2% | 476.6 | 93% | 44.1 | 14.5% | 1.00000 | 0.1 |
| 96x8192x5120 | HeyGen | 3 | 4,3,1,2,2 | 96 | reduce-scatter | bank-local | 187.73 | 187.47 | +0.1% | 477.2 | 93% | 42.9 | 14.1% | 1.00000 | 0.3 |
| 32x15360x768 | corpus | 1 | 6,1,1,2,3 | 48 | chain | bank-local | 51.49 | 51.24 | +0.5% | 478.2 | 93% | 14.7 | 4.8% | 0.99999 | 0.2 |
| 32x1280x30720 | HeyGen | 1 | 5,2,1,1,2 | 80 | chain | bank-local | 167.28 | 167.06 | +0.1% | 482.9 | 94% | 15.0 | 4.9% | 1.00002 | 0.3 |
| 64x6144x6144 | corpus | 2 | 12,1,1,2,2 | 96 | chain | bank-local | 159.06 | 158.72 | +0.2% | 484.5 | 95% | 30.4 | 10.0% | 0.99999 | 0.1 |
| 48x6144x4608 | corpus | 2 | 6,1,1,4,2 | 48 | chain | bank-local | 118.68 | 118.38 | +0.3% | 485.8 | 95% | 22.9 | 7.5% | 1.00000 | 0.1 |
| 64x15360x1536 | corpus | 2 | 6,1,1,2,3 | 48 | chain | bank-local | 101.33 | 101.40 | -0.1% | 487.0 | 95% | 29.8 | 9.8% | 0.99999 | 0.1 |
| 64x2304x6144 | corpus | 2 | 3,4,1,1,3 | 96 | chain | bank-local | 62.06 | 62.01 | +0.1% | 487.8 | 95% | 29.2 | 9.6% | 1.00000 | 0.2 |
| 32x6100x4608 | corpus | 1 | 12,1,1,2,1 | 96 | chain | bank-local | 116.41 | 116.34 | +0.1% | 488.8 | 95% | 15.5 | 5.1% | 0.99999 | 0.0 |
| 32x6144x4600 | corpus | 1 | 12,1,1,2,1 | 96 | chain | bank-local | 116.93 | 116.73 | +0.2% | 489.3 | 96% | 15.5 | 5.1% | 0.99999 | 0.1 |
| 32x2304x6144 | corpus | 1 | 3,2,1,1,6 | 48 | chain | bank-local | 59.24 | 59.63 | -0.7% | 489.5 | 96% | 15.3 | 5.0% | 0.99999 | 0.2 |
| 32x6144x4608 | corpus | 1 | 12,1,1,2,1 | 96 | chain | bank-local | 117.05 | 116.91 | +0.1% | 489.6 | 96% | 15.5 | 5.1% | 0.99999 | 0.0 |
| 64x6144x4608 | corpus | 2 | 6,1,1,4,2 | 48 | chain | bank-local | 118.46 | 118.38 | +0.1% | 489.6 | 96% | 30.6 | 10.1% | 0.99999 | 0.0 |
| 32x6144x2304 | corpus | 1 | 3,1,1,4,5 | 24 | chain | bank-local | 58.74 | 58.59 | +0.3% | 491.1 | 96% | 15.4 | 5.1% | 0.99999 | 0.3 |
| 32x6144x3072 | corpus | 1 | 3,1,1,4,6 | 24 | chain | bank-local | 78.05 | 77.80 | +0.3% | 491.2 | 96% | 15.5 | 5.1% | 0.99999 | 0.4 |
| 64x4608x6144 | corpus | 2 | 3,2,1,2,3 | 48 | chain | bank-local | 118.91 | 118.90 | +0.0% | 492.7 | 96% | 30.5 | 10.0% | 0.99999 | 0.3 |
| 32x15360x1536 | corpus | 1 | 6,1,1,2,3 | 48 | chain | bank-local | 97.76 | 97.92 | -0.2% | 493.7 | 96% | 15.4 | 5.1% | 0.99999 | 0.1 |
| 64x6144x9216 | corpus | 2 | 6,1,1,4,2 | 48 | chain | bank-local | 231.57 | 231.81 | -0.1% | 497.5 | 97% | 31.3 | 10.3% | 1.00000 | 0.1 |
| 32x6144x6144 | corpus | 1 | 6,1,1,4,2 | 48 | chain | bank-local | 152.84 | 152.99 | -0.1% | 499.1 | 97% | 15.8 | 5.2% | 0.99999 | 0.1 |
| 32x6144x9216 | corpus | 1 | 3,1,1,4,6 | 24 | chain | bank-local | 227.96 | 227.89 | +0.0% | 501.1 | 98% | 15.9 | 5.2% | 0.99999 | 0.3 |

**78 shapes measured on HEAD** (66 corpus, 11 HeyGen, 1 newly runnable). All at defaults (config=None).

- effective DRAM BW: min 76.3, median 438.1, max 501.1 GB/s (peak 512); median 86% of peak
- FPU vs full 110-core grid: median 11.2%, max 52.9% (512x6144x4608)
- correctness: 78/78 PCC >= 0.999 (min 0.99999)
- stability: median block spread 0.2%, worst 1.8% (256x2048x1024)
- **vs the pre-fix reports (77 comparable): median +0.1%, best -0.7%, worst +1.7%**
  - beyond +-3%: 0 slower, 0 faster
- reduction: 16 reduce-scatter / 62 chain    placement: 15 mesh / 9 in1-near / 54 bank-local
- config changes (*): 0
- reduction-strategy changes: 0

**Not measured (1):**

- `32x128x30720` (new): TT_FATAL @ /localdev/cglagovich/tt-metal/ttnn/cpp/ttnn/operations/experimental/regime_a_matmul/

---

## Method

`config=None` throughout, i.e. the production picker chooses -- no diagnostic mask, no env override that
changes behaviour. Device time comes from the device-profiler CSV demuxed by run-host-id, so it is the op
alone rather than host wall. Each shape runs 2 blocks x (2 warmup + 12 timed) iterations on resident
inputs; `dev us` is the median over all 24 timed iterations. One worker SUBPROCESS per shape, because the
profiler CSV is only flushed at `close_device`.

Regenerate: `python tools/mm_sweep/picker_gen/head_sweep.py` (resumable) then `unified_report.py`.
The `was us` baseline is read from an `old.json` extracted from the superseded reports; without it the
generator still produces the table, just without the comparison columns.

### Column definitions

- **eff GB/s** = `(Ns*M*K + K*N + M*N)*2 / device time`. in0 counts `Ns` times because each of the `Ns`
  n-slice groups reads all of in0; within an 8-bank ring the 8 cores read DIFFERENT shards, so there is no
  duplication across a ring, nor across `Pk` or `Sm`. The in0 ring FORWARD traffic is NoC-only and is
  excluded -- this is a DRAM metric. Padded positions are never DRAM-read (balanced tails), so these bytes
  are the bytes physically moved.
- **%pk** against a 512 GB/s DRAM peak (the measured BH ceiling used throughout this campaign).
- **FPU%** against this BOARD's full compute grid, device-queried at 11x10 = 110 cores => 304 TFLOP/s.
  Per `tech_reports/GEMM_FLOPS/GEMM_FLOPS.md` the BH matrix engine computes 8x16 x 16x16 per cycle =
  `2*8*16*16` = 4096 FLOP/cycle, and MATH_FIDELITY divides it; this op is bf16 in/out, HiFi2, fp32
  accumulate => 2048 FLOP/cycle/core = 2.765 TFLOP/s per core at 1.35 GHz. Note 110 cores is NOT the
  13x10 = 130 the tech report quotes for Blackhole generally -- this board has harvested columns.
  fp32 dest accumulation costs DST capacity, not MAC throughput.
- **blk%** = spread between the two 12-iteration blocks. The noise floor on this board is +-2.4% on the
  small 2048-K shapes and +-1.5% on the large ones.
- **cfg / reduction / placement** come from the factory's own log (`TT_REGIME_A_LOG_CFG=1`), i.e. ground
  truth for what was actually built rather than a re-derivation of the picker.

## Filter applied to the 23 HeyGen shapes

Kept a shape only if it is BOTH memory-bound by roofline AND servable by regime-A. Machine balance on this
board = 304 TFLOP/s (110 cores x 2.765 TFLOP/s, bf16 HiFi2) / 512 GB/s = **594 FLOP/byte**; a shape is
memory-bound when its arithmetic intensity `MNK/(MK+KN+MN)` is below that.

Regime-A structural requirements (not tunable -- they follow from the in0 8-bank ring and the in1 8-bank width
shard):
- **Nt wide enough to width-shard over 8 banks**: `7*ceil(Nt/8) < Nt`.
- **Kt >= 8**: the k-slice is distributed over exactly 8 banks, so the smallest slice is 8 tiles; the picker
  rejects anything needing more than 20% K padding.
- **M < N**: regime A shards in1, so it assumes in1 is the big operand. M >= N is regime B.

| shape | Mt | Kt | Nt | AI | t_dram us | t_comp us | verdict |
|---|---|---|---|---|---|---|---|
| 96x8192x5120 | 3 | 256 | 160 | 93 | 168.8 | 26.5 | KEEP |
| 96x5120x5120 | 3 | 160 | 160 | 93 | 106.2 | 16.5 | KEEP |
| 96x5120x2560 | 3 | 160 | 80 | 91 | 54.1 | 8.3 | KEEP |
| 96x2048x5120 | 3 | 64 | 160 | 90 | 43.6 | 6.6 | KEEP |
| 32x256x512 | 1 | 8 | 16 | 27 | 0.6 | 0.0 | KEEP |
| 32x256x5120 | 1 | 8 | 160 | 28 | 5.8 | 0.3 | KEEP |
| 32x5120x1280 | 1 | 160 | 40 | 31 | 26.4 | 1.4 | KEEP |
| 32x1280x30720 | 1 | 40 | 960 | 31 | 157.6 | 8.3 | KEEP |
| 512x4096x5120 | 16 | 128 | 160 | 418 | 100.4 | 70.6 | KEEP |
| 512x5120x5120 | 16 | 160 | 160 | 427 | 122.9 | 88.3 | KEEP |
| 512x5120x2560 | 16 | 160 | 80 | 394 | 66.6 | 44.1 | KEEP |
| 32x128x30720 | 1 | 4 | 960 | 26 | 19.2 | 0.8 | kept by roofline, then REJECTED BY THE PICKER: Kt=4 < 8 |
| 2656x5120x3840 | 83 | 160 | 120 | 1202 | 169.8 | 343.4 | drop: compute-bound |
| 2656x5120x3456 | 83 | 160 | 108 | 1161 | 158.1 | 309.1 | drop: compute-bound |
| 2656x5120x1280 | 83 | 160 | 40 | 739 | 92.0 | 114.5 | drop: compute-bound, M>=N |
| 10560x5120x3840 | 330 | 160 | 120 | 1817 | 446.4 | 1365.3 | drop: compute-bound, M>=N |
| 10560x5120x3456 | 330 | 160 | 108 | 1726 | 422.9 | 1228.8 | drop: compute-bound, M>=N |
| 10560x3456x5120 | 330 | 108 | 160 | 1726 | 422.9 | 1228.8 | drop: compute-bound, M>=N |
| 10560x5120x1280 | 330 | 160 | 40 | 933 | 289.6 | 455.1 | drop: compute-bound, M>=N |
| 2656x5120x64 | 83 | 160 | 2 | 62 | 55.1 | 5.7 | drop: Nt=2 too narrow, M>=N |
| 10560x5120x64 | 330 | 160 | 2 | 63 | 215.1 | 22.8 | drop: Nt=2 too narrow, M>=N |
| 32x512x128 | 1 | 16 | 4 | 24 | 0.3 | 0.0 | drop: Nt=4 too narrow |
| 10560x224x1280 | 330 | 7 | 40 | 187 | 63.2 | 19.9 | drop: M>=N (also Kt=7 < 8) |

Column definitions: see "Method" above.
