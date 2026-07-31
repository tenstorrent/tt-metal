# Regime-A matmul: p150 vs Blackhole Galaxy

Same commit, same 78 shapes, `config=None` (production picker) throughout.
The two boards have different memory systems, so **wall time is not comparable directly** -- utilisation against each board's own DRAM peak is the metric that controls for that.

| | p150 | Galaxy |
|---|---|---|
| GDDR6 pin rate | 16 Gbps | **14 Gbps** |
| DRAM peak (32-bit x 8ch) | 512 GB/s | **448 GB/s** |
| Tensix cols enabled | 11 of 14 | 13 of 14 (varies per chip) |

`util` = effective DRAM bandwidth as a % of that board's peak. `Dpp` = Galaxy util minus p150 util, in percentage points; **positive = Galaxy uses its narrower bus more efficiently**.
Picker output (`cfg` / `reduction` / `placement`) was byte-identical on both boards for all 78 shapes, so it is listed once.

| shape | cfg | reduction | placement | p150 us | glx us | p150 GB/s | glx GB/s | p150 util | glx util | Dpp |
|---|---|---|---|---|---|---|---|---|---|---|
| 32x256x512 | 1,2,1,1,1 | chain | bank-local | 4.29 | 4.41 | 76.3 | 74.3 | 14.9% | 16.6% | +1.68 |
| 256x2048x512 | 4,1,3,2,2 | reduce-scatter | mesh | 13.70 | 14.14 | 248.8 | 241.1 | 48.6% | 53.8% | +5.22 |
| 128x2048x512 | 4,1,2,2,2 | reduce-scatter | in1-near | 10.29 | 10.53 | 267.6 | 261.4 | 52.3% | 58.3% | +6.08 |
| 512x6144x768 | 12,1,1,2,1 | chain | mesh | 61.57 | 63.53 | 268.2 | 259.9 | 52.4% | 58.0% | +5.63 |
| 512x15360x768 | 6,1,2,2,1 | chain | mesh | 144.33 | 145.80 | 277.9 | 275.1 | 54.3% | 61.4% | +7.13 |
| 256x2048x1536 | 4,1,3,2,3 | reduce-scatter | in1-near | 27.73 | 28.88 | 293.1 | 281.4 | 57.2% | 62.8% | +5.57 |
| 256x2048x1024 | 4,1,2,2,4 | reduce-scatter | in1-near | 19.47 | 19.26 | 296.2 | 299.4 | 57.9% | 66.8% | +8.98 |
| 64x2048x512 | 4,2,1,2,1 | chain | bank-local | 8.85 | 9.15 | 303.5 | 293.6 | 59.3% | 65.5% | +6.26 |
| 256x2048x2048 | 4,1,3,2,4 | reduce-scatter | in1-near | 34.45 | 35.75 | 304.3 | 293.3 | 59.4% | 65.5% | +6.04 |
| 512x6144x1536 | 12,1,1,2,1 | chain | mesh | 87.17 | 88.13 | 306.7 | 303.4 | 59.9% | 67.7% | +7.82 |
| 512x5120x2560 | 10,1,1,2,1 | chain | mesh | 109.79 | 110.56 | 310.4 | 308.2 | 60.6% | 68.8% | +8.17 |
| 512x2304x6144 | 3,4,1,1,1 | chain | mesh | 141.59 | 142.88 | 311.0 | 308.2 | 60.7% | 68.8% | +8.05 |
| 512x3072x6144 | 6,2,1,2,1 | chain | mesh | 158.68 | 164.37 | 317.2 | 306.2 | 62.0% | 68.3% | +6.40 |
| 32x2048x512 | 4,2,1,2,1 | chain | bank-local | 7.34 | 7.76 | 325.7 | 308.1 | 63.6% | 68.8% | +5.16 |
| 512x5120x5120 | 10,1,1,2,1 | chain | mesh | 187.63 | 192.85 | 335.3 | 326.2 | 65.5% | 72.8% | +7.32 |
| 512x6144x2304 | 12,1,1,2,1 | chain | mesh | 109.98 | 112.51 | 336.1 | 328.5 | 65.6% | 73.3% | +7.68 |
| 128x2048x1024 | 4,1,2,2,4 | reduce-scatter | in1-near | 14.50 | 15.17 | 343.5 | 328.4 | 67.1% | 73.3% | +6.21 |
| 512x4096x5120 | 4,3,1,2,1 | chain | mesh | 173.44 | 181.52 | 344.6 | 329.3 | 67.3% | 73.5% | +6.20 |
| 256x6144x768 | 12,1,1,2,1 | chain | mesh | 36.11 | 38.19 | 359.3 | 339.8 | 70.2% | 75.8% | +5.67 |
| 256x15360x768 | 6,1,2,2,3 | reduce-scatter | mesh | 87.02 | 88.84 | 366.0 | 358.5 | 71.5% | 80.0% | +8.54 |
| 128x6144x768 | 12,1,1,2,1 | chain | bank-local | 30.48 | 33.32 | 367.7 | 336.3 | 71.8% | 75.1% | +3.25 |
| 256x6144x1536 | 6,1,2,4,2 | reduce-scatter | in1-near | 61.00 | 66.41 | 373.9 | 343.4 | 73.0% | 76.7% | +3.62 |
| 512x6144x4608 | 12,1,1,2,1 | chain | mesh | 180.31 | 186.36 | 375.1 | 362.9 | 73.3% | 81.0% | +7.74 |
| 128x2048x1536 | 4,3,1,2,1 | chain | bank-local | 21.95 | 24.32 | 376.2 | 339.5 | 73.5% | 75.8% | +2.30 |
| 64x2048x1024 | 4,2,1,2,2 | reduce-scatter | bank-local | 12.20 | 13.39 | 397.5 | 362.3 | 77.6% | 80.9% | +3.23 |
| 64x6144x768 | 12,1,1,2,1 | chain | bank-local | 25.63 | 28.13 | 402.7 | 366.9 | 78.7% | 81.9% | +3.24 |
| 256x6144x2304 | 12,1,1,2,1 | chain | mesh | 79.98 | 81.61 | 408.1 | 399.9 | 79.7% | 89.3% | +9.56 |
| 256x15360x1536 | 6,1,2,2,6 | reduce-scatter | in1-near | 136.04 | 144.83 | 410.4 | 385.5 | 80.2% | 86.0% | +5.89 |
| 32x256x6144 | 1,3,1,1,8 | chain | bank-local | 8.65 | 9.39 | 414.7 | 382.3 | 81.0% | 85.3% | +4.34 |
| 256x6080x4640 | 12,1,1,2,1 | chain | mesh | 148.48 | 153.38 | 417.0 | 403.6 | 81.4% | 90.1% | +8.64 |
| 64x2048x1536 | 4,3,1,2,1 | chain | bank-local | 17.44 | 19.42 | 417.0 | 374.6 | 81.4% | 83.6% | +2.17 |
| 32x256x5120 | 1,5,1,1,4 | chain | bank-local | 7.23 | 7.88 | 419.1 | 384.5 | 81.9% | 85.8% | +3.97 |
| 256x2304x6144 | 3,4,1,1,3 | chain | bank-local | 85.63 | 96.77 | 422.5 | 373.8 | 82.5% | 83.4% | +0.92 |
| 32x6144x768 | 6,1,1,2,3 | chain | bank-local | 23.20 | 25.89 | 425.8 | 381.6 | 83.2% | 85.2% | +2.01 |
| 128x2048x2048 | 4,3,1,2,3 | reduce-scatter | bank-local | 24.49 | 27.61 | 428.2 | 379.7 | 83.6% | 84.8% | +1.12 |
| 128x6144x1536 | 12,1,1,2,2 | chain | bank-local | 48.42 | 55.23 | 430.4 | 377.3 | 84.1% | 84.2% | +0.16 |
| 128x15360x768 | 6,1,1,2,3 | reduce-scatter | bank-local | 63.89 | 68.61 | 433.9 | 404.1 | 84.7% | 90.2% | +5.45 |
| 32x2048x1024 | 2,4,1,4,1 | chain | bank-local | 10.94 | 12.15 | 437.3 | 393.7 | 85.4% | 87.9% | +2.47 |
| 32x5120x1280 | 10,1,1,2,1 | chain | bank-local | 30.86 | 34.29 | 438.0 | 394.2 | 85.5% | 88.0% | +2.44 |
| 256x6144x6144 | 6,1,2,4,2 | reduce-scatter | in1-near | 186.68 | 200.02 | 438.1 | 408.9 | 85.6% | 91.3% | +5.71 |
| 256x6144x4608 | 6,1,2,4,2 | reduce-scatter | in1-near | 141.57 | 151.98 | 438.9 | 408.8 | 85.7% | 91.2% | +5.53 |
| 64x2048x2048 | 2,3,1,2,3 | chain | bank-local | 21.49 | 23.81 | 439.1 | 396.4 | 85.8% | 88.5% | +2.72 |
| 32x2048x1536 | 2,2,1,4,3 | chain | bank-local | 15.04 | 16.79 | 442.4 | 396.2 | 86.4% | 88.4% | +2.03 |
| 128x6080x4640 | 12,1,1,2,2 | chain | bank-local | 133.26 | 148.28 | 444.0 | 399.0 | 86.7% | 89.1% | +2.34 |
| 64x15360x768 | 10,1,1,2,3 | chain | bank-local | 57.69 | 64.57 | 444.7 | 397.3 | 86.9% | 88.7% | +1.83 |
| 128x6144x2304 | 12,1,1,2,1 | chain | bank-local | 67.66 | 75.55 | 450.4 | 403.4 | 88.0% | 90.0% | +2.08 |
| 128x2304x6144 | 3,2,1,1,6 | chain | bank-local | 68.54 | 75.15 | 453.2 | 413.4 | 88.5% | 92.3% | +3.76 |
| 96x5120x2560 | 10,1,1,2,2 | chain | bank-local | 61.05 | 68.47 | 453.6 | 404.4 | 88.6% | 90.3% | +1.67 |
| 32x2048x2048 | 2,2,1,4,4 | chain | bank-local | 19.34 | 21.75 | 454.0 | 403.7 | 88.7% | 90.1% | +1.44 |
| 64x6080x4640 | 12,1,1,2,2 | chain | bank-local | 126.93 | 142.80 | 455.3 | 404.7 | 88.9% | 90.3% | +1.41 |
| 64x6144x2304 | 12,1,1,2,2 | chain | bank-local | 64.04 | 71.48 | 459.0 | 411.2 | 89.6% | 91.8% | +2.14 |
| 128x15360x1536 | 12,1,1,1,3 | chain | bank-local | 111.83 | 124.14 | 460.6 | 415.0 | 90.0% | 92.6% | +2.67 |
| 64x6144x1536 | 3,1,1,8,2 | chain | bank-local | 42.82 | 48.27 | 463.8 | 411.4 | 90.6% | 91.8% | +1.24 |
| 96x2048x5120 | 4,3,1,2,2 | reduce-scatter | bank-local | 49.78 | 56.01 | 464.7 | 413.1 | 90.8% | 92.2% | +1.45 |
| 32x6080x4640 | 12,1,1,2,1 | chain | bank-local | 122.75 | 138.56 | 465.2 | 412.2 | 90.9% | 92.0% | +1.15 |
| 128x6144x6144 | 12,1,1,2,2 | chain | bank-local | 167.94 | 186.56 | 468.3 | 421.5 | 91.5% | 94.1% | +2.62 |
| 32x6144x1536 | 6,1,1,4,2 | chain | bank-local | 41.24 | 46.46 | 469.6 | 416.8 | 91.7% | 93.0% | +1.32 |
| 128x6144x4608 | 12,1,1,2,1 | chain | bank-local | 125.52 | 140.31 | 473.1 | 423.2 | 92.4% | 94.5% | +2.06 |
| 96x5120x5120 | 10,1,1,2,2 | chain | bank-local | 114.12 | 127.73 | 476.6 | 425.9 | 93.1% | 95.1% | +1.98 |
| 96x8192x5120 | 4,3,1,2,2 | reduce-scatter | bank-local | 187.73 | 211.84 | 477.2 | 422.9 | 93.2% | 94.4% | +1.19 |
| 32x15360x768 | 6,1,1,2,3 | chain | bank-local | 51.49 | 58.31 | 478.2 | 422.3 | 93.4% | 94.3% | +0.86 |
| 32x1280x30720 | 5,2,1,1,2 | chain | bank-local | 167.28 | 186.11 | 482.9 | 434.0 | 94.3% | 96.9% | +2.56 |
| 64x6144x6144 | 12,1,1,2,2 | chain | bank-local | 159.06 | 179.62 | 484.5 | 429.1 | 94.6% | 95.8% | +1.15 |
| 48x6144x4608 | 6,1,1,4,2 | chain | bank-local | 118.68 | 134.21 | 485.8 | 429.6 | 94.9% | 95.9% | +1.01 |
| 64x15360x1536 | 6,1,1,2,3 | chain | bank-local | 101.33 | 114.50 | 487.0 | 431.0 | 95.1% | 96.2% | +1.09 |
| 64x2304x6144 | 3,4,1,1,3 | chain | bank-local | 62.06 | 70.06 | 487.8 | 432.2 | 95.3% | 96.5% | +1.20 |
| 32x6100x4608 | 12,1,1,2,1 | chain | bank-local | 116.41 | 132.05 | 488.8 | 430.9 | 95.5% | 96.2% | +0.71 |
| 32x6144x4600 | 12,1,1,2,1 | chain | bank-local | 116.93 | 131.90 | 489.3 | 433.8 | 95.6% | 96.8% | +1.26 |
| 32x2304x6144 | 3,2,1,1,6 | chain | bank-local | 59.24 | 66.91 | 489.5 | 433.4 | 95.6% | 96.7% | +1.14 |
| 32x6144x4608 | 12,1,1,2,1 | chain | bank-local | 117.05 | 132.28 | 489.6 | 433.3 | 95.6% | 96.7% | +1.09 |
| 64x6144x4608 | 6,1,1,4,2 | chain | bank-local | 118.46 | 134.03 | 489.6 | 432.7 | 95.6% | 96.6% | +0.96 |
| 32x6144x2304 | 3,1,1,4,5 | chain | bank-local | 58.74 | 66.29 | 491.1 | 435.2 | 95.9% | 97.1% | +1.22 |
| 32x6144x3072 | 3,1,1,4,6 | chain | bank-local | 78.05 | 87.80 | 491.2 | 436.6 | 95.9% | 97.5% | +1.52 |
| 64x4608x6144 | 3,2,1,2,3 | chain | bank-local | 118.91 | 134.11 | 492.7 | 436.9 | 96.2% | 97.5% | +1.29 |
| 32x15360x1536 | 6,1,1,2,3 | chain | bank-local | 97.76 | 111.01 | 493.7 | 434.8 | 96.4% | 97.1% | +0.63 |
| 64x6144x9216 | 6,1,1,4,2 | chain | bank-local | 231.57 | 262.91 | 497.5 | 438.2 | 97.2% | 97.8% | +0.64 |
| 32x6144x6144 | 6,1,1,4,2 | chain | bank-local | 152.84 | 174.16 | 499.1 | 438.0 | 97.5% | 97.8% | +0.29 |
| 32x6144x9216 | 3,1,1,4,6 | chain | bank-local | 227.96 | 259.29 | 501.1 | 440.6 | 97.9% | 98.3% | +0.48 |

## Summary

| | p150 (512 GB/s) | Galaxy (448 GB/s) |
|---|---:|---:|
| median util | 85.6% | **88.9%** |
| max util | 97.9% | **98.3%** |
| min util | 14.9% | 16.6% |
| median eff BW | 438.1 GB/s | 398.1 GB/s |

- **utilisation delta: median +2.39 pp, range +0.16 to +9.56 pp**
- **78/78 shapes improved, 0 regressed**
- picker config / reduction / placement changes: **0**
- correctness: 78/78 PCC >= 0.999 on both boards
- absolute bandwidth retention (glx/p150): median 0.900 vs DRAM clock ratio 0.875 -- the op gives back less than the clock drop, i.e. part of the corpus is not purely memory-bound

Gains concentrate in shapes that were far from the roofline on p150:

| p150 util band | median Dpp | n |
|---|---:|---:|
| 0-60% | +6.06 pp | 10 |
| 60-80% | +6.21 pp | 17 |
| 80-90% | +2.34 pp | 25 |
| 90-95% | +1.28 pp | 12 |
| 95-100% | +1.09 pp | 14 |

Roofline-pinned shapes hold (+1.1 pp median at 95-100%); shapes with a latency or compute component gain most, because that component does not scale with the DRAM clock.

**Not measured (1):** `32x128x30720` -- TT_FATAL on both boards (Kt=4 < 8, rejected by the picker).

---

*The two sections below are carried over from `HEAD_SWEEP_UNIFIED.md`, which this file
supersedes and replaces. They describe the original p150 campaign; the same method and the
same byte accounting were used for the Galaxy run, with two board-specific caveats:*

- *`%pk` below is quoted against **512 GB/s**, which is the p150 peak (16 Gbps x 32-bit x 8ch). The Galaxy peak is **448 GB/s** (14 Gbps). The `util` columns in the table above already use the correct peak for each board.*
- *`FPU%` below is computed against a **110-core** grid, which describes the p150. This Galaxy has 13 of 14 Tensix columns enabled (differing column per chip), so that denominator does not apply here -- FPU% is deliberately omitted from the table above rather than reported wrongly.*

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
