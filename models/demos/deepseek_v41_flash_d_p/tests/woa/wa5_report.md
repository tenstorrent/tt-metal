# WA.5: wo_a bank-local reads, the report (CPU only, 2026-11-27)

Written by tests/woa/test_woa_report.py from tests/woa/wa5_report.md. Sources: results/WA.2_probe.md, WA.3_offon.md (OV-79 table: the model's 11 x 10 grid), WA.4_ladder.md (re-rendered
from the WA.4 A/B, OV-80), PB.MM256.json / PB.MM256_ladder.md, P.81 (BREADCRUMBS.0004, DS41-306). Device us a call =
real-time profiler, slowest chip, 20 calls. A chunk = layers 0-20 (DECODER_SKIP=decode), 20 calls of each projection.
The estimate in part 3: `python3 -m models.demos.deepseek_v41_flash_d_p.tests.woa.woa_estimate` (gated by
tests/woa/test_woa_report.py).

## 1. Result

| wo_a, us a call | 256 rows (8192 chunk) | 160 rows (5120 chunk) |
|---|---|---|
| today (interleaved, 88 cores) | 151.1 | 131.7 |
| bank-local, as the model runs it (band 10, 80 cores on the 11 x 10 grid) | **130.3** (-20.8, -14 %) | **114.2** (-17.5, -13 %) |
| bank-local on the full 12 x 10 grid (WA.2; not legal in the model, DS41-382) | 124.9 | 104.1 |
| DRAM floor: weights 35.65 MB at 512 / 410 GB/s | 69.6 / 87.0 | 69.6 / 87.0 |
| all bytes (w + x + y) at 410 GB/s | 114.2 | 104.0 |
| busiest core's compute at the HiFi2 peak (band 10: 4 column tiles) | 97.1 | 60.7 |

- **In the model, 8192 chunk (the only real off vs on: test_woa_offon, one load, switch per call):** wo_a 3.058 ->
  2.656 ms a chunk (152.9 -> 132.8 us a call); device ms a chunk 85.41 -> 84.82 (-0.60, single 8192), 94.72 -> 94.20
  (-0.52, packed 4 x 2k). Bit-identical off == on over 40 layers; host a call unchanged within noise; load +1.97 s.
- **5120 chunk (160 rows):** no real in-model off vs on was measured (test_woa_offon's 5120 case pads to 256 rows in
  the 8192 load; the 5120 ladder's off side ran with the switch on, below). From the op probe: -17.5 us x 20 =
  **about -0.35 ms a 5120 chunk** (on-side device 60.35-76.44 ms over the ladder).
- **Ladder deltas: none usable.** The WA.4 A/B's off side ran its ladder, 5120 and no-rebuild units with
  DS41_WOA_BANK=on (each gate command names the switch, and a command's prefix wins over the side's env; DS41-397).
  So the re-rendered tables are on vs on: single 8192 device -0.58..+0.26 ms, attention -0.22..+0.23; 4 x 2k device
  +0.06..+1.01; 4 x 1792s -0.94..+1.23; 5120 -0.26..+0.39. That is session noise; a real off vs on would show the
  attention section ~0.40 ms lower at every rung. The ladders neither confirm nor refute the -0.40 ms; the accept rests
  on test_woa_offon.

## 2. What closed the gap, and what did not

Before the build (WA.2 step 0, today's op with stubs): no weight reads 151.6 -> 107.2 us (256 rows) / 131.4 -> 70.9
(160); no compute 149.4 / 129.7 (about today); no x 117.8 / 118.9. Today's wo_a was reader-bound, weights first.

| lever (each measured, every output bit-identical) | 256 rows us | 160 rows us |
|---|---|---|
| bank-local, K-major data, ONE contiguous read a K block, on today's cores (rect 11): fewer packets, no cross-bank conflicts | -3.4 (151.1 -> 147.7) | -10.5 (131.7 -> 121.2) |
| cores moved next to their bank, 10 a bank in a 5 x 2 cluster (short NoC distance, clusters run independently) | **-17.4** (147.7 -> 130.3) | -7.0 (121.2 -> 114.2) |
| reads in flight (w_inflight 1 vs 2, at 11 wide) | within 0.7 | within 0.7 |
| fewer cores a bank (band 10 = 80 cores vs band 12 = 96, 12 wide) | -4.7 (128.8 -> 124.1) | -5.4 (110.1 -> 104.7) |
| x sender at the cluster's first core, not inside it (NoC-0 multicast wraps the chip; DS41-393) | 342 -> 155 (pre-tuning) | |
| x read-ahead that never blocks (no-weight stub) | 152 -> 106 | |
| host: one kernel group, 24 -> 4 core ranges (DS41-394) | host +23-45 -> +3-4 us | |

- **What paid: where the cores sit, not the layout alone.** Bank-local data on today's cores kept 3 of the 21 us at
  256 rows (half at 160 rows); putting each bank's cores next to it gave the rest. Bank conflicts were not measured on
  their own (no per-bank counter); the rect 11 row is the nearest proxy. Keeping more reads in flight did not matter
  once a block is one contiguous read.
- **What did not close.** At 256 rows the bank path is compute-bound: its no-x, no-weight and no-compute stubs all ran
  116-121 us. Its busiest core has 4 column tiles (4096 tile products, 97.1 us at the HiFi2 peak) and runs at ~75 % of
  peak. The 110-core matmul cap (11 wide) costs 5.4 us (256) / 10.1 us (160) against the full grid. x still comes
  interleaved (the previous op's output). The remaining gap at 256 rows: 130.3 vs all bytes at 410 GB/s 114.2 / at
  512 GB/s 91.4. At 160 rows 114.2 vs 104.0: within 10 %, the 11-wide cap is most of it.

## 3. Would the same layout help wq_b and wo_b? ESTIMATE ONLY (not built: owner decision)

As the model holds them (ATTN_SP32_PROJ: replicated on every chip): wq_b x [rows, 1280] bfp8 @ w [1280, 32768] bfp8 ->
bf16; wo_b x [rows, 8192] bfp8 @ w [8192, 5120] bfp8 -> bf16; HiFi2, fp32 DEST. Today at 256 rows: wq_b 238.2 us
(minimal_matmul M2 K8 N16, PB.MM256), wo_b 157.2 us (1D in0 pm8 pn2 bw16); at 160 rows wq_b 182.0 (P.81 ROOFLINE),
wo_b 142.3. Gap to the weights at 410 GB/s at 256 rows: wq_b 2.19x, wo_b 1.45x (all bytes: 1.58x / 1.30x).

What the stock DRAM-sharded matmul gave them (P.81, 160 rows, 8-13 cores a bank, DS41-306): wq_b's matmul
153-174 us, but its output is L1-only and the copy to DRAM adds 37 us: no gain; wo_b 280-650 us: worse. Not tried at
256 rows.

Method (woa_estimate.py), calibrated on the one bank path measured (wo_a): A = today minus the share of
(today - realistic roofline) that the bank path closed for wo_a (0.70 at 256 rows, 0.63 at 160); B = all bytes at the
byte rate wo_a's bank path reached (359 / 373 GB/s) or the busiest core at wo_a's measured time a tile product,
whichever is slower; 13 cores a bank (104 <= 110). Best = mean of A and B; range = 0 to (today - realistic).

| proj, rows a chip | w / x / y MB | w at 512 / 410 GB/s us | all bytes 410 us | compute peak us (cores) | realistic us | today us | est A / B us | best est us | gain ms a chunk best (range) |
|---|---|---|---|---|---|---|---|---|---|
| wq_b, 256 | 44.56 / 0.35 / 16.78 | 87.0 / 108.8 | 150.6 | 75.9 (104) | 150.6 | 238.2 | 176.9 / 171.8 | 174.4 | 1.28 (0-1.75) |
| wq_b, 160 | 44.56 / 0.22 / 10.49 | 87.0 / 108.8 | 134.9 | 47.4 (104) | 134.9 | 182.0 | 152.2 / 148.1 | 150.2 | 0.64 (0-0.94) |
| wo_b, 256 | 44.56 / 2.23 / 2.62 | 87.0 / 108.8 | 120.6 | 97.1 (104) | 121.4 | 157.2 | 132.1 / 137.6 | 134.9 | 0.45 (0-0.72) |
| wo_b, 160 | 44.56 / 1.39 / 1.64 | 87.0 / 108.8 | 116.2 | 60.7 (104) | 116.2 | 142.3 | 125.8 / 127.5 | 126.7 | 0.31 (0-0.52) |

**Best estimate, device ms saved a 8192 chunk (layers 0-20): wq_b 1.28 ms (4.76 -> ~3.5), wo_b 0.45 ms
(3.14 -> ~2.7). A 5120 chunk: 0.64 / 0.31 ms.**

Uncertainty (large; the honest range is the column's 0 to the high end):
- No stub data for wq_b or wo_b. wo_a was proven reader-bound before the build (no-compute stub = today); nobody has
  shown that for these two. Their kernels differ (minimal_matmul; stock 1D with double-buffered reads), so wo_a's
  "one block in flight, then drain" may not apply. The first step of any build is the same step 0: stubs.
- **wq_b is the better candidate.** Its x is tiny (0.35 MB), so each bank cluster can read its own copy (+2.5 MB,
  +7 us) and stay independent as wo_a's did. It has the largest gap (88 us a call above its realistic roofline) and
  compute is far from the limit (76 us at peak vs 151 us of bytes). Against it: 16.8 MB of bf16 output (27 % of its
  bytes) goes to interleaved DRAM, which the layout does not touch, and K is only 40 tiles (few blocks, so fill and
  drain weigh more). The P.81 DRAM-sharded matmul at 160 rows (153-174 us without its output copy) agrees with
  estimate A / B (152 / 148 us). A fork writing DRAM directly would not pay that copy.
- **wo_b is weaker and riskier.** Compute is near the bytes (97 us at peak vs 121 us of bytes: like wo_a at 256
  rows, compute-bound once reads are fixed). Every bank cluster needs ALL of x (2.23 MB), unlike wo_a where a cluster
  needs only its own group. If each cluster reads its own copy, estimate B becomes 181 us, slower than today's 157. So
  it needs one multicast across the whole grid, which ties the clusters together again, and independent clusters were
  most of wo_a's gain. Realistic: 0 to 0.45 ms.
- The calibration rests on one op at two row counts. The 11 x 10 cap and the 110-core rule hold the same way.
- Build cost: grouped_matmul's bank path is built for G groups with x per group. These two are G = 1 with N split
  over the 8 banks, and they need an x path shared by every cluster, which is new reader work. Added together the
  best estimate is ~1.7 ms a 8192 chunk (~2 % of 84.8 ms device). wq_b alone has ~3x wo_a's realised 0.40 ms.

## 4. Known issue

- **DS41-396** (bringup/known_issues.md): the wo_a lesson. Most of the gain came from where the cores sit; the layout
  alone gave little; at 256 rows the call ends compute-bound on 4-column cores. The wq_b / wo_b estimate is there too.
- **DS41-397**: an A/B side's env loses to a gate command's own switch prefix, so WA.4's off ladders ran with the
  change on. Never name the A/B switch in a gate command; check the side's batch plan
  (tests/woa/woa_ladder_report.switch_as_run).
