# moe_fused_swiglu (Kimi K2.6 routed expert) precision fix + optimization log

Branch: `mstaletovic/routed_expert_fixes` (forked from PR #54677 branch `pmilojevic/52183-REdoubleOP`).
Box: Blackhole (bh-qb, 4 chips, IOMMU on), compute grid 11x10 per chip, RT program profiler active.
All device numbers are on-device program durations from the realtime profiler (median of 3 dispatches, one
profiler window) unless stated otherwise. Run-to-run noise is about ±2%.

Shape under study (production): Kimi K2.6, emb 7168, hidden 2048, weights **bfloat4_b**, x bf16 ROW_MAJOR,
grid 11x8 (88 cores), region capacity 5120 tokens (`input_m_tiles=160`), one expert, active M = 64 ... 5120.
The compute config is the model's: LoFi, approx off, fp32_dest_acc off, packer_l1_acc off.

Tools (all in `routed_expert_work/`):
* `test_bench.py` -- perf (RT profiler) + error vs fp32 torch reference (PCC and relative RMS), optionally the
  old composite `unified_routed_expert_moe` on identical inputs. Env-driven (`BENCH_M`, `BENCH_WDTYPE`,
  `BENCH_OLD`, `BENCH_OUT_BF16`, `BENCH_IDENTITY`, `BENCH_SAVE`, ...). Results append to `results/<tag>.jsonl`.
* `zones.py` -- per-stage report from `MOE_FUSED_SWIGLU_STAGE_PROFILE=1 TT_METAL_DEVICE_PROFILER=1` captures
  (median/min/max per zone across the 88 cores + start offset from kernel start). Captures in `zones/`.
* `emulate_precision.py`, `emulate_precision2.py` -- host-side torch emulation of the quantization points.
* `test_matmul_kblock.py` -- standalone ttnn.matmul: error vs K tiles accumulated in bf16 DEST.

Host knobs added to the op for A/B without rebuilds (all default to the shipped constants):
`MOE_FUSED_SWIGLU_ACC_BF16`, `_DEPTH_X`, `_DEPTH_H`, `_HACK_AHEAD`, `_WD_MROW`, `_GU_CHUNKS`,
`MOE_FUSED_SWIGLU_DEFINES="A=1,B"` (extra kernel defines), `MOE_FUSED_SWIGLU_LOG_L1=1` (prints the CB layout).

## 0. Problem statement

CI job `(Kimi-K2.6-1T) prefill runner accuracy code_debug 55k@5k` failed with KV-cache PCC 0.9210 < 0.93
(mean over 61 layers, `test_producer_runner_pcc[single_user_full_depth]`). The same test passes with the old
`unified_routed_expert_moe` composite. Hypothesis (user): gate/up partials are quantized to bfp8 before the
8-way K reduce-scatter; the old op keeps K-loop partials in bf16 (`partials_gu_df = Float16_b`).

## 1. Baseline (as inherited, commit 753718a7412)

### 1.1 Intermediate formats
| Stage | CB | format | old op |
|---|---|---|---|
| x resident | CB_X_TILES | bfp8 (tilized from bf16 sticks) | bfp8 |
| gate/up matmul out (K/8 partial per core) | CB_GATE_ACC / CB_UP_ACC | **bfp8** | bf16 |
| reduce-scatter landing (8 partials) | CB_GATHER_GATE / CB_GATHER_UP | **bfp8** | n/a (single core K loop) |
| fold in DEST -> slice, silu | CB_SLICE_*, CB_GATE_SILU | bf16 | bf16 |
| h = silu(g)*u | CB_H_SLICE -> CB_H_LOCAL -> CB_H | bfp8 | bfp8 |
| output | CB_OUT_TILES | bfp8 (precise pack) | bfp8 |

### 1.2 L1 (11x8, kimi, bfp4 weights, rm x, 5120-token region): 1,389,120 of 1,461,248 B (72 KB free)
| CB | bytes | note |
|---|---:|---|
| CB_X_TILES (depth_x=2 x 8 rows x 28 tiles, bfp8) | 487,424 | **35% of L1**; the second slot is the cross-M-block prefetch target |
| CB_H (depth_h=3 x 64 tiles bfp8) | 208,896 | H row broadcast pipeline |
| CB_W_DOWN (resident, 11 x 6 x 3 bfp4) | 114,048 | |
| CB_W_GATE, CB_W_UP (28 x 6 bfp4 each) | 96,768 x2 | resident |
| CB_H_LOCAL (64 bfp8) | 69,632 | assembled H row (diagonal cores) |
| CB_X_IN (32 sticks x 1792 B) | 57,344 | |
| CB_GATE_ACC, CB_UP_ACC (48 bfp8 each) | 52,224 x2 | |
| CB_GATHER_GATE(+alias H_SLICE, OUT_TILES), CB_GATHER_UP | 52,224 x2 | |
| CB_GATE_SILU+CB_OUT_INTERM (alias), SLICE_GATE, SLICE_UP | 24,576 + 12,288 x2 | |

### 1.3 Baseline perf and error (bfp4 weights), fused vs old composite
| M | fused ns | old ns | fused rel-RMS | old rel-RMS |
|---:|---:|---:|---:|---:|
| 64 | 84,581 | 203,663 | 0.2078 | 0.2031 |
| 128 | 93,536 | 198,038 | 0.2093 | 0.2046 |
| 256 | 115,636 | 210,714 | 0.2171 | 0.2043 |
| 512 | 195,108 | 264,737 | 0.2169 | 0.2041 |
| 1024 | 341,589 | 376,619 | 0.2169 | 0.2040 |
| 5120 | 1,522,604 | 1,648,480 | 0.2167 | 0.2038 |

PCC vs the fp32 reference is 0.980 for both ops (dominated by bfp4 weight quantization), so relative RMS is
the sensitive metric. The fused op carries ~6% more error than the composite at M>=256, ~2% at M<=128.

Diagnostic only (not the production config): with bfp8 weights the fused op is MORE precise than the old op
(rel-RMS 0.0421 vs 0.0468 at M=256), so the old op is not uniformly "bf16 everywhere better".

### 1.4 Stage profile at M=256 (median core, µs from kernel start; `zones/baseline_m256.csv`)
Compute (TRISC): tilize 2-8 | gate/up 7-66 (up 37.5 incl. weight waits, gate 17.9 ~= FPU bound) | reduce 67-83
(15.8, mostly waiting for the column's slowest core + scatter landing) | swiglu 85-90 (5.5) | down 90-115 (24.5).
Reader (NoC0): x stage+mcast 2-26 | W_gate issue/wait to 46 | W_down issue 50-54 | reduce 54-82 (up wait 9.9,
invite wait 2.3, payload 1.4, data wait 6.5) | phase 2 82-112 (H-row waits 29 over 8 rounds).
Writer (NoC1): W_up 1.7-44 (42.7) | W_down tail 44-49 | scatter 51-77 (invite wait 19.9 = waiting for peers'
gate/up) | h-slice 76-91 (waits for compute swiglu) | output issue 91-115.
Core skew: compute_gateup min 41 / med 59 / max 75 µs -> the column reduce waits for the slowest core.

Reduce-scatter + SiLU share of the op (compute_reduce + compute_swiglu zones):
M=64: 7.8/84.6 = 9% | M=128: 11.5/93.5 = 12% | M=256: 21.3/115.6 = 18% | M=512: 38/195 = 19% | M=1024: 68/342 = 20%.

Roofline for M=256, one expert: weights 24.8 MB (bfp4) -> 48 µs at 512 GB/s; FPU 3 x 256x7168x2048 LoFi
= 3909 tile-matmuls/core x 16 cycles = 46 µs; x read 3.7 MB = 7 µs. The gate/up phase alone (17 MB of weights,
33 µs DRAM floor; 32 µs FPU floor) takes 59 µs; weight streams reach ~75% of DRAM peak.

## 2. Precision attribution

### 2.1 Host emulation (`emulate_precision2.py`, bfp4 weights, M=256)
Rounding the 8 K-shard partials to bfp8 before the fold costs rel-RMS +0.0002 (0.19888 -> 0.19909); LoFi
truncation, per-K-step bf16 DEST rounding, bfp8 truncating vs precise pack all move the number by <0.001. The
emulation floor (0.198) is well below both devices (0.204 old / 0.217 fused), so the device has error sources
the emulation does not model.

### 2.2 On-device attribution (M=256, bfp4 weights, rel-RMS)
| variant | rel-RMS | note |
|---|---:|---|
| baseline (bfp8 partials, mrow down path) | 0.2171 | |
| bf16 partials + landing (`ACC_BF16=1`, fell back to no wd residency/no mrow) | 0.2072 | |
| bfp8 partials, `WD_MROW=0` (11 K-block down path with bf16 L1 acc) | 0.2093 | mrow down costs +0.008 |
| bfp8 partials, mrow, **bf16 output** | 0.2082 | identical output bits to WD_MROW=0 + bf16 output |
| bf16 partials, mrow, depth_x=1 (the fix) | 0.2145 | |
| old composite | 0.2043 | |

Standalone `ttnn.matmul` (h bfp8 @ W bfp4, LoFi, bf16 DEST, single core, `test_matmul_kblock.py`): rel-RMS vs
the quantized-exact product is 0.032 when 64 K tiles accumulate in DEST, 0.013 at 8, 0.009 at 4, and 0.0017
with fp32 DEST. bf16 DEST accumulation over long K is a real error source on hardware (much larger than the
round-to-nearest emulation predicts), but in the fused op the mrow path's extra error is NOT from accumulation:
with bf16 output mrow and non-mrow produce bit-identical results. The extra error appears only when the mrow
path packs to **bfp8**: its output deviates from nearest-bfp8 of the bf16 result by 0.033 rel-RMS (non-mrow:
0.008), values look coarser than 7-bit mantissas (e.g. 0.203 -> 0.25, -0.095 -> -0.156) and carry a +0.9%
signed bias. Uniform over rows, columns, in-tile positions; grows with |value|. OPEN ITEM: the bfp8 pack in
the full-M down path (pack_row_strided of 1x3 tiles straight from a 64-step DEST accumulation) is coarser than
the reload+pack of the ragged path. Not chased further per direction (perf under the two fixed regimes is the
objective); worth a look by whoever owns the numerics because it is the larger of the two error terms.

### 2.3 Chosen precision fix (regime "bf16"): `MOE_FUSED_SWIGLU_ACC_BF16=1`
CB_GATE_ACC / CB_UP_ACC / CB_GATHER_GATE / CB_GATHER_UP become bf16 (FormatKey::Acc; ACC_TILE_BYTES CT arg
drives the scatter payload sizes; the bfp8 phase alias is disabled because the page sizes no longer agree).
Extra L1: +92,160 (ACC) +92,160 (landing) +6,528 (alias break) = +190,848 -> 118,720 over budget with the
shipped depths. See 3.1 for how it is paid for.

## 3. Experiments

### 3.1 L1 levers measured in the bfp8 regime (perf cost of each candidate)
| knob | L1 saved | M=64 | M=256 | M=512 | M=1024 | M=5120 |
|---|---:|---:|---:|---:|---:|---:|
| baseline | -- | 84.7 | 117.8 | 198.2 | 344.1 | 1519.8 |
| depth_x 2->1 | 243,712 | 84.8 | 115.9 | 197.6 | 342.1 | 1524.4 |
| depth_h 3->2 (hack_ahead 1) | 69,632 | 85.9 | 121.2 (+3%) | 205.1 (+3.5%) | 363.8 (+5.7%) | 1613.1 (+6%) |
| depth_x 1 + depth_h 2 | 313,344 | 85.7 | 120.9 | 205.5 | 363.8 | 1612.4 |

**depth_x=1 is free at every M** (within the ±2% noise), including 20-block dispatches. Reason: with row-major
x the cross-block prefetch lands in `cb_x_in` (sticks), not in the second resident slot; the second slot only
lets the reader reserve block b+1's slot during block b's phase 2, but the reader reaches the block b+1 x
multicast only after its phase-2 loop, by which time compute has nearly finished block b anyway. depth_h=2
costs 3-6% (the 3-deep H row pipeline with 2-ahead acks matters). Decision: pay for bf16 with depth_x=1.
depth_h=4 (with the freed L1) HANGS: only three SEM_H_RDY flag cells exist (ids 8..10, SEM_H_FREE is 11), so
depth 4 needs a semaphore renumbering; parked.

### 3.2 bf16 regime with depth_x=1 (`ACC_BF16=1 DEPTH_X=1`), L1 1,388,480 (72 KB free), wd resident + mrow kept
| M | bfp8 regime ns | bf16 regime ns | delta | bf16 rel-RMS (bfp8: see 1.3) |
|---:|---:|---:|---:|---:|
| 64 | 84,748 | 85,006 | +0.3% | 0.2058 |
| 128 | 93,536 | 94,211 | +0.7% | 0.2073 |
| 256 | 117,827 | 115,399 | -2% (noise) | 0.2145 |
| 512 | 198,172 | 200,030 | +0.9% | 0.2142 |
| 1024 | 344,124 | 353,741 | +2.8% | 0.2142 |
| 5120 | 1,519,762 | 1,589,367 | +4.6% | 0.2140 |
The doubled scatter payload (per core 2 x 48 x 2 KB per block, 17 MB grid-wide per block instead of 9 MB)
costs ~3-5% at high M and nothing measurable at low M. Both regimes stay selectable.

### 3.3 Weight chunk count (GU_CHUNKS knob, bfp8 regime, depth_x=1)
| chunks | M=64 | M=256 | M=1024 |
|---:|---:|---:|---:|
| 1 | 86.7 | 128.3 | 345.2 |
| 2 | 86.6 | 122.7 | 343.8 |
| **3 (shipped)** | 84.8 | 115.9 | 342.1 |
| 6 | 90.3 | 140.2 | 441.7 |
3 stays. 6 makes the in1 sub-block one tile wide (no SRC reuse) and doubles the per-chunk barriers.

### 3.4 Issue all weight chunks up front with per-chunk transaction ids (REJECTED)
Reader W_gate and writer W_up: one reservation, all chunks issued at once, each with its own trid,
published per trid. Result: M=64 89.7 (+6%), M=256 132.1 (+14%), M=1024 359.9 (+5%). Under a saturated DRAM,
queueing three chunks per core triples the queue depth in front of chunk 0, so chunk 0 lands ~3x later and
the compute/stream overlap of the first chunks is lost. Confirms the shipped design note: only the chunk
compute needs next should be outstanding. Reverted.

### 3.5 Grid-wide x-staged barrier before any weight stream (block 0) (REJECTED)
Per-row zone medians at M=256 (baseline) showed a strong ROW skew: x reads done at 4.8 us (row y=9) ... 16 us
(row y=2), W_gate done at 28 ... 64 us, gate/up done at 51 ... 80 us; every column's reduce then waits for its
slowest row (84.7). Hypothesis: fast rows' weight streams starve slow rows' x reads. A grid-wide barrier
(88 acks to core (0,0), multicast go) made x land by 12 us for every row, but W_gate on NoC0 stayed skewed
(33 ... 66 us) and the fast rows lost their head start: M=64 91.8 (+8%), M=256 125.5 (+8%). Total phase-1
DRAM traffic (20.7 MB) still took ~68 us in both runs. The skew is NoC0 read-return arbitration by row, not
start time. Reverted.

### 3.6 DRAM transaction size (bfp4 tile = 576 B)
Probe (`test_dram_bw.py`, clone of a 7168x2048 interleaved tensor): 576 B pages 288 GB/s, 1088 B 408 GB/s,
2048 B 399 GB/s (read+write). Ablation inside the op (wrong data, bank-uniform runs): 2-tile W_gate/W_up runs
(1152 B) M=64 79.4 (-6%), M=256 120.6 (+4%), M=1024 347 (+1%); 3-tile W_down runs (1728 B) M=256 +3.5%;
6-tile runs (GU_CHUNKS=1) M=256 130.7 vs 128.3 for chunks=1 alone. A first ablation with runs starting at
hstart hit only even banks (hstart = 6x) and was 25% slower -- bank coverage dominates transaction size here.
Conclusion: a bank-aware hidden/output ownership (stride-8 column sets, contiguous in-bank runs) would buy
~5 us at M=64 and nothing at M=256; parked, not worth the ownership rewrite now.

### 3.7 x row-multicast protocol (three variants, all REJECTED)
Baseline chain: per round ack -> payload -> VALID -> reset, 8 sequential rounds, 18 us (median row) to 47 us
(slowest row) at M=256, ~8 us per block in steady state.
* Fully parallel rounds (per-round monotone flag words in a 64 B CB, one ack per sender per block): M=64 84.9,
  M=128 92.3 (-1%), M=256 114.1 (-2%), M=512 204 (+3%), M=1024 371 (+8%), M=5120 1758 (+16%). Fine zones:
  senders' 30 KB payload issue took 3-16 us -- eight concurrent multicasts on the unidirectional NoC0 row
  ring with path reservation serialize badly; steady-state blocks also collide with the previous block's H
  rounds. Skew grows block over block.
* Ordered rounds on flags (sender t waits flag t-1, no consumer ack round trip): M=256 123.0 (+6%),
  M=1024 349.8 (+2%), M=5120 1569 (+3%).
* Lesson: the chain's cost is NoC transfer + one payload at a time on the row ring, not handshake latency.
  Reverted to the original protocol. (Also: CB pages are not zeroed -- a monotone flag in a fresh CB page
  read stale L1 as "arrived" and produced inf output until zero-initialised at kernel start.)

### 3.8 Stream-pair ablations (perf only, wrong data; bfp8 regime, depth_x=1)
| variant | M=64 | M=256 |
|---|---:|---:|
| baseline | 84.2 | 114.7 |
| no W_gate reads | 82.8 | 110.4 |
| no W_up reads | 75.0 | 107.1 |
| no W_gate + no W_up | 48.6 | 87.2 |
| no W_down reads | 66.0 | 107.5 |
| no W_down + no W_gate | 62.9 | 102.6 |
Either gate/up stream alone is worth 4-9 us, both together 27-36 us: the pair streams at ~470 GB/s, i.e. phase 1
is DRAM-bound as a pair, and which rows finish last is decided by NoC0 read-return arbitration. W_down is worth
19 us at M=64 (it is simply the serial DRAM tail there) and 7.5 us at M=256. With both NoC0 weight streams
removed the per-row skew vanishes (x multicast done at 15-21 us on every row) and the op is STILL 101 us:
x ~20 + gate/up FPU 36 (6 chunks x 6.0 us; 5.3 us ideal) + reduce 10 + SiLU 5.5 + down 24.5. That is the
structural floor of the current design at M=256; the extra ~15 us of the real run is the slow rows' late x.

### 3.9 Early-issue of the next W_gate chunk from inside the multicast loop (REJECTED)
Chunk c+1 issued as soon as chunk c's own transaction id drains (polled per multicast round). M=256 133.5
(+15%), M=128 102.8 (+9%): same mechanism as issuing all at once -- more W_gate bytes in flight during the x
phase slow the losing rows' x further. One chunk in flight, issued only after the multicast loop, is optimal.

### 3.10 W_gate split across NoCs, WD_SPLIT-style (REJECTED, code removed)
Writer reads the tail WG_SPLIT/8 K rows of every W_gate chunk on NoC1 into the reader's chunk slot and
publishes a monotone counter. Share 4/8: M=64 110 (+31%), M=256 135 (+16%); share 2/8: 98 / 123; two chunks
in flight without the split: M=256 122.7 (+6%). NoC1 reads are the weak path; any W_gate byte moved there
loses monotonically with the share.

### 3.11 W_gate issued only after the x multicast (REJECTED): M=256 123.8 (+6%), M=128 96.3 (+2%).

### 3.12 Full-row down schedule for partial blocks (m_eff 1/2/4) -- implemented, gated OFF (`MROW_PARTIAL`)
m_eff rounds of one complete H row (assembled from HN_PAD/a workers per row on the diagonal cores) instead of 11
hidden-block rounds. Correct (functional suite green), and the down phase at M=64 falls from 32 to 8.6 us -- but
the op does not move (M=64 82.0 vs 84.2, M=128 94.2 vs 94.1): at small M the resident W_down DRAM read
(8.3 MB, issued after the gate/up streams) is what the down phase was really waiting for, and the ordinary
path had been consuming it block by block as it landed. It also inherits the full-row bfp8 pack error
(M=64 rel-RMS 0.2156 vs 0.2078). Kept behind the knob for a W_down-resident-early future.

### 3.13 Issue the resident W_down batch first (REJECTED, `WD_EARLY`)
Both NoCs issue their W_down share at the top of block 0 with per-stream transaction ids (x, W_gate, W_up,
phase 2) so no scoped barrier waits for it. M=64 89.5 (+7%), M=128 101 (+8%), M=256 135 (+16%): the batch
competes with x and the gate/up streams, which compute is waiting on, while W_down is not needed until the
down phase. (Side finding while debugging: dropping the writer's 3.1 MB W_down share entirely is worth only
~3 us at M=256.) The per-stream transaction ids stay (harmless, and they are what makes the batch position
a free choice); default off.

### 3.14 Other knobs measured (bfp8 regime): gate/up sub-block height 2: M=256 121.8 (+6%), 4: 117 / M=64 93.6
(+10%); depth_h 4 (needs the 4th flag cell, added): neutral within noise at every M, and ILLEGAL for partial
blocks with HN_PAD 6 (4 x 64 tiles is not a multiple of the 24-tile round: output garbage at M=128) -- the
geometry now rejects depth_h > SEM_H_RDY_CELLS and keeps 3. x stick read split across both RISCs (block 0,
odd sticks on NoC1): M=64 88.2 (+4%), M=256 124.7 (+7%); off.

### State after 3.14 (both regimes, depth_x=1 default)
| M | bfp8 regime ns | bf16 regime ns | bfp8 rel-RMS | bf16 rel-RMS |
|---:|---:|---:|---:|---:|
| 64 | 84.5 | 84.8 | 0.2078 | 0.2058 |
| 128 | 93.7 | 93.8 | 0.2093 | 0.2073 |
| 256 | 116.1 | 116.9 | 0.2171 | 0.2145 |
| 512 | 195.4 | 202.5 | 0.2169 | 0.2142 |
| 1024 | 343.5 | 354.3 | 0.2169 | 0.2142 |
| 5120 | 1518.2 | 1590.8 | 0.2167 | 0.2140 |
Goal check: bf16 regime at M=256 is 116.9 us (<= 120) with lower error than the shipped bfp8 path (0.2145 vs
0.2171); the bfp8 regime is unchanged within noise. Remaining levers are compute-side: down-matmul two-row
sub-blocks, scatter overlap with the last gate/up chunk.

### 3.15 Two H rows per down-matmul call (REJECTED, kept behind `MOE_DOWN_ROWS_MAX`, default 1)
Halves the W_down unpacks per K step. Correct once a pair is not allowed to straddle the cb_h wrap (row r
sits in slot r % DEPTH_H because every block returns to the base), but M=256 131.7 (+13%), M=1024 398
(+15%): popping rows in pairs halves the effective depth of the 3-deep H pipeline, and the down phase is
paced by round delivery, not by the matmul. Would need DEPTH_H 4, which is illegal for partial blocks here.

## 4. Deliverable state

* `intermediate_dtype` op argument (BFLOAT8_B default = original, BFLOAT16 = bf16 partials + landing);
  the env knob `MOE_FUSED_SWIGLU_ACC_BF16` still overrides it for A/B runs. L1: bfp8 1,145,408 B,
  bf16 1,388,480 B of 1,461,248 (depth_x=1 in both).
* Everything else defaults to the shipped schedule; every rejected experiment is behind a knob or removed.

Final numbers (RT profiler, median of 3, bfp4 weights, x bf16 RM, 11x8; rel-RMS vs fp32 reference):
| M | bfp8 regime ns | bf16 regime ns | delta | bfp8 rel-RMS | bf16 rel-RMS | old composite ns / rel-RMS |
|---:|---:|---:|---:|---:|---:|---:|
| 64 | 84,033 | 85,144 | +1.3% | 0.2078 | 0.2058 | 203,663 / 0.2031 |
| 128 | 94,067 | 94,387 | +0.3% | 0.2093 | 0.2073 | 198,038 / 0.2046 |
| 256 | 115,769 | 117,117 | +1.2% | 0.2171 | 0.2145 | 210,714 / 0.2043 |
| 512 | 198,307 | 201,097 | +1.4% | 0.2169 | 0.2142 | 264,737 / 0.2041 |
| 1024 | 346,536 | 358,166 | +3.4% | 0.2169 | 0.2142 | 376,619 / 0.2040 |
| 5120 | 1,523,199 | 1,593,615 | +4.6% | 0.2167 | 0.2140 | 1,648,480 / 0.2038 |

What did NOT move and why (the honest summary): phase 1 is DRAM-bound as a pair of weight streams (~470 GB/s)
with a NoC0 row-arbitration skew that decides which cores finish last; every reordering of that phase
(issue-all, early next chunk, grid barrier, W_gate after x, NoC split, early W_down, x read split, parallel or
flag-ordered x rounds, chunk count, sub-block height) measured worse than the shipped schedule. The remaining
gap to the old composite's error is dominated by the full-row down path's bfp8 pack (section 2.2), which is
outside the two agreed precision regimes.

Heavy-tailed inputs (`BENCH_SPIKY=1`: 1% of positions x16, 8 shared outlier channels x32), rel-RMS vs fp32:
| M | bfp8 regime | bf16 regime | old composite |
|---:|---:|---:|---:|
| 64 | 0.2080 | 0.2057 | 0.2038 |
| 256 | 0.2088 | 0.2060 | 0.2018 |
| 1024 | 0.2094 | 0.2072 | 0.2001 |
Same ordering as the Gaussian case: bf16 partials recover roughly a third to a half of the gap to the composite.
