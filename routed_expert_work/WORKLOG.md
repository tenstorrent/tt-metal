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

## 5. Second round (after the goal check): two wins

### 3.16 Chunked reduce-scatter (ACCEPTED, `MOE_FUSED_SWIGLU_CHUNKED`, default on)
Full blocks only (slice == one token row, a == HN_PAD). The gate/up accumulator becomes chunk-major
([chunk][row][GU_CHUNK_W]); compute publishes each N-chunk as it completes; reader/writer scatter that chunk
(GU_CHUNK_W tiles per worker, one signal per chunk after both halves land); the landing CBs are pushed per
chunk and compute folds/SiLUs/multiplies per chunk. Only the last chunk's transfer stays exposed. The bfp8
phase alias (gather_gate / h_slice / out_tiles) had to be dropped under this schedule (h_slice tiles are
produced while peers still land later chunks): +58 KB, plenty of room after depth_x=1.
Two bugs on the way: (1) the pack index had the chunk offset added although the per-chunk push already
moved the CB write pointer (double offset -> inf); (2) the alias corruption above.
| M | bfp8 before | bfp8 chunked | bf16 before | bf16 chunked |
|---:|---:|---:|---:|---:|
| 256 | 116.3 | 116.2 | 117.1 | 117.6 |
| 512 | 195-198 | 194.1 | 201.1 | 194.4 |
| 1024 | 343.5 | 338.0 | 358.2 | 339.5 |
| 5120 | 1523 | 1490 | 1594 | 1486 |
The bf16 regime's doubled payload is now fully hidden at high M (it matches bfp8).

### 3.17 Reduce chunk c-1 between gate/up matmul chunks (REJECTED: hangs)
Meant to overlap the PACK-thread SiLU of chunks 0-1 with the next chunk's matmul. Deadlocks on every core
(watcher: MATH past its DEST acquire, PACK spinning in the SiLU helper's manual MATH_PACK wait, reader in a
semaphore wait, writer in a CB wait). Also hangs with the SFPU loop removed and with a standard
`tile_regs_wait()` instead of the raw wait, so the eltwise-between-matmul-chunks DEST handshake itself is the
problem, not the SiLU math. Not pursued further; all chunks are reduced after the last matmul.

### 3.18 K taper across grid rows (ACCEPTED, `MOE_FUSED_SWIGLU_KR_TAPER`=4, `_KR_SPLIT` explicit override)
The rows that lose NoC0 read arbitration (logical rows 0-1) get fewer K tiles, the winners (rows 6-7) more:
24,24,26,28,28,30,32,32 at emb 7168. Tried 22..34 and 20..34 too (worse at small M: kr_pad grows the padded x
slot and multicast). With the taper the x row-multicast sends only the row's real kr_rows tiles instead of the
KR_PAD slot. The taper is the first rung of the L1 fallback ladder (dropped before any pipeline depth).
| M | bfp8 even | bfp8 taper | bf16 even | bf16 taper |
|---:|---:|---:|---:|---:|
| 64 | 84.3 | 85.5 | 85.1 | 84.9 |
| 128 | 94.4 | 93.2 | 94.4 | 97.3 |
| 256 | 116.7 | 111.8 | 117.6 | 111.4 |
| 512 | 194.1 | 187.2 | 194.4 | 190.5 |
| 1024 | 336.3 | 327.4 | 339.5 | 331.7 |
| 5120 | 1490 | 1449 | 1486 | 1459 |
Caveat: this tunes to a measured arbitration pattern of the 11x8 grid on this Blackhole; if another board's
pattern differs the cost is bounded by the same few percent, and the knob turns it off.

## 6. Final numbers vs the inherited baseline (bfp4 weights, x bf16 RM, 11x8, RT profiler, median of 3)
| M | baseline bfp8 | **bfp8 now** | **bf16 now** | old composite |
|---:|---:|---:|---:|---:|
| 64 | 84.6 | 85.5 (+1%) | 84.9 | 203.7 |
| 128 | 93.5 | 93.2 (0%) | 97.3 | 198.0 |
| 256 | 115.6 | 111.8 (-3.3%) | 111.4 (-3.6%) | 210.7 |
| 512 | 195.1 | 187.2 (-4.1%) | 190.5 (-2.4%) | 264.7 |
| 1024 | 341.6 | 327.4 (-4.2%) | 331.7 (-2.9%) | 376.6 |
| 5120 | 1522.6 | 1449.2 (-4.8%) | 1459.1 (-4.2%) | 1648.5 |
rel-RMS vs fp32 at M=256: baseline 0.2171, bfp8 now 0.2172 (partition changes rounding by 1e-4), bf16 now 0.2145.
Functional suite (kimi_k26 + glm_51, ragged counts 251/768/3001, x RM and TILE, 0..5120 sweep): 28/28.

## 7. DRAM ND-sharded weights (2026-09-07, BH p150a, bfp4 weights, x bf16 RM, intermediate bfp8, 11x8)

The placement support is already in the shipped op — `geometry::nd_shard_n_tiles` reads the shard's N
extent off the tensor and `WeightRuns<SHARD_W>` coalesces each read to the shard boundary — but no
harness exercised it. The spec itself was removed with the Python descriptor; it is
`weight_memory_configs()` in commit `220f7ee87aa`, restored here as `_nd_shard_config` /
`_preferred_shard_widths` in `test_bench.py` behind `BENCH_WSHARD`.

Shard = ONE tile-row tall x per-core-N wide, on the full DRAM grid, ROW_MAJOR (ROUND_ROBIN_1D):
gate/up `hn_pad = ceil(hid_t / gx) = 6` tiles, down `ec_max = max(split(emb_t, gx*gy)) = 3` tiles.

Median of 3 sweeps, 5 dispatches per sweep, `BENCH_WSHARD_VIA=reshard`:

| M | interleaved | ND-sharded | x | gate/up only | down only |
|---:|---:|---:|---:|---:|---:|
| 64 | 85,617 | 76,497 | **1.119** | 1.051 | 1.037 |
| 128 | 96,533 | 80,243 | **1.203** | 1.112 | 1.063 |
| 256 | 111,891 | 100,665 | **1.112** | 1.107 | 1.017 |
| 512 | 187,336 | 178,245 | **1.051** | 1.046 | 1.024 |
| 1024 | 327,171 | 317,868 | **1.029** | 1.025 | 1.009 |
| 2048 | 608,333 | 597,888 | **1.017** | 1.014 | 1.007 |
| 4096 | 1,169,379 | 1,159,803 | **1.008** | 1.007 | 1.004 |
| 5120 | 1,449,910 | 1,440,487 | **1.007** | 1.006 | 1.004 |

The saving is a near-constant **9.1-9.5 us per dispatch** (M=128 reads 16.3 us, and that is the one
cell whose interleaved baseline carries a 3.5% cross-rep spread). W_RESIDENT reads the weights once
per program, so a fixed saving decays as 1/m_blocks — 1.20x at one M-block, 1.007x at twenty. That is
the same predicate `220f7ee87aa` recorded, and the same absolute saving (~9 us then, ~9.5 us now);
the RELATIVE win is larger now only because the op itself got faster (256: 133.9 -> 111.9 us
interleaved since that commit, 5120: 1991.7 -> 1449.9). Sharding also halves the run-to-run spread
(median 0.4% vs 0.3%, max 1.6% vs 3.5%).

Attribution: gate/up carries ~90% of it (two of the three weight streams, and phase 1 is the
DRAM-bound pair — section 3), down ~4 us on its own; the two do not quite add.

Shard HEIGHT confirmed load-bearing on the current op, M=256: h=1 102.1 us, h=2 102.5, h=8 106.4.
A taller shard pins a core to one bank for that many K-rows instead of rotating every row and gives
back about half the win — still ahead of interleaved (111.9), so height is a gradient, not a cliff.

**`from_torch` into an ND config RE-QUANTISES bfp4.** Handing `from_torch` the ND memory config
directly changes 0.74% of the weight values by one quantum (maxabs 0.015625, rel-L2 3.1e-2 on both
the gate/up and the down shape) against the same torch tensor written interleaved, which leaks into
the output and turns a placement A/B into a numerics A/B — the fused op's PCC moved 0.979952 ->
0.979868 at M=256 purely from that. Building interleaved and `to_memory_config`-ing onto the shard
moves the bytes exactly (0/14680064 differ), and then the op's output is **bit-identical** across the
two placements (maxabsdiff 0.0), which is the real correctness gate on the coalesced read path. This
is why the removed `dram_nd_shard_spec` docstring insisted on a device-side reshard; `BENCH_WSHARD_VIA`
defaults to `reshard` for that reason. Reproducer: `ndcheck/test_weight_bytes.py`.

Anchor, so these are comparable to the numbers in PR #54677 and #55636: the old composite measured
here reads 200.8 / 211.8 / 266.6 / 377.1 / 1642.6 us at M=128/256/512/1024/5120, i.e. within 1% of
PR #54677's own composite column (203.4 / 213.4 / 266.8 / 377.8 / 1641.0).

Two things NOT to reuse from this round:
* `--dev` cannot run the sharded variant: watcher pushes the program to 76,592 B against a 70,656 B
  kernel-config buffer. The interleaved variant fits. Correctness here rests on the bit-identity
  check above, not on watcher.
* **bfp8 WEIGHTS are broken, independent of placement** — `BENCH_WDTYPE=bfp8` gives PCC 0.032-0.036
  interleaved and inf/nan rel-RMS sharded, at every M. Off the shipped path (routed experts ship
  bfp4, `DEFAULT_ROUTED_EXPERT_WEIGHTS_DTYPE`) and never measured in this harness before, but the op
  validates BFLOAT8_B weights as legal, so it accepts an input it cannot serve. Separate bug.

## 8. Cross-expert weight prefetch — opportunity sizing (2026-09-07, not implemented)

One program serves every local expert in BOTH ops (`for local_expert_id < EXPERTS_PER_CHIP` in
`moe_fused_swiglu_reader.cpp:417` and `unified_routed_expert_ffn_reader.cpp:319`), so the expert
boundary is now a place where the next expert's weight read could be issued under the current one's
tail. Nothing does that today.

**Nothing overlaps across the boundary today.** N-expert dispatch, same count per expert, `BENCH_EXPERTS`:

| N | fused interleaved | fused ND-sharded | composite |
|---:|---:|---:|---:|
| 1 | 111,861 | 102,208 | 214,658 |
| 2 | 224,156 | 204,046 | — |
| 4 | 443,233 | 405,815 | — |
| 8 | 881,192 | 804,529 | 1,657,437 |
| marginal / expert | 109,904 | 100,332 | 206,111 |

Marginal is 96-98% of the first expert in all three, i.e. every expert pays full price for its own
weight read. At M=64 the same holds (fused ND 76,879 -> 599,531, marginal 74,664).

**The prize** is section 3.8's stream-pair ablation, which is exactly "the gate/up read costs nothing":
M=256 114.7 -> 87.2 us (-27.5), M=64 84.2 -> 48.6 us (-35.6). Removing ONE stream is worth only
4-9 us, both together 27-36 — the pair shares bandwidth, so a prefetch has to move BOTH to collect
anything. ND sharding has since banked ~9.5 us of that same pool, so the residual is ~18 us at M=256
and ~26 us at M=64 per expert.

**The window** is phase 2. Stage zones at M=256, interleaved (kernel 111.3 us, so the profile costs
~1%): `compute_down` 25.4 us at offset 84.8, and across it NCRISC sits in `p2_hwait` 26.8 of its
27.9 us `reader_phase2` — 96% blocked on h delivery. The weight DRAM path is silent there:
`reader_wd_wait` is 0.1 us at offset 80.3, `p2_wd_barrier` never fires, because `depth_wd = hgroups`
holds the whole resident W_down shard and `reader_wd_issue` (4.2 us at 49.4) + `writer_wd_issue`
(2.4 us at 45.5) landed it during phase 1. So there is a genuine ~25-30 us hole per expert with the
weight readers idle and the reader RISC free to issue.

Ceiling for 8 experts (the Kimi K2.6/2.7 shape), ND-sharded, hiding 7 of 8 reads:
M=256 804.5 -> ~678 us (**1.19x**), M=64 599.5 -> ~417 us (**1.44x**). Window-bounded (the hole is
~25 us and the residual prize ~18-26 us, so capture is partial): **~1.1-1.2x at M=256, ~1.3-1.45x at
M=64**. Same 1/m_blocks decay as ND sharding — nil past ~1K tokens.

**L1 is NOT a blocker — the first pass of this note claimed it was, wrongly.** No second slot is
needed: the prefetch OVERWRITES the weights the current expert is finished with. The CB slots are
already cycled every M-block. `matmul_row_major<..., retain_in1=false>` pops each gate/up chunk as it
consumes it, and residency is implemented purely as "do not rewrite": the reader does
`read_wg = (block_idx == 0) || !W_RESIDENT` and the writer `read_w = (b == 0) || !W_RESIDENT`, so
blocks after the first reserve/push the SAME slot with no DRAM read and re-consume the bytes still
sitting in it (reader:589-591, writer:90-91). So the residency gate simply INVERTS: instead of
skipping the read on blocks > 0, issue the NEXT expert's read on the LAST block, and skip it on the
next expert's block 0. Same number of DRAM reads, same L1, moved one phase earlier.

The only extra space is one chunk of slack, so the prefetch of chunk c can be in flight while the
current expert still consumes chunk c: `GU_CHUNKS = 3`, `CHUNK_W_TILES = KR_PAD * GU_CHUNK_W = 32*2
= 64` tiles = **36,864 B per weight CB, 73,728 B for the pair**, against 186,432 B free
(`MOE_FUSED_SWIGLU_LOG_L1=1`: 1,274,816 of 1,461,248 used). Fits with ~112 KB spare. Serialising the
last chunk instead costs zero L1 and one small bubble.

Correspondingly the window is bigger than the down phase alone. The current expert's weight DRAM
traffic ends at ~49 us (`reader_wg_wait` 27.9+21.4, `writer_wup` ends 45.4) and the kernel runs to
110 us, so the DRAM-quiet stretch is **~60 us against an ~18 us residual prize** — bandwidth is not
the constraint, and the prefetch can be throttled to ~0.3x phase 1's instantaneous rate, which is the
regime least likely to reproduce the contention losses below.
**The one real blocker is contention.** FIVE in-expert variants of "more weight bytes in flight during another phase" all lost:
  issue-all chunks +14% (3.9), early next W_gate chunk from the multicast loop +15% (3.9), W_gate
  split to NoC1 +16%, early W_down +16%, x sticks split to NoC1 +7% (all in 3.x). The mechanism that
  killed them is NoC0 read-return contention with the phase's own traffic, and phase 2 is paced by
  h delivery on the NoC, not by FPU. The one asymmetry in a CROSS-expert prefetch's favour: the bytes
  are not needed until the next expert starts, so it can be throttled (a K-block per h round, dropped
  if the round is late) in a way an in-expert prefetch of this-block bytes cannot. That throttle is
  the whole experiment.

**Fused benefits more than the composite, structurally.** The fused op holds weights resident
(`read_wg = (block_idx == 0) || !W_RESIDENT`) and so has the concentrated read + quiet tail above.
The composite reads gate/up inside its `for kb < num_blocks_gu` loop within the chunk loop — the
weight stream is spread across the whole expert, so there is no hole to move the next expert's read
into. Its average weight rate is also only 24.8 MB / 214.7 us = 115 GB/s against the fused op's
216 GB/s, i.e. it is issue-bound rather than bandwidth-bound, which is what PR #55636 attacks
directly; closing that gap makes it MORE bandwidth-bound and leaves it less prefetch headroom, not
more. Not measured for the composite — this is read off its kernel structure and the two rates.

### 8.1 Cross-expert W_gate prefetch — IMPLEMENTED, correct, and REJECTED on perf

`-D MOE_WG_PREFETCH=1|2|3` in the reader (default 0, off; baseline provably unchanged — mode 0 reads
880,030 / 682,784 ns at M=256 / M=64 against the pre-patch 881,192 / 682,753, i.e. 0.13%).

The mechanism is the residency gate inverted, and it needs no second buffer, exactly as predicted:
on the LAST M-block the next expert's W_gate is read into THIS expert's slot
(`cb_reserve_back(cb_w_gate, WG_BLOCK_TILES)` is the anti-dependency — it returns only once compute
has popped all GU_CHUNKS chunks), the next expert's block 0 then publishes it with no DRAM read, and
the existing `WG_TRID` publish barrier drains it. Three placements: 1 EAGER (all chunks at the top of
phase 2), 2 THROTTLED (one chunk per h round), 3 LATE (after the h rounds).

**Correct.** With `BENCH_DISTINCT_W=1` (expert i's weights scaled by 1 + i/4, so an off-by-one that
hands expert i+1 expert i's weights is visible) all three modes reproduce the baseline exactly:
pcc 0.979358 rel_rms 0.21797 at M=256 N=8, identical to prefetch off. Note the earlier N-expert
scaling runs used IDENTICAL weights per expert and could not have caught a wrong-expert read at all.

**Every variant is slower.** 8 experts, 3 M values, baseline cross-rep spread 0.4%:

| weights | M | off | eager | throttled | late |
|---|---:|---:|---:|---:|---:|
| interleaved | 64 | 682,784 | 0.900x | 0.897x | 0.897x |
| interleaved | 256 | 880,030 | 0.933x | 0.833x | 0.891x |
| interleaved | 512 | 1,474,139 | 0.904x | 0.900x | 0.921x |
| ND-sharded | 64 | 599,723 | 0.872x | 0.901x | 0.902x |
| ND-sharded | 256 | 806,553 | **1.007x** | 0.910x | 0.895x |
| ND-sharded | 512 | 1,400,967 | 0.965x | 0.962x | 0.918x |

The single non-losing cell is a 0.7% wash. This is the SIXTH variant of "more weight bytes in flight
during another phase" to lose in this op (3.9 has the other five), and throttling — the one thing
section 8 argued would distinguish a cross-expert prefetch from those — made it WORSE, not better
(0.833x at M=256): a `set_trid` + issue + `set_trid` per h round lands directly on the delivery that
paces phase 2.

**The concrete blocker, and what has to be fixed before a retry is worth anything.** Phase 0 of every
expert ends in a GLOBAL `noc_async_read_barrier()` (reader:448, draining the idx/counts pair), which
force-drains ANY outstanding read — including a prefetch — at the next expert's first instruction.
So the reachable overlap is only the phase-2 tail, and LATE mode gets essentially none, which is why
it loses about as much as EAGER despite touching h delivery least. Scoping that barrier to a trid is
a prerequisite, not an optimisation.

Also note ND-sharded eager is the best cell (1.007x) while interleaved eager is 0.933x: the same
prefetch coalesced into ~57% fewer transactions stops hurting. That is consistent with the cost being
NoC transaction pressure rather than bytes, and it means the DRAM-idle window measured in section 8 is
real but NOT usable — the window is idle in DRAM terms and busy in NoC terms.

(SUPERSEDED by 8.2 -- the diagnosis in this section was right about the symptom and wrong about the
conclusion. Gate-only cannot win; the pair can.)

### 8.2 Why gate-only lost, and the fix: prefetch BOTH streams

Zone probe (`-D MOE_PF_PROFILE=1`, four zones only -- the full stage set pushes the program past the
70,656 B TENSIX kernel-config buffer at EXPERTS_PER_CHIP >= 2, which is the only configuration the
prefetch runs in). 8 experts, M=256, interleaved, gate-only prefetch vs off:

| zone | off | gate prefetch |
|---|---:|---:|
| `pf_publish` (the exposed W_gate wait) | 173.8 us | **23.8 us** |
| `pf_issue` (issuing the prefetch) | -- | **55.5 us** |
| `pf_reserve` (the anti-dependency) | -- | 0.3 us |
| `pf_ph0_barrier` | 11.3 us | 11.6 us |
| NCRISC-KERNEL | 875.0 | 939.1 |

So the mechanism WORKS -- it removes 150 us of exposed W_gate wait -- and still loses, for two
reasons the probe makes measurable:

1. **Issue cost.** `pf_issue` is 7.9 us per prefetch. DRAM-interleaved weights page at one tile, so a
   chunk is `KR_PAD * GU_CHUNK_W = 64` separate `noc_async_read` calls and the whole prefetch is 192
   NoC commands; at ~40 ns of command issue each that is 7.7 us, which is what the zone reads. This
   is CPU time on NCRISC, not DRAM time.
2. **The prize for ONE stream is only 4-9 us** (section 3.8). Of `pf_publish`'s 17-22 us per expert,
   most is the reader waiting while compute works -- not critical path. So gate-only is structurally
   incapable of clearing its own 7.9 us overhead. That, not contention, is why all three placements
   lost, and why throttling could never have helped.

**The fix is the pair.** Section 3.8 says either gate/up stream alone is worth 4-9 us and both
together 27-36: they share DRAM bandwidth, so removing one leaves the other saturating phase 1.
Prefetching both empties phase 1 of weight traffic. W_up's producer is the WRITER, so its half is
issued on a different RISC and a different NoC -- the two issue costs are paid in PARALLEL, not
added. `-D MOE_WU_PREFETCH=1` (writer, `wu_cb_base`, WU_TRID, same reserve-as-anti-dependency).

Superadditive exactly as predicted (8 experts, one rep, M=64/256/512):

| | interleaved | ND-sharded |
|---|---|---|
| gate only | 0.895 / 0.934 / 0.906x | 0.874 / 1.003 / 0.969x |
| up only | 0.964 / 0.998 / 0.957x | 1.008 / 1.027 / 0.993x |
| **pair** | **1.070 / 1.025 / 1.027x** | **1.063 / 1.021 / 1.021x** |

Also fixed while here: phase 0's `noc_async_read_barrier()` (reader:448) was GLOBAL, so it
force-drained the prefetch at the next expert's first instruction and threw away the overlap. Now
tagged IDX_TRID and waited on by trid (plus P2_READ_TRID, so a phase-2 read left in flight is still
retired). Worth ~0.4% and removes a hidden coupling.

**Verified, 3 reps, 8 experts, bfp4, `BENCH_DISTINCT_W=1`:**

| M | interleaved off | pair | x | ND off | pair | x | STACK (int-off -> ND-pair) |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 64 | 683,131 | 639,380 | 1.068 | 599,692 | 561,172 | 1.069 | **1.217x** |
| 128 | 780,921 | 743,804 | 1.050 | 671,233 | 665,866 | 1.008 | **1.173x** |
| 256 | 879,309 | 860,255 | 1.022 | 805,579 | 789,987 | 1.020 | **1.113x** |
| 512 | 1,473,718 | 1,436,796 | 1.026 | 1,404,806 | 1,374,540 | 1.022 | **1.072x** |
| 1024 | 2,597,773 | 2,558,255 | 1.015 | 2,532,861 | 2,497,786 | 1.014 | **1.040x** |
| 5120 | 11,580,004 | 11,541,476 | 1.003 | -- | -- | -- | -- |

The saving is a near-constant **~5.5 us per expert** at every M -- the same fixed-cost signature as
ND sharding, and it decays as 1/m_blocks for the same reason. Never a regression at any M.

**Correctness.** The shipped functional suite is SINGLE-expert (`counts = idx_tensor([active_tokens])`)
so it never enters this path. `routed_expert_work/test_prefetch_ragged.py` covers the two things that
can break and that uniform dispatches hide: (a) per-expert weight scales, so an off-by-one handing
expert i+1 expert i's weights is visible; (b) `counts = [c, 0, c]`, because a zero-count expert never
enters the block loop, so a prefetch issued FOR it is never consumed and never barriered -- both
kernels now retire it at `m_blocks == 0` or the expert after it silently runs on the skipped expert's
weights. That drain was missing from the writer half when first written.

**Remaining headroom.** Gross prize ~13 us/expert (the critical-path part of `pf_publish`), paid ~7.5
in issue cost, netting 5.5. Cutting the issue cost is the only lever left and it is transaction-bound:
192 commands interleaved, ~96 ND-sharded (a chunk is only GU_CHUNK_W = 2 tiles wide, so ND-shard
coalescing has almost nothing to join along N). One transaction per chunk would need a shard of
`[kr_rows x GU_CHUNK_W]`, which is the tall shard section 7 measured as no better than interleaved for
the MAIN read path -- and gate/up carry one memory config for both paths, so the shapes conflict.

**Sender-deferred placement (mode 4) — tried, essentially null.** Hypothesis: the 7.9 us of issue is
CPU time on NCRISC, absorbed on a core that only RECEIVES h (it then spends 7.9 us less in
`p2_hwait`) but added to an 88-core-wide critical path on a core that SENDS h. So mode 4 issues early
on receivers and late on senders (`wd_mrow ? mrow_sender : is_root`). Measured against mode 1:
interleaved 639,675 / 746,063 / 865,689 / 1,451,591 vs 638,675 / 745,527 / 860,767 / 1,438,608
(neutral to 0.9% worse); ND-sharded 554,424 / 662,898 / 794,488 / 1,382,467 vs 563,257 / 665,474 /
789,361 / 1,374,647 (1.6% and 0.4% BETTER at M=64/128, 0.6% worse at 256/512). So the issue cost is
spread across all cores rather than concentrated on the senders, and the hypothesis is wrong. Mode 1
stays the default; mode 4 is kept as a low-M micro-variant (best single cell: ND M=64 599,692 ->
554,424 = 1.082x) but is not worth its branch.

### 8.3 The window was the constraint: PIPELINED prefetch, 1.10-1.13x at low M

8.2 left the prefetch at ~5.5 us/expert and blamed the issue cost. The floor measurement says
otherwise. All three weight streams ablated (`MOE_ABL_NO_WG` + the new `MOE_ABL_NO_WU` +
`MOE_ABL_NO_WD`), ND-sharded, 8 experts, per expert:

| | M=64 | M=256 |
|---|---:|---:|
| baseline | 74.9 us | 101.1 us |
| no gate/up reads | 42.9 us | 79.9 us |
| no weight reads at all | 35.7 us | 77.7 us |
| **gate/up exposure** | **32.0 us** | **21.1 us** |

So gate/up alone expose 21-32 us/expert and 8.2 was capturing 5.5. The reason is not the issue cost:
gate+up are ~19.4 MB/expert across 88 cores, ~48 us of DRAM at the rate ND sharding achieves, against
a phase-2 window of only ~40 us at M=256 and ~25 us at M=64. **An all-at-phase-2 prefetch cannot fit
inside its own window** and spills into the next expert's phase 1, which is where the benefit was
going.

**Fix: refill chunk c as soon as compute POPS chunk c of the current expert.** A cumulative
`cb_reserve_back(cb_w_gate, (c+1) * WG_CHUNK_TILES)` is exactly that signal -- compute pops in order --
so the prefetch starts in phase 1 and gets the whole rest of the expert as its window instead of just
phase 2. Placed after this expert's own W_down batch so that batch, needed at phase 2 entry, gets DRAM
first. Polled again after the reduce and at phase 2, with a blocking tail so the bytes are always
there before the next expert.

Blocking on the pop vs polling with `cb_pages_reservable_at_back` splits cleanly by regime, and the
split lands exactly on the PARTIAL-BLOCK boundary:

| mode | M=64 | M=128 | M=256 | M=512 |
|---|---:|---:|---:|---:|
| 5 blocking | 1.084x | **1.128x** | 1.006x | 0.980x |
| 6 polling | 1.080x | 1.020x | 1.028x | 1.021x |
| **7 = blocking if m_t < M_BLOCK, else polling** | **1.084x** | **1.133x** | **1.033x** | **1.019x** |

The mechanism: with `m_t < M_BLOCK` compute has little to do, so the reader has slack and waiting for
a pop is free; on a full block the reader is on the reduce's critical path and must not wait. Mode 7
takes both wins. `-D MOE_WG_PREFETCH=7,MOE_WU_PREFETCH=1`.

**Mode 7 verified, 3 reps, 8 experts, bfp4, `BENCH_DISTINCT_W=1`, off cross-rep spread <=1.0%:**

| M | int. off | int. pf | x | ND off | ND pf | x |
|---:|---:|---:|---:|---:|---:|---:|
| 64 | 682,875 | 653,284 | 1.045 | 595,105 | 548,978 | 1.084 |
| 128 | 774,492 | 706,762 | **1.096** | 673,106 | 593,331 | **1.134** |
| 256 | 882,864 | 830,436 | **1.063** | 808,896 | 784,612 | 1.031 |
| 512 | 1,473,862 | 1,437,579 | 1.025 | 1,404,581 | 1,377,974 | 1.019 |
| 1024 | 2,597,260 | 2,560,470 | 1.014 | 2,532,275 | 2,498,407 | 1.014 |

**Then modes 8 and 9 beat it.** Mode 7 blocks for EVERY chunk on a partial block; blocking for chunk 0
ONLY is better, because chunk 0 is popped early (so the wait is short) and it is the one chunk whose
window would otherwise depend on when a poll happens to land, while chunks 1-2 are better left to the
polls. Mode 8 = block chunk 0 always; mode 9 = block chunk 0 only on a partial block.

| variant | int 64 | int 128 | int 256 | int 512 | ND 64 | ND 128 | ND 256 | ND 512 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 7 | 1.045 | 1.096 | 1.063 | 1.025 | 1.084 | 1.134 | 1.031 | 1.019 |
| **8** | 1.096 | **1.161** | 1.046 | 1.012 | **1.173** | 1.098 | **1.066** | **1.038** |
| 9 | 1.095 | **1.167** | 1.060 | 1.025 | 1.167 | 1.096 | 1.030 | 1.021 |

Mode 8 is the pick: simplest rule, best on the ND-sharded layout that ships (which is itself a
1.09-1.14x win, section 7), and no regression anywhere. Mode 9 is better on interleaved at M>=256, so
the two differ only in which layout they favour past the low-M band.

**Prefetching ALONE now clears 10% at BOTH of the two lowest token counts on both layouts:** mode 8
reads 1.096x / 1.161x interleaved and **1.173x / 1.098x** ND-sharded at M=64 / M=128. Against 8.2's
single-shot pair (1.069x / 1.008x at the same points) that is 2-12x the saving.

**Why M=64 tops out at 8.4% and not more.** Weight DRAM there is ~69 us of a 74.9 us expert, i.e.
~93% DRAM occupancy -- the expert is bandwidth-bound and prefetching can only re-time bytes, not
remove them. M=256 sits at ~68% occupancy, which is why there is more left there (exposure 21.1 us,
captured 3.7-7.3).

**Rejected on the way:** W_down prefetch, even with the second cb_w_down slot built for it
(`MOE_FUSED_SWIGLU_WD_SLOTS=2` + `MOE_WD_SLOTS=2` + `MOE_WD_PREFETCH=1`; +114,048 B, fits in the
186,432 B free, and the reader/writer address the slot by EXPLICIT PARITY because the writer never
pushes cb_w_down and its write pointer would never advance). Correct but slower at every placement
tried: phase 2 (0.90x at M=64 -- it competes with the W_down stream it duplicates), phase 1 before
this expert's own batch (0.965x), phase 1 after it (0.973x), and on top of mode 7 it is 0.91x. W_down
is read in phase 1 alongside everything else, so there is no idle window for it; the second slot is
kept behind its knobs and defaults off. Pipelining the W_up half too (writer modes 2/3) is neutral
(1.084/1.133 -> 1.084/1.133). Sender-deferred (mode 4) null. Mode 8 (block for chunk 0 only) — see below.

**Mode 8 verified (3 reps, prefetch cross-rep spread <=1.0%), the shipping configuration
`-D MOE_WG_PREFETCH=8,MOE_WU_PREFETCH=1`:**

| M | int. off | int. pf | x | ND off | ND pf | x | saved/expert |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 64 | 682,875 | 623,929 | **1.094** | 595,105 | 508,834 | **1.170** | 8.4-12.3 us |
| 128 | 774,492 | 665,879 | **1.163** | 673,106 | 613,616 | **1.097** | 8.5-15.5 us |
| 256 | 882,864 | 845,301 | 1.044 | 808,896 | 758,393 | **1.067** | 5.4-7.2 us |
| 512 | 1,473,862 | 1,452,197 | 1.015 | 1,404,581 | 1,358,725 | 1.034 | 3.1-6.6 us |
| 1024 | 2,597,260 | 2,578,087 | 1.007 | 2,532,275 | 2,474,560 | 1.023 | 2.7-8.2 us |

Correctness: `test_prefetch_ragged.py` 4/4 with PCC identical to prefetch-off, modes 7 and 8.

Unlike every earlier round the saving is NO LONGER a constant per expert -- it is 12.3 us/expert at
M=64 against 2.6-8.5 at M>=512 -- which is the signature of the change: the prefetch is now paced by
how fast compute frees chunk slots rather than by a fixed window, so it collects more where the
exposure is larger.

### 8.4 Pushing further: the prefetch changes which GU_CHUNKS is optimal

Two more levers on top of mode 8.

**(a) A poll per REDUCE CHUNK, not just once after the reduce.** At a full M-block compute frees the
gate/up slots across the whole reduce, so a slot found inside the reduce loop gets the rest of the
reduce plus all of phase 2 as its window. The poll sits in the loop the reader already spins in
(`reader_reduce_up_wait` / `_data_wait`) so it costs nothing when it misses, and it can never block
(chunk 0 was placed by the phase-1 poll long before). Worth ~1% at M>=256 on its own
(ND M=256 758,393 -> 752,390; M=512 1,358,725 -> 1,344,858), nil below.

**(b) GU_CHUNKS.** The prefetch's cost is transaction count, and GU_CHUNKS sets it: the prefetch
issues `GU_CHUNKS x kr_rows` reads, so 2 chunks is 64 transactions where 3 is 96 and 6 is 192. Swept
against its own baseline (ND, 8 experts, one rep):

| GU_CHUNKS | prefetch x at M=64 / 128 / 256 / 512 |
|---|---|
| 2 | **1.165 / 1.191 / 1.200 / 1.072** |
| 3 (default) | 1.167 / 1.090 / 1.070 / 1.044 |
| 6 | 1.007 / **0.898** / 0.979 / 0.948 |

GU_CHUNKS=6 makes the prefetch LOSE, which is the 8.2 failure mode reappearing at 192 transactions
and is the cleanest confirmation that transaction count is the cost. At GU_CHUNKS=2 the prefetch
saves 20.8 us/expert at M=256 against the 21.1 us of gate/up exposure the floor measurement found --
i.e. it now captures essentially ALL of it, and there is nothing left to take at that M.

**Verified at GU_CHUNKS=2 (prefetch cross-rep spread <=0.2%):**

| M | int off | int pf | x | ND off | ND pf | x | saved/expert |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 64 | 692,303 | 646,848 | **1.070** | 585,708 | 503,412 | **1.163** | 6.5-11.8 us |
| 128 | 800,664 | 701,222 | **1.142** | 694,580 | 583,163 | **1.191** | 14.2-15.9 us |
| 256 | 970,587 | 880,670 | **1.102** | 898,179 | 752,393 | **1.194** | 12.8-20.8 us |
| 512 | 1,532,027 | 1,463,043 | 1.047 | 1,476,623 | 1,376,635 | 1.073 | 9.9-14.3 us |

**Read this honestly.** GU_CHUNKS=2 is a WORSE baseline than 3 at M>=128 (ND M=256 off: 898,179 vs
804,964), so part of that 1.194x is the prefetch recovering ground the blocking change gave away. In
ABSOLUTE terms the two land in the same place at M=256 (752,393 vs 752,390) and GU_CHUNKS=2 wins only
at M<=128 (M=128: 583,163 vs 613,369, -5.0%; M=64: 502,938 vs 510,600, -1.5%). So:

* **Best absolute, ND-sharded:** GU_CHUNKS=2 + prefetch at M<=128, GU_CHUNKS=3 + prefetch at M>=256.
* **Prefetching's own worth at the blocking that suits it:** 1.163-1.194x at M=64..256 (ND),
  1.070-1.142x (interleaved).
* **Prefetching's own worth at the shipped default blocking (3):** 1.167 / 1.090 / 1.070 / 1.044.

The real conclusion is that the two are not independent: prefetching removes the reason GU_CHUNKS=3
was chosen (it existed to keep ONE chunk in flight against the phase-1 read, which no longer happens),
so the blocking should be re-tuned WITH the prefetch on rather than inherited from the no-prefetch op.
GU_CHUNKS is already a per-dispatch host knob, so picking 2 below M=256 and 3 above is a host-side
change, not a kernel one.

**Shipping configuration (superseded by 8.5's taper -- see the final table at the end of 8.5).**
`GU_CHUNKS=2`, `kr_taper=4`, ND-sharded: 1.164 / 1.191 / 1.195 / 1.072 at M=64/128/256/512.

### 8.5 The prefetch invalidates the K taper too (2026-09-07)

Same theme as GU_CHUNKS in 8.4, and a bigger effect. `kr_taper` exists ONLY to compensate for NoC0
read-return unfairness under WEIGHT load -- its docstring: "x sticks land after 5 us on the bottom rows
and 16 us on the top rows, W_gate after 28 vs 64 us", and 4 was chosen because it measured -4% at
M=256 and -5% at M=1024/5120. With gate/up prefetched, phase 1 no longer reads weights on NoC0 at all,
so the arbitration the taper corrects for is a different one.

The probe pointed here first. At gc=2, M=256, 8 experts: prefetch off TRISC-KERNEL 901.3 us, on
748.8 us (93.6 us/expert) against a 77.7 us/expert compute-only floor -- 15.9 us/expert still exposed,
which matches section 3.8's note that "the extra ~15 us of the real run is the slow rows' late x"
almost exactly. So the residual after prefetching is x delivery, and the taper is x's knob.

Sweep with the prefetch ON (gc=2, ND, 8 experts, absolute ns):

| taper | M=64 | M=128 | M=256 | M=512 |
|---:|---:|---:|---:|---:|
| 0 | 498,922 | 588,780 | 772,820 | 1,330,580 |
| **2** | 499,053 | **583,733** | 762,318 | **1,320,398** |
| 4 (default) | 502,146 | 585,010 | **747,540** | 1,375,319 |
| 6 | 505,903 | 588,941 | 769,772 | 1,409,415 |
| 8 | 512,092 | 599,793 | 782,666 | 1,450,466 |

Monotone above 2, so the optimum moved DOWN from 4. Isolated at large M (2 reps):

| M | gc=2 t=4 | gc=2 t=2 | gain | gc=3 t=4 | gc=3 t=2 | gain |
|---:|---:|---:|---:|---:|---:|---:|
| 512 | 1,375,048 | 1,322,181 | **3.8%** | 1,347,232 | 1,329,494 | 1.3% |
| 1024 | 2,490,493 | 2,403,416 | **3.5%** | 2,473,873 | 2,433,949 | 1.6% |
| 5120 | 11,410,312 | 11,005,404 | **3.5%** | 11,518,378 | 11,269,143 | 2.2% |

Positive at both blockings and every large M, and larger at gc=2 -- the two re-tunings compound.
Retuning the taper is a clean **3.5-3.8% at M>=512** on top of everything else, and it also flips the
answer to "which blocking": with taper=2, **gc=2 is now best at 5 of 6 M values** (it was only best at
M<=128 with taper=4), losing only at M=256 by 2.0%.

Net effect of this round against the previous best (gc=2, taper=4): +0.6 / +0.2 / -2.0 / +3.8 / +3.5 /
+3.5% at M=64/128/256/512/1024/5120. Against the original default (gc=3, taper=4) with the prefetch:
+2.3 / +4.8 / -1.3 / +1.9 / +2.8 / +4.5%.

**The general lesson, now shown three times** (GU_CHUNKS, kr_taper, and the five NoC-placement
variants of 3.7/3.9 that were rejected under weight load): every parameter in this op that exists to
HIDE or ARBITRATE the phase-1 weight stream is mistuned once that stream is prefetched away. The
op's whole knob set should be re-swept with the prefetch on rather than inherited.

**FINAL configuration and numbers.** ND-sharded weights, `MOE_FUSED_SWIGLU_GU_CHUNKS=2`,
`MOE_FUSED_SWIGLU_KR_TAPER=2`, `MOE_FUSED_SWIGLU_DEFINES="MOE_WG_PREFETCH=8,MOE_WU_PREFETCH=1"`.
Matched off-vs-on, 2 reps, off spread <=0.6% / pf <=0.4%, 8 experts, bfp4, `BENCH_DISTINCT_W=1`:

| M | off | prefetch | x | pcc off | pcc pf |
|---:|---:|---:|---:|---:|---:|
| 64 | 581,840 | 499,974 | **1.164** | 0.979423 | 0.979423 |
| 128 | 684,045 | 582,811 | **1.174** | 0.979471 | 0.979471 |
| 256 | 907,713 | 762,180 | **1.191** | 0.979367 | 0.979367 |
| 512 | 1,490,643 | 1,327,836 | **1.123** | 0.979386 | 0.979386 |
| 1024 | 2,570,748 | 2,399,686 | 1.071 | 0.979369 | 0.979369 |

PCC is BIT-IDENTICAL off vs on at every M, and the ragged gate is 4/4 at this taper as well (the
values shift ~1e-5 from taper=4 because the taper repartitions the K accumulation -- that is the
taper's rounding, not the prefetch's). Prefetching alone now clears 10% from M=64 through M=512,
where before 8.5 it stopped at M=256.

Absolute gain of 8.5 over 8.4's configuration: +0.7 / +0.0 / -1.4 / +3.6% at M=64/128/256/512 and
+4.0% at M=1024 (against gc=3 there, which was 8.4's best at that M).

### 8.6 Remaining knobs re-swept with the prefetch on — all null or negative

Continuing 8.5's premise that the op's knob set should be re-swept with the prefetch on. At the final
config (ND shard, gc=2, taper=2, mode 8), against the verified baseline 499,974 / 582,811 / 762,180 /
1,327,836 ns at M=64/128/256/512.

**`WD_SPLIT` (eighths of each W_down K-block the WRITER reads on NoC1) — NEGATIVE, and instructive.**
The hypothesis was that W_down's head read is the only OTHER NoC0 traffic left competing with x in
phase 1, so pushing it onto NoC1 should relieve x. It does the opposite:

| wd_split | M=64 | M=128 | M=256 | M=512 |
|---:|---:|---:|---:|---:|
| **3 (default)** | **499,168** | 582,585 | **764,838** | **1,326,280** |
| 5 | 521,828 | 578,699 | 774,907 | 1,377,364 |
| 6 | 576,387 | 676,277 | 777,982 | 1,427,874 |
| 8 | 677,073 | 719,368 | 783,452 | 1,499,770 |

Monotone and severe -- 36% worse at M=64 with the whole stream on NoC1. The reason matters for
anything else in this area: **the prefetch made NoC1 the busy one**, since it now carries the W_up
prefetch on top of the output writes, and section 3.x already recorded NoC1 reads as the weak path.
Prefetching did not free NoC0 for x so much as EQUALISE the two NoCs, so there is no longer slack to
shift work into in either direction. `wd_split=3` stays, and this also predicts `X_SPLIT` (which moves
half the x reads to NoC1) still loses -- not retested on that basis.

**`DEPTH_H` — NULL.** 4 measures 499,965 at M=64 against the baseline's 499,974, i.e. identical.
`DEPTH_H=2` fails a legality/L1 constraint and never dispatches. h depth is not the constraint.

**`HACK_AHEAD` — the default 2 is already optimal.** 1 is worse at every M
(0.992 / 0.986 / 0.970 / 0.977x at M=64/128/256/512) and 3 is neutral
(1.003 / 1.001 / 0.996 / 1.001x). So the h-round look-ahead was NOT mistuned by the prefetch, unlike
GU_CHUNKS and kr_taper -- which fits: it paces phase 2's collective, and phase 2 never had a weight
stream of its own to hide.

**`DEPTH_X` — NULL/negative, and it DOES fit.** A second resident x slot was worth testing because
at M>=512 there are several M-blocks and block N+1's x cannot stage during block N at depth 1. At
gc=2/taper=2 it fits with 9,280 B to spare (CB L1 1,451,968 of 1,461,248, and the log confirms
`depth_x 2` was honoured rather than guard-lowered), but it does not pay: M=512 1,322,927 vs 1,327,773
(+0.4%, inside noise) and M=1024 2,424,353 vs 2,397,630 (**-1.1%**). So the x staging depth is not the
constraint either -- consistent with 8.6's finding that the two NoCs are now balanced.

So 8.5's taper was the last knob with anything in it. The residual at M=256 (93.6 us/expert against a
77.7 us compute-only floor) is not reachable by re-balancing the existing schedule; it would need the
x-multicast protocol itself changed, which is section 3.7's territory and a kernel rewrite rather than
a knob.

### 8.7 Is the taper's DIRECTION real? Yes at M>=256 — and 8.5's framing needs correcting

`kr_taper`'s docstring asserts a physical direction ("x sticks land after 5 us on the bottom rows and
16 us on the top rows") that no measurement in this log establishes. Tested directly with explicit
`MOE_FUSED_SWIGLU_KR_SPLIT` lists, all summing to the same 224 tiles, at the final config:

| M | row0=FEWEST (26,26,27,28,28,29,30,30) | row0=MOST (reversed) | rev/fwd |
|---:|---:|---:|---:|
| 64 | 500,421 | 498,683 | 0.997 (null) |
| 256 | 758,556 | 775,496 | **1.022** |
| 512 | 1,326,401 | 1,371,036 | **1.034** |

So the asymmetry IS real and its sign in ROW-INDEX terms is confirmed: row 0 receives data latest and
must carry the least work; reversing costs 2.2-3.4% at M>=256, and is a null at M=64 (too little work
per row for it to show). What is NOT established -- here or anywhere in this log -- is which PHYSICAL
edge row 0 is, or the mechanism. The dimension-order-routing + per-hop-arbitration story is plausible
but which end starves depends on the GDDR endpoint placement relative to the Tensix grid. Settling
that needs a NoC trace (`--collect-noc-traces` + tt-npe), not a bench sweep. Do not quote the
docstring's 5-vs-16 us figures as measured.

**Correction to 8.5.** I reported retuning the taper as "worth 3.5-3.8%". Against the EVEN split
rather than against taper=4, at M=512:

| | ns | vs even |
|---|---:|---:|
| taper=0 (even) | 1,330,580 | -- |
| taper=2 | 1,320,398 | -0.8% |
| taper=4 (old default) | 1,375,319 | **+3.4% WORSE** |

The 4->2 change is therefore mostly **removing harm, not adding gain**: with the prefetch on, the
residual unfairness worth correcting is only ~0.8%, while the inherited taper=4 actively costs 3.4%.
That is what the mechanism predicts -- the prefetch removed the weight-load unfairness taper=4 existed
to correct, so the right response is to taper LESS, and near-zero would do almost as well. The
3.5-3.8% delta against the old default is arithmetically right but should be read as "stop
over-tapering", which also means the exact value is not sensitive: anything in 0..2 is fine.

### 8.8 The row asymmetry is NOT a DRAM effect — the probe was measuring its own consumer (2026-09-07)

8.5/8.7 rest on "some grid rows receive their weights later than others". That is wrong, and the
error was in the instrument, not the device.

**What `pf_publish` actually brackets.** The zone wraps the `WG_TRID` barrier *and* the following
`issue_wg_chunk(c + 1)`, and `issue_wg_chunk` opens with `cb_reserve_back(cb_w_gate, ...)`. So at
`GU_CHUNKS >= 2` the zone includes a wait for **compute to pop chunk 0**. It is a consumer-progress
probe wearing a producer's name.

**The control.** At `GU_CHUNKS=1` there is no chunk 1, so nothing inside the zone can block on the
consumer and it degenerates to a pure DRAM read barrier. Same 8 experts, M=256, ND-sharded weights,
even 28-tile split, prefetch on:

| phys y | logical row | gc=2 (barrier + reserve) | gc=1 (barrier only) |
|---:|---:|---:|---:|
| 2 | 0 | 44.9 | 0.7 |
| 3 | 1 | 46.2 | 0.7 |
| 4 | 2 | 124.0 | 0.8 |
| 5 | 3 | 177.9 | 0.8 |
| 6 | 4 | 197.6 | 0.8 |
| 7 | 5 | 144.6 | 0.8 |
| 8 | 6 | 102.9 | 0.8 |
| 9 | 7 | 105.6 | 0.8 |
| | **spread** | **4.40x** | **1.08x** |

**DRAM read return is row-uniform to within 8%.** The 4.4x is entirely back-pressure.

Two independent checks agree:

* **Hop distance cannot explain a hump anyway.** Blackhole DRAM sits only in columns x=0 and x=9
  (`blackhole_140_arch.yaml`); the NoC0 worker endpoints are y in {11,2,9,5} at x=0 and {11,3,8,6}
  at x=9. Mean hops for worker rows y=2..9 are [5.6, 5.1, 6.1, 5.6, 5.1, 6.1, 5.6, 5.1] -- flat and
  period-3, against a measured smooth 4.4x hump. No positional model fits.
* **The hump is pinned to the row, not to the split** -- it survives even / U-shaped / reversed splits
  unchanged (row y=6 with 22 tiles is still 4.4x slower than y=2 with 32). That is what a
  compute-schedule effect looks like: the reduce-scatter dependency order is a function of `my_row`
  (`slice_assigned(gu_block_tiles, KGROUPS, my_row)`), not of how many K tiles the row reads.

**Why no shaped split can exploit it.** `kr_pad = *max_element(kr_sizes)` and it sizes `CB_W_GATE`,
`CB_W_UP` and `CB_X_TILES` on *every* core. Since the split must sum to 224 over 8 rows, any shaping
raises the max above the mean 28, so compensation always costs L1 on all 88 cores. `kr_taper = t`
gives deltas `{-t,-t,-t/2,0,0,+t/2,+t,+t}`, hence `kr_pad = 28 + t` exactly.

Controlled A/B, all summing to 224, prefetch on, gc=2, ND, 7 iters:

| split | kr_pad | M=64 | M=256 | M=512 |
|---|---:|---:|---:|---:|
| even `28x8` | 28 | 499,544 | 774,261 | 1,329,499 |
| taper=2 `26,26,27,28,28,29,30,30` | 30 | 499,788 | 768,507 | 1,331,193 |
| hump-matched `30,30,28,26,26,28,28,28` | 30 | 498,589 | 775,016 | 1,354,676 |
| hump-matched `29,29,28,27,27,28,28,28` | 29 | 499,050 | 768,050 | 1,343,884 |
| U-shape `32,32,28,24,22,26,30,30` | 32 | 501,850 | 798,741 | 1,393,289 |
| gentle U `30,30,28,26,24,27,29,30` | 30 | 501,728 | 772,997 | 1,354,250 |

**Shape is a null; only the max matters.** kr_pad 28/29/30 are all equal to within run noise (+-0.8%);
kr_pad 32 costs 3.4-4.5% (matching taper=4's 3.4% in 8.7). Fitting the split to the measured per-row
profile made things *worse*, which is the clearest possible evidence the profile is not a supply
constraint.

**Corrections to 8.5 and 8.7.**
* 8.7's claim "the asymmetry IS real and its sign in ROW-INDEX terms is confirmed" is too strong. The
  monotone-reversal cost (2.2-3.4% at M>=256) reproduces, but it is **not** a data-arrival effect, and
  a hump fitted to the arrival profile at the same `kr_pad` is a null -- so it cannot be modelled or
  tuned from arrival times. The residual mechanism (why *monotone* order matters at all when the
  *amount* per row does not) is unexplained and is a reduce-scatter scheduling question, not a NoC one.
* 8.7's "residual unfairness worth correcting is only ~0.8%" **stands, and my first reading of this
  section ("worth nothing measurable") was wrong** -- it came from single runs read against an
  eyeballed +-0.8% noise floor. Measured properly (3 interleaved sessions per config, 7 iters each,
  actual session spread 0.1-0.6%), `kr_taper=2` beats the even split reproducibly with non-overlapping
  ranges:

  | M | taper=0 reps (us) | taper=2 reps (us) | t0/t2 |
  |---:|---|---|---:|
  | 64 | 499.2, 498.8, 500.2 | 499.4, 499.2, 501.0 | 1.000 |
  | 256 | 773.1, 774.2, 777.0 | 765.6, 770.5, 766.3 | **1.010** |
  | 512 | 1327.7, 1330.1, 1328.0 | 1322.8, 1324.2, 1324.9 | 1.003 |
  | 1024 | 2424.1, 2425.6, 2427.1 | 2398.1, 2400.2, 2403.5 | **1.011** |

  So the taper is worth ~1% at M=256 and M=1024, 0.3% at M=512, nothing at M=64. Recommended default:
  `kr_taper = 2` (the shipped 4 is still wrong -- it costs 3.4-4.5% via `kr_pad`). Do not go above 2.

**What survives and what does not.** The ~1% is real but it is NOT a data-arrival effect -- the
barrier-only measurements are flat (1.08x in-op, 1.07x stock-op) and a split fitted to the arrival
profile *lost*. What the evidence supports is a **monotone row-ORDER** effect and nothing more: the
consistent ranking is forward taper > even > reversed (reversal costs 2.2-3.4%, 8.7), while the
*amount* per row cannot be fitted from arrival times. Row 0 wants the least K work and row 7 the most,
for reasons in the reduce-scatter schedule (`slice_assigned(..., my_row)`), not in the NoC. A +-2 ramp
captures it; +-4 over-pays in `kr_pad`.
* The docstring's "5 us on the bottom rows and 16 us on the top rows" is not just unmeasured, it is
  contradicted: with the consumer wait removed, every row is served in 0.7-0.8 us. It should be deleted.

### 8.9 The x stream IS row-unfair — and that is what the taper corrects (2026-09-07)

8.8 disproved the row asymmetry for **W_gate**. The `kr_taper` docstring makes a *second*, separate
claim about the **x** stream ("x sticks land after 5 us on the bottom rows and 16 us on the top"), and
`reader_x_read` has the same confound: it opens with `cb_reserve_back(cb_x_in, TILE_H)`. Split the zone
in two (`pf_x_reserve` = the CB wait, `pf_x_barrier` = the read return alone; both under
`MOE_PF_PROFILE`) and the two halves separate completely. M=256, 8 experts, even 28-tile split, gc=1:

| phys y | logical row | `pf_x_barrier` (us) | `pf_x_reserve` (us) |
|---:|---:|---:|---:|
| 2 | 0 | **449.9** | 0.6 |
| 3 | 1 | 340.3 | 0.6 |
| 4 | 2 | 247.8 | 0.6 |
| 5 | 3 | 140.0 | 0.6 |
| 6 | 4 | 110.1 | 0.6 |
| 7 | 5 | 107.3 | 0.6 |
| 8 | 6 | 89.0 | 0.6 |
| 9 | 7 | **60.4** | 0.6 |
| | spread | **7.45x** | 1.02x |

**The x read return is genuinely row-unfair, monotone, and row 0 is served LAST** -- 7.45x, with the CB
wait flat at 0.6 us. The docstring's direction was right all along for x; only its W_gate figure
(28 vs 64 us) was the contaminated one.

This makes the whole picture coherent, including the failures in 8.8:

* `kr_taper`'s direction is correct and mechanistically justified: row 0's x arrives last, so row 0
  must carry the least K work. The reproducible ~1% (8.8) is this.
* The 4.4x W_gate "asymmetry" is the **mirror** of this, not an independent effect: a reader that
  spent 450 us in `pf_x_barrier` reaches the W_gate barrier long after compute is ready, so it waits
  ~0; a reader that got x in 60 us arrives early and waits. The two profiles are anti-correlated
  because one causes the other.
* 8.8's fitted-hump splits failed because **the hump was the artifact profile**. The real profile is
  monotone, which is exactly the family `kr_taper` generates -- and monotone ramps do work
  (forward > even > reversed).
* `kr_pad = max(kr_sizes)` still caps how much correction is affordable, which is why +-2 wins and
  +-4 (kr_pad 32) over-pays by 3.4-4.5%.

**Why x and not the weights?** x is DRAM-**interleaved** bf16 ROW_MAJOR sticks (1792 B) and each grid
row reads a different emb slice, so a row's pages can concentrate on a subset of banks; the weights are
**ND-sharded** (one tile-row x per-core-N), which spreads every row's tiles across banks by
construction. That is a data-placement hypothesis, not yet a measurement -- the clean test is to
re-probe `pf_x_barrier` with x in a tile/ND layout and see whether the 7.45x collapses.

**Instrument lesson.** A zone that contains a `cb_reserve_back` measures the consumer, not the
producer. Any per-core "when did my data arrive" probe must bracket the barrier alone. Both misleading
claims in this op's history came from zones that violated that.

Tooling added: `routed_expert_work/rowprobe.py` (per-row aggregation of a named zone from a
device-profiler CSV), `routed_expert_work/rowprobe_run.sh` (capture one K split; `GC=` overrides
`GU_CHUNKS`), `routed_expert_work/test_row_asym.py` (stock-op `ttnn.clone` per-row DRAM probe with a
core-grid sweep, independent of this op).

**Independent stock-op confirmation.** `ttnn.clone` on a 64 MB DRAM-interleaved bf16 tile tensor gives
every core the same page count with no cross-core reduce and no CB coupling to anything but its own
writer. Per-core NCRISC (reader) kernel duration, 110 cores, all ten worker rows:

| phys y | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| mean us | 63.0 | 63.4 | 63.5 | 64.7 | 65.6 | 65.1 | 65.6 | 65.9 | 65.4 | 61.4 |

Spread 1.07x. The simple case does not reproduce the hump -- which is the result: there is no
row-dependent DRAM service rate on this part, at any core count that matters here.

### 8.10 Fitting the split to the REAL (x) profile beats the taper family — ACCEPTED

8.9 says the true per-row profile is the monotone `pf_x_barrier` one, steepest at row 0
(449.9 / 340.3 / 247.8 / 140.0 / 110.1 / 107.3 / 89.0 / 60.4 us). `kr_taper` can only generate the
symmetric family `{-t,-t,-t/2,0,0,+t/2,+t,+t}`, which is too gentle at row 0 and wastes shift on row 1.
Explicit `KR_SPLIT` lists fitted to the measured profile, all summing to 224 with max 30 so `kr_pad`
(and therefore L1) is identical to taper=2:

| split | deltas from 28 | M=256 | M=1024 |
|---|---|---:|---:|
| taper=2 `26,26,27,28,28,29,30,30` | -2,-2,-1,0,0,+1,+2,+2 | 762,924 | 2,399,885 |
| **fitB `24,26,27,28,29,30,30,30`** | **-4,-2,-1,0,+1,+2,+2,+2** | **744,629 (+2.5%)** | **2,383,175 (+0.7%)** |
| fitE `23,26,27,28,30,30,30,30` | -5,-2,-1,0,+2,+2,+2,+2 | 747,031 (+2.1%) | 2,392,741 (+0.3%) |
| fitF `22,26,28,29,29,30,30,30` | -6,-2,0,+1,+1,+2,+2,+2 | 759,343 (+0.5%) | 2,398,294 (+0.1%) |

Row 0 wants about -4; -5 is slightly worse and -6 gives most of the win back, so the optimum is a
genuine interior one. Validated on two fresh sessions across the full M range:

| M | taper=2 reps (us) | fitB reps (us) | fitB/taper2 |
|---:|---|---|---:|
| 64 | 500.9, 500.6 | 499.0, 501.3 | 1.001 |
| 256 | 763.2, 766.5 | 747.0, 746.5 | **1.024** |
| 512 | 1323.4, 1324.7 | 1301.2, 1302.1 | **1.017** |
| 1024 | 2399.6, 2398.0 | 2384.8, 2382.3 | 1.006 |

Reproducible (first pair gave +2.5% / +0.7%), non-overlapping at M=256/512, free in L1, and null at
M=64 like every other row-balance knob. **Recommend `kr_split = {24,26,27,28,29,30,30,30}` as the
default at emb 7168 / 11x8**, which needs the delta table replaced rather than a new `kr_taper` value
-- `-4,-2,-1,0,+1,+2,+2,+2` is not in the symmetric family the knob can express. The shipped default
`kr_taper = 4` remains the worst of everything measured (kr_pad 32, +3.4-4.5%).

Caveat: the fit is to a measurement at emb 7168 on an 11x8 grid with ND-sharded weights and
interleaved bf16 RM x. It should be re-derived, not transplanted, for a different grid or x layout --
and if the 8.9 bank-concentration hypothesis is right, an ND/tile x layout could flatten the profile
and make the whole knob unnecessary.

## 9. Branch cleanup: the winning configuration IS the code (2026-09-07)

Every experimental switch this work introduced is gone; what measured best is now unconditional.

**Removed**
* `kr_taper` / `kr_split` and both env vars. 8.8/8.9/8.10 establish that the row imbalance is real (the
  x stream, 7.45x) but that correcting it needs a per-row tile table fitted to one grid, one embedding
  and one x layout, with `kr_pad = max(kr_sizes)` taxing all 88 cores for the privilege. The split is
  now even. Cost vs the best fitted list: ~2.4% at M=256, ~1.7% at M=512, ~0.6% at M=1024, nothing at
  M=64. `24,26,27,28,29,30,30,30` is recorded in 8.10 if it is ever wanted back as a tuned constant.
* The `MOE_ABL_NO_WG` / `MOE_ABL_NO_WU` / `MOE_ABL_NO_WD` perf ablations (they deliberately compute
  wrong results; they existed to size the streams and have served their purpose).
* The W_gate prefetch mode switch (`MOE_WG_PREFETCH`, modes 1-9) -> the winning schedule inlined:
  pipelined against compute's chunk pops, blocking on chunk 0 and polling the rest.
* The W_up prefetch mode switch (`MOE_WU_PREFETCH`, modes 1-3) -> the winning single shot inlined.
* The whole W_down prefetch (`MOE_WD_PREFETCH`), its `WD_PF_TRID`, its `next_m_blocks` phase-0 peek,
  and the two-slot `cb_w_down` machinery (`MOE_WD_SLOTS`, host `wd_slots`, `WD_SLOT_TILES`, the parity
  `wd_base`). Rejected in 8.2: the shard is consumed through phase 2, so no slot is ever free, and the
  second slot costs L1 the gate/up prefetch does not.
* 324 MB of committed benchmark JSONL, 604 MB of CI logs, ~200 scratch `.log` files and six one-off
  report scripts. `routed_expert_work/{results,pf_zones}/`, `*.log` and `__pycache__` are gitignored --
  the findings are in this file and the raw data regenerates.

**Changed**
* Nothing in the defaults. `GU_CHUNKS` 3 -> 2 was tried and REVERTED: it is worth 1.2-2.5% with the
  prefetch on (8.4), but it pushes the program past the 70,656 B TENSIX kernel-config buffer on the
  INTERLEAVED-weight path at kimi dims -- 70,688 B and 70,896 B on the ragged multi-expert configs,
  4/4 undispatchable, while gc=3 passes 4/4. It fits with ND-sharded weights (the 8-expert bench runs
  clean), which is why the perf sweeps never saw it: they were all ND. Left as
  `MOE_FUSED_SWIGLU_GU_CHUNKS=2`, worth setting on the ND path.

**Kept**
* `MOE_PF_PROFILE`, including the `pf_x_reserve` / `pf_x_barrier` split from 8.9. It is off by default,
  it is the only profiling that fits in the kernel-config buffer at EXPERTS_PER_CHIP >= 2, and it is
  what distinguishes a producer wait from a consumer wait -- the mistake that cost this work two
  wrong conclusions.

**Shipped perf, defaults only, no environment overrides** (8 experts, distinct ND-sharded bfp4 weights,
x bf16 RM, 11x8, RT profiler, 7 iters):

| M | default (gc=3) | with `GU_CHUNKS=2` |
|---:|---:|---:|
| 64 | 505,093 | 497,212 |
| 256 | 776,179 | 774,281 |
| 512 | 1,348,779 | 1,328,820 |
| 1024 | 2,486,179 | 2,423,556 |

The gc=2 column matches the hand-tuned `KR_TAPER=0 GU_CHUNKS=2 MOE_WG_PREFETCH=8 MOE_WU_PREFETCH=1`
numbers (499/774/1328/2426) to within run noise, so the only tuning left worth doing is that one knob,
and only on the ND-sharded path where it fits.

**Verification**: shipped functional suite 74/74; ragged multi-expert prefetch gate 4/4 with PCC
matching pre-cleanup to ~1e-6 (`routed_expert_work/test_prefetch_ragged.py` -- the shipped suite is
single-expert and never enters the prefetch path, which is why that file exists).

## 10. Per-expert scaling, bf16 intermediate (2026-09-07)

Expert count x M, ND-sharded bfp4 weights, distinct per-expert weights, x bf16 RM, 11x8, bf16
partials, median of 5. Each expert gets its own count of M tokens, so per-expert = total / experts.

**Both `GU_CHUNKS` settings, because the optimum moves with expert count.** At E=1 there is no next
expert, so the prefetch is inert and gc=3 (the pre-prefetch optimum) is right; as E grows the prefetch
starts paying and gc=2 takes over.

Per expert (us), gc=2 / gc=3, and the best of the two:

| M | e=1 | e=2 | e=4 | e=8 | e=16 |
|---:|---:|---:|---:|---:|---:|
| 64 | 74.5 / 75.9 | 70.2 / 72.9 | 65.1 / 69.2 | 62.4 / 64.4 | 61.0 / 61.5 |
| 256 | 116.8 / **104.9** | 106.4 / **103.0** | 98.9 / 99.6 | 96.4 / 97.0 | 95.0 / 96.4 |
| 512 | 191.1 / **187.1** | 179.1 / **176.5** | 170.9 / **170.1** | 167.1 / 168.8 | 165.4 / 167.2 |
| 1024 | 330.2 / 329.1 | 316.1 / 320.2 | 308.4 / 313.2 | 304.4 / 311.0 | 302.7 / 309.9 |
| 2048 | 606.5 / 616.6 | 589.1 / 604.1 | 582.3 / 599.3 | 578.8 / 597.3 | 576.8 / 595.8 |

gc=2 costs **11.3% at E=1, M=256** and 3.2% at E=2; it wins by 1.5-3.3% everywhere from e=4 up and at
both ends of the M range. Do not ship gc=2 for single-expert dispatches.

Marginal cost of one more expert (linear fit of total vs experts, best-of data): 59.9 / 94.1 / 163.9 /
300.9 / 575.0 us at M = 64 / 256 / 512 / 1024 / 2048, with 16-30 us of M-independent fixed cost.
Within 1% of the bfp8 sweep's 60.0 / 94.1 / 162.1 / 299.7 / 573.5 -- **the intermediate format does not
move the steady-state per-expert cost.**

### 10.1 The taper removal DID cost ~4% at E=1 -- 8.8's null is specific to the 8-expert regime

Section 7's ND single-expert numbers were measured with the then-default `kr_taper = 4`. Re-running the
same configuration now (ND, E=1, gc=3, bfp8 intermediate) with the taper deleted isolates its worth:

| M | WORKLOG 7 (taper=4) | now (even split) | cost of removal |
|---:|---:|---:|---:|
| 64 | 76.5 | 76.4 | -0.1% (null) |
| 256 | 100.7 | 105.0 | **+4.3%** |
| 512 | 178.2 | 184.5 | **+3.5%** |
| 1024 | 317.9 | 330.6 | **+4.0%** |
| 2048 | 597.9 | 617.8 | **+3.3%** |

That reproduces 3.18's original "-4% at M=256" claim exactly, and it means 8.8/8.10's "the taper is
worth nothing measurable" is true ONLY with the prefetch on and 8 experts -- the regime those sweeps
ran in. At E=1 the prefetch is inert, the weight stream is unhidden, and the taper's original
justification applies. Production (kimi, 8 local experts) is the 8-expert regime, so the shipped
even split is still the right default; but the knob was not worthless, it was worthless *given the
prefetch*, and that distinction was missing from 8.8.

This also fully accounts for the 116 us E=1/M=256 figure first reported: 100.7 -> 105.0 (taper removal,
+4.3%) -> 116.8 (gc=2 at E=1, +11.3%).

### 10.2 bf16 vs bfp8 partials is a wash

Same config (ND, E=1, gc=3), M = 64 / 256 / 512 / 1024 / 2048:
bfp8 76.4 / 105.0 / 184.5 / 330.6 / 617.8 vs bf16 75.9 / 104.9 / 187.1 / 329.1 / 616.6 us.
Within +-1.4% either way, no consistent winner -- matching section 6's interleaved result (111.4 bf16
vs 111.8 bfp8 at M=256). Choose the intermediate format on accuracy, not speed.

## 11. Ragged per-expert counts (2026-09-07)

Real routing does not give every expert the same count. `BENCH_COUNTS` (new) takes an explicit
per-expert list; grading uses the last non-zero expert, since all experts write the same output rows.
Patterns and their predicted block counts are in `routed_expert_work/ragged_sweep.py`.

The op quantises each expert to `ceil(count/32)` tiles and then to `ceil(tiles/M_BLOCK)` blocks with
`M_BLOCK = 8`, i.e. **256 tokens**. The patterns below separate tokens, blocks and non-zero experts.
ND-sharded bfp4 weights, bf16 partials, gc=2, 8 local experts, 11x8, median of 5. PCC 0.979 on every
pattern.

### 11.1 Total fixed at 2048 tokens, distribution varied

| pattern | counts | nz | blocks | total us | ns/token | vs balanced |
|---|---|---:|---:|---:|---:|---:|
| balanced | 256 x8 | 8 | 8 | **768.6** | 375.3 | 1.000x |
| mild_25pct | 192..304 | 8 | 12 | 888.1 | 433.6 | **0.865x** |
| moderate_3x | 128..384 | 8 | 12 | 898.6 | 438.8 | **0.855x** |
| heavy_hot | 1024,256x3,64x4 | 8 | 11 | 845.4 | 412.8 | 0.909x |
| zipf | 757,376,251,188,150,125,107,94 | 8 | 11 | 884.6 | 432.0 | 0.869x |
| sparse4 | 512 x4, 0 x4 | 4 | 8 | 701.2 | 342.4 | **1.096x** |
| sparse2 | 1024 x2, 0 x6 | 2 | 8 | 652.2 | 318.5 | **1.178x** |
| sparse1 | 2048, 0 x7 | 1 | 8 | 627.5 | 306.4 | **1.225x** |

Two effects, opposite in sign:
* **Skew costs 8.5-14.5% at identical total tokens** -- not because of the tokens but because uneven
  counts straddle block boundaries: 12 blocks instead of 8. A zipf-shaped router (the realistic case)
  costs 13.1%.
* **Sparsity PAYS.** Every zero-count expert is skipped whole, weight read included, and that read is
  the op's dominant cost at these sizes: 8 active experts -> 1 is **1.225x faster at the same 2048
  tokens**. Fewer, fuller experts beat more, emptier ones.

### 11.2 The 256-token staircase

Uniform counts stepped across one block boundary:

| pattern | tokens | blocks | total us | ns/token | vs unif_256 |
|---|---:|---:|---:|---:|---:|
| unif_224 | 1792 | 8 | 734.0 | 409.6 | 1.048x |
| unif_251 | 2008 | 8 | 766.5 | 381.7 | 1.003x |
| unif_256 | 2048 | 8 | **769.0** | 375.5 | 1.000x |
| unif_257 | 2056 | 16 | **1026.9** | 499.5 | **0.749x** |
| unif_288 | 2304 | 16 | 1024.5 | 444.7 | 0.751x |
| unif_512 | 4096 | 16 | 1339.4 | 327.0 | 0.574x |

* **Tile padding is free.** 251 vs 256 tokens per expert: 2% fewer tokens, identical time (766.5 vs
  769.0). Any count in 225..256 costs what 256 costs.
* **Block padding is a cliff.** 257 vs 256 is **+0.4% tokens for +33.5% time** -- one extra token per
  expert doubles the block count. 257 costs the same as 288, and 77% of what 512 costs.
* So per-expert cost is a **staircase with 256-token treads**. The best place to be is just under a
  multiple of 256; the worst is just over.

### 11.3 Cost model

Least squares over all 14 patterns:

    total_us = 254 + 45.9 * blocks + 15.6 * nonzero_experts

Within +-5% on 12 of 14. It misses `unif_512` by -16.9%, and that miss is informative: blocks are NOT
equal cost. Against the `unif_256` baseline, the 8 extra nearly-empty blocks of `unif_257` cost
(1026.9-769.0)/8 = 32 us each, while the 8 extra FULL blocks of `unif_512` cost
(1339.4-769.0)/8 = 71 us each. **A partial block costs ~45% of a full one** -- it is not free, which is
exactly why the 257 cliff hurts, but it is not a full block either.

### 11.4 Consequence for section 10

Section 10's per-expert table is the BALANCED case and is therefore optimistic. With all 8 experts
active and a realistic zipf-shaped router, add ~13% to the e=8 column. The pessimistic corner is
uniform counts one token over a block boundary (+33%); the optimistic one is a sparse dispatch where
most experts are empty (-18% at 2 active experts).

## 12. Widening the x-stage protection to every cold block -- ACCEPTED (2026-09-07)

Chasing the ~16 us/expert of "late x" above the 77.7 us compute floor (3.x), at E=1 where the weight
stream is fully exposed in phase 1 (the prefetch has no next expert and is inert). ND weights, bf16
partials, gc=3, M in {256, 512, 1024}.

### 12.1 What did NOT work, and why the obvious reading was backwards

| config | M=256 | M=512 |
|---|---:|---:|
| base (`WD_SPLIT=3`) | **105.0** | 187.1 |
| `WG_AFTER_X=1` | 114.4 (-8.2%) | 190.6 (-1.9%) |
| `WD_SPLIT=0` | 103.9 (+1.0%) | 187.3 (-0.1%) |
| `WD_SPLIT=5` | 106.0 (-1.0%) | 182.0 (+2.8%) |
| `WD_SPLIT=8` | 115.7 (-9.3%) | 192.4 (-2.7%) |
| `WG_AFTER_X=1` + `WD_SPLIT=8` | 116.2 (-9.7%) | 187.7 (-0.3%) |

Deferring W_gate past x costs **8.2%**: the weight read is not starving x, it is FILLING DRAM that
would otherwise idle during x staging, and delaying it just makes compute wait. `WD_SPLIT=8` losing
9.3% says NoC1 is not a free resource either -- `XPRIO` already defers W_up there, so adding W_down
builds a NoC1 backlog that phase 2 waits on. (`WD_SPLIT=5` at M=512 and `WD_SPLIT=0` at M=256 are
1-3% and M-dependent; not pursued, they need their own per-M story.)

### 12.2 The actual gap: `protect_x_stage` was gated to `m_blocks == 1`

`protect_x_stage` releases W_gate inside multicast round 0, only after every core in the row has
staged x, so an early core cannot starve a lagging core on NoC0. It was restricted to
`m_blocks == 1` on the reasoning that multi-block work has its later x prefetched -- true, but
**block 0 never is** (`prefetch_next_x` requires `block_idx + 1 < m_blocks`). So at m_blocks > 1,
block 0 staged x cold with W_gate already in flight: exactly the case the mechanism exists for.
`!staged_early` alone is the correct condition and is what ships now.

| M | m_blocks | before | after | speedup |
|---:|---:|---:|---:|---:|
| 256 | 1 | 105.0, 105.3, 104.1 | 105.3, 104.7, 104.0 | 1.003x (null -- already protected) |
| 512 | 2 | 187.4, 184.1, 186.5 | 181.4, 181.1, 181.2 | **1.029x** |
| 1024 | 4 | 329.0, 328.9, 326.8 | 323.7, 323.5, 323.1 | **1.017x** |

Non-overlapping at M=512 and M=1024, and null at M=256 -- the exact signature the mechanism predicts,
since M=256 was already protected. At **8 experts** it is 1.001x / 1.005x / 1.002x: neutral, no
regression, and the small size is itself consistent -- with the prefetch the weight stream is largely
already out of phase 1, so there is less interference left to remove.

### 12.3 It lowers the curve, it does not flatten it

`pf_x_barrier` per row at M=512, E=1, barrier only:

| phys y | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | spread |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| before | 21.5 | **32.5** | 23.3 | 18.4 | 19.5 | 14.1 | 10.5 | 6.9 | 4.71x |
| after | 19.4 | **24.0** | 18.7 | 14.7 | 13.6 | 9.7 | 7.0 | 5.1 | 4.66x |

Every row improves and the slowest drops 32.5 -> 24.0 us, but the **spread is unchanged**. So this
removes weight-stream interference from the x stream; it does not touch x's own inter-row arbitration,
which stays at ~4.7x and remains the open item. 8.5 us off the slowest row for 4.6 us end-to-end at
M=512 is the expected ratio -- not all of it is on the critical path, and only block 0 of two is
protected.

Verified 78/78 (shipped suite + ragged multi-expert gate) on the default build.

### 12.4 Also settled here: the 8+1 vs 5+4 block split question

Splitting m_t = 9 as 5+4 instead of 8+1 would be **worse**, not better. `m_tiles_eff` rounds the last
block's remainder up to a POWER OF TWO (`while (p < rem) p <<= 1`), so with `m_eff_min = 1` the legal
sizes are {1, 2, 4, 8}: 8+1 charges 9 m_eff units, 5+4 charges 8+4 = 12. The FPU work is
`m_tiles_real` (9 either way, the real rows are a contiguous prefix) but every CB reserve/push/pop and
the whole reduce-scatter slice plan are in m_eff units, so a balanced split moves 33% more data
through the reduce for identical arithmetic. The pow2 ladder is load-bearing: the slice plan must
divide M_BLOCK and agree across cores without communication.

The real cost of m_t = 9 is the second block's FIXED cost, ~26 us (section 11.3: partial block 32 us,
full 71 us). That is a per-block-overhead problem, not a tile-distribution one.
