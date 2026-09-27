# Flat streamed routed expert: generalization / raggedness work log

Op under test: `test_stream_expert_flat.py` (+ `kernels/stream_mm/se*`), one Blackhole p150a, bfp4 weights, LoFi.
"Hard case" = row-major bf16 dispatch buffer in DRAM (`MIMO_FL_DYN=1`, implies E2E), per-expert token counts read
on device, output bfp8 tiles at each expert's region.

Timing: Tracy device-kernel duration between signposts, mean of 3 iterations (`analyze_tags.py`). "us" below is
the whole launch (all local experts), not per expert, unless marked.

## 2026-09-27

### Setup
- Workload reference: Kimi K2.6 routing dump (artifact ad3ca641, 60 layers x 11 chunks x 384 experts, 5120 tok/chunk,
  top-8, 32 chips x 12 local experts). Per chip: mean 1280 tokens, p50 1153, p90 1910, p99 3442, max 6139; per
  expert: mean 107, p50 69, p99 756, max 3775; within a chip max/mean expert 3.1x (p50), 5.7x (p90), 8.8x (p99).
  Program capacity must cover the max -> built with M (capacity) = 4096, E = 12.
- Harness changes (DYN mode): count sets (`MIMO_FL_COUNTS "a,b;c,d"` or `MIMO_FL_COUNTS_FILE` json
  [[label, counts]]) run back to back on one program (same buffers, contents rewritten); regions packed like a real
  dispatch buffer (32-row aligned prefix sums) instead of `e * capacity`; pre-tiled x no longer built in E2E mode;
  PCC checked on up to 160 rows at each end of every expert (`MIMO_FL_CHECK_ROWS`).
- Shapes survey (repo): K2.6/K2.7/DSv3 7168x2048 SiLU; Kimi K3 3584x3072 SiTU-GLU; MiniMax-M3 6144x3072 swigluoai;
  MiniMax-M2.7 3072x1536; GLM-5.x 6144x2048; DSv4-Flash 4096x2048 / Pro 7168x3072 clamped_silu_glu;
  Mistral-S4 4096x2048; Qwen3.5-397B 4096x1024; GPT-OSS 2880x2880 swigluoai + biases; Gemma-4 26B 2816x704 GeGLU.
  Activation reference math: unified_routed_expert_ffn/.../compute/fused_swiglu.cpp.

### Smoke (K2 7168x2048, E=12, capacity 4096)
| counts | tokens | us |
|---|---|---|
| balanced 12 x ~107 | 1280 | 825.2 |
| real p50 chip [74,196,148,67,65,94,330,65,34,41,10,29] | 1153 | 836.5 |
PCC min 0.9943 (quantized-weight reference).

### Baseline ragged sweep (K2, E=12, capacity 4096, before any change)
Real chips from the K2.6 dump (tot-pN = chip at the N-th percentile of total tokens, max-pN = of its largest expert),
each vs a "balanced twin" (same total spread evenly over the 12 experts), plus uniform / synthetic sets.

| set | counts | tokens | ragged us | balanced us | ragged / bal |
|---|---|---|---|---|---|
| tot-p50 | 74,196,148,67,65,94,330,65,34,41,10,29 | 1153 | 836.9 | 814.8 | +2.7% |
| tot-p90 | 117,108,241,66,19,1000,43,40,87,48,34,107 | 1910 | 957.2 | 930.0 | +2.9% |
| tot-p99 | 2876,49,90,49,72,21,24,73,68,55,41,26 | 3444 | 1277.9 | 1146.9 | +11.4% |
| tot-p100 | 13,1050,30,30,36,117,66,29,725,3674,350,19 | 6139 | 1714.2 | 1469.4 | +16.7% |
| max-p50 | 93,46,19,39,289,91,43,84,4,50,94,94 | 946 | 810.5 | 801.0 | +1.2% |
| max-p99 | 314,23,31,157,8,85,151,42,17,2320,33,77 | 3258 | 1245.5 | 1150.4 | +8.3% |
| max-p100 | 12,1189,11,47,25,65,73,50,468,3775,104,6 | 5825 | 1674.1 | 1460.1 | +14.7% |

Uniform (12 x M): M 32/64/128/192/256/384/512/1024 = 770/796/839/945/1001/1199/1463/2482 us -> per expert 64 us at
the weight floor (24.8 MB bf4 at ~390 GB/s), incremental 0.166 us/token above M=512 (~530 TFLOP/s, 87% of LoFi peak).
Synthetic: spike1024 [1024 + 11 x 24] 910; spike3000 [3000 + 11 x 25] 1285; two-hot [1500,1500 + 10 x 30] 1232.

Diagnosis: a big expert is compute-bound (DRAM mostly idle) while every small expert costs the full weight floor
(~64 us) with compute idle; the gate/up ring holds 2 experts so only ONE expert's weights prefetch under the big
expert. Sum-of-parts model (per-expert cost from the uniform curve) reproduces the ragged numbers within ~5%.

### Pinned schedule (se_dyn.hpp `SE_PIN_MIN`, host `MIMO_FL_PIN`)
- Schedule = list of entries (runs of sub-blocks of one expert); the biggest expert B is cut into K chunks
  interleaved with the small experts (<= `SE_PIN_SMALL` = 2 sub-blocks): B1 s1 B2 s2 .. BK sK .., then the larger
  experts. B's weights load once and stay in ring region 0 until BK; the small experts cycle through region 1, so
  each small expert's weights stream in while a chunk of B computes. Everything but the weight path (x relays, h,
  y, down compute counts) just sees more entries.
- Ring mechanics: loads land in regions; receivers keep granting slots by count; compute pops by count after a
  block's last use and addresses block (region, j) relative to the read pointer (popped % ring). Load l goes to the
  region of the (l - NREG)-th load to retire (retirement order = schedule order of last uses); the builder checks
  that retirement comes before the load's first use, else drops pinning (never deadlocks).
- Down weights too (`SE_DN_REG`): pinning only the gate/up weights re-read B's down weights per chunk (8.3 MB) and
  gained just 5%; pinned down rings need 2 whole experts (DRING 2.0, default with PIN) -> did not fit L1 (1434 KB
  vs 1427 KB/bank) until the e2e output double buffer was sized for bfp8 pages (was 2 KB pages: -53 KB, arena 1382 KB).
- Row tiles: a sub-block's compute (gate/up rows, down row passes) now skips row tiles past the token count (the
  last sub-block of each entry: `SE_META_LMT`); small experts cost 1-3 row tiles instead of 4.
- Profiling (p99 chip): gate/up MM and down MM both ~24 us per full 128-row sub-block (~18 cyc/tile = LoFi peak);
  small-expert down 6 us per row tile. After pinning the p99 chip is near its DRAM floor (12 x 24.8 MB weights +
  75 MB bf16 x / bf8 y at ~420 GB/s ~ 890 us) vs 1068 measured.
- Profiler gotcha: `SE_DOWN` in se3_compute.cpp (unused down path) hash-collided with BRISC-FW once line numbers
  moved -> renamed `SE_DOWN3`.
- Tried, no gain: `MIMO_FL_DW_DELAY` 40k/80k cycles (down weights start late so expert 0's gate/up weights own DRAM
  during fill): within +-1%.

Results (PIN=1, SMALL=2; same program and capacity for every set):
| set | before | pinned | balanced twin (pinned build) | ragged / bal |
|---|---|---|---|---|
| tot-p50 | 836.9 | 821.3 | 814.8 | +0.8% |
| tot-p90 | 957.2 | 904.3 | 943.6 | -4.2% |
| tot-p99 | 1277.9 | 1068.1 | 1176.3 | -9.2% |
| tot-p100 | 1714.2 | 1590.6 | 1456.7 | +9.2% |
| max-p50 | 810.5 | 809.6 | 806.4 | +0.4% |
| max-p99 | 1245.5 | 1101.8 | 1165.7 | -5.5% |
| max-p100 | 1674.1 | 1543.1 | 1464.3 | +5.4% |
| spike1024 | 909.8 | 834.7 | | |
| spike3000 | 1285.0 | 1106.9 | | |
| two-hot | 1232.3 | 1169.5 | | |
Uniform 12 x M unchanged within noise (32: 754, 128: 845, 512: 1467, 1024: 2466). Interleaving every other expert
(not only small ones) was 1562 on tot-p100 but cost 2-3% on uniform 384/1024 (chunking between equally big experts
only adds switches) -> small-only interleave. PIN 1 vs 2 vs 3: p99 1122 / 1128 / 1136 (before row-tile skipping).
Balanced 12 x 512 profile: 23.6 us per sub-block steady; ~140 us launch fill (expert 0's gate/up weights share DRAM
with 16.6 MB of down weights), boundary jitter from DRAM at ~85-90% (weights + bf16 x + y ~375 GB/s).

### Activation functions (se3_compute.cpp `SE_ACT`, host `MIMO_FL_ACT`)
Run on the PACK thread's SFPU over the raw gate / up DST accumulators (as SiLU was, so MATH starts the next
sub-block's matmul): the unified_routed_expert_ffn variants, same SFPU functions, invoked from PACK:
`silu` (0), `swigluoai` (1, GPT-OSS / MiniMax-M3: (clamp(u,+-7)+1) g' sigmoid(1.702 g'), g' = min(g, 7), via
moe_gpt's swiglu_sfpu.h), `situ` (2, Kimi K3: 4 tanh(g/4) sigmoid(g) * 25 tanh(u/25), calculate_situ_glu),
`clamped_silu` (3, DeepSeek V4: silu(min(g,10)) clamp(u,+-10)), `gelu_tanh` (4, Gemma 4: gelu_tanh_tile_pack + mul).
Check is discriminating: the test also computes the PCC against the SiLU-GLU reference and asserts the chosen
activation's reference matches strictly better (weights scaled with `MIMO_FL_WSTD` so the clamps / tanh engage:
0.06 -> gate std ~5; GeGLU vs SiLU-GLU only separate at small gates: 0.006).
| act | min PCC (own ref) | min PCC (silu-glu ref) | uni128 us | tot-p99 us |
|---|---|---|---|---|
| silu | 0.9947 | - | 844.1 | 1093.9 |
| swigluoai | 0.9923 | 0.9021 | 844.0 | 1084.3 |
| situ | 0.9935 | 0.9328 | 848.5 | 1077.9 |
| clamped_silu | 0.9928 | 0.9751 | 843.7 | 1089.2 |
| gelu_tanh | 0.9938 | 0.9777 | 848.0 | 1080.5 |
No measurable cost (the activation hides under the next matmul). Not done: GPT-OSS gate/up/down biases.

### Shape generalization
- `MIMO_FL_I` (per-device intermediate size, I / TP) and any H with Ht % 8 == 0.
- Gate/up split `_gu_split`: NP column pairs per gate/up core x G M-groups (group g computes row tiles
  g*MT/G.. of every sub-block; the x block still carries all MT rows, compute starts at row offset g*MT/G). The
  It/NP pair-sets go round the 16 readers (multiple of 16), each forwarded to its G cores (forwarder receiver j gets
  block j / G). Picks the most cores (<= 64), then the fewest pairs per core, subject to DST ((MT/G) * 2 NP <= 8)
  and gate/up L1 (2-expert ring of the pairs' full-K slice + x ring <= 1400 KB; the x ring shrinks 24 -> 16 -> 12
  slots first). Unused gate/up cores idle inside the multicast rectangles (arena allocated there).
  I 2048: NP1 G1 (64 cores); 1024: NP1 G2; 512: NP1 G4; 3072 @ H 6144/7168: NP2 G1 (48 cores, MT 2);
  3072 @ H 3584: NP3 G2 (64, MT 2); 1536: NP1 G1 (48 cores).
- Relays: super-block width `SE_SBT` (K tiles per row-major chunk: 32, 16 for H 3584).
- Down: HBUF drops to 2 when 3 h buffers + the (pinned, 2-expert) down ring overflow L1 (I 3072).
- Bugs found on the way: (1) se5_recv read RT 14 both as the M-group and as the old pair role (`is_b`), so group-1
  cores never granted weight credits -> hang (DPRINT of receiver/forwarder counters: credits 56 0 56 0);
  (2) the forwarder sent a whole block as one NoC packet; NP 2 blocks are 18 KB > NOC_MAX_BURST_SIZE (16 KB) -> hang;
  now bursts under one trid; (3) my own L1 check at 1300 KB silently moved K2 from 24 to 16 x slots (fixed: 1400).
- Also fixed: an indentation slip in the e2e check block that had skipped the per-expert PCC assert for SiLU runs
  (introduced with the activation check; only two H-sweep runs affected, rerun).

### TP shapes: M-groups vs fewer cores; x path
- G > 1 re-forwards every gate/up weight block to G cores. G = 1 on fewer cores (K2 TP2: 32, TP4: 16) is faster at
  small M and equal at large M (x-path bound anyway), 8 experts, us:
  | shape | G>1 uni32 / 128 / 512 / rag | G=1 uni32 / 128 / 512 / rag |
  |---|---|---|
  | K2 TP2 7168x1024 | 353 / 398 / 826 / 612 (G2) | 283 / 341 / 826 / 603 |
  | K2 TP4 7168x512 | 306 / 343 / 807 / 584 (G4) | 183 / 237 / 794 / 560 |
  | K3 3584x3072 | 453 / 540 / 1130 / 887 (NP3 G2) | 403 / 464 / 988 / 752 (NP2, 48 cores) |
  -> `_gu_split` now prefers the fewest M-groups.
- Large-M TP profile (TP4, 8 x 512, 794 us): gate/up and down compute ~6 us per 128-row sub-block but the pipeline
  runs at ~25 us; the relays' tilizer is the limit: `fast_tilize_block` of 32 tiles to bfp8 ~2400-2800 cycles
  (~80 cyc/tile), each relay (primary + helper per rectangle) tilizes 448 tiles per sub-block, and every x tile is
  tilized twice (once per rectangle).
- Tried: shared tilizing (XSHARE: the 4 relays each tilize every 4th super-block once and write it to BOTH
  primaries' landing rings): correct (PCC ok) but 1.6-1.8x SLOWER (8 x 512: 1261 vs 794 us; 8 x 32: 333 vs 183).
  Zones: primaries wait 450-660 us per launch for landed super-blocks; the helpers' 139 KB unicasts now cross the
  chip (6 hops, NOC0, through the rows the linked multicasts reserve) instead of going to the adjacent primary. A
  deeper landing ring (6 slots, 2 own) changed nothing. Reverted.
- Re-diagnosis (the relay is not the whole story): with G = 1 each of the 16 (TP4) gate/up cores does a full
  128-row sub-block for its pair, gate AND up: 4 x 224 x 2 = 1792 tile-matmuls ~24 us -> compute-bound on 16 cores;
  with G = 4 the 4x duplicated forwarding (16.5 MB per expert over its 4 sub-blocks at M = 512) is also ~25 us.
  Probes that changed nothing at 8 x 512 (794 us): DRAM x reads off (`MIMO_FL_XRD_SKIP=1`: 791), contiguous 35 KB
  multicast pieces instead of 8.7 KB (795), h chain pieces 32 -> 8 / 4 (793 / 794), 2 helpers per rectangle (797).
- Tried: multicast forwarding for G > 1 (each block once to its G cores' 1x2 / 2x2 / 1x4 rectangle). With a
  transaction id on the multicast -> HANG (the per-id outstanding counters do not track multicast acks: confirms the
  earlier trid gotcha). Without trid (chunks complete in batches when all this RISC's writes are acked, 2-8 deep):
  correct but 2.1x slower at small M (8 x 32: 389-414 vs 183 us), ~4% faster at 512 (754 vs 794): 16 forwarders'
  path-reserved multicasts on NOC1 contend. Reverted.
- New: `MIMO_FL_XHELP_N` helpers per rectangle (se11_xmc round of 1 + NH, per-helper landing ring + arrival sem;
  2 needs `MIMO_FL_LAND_SLOTS=2` to fit L1). Best large-M TP4 point: G 2 (32 cores, 2x forwarding) + 2 helpers:
  | K2 TP4, 8 experts | uni32 | uni128 | uni512 | rag |
  |---|---|---|---|---|
  | G1 (default) | 183 | 236 | 794 | 560 |
  | G2 | 199 | 247 | 732 | 519 |
  | G2 + 2 helpers | 198 | 253 | 675 | 520 |
  | G4 + 2 helpers | 285 | 344 | 756 | 569 |
  Real routing is small-expert dominated, so G 1 stays the default; G 2 + 2 helpers is the opt-in for large M.
