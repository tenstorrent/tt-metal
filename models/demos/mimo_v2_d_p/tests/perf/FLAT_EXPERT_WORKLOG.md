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
