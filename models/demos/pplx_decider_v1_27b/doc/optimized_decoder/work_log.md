# Stage 3A work log: prefill matmul optimization

Labels: **measured** (command and value recorded here, logs under
`/local/ttuser/gtobar/artifacts/pplx_decider/stage3a/`), **inferred** (follows from code or
arithmetic), **guess**. Baseline: `78e35968f19` (stage 2 code; later stages did not touch the
text-layer matmul path). Precision is fixed: BF16 activations, BFP8 weights, HiFi2 with fp32
accumulation (`act_bf16__w_bfp8_all__hifi2`). Prefill only; decode, KV-cache and decode-trace items
of the optimize skill are N/A for this model.

## Method

- **Op level (screening).** `mm_bench.py` (scratch script, not in the repo): one projection at its
  real K x N, a random BF16 input [1, M, K] and a random BFP8 weight, both DRAM interleaved; HiFi2 +
  fp32 accumulation. Time = best of 3 x (10 back-to-back calls, one `synchronize_device`) / 10, so it
  is device time plus a small dispatch share. M = 128 (bucket 128), 1024 (bucket 1024) and 2048 (the
  prefill chunk; buckets 2048, 4096 and 8192 run 2048-row chunks). Each line also prints the PCC of
  the output against the first candidate of the line (always >= 0.99996). Logs: `stage3a/micro/*.log`.
- **Layer level (decision).** `tests/perf/test_prefill_perf.py::test_layer_and_module_perf` (stage
  1/2 harness: 2 warm-ups, median of 7, `synchronize_device` around each pass), run A1 B1 A2 B2,
  where A is `tt/common.py` + `tt/optimizations.py` of `78e35968f19` and B is the stage-3A commit.
  Each cell is the mean of the two run medians.
- **Device profile.** Tracy `test_profile_layer` (one warmed layer between `PREFILL_START` /
  `PREFILL_END`) and `tt-perf-report` with advice, at S = 128, 2048, 8192 for both layer kinds.

## 1. Operation-topology audit (before tuning)

Read from the code (`tt/mlp.py`, `tt/attention.py`, `tt/gated_deltanet.py`, `tt/common.py`) and the
stage-2 `tt-perf-report` CSVs (`doc/fused_decoder/perf/`). Shapes are per 2048-row chunk for
S >= 2048, so S=8192 is 4 chunks. Every matmul reads a DRAM-interleaved BFP8 weight and a
DRAM-interleaved BF16 activation and writes BF16 to DRAM.

**Topology.** The stage-2 topology is already fused where it pays: gate and up are one tile-pair
interleaved weight with the SwiGLU in the `minimal_matmul` epilogue; attention q|k|v is one matmul
plus a separate gate matmul; GDN q|k|v is one matmul plus a packed z|b|a matmul. Stage 2 measured
the fully packed alternatives (q|k|v|gate, q|k|v|z|b|a) and lost to the slices they need (161 + 121
and 211 us at 2048). There are no collectives (one device), no reshards around the matmuls and no
host ops. So stage 3A is per-op configuration, not topology.

**Stage-2 op and config.** `prefill_linear` used one `MinimalMatmulConfig(M_block 4, K_block 8,
N_block 16, subblock 1x4, grid 13x10)` for every projection at rows >= 512, and `ttnn.linear` with
its auto program config below 512 rows; the fused SwiGLU always used `minimal_matmul`.

Device time per layer, stage 2 (share of the layer's device time) -> stage 3A, in us:

`linear_attention` (L0)

| projection | M x K x N | stage-2 op | S=128 | S=2048 | S=8192 (4 chunks) |
|---|---|---|---|---|---|
| GDN in q\|k\|v | M x 5120 x 10240 | linear auto (128), minimal | 160 (7.1 %) -> 161 | 860 (8.2 %) -> 779 | 3443 (8.1 %) -> 3126 |
| GDN in z\|b\|a | M x 5120 x 6272 | linear auto, minimal | 148 (6.5 %) -> 98 | 510 (4.9 %) -> 503 | 2039 (4.8 %) -> 2026 |
| GDN out_proj | M x 6144 x 5120 | linear auto, minimal | 160 (7.1 %) -> 102 | 594 (5.7 %) -> 504 | 2373 (5.6 %) -> 2017 |
| MLP gate\|up, SwiGLU | M x 5120 x 34816 (-> 17408) | minimal fuse_swiglu | 815 (35.9 %) -> 792 | 2797 (26.7 %) -> 2565 | 11196 (26.2 %) -> 10275 |
| MLP down | M x 17408 x 5120 | linear auto, minimal | 438 (19.3 %) -> 264 | 1602 (15.3 %) -> 1317 | 6410 (15.0 %) -> 5268 |
| all matmuls | | | 1721 (75.9 %) -> 1417 (72.0 %) | 6362 (60.7 %) -> 5668 (58.0 %) | 25461 (59.6 %) -> 22711 (56.9 %) |
| layer device time | | | 2269 -> 1968 (-13.3 %) | 10487 -> 9773 (-6.8 %) | 42734 -> 39894 (-6.6 %) |

`full_attention` (L3)

| projection | M x K x N | stage-2 op | S=128 | S=2048 | S=8192 (4 chunks) |
|---|---|---|---|---|---|
| attention q\|k\|v | M x 5120 x 8192 | linear auto, minimal | 168 (7.9 %) -> 129 | 784 (8.5 %) -> 628 | 3068 (6.1 %) -> 2533 |
| attention gate | M x 5120 x 6144 | linear auto, minimal | 147 (6.9 %) -> 97 | 507 (5.5 %) -> 494 | 2028 (4.0 %) -> 1975 |
| attention o_proj | M x 6144 x 5120 | linear auto, minimal | 161 (7.5 %) -> 103 | 594 (6.5 %) -> 503 | 2376 (4.7 %) -> 2013 |
| MLP gate\|up, SwiGLU | M x 5120 x 34816 (-> 17408) | minimal fuse_swiglu | 815 (38.3 %) -> 760 | 2802 (30.5 %) -> 2568 | 11199 (22.2 %) -> 10274 |
| MLP down | M x 17408 x 5120 | linear auto, minimal | 437 (20.5 %) -> 267 | 1602 (17.5 %) -> 1320 | 6410 (12.7 %) -> 5272 |
| all matmuls | | | 1728 (81.1 %) -> 1356 (77.1 %) | 6289 (68.6 %) -> 5513 (65.6 %) | 25081 (49.8 %) -> 22068 (46.6 %) |
| layer device time | | | 2130 -> 1758 (-17.5 %) | 9172 -> 8399 (-8.4 %) | 50370 -> 47337 (-6.0 %) |

**Bound classification (stage 2, tt-perf-report).** S=128: every projection is weight-read bound;
`ttnn.linear` auto reached 42-69 % of DRAM peak, and the fused SwiGLU `minimal_matmul` 44 % (13
weight readers, below). S=2048: compute bound at 60-73 % of the HiFi2 FLOP peak; the down
projection and the 5120-wide outputs were lowest (60-63 %).

**Candidates per projection (from the audit).**

| candidate | applies to | dtype/fidelity constraint | action |
|---|---|---|---|
| per-shape `minimal_matmul` blocks/subblocks (N block = N tiles per core, 4x1 subblocks under fp32 dest) | every projection, M >= 1024 | none (same dtypes, accumulation order only) | swept, kept |
| 1D in0-multicast `ttnn.linear` config with explicit `in0_block_w` / `per_core_N` | every non-SwiGLU projection, M = 128 | none | swept, kept for 5 of 6 shapes |
| fused-SwiGLU `minimal_matmul` with blocks and grid for short M | gate\|up, M = 128 | none | swept, kept |
| DRAM-sharded weights + `MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig` (tt-perf-report advice) | every projection, M = 128 | weight memory config change (one copy) | measured, rejected |
| `ttnn.linear` (more weight readers) on the interleaved gate\|up weight + on-device de-interleave | gate\|up, M = 128 | none | rejected (stage 2 measured; see below) |
| DRAM-sharded weight read by `minimal_matmul` (one copy serving both paths) | gate\|up | weight memory config change | measured, rejected |
| 2D multicast `ttnn.linear` instead of `minimal_matmul` | down, M = 128 and 1024 | none | measured, rejected |
| activation input in L1 (tt-perf-report advice) | gate\|up, M = 128 | none | measured, rejected |
| SwiGLU product in L1 for short rows | gate\|up -> down, M = 128 | none | measured, rejected |
| larger M per call (MLP over 4096 rows) | gate\|up, down | none | measured at op level; not in 3A scope |
| BFP4 / LoFi | MLP | forbidden by the C0 decision (stage 8) | not tried |

## 2. Matmul sweep

All numbers **measured** with `mm_bench.py`, us per call. "cur" is the stage-2 config.
`minimal(Mb, Kb, Nb, sh, sw[, gx, gy])` is a `MinimalMatmulConfig`; `d1(per_core_N, in0_block_w)` a
`MatmulMultiCoreReuseMultiCast1DProgramConfig` (mcast_in0, grid 13x10, per_core_M = M tiles,
subblock 1 x largest divisor of per_core_N <= 4).

### M = 2048 (buckets 2048 / 4096 / 8192) and M = 1024

`minimal_matmul` puts N on the 13 grid columns and M on the 10 rows when M <= N
(`minimal_matmul_program_factory.cpp:251-259`), so each core owns ceil(N_tiles / 13) N tiles and
ceil(M_tiles / 10) M tiles. The stage-2 N block of 16 left a ragged block on every core for the
5120-wide outputs (13 tiles per core) and the 8192-wide q|k|v (20 per core). With fp32 accumulation
the dest holds 4 tiles; a 4x1 subblock beat 1x4 for the narrow-N shapes.

| projection | cur 2048 | candidates at 2048 (us) | kept | 2048 | 1024: cur -> kept |
|---|---:|---|---|---:|---|
| gate\|up SwiGLU 5120x34816 | 2810 | (8,8,8,2,2) 3182; (4,16,16,1,4) 2694; (7,16,8,1,4) 2675; (4,16,8,1,4) 2653; (7,16,12,1,4) 2636; (4,20,12,1,4) 2582; (4,32,8,1,2) 2611; (5,32,8,1,4) 2627; (4,32,8,2,2) 2953; (4,32,8,4,1) 3037; (4,32,12/14,..) and (7,32,8,..) fail (L1) | (4,32,8,1,4) | 2577 | 1621 -> 1495 |
| down 17408x5120 | 1607 | (8,8,8,2,2) 1629; (4,8,8,1,4) 1604; (7,8,14,1,2) 1479; (7,8,13,1,1) 1434; (7,16,7,1,1) 1388; (4,16,7,1,1) 1375; (4,16,13,4,1) 1329; (4,16,7,2,1) 1524; (8,16,7,4,1) 1514 | (4,16,7,4,1) | 1328 | 942 -> 771 |
| attn q\|k\|v 5120x8192 | 780 | (4,8,8,1,4) 697; (4,8,10,1,2) 646; (4,8,20,1,4) 647; (4,8,10,2,2) 715; (7,8,10,1,2) 685 | (4,8,10,4,1) | 637 | 460 -> 377 |
| GDN q\|k\|v 5120x10240 | 867 | (7,8,14,1,2) 845; (4,8,13,1,1) 828; (4,8,8,1,4) 916; (8,8,8,2,2) 1056 | (4,8,13,4,1) | 788 | 507 -> 461 |
| out proj 6144x5120 | 601 | (7,16,7,1,1) 596; (7,8,13,1,1) 556; (4,8,13,1,1) 524; (4,12,13,4,1) 500 | (4,8,13,4,1) | 508 | 362 -> 301 |
| attn gate 5120x6144 | 513 | (4,8,8,1,4) 508; (4,8,15,4,1) 488; (7,16,8,1,4) 562 | (4,8,8,4,1) | 497 | 316 -> 291 |
| GDN z\|b\|a 5120x6272 | 515 | (4,8,8,4,1) 515; (7,8,16,1,4) 562; (4,16,16,1,4) 529 | (4,8,8,1,4) | 513 | 318 -> 303 |

The z|b|a and gate rows are flat within ~1-3 %; the config is kept because it is not slower at either M.

Grid variants at M = 1024 (32 M tiles, padded to 40 on 10 rows): a 13x8 grid (4 M tiles per core,
no padding, 104 cores) gave gate|up 1493 vs 1495 and down 798 vs 804 us; no gain, not kept. A 2D
multicast `ttnn.linear` for down at M=1024 (`grid 13x8, per_core 4x13, in0_block_w 8`): 862 vs 798
us, rejected.

### M = 128 (bucket 128)

All short-row projections are weight-read bound. `ttnn.linear` auto chose `in0_block_w` 2 and
42-69 % of DRAM peak.

| projection | stage-2 op, us | candidates (us) | kept | us |
|---|---:|---|---|---:|
| down 17408x5120 | linear auto 442 | minimal (4,8,16,1,4) 430; d1(2,4) 290; d1(2,8) 272; d1(2,17) 304; d1(2,8) subblock 2x2 274; 2D grid 10x4 423 | d1(2,8) | 272 |
| out proj 6144x5120 | linear auto 165 | d1(2,4) 112; d1(2,8) 106; d1(2,12) 111; d1(2,16) 117 | d1(2,8) | 106 |
| GDN z\|b\|a 5120x6272 | linear auto 152 | d1(2,8) 107; d1(2,4) 102; d1(4,8) 106 | d1(2,4) | 102 |
| attn q\|k\|v 5120x8192 | linear auto 173 | d1(2,8) 139; d1(4,8) 137; d1(4,4) 130; d1(2,4) 134; d1(4,5) 134 | d1(4,4) | 130 |
| attn gate 5120x6144 | linear auto 151 | d1(2,8) 109; d1(2,4) 102; d1(3,8) 101 | d1(3,8) | 101 |
| GDN q\|k\|v 5120x10240 | linear auto 165 | d1(3,2) 165; d1(3,4) 167; d1(3,8) 169; d1(4,8) 182; d1(5,8) 171 | linear auto (unchanged) | 165 |
| gate\|up SwiGLU 5120x34816 | minimal (4,8,16,1,4) 821 | (4,8,16,1,4) grid 13x4 812; (1,8,16,1,4) 13x4 813; (4,32,8,1,4) 13x10 785; (4,32,8,1,4) 13x4 764; (4,32,4,1,4) 13x4 767; (4,16,8,1,4) 13x4 787; (4,32,8,1,4) 13x2 798; (4,32,2,1,2) 975 | (4,32,8,1,4), grid 13 x M tiles | 764 |

### The S=128 gate|up path (the stage-2 regression)

Cause (stage 2, re-read): for M <= N, `minimal_matmul` reads the weight on the 13 cores of one grid
row and multicasts it down the columns, so 13 cores read 189 MB. A larger K block (32) and a grid
cut to the 4 M tiles (13x4: 4 receivers per multicast instead of 10, no padded M rows) bring it from
815 to 760-792 us in the layer profile (45-47 % of DRAM peak). The remaining gap needs more weight
readers, and every path to them was measured or ruled out with the same stored weight:

- **DRAM-sharded matmul** (`tt-perf-report` advice). Contract: `matmul_device_operation.cpp:1455`,
  "currently only support in0 tensor height of tile height" (M = 32). Adapted to the contract (a
  DRAM width-sharded weight, M = 32 per call, in0 resharded to L1, output back to DRAM): down 308 us
  (32 cores) / 318 us (16 cores, 2 readers per bank) per 32-row call, against 257 us for the 1D
  config at the same M = 32, and 272 us for all 128 rows with the 1D config. The 128 bucket would
  need 4 such calls: >= 1232 us. out_proj at M = 32: 104-124 us vs 96 us. **Rejected.** It also
  has no SwiGLU epilogue, so it cannot replace the gate|up op.
- **The same DRAM-sharded weight under `minimal_matmul`** (one copy serving both paths): gate|up
  at M = 128 1208 us vs 821 us. **Rejected.**
- **`ttnn.linear` on the interleaved gate|up weight + de-interleave.** Only `minimal_matmul` has a
  SwiGLU epilogue; any other op leaves the tile-pair interleaved [128, 34816] output to split.
  Stage 2 measured this at 1.90 ms against 0.64 ms (work log 1b). A transpose/view route (output
  transposed, `[34816,128] -> [544,64,128]` is a free TILE view, then two tile slices, the mul and a
  transpose back) still moves about 22 MB on top of a >= 470 us matmul (inferred: 189 MB at
  ~400 GB/s), so it cannot beat 764 us. Not built.
- **A second, non-interleaved gate/up copy** for short rows: forbidden (about 12 GB over 64 layers).
- **Input in L1** (`tt-perf-report` advice "place input 0 in L1"): 764 us with the input already in
  L1, 769 us including the conversion. **Rejected.**

What remains is an op change: let `minimal_matmul` spread the in1 DRAM reads over the column cores
when M is short, or give the 1D / DRAM-sharded matmul a tile-pair SwiGLU epilogue. Outside this
stage's scope (`ttnn/cpp`); recorded for the backlog.

## 3. Layer-level measurements (one change per measurement)

| # | change | layer medians (ms), linear / full | decision |
|---|---|---|---|
| 1 | per-shape `minimal_matmul` configs (first table: 1x4 / 1x2 / 1x1 subblocks) + short-row 1D configs + short gate\|up config | S128 2.37 -> 2.03 / 2.18 -> 1.81; S2048 10.64 -> 10.01 / 9.26 -> 8.56 (vs the stage-2 README) | kept |
| 2 | 4x1 subblocks for down, out proj, both q\|k\|v, attn gate (minimal path only) | S1024 5.65 -> 5.59 / 4.57 -> 4.51; S2048 10.01 -> 9.91 / 8.56 -> 8.48; S8192 42.28 -> 41.74 / 51.01 -> 50.72; S128 unchanged (2.03 / 1.81) | kept |
| 3 | SwiGLU product (gate\|up output) in L1 at S=128 | two runs: MLP 1.081 -> 1.076 / 1.078 ms, layer 2.033 -> 2.033 / 2.028 (linear), 1.808 -> 1.801 / 1.823 (full) | rejected (within noise) |

Rows 1 and 2 touch disjoint row ranges per bucket (the short tables only act at S=128, the
`minimal_matmul` table only at S >= 1024), so each bucket's change is attributable.
Logs: `stage3a/perf_cand{1,2,3}.jsonl`, `stage3a/logs/perf_cand*.log`.

## 4. Before / after (interleaved A/B)

A1 B1 A2 B2 on the final commit (A = `78e35968f19` `tt/common.py` + `tt/optimizations.py`), layer
medians, mean of two runs, ms (all modules: README). Logs `stage3a/ab/perf_ab_{A,B}{1,2}.{jsonl,log}`.

| kind | S=128 | S=1024 | S=2048 | S=4096 | S=8192 |
|---|---|---|---|---|---|
| linear_attention | 2.41 -> 2.03 (-15.7 %) | 6.02 -> 5.60 (-6.9 %) | 10.63 -> 9.94 (-6.5 %) | 21.66 -> 20.28 (-6.4 %) | 44.90 -> 41.95 (-6.6 %) |
| full_attention | 2.20 -> 1.82 (-17.3 %) | 5.04 -> 4.55 (-9.6 %) | 9.25 -> 8.49 (-8.2 %) | 21.07 -> 19.54 (-7.3 %) | 54.34 -> 51.21 (-5.8 %) |

Every bucket beats stage 2 for both kinds; S=128 is below stage 1 (2.11 / 1.91 ms).

## 5. Device profiles after optimization

`doc/optimized_decoder/perf/opt_<layer>_S<len>_perf_report.{txt,csv}` (txt with advice; csv and
stacked csv). Raw Tracy output: `stage3a/tracy/opt_*`. Every capture shows 28 (linear) / 22 (full)
device ops per chunk, the same op sequence as stage 2, and 0 host ops. The matmul rows show the
kept configs (for example the S=128 rows: `in0_block_w` 8 / 4, subblock 1x2 / 1x3 / 1x4, 80 / 98 /
64 cores; the gate|up `minimal_matmul` on 52 cores = 13x4).

Remaining `tt-perf-report` advice and its disposition:

- DRAM-sharded program config (S=128 matmuls): measured and rejected (section 2).
- HiFi4 "if not FLOP-bound": fidelity is fixed by the C0 decision.
- Input 0 in L1 for the S=128 gate|up: measured and rejected (section 2).
- "No program_config specified" on the `minimal_matmul` rows: a false positive; the op takes
  `config=` (`MinimalMatmulConfig`), which is set per shape. Candidate improvement for
  `tt-perf-report`: classify `MinimalMatmulDeviceOperation` (it is reported as unclassified) and read
  its config.
- Op-to-op gaps (for example 57 us before `ChunkGdnPrepOperation` at S=128): host dispatch; the full
  model already has a traced-forward path (stage 6). Not a matmul item.

## 6. Gates

| check | command | result |
|---|---|---|
| PCC suite | `pytest models/demos/pplx_decider_v1_27b/tests/pcc -q -p no:cacheprovider` | **142 passed in 669 s, exit 0**; 124 PCC records, 0 failures, min 0.999897 (`gated_deltanet` L61 S8192, the stage-2 minimum case); vs stage 2 (123 matching keys) per-case delta -1.3e-6 .. +1.7e-7. Log `stage3a/logs/pcc_suite_final.log`, records `stage3a/pcc_suite_final.jsonl` |
| runtime fallback audit | `tests/pcc/test_runtime_audit.py` (in the suite) | 4/4 passed (L0/L3 x S128/S4096): 0 host conversions, 0 torch ops; Tracy: 0 host ops in all 6 captures |
| determinism | `tests/pcc/test_prefill_contract.py` (in the suite; bit-identical repeat) | 2/2 passed |
| e2e text | `pytest models/demos/pplx_decider_v1_27b/tests/e2e/test_model.py -q -s` | 6 passed, exit 0; **decision agreement 25/25**, readout-logit PCC min 0.99754 (bar 0.99), median 0.99996; bucket padding, determinism and on-device audits pass |
| e2e image | `pytest models/demos/pplx_decider_v1_27b/tests/e2e/test_image_model.py -q -s` | 7 passed, exit 0; **image agreement 8/8**, logit PCC min 0.99987 |
| watcher, linear_attention | `TT_METAL_WATCHER=10 TT_METAL_LOGS_PATH=stage3a/watcher/L0 pytest .../tests/pcc/test_decoder_layer.py -q -k "L0 and (S128 or S2048)"` | 2 passed, exit 0; watcher on, no disabled features; 0 error / assert / sanitize / overflow / hang lines |
| watcher, full_attention | same with `L3` (`stage3a/watcher/L3`) | 2 passed, exit 0; 0 error lines |

Logs: `stage3a/logs/e2e.log`. The image test writes its JSONs to its default
`stage12b/image_e2e/` (no env override in the test); copies are in `stage3a/e2e_image/`.

## 7. Full-model request latency

Stage-6 harness `tests/perf/test_model_perf.py::test_model_perf` (burst: median of 5 passes, each
after 10 s idle; sustained: 2 warm-ups + median of 7 back to back), run B then A on the same
loaded-weight cache. Projected = 48 x linear + 16 x full layer medians from section 4. ms.
JSON: `stage3a/model_perf_{A,B}/model_perf.json`; logs `stage3a/logs/model_perf_{A,B}.log`.

| bucket | prompt tokens | projected A -> B | device forward, burst | request, burst | device forward, sustained | request, sustained |
|---:|---:|---|---|---|---|---|
| 128 | 100 | 150.9 -> 126.6 | 146.9 -> 124.6 (-15.2 %) | 148.6 -> 126.4 (-14.9 %) | 146.8 -> 124.0 (-15.5 %) | 152.7 -> 125.1 (-18.1 %) |
| 1024 | 178 | 369.6 -> 341.6 | 363.8 -> 336.6 (-7.5 %) | 366.0 -> 338.7 (-7.4 %) | 458.1 -> 431.8 (-5.8 %) | 461.9 -> 438.5 (-5.1 %) |
| 2048 | 2006 | 658.2 -> 613.0 | 656.5 -> 610.1 (-7.1 %) | 665.1 -> 619.2 (-6.9 %) | 891.0 -> 844.5 (-5.2 %) | 903.0 -> 857.4 (-5.1 %) |
| 4096 | 3906 | 1376.8 -> 1286.1 | 1470.5 -> 1351.3 (-8.1 %) | 1481.3 -> 1366.4 (-7.8 %) | 1846.5 -> 1745.6 (-5.5 %) | 1884.5 -> 1779.3 (-5.6 %) |
| 8192 | 7928 | 3024.6 -> 2833.0 | 3459.0 -> 3226.4 (-6.7 %) | 3498.3 -> 3267.0 (-6.6 %) | 3897.1 -> 3697.6 (-5.1 %) | 3944.6 -> 3754.6 (-4.8 %) |

The A side reproduces stage 6 (burst request 148.4 / 366.0 / 665.4 / 1481.5 / 3512.9 ms). Burst
matches the projection up to 2048 (0.98-1.00); 4096 and 8192 stay above it (1.05 / 1.14), the
same long-pass slowdown stage 6 recorded.

## 8. Remaining candidates for stage 3B

- **S=128 LayerNorm**: 82 us x 2 per layer on 4 cores (8.3 % of the linear layer at S=128). A
  width-sharded or multi-core norm config should take most of it.
- **S=128 gate|up** (760-790 us, 45 % DRAM peak): needs an op change (section 2).
- **GDN chunk ops** at 2048: `ChunkGdnPrepOperation` 1164 us (11 %), `ChunkGdnScanOperation`
  730 us (7 %), `SigmoidGatedRmsNormOperation` 337 us, the conv (468 us) and its untilize (206 us).
- **SDPA** at long S: 1277 us at 2048 (14 % of the full layer) and the largest single term at 8192.
- **Residual adds and the GDN output mul** (140-171 us each at 2048, BinaryNg in DRAM).
- **Layer-level chunk of 4096 for the MLP** (decoder.py / model.py, outside 3A scope): at M=4096
  gate|up takes 5074 us vs 2 x 2578 and down 2634 vs 2 x 1325 (measured, op level): about 100 us
  per 4096 tokens per layer (1-2 % of the MLP), small.
- The 9 % M padding at 2048 rows (64 M tiles on 10 grid rows) is inherent to the 13x10 grid.
