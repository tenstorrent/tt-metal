# Stage 3A work log: prefill matmul optimization

Labels: **measured** (command and value recorded here, logs under
`$PPLX_DECIDER_ARTIFACTS/stage3a/`), **inferred** (follows from code or
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
`stage12b/image_e2e/` (no env override in the test); copies are in `stage3a/e2e_image/`. (Correction, stage 3B: the test did read `$PPLX_DECIDER_STAGE12B_DIR`; 3B adds the stage-neutral `$PPLX_DECIDER_IMAGE_E2E_DIR`.)

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

---

# Stage 3B work log: non-matmul ops, chunking

Baseline: `57f76869913` (stage 3A, reviewer clean-pass). Precision unchanged (C0: BF16 activations,
BFP8 weights, HiFi2 + fp32 accumulation for every matmul, fp32 DeltaNet state); only SDPA / norm
compute configs, op program configs, chunking and op choice changed. Artifacts:
`$PPLX_DECIDER_ARTIFACTS/stage3b/` (`bench/*.jsonl`, `logs/`, `ab/`, `tracy/`,
`scripts/`).

## 9. Method

- **Candidate bench (decision).** `scripts/bench3b.py` (scratch, copied to the artifacts): one real
  layer (L0 linear or L3 full, real weights, real-prompt input) is built once; each candidate is one
  field of `Optimizations` toggled and reset between measurements. Per candidate and length: 2
  warm-ups, median of 7 passes (`synchronize_device` around each pass, the stage-1 harness method),
  rounds interleaved (base, c1, c2, ..., base, c1, ...), 5 s idle before every candidate. Round 0
  also records the PCC of the layer output and of the mixer output (input-normed golden in) against
  the HF fp32 goldens. Records: `bench/bench.jsonl`.
- **Why the idle gap.** Without it the S=8192 full layer drifted upward within a round (base 50.45
  ms, then 51.9 / 57.3 / 60.2 / 65.8 ms for later candidates, `logs/bench_sdpa1.log`), the
  pattern the 3A reviewer saw on the A side. With 5 s idle, repeated base medians agree within
  0.06 ms at 8192 and 0.04 ms at 2048. Inferred: thermal / power throttling under sustained load.
- **Layer stack.** For S=128 changes a single synced layer is partly host-dispatch bound (Tracy
  op-to-op gaps of 18-35 us around short ops), so S=128 candidates were also run as 8 layer passes
  back to back without a sync (`BENCH_STACK=8`, `bench/bench_stack.jsonl`), closer to the model.
- **A/B and gates** as stage 3A: `tests/perf/test_prefill_perf.py::test_layer_and_module_perf`, A1 B1
  A2 B2 with A = a worktree of `57f76869913`.

## 10. Topology audit of the non-matmul ops (3A captures, `doc/optimized_decoder/perf/opt_*`)

Device time per layer in us (share of the layer). S=8192 is 4 chunks of 2048 in 3A.

| op | S=128 L0 / L3 | S=2048 L0 / L3 | S=8192 L0 / L3 | candidate | action |
|---|---|---|---|---|---|
| SDPA (chunked, paged cache) | - / 54 (3.1 %) | - / 1276 (15.2 %) | - / 18023 (38.1 %; chunks 1276 / 3426 / 5578 / 7743) | program config (q/k chunks, exp approx, fidelity, cores per head); plain SDPA; whole-request chunk | kept: exp approx, whole request, plain SDPA + concat heads, q 256 from 2048 rows |
| ChunkGdnPrep + ChunkGdnScan (phased) | 103 + 66 (8.6 %) | 1145 + 730 (19.2 %) | 4583 + 2913 (18.8 %) | fused prep->scan program config, geometry | kept: fused program |
| QkvCausalConv1dSilu | 59 (3.0 %) | 473 (4.8 %) | 1870 (4.7 %) | channel chunk | kept: 512 |
| Untilize (conv input to ROW_MAJOR) | 40 (2.0 %) | 212 (2.2 %) | 835 (2.1 %) | none: the conv op requires ROW_MAJOR input (`qkv_causal_conv1d_silu_device_operation.cpp:34`) | op limit |
| RMSNorm input / post (interleaved, 1 core per tile row) | 2 x 82 (8.3 %) / 2 x 75 | 2 x 192 (3.9 %) | 8 x 192 (3.8 %) | sharded norm at short S; compute config | kept: block-sharded at <= 128 rows; no fp32 dest acc |
| q / k head RMSNorm (L3) | 2 x ~16 | 134 + 27 | 4 x (134 + 27) | compute config | kept (same norm config) |
| SigmoidGatedRmsNorm + mul by z | 32 + 14 | 328 + 170 | 1304 + 680 | none in the op API (fixed fused kernel; the `* z` completes silu) | op limit |
| residual adds (BinaryNg, DRAM) | 2 x 10 | 2 x 139 (each ~450 GB/s, ~88 % of DRAM peak) | 8 x 139 | in-place output; fusion into the matmul epilogue | in-place measured, rejected; `minimal_matmul` has no residual epilogue (bias / activation / SwiGLU only) |
| z slice out of z\|b\|a | 9 | 123 | 4 x 123 | split z and b\|a into two matmuls | measured, rejected |
| ReshapeView x2 (g, beta inside `chunk_gated_delta_rule`, `[48,T] -> [48,NC,32,1]`) | 2 x 10 | 2 x 112 | 8 x 112 | none from the caller (inside the op launch, `chunk_gated_delta_rule.cpp:238`) | op limit, backlog |
| RotaryEmbeddingLlama q / k (L3) | 76 | 232 + 32 | 4 x 264 | none tried (one fused op already) | - |
| NlpCreateHeads / NLPConcatHeads (L3) | 25 / 21 | 161 / 131 | 640 / 521 | SDPA `output_concat_heads` | kept (concat heads removed) |
| PagedFillCache x2 (L3) | 5 | 44 | 177 | skip when no later chunk reads the cache | kept |
| chunk slices + output concat | 0 | 0 | L0 ~400 + 401, L3 ~400 + 395 | whole-request chunk (L3) | kept for L3; L0 measured slower with longer chunks |
| typecast / transpose / small slices | < 15 | < 25 | < 100 | - | - |

There are no collectives (one device) and no host ops (0 in all six 3A captures).

## 11. Candidates, one change per measurement (measured)

Layer medians in ms, two interleaved rounds (r0 / r1) unless noted; "base" is the current best
at the time of the run, so rows build on each other in the order listed. PCC = layer output vs HF
(mixer output in brackets), from round 0. Logs: `logs/bench_*.log`; records `bench/bench.jsonl`.

### Full-attention layer (L3)

| # | candidate | S=128 | S=1024 | S=2048 | S=4096 | S=8192 | PCC (8192) | decision |
|---|---|---|---|---|---|---|---|---|
| 1 | base: 3A config (chunked SDPA, q/k 128/128, HiFi4 + fp32 acc, 2048 chunks) | | | 8.64 / 8.65 | | 50.44 / 50.50 | 0.999993 (0.999956) | |
| 1a | SDPA q/k 256/256, 256/512, 512/512 | | | L1 overflow | | L1 overflow (CBs 1.68 / 2.46 / 3.76 MB > 1.57 MB) | | rejected (illegal, head_dim 256) |
| 1b | SDPA q/k 128/256 | | | 8.72 / 8.70 | | 50.09 (r0) | 0.999993 | rejected (flat / slower) |
| 1c | SDPA q/k 128/64, 64/128, 256/64, 64/256 | | | 8.79, 9.41, 8.81, 9.52 | | 52.2, 61.6, 52.0, 62.1 | | rejected (slower) |
| 1d | **SDPA exp approx** | | | **8.51 / 8.54** | | **47.53 / 47.60** | 0.999993 (0.999955) | **kept** (-2.9 ms at 8192) |
| 1e | SDPA HiFi2 (+ approx) | | | 8.51 / 8.48 | | 47.49 / 47.59 | 0.999993 (0.999954) | rejected (= approx alone) |
| 1f | SDPA LoFi + approx | | | 8.52 / 8.51 | | 47.52 / 47.53 | 0.999946 (0.999658) | rejected (no gain, PCC drop) |
| 1g | SDPA HiFi2 / HiFi4 without fp32 dest acc | | | 8.47-8.50 | | 47.49-47.56 | 0.999993 | rejected (no gain) |
| 1h | plain causal SDPA for the first 2048 chunk (cache still filled, concat_heads kept) | | | 8.69 / 8.76 | | 50.47 (r0) | 0.999993 | rejected alone (flat); see 3 |
| 2 | **full-attention chunk = whole request** (`prefill_chunk_full_attention` = max_seq_len) vs 2048 | 1.81 / 1.81 | 4.60 / 4.62 vs 4.56 / 4.60 | = | **18.28 / 18.34 vs 19.46 / 19.52** | **42.64 / 42.66 vs 47.52 / 47.64** | 0.999993 (0.999955) | **kept** (4096-token chunks: 45.20 at 8192) |
| 2a | SDPA cores per head batch 8 / 32 (whole request) | 1.83 / 1.81 | | 8.47-8.52 | | 42.63-42.65 | | rejected (flat) |
| 3 | **plain SDPA + `output_concat_heads` (no concatenate_heads) + no cache write for a single-chunk request**, vs chunked SDPA on the whole request | **1.770 / 1.771 vs 1.803 / 1.805** | | **8.30 / 8.34 vs 8.47 / 8.47** | | **42.08 / 42.09 vs 42.63 / 42.64** | 0.999993 (0.999955) | **kept**; the cache-write skip alone is ~0.1 ms at 8192 (plain + concat with the cache write: 42.24 / 42.25) |
| 4 | **plain SDPA q chunk 256 for >= 2048 rows** (k 128) | 128: = (q halves to t) | 1024 with q 256: 4.63 / 4.52 vs 4.50 / 4.41 (slower, so 128 below 2048) | **7.85 / 7.86 vs 8.31 / 8.31** | **16.78 / 16.73 vs 18.01 / 17.99** | **34.51 / 34.49 vs 42.06 / 42.10** | 0.999993 (0.999955) | **kept** (-7.6 ms at 8192) |
| 4a | plain SDPA q/k 512/128, 512/64, 1024/64, 512/32, 1024/32 | | L1 overflow | L1 overflow | | L1 overflow | | rejected (illegal) |
| 4b | plain SDPA q/k 256/64, 256/32 | | 4.67, 4.96 | 8.10, 8.52 | | 36.52, 41.18 | | rejected (slower than 256/128) |
| 5 | **RMSNorm compute config without fp32 dest accumulation** (all norms incl. q/k head norms; HiFi4 kept) | 1.79 / 1.80 vs 1.78 / 1.84 | | **7.82 / 7.83 vs 7.91 / 7.90** | | **34.10 / 34.10 vs 34.59 / 34.54** | 0.999988 (0.999954) vs 0.999993 (0.999955) | **kept** (op level 2048 rows 194 -> 153 us, 8192 rows 737 -> 482 us) |
| 5a | same, layer norms only (head norms unchanged) | 1.78 / 1.77 | | 7.84 / 7.83 | | 34.03 / 34.04 | 0.999988 (0.999955) | equivalent; one config for all norms kept |
| 5b | RMSNorm HiFi2 / LoFi (fp32 acc), legacy reduction / rsqrt; Welford | | | op level: 193.5 us at 2048 (= HiFi4) | | op level: 736 us | | rejected (no gain); Welford is LayerNorm-only (`layernorm_device_operation.cpp:333`) |
| 6 | residual adds in place (output into the mixer / MLP buffer) | 1.776 vs 1.770 | | 8.31 / 8.32 vs 8.30 / 8.34 | | 42.07 / 42.06 vs 42.08 / 42.09 | | rejected (flat; the adds already run at ~88 % of DRAM peak) |

### Linear-attention layer (L0)

| # | candidate | S=128 | S=2048 | S=8192 | PCC (8192) | decision |
|---|---|---|---|---|---|---|
| 1 | base: 3A config (phased GDN chosen by the op's cost model, conv channel chunk 256) | 2.035 / 2.032 | 9.944 / 9.943 | 40.34 / 40.37 | 0.999993 (0.999994) | |
| 2 | **`ChunkGdnFusedProgramConfig()`** (fused prep -> scan; the cost model predicts 2118 vs 2054 us and picks phased) | **1.950 / 1.972** | **9.177 / 9.221** | **37.38 / 37.43** | identical | **kept** (-0.76 ms at 2048, -3.0 ms at 8192) |
| 2a | phased, `use_mcast=False` | 2.087 / 2.073 | 10.37 / 10.37 | 42.01 / 42.01 | identical | rejected |
| 2b | fused geometry: `row_local=False` / `handoff_depth` 1, 4 / `posted` / multicast hand-off | | 9.50, 9.06-9.10 | 38.68, 36.98-37.05 vs 37.00 / 37.01 | identical | rejected (worse or flat); NP + NV <= 2 per head on 130 cores (48 heads), so 1 + 1 is the only fused geometry |
| 3 | **conv channel chunk 512** (on the fused GDN) | 1.933 / 1.941 vs 1.930 / 1.934 | **9.094 / 9.079 vs 9.207 / 9.214** | **36.97 / 36.97 vs 37.45 / 37.44** | identical | **kept** (128: within noise) |
| 3a | conv channel chunk 128 / 1024 (on phased) | 2.045, 2.065 | 10.08, 9.90 vs 9.94 | 40.83, 40.17 vs 40.34 | identical | rejected (1024: smaller gain than 512) |
| 4 | prefill chunk 4096 / 8192 for linear layers | 2.04 / 2.04 | = | 74.23 / 71.30 vs 40.34 | identical | rejected (+31-34 ms; not investigated further) |
| 5 | split z and b\|a into two matmuls (weight cut once at load, same bytes) | 1.991 / 1.991 / 2.005 vs 1.889 / 1.898 / 1.900 | 9.058 / 9.065 / 9.055 vs 9.060 / 9.095 / 9.098 | 36.79 / 36.78 / 36.76 vs 37.01 / 36.96 / 36.96 | identical | rejected (+0.10 ms at 128, -0.03 at 2048, -0.2 at 8192) |
| 6 | residual adds in place | 1.888 / 1.934 vs 1.892 / 1.886 | 9.10 / 9.10 vs 9.09 / 9.10 | 37.00 / 37.01 vs 36.98 / 36.97 | | rejected (flat) |
| 7 | **RMSNorm without fp32 dest accumulation** (as L3 #5) | 1.884 / 1.895 vs 1.896 / 1.925 | **8.968 / 8.974 vs 9.048 / 9.054** | **36.63 / 36.62 vs 37.00 / 36.99** | 0.999931 (0.999994) vs 0.999993 | **kept** |

### S=128 norm: block-sharded L1 RMSNorm (both kinds)

Op level (`scripts/norm_bench.py`, random input, includes the DRAM -> L1 -> DRAM reshards): the
interleaved norm runs 4 cores at 128 rows (83.6 us). Block-sharded: 8x1 grid 39.8 us, 4x2 39.9,
4x4 41.5, 10x2 43.6, 10x4 62.1 (PCC vs interleaved 0.999999). 1024 rows: 106.5 -> 89.2 us (10x8, the
only legal grids are 8x8 / 10x8); 2048 rows: every grid overflows L1. In-layer Tracy (L0, S=128):
reshard 5 us + norm 23 us + reshard 5 us per norm, against 82 us.

| measurement | interleaved | sharded 8x1 | sharded 4x2 |
|---|---|---|---|
| single synced layer, L3 S=128 (ms) | 1.809 / 1.809 | 1.807 / 1.801 | 1.821 / 1.811 |
| 8 L0 layers back to back, S=128 (ms, 3 rounds) | 15.41 / 15.43 / 15.43 | **14.77 / 14.85 / 14.69** | 14.76 / 14.79 / 14.84 |
| 8 L0 layers, S=1024 (sharded not active) | 41.01 / 41.00 / 41.00 | 41.00 / 41.01 / 41.01 | |

Kept (8x1, <= 128 rows): -81 us per layer in the stack. The single synced layer does not show it:
the two extra short ops add host dispatch gaps (18 and 35 us in the Tracy capture) that cover the
device saving when nothing else is queued.

## 12. Before / after (interleaved A/B, the brief's harness)

`tests/perf/test_prefill_perf.py::test_layer_and_module_perf` (2 warm-ups, median of 7, no idle
gaps), run A1 B1 A2 B2; A = worktree of `57f76869913` (a temporary worktree next to the checkout, removed
after the run), B = this stage. Records `ab/perf_ab_{A,B}{1,2}.jsonl`. Layer medians, ms, (run 1, run 2):

| kind | S | A (3A) | B (3B) | change |
|---|---:|---|---|---|
| linear_attention | 128 | 2.05, 2.04 | 1.89, 1.89 | -7.5 % |
| linear_attention | 1024 | 5.59, 5.60 | 5.14, 5.16 | -8.0 % |
| linear_attention | 2048 | 9.95, 9.94 | 9.00, 9.02 | -9.4 % |
| linear_attention | 4096 | 20.20, 20.22 | 18.43, 18.42 | -8.8 % |
| linear_attention | 8192 | 41.07, 42.41 | 39.31, 39.23 | -5.9 % |
| full_attention | 128 | 1.80, 1.81 | 1.78, 1.78 | -1.3 % |
| full_attention | 1024 | 4.54, 4.57 | 4.44, 4.46 | -2.3 % |
| full_attention | 2048 | 8.48, 8.52 | 7.85, 7.84 | -7.7 % |
| full_attention | 4096 | 19.51, 19.58 | 16.69, 16.68 | -14.6 % |
| full_attention | 8192 | 47.56, 51.79 | 36.53, 36.91 | -26.1 % |

Every bucket of both kinds improves by more than the run-to-run spread of either side. The small
gains are explained: full S=128 and 1024 have little SDPA time (54 us / ~0.3 ms) and the S=128
changes (sharded norm, no `concatenate_heads`) mostly land in host dispatch gaps of a single synced
layer (section 11, 8-layer stack: -81 us per layer from the norm alone).

## 13. Device profiles after (Tracy, `doc/optimized_decoder/perf/opt3b_*`)

One warmed layer between `PREFILL_START` / `PREFILL_END`; `tt-perf-report` with advice (txt) and
CSV at S=128, 2048 and 8192 for both kinds. Raw captures: `stage3b/tracy/final_*`.

| layer | 3A device us | 3B device us | ops | notes |
|---|---:|---:|---:|---|
| L0 S=128 | 1968 | 1791 (-9.0 %) | 31 | norm 82 -> 22-23 us (+ 2 x 5 us reshard); GDN prep+scan 169 -> fused 118 us |
| L3 S=128 | 1758 | 1625 (-7.6 %) | 22 | concat heads and cache writes gone; 153 us of op-to-op gaps (host) |
| L0 S=2048 | 9773 | 8923 (-8.7 %) | 27 | GDN 1875 -> 1219 us, conv 473 -> 344 us, norms 384 -> 303 us |
| L3 S=2048 | 8399 | 7706 (-8.3 %) | 18 | SDPA 1276 -> 839 us; concat heads, 2 cache writes, page-table slice gone |
| L0 S=8192 | 39894 | 36442 (-8.7 %) | 113 | 4 chunks of 2048 (as 3A) |
| L3 S=8192 | 47337 | 33871 (-28.4 %) | 16 | one chunk: SDPA 18023 -> 8653 us (95 TFLOP/s on the causal half, 53 % of the HiFi4 peak); MLP at M=8192: gate\|up 10274 -> 9320 us (87 % FLOP util), down 5272 -> 5040 us; chunk slices / concat gone |

Remaining non-matmul shares at 2048 (L0): fused GDN 13.7 %, BinaryNg 5.3 % (residual adds at ~88 %
of DRAM peak, the z gate mul, the softplus/gate math), conv 3.9 %, gated norm 3.7 %, norms 3.4 %,
`ReshapeView` (inside the GDN op launch) 2.5 %, untilize 2.4 %, z slice 1.5 %. L3 at 8192: SDPA
25.6 %, BinaryNg 5.2 %, norms 4.6 %, RoPE 2.9 %, create-heads 1.8 %.

Advice in the reports and its disposition: "High op-to-op gap" at S=128 (tracing would save ~112
us): the request path is eager by design of stage 6; the traced forward exists as
`test_traced_forward`, so this is a full-model item, not a layer item. "Matmul ... Optimized" on
every matmul row; `MinimalMatmulDeviceOperation` still reported as unclassified (tt-perf-report
candidate improvement, as in 3A). No other advice.

## 14. Gates on the final code (measured)

| check | command | result |
|---|---|---|
| PCC suite | `PPLX_DECIDER_PCC_LOG=stage3b/pcc_suite_final.jsonl pytest models/demos/pplx_decider_v1_27b/tests/pcc -q` | **143 passed in 679 s, exit 0** (142 of 3A + the new chunked full-attention contract test); 128 records, 0 below 0.995, **min 0.999535** (`rmsnorm_input_norm` L0 S1024). Against 3A (124 matching cases): norms -1e-5 .. -4.6e-4, decoder layers -4e-6 .. -2.3e-4, attention -1e-6 .. -2e-5, GDN / MLP / readout / embedding identical. The shift is the norm fp32-accumulation change (section 11, L0 #7 / L3 #5) |
| chunked full-attention path | `tests/pcc/test_prefill_contract.py::test_prefill_contract_full_attention_chunked` (in the suite; also rerun alone, `logs/contract_chunked.log`) | passed: 8192 (4 chunks) -> 128 -> 2048 -> 4096 on one instance with 2048-token chunks, PCC 0.999987-0.999988, repeat bit-identical |
| non-bucket lengths | `scripts/bench3b.py` at S = 100, 1000, 2100 (L0) and 100, 1000, 2100, 4100 (L3), `bench/awkward.jsonl` | layer PCC vs HF 0.999927-0.999990 (L0), 0.999986-0.999988 (L3); mixer PCC 0.999993-0.999994 / 0.999957-0.999961 |
| runtime audit | in the suite (`test_runtime_audit`, L0/L3 x S128/S4096) and the e2e audits | 0 host conversions, 0 torch ops; e2e `runtime_audit_*.json` host_calls {} (buckets 1024, 8192, image 1024); 0 host ops in all 6 Tracy captures |
| determinism | `test_prefill_contract` (both kinds + chunked), e2e `test_determinism`, image determinism | bit-identical repeats |
| e2e text | `PPLX_DECIDER_STAGE6_DIR=stage3b/e2e pytest tests/e2e/test_model.py tests/e2e/test_precision_config.py -q -s` | 7 passed, exit 0; **decision agreement 25/25**, readout-logit PCC min 0.99721 (3A 0.99754; bar 0.99), median 0.99994; precision check: every role BFLOAT8_B + HiFi2 |
| e2e image | `PPLX_DECIDER_IMAGE_E2E_DIR=stage3b/e2e_image pytest tests/e2e/test_image_model.py -q -s` | 7 passed, exit 0; **8/8** agree, logit PCC min 0.99923 (3A 0.99987); stage-12B evidence untouched |
| watcher, linear_attention | `TT_METAL_WATCHER=10 TT_METAL_LOGS_PATH=stage3b/watcher/L0 pytest tests/pcc/test_decoder_layer.py -q -k "L0 and (S128 or S2048 or S8192)"` | 3 passed, exit 0; disabled features: None; watcher.log 648 lines, 0 error / assert / sanitize / hang / overflow lines |
| watcher, full_attention | same with `L3` | 3 passed, exit 0; 649 lines, 0 such lines |

### Full-model request latency (stage-6 harness, measured)

`tests/perf/test_model_perf.py::test_model_perf`, B (this stage) then A (worktree of
`57f76869913`) on the same weight cache, same session. Burst = median of 5 passes, each after 10 s
idle; sustained = 2 warm-ups + median of 7 back to back. ms. JSON `stage3b/model_perf_{A,B}/`.

| bucket | prompt tokens | device forward, burst | request, burst | device forward, sustained | request, sustained | decisions/s (sustained) |
|---:|---:|---|---|---|---|---|
| 128 | 100 | 124.7 -> 113.2 (-9.2 %) | 126.3 -> 114.8 (-9.1 %) | 124.3 -> 112.8 (-9.2 %) | 125.4 -> 114.0 (-9.1 %) | 7.97 -> 8.77 |
| 1024 | 178 | 336.5 -> 313.0 (-7.0 %) | 338.6 -> 315.2 (-6.9 %) | 435.1 -> 417.0 (-4.2 %) | 436.8 -> 419.9 (-3.9 %) | 2.29 -> 2.38 |
| 2048 | 2006 | 610.1 -> 555.0 (-9.0 %) | 619.3 -> 563.9 (-8.9 %) | 844.8 -> 807.4 (-4.4 %) | 861.6 -> 828.9 (-3.8 %) | 1.16 -> 1.21 |
| 4096 | 3906 | 1352.9 -> 1212.9 (-10.3 %) | 1370.0 -> 1228.4 (-10.3 %) | 1751.0 -> 1665.5 (-4.9 %) | 1790.4 -> 1702.2 (-4.9 %) | 0.56 -> 0.59 |
| 8192 | 7928 | 3238.6 -> 2856.5 (-11.8 %) | 3255.8 -> 2894.6 (-11.1 %) | 3696.9 -> 3459.6 (-6.4 %) | 3751.9 -> 3518.6 (-6.2 %) | 0.27 -> 0.28 |

The A side reproduces the 3A reviewer's burst requests (126.3 / 338.8 / 619.7 / 1368.0 / 3272.7).
The 128 bucket gains -9 % in the model against -1.3 % / -7.5 % for a single synced layer: in the
64-layer forward the host runs ahead and the S=128 device savings (sharded norms, no concat heads,
fused GDN) are no longer hidden behind dispatch gaps (section 11, stack measurement). Burst vs
projection from the A/B layer medians (section 12): 114.8 / 119.2, 315.2 / 318.4, 563.9 / 557.9,
1228.4 / 1151.7, 2894.6 / 2472.5 ms; above 2048 the model stays above the projection as in 3A and
stage 6 (inferred: the same long-pass slowdown under sustained load). Sustained gains are smaller
than burst at 1024-8192, the throttling seen in section 9.

## 15. Optimize-skill checklist (stage 3B, prefill only)

| item | status / evidence |
|---|---|
| decoder path traced, no host fallbacks | runtime audit 0 host calls (PCC suite `test_runtime_audit`, e2e audits at buckets 1024 / 8192, image audit); 0 host ops in all 6 Tracy captures. Decode trace: N/A (prefill-only model); the eager request path is the stage-6 design (`test_traced_forward` exists) |
| decode activations width-sharded in L1 | N/A (no decode) |
| prefill activations DRAM interleaved, 2D matmul configs | yes (3A configs unchanged); the only L1 sharding is the S <= 128 norm (section 11) |
| operation-topology audit | section 10 (non-matmul ops) and section 1 (matmuls, 3A) |
| multi-device topology / lower-movement residual / CCL / fused CCL+matmul / persistent CCL buffers | N/A (one device, no collectives) |
| best-candidate comparison | every candidate measured against the current best in the same process, interleaved; final A/B against 3A (section 12) |
| final default reproduces the selected candidates | A/B B side vs the last bench base: linear 2048 9.01 vs 8.97, full 2048 7.84 vs 7.83; full 8192 36.72 vs 34.10 (the bench idles 5 s before each candidate, the A/B harness does not: throttling, section 9); Tracy device time of the final code: full 8192 33.87 ms |
| dtype/fidelity policy verified in the rows | every matmul row in `opt3b_*` reads `HiFi2 BF16 x BFP8 => BF16` (unchanged C0) |
| SDPA and composite ops instead of hand-built attention | yes: `scaled_dot_product_attention` (plain, causal, concat heads) and `chunked_scaled_dot_product_attention` for later chunks; `chunk_gated_delta_rule` (fused), `qkv_causal_conv1d_silu`, `sigmoid_gated_rms_norm` |
| packed / fused same-input projections | unchanged from stage 2/3A; z\|b\|a split re-measured and rejected (section 11) |
| explicit memory / program / compute configs for important ops | SDPA program + compute config, GDN program config, conv program config, norm program config (sharded at <= 128 rows) and compute config, matmul configs (3A) |
| matmul sweeps, fidelity sweep, BFP4, DRAM-sharded, readers per bank | 3A (sections 1-4); BFP4 / LoFi matmuls forbidden by C0 |
| SDPA program config explicit, swept | section 11: q/k chunks (12 legal and 9 illegal pairs), exp approx, fidelity (HiFi4 / HiFi3 / HiFi2 / LoFi, fp32 acc on/off), cores per head, plain vs chunked, whole request vs chunks |
| KV-cache dtype (OPT-002 reduced cache) | N/A for decode; the prefill cache is now not written for single-chunk requests. BF16 cache kept for the multi-chunk path (not a measured cost at bucket lengths) |
| norm / residual layout (OPT-003) | interleaved DRAM residual (prefill); sharded norm kept at <= 128 rows, illegal (L1) at 2048, < 20 us at 1024; residual adds at ~88 % DRAM peak, no fusing epilogue in `minimal_matmul` |
| reduced precision experiments | C0 fixed; only SDPA exp approx and the norm fp32-accumulation switch, both PCC-checked on real weights (layer and full-model) |
| performance accounting (roofline / device / e2e) | per layer: device time (Tracy) vs warm wall time (A/B) in sections 12-13; full request in section 14. Decode roofline: N/A |
| batch capability | batch 1 by contract (prefill buckets); unchanged |
| watcher clean | section 14 |
| tt-perf-report with advice | `perf/opt3b_*_perf_report.txt` (S=128 / 2048 / 8192, both kinds), section 13 |

## 16. Not done / remaining candidates

- **MLP and norms over the whole request in linear-attention layers** (only the GDN mixer needs
  2048 chunks): the full-attention profile shows gate|up 10274 -> 9320 us and down 5272 -> 5040 us at
  8192 rows when run once instead of 4 x 2048, and the norm 4 x 153 -> 482 us (measured, op rows);
  about -1.3 ms per linear layer at 8192 (inferred, ~-60 ms per 8192 request). Not built: it needs a
  decoder restructure and a full gate rerun that did not fit the timebox.
- **Linear-layer chunk slices / concat** at 8192 (~800 us per layer): passing chunk lists between
  consecutive linear layers would remove most of it; model-level change.
- **Op-level items (ttnn source is read-only here)**: `chunk_gated_delta_rule` reshapes g / beta
  `[48, T] -> [48, NC, 32, 1]` with two 112 us `ReshapeView` ops per 2048 chunk inside its launch;
  the conv requires a ROW_MAJOR input (untilize 212 us per chunk); `minimal_matmul` has no residual
  epilogue; the GDN fused-geometry cost model picks the slower phased path on the 13x10 grid (fused
  is 1.08x faster here); `sigmoid_gated_rms_norm` returns `w * norm(o) * sigmoid(z)` so a separate
  `* z` multiply (170 us per chunk) completes silu.
- **S=128 host gaps**: 153-156 us of op-to-op gaps per layer in the S=128 captures; a traced request
  path (stage 6 `test_traced_forward`) removes them; outside this stage.
- tt-perf-report: `MinimalMatmulDeviceOperation`, `ChunkGdnDeviceOperation`, `QkvCausalConv1dSilu`,
  `SigmoidGatedRmsNorm`, `ReshapeView` are reported as unclassified (candidate tool improvement).
