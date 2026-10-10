# Stage 3A: optimized decoder layers, part 1 (prefill matmuls)

Model `perplexity-ai/pplx-decider-v1-27b`, one Blackhole p150a (13x10 compute grid). Branch
`gtobarTT/pplx-decider-bringup`. Baseline: `78e35968f19` (the stage-2 matmul path; later stages did
not change it). Precision is the person's C0 choice and is unchanged: BF16 activations, BFP8 weights,
HiFi2 matmuls with fp32 accumulation, fp32 DeltaNet state. Length contract: buckets 128 / 1024 /
2048 / 4096 / 8192 at batch 1; buckets above 2048 run 2048-row chunks. Labels: **measured** means a
command and its output are recorded here or in [`work_log.md`](work_log.md); **inferred** means it
follows from code or arithmetic.

## What changed

Only matmul program configs. The stored weights, their layout, dtype and memory config are the
same, so the C0 weight cache stays valid (no new cache files). Files: `tt/optimizations.py` (the
config tables) and `tt/common.py` (`prefill_linear` picks the config).

| rows | projection(s) | stage 2 | stage 3A |
|---|---|---|---|
| >= 512 (1024 and 2048 chunks) | all seven projection shapes | one `MinimalMatmulConfig` (4, 8, 16, 1x4) | one config per weight shape (`_MINIMAL_BLOCKS`): N block = N tiles per core or half of it, K block 16-32 for the MLP, 4x1 subblocks for the narrow outputs |
| <= 128 (128 bucket) | down, o_proj / out_proj, z\|b\|a, attention q\|k\|v and gate | `ttnn.linear`, auto program config (`in0_block_w` 2) | `ttnn.linear` with an explicit 1D in0-multicast config (`_SHORT_MCAST1D`: `in0_block_w` 4-8, `per_core_N` 2-4) |
| <= 128 | gate\|up (fused SwiGLU) | `minimal_matmul` (4, 8, 16, 1x4), grid 13x10 | `minimal_matmul` (4, 32, 8, 1x4), grid 13 x (M tiles) |
| <= 128 | GDN q\|k\|v | `ttnn.linear` auto | unchanged (no config beat it) |

The decision-head readout (weight 5120x256) is not in the tables and keeps the stage-2 path; vision
is untouched.

## Result: per-layer warmed prefill latency (measured)

Harness `tests/perf/test_prefill_perf.py::test_layer_and_module_perf` (2 warm-ups, median of 7),
run interleaved A1 B1 A2 B2 (A = `78e35968f19`, B = stage 3A); each cell is the mean of two run
medians, ms. Logs: `artifacts/pplx_decider/stage3a/ab/perf_ab_*.jsonl`.

| target | kind | S=128 | S=1024 | S=2048 | S=4096 | S=8192 |
|---|---|---|---|---|---|---|
| decoder layer | linear_attention | 2.41 -> 2.03 (-15.7 %) | 6.02 -> 5.60 (-6.9 %) | 10.63 -> 9.94 (-6.5 %) | 21.66 -> 20.28 (-6.4 %) | 44.90 -> 41.95 (-6.6 %) |
| decoder layer | full_attention | 2.20 -> 1.82 (-17.3 %) | 5.04 -> 4.55 (-9.6 %) | 9.25 -> 8.49 (-8.2 %) | 21.07 -> 19.54 (-7.3 %) | 54.34 -> 51.21 (-5.8 %) |
| GDN mixer | linear_attention | 0.93 -> 0.83 (-11.3 %) | 3.05 -> 2.93 (-4.0 %) | 5.52 -> 5.35 (-3.0 %) | 11.43 -> 11.12 (-2.8 %) | 23.55 -> 22.87 (-2.9 %) |
| attention mixer | full_attention | 0.82 -> 0.66 (-18.9 %) | 2.14 -> 1.94 (-9.2 %) | 4.28 -> 4.04 (-5.7 %) | 10.99 -> 10.44 (-5.0 %) | 34.04 -> 33.09 (-2.8 %) |
| MLP | linear_attention | 1.30 -> 1.08 (-17.4 %) | 2.63 -> 2.32 (-12.0 %) | 4.52 -> 4.00 (-11.4 %) | 9.10 -> 8.02 (-11.9 %) | 19.91 -> 17.71 (-11.0 %) |
| MLP | full_attention | 1.31 -> 1.08 (-17.5 %) | 2.63 -> 2.31 (-12.1 %) | 4.49 -> 3.98 (-11.4 %) | 9.14 -> 8.01 (-12.4 %) | 20.76 -> 18.33 (-11.7 %) |

Run-to-run spread of the layer medians: <= 0.09 ms up to S=4096, up to 0.71 ms at S=8192 (both
sides). The A side reproduces the stage-2 README numbers within 0.04 ms at S <= 4096 (stage 2:
2.37 / 6.03 / 10.64 / 21.67 / 45.23 and 2.18 / 5.03 / 9.26 / 21.09 / 55.20).

Projected 64 layers (48 x linear + 16 x full, from the table, ms): 150.9 -> 126.6 (128, -16.1 %),
369.6 -> 341.6 (1024, -7.6 %), 658.2 -> 613.0 (2048, -6.9 %), 1376.8 -> 1286.1 (4096, -6.6 %),
3024.6 -> 2833.0 (8192, -6.3 %).

Stage-1 S=128 reference: 2.11 (linear) / 1.91 (full) ms. **Both kinds beat stage 2 at every bucket,
and S=128 is below stage 1 for both kinds** (2.03 / 1.82 ms).

Device time (Tracy, one warmed layer, `perf/`): linear 2269 -> 1968 us at S=128 (-13.3 %), 10487 ->
9773 us at 2048 (-6.8 %), 42734 -> 39894 us at 8192 (-6.6 %); full 2130 -> 1758 (-17.5 %), 9172 ->
8399 (-8.4 %), 50370 -> 47337 (-6.0 %). Matmuls are now 58 % (linear) / 66 % (full) of the 2048
layer, at 71-79 % of the HiFi2 FLOP peak.

## Why it is faster

- **S >= 1024.** `minimal_matmul` gives each core ceil(N_tiles / 13) N tiles. The stage-2 N block of
  16 left a ragged last block on every core for the 5120-wide outputs (13 tiles per core) and the
  8192-wide q|k|v (20 per core). An N block that divides the per-core range, a 4x1 subblock (the
  fp32 dest holds 4 tiles) and, for the MLP, a 16-32 deep K block raise the FLOP utilization from
  60-73 % to 71-79 %. Down 1602 -> 1317 us, gate|up 2797 -> 2565 us, q|k|v 784 -> 628 us per chunk.
- **S = 128.** The projections are weight-read bound. With `in0_block_w` 2 the auto config issued
  small reads; `in0_block_w` 4-8 on the same interleaved weight raises the read rate from 42-69 % to
  65-70 % of DRAM peak. Down 438 -> 264 us, out_proj 160 -> 102 us.
- **The S=128 regression of stage 2** (fused SwiGLU on 13 weight readers) is reduced, not removed:
  815 -> 760-792 us with a deeper K block and a grid cut to the 4 M tiles. The other short-row
  projections more than recover the rest. Removing the 13-reader limit needs an op change; all
  model-side routes were measured or ruled out (work log, section 2).

## Rejected (measured)

| option | result | why rejected |
|---|---|---|
| DRAM-sharded weights + DRAM-sharded matmul at S=128 (tt-perf-report advice) | op allows M = 32 only (`matmul_device_operation.cpp:1455`); adapted at M = 32: down 308 us vs 257 us (1D config, same M); the 128 bucket needs 4 calls | slower even per call |
| `minimal_matmul` on a DRAM width-sharded gate\|up weight | 1208 vs 821 us at M=128 | slower |
| input 0 in L1 for the S=128 gate\|up (advice) | 769 vs 764 us | no gain |
| SwiGLU product in L1 at S=128 | MLP 1.081 -> 1.076 / 1.078 ms | within noise |
| 2D multicast `ttnn.linear` for down | M=128: 423 vs 272 us; M=1024: 862 vs 798 us | slower |
| 13x8 grid at M=1024 (no M padding) | gate\|up 1493 vs 1495, down 798 vs 804 us | flat |
| 2x2 / 4x1 subblocks for gate\|up at 2048 | 2953 / 3037 vs 2577 us | slower |

## Gates (measured)

- **PCC suite** `pytest models/demos/pplx_decider_v1_27b/tests/pcc -q`: 142 passed, exit 0. All 124
  PCC records >= 0.995; minimum 0.999897 (GDN L61 S8192, the same case as stage 2). Against stage 2
  the per-case delta is -1.3e-6 .. +1.7e-7 (123 matching cases).
- **Runtime audit** (in the suite): 0 host conversions and 0 torch ops (L0/L3, S128/S4096); 0 host
  ops in all 6 Tracy captures. **Determinism** (`test_prefill_contract`): bit-identical repeat.
- **e2e text** `tests/e2e/test_model.py`: 6 passed; decision agreement **25/25**, readout-logit PCC
  min 0.99754 (bar 0.99). **e2e image** `tests/e2e/test_image_model.py`: 7 passed; **8/8** agree.
- **Watcher** (`TT_METAL_WATCHER=10`, `test_decoder_layer.py` at S128 and S2048), one run per layer
  kind: L0 2 passed, L3 2 passed, no error / assert / sanitize / hang entries.

## Full-model request latency (measured)

Stage-6 harness `tests/perf/test_model_perf.py::test_model_perf`, A (`78e35968f19` configs) vs B
(stage 3A), ms. Burst = median of 5 passes, each after 10 s idle.

| bucket | projected (64 layers) | device forward, burst | request, burst | request, sustained |
|---:|---|---|---|---|
| 128 | 150.9 -> 126.6 | 146.9 -> 124.6 (-15.2 %) | 148.6 -> 126.4 (-14.9 %) | 152.7 -> 125.1 (-18.1 %) |
| 1024 | 369.6 -> 341.6 | 363.8 -> 336.6 (-7.5 %) | 366.0 -> 338.7 (-7.4 %) | 461.9 -> 438.5 (-5.1 %) |
| 2048 | 658.2 -> 613.0 | 656.5 -> 610.1 (-7.1 %) | 665.1 -> 619.2 (-6.9 %) | 903.0 -> 857.4 (-5.1 %) |
| 4096 | 1376.8 -> 1286.1 | 1470.5 -> 1351.3 (-8.1 %) | 1481.3 -> 1366.4 (-7.8 %) | 1884.5 -> 1779.3 (-5.6 %) |
| 8192 | 3024.6 -> 2833.0 | 3459.0 -> 3226.4 (-6.7 %) | 3498.3 -> 3267.0 (-6.6 %) | 3944.6 -> 3754.6 (-4.8 %) |

Decisions/s at batch 1 (burst request): 6.73 -> 7.91 (128), 2.73 -> 2.95 (1024), 1.50 -> 1.62
(2048), 0.29 -> 0.31 (8192).

## Remaining candidates for stage 3B

See [`work_log.md`](work_log.md#8-remaining-candidates-for-stage-3b): S=128 norms (2 x 82 us on 4
cores), the S=128 gate|up op limit, GDN chunk prep/scan and conv, SDPA at long S, residual adds.

## Commands

Environment: `source python_env/bin/activate; export TT_METAL_HOME=$PWD PYTHONPATH=$PWD HF_HOME=/local/ttuser/gtobar/hf`.

| purpose | command |
|---|---|
| PCC suite | `pytest models/demos/pplx_decider_v1_27b/tests/pcc -q` |
| warmed layer perf | `PPLX_DECIDER_PERF_LOG=<log> pytest models/demos/pplx_decider_v1_27b/tests/perf/test_prefill_perf.py -q -k test_layer_and_module_perf` |
| baseline (A) | `git checkout 78e35968f19 -- models/demos/pplx_decider_v1_27b/tt/common.py models/demos/pplx_decider_v1_27b/tt/optimizations.py`, run the line above, then `git checkout HEAD -- <same files>` |
| device profile | `python -m tracy -r -p -v --no-web-server -o <dir> -m pytest "models/demos/pplx_decider_v1_27b/tests/perf/test_prefill_perf.py::test_profile_layer[L0_linear-S2048-device_params0]"` |
| perf report | `tt-perf-report <ops_perf_results.csv> --start-signpost PREFILL_START --end-signpost PREFILL_END [--csv out.csv]` (advice on) |
| full-model latency | `PPLX_DECIDER_STAGE6_DIR=<dir> pytest models/demos/pplx_decider_v1_27b/tests/perf/test_model_perf.py::test_model_perf -q -s` |
| watcher | `TT_METAL_WATCHER=10 TT_METAL_LOGS_PATH=<dir> pytest models/demos/pplx_decider_v1_27b/tests/pcc/test_decoder_layer.py -q -k "S2048 and (L0 or L3)"` |

Artifacts: `/local/ttuser/gtobar/artifacts/pplx_decider/stage3a/` (`micro/`, `ab/`, `tracy/`,
`logs/`, `pcc_suite_final.jsonl`, `model_perf_{A,B}/`, `watcher/`).
