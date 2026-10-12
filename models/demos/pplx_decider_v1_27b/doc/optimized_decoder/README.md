# Stage 3: optimized decoder layers (3A prefill matmuls, 3B non-matmul ops and chunking)

Stage 3B is described first; the stage-3A text follows unchanged from "Stage 3A" below.

## Stage 3B: non-matmul ops, SDPA, Gated DeltaNet, norms, chunking

Baseline `57f76869913` (stage 3A). Precision is unchanged (C0): BF16 activations, BFP8 weights,
HiFi2 + fp32 accumulation for every matmul, fp32 DeltaNet state; no weight, dtype or matmul config
changed and the C0 weight cache is reused as is. Labels as below; numbers are **measured** unless
marked. Full log, audit and every rejected candidate: [`work_log.md`](work_log.md#stage-3b-work-log-non-matmul-ops-chunking)
(sections 9-15).

### What changed (kept, one change per measurement)

| # | change | where | evidence (layer median, ms, interleaved bench) |
|---|---|---|---|
| 1 | SDPA `exp_approx_mode=True` | `optimizations.py` | full S=8192 50.47 -> 47.57; 2048 8.65 -> 8.53; PCC unchanged to 1e-6 |
| 2 | full-attention layers prefill the whole request as one chunk (`prefill_chunk_full_attention`, default `max_seq_len`; linear layers keep 2048) | `optimizations.py`, `decoder.py` | full 8192 47.58 -> 42.65, 4096 19.49 -> 18.31 |
| 3 | first chunk: plain causal SDPA on the chunk's own q/k/v with `output_concat_heads=True` (drops `concatenate_heads`), no paged-cache write when no later chunk follows | `attention.py` | full 8192 42.64 -> 42.09, 2048 8.47 -> 8.32, 128 1.80 -> 1.77 |
| 4 | plain SDPA q chunk 256 for >= 2048 rows (k 128; 128/128 below) | `optimizations.py`, `attention.py` | full 8192 42.08 -> 34.50, 4096 18.00 -> 16.76, 2048 8.31 -> 7.86 |
| 5 | GDN `chunk_gated_delta_rule` with `ChunkGdnFusedProgramConfig()` (the op's cost model picks the phased path on this grid) | `optimizations.py`, `gated_deltanet.py` | linear 8192 40.36 -> 37.41, 2048 9.94 -> 9.20, 128 2.03 -> 1.96; output bit-identical |
| 6 | causal conv `channel_chunk_size` 256 -> 512 | `optimizations.py` | linear 8192 37.45 -> 36.97, 2048 9.21 -> 9.09 |
| 7 | RMSNorm at <= 128 rows block-sharded in L1 on 8 cores (82 -> 23 us + 2 x 5 us reshards) | `norm.py`, `optimizations.py` | 8 L0 layers back to back at S=128: 15.42 -> 14.77 (-81 us per layer); flat for a single synced layer (host dispatch gaps) |
| 8 | RMSNorm compute config without fp32 dest accumulation (HiFi4 kept) | `optimizations.py` | linear 8192 37.00 -> 36.62, full 8192 34.56 -> 34.10; norm op 8192 rows 737 -> 482 us; layer PCC >= 0.99993 for the benched L0 / L3, but the PCC suite's decoder-layer minimum is 0.999742 (L63 S4096, up to -2.3e-4 vs 3A; see Gates) |

Also: `tests/e2e/test_image_model.py` reads `$PPLX_DECIDER_IMAGE_E2E_DIR` (default unchanged) so
reruns do not overwrite stage-12B evidence; `tests/pcc/test_prefill_contract.py` gains a test of the
multi-chunk full-attention path (paged cache + chunked SDPA), which the new default no longer uses
for bucket lengths; the perf harness times the MLP target with each kind's chunk.

### Result: per-layer warmed prefill latency (measured)

Harness `tests/perf/test_prefill_perf.py::test_layer_and_module_perf` (2 warm-ups, median of 7),
A1 B1 A2 B2, A = worktree of `57f76869913`, B = stage 3B; mean of the two run medians, ms. Logs:
`artifacts/pplx_decider/stage3b/ab/perf_ab_*.{jsonl,log}`.

| target | kind | S=128 | S=1024 | S=2048 | S=4096 | S=8192 |
|---|---|---|---|---|---|---|
| decoder layer | linear_attention | 2.04 -> 1.89 (-7.5 %) | 5.60 -> 5.15 (-8.0 %) | 9.95 -> 9.01 (-9.4 %) | 20.21 -> 18.43 (-8.8 %) | 41.74 -> 39.27 (-5.9 %) |
| decoder layer | full_attention | 1.81 -> 1.78 (-1.3 %) | 4.56 -> 4.45 (-2.3 %) | 8.50 -> 7.84 (-7.7 %) | 19.54 -> 16.69 (-14.6 %) | 49.68 -> 36.72 (-26.1 %) |
| GDN mixer | linear_attention | 0.83 -> 0.76 | 2.92 -> 2.57 | 5.35 -> 4.56 | 11.09 -> 9.50 | 22.83 -> 20.56 |
| attention mixer | full_attention | 0.66 -> 0.63 | 2.00 -> 1.90 | 4.04 -> 3.47 | 10.48 -> 8.00 | 31.68 -> 20.03 |
| MLP (code unchanged) | linear / full | 1.08 / 1.08 -> 1.08 / 1.08 | 2.31 / 2.36 -> 2.36 / 2.36 | 3.98 / 3.98 -> 4.00 / 4.01 | 8.04 / 8.00 -> 8.10 / 8.00 | 17.46 / 17.68 -> 18.28 / 17.18 |

Per-run medians (A1, A2 | B1, B2): linear 2.05, 2.04 | 1.89, 1.89; 5.59, 5.60 | 5.14, 5.16; 9.95,
9.94 | 9.00, 9.02; 20.20, 20.22 | 18.43, 18.42; 41.07, 42.41 | 39.31, 39.23. Full 1.80, 1.81 | 1.78,
1.78; 4.54, 4.57 | 4.44, 4.46; 8.48, 8.52 | 7.85, 7.84; 19.51, 19.58 | 16.69, 16.68; 47.56, 51.79 |
36.53, 36.91. **Both kinds beat 3A at every bucket**, beyond the run-to-run spread (largest B-side
spread 0.38 ms at full 8192; the A side spreads 4.2 ms there, the throttling drift of section 9).
The MLP module has no code change; its +/- 0.05 ms rows at S <= 4096 are run-to-run variation and
the 8192 rows follow the A-side drift. The full-attention S=128 gain is small because S=128 changes
(sharded norm, no `concatenate_heads`) are hidden by host dispatch gaps in a single synced layer.

Stage history, decoder layer ms (stage 1 -> stage 2 -> 3A -> 3B; stage 1 / 2 from their READMEs, 3A
and 3B from this A/B):

| kind | S=128 | S=1024 | S=2048 | S=4096 | S=8192 |
|---|---|---|---|---|---|
| linear_attention | 2.11 -> 2.37 -> 2.04 -> **1.89** | 6.16 -> 6.03 -> 5.60 -> **5.15** | 10.98 -> 10.64 -> 9.95 -> **9.01** | 22.47 -> 21.67 -> 20.21 -> **18.43** | 46.95 -> 45.23 -> 41.74 -> **39.27** |
| full_attention | 1.96 -> 2.18 -> 1.81 -> **1.78** | 5.47 -> 5.03 -> 4.56 -> **4.45** | 10.07 -> 9.26 -> 8.50 -> **7.84** | 22.70 -> 21.09 -> 19.54 -> **16.69** | 58.84 -> 55.20 -> 49.68 -> **36.72** |

Projected 64 layers (48 linear + 16 full, ms): 3A -> 3B 126.9 -> 119.2 (128), 341.8 -> 318.4
(1024), 613.6 -> 557.9 (2048), 1282.7 -> 1151.7 (4096), 2798.4 -> 2472.5 (8192).


Model `perplexity-ai/pplx-decider-v1-27b`, one Blackhole p150a (13x10 compute grid). Branch
`gtobarTT/pplx-decider-bringup`. Baseline: `78e35968f19` (the stage-2 matmul path; later stages did
not change it). Precision is the person's C0 choice and is unchanged: BF16 activations, BFP8 weights,
HiFi2 matmuls with fp32 accumulation, fp32 DeltaNet state. Length contract: buckets 128 / 1024 /
2048 / 4096 / 8192 at batch 1; buckets above 2048 run 2048-row chunks. Labels: **measured** means a
command and its output are recorded here or in [`work_log.md`](work_log.md); **inferred** means it
follows from code or arithmetic.

### Full-model request latency (measured)

Stage-6 harness `tests/perf/test_model_perf.py::test_model_perf`, A (`57f76869913`, worktree) vs B
(stage 3B), same session and weight cache, ms. Burst = median of 5 passes, each after 10 s idle.

| bucket | projected 64 layers (3A -> 3B) | device forward, burst | request, burst | request, sustained |
|---:|---|---|---|---|
| 128 | 126.9 -> 119.2 | 124.7 -> 113.2 (-9.2 %) | 126.3 -> 114.8 (-9.1 %) | 125.4 -> 114.0 (-9.1 %) |
| 1024 | 341.8 -> 318.4 | 336.5 -> 313.0 (-7.0 %) | 338.6 -> 315.2 (-6.9 %) | 436.8 -> 419.9 (-3.9 %) |
| 2048 | 613.6 -> 557.9 | 610.1 -> 555.0 (-9.0 %) | 619.3 -> 563.9 (-8.9 %) | 861.6 -> 828.9 (-3.8 %) |
| 4096 | 1282.7 -> 1151.7 | 1352.9 -> 1212.9 (-10.3 %) | 1370.0 -> 1228.4 (-10.3 %) | 1790.4 -> 1702.2 (-4.9 %) |
| 8192 | 2798.4 -> 2472.5 | 3238.6 -> 2856.5 (-11.8 %) | 3255.8 -> 2894.6 (-11.1 %) | 3751.9 -> 3518.6 (-6.2 %) |

The A side reproduces the 3A review (burst request 126.3 / 338.8 / 619.7 / 1368.0 / 3272.7 ms).
Against stage 6 (149 / 366 / 665 / 1481 / 3498 ms burst request) the optimized decoder is -23 %,
-14 %, -15 %, -17 % and -17 %. Decisions/s at batch 1 (sustained): 8.77 / 2.38 / 1.21 / 0.59 / 0.28.

### Device time (Tracy, one warmed layer, `perf/opt3b_*`)

3A -> 3B, us: linear 1968 -> 1791 (S=128), 9773 -> 8923 (2048), 39894 -> 36442 (8192); full 1758 ->
1625, 8399 -> 7706, 47337 -> 33871. At 8192 the full layer's SDPA is 18023 -> 8653 us (one causal
SDPA, q chunk 256, approximate exp) and its MLP runs once at M=8192 (gate|up 87 % of the HiFi2 FLOP
peak). Every matmul row still reads `HiFi2 BF16 x BFP8 => BF16`. 0 host ops.

### Rejected in 3B (measured; full list with numbers in the work log, section 11)

| option | result |
|---|---|
| SDPA q/k 256/256, 256/512, 512/512 (chunked) and 512/x, 1024/x (plain) | L1 overflow (head_dim 256) |
| SDPA q/k 128/256, 128/64, 64/128, 256/64, 64/256 (chunked); 256/64, 256/32 (plain) | slower than 128/128 resp. 256/128 |
| plain SDPA q 256 at 1024 rows | +0.1 ms (q 128 kept below 2048 rows) |
| SDPA HiFi2 / HiFi3 / LoFi, fp32 dest acc off | no gain over exp approx alone (LoFi also lowers mixer PCC to 0.99966) |
| SDPA max cores per head batch 8 / 32 | flat |
| full-attention chunks of 4096 | 45.2 vs 42.6 ms (whole request) at 8192 |
| linear-attention chunks of 4096 / 8192 | 74.2 / 71.3 vs 40.3 ms at 8192 |
| GDN phased without multicast; fused `row_local=False`, hand-off depth 1 / 4, posted, multicast hand-off | slower or flat |
| conv channel chunk 128 / 1024 | slower / smaller gain than 512 |
| z and b\|a as two matmuls (weight cut at load, same bytes) | +0.10 ms at 128, -0.03 at 2048, -0.2 at 8192 |
| residual adds in place | flat (adds run at ~88 % of DRAM peak) |
| sharded RMSNorm at 1024 / 2048 rows | 1024: -17 us per norm (op); 2048: L1 overflow |
| RMSNorm HiFi2 / LoFi, legacy reduction / rsqrt | no gain (Welford: LayerNorm only) |

### Gates (measured, final code)

- **PCC suite** `pytest models/demos/pplx_decider_v1_27b/tests/pcc -q`: **143 passed**, exit 0; all
  128 records >= 0.995, min **0.999535** (`rmsnorm_input_norm` L0 S1024; the norm accumulation
  change moves norm / layer PCC by at most -4.6e-4; GDN, MLP, readout identical to 3A).
- **Chunked full-attention path** (new contract test, 2048-token chunks, stale-page order): PCC
  0.999987-0.999988, repeat bit-identical. **Non-bucket lengths** S = 100 / 1000 / 2100 / 4100:
  layer PCC >= 0.999927.
- **Runtime audit** 0 host calls (suite, e2e text and image); **determinism** bit-identical.
- **e2e text** 7 passed: **25/25**, readout-logit PCC min 0.99721 (bar 0.99). **e2e image** 7 passed:
  **8/8**, min 0.99923 (written to `stage3b/e2e_image/` via `$PPLX_DECIDER_IMAGE_E2E_DIR`).
- **Watcher** (`TT_METAL_WATCHER=10`, `test_decoder_layer.py` at S128 / S2048 / S8192, one run per
  kind): 3 + 3 passed, no disabled features, 0 error / assert / sanitize / hang lines.

Artifacts: `$PPLX_DECIDER_ARTIFACTS/stage3b/` (`bench/`, `ab/`, `tracy/`, `logs/`,
`model_perf_{A,B}/`, `e2e/`, `e2e_image/`, `watcher/`, `scripts/`).

## Stage 3A

### What changed

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

### Result: per-layer warmed prefill latency (measured)

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

### Why it is faster

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

### Rejected (measured)

| option | result | why rejected |
|---|---|---|
| DRAM-sharded weights + DRAM-sharded matmul at S=128 (tt-perf-report advice) | op allows M = 32 only (`matmul_device_operation.cpp:1455`); adapted at M = 32: down 308 us vs 257 us (1D config, same M); the 128 bucket needs 4 calls | slower even per call |
| `minimal_matmul` on a DRAM width-sharded gate\|up weight | 1208 vs 821 us at M=128 | slower |
| input 0 in L1 for the S=128 gate\|up (advice) | 769 vs 764 us | no gain |
| SwiGLU product in L1 at S=128 | MLP 1.081 -> 1.076 / 1.078 ms | within noise |
| 2D multicast `ttnn.linear` for down | M=128: 423 vs 272 us; M=1024: 862 vs 798 us | slower |
| 13x8 grid at M=1024 (no M padding) | gate\|up 1493 vs 1495, down 798 vs 804 us | flat |
| 2x2 / 4x1 subblocks for gate\|up at 2048 | 2953 / 3037 vs 2577 us | slower |

### Gates (measured)

- **PCC suite** `pytest models/demos/pplx_decider_v1_27b/tests/pcc -q`: 142 passed, exit 0. All 124
  PCC records >= 0.995; minimum 0.999897 (GDN L61 S8192, the same case as stage 2). Against stage 2
  the per-case delta is -1.3e-6 .. +1.7e-7 (123 matching cases).
- **Runtime audit** (in the suite): 0 host conversions and 0 torch ops (L0/L3, S128/S4096); 0 host
  ops in all 6 Tracy captures. **Determinism** (`test_prefill_contract`): bit-identical repeat.
- **e2e text** `tests/e2e/test_model.py`: 6 passed; decision agreement **25/25**, readout-logit PCC
  min 0.99754 (bar 0.99). **e2e image** `tests/e2e/test_image_model.py`: 7 passed; **8/8** agree.
- **Watcher** (`TT_METAL_WATCHER=10`, `test_decoder_layer.py` at S128 and S2048), one run per layer
  kind: L0 2 passed, L3 2 passed, no error / assert / sanitize / hang entries.

### Full-model request latency (measured)

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

### Remaining candidates for stage 3B (taken up above)

See [`work_log.md`](work_log.md#8-remaining-candidates-for-stage-3b): S=128 norms (2 x 82 us on 4
cores), the S=128 gate|up op limit, GDN chunk prep/scan and conv, SDPA at long S, residual adds.

## Commands

Environment: `source python_env/bin/activate; export TT_METAL_HOME=$PWD PYTHONPATH=$PWD HF_HOME=<your HF cache>`.

| purpose | command |
|---|---|
| PCC suite | `pytest models/demos/pplx_decider_v1_27b/tests/pcc -q` |
| warmed layer perf | `PPLX_DECIDER_PERF_LOG=<log> pytest models/demos/pplx_decider_v1_27b/tests/perf/test_prefill_perf.py -q -k test_layer_and_module_perf` |
| baseline (A) | `git checkout 78e35968f19 -- models/demos/pplx_decider_v1_27b/tt/common.py models/demos/pplx_decider_v1_27b/tt/optimizations.py`, run the line above, then `git checkout HEAD -- <same files>` |
| device profile | `python -m tracy -r -p -v --no-web-server -o <dir> -m pytest "models/demos/pplx_decider_v1_27b/tests/perf/test_prefill_perf.py::test_profile_layer[L0_linear-S2048-device_params0]"` |
| perf report | `tt-perf-report <ops_perf_results.csv> --start-signpost PREFILL_START --end-signpost PREFILL_END [--csv out.csv]` (advice on) |
| full-model latency | `PPLX_DECIDER_STAGE6_DIR=<dir> pytest models/demos/pplx_decider_v1_27b/tests/perf/test_model_perf.py::test_model_perf -q -s` |
| 3B candidate bench | `BENCH_OUT=<jsonl> python $PPLX_DECIDER_ARTIFACTS/stage3b/scripts/bench3b.py <layer> <S,S,..> base,<cand>,.. <rounds>` (`BENCH_STACK=8` for back-to-back layers) |
| 3B A/B | `$PPLX_DECIDER_ARTIFACTS/stage3b/scripts/ab.sh` (A = `git worktree add --detach <dir> 57f76869913`) |
| image e2e to its own dir | `PPLX_DECIDER_IMAGE_E2E_DIR=<dir> pytest models/demos/pplx_decider_v1_27b/tests/e2e/test_image_model.py -q -s` |
| watcher | `TT_METAL_WATCHER=10 TT_METAL_LOGS_PATH=<dir> pytest models/demos/pplx_decider_v1_27b/tests/pcc/test_decoder_layer.py -q -k "S2048 and (L0 or L3)"` |

Artifacts: `$PPLX_DECIDER_ARTIFACTS/stage3a/` (`micro/`, `ab/`, `tracy/`,
`logs/`, `pcc_suite_final.jsonl`, `model_perf_{A,B}/`, `watcher/`).
