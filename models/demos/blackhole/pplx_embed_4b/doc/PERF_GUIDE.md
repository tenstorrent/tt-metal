# pplx-embed-v1-4B on Blackhole P150 — performance guide

How to run the model, measure it the way the numbers in `POSITIVE_RESULTS.md` were measured, profile it,
check accuracy, and where every tuning knob lives. Everything here is relative to the tt-metal checkout
(branch `arg/pplx-embed-upstream`). Model directory: `models/demos/blackhole/pplx_embed_4b/` (`$M` below).

## 1. Environment

```bash
cd tt-metal
export TT_METAL_HOME=$PWD PYTHONPATH=$PWD HF_MODEL=perplexity-ai/pplx-embed-v1-4b MESH_DEVICE=P150
PY=./python_env/bin/python              # the system python3 has no torch/ttnn
export TT_VISIBLE_DEVICES=<chip>        # Galaxy host: one process per chip; the chip appears as device 0
```

- Weights: `HF_MODEL` (tensor cache under `model_cache/…/P150`, built on first run, ~2 min per process).
- Device reset on the shared Galaxy host: `tt-smi -r` only; never `tt-smi -glx_reset`.
- The P150s here expose a 12×10 = 120-worker grid (a p150a card exposes 13×10); every grid in the
  configs is clamped to the device.

## 2. Run the demo

Latency demo, one batch size per script (10 iterations, prints per-iteration and best/avg wall time of the
extended trace = forward + pooling + I/O in one replay):

```bash
$PY $M/demo/demo_bs1_isl512.py      # also bs4 / bs8 / bs32, and isl1024 / isl2048 variants
$PY $M/demo/demo_bs8_isl512.py
pytest $M/demo/demo_bs8_isl512.py -sv       # same thing under pytest
```

Same run with an arbitrary batch size and iteration count (what every A/B in the records used):

```bash
TT_VISIBLE_DEVICES=7 $PY $M/perf_tools/e2e_run_fp.py 8 10        # <batch> <iterations>
```

Serving / your own inputs: `$M/demo/live_demo.py` (see README §3.3), data-parallel over chips:
`$M/demo/dp32_multiprocess.py`.

## 3. Where the tuning lives

`$M/demo/_common.py::apply_workload_env(batch_size, seq_len)` sets every per-batch default with
`os.environ.setdefault`, so any knob is overridable from the shell (that is how every A/B was run).
Shipped defaults at ISL 512:

| batch | knobs |
|---|---|
| all | `QWEN_FUSED_HEADS_NORM=1` (head split + Q/K RMSNorm + RoPE in one generic op), `QWEN_FUSED_Q_BFP8=force`, `QWEN_FUSED_KV_BFP8=1`, `QWEN_QKV_OUT_BFP8=1`, `QWEN_FUSED_ADD_NORM=1`, `QWEN_MM_MAX_DIVISOR=38` |
| 1 | `QWEN_SDPA_Q_CHUNK=256 QWEN_SDPA_K_CHUNK=256`; legacy 2D matmuls on 12×8: `QWEN_QKV_GRID_X=12`, `QWEN_LEGACY_GRID_{FF13,FF2,WO}=12,8`, `QWEN_LEGACY_TIGHT_PER_CORE_N=1`, `QWEN_LEGACY_SUBBLOCK_K<k>_N<n>` (2×2 FF1/FF3, 2×1 FF2/WO, 1×4 QKV); `QWEN_SDPA_CONCAT_OUT_BS1=1` (SDPA writes `[1,1,S,H·d]` at bs1 too, model-local concat gone), `QWEN_BS1_RESID_SHARDED=1` (residual adds write the norm's 10×8 block-shard layout; the I2S before each norm is a no-op); `QWEN_SDPA_GQA_PACK=1` + `QWEN_SDPA_K_CHUNK=512` (SDPA `pack_gqa_heads`: one K/V stream per KV head), packed calls on q192 / 11×8: `QWEN_SDPA_GQA_PACK_Q_CHUNK=192 QWEN_SDPA_GQA_PACK_GRID=11,8`; `QWEN_FUSED_RESIDENT_CONSTS=1` (heads-op constants in a per-core L1 shard aliased to its CBs), `QWEN_FUSED_COMPUTE_V3=1` (heads-op phases batched across a unit's heads), `QWEN_BS1_NORM_SHARDED_OUT=1` (norm output stays 10×8 block-sharded; QKV/FF1/FF3 read it directly, no S2I) |
| >1 | `QWEN_SDPA_CONCAT_OUT=1` (SDPA writes `[B,1,S,H·d]`, no concat pass), `QWEN_SDPA_K_CHUNK=512`, `QWEN_FUSE_SWIGLU=1` (fused SwiGLU `minimal_matmul`, SwiGLU applied on the pack thread in the last K block); at ISL 512 the fused add+norm's operands in L1 (`QWEN_BATCHED_L1_INTERMEDIATES=1`): `TT_PREFILL_WO_L1=1 TT_PREFILL_FF2_L1=1 QWEN_FUSED_ADD_NORM_OUT_L1=1`, plus `QWEN_FUSED_ADD_NORM_SUM1_L1=1` at bs8/16 (at bs16 unless the QKV output is in L1 without `QWEN_FUSED_ADD_NORM_PREALLOC=1`); at bs8/16 the QKV output in L1 (`TT_PREFILL_QKV_L1=1`) and with it the v3 heads compute (`QWEN_FUSED_COMPUTE_V3=1`); SDPA K/V reuse on eligible calls with q128 (`QWEN_SDPA_REUSE_KV=1 QWEN_SDPA_REUSE_Q_CHUNK=128`); K / V out of the heads op in L1 (`QWEN_HEADS_KV_L1=1`, bs32 with 4 QKV chunks); heads-op cos / sin double-buffered (`QWEN_HEADS_ROT_DB=1`) and the add+norm partial exchange on its own NoC transaction id (`QWEN_ADD_NORM_PART_TRID=1`), both on in the op itself |
| 8 | `QWEN_SDPA_GRID=12,8 QWEN_SDPA_Q_CHUNK=512`, `QWEN_MM_BLOCK_FF2=16,8,8 QWEN_MM_BLOCK_QKV=8,4,8 QWEN_MM_BLOCK_WO=16,8,8`, `QWEN_MM_BLOCK_FF13=8,40,6 QWEN_MM_SUBBLOCK_FF13=1,6` (fused SwiGLU), `QWEN_FUSED_ADD_NORM_MIN_ROWS=4096 QWEN_FUSED_ADD_NORM_R=5` |
| 16 | `QWEN_SDPA_GRID=12,10`, `QWEN_MM_BLOCK_FF13=4,40,8 QWEN_MM_SUBBLOCK_FF13=1,8` (fused SwiGLU), `QWEN_FUSED_ADD_NORM_R=5`, `QWEN_FUSED_ADD_NORM_PREALLOC=1` (next layer's norm output allocated before FF2, at the top of L1) |
| 32 | `QWEN_SDPA_GRID=12,10`, `QWEN_MM_BLOCK_FF13=4,40,8 QWEN_MM_SUBBLOCK_FF13=1,8` (with the fused SwiGLU; fits with `QWEN_FUSED_ADD_NORM_PREALLOC_FF=1`: the post-attention norm output allocated before WO, at the top of L1), `QWEN_SILU_MUL=1` (unfused path only), `QWEN_FUSED_ADD_NORM_R=4`, `QWEN_QKV_CHUNKS=4 QWEN_FUSED_ADD_NORM_PREALLOC=1` (QKV + heads in four quarter-batch chunks, each output in L1, `tt/qkv_chunks.py`), `QWEN_WEIGHT_INTERLEAVED_K{2560_N6144,4096_N2560,2560_N9728}=1` |

Where the knobs are read: `models/tt_transformers/tt/model_config.py` (matmul program configs, SDPA
config, weight layout), `$M/tt/attention.py` (fused heads op, SDPA wrapper, concat), `$M/tt/mlp.py`
(fused SwiGLU / `silu_mul`), `$M/tt/decoder_fusion.py` (fused add + RMSNorm), `$M/tt/custom_ops/*`
(the model-local `generic_op` kernels). Opt-out knobs are documented next to each default in
`apply_workload_env`.

## 4. Measuring — the rules that made the numbers comparable

- **Rebuild after any host-side C++ change** (`./build_metal.sh`). Kernels JIT-compile from source, but program
  factories (CB sizes, compile args) live in the built `_ttnncpp.so`: new kernels on an old build deadlock instead of
  erroring (2026-09-30: every run hung in the first warmup prefill; NEGATIVE_RESULTS 67). Compare
  `ls -l build/lib/_ttnncpp.so` with `git log -1 --format=%ci -- <file>`. A live hang: `TT_VISIBLE_DEVICES=<chip>
  ./python_env/bin/python tools/tt-triage.py` while the process is still up (it halts cores: `tt-smi -r <chip>` after).
- **Same chip, sequential A/B.** Chip-to-chip spread is ≈1.5% cold and ≈5% sustained (power cap). Convention:
  bs1 → chip 4, bs8 → 7, bs16 → 8, bs32 → 6; other chips for standalone benches and accuracy.
  ```bash
  $M/perf_tools/ab_one.sh 8 7 "QWEN_NOOP=1" "QWEN_FUSED_ADD_NORM_R=10" 10   # <bs> <chip> "<ENV_A>" "<ENV_B>" [iters]
  ```
  Logs: `/tmp/ab_<bs>_c<chip>_{A,B}.log`, result line `RES ab …`.
- **Cold vs sustained.** "Best of 10" is the first iterations on a cool chip at 1.35 GHz; the board's power
  manager settles AICLK at ≈1.1–1.2 GHz within ~0.6 s of a batched load. Report both:
  ```bash
  $M/perf_tools/sustained_run.sh 32 6 30 mytag "QWEN_NOOP=1"   # <bs> <chip> <iters> <tag> "<ENV>"
  # RES sus … cold_best=… sustained_median=… aiclk=… power=…  (tt-smi sampled during the run)
  ```
- **bs16 flips between the two clock states mid-run**; compare arms with alternating launches:
  `$M/perf_tools/ab_multi.sh 16 8 "<A>" "<B>" 3`.
- Standalone op benches must reproduce the model's operand placement (bfp4 weights DRAM width-sharded over
  the 8 banks for bs1, interleaved for `minimal_matmul`; bfp8 activations in L1 at bs1, DRAM at bs>1;
  RoPE tables in L1). Below ≈20 µs/layer standalone, only the e2e A/B decides.
- Kernel `.cpp` edits are JIT-compiled by every process started after the edit — keep kernel sources frozen
  across both arms of a config A/B. Host C++ changes need `ninja -C build ttnn/_ttnncpp.so ttnn/_ttnn.so`,
  then copy `build/ttnn/*.so` over `build/lib/` and `ttnn/ttnn/_ttnn.so` (cp to a temp name + mv).

## 5. Profiling

Per-op device time (one trace replay), Tracy + device profiler:

```bash
TT_VISIBLE_DEVICES=0 TT_METAL_DEVICE_PROFILER=1 TT_METAL_PROFILER_PROGRAM_SUPPORT_COUNT=40000 \
  $PY -m tracy -p -r -v -m pytest $M/tests/perf/new_perf_bs8_isl512.py -sv
```

- Run profiles one at a time (concurrent tracy runs collide on port 8086 and lose their reports).
- `TT_METAL_PROFILER_PROGRAM_SUPPORT_COUNT=40000` is required or the batched-shape ops get no device data.
- Output: `generated/profiler/reports/<timestamp>/ops_perf_results_*.csv` (+ `profile_log_device.csv`,
  `tracy_profile_log_host.tracy`). Use `DEVICE KERNEL DURATION`, and take the last pass after an
  `EmbeddingsDeviceOperation` row (the Generator warm-up shapes precede it). Kernel durations are cycles
  converted at the nominal 1.35 GHz, so they read ~20% optimistic for sustained batched runs.
- `tt-perf-report <ops_perf_results.csv>` gives the per-op table used for the `ttperfreport_*.txt` files.
- The per-op profile page (claude.ai artifact `EyeLiogdyYu3soMry6akYn`): read the artifact (saves its HTML), write a
  `runs.json` with the new report timestamps and e2e numbers, then `python3 $M/perf_tools/profile_page.py <page.html>
  runs.json <out.html>` (keeps the page's Baseline level, rebuilds Optimized, adds the roofline fields) and republish.
  Its measured floors (`MEASURED_FLOORS`) are device kernel time: run a floor bench under
  `TT_METAL_DEVICE_PROFILER=1 TT_METAL_PROFILER_DIR=<dir>` and read it with `$M/perf_tools/device_kernel_us.py <dir>`
  (wall-clock µs per call include the trace's op-to-op gap, 3-8 µs here). Profile on a cool chip and check tt-smi
  holds 1350 MHz during the replay, or the DRAM floors do not compare with the durations.
- Per-op L1/DRAM buffer placement: `$PY $M/tests/perf/gen_mem_report.py --batch 8 --seq 512 --out /tmp/mem_bs8.csv` (turns off ttnn's fast runtime mode, without which the op hooks never fire); addresses and fragmentation: `$PY $M/perf_tools/l1_map_first_layer.py 16`.

Raw profiles and reports are on the host under `/home/ttuser/ashai/pplx-embedding-models/perf_csv/`
(not in the repo: 2–10 GB per folder):

- `visualizer_reports/pplx4b_FINAL_bs{1,8,16,32}_isl512/` — the shipped defaults at commit f970e9199d5,
  regenerated 2026-09-24 (ops CSV + `profile_log_device.csv` + tracy file + `ttperfreport_*.txt`; each folder
  loads into ttnn-visualizer as a Performance report). The 2026-09-23 set is under `superseded_20260923/`,
  the pre-optimization set under `pplx4b_baseline_bs*`.
- `ttperfreport_pplx4b_FINAL_bs{1,8,16,32}_isl512.txt` — rendered `tt-perf-report` tables (signpost window).
- `pplx4b_FINAL_bs{1,8,16,32}_isl512_ops.csv`, `pplx4b_FINAL_all_batches_comparison.csv` — per-op totals of
  the signposted replay (op, calls, total_ms, avg_ms, max_ms, %), same format as the `pplx4b_baseline_*` files.
- `doc/optimized/perf_summary.json` in the repo carries the same per-op breakdown next to the e2e numbers.

## 6. Accuracy

```bash
TT_VISIBLE_DEVICES=9 $PY $M/demo/eval_accuracy_tt.py                     # STS-B, batch-1 path, bucketed: 0.8161
TT_VISIBLE_DEVICES=10 $PY $M/demo/eval_accuracy_batched.py --batch 8 --save /tmp/embs_B8.pt   # batched path
TT_VISIBLE_DEVICES=11 $PY $M/demo/eval_accuracy_batched.py --batch 16 --compare /tmp/embs_B8.pt
```

The batched script runs STS-B through the exact batch-B perf configuration (fixed ISL 512, masked mean over
real tokens): 0.8121 / 0.8123 / 0.8140 / 0.8159 at batch 1 / 8 / 16 / 32. Per-op checks for batched-only
kernels: `QWEN_FUSED_ADD_NORM_VERIFY=1`, `QWEN_SILU_MUL_VERIFY=1` (per-call PCC vs the stock ops on live
tensors). End-of-model hidden-state cosines are not an equivalence test on this bfp8 pipeline.

## 7. Standalone benches (`$M/perf_tools/`, all `TT_VISIBLE_DEVICES=<chip> $PY <script>`)

| script | measures |
|---|---|
| `bench_bs1_legacy_blocks.py` (`ONLY=FF2`) | legacy 2D matmul in0_block_w × subblock sweep at M=512, sharded weights |
| `bench_bs1_model_mm.py` (`GRIDS=8x8,12x8`) | the four bs1 matmuls per grid: µs + PCC vs torch |
| `bench_bs1_wide_sharded.py`, `bench_bs1_wide_fine.py` | wider grids on the 8-bank sharded weights (numerics + time), 12×8 fine sweep |
| `bench_bs1_mm_bound.py` | what bounds the bs1 matmul: weight placement, fidelity, in0 dtype, grid |
| `bench_minimal_bs1.py`, `bench_batched_legacy_vs_minimal.py` | `minimal_matmul` vs legacy at M=512 and at M=4096/8192 |
| `bench_minimal_weight_layout.py` | interleaved vs width-sharded weights for `minimal_matmul` at bs8/16/32 shapes |
| `bench_sdpa_bs1.py`, `bench_sdpa_batched.py` (`BS=8`) | SDPA grid × q/k chunk × fp32-acc sweeps |
| `bench_add_norm_batched.py` | stock add + rms_norm vs fused / row-split add+RMSNorm (R sweep) |
| `bench_heads_bs16_ablate.py [batch]` (`QWEN_FUSED_COMPUTE_V3=1`, `ABL_ROPE=DRAM`) | fused heads op: full vs compute-only / data-movement-only / read-only / write-only scratch kernel variants, DRAM and L1 input |
| `bench_heads_placement_ablate.py <B_in> <B_out>` (`HQ_L1=1`, `HV_ONLY=`) | the same ablation at the model's batched placement: v3, L1 input, Q DRAM, K / V L1, preallocated outputs (`8 32` = a bs32 quarter chunk) |
| `bench_add_norm_placement.py <tag>` (`OUT_DIR=`) | row-split add+RMSNorm at the model's bs8 / 16 / 32 placement; saves outputs for bit-identity across env arms |
| `bench_heads_bs16_kernels.py [batch] [v1 v2 v3 path.cpp …]` | heads-op compute kernels side by side: traced µs (L1 / DRAM input) + PCC / exactness vs v1 |
| `bench_heads_bs16_phases.py [batch] [L1\|DRAM]` (`PH_MODE=unit`, `QWEN_FUSED_COMPUTE_V3=1`) | per-phase (v1) or per-unit wait/compute split of the heads compute from per-TRISC wall-clock accumulators (DPRINT), per core |
| `bench_qkv_mm_bs16_ablate.py [batch]` | QKV `minimal_matmul` at the model's config (bfp4 width-sharded weights; L1 and DRAM output) with the in0 / in1 reads and / or the output write skipped (patches `matmul_dataflow_common_metal2.hpp` per variant, restores it; PCC shows each patch compiled in) |
| `capture_qkv_call.py [batch]` (`CAP_N=9728,19456` for FF1/FF3 / fused w13) | the model's actual QKV (or other) `minimal_matmul` call: operand shapes, dtypes, memory configs (incl. shard specs), program config |
| `l1_map_first_layer.py [batch] [n_ops]` | live L1 buffers (lowest address, total, list) after each of the first `n_ops` ops of one eager prefill, next to the CB-region start: the room a program's static CBs have |
| `bench_mm_ablate.py <preset> [batch]` (`ff13`, `ff13fused`, `ff13plain`, `qkv`, `ff2`, `wo`; `MM_BLOCKS=`, `ABL_ONLY=`) | any model matmul at its per-batch config with reads / writes / both skipped |
| `bench_mm_sweep.py <preset> <batch> [top_n]` (`SWEEP=`) | plain `minimal_matmul` block / subblock sweep for a model projection at its operands (presets of `bench_mm_ablate.py`) |
| `bench_ff13_sweep.py <batch> [top_n]` (`SWEEP=`, `FF13_FP32=1`) | fused-SwiGLU FF13 block / subblock sweep at the model's operands, one device session |
| `bench_ff13_fused_ablate.py <batch>` (`MM_BLOCKS=`, `ABL_ONLY=`) | fused-SwiGLU FF13 at the model's blocks with the pack-thread SwiGLU skipped / doubled, the partial-sum add skipped, and / or reads and writes skipped |
| `bench_swiglu_epilogue.py [batch …]` (`EPI_PRESET=ff13fused`, `EPI_ONLY=`) | fused-SwiGLU epilogue variants (their `swiglu_block` patches apply to the kernel before §58; the landed in-loop path is `base`) with PCC vs base and vs an fp32 torch SwiGLU |
| `bench_swiglu_variants.py <batch> [xscale …]` (`MM_BLOCKS=`, `SW_ONLY=`, `SW_REPEAT=3`) | pack-thread SwiGLU SFPU pass variants (`swiglu_sfpu.hpp` patched) timed at the model's blocks, with their error vs an fp32 SwiGLU of the device's own pre-activations; `SW_REPEAT=3` exposes a pass's cost |
| `test_qkv_chunks.py` | bit-identity of the chunked bs32 path's ops: add+norm half-batch output pair, heads op per half batch at a batch offset |
| `test_heads_qsplit.py`, `test_sdpa_concat_out.py` | bit-identity + timing of the fused-heads Q split and the concat-free SDPA output |
| `bench_bs1_swiglu.py` | bs1 FF1 + FF3 + SwiGLU product at the model's call: stock mul vs `silu_mul` mode 3 on interleaved and on block-sharded FF1 / FF3 outputs, with each arm's error vs an fp32 SwiGLU (device times via `device_kernel_us.py`) |
| `parse_smi2.py <dev> <log>` | aligns tt-smi clock/power samples (`/tmp/smi_samples/*.json`) with iteration timestamps |

## 8. Records

- `POSITIVE_RESULTS.md` — every landed optimization with its measured effect; current cold/sustained table.
- `NEGATIVE_RESULTS.md` — every rejected experiment with numbers (§0–§49), including measurement corrections.
- `github_issues/` — the kernel/op asks filed on tenstorrent/tt-metal (#57626–#57630, #57722) with repro scripts.
- `../README.md` §5 — the long-form notes per landing.

## 9. Qwen3-Embedding-4B through the same stack

`Qwen/Qwen3-Embedding-4B` is the backbone pplx-embed-4B was trained from (2560 hidden, 9728 FFN, 36 layers,
32 Q / 8 KV heads, d 128), so every landing above applies unchanged. The two recipe differences follow
`HF_MODEL`: attention is causal (`apply_workload_env` sets `QWEN_SDPA_CAUSAL=1` for any non-pplx checkpoint,
so the SDPA wrapper leaves the base forward's `is_causal=True` alone and passes no pad mask), and the
embedding is the last real token with an EOS appended (the traced pipeline already slices the last-token
tile; the accuracy script has `--pool last --eos`).

```bash
export HF_MODEL=Qwen/Qwen3-Embedding-4B                    # weights from the HF cache; tensor cache built on first run
TT_VISIBLE_DEVICES=4 $PY $M/perf_tools/e2e_run_fp.py 1 10    # cold best of 10
bash $M/perf_tools/sustained_run.sh 8 7 30 q3e8 "HF_MODEL=Qwen/Qwen3-Embedding-4B"    # cold + sustained, tt-smi sampled
TT_VISIBLE_DEVICES=9 $PY $M/demo/eval_accuracy_batched.py --batch 8 --pool last --eos  # STS-B, Qwen recipe
IS_CAUSAL=1 TT_VISIBLE_DEVICES=3 $PY $M/perf_tools/bench_sdpa_bs1.py                   # SDPA sweeps under causal attention
IS_CAUSAL=1 TT_VISIBLE_DEVICES=3 $PY $M/perf_tools/bench_sdpa_batched.py
```

Measured 2026-09-24 (same chips and method as the pplx table in `POSITIVE_RESULTS.md`; the H200 reference
is the pplx-embed-4B H200 measurement, identical compute):

| batch | cold best | sustained (it 15–29) | × H200 cold / sustained | pplx-embed-4B cold / sustained | STS-B (last token + EOS) |
|---|---|---|---|---|---|
| 1 | 18.1 ms | 18.3 | 3.33× / 3.37× | 17.6 / 18.3 | 0.8190 |
| 8 | 115.7 | 121.9 | 3.50× / 3.68× | 115.3 / 121.0 | 0.8095 |
| 16 | 217.6 | 228.3 | 3.24× / 3.40× | 221.0 / 228.2 | 0.8076 |
| 32 | 426.9 | 445.5 | 3.07× / 3.20× | 425.5 / 451.1 | 0.8073 |

The per-batch SDPA gates are also the optimum under causal attention (bs1 8×8 q256/k256 with fp32
accumulation off, 54.5 µs; batched 12×8 q512/k512 at 237.6 / 454.5 / 804.4 µs for bs8 / 16 / 32 — the same
picks, and for the batched sweep the same times, as bidirectional), so there is no Qwen-specific gating.
The previous Qwen demo (`../qwen3_embedding_4b/`, README numbers) read 32.3 ms at bs1 and 725 ms at bs32.
