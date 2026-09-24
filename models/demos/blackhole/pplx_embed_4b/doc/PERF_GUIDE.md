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
| 1 | `QWEN_SDPA_Q_CHUNK=256 QWEN_SDPA_K_CHUNK=256`; legacy 2D matmuls on 12×8: `QWEN_QKV_GRID_X=12`, `QWEN_LEGACY_GRID_{FF13,FF2,WO}=12,8`, `QWEN_LEGACY_TIGHT_PER_CORE_N=1`, `QWEN_LEGACY_SUBBLOCK_K<k>_N<n>` (2×2 FF1/FF3, 2×1 FF2/WO, 1×4 QKV) |
| >1 | `QWEN_SDPA_CONCAT_OUT=1` (SDPA writes `[B,1,S,H·d]`, no concat pass), `QWEN_SDPA_K_CHUNK=512`, `QWEN_FUSE_SWIGLU=1` at bs8/16 |
| 8 | `QWEN_SDPA_GRID=12,8 QWEN_SDPA_Q_CHUNK=512`, `QWEN_MM_BLOCK_FF2=16,8,8 QWEN_MM_BLOCK_QKV=8,4,8 QWEN_MM_BLOCK_WO=16,8,8`, `QWEN_FUSED_ADD_NORM_MIN_ROWS=4096 QWEN_FUSED_ADD_NORM_R=5` |
| 16 | `QWEN_SDPA_GRID=12,10`, `QWEN_MM_BLOCK_FF13=4,20,8 QWEN_MM_SUBBLOCK_FF13=1,4`, `QWEN_FUSED_ADD_NORM_R=5` |
| 32 | `QWEN_SDPA_GRID=12,10`, `QWEN_SILU_MUL=1`, `QWEN_FUSED_ADD_NORM_R=4`, `QWEN_WEIGHT_INTERLEAVED_K{2560_N6144,4096_N2560,2560_N9728}=1` |

Where the knobs are read: `models/tt_transformers/tt/model_config.py` (matmul program configs, SDPA
config, weight layout), `$M/tt/attention.py` (fused heads op, SDPA wrapper, concat), `$M/tt/mlp.py`
(fused SwiGLU / `silu_mul`), `$M/tt/decoder_fusion.py` (fused add + RMSNorm), `$M/tt/custom_ops/*`
(the model-local `generic_op` kernels). Opt-out knobs are documented next to each default in
`apply_workload_env`.

## 4. Measuring — the rules that made the numbers comparable

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
- Per-op L1/DRAM buffer placement: `$PY $M/tests/perf/gen_mem_report.py --batch 8 --seq 512 --out /tmp/mem_bs8.csv`.

Raw profiles and reports collected so far are on the host under
`/home/ttuser/ashai/pplx-embedding-models/perf_csv/` (`visualizer_reports/pplx4b_20260923_bs{8,32}_isl512/`,
`ttperfreport_*.txt`, baseline op CSVs); they are not in the repo (tens of MB each).

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
| `test_heads_qsplit.py`, `test_sdpa_concat_out.py` | bit-identity + timing of the fused-heads Q split and the concat-free SDPA output |
| `parse_smi2.py <dev> <log>` | aligns tt-smi clock/power samples (`/tmp/smi_samples/*.json`) with iteration timestamps |

## 8. Records

- `POSITIVE_RESULTS.md` — every landed optimization with its measured effect; current cold/sustained table.
- `NEGATIVE_RESULTS.md` — every rejected experiment with numbers (§0–§49), including measurement corrections.
- `github_issues/` — the kernel/op asks filed on tenstorrent/tt-metal (#57626–#57630, #57722) with repro scripts.
- `../README.md` §5 — the long-form notes per landing.
