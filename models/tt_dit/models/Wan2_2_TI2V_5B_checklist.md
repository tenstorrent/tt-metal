# Wan2.2 TI2V-5B on Blackhole Galaxy — completion checklist

Status as of 2026-09-25 (branch `nkira/wan2.2-5B-i2v`, sprint-4 tip on `nkira/wan2.2-5B-i2v-step3`). Evidence links point at
`Wan2_2_TI2V_nadim_opt.md` (the optimisation notes, "notes" below), `Wan2_2_TI2V_5B.md` (Teja's
bring-up doc) and `~/nkira/Wan2_2_TI2V_5B_benchmarks.md` (the benchmark reference, outside git).
Legend: ✅ done, 🟡 done pending one action, ❌ open, ⛔ not done with a documented blocker.

## Functional completion

| item | status | evidence / action |
|---|---|---|
| Runs end-to-end on TT hardware | ✅ | `test_pipeline_performance_ti2v_5b` (T2V 720p/480p) and `_i2v` (720p) pass on the 4x8 BH Galaxy, warm-traced, 81 f / 40 steps. `test_pipeline_ti2v_5b_generate` runs 121 f at 720p. |
| Produces correct outputs on real inputs | ✅ | Transformer vs torch: PCC 99.9893 / 99.9894 % (scalar / per-token timestep), 100.0000 % two-row-vs-scalar (notes §9). VAE chunked decode PCC 1.0, `WanDupUp3D` bit-exact. I2V frame-0 vs seed image PCC 0.9984. CLIP prompt alignment 40.38 on Teja's 121 f generate (gate 36.00). Visual check of 720p frames (notes §8 on why CLIP is not a quality gate). |
| Working demo: real input → model → real output | ✅ | `models/tt_dit/demos/wan2_2_ti2v_5b_demo.py`: CLI, opens the mesh itself, T2V from `--prompt`, I2V from `--prompt --image`, writes mp4 + first/mid/last PNGs, honours `WAN5B_QUANT_CONFIG` and `WAN5B_TRACE_MODE`. Run with `HF_HOME=/mnt/tt-data/teja/hf python models/tt_dit/demos/wan2_2_ti2v_5b_demo.py --prompt "..." [--image <png>] --out <mp4> --repeat 1`. T2V (2026-09-23): 1280x704, 81 f, 40 steps, seed 42, pipeline ready 187.5 s, cold generation 17.10 s, **warm traced 11.57 s**, `/home/ttuser/wan5b_demo_t2v.mp4`. I2V (2026-09-24, conditioning image `/home/ttuser/wan5b_t2v_720p_bf8_first.png`): ready 203.5 s, cold 19.55 s, **warm 13.76 s**, `/home/ttuser/wan5b_demo_i2v.mp4`. Both with the 2cq trace mode, bf16. |

## Performance validation

| item | status | evidence / action |
|---|---|---|
| Performance within an acceptable range | ✅ | Default (bf16, 2cq, mean of 3, 2026-09-24/25): 720p T2V **11.50 s** e2e / 10.434 s denoise, I2V **13.49 s**, 480p **6.03 s**. Opt-in `WAN5B_QUANT_CONFIG=all_bf8_lofi`: 720p T2V **9.95 s** / 8.914 s denoise (PCC 99.965 %, CLIP 41.34). Sprint 3 for reference: 11.77 / 13.96 / 6.34 s. Benchmarks §4-5 place this against the 14B in tt-metal and against H100/H200/B200 and RTX 4090 figures, normalised per forward and per TFLOP; §0 states the target. |
| Results in the required tabular report format | 🟡 | The perf tests emit the CI `BenchmarkData` JSON (`save_partial_run_json`, `ml_model_name="Wan2.2-TI2V-5B"`, `run_type="BH_GLX"`, per-section `add_measurement` with targets) and print the section table (Text Encoding / Image Encode / Denoising / VAE Decoding / Total). Human tables: bring-up doc "Performance", notes §1, benchmarks §3. **Action:** confirm which template "required" means and transcribe if it differs. |
| Reported numbers validated with Tracy | ⛔ | Tracy works on this pipeline (notes §5: `TT_METAL_PROFILER_PROGRAM_SUPPORT_COUNT=20000`, 0 dropped markers) and device time per step is measured without it: blocking `execute_trace` = 258.5 ms of a 266.4 ms step (notes §7.5). The validation capture itself was attempted three times on 2026-09-25 and each died with `ENOSPC` on the root disk (notes §7.8): the profiler JIT-recompile plus the device log need more than the ~12 GB free. **Blocker is disk, not code.** Action: point `generated/profiler` (and ideally `~/.cache/tt-metal-cache`) at `/mnt/tt-data`, then run the block below. |
| Manual Tracy analysis (modules, blocks, ops) | ✅ / 🟡 | Ranking exists (notes §6): ring SDPA ~28 %, ff2 MMRS ~27 %, AGMM ~17 %, norms ~11 % of denoise; VAE conv3d 42 % of decode; every DiT matmul shape swept under Tracy (notes §7.1, 11 shapes, per-shape kernel time before/after). **Action:** the same Tracy capture above gives the op ranking from `DEVICE KERNEL DURATION` directly; paste its top-10 here. |

## Optimisation

| item | status | evidence / action |
|---|---|---|
| Trace support | ✅ | `combined_step` is captured with `traced_function` (`models/tt_dit/utils/tracing.py`); the whole 40-step denoise runs from one trace per step with per-step inputs updated in place (latent, timestep, guidance). Warm-traced is the gated configuration. `trace_region_size=150 MB`. Since 2026-09-24 the trace executes **non-blocking on two command queues** by default (`WAN5B_TRACE_MODE`, notes §7.5); `test_trace_modes_ti2v_5b` asserts blocking, nonblocking and 2cq are bit-identical. |
| 2CQ support | ✅ | Built 2026-09-24 as opt-in, then made the default (`b5dd469bd09`). The `Tracer` issues the per-call input copies on a second command queue fenced with events and runs `execute_trace` non-blocking; `WanPipeline.configure_trace_execution(blocking, input_cq_id)` selects the mode. Measured on full 81 f / 40 step runs (mean of 3): 720p T2V denoise 10.747 → **10.434 s** (-2.9 %, 7.8 ms/step), total 11.825 → **11.50 s**; 480p 5.649 → **5.350 s** (-5.3 %). The pre-build estimate (notes §7.5) put the strict ceiling at 1.4 ms/step and the outside bound at 7.8 ms/step; the result landed on the outside bound. Compute grid stays 12x10 with two queues, latents bit-identical (`test_trace_modes_ti2v_5b`). `WAN5B_TRACE_MODE=blocking` restores one queue. |
| Missing optimisations have documented blockers | ✅ | Notes §7 lists each remaining item with its measured ceiling: AdaLN chunk layout (host-only, traced runs do not pay it), `+1` fold (bf16 rounding), M=3424 sweep (needs ~75 GB kernel cache, ~12 GB free), `all_lofi` and `all_bf8_lofi_sdpa_lofi` (device hangs: LoFi on bf16 matmul operands and LoFi ring SDPA), `bf8_weights_sdpa_bf8` (preset exists, unmeasured alone), residual encoder on device (would remove the 1.33 s host image encode from I2V; not built), Tracy capture (disk). Quantisation is measured and opt-in: `all_bf8_lofi` is -15.5 % e2e at PCC 99.965 %, awaiting a visual sign-off to become the default (notes §7.7). |

## Hardware selection

| item | status | evidence |
|---|---|---|
| Target hardware justified | ✅ | Benchmarks §0 and §2: 5B dense DiT (30 blocks, d=3072, 24 heads), 48-channel 16x VAE; 720p/81 f is N=18480 tokens, so the model is sequence-parallel bound and wants a full Galaxy ring (SP=8 x TP=4). The 14B A14B reference already runs on the same mesh in tt-metal. |
| Data-driven review of existing references | ✅ | Benchmarks §4: tier 1 Wan2.2 14B in tt-metal, tier 2 other video DiTs in tt-metal (Mochi, LTX), tier 3 Wan2.2 on GPUs, tier 4 other video models on GPUs; §5 normalised comparison. |
| BH systems used whenever possible | ✅ | Single BH Galaxy (4x8) is the only target; all perf gates assert `is_blackhole()`. |

## Engineering due diligence

| item | status | evidence |
|---|---|---|
| Model architecture and execution characteristics | ✅ | Benchmarks §0 (one-page architecture, token counts, FLOP model); notes §6 (where the time goes, compute-bound denoise). |
| Closest existing TTNN implementation identified | ✅ | Wan2.2 T2V/I2V A14B in `models/tt_dit/models/transformers/wan2_2` and `pipelines/wan`; the 5B reuses the transformer, attention, VAE and pipeline with 5B-specific tables and the two-row per-token timestep path (bring-up doc "What already differs from 14B"). |
| Realistic performance target from references and GPU benchmarks | ✅ | Benchmarks §0 target and §4.3-4.4 GPU rows (RTX 4090 official, B200 figure with caveats) with normalisation in §5 and caveats in §6. |
| Expected bottlenecks and optimisation opportunities | ✅ | Notes §6 and §7; the matmul sweep and VAE rewrite came out of that list and delivered -29.8 % e2e at 720p (notes §1). |

## Device runs still needed for this checklist

1. Tracy validation capture (block in notes §5 / here), after freeing disk (notes §7.8):
   ```bash
   export TT_METAL_PROFILER_PROGRAM_SUPPORT_COUNT=30000
   WAN5B_GAP_PASSES=A WAN5B_GAP_STEPS=20 python_env/bin/python -m tracy -p -r -v -m pytest \
     "models/tt_dit/tests/models/wan2_2/test_step_gap_ti2v_5b.py::test_step_gap_ti2v_5b[blackhole-bh_4x8]" -sv --timeout=0
   python models/tt_dit/tests/models/wan2_2/tracy_summarize_ops.py generated/profiler/reports/<ts>/ops_perf_results_<ts>.csv --steps 24
   ```
   Expected: device kernel time per step within a few % of 258.5 ms (the text encoder's three
   calls add ≤ 0.3 s total to the capture, ≤ 12 ms/step at 24 steps; subtract or filter).
