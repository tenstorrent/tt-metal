# Wan2.2 TI2V-5B on Blackhole Galaxy — completion checklist

Status as of 2026-09-22 (branch `nkira/wan2.2-5B-i2v`, sprint 3). Evidence links point at
`Wan2_2_TI2V_nadim_opt.md` (the optimisation notes, "notes" below), `Wan2_2_TI2V_5B.md` (Teja's
bring-up doc) and `~/nkira/Wan2_2_TI2V_5B_benchmarks.md` (the benchmark reference, outside git).
Legend: ✅ done, 🟡 done pending one action, ❌ open, ⛔ not done with a documented blocker.

## Functional completion

| item | status | evidence / action |
|---|---|---|
| Runs end-to-end on TT hardware | ✅ | `test_pipeline_performance_ti2v_5b` (T2V 720p/480p) and `_i2v` (720p) pass on the 4x8 BH Galaxy, warm-traced, 81 f / 40 steps. `test_pipeline_ti2v_5b_generate` runs 121 f at 720p. |
| Produces correct outputs on real inputs | ✅ | Transformer vs torch: PCC 99.9893 / 99.9894 % (scalar / per-token timestep), 100.0000 % two-row-vs-scalar (notes §9). VAE chunked decode PCC 1.0, `WanDupUp3D` bit-exact. I2V frame-0 vs seed image PCC 0.9984. CLIP prompt alignment 40.38 on Teja's 121 f generate (gate 36.00). Visual check of 720p frames (notes §8 on why CLIP is not a quality gate). |
| Working demo: real input → model → real output | 🟡 | `models/tt_dit/demos/wan2_2_ti2v_5b_demo.py` (added 2026-09-22): CLI, opens the mesh itself, T2V from `--prompt`, I2V from `--prompt --image`, writes mp4 + first/mid/last PNGs, honours `WAN5B_QUANT_CONFIG`. **Action:** one device run of each mode to record the command, timing and output path here. Until then the closest thing that has run is `test_pipeline_ti2v_5b_generate` (prompt → 121 f 720p → PNG previews; mp4 needs `imageio_ffmpeg`, see notes §7.7). |

## Performance validation

| item | status | evidence / action |
|---|---|---|
| Performance within an acceptable range | ✅ | 720p T2V 11.77 s e2e / 10.701 s denoise (mean of 3, 2026-09-22), I2V 13.96 s, 480p 6.34 s. Benchmarks §4-5 place this against the 14B in tt-metal and against H100/H200/B200 and RTX 4090 figures, normalised per forward and per TFLOP; §0 states the target. |
| Results in the required tabular report format | 🟡 | The perf tests emit the CI `BenchmarkData` JSON (`save_partial_run_json`, `ml_model_name="Wan2.2-TI2V-5B"`, `run_type="BH_GLX"`, per-section `add_measurement` with targets) and print the section table (Text Encoding / Image Encode / Denoising / VAE Decoding / Total). Human tables: bring-up doc "Performance", notes §1, benchmarks §3. **Action:** confirm which template "required" means and transcribe if it differs. |
| Reported numbers validated with Tracy | 🟡 | Tracy works on this pipeline (notes §5: `TT_METAL_PROFILER_PROGRAM_SUPPORT_COUNT=20000`, 0 dropped markers). Device time per step is already measured without Tracy: blocking `execute_trace` = 258.5 ms of a 266.4 ms step (notes §7.5). **Action:** one Tracy capture of `test_step_gap_ti2v_5b` (`WAN5B_GAP_PASSES=A`, `WAN5B_GAP_STEPS=20`) and `tracy_summarize_ops.py --steps 24` to show device kernel ms/step ≈ 258 ms; block below. |
| Manual Tracy analysis (modules, blocks, ops) | ✅ / 🟡 | Ranking exists (notes §6): ring SDPA ~28 %, ff2 MMRS ~27 %, AGMM ~17 %, norms ~11 % of denoise; VAE conv3d 42 % of decode; every DiT matmul shape swept under Tracy (notes §7.1, 11 shapes, per-shape kernel time before/after). **Action:** the same Tracy capture above gives the op ranking from `DEVICE KERNEL DURATION` directly; paste its top-10 here. |

## Optimisation

| item | status | evidence / action |
|---|---|---|
| Trace support | ✅ | `combined_step` is captured with `traced_function` (`models/tt_dit/utils/tracing.py`); the whole 40-step denoise runs from one trace per step with per-step inputs updated in place (latent, timestep, guidance). Warm-traced is the gated configuration. `trace_region_size=150 MB`. |
| 2CQ support | ⛔ | **Not implemented, by measurement.** A second command queue only overlaps host↔device transfers with compute. In this pipeline the per-step transfers are two scalars (timestep, guidance) and one device-side latent copy; everything else stays on device for all 40 steps and the video is read back once. Measured on the traced production path (notes §7.5): host-only time between traces is **1.37 ms/step of 266.35 ms (0.5 %)**, and that figure already contains the input writes, the solver dispatch and `execute_trace` re-entry. The whole-pipeline transfers outside the loop (T5 upload, one latent upload/readback, ~220 MB uint8 video readback) are one-off and inside the ~1 s non-denoise time. Ceiling for 2CQ is therefore < 0.5 % of denoise and < 1 % end to end, below the 0.4-0.8 % run-to-run spread. Cost would be `num_command_queues=2` in `DEVICE_PARAMS`, `tracer_cq_id` plumbing and event fences around `_update_input` and the solver, with hang risk on a 32-chip mesh. Model-specific blocker: **there is no transfer time to hide**; the denoise is compute-bound (notes §6). Revisit only if the trace shortens by >10x or the VAE encoder/decoder move to a streaming design. |
| Missing optimisations have documented blockers | ✅ | Notes §7 lists each remaining item with its measured ceiling: non-blocking trace (1.4 ms/step, skipped), AdaLN chunk layout (host-only, traced runs do not pay it), `+1` fold (bf16 rounding), M=3424 sweep (needs ~75 GB kernel cache, 14 GB free), `all_lofi` (device hang in the first LoFi matmul), residual encoder on device (would remove the 1.38 s host image encode from I2V; not built). |

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

1. Demo, T2V and I2V, one run each (records command, time, output path).
2. Tracy validation capture (block in notes §5 / here):
   ```bash
   export TT_METAL_PROFILER_PROGRAM_SUPPORT_COUNT=30000
   WAN5B_GAP_PASSES=A WAN5B_GAP_STEPS=20 python_env/bin/python -m tracy -p -r -v -m pytest \
     "models/tt_dit/tests/models/wan2_2/test_step_gap_ti2v_5b.py::test_step_gap_ti2v_5b[blackhole-bh_4x8]" -sv --timeout=0
   python models/tt_dit/tests/models/wan2_2/tracy_summarize_ops.py generated/profiler/reports/<ts>/ops_perf_results_<ts>.csv --steps 24
   ```
   Expected: device kernel time per step within a few % of 258.5 ms (the text encoder's three
   calls add ≤ 0.3 s total to the capture, ≤ 12 ms/step at 24 steps; subtract or filter).
