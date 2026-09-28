# Wan2.2 TI2V-5B on Blackhole Galaxy — completion checklist

Status as of 2026-09-28 (branch `nkira/wan2.2-5B-i2v`, sprint-6 tip on `nkira/wan2.2-5B-i2v-step6`, sprint 5 on `-step5`, sprint 4 on `-step3`). Evidence links point at
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
| Performance within an acceptable range | ✅ | Default (bf16, 2cq, AdaLN modulation hoisted and in the split tile layout; mean of 3, 2026-09-28, sprint 6): 720p T2V **10.83 s** e2e / 9.787 s denoise, I2V **12.52 s** / 10.260 s, 480p **5.14 s** / 4.438 s (same-hour control of the sprint-5 file at 720p: 11.26 / 10.201 s). Sprint 5: 11.23 / 12.90 / 5.49 s; sprint 4: 11.50 / 13.49 / 6.03 s; sprint 3: 11.77 / 13.96 / 6.34 s. Opt-in `WAN5B_QUANT_CONFIG=all_bf8_lofi` on the sprint-6 tip: 720p T2V **9.38 s** / 8.328 s denoise (notes §7.7; sprint 5: 9.61 s). Gates still the sprint-5 means + 30 % (`test_performance_wan.py`), 26-33 % headroom. Benchmarks §4-5 place this against the 14B in tt-metal and against H100/H200/B200 and RTX 4090 figures, normalised per forward and per TFLOP; §0 states the target. |
| Results in the required tabular report format | 🟡 | The perf tests emit the CI `BenchmarkData` JSON (`save_partial_run_json`, `ml_model_name="Wan2.2-TI2V-5B"`, `run_type="BH_GLX"`, per-section `add_measurement` with targets) and print the section table (Text Encoding / Image Encode / Denoising / VAE Decoding / Total). Human tables: bring-up doc "Performance", notes §1, benchmarks §3. **Action:** confirm which template "required" means and transcribe if it differs. |
| Reported numbers validated with Tracy | ✅ | Sprint-6 tip capture of 2026-09-28 (`/mnt/tt-data/nkira/profiler/s6_f1_blocks6/reports/2026_09_28_21_15_41/`; 6 of 30 blocks so the host survives, 4 traced steps, 0 dropped markers): **45.86 ms device kernel time and 328 programs per replay per device** (sprint 5: 48.15 ms / 404) against 48.4 ms `execute_trace`; the `TilizeWithValPadding` / `UntilizeWithUnpadding` rows are gone and every matmul / SDPA / norm row is unchanged (notes §6, §7.9). Scaled to 30 blocks, ~223 ms kernel and ~1460 programs of the ~248 ms production step. Sprint-5 capture for reference: `/mnt/tt-data/nkira/profiler/blocks6/reports/2026_09_28_19_07_57/`. Full-model captures are not possible on this host (~1.2 TB of host RAM at the end-of-run read). |
| Manual Tracy analysis (modules, blocks, ops) | ✅ | Device ranking (notes §6, 720p bf16, sprint-6 tip): ring SDPA 30 %, the three N=768 AGMM projections 21 % (`to_out`, cross-attn q / out; sprint 6 measured them as bound by the TP-ring bytes, not by N: the MMRS form is only 4-18 us faster per call, notes §7.11), ff1 11 %, ff2 10 %, qkv 8 %, fused norms 11 %, cross-attention (to_kv + SDPA + heads) ~4 %; the AdaLN round trip (4-5 % in sprint 5) is gone (§7.9). VAE conv3d 42 % of decode (host profiler). Every DiT matmul shape swept under Tracy (notes §7.1), plus the MMRS form of the N=768 projections (§7.11). |

## Optimisation

| item | status | evidence / action |
|---|---|---|
| Trace support | ✅ | `combined_step` is captured with `traced_function` (`models/tt_dit/utils/tracing.py`); the whole 40-step denoise runs from one trace per step with per-step inputs updated in place (latent, timestep, guidance). Warm-traced is the gated configuration. `trace_region_size=150 MB`. Since 2026-09-24 the trace executes **non-blocking on two command queues** by default (`WAN5B_TRACE_MODE`, notes §7.5); `test_trace_modes_ti2v_5b` asserts blocking, nonblocking and 2cq are bit-identical. |
| 2CQ support | ✅ | Built 2026-09-24 as opt-in, then made the default (`b5dd469bd09`). The `Tracer` issues the per-call input copies on a second command queue fenced with events and runs `execute_trace` non-blocking; `WanPipeline.configure_trace_execution(blocking, input_cq_id)` selects the mode. Measured on full 81 f / 40 step runs (mean of 3): 720p T2V denoise 10.747 → **10.434 s** (-2.9 %, 7.8 ms/step), total 11.825 → **11.50 s**; 480p 5.649 → **5.350 s** (-5.3 %). The pre-build estimate (notes §7.5) put the strict ceiling at 1.4 ms/step and the outside bound at 7.8 ms/step; the result landed on the outside bound. Compute grid stays 12x10 with two queues, latents bit-identical (`test_trace_modes_ti2v_5b`). `WAN5B_TRACE_MODE=blocking` restores one queue. |
| Missing optimisations have documented blockers | ✅ | Notes §7 lists each remaining item with its measured ceiling: per-block AdaLN modulation hoist **done in sprint 5** (§7.2) and its tile/row-major round trip **removed in sprint 6** (bit-exact, -4.1 % denoise at 720p vs a same-hour control, -8.3 % at 480p, §7.9); the N=768 projections as fused matmul + reduce-scatter **measured and dropped** (4-18 us per call, the ops are bound by the TP-ring bytes, §7.11); the cross-attention residual fusion **measured and dropped** (no gain, §7.13); head split/merge (ring SDPA needs 4-D inputs, §7.13); `+1` fold (rounding, ~1.2 ms/step); the two-row expansion's misaligned row slice (~0.3 ms per I2V step, §7.12); M=3424 sweep (kernel cache now on NFS, so runnable); `all_lofi` and `all_bf8_lofi_sdpa_lofi` (device hangs); `bf8_weights_sdpa_bf8` (unmeasured alone); residual encoder on device (1.24 s host image encode in I2V; not built). Quantisation is measured and opt-in: `all_bf8_lofi` on the sprint-6 tip is 9.38 s at 720p (notes §7.7), awaiting a visual sign-off to become the default; its next lever is the dead `activation_dtype` (bf8 on the gather of the bandwidth-bound projections). |

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

None open: the Tracy validation capture below succeeded on 2026-09-28 (sprint 5, then again
on the sprint-6 tip as `s6_f1_blocks6`; notes §7.8 has the failed attempts and why each died).
Kept for the recipe:

1. Tracy validation capture:
   ```bash
   free -g   # a few hundred GB must be free; the end-of-run read holds every marker on the host
   WAN5B_GAP_BLOCKS=6 WAN5B_GAP_STEPS=4 WAN5B_GAP_PASSES=A TT_METAL_PROFILER_DISABLE_PUSH_TO_TRACY=1 \
     python_env/bin/python -m tracy -p -r -v -t 8086 -o /mnt/tt-data/nkira/profiler/blocks6 --op-support-count 30000 \
     -m pytest "models/tt_dit/tests/models/wan2_2/test_step_gap_ti2v_5b.py::test_step_gap_ti2v_5b[blackhole-bh_4x8]" -sv --timeout=0
   python models/tt_dit/tests/models/wan2_2/tracy_summarize_ops.py /mnt/tt-data/nkira/profiler/blocks6/reports/<ts>/ops_perf_results_<ts>.csv --steps 4 --traced-only
   ```
   Expected: with `--traced-only`, per-step device kernel time of the 6-block model; the block
   ops scale by 30/6 to the full model and should land within a few % of 258.5 ms once the
   non-block ops (patch embed, norm_out, proj_out, gathers, lerp) are added back unscaled.
