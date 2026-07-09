# FIBO on TTNN — Perf dev-harness (stage-timing instrument)

**Date:** 2026-07-09
**Status:** Design approved (user-approved: chose B, then "implement spec and start working"), ready for implementation plan
**Target hardware:** Blackhole Quietbox (4× P150), 2×2 mesh
**Branch:** `fibo-pipeline` (perf follow-up on the completed sub-project 4)
**Scope:** a lightweight, developer-facing instrument that measures where wall-clock time goes in `BriaFiboPipeline` — a per-stage breakdown (encode / prepare / denoise it/s / decode / total) — so the documented perf follow-ups (tracing, batched+masked CFG, matmul-config tuning) can be validated with before/after numbers.

---

## Context
Sub-project 4 (end-to-end FIBO text→image pipeline, `models/tt_dit/pipelines/bria_fibo/pipeline_bria_fibo.py`) is functionally complete and PCC-validated on the 2×2 Blackhole mesh. It currently has **no perf instrumentation** — the recorded baseline (~1.5 it/s denoise, ~0.7s encode, ~10s on-device decode, 1024²/30-step, bf16) was gathered ad hoc. The remaining perf work (productionize tracing, batched+masked CFG for ~2×, matmul tuning) needs a repeatable way to see the current stage breakdown and measure deltas.

**How tt-metal measures perf (research summary).** Two orthogonal paths exist: (1) **host/e2e perf** — in-process wall-clock via a lightweight `Profiler` + `prep_perf_report`, or in `tt_dit` an `on_event` callback → `BenchmarkProfiler` consumed by `tests/models/*/test_performance_*.py` (warmups + measured runs, steps/s + images/s, per-mesh `expected_metrics` asserts, CI benchmark JSON); (2) **device perf** — on-silicon kernel time via the Tracy device profiler (`TT_METAL_DEVICE_PROFILER=1` under `python3 -m tracy`), `run_model_device_perf_test`, `ops_perf_results_*.csv`, needs a Tracy-enabled build. This spec deliberately builds **neither** of the heavier CI/device paths — it is the minimal host-side dev instrument, and is designed so it can grow into the `on_event`/CI shape later.

## Goal
Produce a per-stage wall-clock breakdown of one `BriaFiboPipeline` image generation, run by hand, that reproduces the current baseline and cleanly attributes time to encode / prepare / denoise / decode — measured on the **real** pipeline code path (not a replica).

### In scope
- A **behavior-preserving refactor** of `BriaFiboPipeline.__call__` into named stage methods (`_encode`, `_prepare`, `_denoise`; decode already exists as `_decode_latents`/`_gather_latent`) so `__call__` and the harness invoke the same code.
- A new **standalone harness** `models/tt_dit/tests/models/bria_fibo/test_performance_bria_fibo.py` that builds the pipeline, does 1 warmup + N measured passes driving the stage methods, times each with boundary syncs, and prints a breakdown (per-stage seconds + %, denoise it/s, overall images/s).
- Re-running the existing PCC/smoke/e2e tests to prove the refactor changed no behavior.

### Out of scope (YAGNI — documented follow-ups)
- Tracing / `Tracer` integration, `on_event`/`SectionStart` plumbing, `PipelineAPIMixin`.
- CI regression gate: no `expected_metrics` asserts, no `@pytest.mark.models_performance_bare_metal`, no `BenchmarkData`/`save_partial_run_json`, no benchmark JSON.
- Device-op (Tracy) profiling / per-op kernel breakdown.
- Per-step denoise timing (needs a callback — Approach C territory); v1 reports denoise total + derived it/s.
- Batched+masked CFG (a separate optimization; the harness will *measure* it once it exists).

## Design

### Why B (measure the real path, keep `__call__` free of timing code)
A stage *breakdown* needs stage boundaries inside `__call__`. Of the three options considered:
- **A** (harness reconstructs the stage sequence from sub-helpers, zero `__call__` change) measures a *replica* — it would silently keep measuring the old orchestration after `__call__` is optimized. Rejected: the instrument's whole job is validating changes to `__call__`.
- **C** (`on_event` markers in `__call__`) measures the real path and is the tt_dit convention, but sprinkles ~8 marker lines + a boundary sync into the hot path. Deferred to the CI-gate follow-up.
- **B** (extract stage methods; harness calls the same methods) measures the real path, keeps `__call__` free of any timing/event code, and the extraction is a genuine isolation improvement. **Chosen.**

### Part 1 — `__call__` refactor (no behavior change)
Extract the current `__call__` body into methods. `__call__` becomes a short sequence of calls; the returned tuples thread the state that currently lives in locals.

| Method | Current lines wrapped | Returns |
|---|---|---|
| `_encode(prompt, negative_prompt)` | `:204-205` — the two `SmolLM3TextEncoderWrapper.encode_prompt` forwards only | `(cond_embeds, cond_hidden_states, uncond_embeds, uncond_hidden_states)` |
| `_prepare(encoded, *, height, width, num_inference_steps, seed, latents)` | `:206-210` (`build_text_encoder_layers` + `_prepare_branch`×2, incl. RoPE recompute + host→device uploads), `:214-219` (schedule/shift), `:225` (`_random_latents`) | `(cond_branch, uncond_branch, timesteps, latent, spatial_sequence_length)` |
| `_denoise(cond_branch, uncond_branch, timesteps, latent, spatial_sequence_length, guidance_scale)` | `:229-248` — the loop (keeps its per-step `synchronize_device`) | final `latent` |
| decode | already `_decode_latents` (`:350`) / `_gather_latent` (`:336`) — unchanged | image or latent |

Splitting `_encode` from `_prepare` deliberately gives the **`prepare` bucket its own line**, exposing the RoPE-recompute + 92-tensor-upload per-call overhead (see below) instead of hiding it inside "encode".

**Per-call rebuild note (informs the buckets):** every `__call__` re-runs the two encoder forwards, recomputes RoPE cos/sin (`_pos_embed.forward(ids)` in `_prepare_branch`), rebuilds the 37→46 layer list, re-uploads `prompt` + 92 layer tensors (46×2 branches) + RoPE to device, recomputes the schedule, and regenerates+uploads latents. Only the model **weights**, submesh, `CCLManager`, `_pos_embed` module, scheduler/solver objects are built once in `__init__`. So a warmup run amortizes **op compilation**, not these rebuilds — encode/prepare/decode work correctly appears in every measured run. The deterministic RoPE/latent rebuild is itself a caching candidate the `prepare` bucket will surface.

### Part 2 — The harness
`models/tt_dit/tests/models/bria_fibo/test_performance_bria_fibo.py`, mirroring `test_pipeline.py`:
- Same fixtures: `mesh_device` `[(2, 2)]` indirect; `device_params` `{fabric_config: FABRIC_1D, l1_small_size: 32768, trace_region_size: 50000000}`; `_fibo_local()` checkpoint (skip if uncached).
- `test_fibo_pipeline_perf_breakdown` parametrized on `(height, width, num_inference_steps, num_measured_runs)`, default `(1024, 1024, 30, 3)`.
- Build the pipeline once. Define `time_stage(name, fn)`: `synchronize_device(submesh)` → `perf_counter()` → `fn()` → `synchronize_device(submesh)` → `perf_counter()`, record `dt`, return `fn`'s result. (The boundary sync is mandatory — TTNN dispatch is async; the denoise loop already syncs per step so denoise stays honest.)
- **1 warmup** pass driving the four stage methods (result discarded) to absorb op compile, then **N measured** passes, accumulating per-stage durations (lightweight local accumulator; avg + min/max across runs).
- Print a breakdown via `logger.info`: per-stage seconds + % of total, `denoise it/s = num_inference_steps / denoise_avg`, `images/s = 1 / total_avg`.
- **Sanity assert only**: the produced image has shape `(height, width, 3)` and is non-degenerate (reuse `test_pipeline.py`'s `std()`/`unique()` checks). No perf thresholds.
- On-demand: no CI perf marker; run explicitly by node id (matches the other on-demand fibo tests).

### Part 3 — Output shape (illustrative)
```
FIBO perf breakdown — 1024x1024, 30 steps, avg of 3 runs (after 1 warmup)
  encode     0.71 s   ( 2.3%)
  prepare    0.15 s   ( 0.5%)   RoPE recompute + 92-tensor upload
  denoise   19.30 s   (63.8%)   -> 1.55 it/s
  decode    10.10 s   (33.4%)
  ----------------------------------------------
  total     30.26 s             -> 0.033 images/s
```

## Testing / verification
- **Refactor is behavior-preserving** → gated by the existing `tests/models/bria_fibo/test_pipeline.py` (`test_fibo_pipeline_latent_pcc` at reduced res is the tight PCC gate; `test_fibo_pipeline_smoke` / `test_fibo_pipeline_e2e_image_golden` cover the full path). Re-run at least the latent-PCC + smoke tests after the refactor; PCC must be unchanged.
- **Harness** → run once, confirm it prints a well-formed breakdown, produces a valid `(1024,1024,3)` image, and the numbers land near the recorded baseline (~1.5 it/s denoise, encode <1s, decode ~10s).
- Run env (per project convention): `HF_HUB_OFFLINE=1 TT_METAL_HOME=$PWD PYTHONPATH=$PWD ARCH_NAME=blackhole python_env/bin/python -m pytest <nodeid> -v -s` (`-s` to see the breakdown print).

## Risks / mitigations
- **Refactor regression** — mitigated by the existing PCC gate; the extraction is mechanical (move locals into method params/returns, no logic change).
- **Inaccurate timings from async dispatch** — mitigated by the mandatory boundary `synchronize_device` in `time_stage`.
- **Harness drift** — eliminated by design: the harness calls the same stage methods `__call__` calls (the reason B was chosen over A).
