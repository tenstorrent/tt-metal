# FIBO on TTNN — Per-component device-op profiles (tt-perf-report per pipeline stage)

**Date:** 2026-07-10
**Status:** Implemented (component-only approach, after a rejected full-pipeline first attempt)
**Target hardware:** Blackhole Quietbox (4× P150), 2×2 mesh
**Branch:** `fibo-pipeline`
**Scope:** add three per-component device-op profile tests (`encode`, `denoise`, `decode`) to
`models/tt_dit/tests/models/bria_fibo/test_performance_bria_fibo.py`, each building **only** that component
with synthetic **production-shape** inputs, so `tt-perf-report` can be generated for each stage in isolation.

---

## Context / goal

`test_performance_bria_fibo.py` already had a whole-pipeline device profile
(`test_fibo_pipeline_device_profile`) whose CSV mixes every stage's ops into one report. The goal here is a
small, focused `tt-perf-report` for each compute-heavy stage — `encode` (SmolLM3), `denoise` (BriaFibo
transformer), `decode` (Wan VAE) — at production 1024² sizes. `prepare` is excluded (host-side RoPE +
uploads; no meaningful device ops).

## Decisions (user-approved)

* **Stages:** `encode`, `denoise`, `decode`, all co-located in `test_performance_bria_fibo.py`.
* **Build only the component, not the pipeline.** This is the load-bearing decision (see below). Each test
  builds just its component on the 2×2 mesh — the same parallel layout the pipeline uses — and feeds
  synthetic inputs at the real 1024² shapes.
* **Synthetic production-shape inputs.** `encode`: a short free-text prompt string ("a luxury sports car").
  `denoise`: random tensors at spatial seq 4096 / full 46 blocks / prompt seq 128. `decode`: a random
  `(1, 48, 1, 64, 64)` BCTHW latent → 1024². Op shapes (hence the device-op report) are input-shape-
  determined, so random values are fine.
* **1 warmup + 1 measured, untraced, Tracy-signposted.** One forward captures every unique device op, so a
  per-op report needs exactly one measured pass after a warmup.

### Why component-only (rejected: drive the pipeline's stage methods)

The first attempt built the full `BriaFiboPipeline` and drove its `_encode`/`_denoise`/`_decode_latents`
methods. It was rejected because **building the pipeline runs a full 2-step allocation generation in
`__init__`**, and under the Tracy profiler *that* gets fully captured — ~15k+ programs/pass, a **17–26 GB**
device log — which overwhelmed the host `tracy-capture` (the denoise CSV never generated) and, even when a
CSV did generate (encode), the trace volume dropped the signpost so `tt-perf-report` fell back to analyzing
the entire file. Building only the component (mirroring the existing
`test_transformer.py::test_fibo_transformer_mesh_profile`) removes the allocation-run bloat entirely, giving
a small, focused capture where the signpost propagates.

### Out of scope (YAGNI)
* A `prepare` device profile (host-bound).
* PCC/correctness asserts (covered by `test_pipeline.py` / `test_transformer.py` / `test_vae.py`); these are
  sanity-only.
* ttnn metal-trace profiling.
* Any change to `pipeline_bria_fibo.py`.

## Design

### One small helper
```
_profile_forward(mesh_device, header, forward) -> out
```
`forward` is a zero-arg callable that (re)builds its device inputs and runs the component once. The helper:
warmup `forward()` → sync → flush → signpost "measured begin" → measured `forward()` → sync → signpost
"measured end". The signpost + flush (`ttnn.ReadDeviceProfiler`) are inlined and **no-op under plain
pytest** (signpost only exists under `python -m tracy`; flush gated on `TT_METAL_PROFILER_MID_RUN_DUMP=1`).

### Three tests (each builds only its component on the 2×2 mesh)
| Test | Builds | Parallel layout (as in the pipeline) | Measured forward |
|---|---|---|---|
| `test_fibo_encode_device_profile` | `SmolLM3TextEncoderWrapper` | `EncoderParallelConfig.from_tuple((1,1))` (replicated) | `encode_prompt("a luxury sports car")` |
| `test_fibo_denoise_device_profile` | `BriaFiboCheckpoint.build()` (full 46 blocks) | `DiTParallelConfig.from_tuples(cfg=(1,0), sp=(2,0), tp=(2,1))` | one `model.forward(...)` at spatial seq 4096 |
| `test_fibo_decode_device_profile` | `WanVAEDecoderAdapter` | `VaeHWParallelConfig.from_tuples(height=(2,1), width=(2,0))` | `vae.decode((1,48,1,64,64), output_type="pt")` |

Fixtures: `mesh_device [(2,2)]` indirect, `device_params _PROFILE_DEVICE_PARAMS` (untraced → no trace
region), 1024² where applicable. Sanity: assert the forward produced non-None output.

### Profiling workflow (per stage; Tracy build required) — two steps
```
# 1. Record the raw device-op CSV (whole process: build + warmup + measured, one row per device).
TT_METAL_PROFILER_PROGRAM_SUPPORT_COUNT=6000 \
  HF_HUB_OFFLINE=1 TT_METAL_HOME=$PWD PYTHONPATH=$PWD ARCH_NAME=blackhole \
  python_env/bin/python -m tracy -r -p -v --dump-device-data-mid-run -m pytest \
  models/tt_dit/tests/models/bria_fibo/test_performance_bria_fibo.py::test_fibo_denoise_device_profile \
  --timeout=1800
# -> generated/profiler/reports/<ts>/ops_perf_results_<ts>.csv

# 2. Save ONLY the measured forward's ops to a per-stage folder (same name for --start/--end-signpost
#    brackets the "measured begin"/"measured end" pair; tt-perf-report does NOT mkdir the folder).
mkdir -p denoise_report
tt-perf-report generated/profiler/reports/<ts>/ops_perf_results_<ts>.csv \
  --start-signpost "fibo denoise" --end-signpost "fibo denoise" \
  --csv denoise_report/ops.csv
```
Per stage swap `::test`, the signpost (`fibo encode` / `fibo denoise` / `fibo decode`), and the folder
(`encode_report` / `denoise_report` / `decode_report`). Buffer default 6000 (verified for one full-depth
transformer forward); encode/decode are lighter.

**Op-count reduction (verified, denoise):** the raw CSV holds every op once per device (2×2 → ×4, merged by
tt-perf-report) plus the model build + compile-warmup forward before the signpost. The signpost keeps only
the measured forward: ~26k raw rows → /4 devices → ~6.5k logical ops → signpost → **2934** measured-forward
ops (`--csv` output row count matches exactly). Encode 885 / decode 204 by the same mechanism.

## Verification
* Plain pytest: all three pass on the real device path (~70 s total — no pipeline build).
* Tracy: at least one stage produces a small `ops_perf_results.csv` and `tt-perf-report` focuses on that
  stage's signpost (no "Device data missing" abort, no "no signposts found" fallback).
* Existing tests unchanged (no edits to `_perf_breakdown` / existing pipeline tests).
