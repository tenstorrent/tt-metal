# FIBO Per-Component Device-Op Profiles — Implementation Plan (as built)

> Design: `docs/superpowers/specs/2026-07-10-fibo-per-stage-device-profile-design.md`

**Goal:** Add three per-component device-op profile tests (`encode`, `denoise`, `decode`) to
`models/tt_dit/tests/models/bria_fibo/test_performance_bria_fibo.py`, each building **only** that component
with synthetic **production-shape** inputs so `tt-perf-report` can be generated per stage in isolation.

**Note on approach change:** a first attempt drove the full `BriaFiboPipeline`'s stage methods; it was
reverted because building the pipeline runs a 2-step allocation generation that, under Tracy, produced a
17–26 GB device log, overwhelmed the host capture, and dropped the signpost. The as-built approach builds
only the component (mirroring `test_transformer.py::test_fibo_transformer_mesh_profile`).

**Only file modified:** `models/tt_dit/tests/models/bria_fibo/test_performance_bria_fibo.py` (additive; the
existing wall-clock + full-pipeline tests are untouched).

---

### Task 1: `_profile_forward` helper + three component tests

**Files:** Modify `models/tt_dit/tests/models/bria_fibo/test_performance_bria_fibo.py`

- [x] **Step 1: Add `_profile_forward(mesh_device, header, forward)`** — 1 warmup + 1 signposted measured
  pass; inlined `_signpost`/`_flush` that no-op under plain pytest (flush gated on
  `TT_METAL_PROFILER_MID_RUN_DUMP=1`).
- [x] **Step 2: `test_fibo_encode_device_profile`** — builds `SmolLM3TextEncoderWrapper` (replicated),
  `encode_prompt("a luxury sports car")`, signpost `"fibo encode"`.
- [x] **Step 3: `test_fibo_denoise_device_profile`** — builds `BriaFiboCheckpoint.build()` (full 46 blocks),
  one `model.forward(...)` at spatial seq 4096 / prompt seq 128 with synthetic inputs (sp=2, tp=2), signpost
  `"fibo denoise"`.
- [x] **Step 4: `test_fibo_decode_device_profile`** — builds `WanVAEDecoderAdapter` (hw-parallel over 2×2),
  `vae.decode((1,48,1,64,64), output_type="pt")`, signpost `"fibo decode"`.
- [x] **Step 5: Update the module docstring** to point at the per-component section.

**Verify collection (no hardware):**
```
TT_METAL_HOME=$PWD PYTHONPATH=$PWD ARCH_NAME=blackhole python_env/bin/python -m pytest \
  models/tt_dit/tests/models/bria_fibo/test_performance_bria_fibo.py --collect-only -q
```
Expected: the three `test_fibo_{encode,denoise,decode}_device_profile` are collected.

### Task 2: Verify

- [x] **Plain pytest (real device path):** run the three tests; all pass (~70 s total, no pipeline build).
```
HF_HUB_OFFLINE=1 TT_METAL_HOME=$PWD PYTHONPATH=$PWD ARCH_NAME=blackhole python_env/bin/python -m pytest \
  models/tt_dit/tests/models/bria_fibo/test_performance_bria_fibo.py::test_fibo_encode_device_profile \
  models/tt_dit/tests/models/bria_fibo/test_performance_bria_fibo.py::test_fibo_denoise_device_profile \
  models/tt_dit/tests/models/bria_fibo/test_performance_bria_fibo.py::test_fibo_decode_device_profile -v -s
```
- [ ] **Tracy (one stage):** confirm a small `ops_perf_results.csv` and that `tt-perf-report` focuses on the
  stage signpost (see the spec's profiling command).

### Task 3: Commit (when the user asks)
Single additive commit to the test file (+ updated spec/plan docs).
