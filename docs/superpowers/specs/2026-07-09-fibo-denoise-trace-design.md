# FIBO on TTNN — Trace the denoise transformer (flux1 pattern)

**Date:** 2026-07-09
**Status:** IMPLEMENTED + verified on 2x2 Blackhole. The design below is the *original* plan; the two-tracer approach it describes proved incorrect on device and was reworked to **one trace per step**. See "Bring-up outcome" immediately below for the final design and results; the original design text is kept for its reasoning.

---

## Bring-up outcome (supersedes the two-tracer parts of the Design below)

The denoise trace is implemented, correct, and faster. Final design (in `pipeline_bria_fibo.py`):

- **One trace per step**, not one per CFG branch. `_traced_step` runs *both* CFG forwards + the guidance `lerp` and returns the combined velocity; a single `Tracer` wraps it. Two separate traces sharing the submesh + CCLManager corrupted each other on replay (~0 PCC). A single trace per device is the pattern every other tt_dit pipeline uses.
- **Captured conditioning is reused on replay.** Fresh conditioning is passed only on the capturing call; every replay passes `tracer.inputs["cond"]`/`["uncond"]`. `ttnn.copy` of the replicated+sharded conditioning into the captured buffers on replay corrupts it (this is exactly why flux1 passes `inputs[...]` on replay). Only `latent`/`timestep` are copied each step.
- **Dedicated VAE CCLManager** (`self._vae_ccl_manager`), like wan/ltx's dit-vs-vae managers, so untraced VAE/latent-gather all-gathers don't desync the resident denoise trace's ping-pong buffers/semaphores.
- **Untraced allocation run in `__init__`** (like flux1), so the on-device VAE decode's buffers are allocated *before* any trace is captured — otherwise the first traced decode allocates during an active trace and corrupts the image.
- `prep_run=True` on the single tracer; re-capture keyed on `(cfg_on, cond_len, uncond_len)`; `trace_region_size` raised to 200 MB (one trace ≈ 70 MB; both CFG forwards resident).

**Verified (2x2 Blackhole, 1024², 30 steps, gs=5.0):** traced latent bit-exact vs untraced (PCC 1.0 across the capture + replay generations and after a prompt-length recapture); untraced golden-PCC gate unchanged (99.12% at gs=5.0, 99.77% at gs=1.0); traced vs untraced denoise **1.57 → 1.99 it/s (~27%)**, total 22.06 s → 18.11 s, decoded images identical. Modest because each step is compute-bound (~0.65 s), so dispatch overhead is a small fraction — but the trace path is now the substrate for the matmul-tuning / batched-CFG follow-ups.

---

**Original status:** Design approved (user chose Approach A + re-capture-per-prompt), ready for implementation plan
**Target hardware:** Blackhole Quietbox (4× P150), 2×2 mesh
**Branch:** `fibo-pipeline` (perf follow-up on the completed sub-project 4; builds on the perf-harness in `2026-07-09-fibo-perf-harness-design.md`)
**Scope:** Add opt-in TTNN tracing of the FIBO denoise transformer forward to `BriaFiboPipeline`, mirroring the `flux1` pipeline (the template FIBO was built from), to cut the denoise wall-clock (the ~64% bucket the perf harness measures). Solver + CFG-combine stay untraced; VAE/encoder tracing is a documented follow-up.

---

## Context

Sub-project 4 (`models/tt_dit/pipelines/bria_fibo/pipeline_bria_fibo.py`) is functionally complete and PCC-validated on the 2×2 Blackhole mesh, but **runs untraced** — the pipeline docstring says so explicitly (`pipeline_bria_fibo.py:24`: *"This first correctness pass runs UNTRACED (tracing is a documented follow-up)."*). The perf harness (`test_performance_bria_fibo.py`) shows the denoise loop is ~64% of wall-clock at ~1.5 it/s; decode is ~33%.

**Trace infra already exists and FIBO is built to accept it.** tt_dit centralizes TTNN trace in `models/tt_dit/utils/tracing.py` (`Tracer` class + `@traced_function` decorator). FIBO's pipeline mirrors `pipelines/flux1/pipeline_flux1.py`, which **already traces its denoise step** (`:172` tracers, `:207` allocation run, `:315-359` loop, `:420` `_traced_step`). The perf harness already reserves the trace region (`trace_region_size: 50_000_000`, `test_performance_bria_fibo.py:47`). FIBO's transformer forward is clean TTNN ops end-to-end (no host fallback, no `.item()`, no data-dependent Python branching) and already uses the trace-safe persistent-buffer CCL path (`all_gather_persistent_buffer`, `transformer_bria_fibo.py:433`).

**How `Tracer` works** (`utils/tracing.py`): first call captures the trace (after an optional `prep_run` untraced pass on cloned inputs, `:198-207`), later calls replay it and copy changed inputs into the captured buffers (`_update_input`, `:319-342`; same-buffer inputs skip the copy, `:337`). Outputs are reused-in-place across calls (`:31-39` caveat). Scalar inputs must compare *equal* across calls (`:340`); tensor inputs must match shape/dtype/layout (`:326`).

## Goal

Make `BriaFiboPipeline` optionally trace the denoise transformer forward, so a traced generation replays a captured graph each step instead of re-dispatching every op from host — cutting denoise wall-clock — while leaving the untraced path (and thus all existing PCC/smoke/e2e correctness gates) byte-for-byte unchanged.

### In scope
- A `traced: bool = False` flag on `__call__` and `_denoise`; untraced path keeps the current loop verbatim.
- A `_traced_step` method wrapping `self._transformer.forward(...)` (velocity prediction only).
- Two `Tracer`s (cond + uncond) because FIBO runs unpadded per-branch CFG at different token lengths.
- `prep_run=True` so each capture compiles all programs at the actual (variable) prompt shape *before* `begin_trace_capture` (the compile-before-capture hazard).
- Re-capture-on-shape-change: track captured `prompt_sequence_length` per branch; release + recapture when a new prompt changes it.
- A `release_traces()` method (mirrors `pipeline_wan.py:743`), called at harness teardown.
- A `traced` param on the perf harness so `test_fibo_pipeline_perf_breakdown` can print traced-vs-untraced denoise it/s.

### Out of scope (YAGNI — documented follow-ups)
- Tracing the VAE decode (the ~33% bucket; has a known all-gather `synchronize_device` gotcha — flux1 defaults `vae_traced=False`).
- Tracing the text encoder.
- Padding prompts to a fixed length for cross-prompt trace reuse (overlaps with the batched+masked-CFG follow-up; user chose re-capture-per-prompt instead).
- Folding the solver step into the trace via a device-tensor sigma-delta.
- 2-CQ input/replay overlap (`num_command_queues: 2`).
- Any CI perf gate / `expected_metrics` asserts.

## Design

### The captured unit — `_traced_step`
```python
def _traced_step(self, *, latent, prompt, timestep, layers, spatial_rope, prompt_rope,
                 spatial_sequence_length, prompt_sequence_length):
    return self._transformer.forward(
        spatial=latent, prompt=prompt, timestep=timestep, text_encoder_layers=layers,
        spatial_rope=spatial_rope, prompt_rope=prompt_rope,
        spatial_sequence_length=spatial_sequence_length,
        prompt_sequence_length=prompt_sequence_length,
    )
```
All args are `Tracer`-valid: tensors, a `list` of tensors (`layers`), `tuple`s of tensors (ropes), and `int` scalars (seq lengths) — `Tracer._tree_map` handles the nesting. The two seq-length scalars are constant for a fixed prompt+resolution, so they satisfy the scalar-equality check on replay.

### Tracer construction (`__init__`)
```python
self._tracers = {
    name: Tracer(self._traced_step, device=self._submesh, prep_run=True, clone_prep_inputs=True)
    for name in ("cond", "uncond")
}
self._captured_lengths = {"cond": None, "uncond": None}  # prompt_sequence_length last captured
```
`prep_run=True` runs one untraced forward at the real shape right before capture, compiling all programs first (fixes the variable-shape compile-before-capture hazard that flux1's fixed-length `""` allocation run avoids). `clone_prep_inputs=True` guards against the forward mutating inputs in the prep pass (to be confirmed harmless; `False` is a perf option if verified safe).

### The denoise loop (`_denoise`, traced path)
Mirrors `pipeline_flux1.py:315-359`. Untraced path (`traced=False`) is the current loop, unchanged.
```python
def _denoise(self, cond_branch, uncond_branch, timesteps, latent, spatial_sequence_length,
             guidance_scale, *, traced=False):
    submesh = self._submesh
    if not traced:
        ...  # current loop verbatim
        return latent

    branches = [("cond", cond_branch)] + ([("uncond", uncond_branch)] if uncond_branch is not None else [])
    self._ensure_traces(branches)  # release+recapture if a branch's prompt length changed

    for i, t in enumerate(tqdm.tqdm(timesteps)):
        timestep = tt_tensor.from_torch(
            torch.full((1, 1), float(t), dtype=torch.bfloat16), device=submesh, dtype=ttnn.bfloat16)
        vels = {}
        for name, branch in branches:
            tr = self._tracers[name]
            inp = tr.inputs  # captured buffers (populated after step 0)
            vels[name] = tr(
                latent=latent,                                                   # changing input, always passed
                prompt=branch["prompt"] if i == 0 else inp["prompt"],            # constants: real tensor on step 0,
                layers=branch["layers"] if i == 0 else inp["layers"],            #   captured buffer thereafter
                spatial_rope=branch["spatial_rope"] if i == 0 else inp["spatial_rope"],
                prompt_rope=branch["prompt_rope"] if i == 0 else inp["prompt_rope"],
                timestep=timestep,
                spatial_sequence_length=spatial_sequence_length,
                prompt_sequence_length=branch["prompt_sequence_length"],
                traced=True,
                tracer_blocking_execution=False,
            )
        # trace replay can clobber the passed latent object; the captured input buffer is the safe handle
        latent = self._tracers["cond"].inputs["latent"]
        if uncond_branch is not None:
            velocity = ttnn.lerp(vels["uncond"], vels["cond"], guidance_scale)
        else:
            velocity = vels["cond"]
        latent = self._solver.step(step=i, latent=latent, velocity_pred=velocity)  # untraced

    ttnn.synchronize_device(submesh)  # single sync at loop end (was per-step)
    return latent
```
Notes:
- The per-step `ttnn.deallocate(...)` calls from the untraced loop are dropped on the traced path — the transformer outputs (`vels[...]`) are tracer-managed (reused in place), and the latent handle is the captured buffer; explicit deallocation would corrupt the trace. Intermediate allocations inside `_traced_step` are internal to the trace.
- `tracer_blocking_execution=False` + a single end-of-loop sync lets the 30 steps pipeline (a win on top of trace itself; the untraced loop synced every step).

### Re-capture on shape change (`_ensure_traces`)
```python
def _ensure_traces(self, branches):
    for name, branch in branches:
        L = branch["prompt_sequence_length"]
        if self._captured_lengths[name] is not None and self._captured_lengths[name] != L:
            self._tracers[name].release_trace()
            self._tracers[name] = Tracer(self._traced_step, device=self._submesh,
                                         prep_run=True, clone_prep_inputs=True)
        self._captured_lengths[name] = L
```
For the perf harness (same prompt across the 3 measured runs) the first run captures, runs 2–3 replay. A different prompt releases and recaptures (amortized over its 30 steps). `spatial_sequence_length` is fixed by resolution, so only prompt length can change.

### `release_traces()`
```python
def release_traces(self):
    for name, tr in self._tracers.items():
        tr.release_trace()
        self._captured_lengths[name] = None
```

### Perf harness (`test_performance_bria_fibo.py`)
- Add `traced: bool` param to `_perf_breakdown` and thread it into the `_denoise` call.
- Parametrize (or add a second test / a `traced` axis) so the breakdown can be printed for `traced=False` and `traced=True`, exposing the denoise it/s delta. The warmup pass captures (traced); the N measured passes replay.
- Call `pipe.release_traces()` at the end (mirrors `test_performance_wan.py:313`).
- `trace_region_size` stays 50 MB (matches flux/SD35); two resident traces may need a bump — verify and adjust if capture OOMs.

## Testing / verification
- **Untraced path unchanged** → the existing `tests/models/bria_fibo/test_pipeline.py` PCC/smoke/e2e gates must still pass (they exercise `traced=False`). The traced path is additive; `__call__(traced=False)` is byte-for-byte the old code.
- **Traced correctness** → a traced generation must produce a valid, non-degenerate image (the harness's `std()`/`unique()` asserts), and ideally match the untraced latent within PCC. Add/allow a traced run of the smoke/latent-PCC test.
- **Perf** → run `test_fibo_pipeline_perf_breakdown` traced vs untraced; denoise it/s should improve and images/s should rise. No hard threshold (dev instrument).
- Run env (project convention): `HF_HUB_OFFLINE=1 TT_METAL_HOME=$PWD PYTHONPATH=$PWD ARCH_NAME=blackhole python_env/bin/python -m pytest <nodeid> -v -s`.

## Risks / mitigations
- **Compile-after-capture hang** (Z-Image gotcha) — mitigated by `prep_run=True` compiling all programs at the real shape before `begin_trace_capture`.
- **Variable prompt shape invalidating the trace** — mitigated by `_ensure_traces` release+recapture keyed on `prompt_sequence_length`.
- **Latent buffer clobbered by replay** — mitigated by using `tracer.inputs["latent"]` as the solver-step latent (flux1 pattern), not the passed object.
- **Trace region too small for two traces** — surfaced by capture OOM; bump `trace_region_size` in the harness/test device params.
- **In-place mutation during `prep_run`** — mitigated by `clone_prep_inputs=True`; revisit for speed only after correctness holds.
- **CCL persistent-buffer state across replays** — the same `CCLManager` persistent/ping-pong buffers that flux1/wan trace with; already the path FIBO uses, so no new exposure.
