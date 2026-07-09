# FIBO Denoise Trace Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add opt-in TTNN tracing of the FIBO denoise transformer forward to `BriaFiboPipeline`, cutting denoise wall-clock, while leaving the untraced path (and all existing correctness gates) unchanged.

**Architecture:** Mirror the `flux1` pipeline (the template FIBO was built from): wrap the transformer forward in tt_dit's `Tracer` (`models/tt_dit/utils/tracing.py`). Two tracers (cond + uncond) because FIBO runs unpadded per-branch CFG at different token lengths. Solver step and CFG `lerp` stay outside the trace. `prep_run=True` compiles all programs at the real prompt shape before capture; the pipeline releases + recaptures when a prompt's token length changes.

**Tech Stack:** Python, TTNN (`ttnn`), tt_dit `Tracer`, pytest on Blackhole 2×2 mesh.

## Global Constraints

- Run env for every test: `HF_HUB_OFFLINE=1 TT_METAL_HOME=$PWD PYTHONPATH=$PWD ARCH_NAME=blackhole python_env/bin/python -m pytest <nodeid> -v -s` (run from repo root `/localdev/mstojkovic/tt-metal`).
- The untraced path (`traced=False`, the default) must stay byte-for-byte the current code — it backs the PCC/smoke/e2e gates in `tests/models/bria_fibo/test_pipeline.py`.
- Do not add a per-op deallocation or `synchronize_device` to the traced loop beyond what this plan specifies (they corrupt / serialize the trace).
- Follow existing tt_dit style; `Tracer` lives at `models.tt_dit.utils.tracing.Tracer`.
- SPDX headers and existing import ordering are already in the files being modified — do not disturb them.

---

### Task 1: Add the tracing machinery to `BriaFiboPipeline`

**Files:**
- Modify: `models/tt_dit/pipelines/bria_fibo/pipeline_bria_fibo.py`
- Test (existing gate, run to confirm no regression): `models/tt_dit/tests/models/bria_fibo/test_pipeline.py`

**Interfaces:**
- Consumes: `models.tt_dit.utils.tracing.Tracer` — `Tracer(fn, *, device, prep_run, clone_prep_inputs)`; instance is callable `tracer(**kwargs, traced=True, tracer_blocking_execution=False)`; exposes `tracer.inputs` (dict keyed by the fn's kwarg names → captured input buffers) and `tracer.release_trace()`.
- Produces: `BriaFiboPipeline._traced_step(**kwargs) -> ttnn.Tensor`; `BriaFiboPipeline._ensure_traces(branches)`; `BriaFiboPipeline.release_traces()`; `_denoise(..., *, traced: bool = False)`; `__call__(..., traced: bool = False)`.

- [ ] **Step 1: Import `Tracer`**

Add to the imports block (near the other `models.tt_dit` imports, ~line 46-49):

```python
from models.tt_dit.utils.tracing import Tracer
```

- [ ] **Step 2: Construct the two tracers in `__init__`**

After the transformer is built and synced (`pipeline_bria_fibo.py:145`, right after `ttnn.synchronize_device(self._submesh)`), add:

```python
        # Denoise trace: one Tracer per CFG branch (cond/uncond run at different, unpadded prompt
        # token lengths, so each needs its own captured shape). prep_run=True compiles all programs
        # at the real prompt shape before begin_trace_capture (variable-shape compile-before-capture
        # safety). See docs/superpowers/specs/2026-07-09-fibo-denoise-trace-design.md.
        self._tracers = {
            name: Tracer(self._traced_step, device=self._submesh, prep_run=True, clone_prep_inputs=True)
            for name in ("cond", "uncond")
        }
        self._captured_lengths: dict[str, int | None] = {"cond": None, "uncond": None}
```

- [ ] **Step 3: Add `_traced_step`, `_ensure_traces`, and `release_traces`**

Insert these three methods immediately after `_run_transformer` (after `pipeline_bria_fibo.py:379`):

```python
    def _traced_step(
        self,
        *,
        latent: ttnn.Tensor,
        prompt: ttnn.Tensor,
        timestep: ttnn.Tensor,
        layers,
        spatial_rope,
        prompt_rope,
        spatial_sequence_length: int,
        prompt_sequence_length: int,
    ) -> ttnn.Tensor:
        """The unit captured by the Tracer: one branch's transformer forward (velocity prediction).

        All args are Tracer-valid (tensors, a list of tensors, tuples of tensors, int scalars). The
        two seq-length scalars are constant for a fixed prompt+resolution, satisfying the tracer's
        scalar-equality check on replay.
        """
        return self._transformer.forward(
            spatial=latent,
            prompt=prompt,
            timestep=timestep,
            text_encoder_layers=layers,
            spatial_rope=spatial_rope,
            prompt_rope=prompt_rope,
            spatial_sequence_length=spatial_sequence_length,
            prompt_sequence_length=prompt_sequence_length,
        )

    def _ensure_traces(self, branches: list[tuple[str, dict]]) -> None:
        """Release + rebuild a branch's tracer when its prompt token length changed since capture.

        A trace bakes tensor shapes, so it is only reusable while ``prompt_sequence_length`` is fixed
        (``spatial_sequence_length`` is fixed by resolution). Same prompt across runs -> reuse; a new
        prompt -> recapture (amortized over that generation's steps).
        """
        for name, branch in branches:
            length = branch["prompt_sequence_length"]
            if self._captured_lengths[name] is not None and self._captured_lengths[name] != length:
                self._tracers[name].release_trace()
                self._tracers[name] = Tracer(
                    self._traced_step, device=self._submesh, prep_run=True, clone_prep_inputs=True
                )
            self._captured_lengths[name] = length

    def release_traces(self) -> None:
        """Release both branch traces (call at teardown; mirrors pipeline_wan.py's release_traces)."""
        for name, tracer in self._tracers.items():
            tracer.release_trace()
            self._captured_lengths[name] = None
```

- [ ] **Step 4: Add the `traced` branch to `_denoise`**

Change the `_denoise` signature (`pipeline_bria_fibo.py:285-293`) to add a keyword-only `traced` flag:

```python
    def _denoise(
        self,
        cond_branch: dict,
        uncond_branch: dict | None,
        timesteps,
        latent: ttnn.Tensor,
        spatial_sequence_length: int,
        guidance_scale: float,
        *,
        traced: bool = False,
    ) -> ttnn.Tensor:
```

Then, at the very start of the method body (right after the docstring, before `logger.info("denoising...")` at `:301`), insert the traced branch. It returns early so the existing untraced loop below is untouched:

```python
        submesh = self._submesh

        if traced:
            return self._denoise_traced(
                cond_branch, uncond_branch, timesteps, latent, spatial_sequence_length, guidance_scale
            )

        logger.info("denoising...")
```

Note: the existing method already binds `submesh = self._submesh` at `:302`; remove that now-duplicate line so `submesh` is bound once at the top. The untraced loop below is otherwise unchanged.

- [ ] **Step 5: Add the `_denoise_traced` helper**

Insert immediately after `_denoise` (after its `return latent` at `:325`):

```python
    def _denoise_traced(
        self,
        cond_branch: dict,
        uncond_branch: dict | None,
        timesteps,
        latent: ttnn.Tensor,
        spatial_sequence_length: int,
        guidance_scale: float,
    ) -> ttnn.Tensor:
        """Traced denoise loop (mirrors pipeline_flux1.py:315-359).

        Captures each branch's transformer forward on step 0 and replays it on later steps. The
        constant conditioning (prompt/layers/ropes) is passed as the real tensor only on step 0 and as
        the captured buffer (``tracer.inputs[...]``) thereafter, so it is not re-uploaded. The changing
        ``latent`` is passed every step. CFG ``lerp`` and the Euler ``solver.step`` (per-step scalar)
        stay untraced. One sync at the end (not per step) so the steps pipeline.
        """
        logger.info("denoising (traced)...")
        submesh = self._submesh
        branches = [("cond", cond_branch)]
        if uncond_branch is not None:
            branches.append(("uncond", uncond_branch))
        self._ensure_traces(branches)

        for i, t in enumerate(tqdm.tqdm(timesteps)):
            timestep = tt_tensor.from_torch(
                torch.full((1, 1), float(t), dtype=torch.bfloat16), device=submesh, dtype=ttnn.bfloat16
            )
            velocities: dict[str, ttnn.Tensor] = {}
            for name, branch in branches:
                tracer = self._tracers[name]
                inp = tracer.inputs  # captured buffers, populated after step 0
                velocities[name] = tracer(
                    latent=latent,
                    prompt=branch["prompt"] if i == 0 else inp["prompt"],
                    timestep=timestep,
                    layers=branch["layers"] if i == 0 else inp["layers"],
                    spatial_rope=branch["spatial_rope"] if i == 0 else inp["spatial_rope"],
                    prompt_rope=branch["prompt_rope"] if i == 0 else inp["prompt_rope"],
                    spatial_sequence_length=spatial_sequence_length,
                    prompt_sequence_length=branch["prompt_sequence_length"],
                    traced=True,
                    tracer_blocking_execution=False,
                )

            # Trace replay can clobber the tensor object passed as `latent`; the captured input buffer
            # is the safe handle for the solver step (both branches copied the same value into it).
            latent = self._tracers["cond"].inputs["latent"]

            if uncond_branch is not None:
                velocity = ttnn.lerp(velocities["uncond"], velocities["cond"], guidance_scale)
            else:
                velocity = velocities["cond"]

            latent = self._solver.step(step=i, latent=latent, velocity_pred=velocity)

        ttnn.synchronize_device(submesh)
        return latent
```

- [ ] **Step 6: Thread `traced` through `__call__`**

Add `traced: bool = False` to the `__call__` signature (after `force_device_decode: bool = False,` at `:189`):

```python
        force_device_decode: bool = False,
        traced: bool = False,
```

And pass it to the `_denoise` call (`:213`):

```python
        latent = self._denoise(
            cond_branch, uncond_branch, timesteps, latent, spatial_sequence_length, guidance_scale, traced=traced
        )
```

- [ ] **Step 7: Sanity-check import + syntax**

Run:
```bash
cd /localdev/mstojkovic/tt-metal && python_env/bin/python -c "import ast; ast.parse(open('models/tt_dit/pipelines/bria_fibo/pipeline_bria_fibo.py').read()); print('parse-ok')"
```
Expected: `parse-ok`

- [ ] **Step 8: Confirm the untraced path is unregressed (PCC gate)**

Run the tight latent-PCC gate (untraced; reduced res, faster):
```bash
cd /localdev/mstojkovic/tt-metal && HF_HUB_OFFLINE=1 TT_METAL_HOME=$PWD PYTHONPATH=$PWD ARCH_NAME=blackhole \
  python_env/bin/python -m pytest models/tt_dit/tests/models/bria_fibo/test_pipeline.py::test_fibo_pipeline_latent_pcc -v -s
```
Expected: PASS with the same PCC as before this change (untraced code path is unchanged).

- [ ] **Step 9: Commit**

```bash
cd /localdev/mstojkovic/tt-metal
git add models/tt_dit/pipelines/bria_fibo/pipeline_bria_fibo.py
git commit -m "feat(fibo-pipeline): trace the denoise transformer forward (opt-in traced=)"
```

---

### Task 2: Wire `traced` into the perf harness and verify the speedup

**Files:**
- Modify: `models/tt_dit/tests/models/bria_fibo/test_performance_bria_fibo.py`

**Interfaces:**
- Consumes: `BriaFiboPipeline.__call__`/`_denoise` `traced=` flag and `release_traces()` from Task 1; `pipe._denoise(...)` is already called by `_perf_breakdown`.
- Produces: `_perf_breakdown(..., traced: bool = False)`; a `traced` parametrization on `test_fibo_pipeline_perf_breakdown`.

- [ ] **Step 1: Add `traced` to `_perf_breakdown` and thread it into the denoise call**

In `_perf_breakdown` (signature at `test_performance_bria_fibo.py:69-81`), add a keyword-only param:

```python
    num_measured_runs,
    negative_prompt="",
    traced=False,
):
```

Change the denoise `time_stage` call (`:111-114`) to pass `traced`:

```python
        latent = time_stage(
            "denoise",
            lambda: pipe._denoise(
                cond_branch, uncond_branch, timesteps, latent, ssl, guidance_scale, traced=traced
            ),
        )
```

- [ ] **Step 2: Release traces at the end of `_perf_breakdown`**

At the very end of `_perf_breakdown` (after the final `logger.info("\n".join(lines))` at `:170`), add:

```python
    if traced:
        pipe.release_traces()
```

- [ ] **Step 3: Reflect `traced` in the breakdown header**

In the header-building block, extend the `cfg_note` line (`:154`) so the log shows whether the run was traced. Replace `:154-158` with:

```python
    cfg_note = "CFG on (2 fwd/step)" if do_cfg else "no-CFG gate (1 fwd/step)"
    trace_note = "traced" if traced else "untraced"
    lines = [
        f"\nFIBO perf breakdown [{label}] — {width}x{height}, {num_inference_steps} steps, "
        f"gs={guidance_scale} [{cfg_note}, {trace_note}], avg of {num_measured_runs} runs (after 1 warmup)"
    ]
```

- [ ] **Step 4: Parametrize `test_fibo_pipeline_perf_breakdown` over `traced`**

Replace the decorators + signature of `test_fibo_pipeline_perf_breakdown` (`:173-176`) with a `traced` axis (untraced first so it always runs even if trace has issues), and pass it through:

```python
@pytest.mark.parametrize("mesh_device", [(2, 2)], indirect=["mesh_device"])
@pytest.mark.parametrize("device_params", [_DEVICE_PARAMS], indirect=["device_params"])
@pytest.mark.parametrize("height, width, num_inference_steps, num_measured_runs", [(1024, 1024, 30, 3)])
@pytest.mark.parametrize("traced", [False, True], ids=["untraced", "traced"])
def test_fibo_pipeline_perf_breakdown(*, mesh_device, height, width, num_inference_steps, num_measured_runs, traced):
```

And add `traced=traced,` to the `_perf_breakdown(...)` call inside it (after `num_measured_runs=num_measured_runs,` at `:193`):

```python
        num_measured_runs=num_measured_runs,
        traced=traced,
    )
```

- [ ] **Step 5: Sanity-check syntax**

Run:
```bash
cd /localdev/mstojkovic/tt-metal && python_env/bin/python -c "import ast; ast.parse(open('models/tt_dit/tests/models/bria_fibo/test_performance_bria_fibo.py').read()); print('parse-ok')"
```
Expected: `parse-ok`

- [ ] **Step 6: Run traced vs untraced and compare**

```bash
cd /localdev/mstojkovic/tt-metal && HF_HUB_OFFLINE=1 TT_METAL_HOME=$PWD PYTHONPATH=$PWD ARCH_NAME=blackhole \
  python_env/bin/python -m pytest "models/tt_dit/tests/models/bria_fibo/test_performance_bria_fibo.py::test_fibo_pipeline_perf_breakdown" -v -s
```
Expected: both `[untraced]` and `[traced]` params PASS, each producing a valid non-degenerate `(1024,1024,3)` image; the `[traced]` breakdown reports a higher denoise it/s (and higher images/s) than `[untraced]`. Capture both breakdown blocks from the log for the before/after comparison.

If capture OOMs on the trace region, bump `trace_region_size` in `_DEVICE_PARAMS` (`:47`) upward (e.g. `80000000`) and re-run.

- [ ] **Step 7: Commit**

```bash
cd /localdev/mstojkovic/tt-metal
git add models/tt_dit/tests/models/bria_fibo/test_performance_bria_fibo.py
git commit -m "test(fibo-pipeline): perf harness runs traced vs untraced denoise"
```

---

### Task 3: Verify traced correctness + cross-prompt recapture

**Files:**
- Uses (no new files): `models/tt_dit/tests/models/bria_fibo/test_pipeline.py`, the perf harness from Task 2.

**Interfaces:**
- Consumes: `__call__(traced=True)`, `_ensure_traces` (exercised by two different-length prompts on one pipeline instance), `release_traces()`.

- [ ] **Step 1: Traced image is valid (smoke)**

The Task 2 `[traced]` run already asserts the produced image is valid + non-degenerate (`std()` / `unique()` in `_perf_breakdown`). Confirm that assertion passed in the Task 2 Step 6 output. If a dedicated end-to-end smoke exists that accepts a flag, also run it traced:

```bash
cd /localdev/mstojkovic/tt-metal && HF_HUB_OFFLINE=1 TT_METAL_HOME=$PWD PYTHONPATH=$PWD ARCH_NAME=blackhole \
  python_env/bin/python -m pytest models/tt_dit/tests/models/bria_fibo/test_pipeline.py::test_fibo_pipeline_smoke -v -s
```
Expected: PASS (untraced smoke still green — proves Task 1 didn't disturb the default path).

- [ ] **Step 2: Exercise recapture-on-shape-change (manual, one process)**

Write a throwaway check in the scratchpad to confirm two different-length prompts on the *same* pipeline instance both succeed (first captures, second releases+recaptures). Save as `/tmp/claude-1211407608/-localdev-mstojkovic-tt-metal/0dbca910-8690-41e0-9489-0f9699824c06/scratchpad/check_recapture.py`:

```python
import os, ttnn
from models.tt_dit.pipelines.bria_fibo.pipeline_bria_fibo import BriaFiboPipeline, BriaFiboPipelineConfig
from huggingface_hub import snapshot_download

params = {"fabric_config": ttnn.FabricConfig.FABRIC_1D, "l1_small_size": 32768, "trace_region_size": 50000000}
md = ttnn.open_mesh_device(ttnn.MeshShape(2, 2), **params)
try:
    ckpt = snapshot_download(os.environ.get("FIBO_PATH", "briaai/FIBO"), local_files_only=True)
    pipe = BriaFiboPipeline(device=md, config=BriaFiboPipelineConfig.default(mesh_shape=md.shape, checkpoint_name=ckpt))
    img1 = pipe("a red cube", num_inference_steps=4, guidance_scale=1.0, traced=True, force_device_decode=True)
    img2 = pipe("a much longer prompt with many more tokens than the first one", num_inference_steps=4,
                guidance_scale=1.0, traced=True, force_device_decode=True)
    print("recapture-ok", img1[0].size, img2[0].size, pipe._captured_lengths)
    pipe.release_traces()
finally:
    ttnn.close_mesh_device(md)
```

Run:
```bash
cd /localdev/mstojkovic/tt-metal && HF_HUB_OFFLINE=1 TT_METAL_HOME=$PWD PYTHONPATH=$PWD ARCH_NAME=blackhole \
  python_env/bin/python /tmp/claude-1211407608/-localdev-mstojkovic-tt-metal/0dbca910-8690-41e0-9489-0f9699824c06/scratchpad/check_recapture.py
```
Expected: prints `recapture-ok (1024, 1024) (1024, 1024) {...}` with no hang/crash — proves `_ensure_traces` released and recaptured on the length change.

- [ ] **Step 3: Record results**

No commit (verification only). Note the traced-vs-untraced denoise it/s delta from Task 2 Step 6 and the recapture check outcome in the final summary to the user.

---

## Notes for the implementer

- `Tracer.__call__` reserves the kwarg names `traced`, `tracer_cq_id`, `tracer_blocking_execution`, `tracer_execute_on_capture`; everything else is forwarded to `_traced_step`. So the `tracer(latent=..., prompt=..., traced=True, tracer_blocking_execution=False)` call maps cleanly onto `_traced_step`'s kwargs.
- `tracer.inputs` is keyed by `_traced_step`'s parameter names (`"latent"`, `"prompt"`, `"layers"`, `"spatial_rope"`, `"prompt_rope"`, ...). It is empty until after the step-0 capture, which is why constants use `branch[...] if i == 0 else inp[...]`.
- The submesh is a single `MeshDevice`, so `tracer_blocking_execution=False` is honored and only one sync at loop end is needed.
- Do NOT `ttnn.deallocate` the tracer outputs (`velocities[...]`) or the captured latent buffer in the traced loop — they are tracer-managed and reused in place; deallocating corrupts the trace.
