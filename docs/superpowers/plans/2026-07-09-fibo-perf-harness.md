# FIBO Perf Dev-Harness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a standalone, on-demand stage-timing instrument for `BriaFiboPipeline` that prints a per-stage wall-clock breakdown (encode / prepare / denoise it/s / decode / total), measuring the real pipeline code path.

**Architecture:** Approach B — first do a behavior-preserving refactor of `BriaFiboPipeline.__call__` into named stage methods (`_encode`, `_prepare`, `_denoise`; decode already exists), gated by the existing PCC test. Then add a new test file that builds the pipeline, does 1 warmup + N measured passes driving those same stage methods, times each with boundary device syncs, and logs the breakdown. `__call__` gains zero timing code; the harness can never drift from the real path because it calls the same methods.

**Tech Stack:** Python, TTNN, pytest, loguru, numpy; 2×2 Blackhole mesh (`briaai/FIBO` weights).

## Global Constraints
- Run env (from repo root): `HF_HUB_OFFLINE=1 TT_METAL_HOME=$PWD PYTHONPATH=$PWD ARCH_NAME=blackhole python_env/bin/python -m pytest <nodeid> -v -s` (`-s` shows the breakdown log). Interpreter is `python_env/bin/python` (no `python` on PATH).
- The `__call__` refactor MUST be behavior-preserving: no logic change, only extraction of locals into method params/returns. Existing PCC gate `test_fibo_pipeline_latent_pcc` (512²/4-step, threshold 0.99, ~3 min) MUST still pass with unchanged PCC.
- The harness is a **dev instrument, not a CI gate**: NO `@pytest.mark.models_performance_bare_metal`, NO `expected_metrics`/threshold asserts, NO `BenchmarkData`/benchmark JSON. Sanity asserts only (image shape + non-degenerate).
- Follow `test_pipeline.py` conventions verbatim: `mesh_device` `[(2, 2)]` indirect; `device_params` `{"fabric_config": ttnn.FabricConfig.FABRIC_1D, "l1_small_size": 32768, "trace_region_size": 50000000}`; `_fibo_local()` checkpoint helper (skip if uncached).
- Precede every host-side timer boundary with `ttnn.synchronize_device(submesh)` (TTNN dispatch is async).

---

### Task 1: Behavior-preserving refactor of `__call__` into stage methods

**Files:**
- Modify: `models/tt_dit/pipelines/bria_fibo/pipeline_bria_fibo.py:176-257` (`__call__`; add `_encode`, `_prepare`, `_denoise`)
- Test (gate, existing — not modified): `models/tt_dit/tests/models/bria_fibo/test_pipeline.py::test_fibo_pipeline_latent_pcc`

**Interfaces:**
- Consumes: existing `self._text_encoder.encode_prompt`, `build_text_encoder_layers`, `self._prepare_branch`, `self._scheduler`, `self._solver`, `self._random_latents`, `self._run_transformer`, `_calculate_shift`, `self._num_blocks`, `self._parallel_config`, `self._submesh`.
- Produces (relied on by Task 2's harness — exact signatures):
  - `_encode(self, prompt: str, negative_prompt: str) -> tuple` → `(cond_embeds, cond_hidden_states, uncond_embeds, uncond_hidden_states)`
  - `_prepare(self, encoded: tuple, *, height: int, width: int, num_inference_steps: int, seed: int, latents) -> tuple` → `(cond_branch, uncond_branch, timesteps, latent, spatial_sequence_length)`
  - `_denoise(self, cond_branch: dict, uncond_branch: dict, timesteps, latent: ttnn.Tensor, spatial_sequence_length: int, guidance_scale: float) -> ttnn.Tensor`

- [ ] **Step 1: Establish the PCC gate passes BEFORE refactor (baseline)**

Run:
```bash
HF_HUB_OFFLINE=1 TT_METAL_HOME=$PWD PYTHONPATH=$PWD ARCH_NAME=blackhole \
  python_env/bin/python -m pytest \
  models/tt_dit/tests/models/bria_fibo/test_pipeline.py::test_fibo_pipeline_latent_pcc -v
```
Expected: PASS (records the pre-refactor PCC, ~0.99+, in the log). Note the PCC value.

- [ ] **Step 2: Extract the three stage methods; rewrite `__call__` to call them**

Replace the current `__call__` body (`pipeline_bria_fibo.py:176-257`) with the version below, and add the three new methods immediately after `__call__` (before `_prepare_branch`). This is a pure move of existing statements — no logic changes.

```python
    @torch.no_grad()
    def __call__(
        self,
        prompt: str,
        *,
        negative_prompt: str = "",
        height: int | None = None,
        width: int | None = None,
        num_inference_steps: int = 30,
        guidance_scale: float = 5.0,
        seed: int = 0,
        latents: torch.Tensor | None = None,
        output_type: str = "pil",
        force_device_decode: bool = False,
    ) -> list[Image.Image] | torch.Tensor:
        height = height if height is not None else self._height
        width = width if width is not None else self._width

        assert height % (_VAE_SCALE_FACTOR) == 0 and width % (_VAE_SCALE_FACTOR) == 0

        # 1-3. Encode, then build per-branch conditioning + schedule + latents.
        encoded = self._encode(prompt, negative_prompt)
        cond_branch, uncond_branch, timesteps, latent, spatial_sequence_length = self._prepare(
            encoded,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            seed=seed,
            latents=latents,
        )

        # 4. Denoise loop (CFG per step).
        latent = self._denoise(
            cond_branch, uncond_branch, timesteps, latent, spatial_sequence_length, guidance_scale
        )

        # 5. Return the pre-VAE latent (PCC gate) or decode to an image.
        if output_type == "latent":
            logger.info("returning pre-VAE latent...")
            return self._gather_latent(latent)
        logger.info("decoding image...")
        return self._decode_latents(
            latent, height=height, width=width, output_type=output_type, force_device_decode=force_device_decode
        )

    def _encode(self, prompt: str, negative_prompt: str) -> tuple:
        """Encode positive and negative prompts SEPARATELY (per-branch, true token lengths)."""
        logger.info("encoding prompts...")
        cond_embeds, cond_hidden_states = self._text_encoder.encode_prompt(prompt)
        uncond_embeds, uncond_hidden_states = self._text_encoder.encode_prompt(negative_prompt)
        return cond_embeds, cond_hidden_states, uncond_embeds, uncond_hidden_states

    def _prepare(
        self,
        encoded: tuple,
        *,
        height: int,
        width: int,
        num_inference_steps: int,
        seed: int,
        latents: torch.Tensor | None,
    ) -> tuple:
        """Build per-branch conditioning (RoPE + host->device uploads), schedule, and initial latents.

        Per-``__call__`` work (recurs on every call): rebuilds the 37->46 layer list, recomputes RoPE
        (``_pos_embed.forward`` in ``_prepare_branch``), uploads ``prompt`` + 2x46 layer tensors + RoPE +
        latents to device, recomputes the schedule. Only weights/submesh/``_pos_embed`` are built once
        (``__init__``), so this is real per-image cost -- a warmup run amortizes op compile, not this.
        """
        cond_embeds, cond_hidden_states, uncond_embeds, uncond_hidden_states = encoded
        sp_axis = self._parallel_config.sequence_parallel.mesh_axis
        latent_h = height // _VAE_SCALE_FACTOR
        latent_w = width // _VAE_SCALE_FACTOR
        spatial_sequence_length = latent_h * latent_w

        cond_layers = build_text_encoder_layers(cond_hidden_states, self._num_blocks)
        uncond_layers = build_text_encoder_layers(uncond_hidden_states, self._num_blocks)
        cond_branch = self._prepare_branch(cond_embeds, cond_layers, latent_h, latent_w, sp_axis)
        uncond_branch = self._prepare_branch(uncond_embeds, uncond_layers, latent_h, latent_w, sp_axis)

        # Timesteps + solver schedule (dynamic shift on the image sequence length).
        logger.info("preparing timesteps...")
        self._scheduler.set_timesteps(
            sigmas=np.linspace(1.0, 1 / num_inference_steps, num_inference_steps),
            mu=_calculate_shift(spatial_sequence_length, self._scheduler),
        )
        self._solver.set_schedule(self._scheduler.sigmas.tolist())
        timesteps = self._scheduler.timesteps

        # Latents (no 2x2 pack): (1, 48, h, w) -> (1, h*w, 48), sequence-sharded on sp.
        logger.info("preparing latents...")
        latent = self._random_latents(height=height, width=width, seed=seed, latents=latents)

        return cond_branch, uncond_branch, timesteps, latent, spatial_sequence_length

    def _denoise(
        self,
        cond_branch: dict,
        uncond_branch: dict,
        timesteps,
        latent: ttnn.Tensor,
        spatial_sequence_length: int,
        guidance_scale: float,
    ) -> ttnn.Tensor:
        """Denoise loop: two per-branch forwards per step, combined via CFG. Syncs per step."""
        logger.info("denoising...")
        submesh = self._submesh
        for i, t in enumerate(tqdm.tqdm(timesteps)):
            timestep = tt_tensor.from_torch(
                torch.full((1, 1), float(t), dtype=torch.bfloat16), device=submesh, dtype=ttnn.bfloat16
            )

            v_cond = self._run_transformer(latent, cond_branch, timestep, spatial_sequence_length)
            v_uncond = self._run_transformer(latent, uncond_branch, timestep, spatial_sequence_length)

            # noise = uncond + guidance_scale * (cond - uncond)
            velocity = ttnn.lerp(v_uncond, v_cond, guidance_scale)
            ttnn.deallocate(v_cond)
            ttnn.deallocate(v_uncond)
            ttnn.deallocate(timestep)

            new_latent = self._solver.step(step=i, latent=latent, velocity_pred=velocity)
            ttnn.deallocate(velocity)
            ttnn.deallocate(latent)
            latent = new_latent

            ttnn.synchronize_device(submesh)
        return latent
```

- [ ] **Step 3: Confirm the refactor changed no behavior — re-run the PCC gate**

Run:
```bash
HF_HUB_OFFLINE=1 TT_METAL_HOME=$PWD PYTHONPATH=$PWD ARCH_NAME=blackhole \
  python_env/bin/python -m pytest \
  models/tt_dit/tests/models/bria_fibo/test_pipeline.py::test_fibo_pipeline_latent_pcc -v
```
Expected: PASS with the **same** PCC as Step 1 (bit-identical trajectory; the extraction moves code without changing it). If PCC differs, the extraction introduced a bug — diff against the original `__call__` before proceeding.

- [ ] **Step 4: Commit**

```bash
git add models/tt_dit/pipelines/bria_fibo/pipeline_bria_fibo.py
git commit -m "refactor(fibo-pipeline): extract __call__ into _encode/_prepare/_denoise stage methods

Behavior-preserving: pure extraction of locals into method params/returns, no
logic change. Gives clean stage boundaries for the perf harness and keeps
__call__ free of timing code. Gated by test_fibo_pipeline_latent_pcc (unchanged PCC).

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: The perf breakdown harness

**Files:**
- Create: `models/tt_dit/tests/models/bria_fibo/test_performance_bria_fibo.py`

**Interfaces:**
- Consumes (from Task 1): `BriaFiboPipeline._encode`, `._prepare`, `._denoise` (signatures above), plus existing `._decode_latents(latent, *, height, width, output_type, force_device_decode)` and `._submesh`.
- Produces: `test_fibo_pipeline_perf_breakdown` (a pytest test; no downstream consumers).

- [ ] **Step 1: Write the harness**

Create `models/tt_dit/tests/models/bria_fibo/test_performance_bria_fibo.py`:

```python
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""On-demand perf dev-harness for ``BriaFiboPipeline`` (NOT a CI perf gate).

Builds the pipeline, does one warmup pass + N measured passes driving the pipeline's own stage methods
(``_encode`` / ``_prepare`` / ``_denoise`` / ``_decode_latents``), times each with boundary device syncs,
and logs a per-stage wall-clock breakdown (seconds + %, denoise it/s, images/s). Approach B from
``docs/superpowers/specs/2026-07-09-fibo-perf-harness-design.md``: measures the real code path; no
tracing / on_event / CI-assert / device-op profiling (those are documented follow-ups).

Run explicitly (not collected into CI perf):
  HF_HUB_OFFLINE=1 TT_METAL_HOME=$PWD PYTHONPATH=$PWD ARCH_NAME=blackhole \\
    python_env/bin/python -m pytest \\
    models/tt_dit/tests/models/bria_fibo/test_performance_bria_fibo.py -v -s
"""

import os
from time import perf_counter

import numpy as np
import pytest
from huggingface_hub import snapshot_download
from loguru import logger

import ttnn

FIBO_PATH = os.environ.get("FIBO_PATH", "briaai/FIBO")

STAGES = ("encode", "prepare", "denoise", "decode")


def _fibo_local():
    try:
        return snapshot_download(FIBO_PATH, local_files_only=True)
    except Exception as e:
        pytest.skip(f"FIBO not cached: {e}")


@pytest.mark.parametrize("mesh_device", [(2, 2)], indirect=["mesh_device"])
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D, "l1_small_size": 32768, "trace_region_size": 50000000}],
    indirect=["device_params"],
)
@pytest.mark.parametrize(
    "height, width, num_inference_steps, num_measured_runs",
    [(1024, 1024, 30, 3)],
)
def test_fibo_pipeline_perf_breakdown(
    *, mesh_device, height, width, num_inference_steps, num_measured_runs
):
    """Print a per-stage wall-clock breakdown of one 1024x1024/30-step generation on the 2x2 mesh.

    Sanity-asserts the produced image is valid + non-degenerate (proves the timed path really ran);
    it does NOT assert on timing (dev instrument, not a regression gate). Use ``-s`` to see the log.
    Runtime ~ (1 warmup + num_measured_runs) full generations (~30s each) + ~44s model build.
    """
    from models.tt_dit.pipelines.bria_fibo.pipeline_bria_fibo import BriaFiboPipeline, BriaFiboPipelineConfig

    ckpt = _fibo_local()
    pipe = BriaFiboPipeline(
        device=mesh_device,
        config=BriaFiboPipelineConfig.default(
            mesh_shape=mesh_device.shape, checkpoint_name=ckpt, height=height, width=width
        ),
    )
    submesh = pipe._submesh
    prompt = "a luxury sports car"
    negative_prompt = ""
    guidance_scale = 5.0
    seed = 0

    def run_once(record: dict):
        """One full generation driving the stage methods; records per-stage seconds into ``record``."""

        def time_stage(name, fn):
            ttnn.synchronize_device(submesh)  # drain enqueued work so t0 is a real boundary
            t0 = perf_counter()
            result = fn()
            ttnn.synchronize_device(submesh)  # wait for this stage's device work to finish
            record[name] = perf_counter() - t0
            return result

        encoded = time_stage("encode", lambda: pipe._encode(prompt, negative_prompt))
        cond_branch, uncond_branch, timesteps, latent, ssl = time_stage(
            "prepare",
            lambda: pipe._prepare(
                encoded,
                height=height,
                width=width,
                num_inference_steps=num_inference_steps,
                seed=seed,
                latents=None,
            ),
        )
        latent = time_stage(
            "denoise",
            lambda: pipe._denoise(cond_branch, uncond_branch, timesteps, latent, ssl, guidance_scale),
        )
        image = time_stage(
            "decode",
            lambda: pipe._decode_latents(
                latent, height=height, width=width, output_type="pil", force_device_decode=True
            ),
        )
        return image

    # 1 warmup pass (absorbs op compilation), then N measured passes.
    logger.info("perf harness: warmup run...")
    run_once({})

    runs = []
    image = None
    for r in range(num_measured_runs):
        logger.info(f"perf harness: measured run {r + 1}/{num_measured_runs}...")
        record = {}
        image = run_once(record)
        runs.append(record)

    # Sanity: the last image is a valid, non-degenerate frame (proves the timed path really ran).
    arr = np.asarray(image[0])
    assert arr.shape == (height, width, 3), f"unexpected image shape {arr.shape}"
    assert arr.std() > 1.0, f"image looks degenerate (std={arr.std():.4f})"

    # Aggregate across measured runs and print the breakdown.
    avg = {s: sum(run[s] for run in runs) / len(runs) for s in STAGES}
    lo = {s: min(run[s] for run in runs) for s in STAGES}
    hi = {s: max(run[s] for run in runs) for s in STAGES}
    total = sum(avg[s] for s in STAGES)

    lines = [
        f"\nFIBO perf breakdown — {width}x{height}, {num_inference_steps} steps, "
        f"avg of {num_measured_runs} runs (after 1 warmup)"
    ]
    for s in STAGES:
        pct = 100.0 * avg[s] / total if total else 0.0
        extra = ""
        if s == "prepare":
            extra = "RoPE recompute + 92-tensor upload"
        elif s == "denoise" and avg[s]:
            extra = f"-> {num_inference_steps / avg[s]:.2f} it/s"
        lines.append(
            f"  {s:<9} {avg[s]:7.2f} s  ({pct:4.1f}%)  [min {lo[s]:6.2f} / max {hi[s]:6.2f}]  {extra}"
        )
    lines.append("  " + "-" * 62)
    images_per_s = 1.0 / total if total else 0.0
    lines.append(f"  {'total':<9} {total:7.2f} s             -> {images_per_s:.4f} images/s")
    logger.info("\n".join(lines))
```

- [ ] **Step 2: Run the harness and verify it produces a well-formed breakdown + valid image**

Run (this is the harness's own verification — there is no separate unit test for a test file):
```bash
HF_HUB_OFFLINE=1 TT_METAL_HOME=$PWD PYTHONPATH=$PWD ARCH_NAME=blackhole \
  python_env/bin/python -m pytest \
  models/tt_dit/tests/models/bria_fibo/test_performance_bria_fibo.py::test_fibo_pipeline_perf_breakdown -v -s
```
Expected: PASS. The log shows a `FIBO perf breakdown` table with four stage rows + total. Confirm the numbers land near the recorded baseline: denoise ~1.4–1.6 it/s, encode <1s, decode ~10s, total ~30s. (If decode raises a `LoadingError` instead of running, the on-device VAE regressed — out of scope here; drop `force_device_decode=True` to fall back and note it.)

- [ ] **Step 3: Commit**

```bash
git add models/tt_dit/tests/models/bria_fibo/test_performance_bria_fibo.py
git commit -m "test(fibo-pipeline): on-demand perf dev-harness (per-stage wall-clock breakdown)

Drives the pipeline's _encode/_prepare/_denoise/_decode stage methods with
boundary device syncs, 1 warmup + N measured runs, and logs encode/prepare/
denoise-it-s/decode/total. Dev instrument (no CI marker, no perf asserts);
Approach B from the 2026-07-09 perf-harness spec.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Self-Review

**Spec coverage:**
- Behavior-preserving `__call__` refactor into `_encode`/`_prepare`/`_denoise` → Task 1. ✓
- Standalone harness with warmup + N measured, boundary syncs, breakdown print → Task 2. ✓
- `prepare` as its own bucket (exposes RoPE + 92-tensor upload) → Task 1 `_prepare` split + Task 2 label. ✓
- Sanity-only asserts, no CI marker / expected_metrics → Task 2. ✓
- Re-run existing PCC gate to prove no regression → Task 1 Steps 1 & 3. ✓
- Out-of-scope items (tracing, on_event, device-op, per-step, batched CFG) → not in any task. ✓

**Placeholder scan:** No TBD/TODO; every code step shows complete code; every run step shows the exact command + expected result. ✓

**Type consistency:** `_encode` returns the 4-tuple consumed by `_prepare(encoded, ...)`; `_prepare` returns `(cond_branch, uncond_branch, timesteps, latent, spatial_sequence_length)` unpacked identically in `__call__` and the harness (`ssl` = `spatial_sequence_length`); `_denoise` params match both call sites; harness `_decode_latents` call matches the existing signature. ✓
