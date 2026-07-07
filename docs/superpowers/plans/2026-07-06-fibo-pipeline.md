# FIBO end-to-end pipeline — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** A `BriaFiboPipeline` that generates a 1024×1024 image from a text prompt end-to-end on the 2×2 Blackhole mesh, wiring the three validated components (SmolLM3 encoder + BriaFibo transformer + Wan VAE/Euler solver), reproducing the reference diffusers pipeline's latent trajectory (reduced-step PCC) and producing a coherent image.

**Architecture:** Mirror `models/tt_dit/pipelines/flux1/pipeline_flux1.py`. Reuse `pipelines/cfg.py` (submeshes/CFG), `Tracer`, `EulerSolver` + `_calculate_shift`, `_latent_image_ids`, the `pos_embed` RoPE helper, and the three components. Net-new: a SmolLM3 `text_encoder.py` wrapper, the 37→46 `text_encoder_layers` pad, unpadded per-CFG-branch denoise wiring, and `_decode_latents` (unpatchify+clamp).

**Tech Stack:** Python, PyTorch + `diffusers` (`BriaFiboPipeline` reference), `ttnn`, tt_dit, Tenstorrent Blackhole.

**Spec:** `docs/superpowers/specs/2026-07-06-fibo-pipeline-design.md`
**Branch:** `fibo-pipeline` (stacked on `fibo-vae-solver`).

## Global Constraints
- **SPDX header** on new `.py` files. **Strong subagents** — sonnet floor; opus for the pipeline integration (Tasks 2-3).
- **Location:** `models/tt_dit/pipelines/bria_fibo/`; tests `models/tt_dit/tests/models/bria_fibo/test_pipeline.py`.
- **Config (verified):** FIBO transformer = 46 blocks (8+38), in_channels=48; VAE `AutoencoderKLWan` z_dim=48, scale_factor_spatial=16, patch_size=2; **no 2×2 latent packing** (in_channels==z_dim==48); scheduler `FlowMatchEulerDiscreteScheduler` dynamic shift (base/max_shift 0.5/1.15, base/max_image_seq_len 256/4096). Defaults: 1024×1024, num_inference_steps=30, guidance_scale=5, negative_prompt="".
- **Reference:** `from diffusers import BriaFiboPipeline` `.from_pretrained("briaai/FIBO", torch_dtype=torch.bfloat16)` (source: `python_env/lib/python3.12/site-packages/diffusers/pipelines/bria_fibo/pipeline_bria_fibo.py`). Component reference classes: `SmolLM3ForCausalLM`, `BriaFiboTransformer2DModel`, `AutoencoderKLWan`.
- **Weights:** gated `briaai/FIBO` — encoder/transformer/vae already cached; **tokenizer + scheduler + model_index.json** to pre-download in Task 1. Run `HF_HUB_OFFLINE=1`; resolve repo id offline via `snapshot_download(local_files_only=True)`.
- **Run:** `HF_HUB_OFFLINE=1 TT_METAL_HOME=$PWD PYTHONPATH=$PWD ARCH_NAME=blackhole python_env/bin/python -m pytest <path>::<test> -v` (interpreter `python_env/bin/python`).
- **Mesh:** `DiTParallelConfig.from_tuples(cfg=(1,0), sp=(2,0), tp=(2,1))` on (2,2), `fabric_config=FABRIC_1D`, `num_links=1` (mirror flux1's BH `(2,2)` preset). Encoder + VAE on the submesh; no inter-stage eviction (start).
- **CFG:** batch=1, **unpadded per-CFG-branch forwards** — encode uncond `""` and cond at their true token lengths, run the transformer once per branch per step, combine `uncond + guidance_scale*(cond-uncond)`. No padding ⇒ no attention mask needed (the tt transformer has none) ⇒ faithful to the reference (which masks padding).
- **Component interfaces (consume as-is):** `SmolLM3TextEncoder(config, device=, parallel_config=, ccl_manager=).encode(input_ids, attention_mask=None, pos_embeds=(cos,sin)) -> (prompt_embeds[B,T,4096], all_hidden_states[list len 37])` + `.create_rope_tensors(batch, seq)`; `BriaFiboCheckpoint(name).build(ccl_manager=, parallel_config=) -> BriaFiboTransformer` with `.pos_embed`; `BriaFiboTransformer.forward(spatial, prompt=, timestep=, text_encoder_layers=, spatial_rope=, prompt_rope=, spatial_sequence_length=, prompt_sequence_length=) -> velocity`; `WanVAEDecoderAdapter(checkpoint_name=, parallel_config=VaeHWParallelConfig, ccl_manager=, height=, width=, num_frames=1, ...).decode(latents_BCTHW, output_type=) -> torch img`; `EulerSolver().set_schedule(sigmas); .step(step=, latent=, velocity_pred=)`.
- **Template:** `pipelines/flux1/pipeline_flux1.py` (+ `flux1/text_encoder.py`, `pipelines/cfg.py`). Pipeline-test precedent (qualitative image smoke): `tests/models/flux1/test_pipeline_flux1.py`.

---

## File Structure
- `models/tt_dit/pipelines/bria_fibo/__init__.py`
- `models/tt_dit/pipelines/bria_fibo/text_encoder.py` — `SmolLM3TextEncoderWrapper` (tokenize + encode + CFG) + `build_text_encoder_layers`.
- `models/tt_dit/pipelines/bria_fibo/pipeline_bria_fibo.py` — `BriaFiboPipeline` (+ config), `__call__`, `_decode_latents`.
- `models/tt_dit/tests/models/bria_fibo/test_pipeline.py`.
- `models/tt_dit/models/BriaFibo.md` — sp4 section.

---

### Task 1: tokenizer download + SmolLM3 text-encoder wrapper + 37→46 list

**Files:** Create `pipelines/bria_fibo/__init__.py`, `pipelines/bria_fibo/text_encoder.py`; Test `tests/models/bria_fibo/test_pipeline.py`.

**Interfaces:**
- Produces: `SmolLM3TextEncoderWrapper(checkpoint, *, device, ccl_manager, parallel_config, use_torch=False)` with `.encode_prompt(prompt: str) -> (prompt_embeds[1,T,4096], all_hidden_states: list[torch.Tensor])` (host tensors; tokenizes at the prompt's TRUE length — no fixed-length pad — handling the empty-prompt `""` case per the reference); and module-level `build_text_encoder_layers(all_hidden_states, num_blocks) -> list` = `list(all_hidden_states) + [all_hidden_states[-1]] * (num_blocks - len(all_hidden_states))` (right-trim if longer, per the reference rule).

- [ ] **Step 1: Pre-download tokenizer/scheduler/model_index** (one-time):
```bash
cd /localdev/mstojkovic/tt-metal
export HF_TOKEN=$(grep -E '^[[:space:]]*export[[:space:]]+HF_TOKEN=' ~/.bashrc | tail -1 | sed -E 's/.*HF_TOKEN=//; s/^["'"'"']//; s/["'"'"']$//')
python_env/bin/python -c "from huggingface_hub import snapshot_download; import os; print(snapshot_download('briaai/FIBO', allow_patterns=['tokenizer/*','scheduler/*','model_index.json','text_encoder/config.json'], token=os.environ['HF_TOKEN']))"
```

- [ ] **Step 2: Write failing tests** in `test_pipeline.py`:
```python
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import os
import pytest
from huggingface_hub import snapshot_download

FIBO_PATH = os.environ.get("FIBO_PATH", "briaai/FIBO")


def _fibo_local():
    try:
        return snapshot_download(FIBO_PATH, local_files_only=True)
    except Exception as e:
        pytest.skip(f"FIBO not cached: {e}")


def test_build_text_encoder_layers_pads_37_to_46():
    from models.tt_dit.pipelines.bria_fibo.text_encoder import build_text_encoder_layers
    hs = [f"h{i}" for i in range(37)]                 # stand-in objects
    out = build_text_encoder_layers(hs, 46)
    assert len(out) == 46
    assert out[:37] == hs
    assert out[37:] == [hs[-1]] * 9                   # last state repeated 9x
    # right-trim when longer than num_blocks
    assert build_text_encoder_layers([f"h{i}" for i in range(50)], 46) == [f"h{i}" for i in range(4, 50)]
```
Run: `... test_pipeline.py::test_build_text_encoder_layers_pads_37_to_46 -v` → FAIL (import).

- [ ] **Step 3: Implement `text_encoder.py`.** `build_text_encoder_layers` per the interface. `SmolLM3TextEncoderWrapper` mirrors `flux1/text_encoder.py`'s `TextEncoder` but for one decoder-LM: load `AutoTokenizer.from_pretrained(checkpoint, subfolder="tokenizer")`; build `SmolLM3TextEncoder` (from `SmolLM3Config.from_hf_config`) + load weights via `cache.load_model` (or reuse the encoder's existing loader); `encode_prompt(prompt)` tokenizes at true length (no max-length pad; replicate the reference's empty-prompt `bot_token_id=128000` handling for `""` — inspect `pipeline_bria_fibo.py get_prompt_embeds`), computes `cos,sin = enc.create_rope_tensors(1, seq)`, pushes tokens/cos/sin to device, calls `enc.encode(...)`, returns `(prompt_embeds, all_hidden_states)` as host tensors. (No CFG concat here — the pipeline runs per-branch, §CFG.)

- [ ] **Step 4: Run** the list test (PASS). Add + run a device `test_wrapper_encode_matches_reference` comparing `SmolLM3TextEncoderWrapper.encode_prompt("a luxury sports car")` `prompt_embeds` to the reference `SmolLM3ForCausalLM(output_hidden_states=True)` → `cat(hs[-1],hs[-2])`, PCC ≥ 0.99 (reuses the sp1-validated encoder; confirms the wrapper's tokenize+encode wiring). Run offline.

- [ ] **Step 5: Commit** — `feat(fibo-pipeline): SmolLM3 text-encoder wrapper + 37->46 layer build`.

---

### Task 2: `BriaFiboPipeline` + `__call__` + full image smoke

**Files:** Create `pipelines/bria_fibo/pipeline_bria_fibo.py`; Test `test_pipeline.py`.

**Interfaces:**
- Produces: `BriaFiboPipeline(device, config=BriaFiboPipelineConfig(...))` with `__call__(prompt: str, *, negative_prompt="", height=1024, width=1024, num_inference_steps=30, guidance_scale=5.0, seed=0, output_type="pil") -> [image]`.

- [ ] **Step 1: Write the failing full-image smoke test** (device, (2,2) mesh) — mirror `tests/models/flux1/test_pipeline_flux1.py`:
```python
import ttnn


@pytest.mark.parametrize("mesh_device", [(2, 2)], indirect=["mesh_device"])
@pytest.mark.parametrize("device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D, "l1_small_size": 32768, "trace_region_size": 50000000}],
    indirect=["device_params"])
def test_fibo_pipeline_smoke(*, mesh_device):
    from models.tt_dit.pipelines.bria_fibo.pipeline_bria_fibo import BriaFiboPipeline, BriaFiboPipelineConfig
    ckpt = _fibo_local()
    pipe = BriaFiboPipeline(device=mesh_device,
        config=BriaFiboPipelineConfig.default(mesh_shape=mesh_device.shape, checkpoint_name=ckpt,
                                              height=1024, width=1024))
    imgs = pipe("a luxury sports car", num_inference_steps=30, guidance_scale=5.0, seed=0)
    import numpy as np
    arr = np.asarray(imgs[0])
    assert arr.shape[:2] == (1024, 1024) and np.isfinite(arr).all()
    imgs[0].save("fibo_smoke.png")
```
Run → FAIL (import).

- [ ] **Step 2-3: Implement `pipeline_bria_fibo.py`.** Copy the structure of `Flux1Pipeline`/`Flux1PipelineConfig` and adapt:
  - **Config**: mesh (2,2) → `cfg=(1,0), sp=(2,0), tp=(2,1)`, `num_links=1`, fabric FABRIC_1D; encoder/vae on the submesh.
  - **`__init__`**: `create_submeshes`; one `CCLManager`; `BriaFiboCheckpoint(checkpoint).build(...)` (transformer, 46 blocks) + `.pos_embed`; `EulerSolver`; `FlowMatchEulerDiscreteScheduler.from_pretrained(checkpoint, subfolder="scheduler")`; `SmolLM3TextEncoderWrapper` (Task 1); `WanVAEDecoderAdapter(checkpoint_name=checkpoint, parallel_config=VaeHWParallelConfig.from_tuples(height=(1,0),width=(1,1)), ccl_manager=, height=1024, width=1024, num_frames=1, ...)`.
  - **`__call__`**: (a) encode `prompt` and `negative_prompt` separately via the wrapper → two `(prompt_embeds, all_hidden_states)`; build two 46-lists via `build_text_encoder_layers(..., num_layers+num_single_layers)`. (b) prepare latents: seeded noise `(1,48,h,w)` with `h=w=height//16`, reshaped to `(1, h*w, 48)` (no pack); `_latent_image_ids(h,w)` + `txt_ids=zeros`; `cos,sin = pos_embed.forward(cat([txt_ids, img_ids]))`, split into spatial/prompt rope, distribute per-submesh (mirror flux1 `:290-310`). (c) `mu=_calculate_shift(h*w, scheduler)`, `scheduler.set_timesteps(sigmas=np.linspace(1,1/N,N), mu=mu)`, `solver.set_schedule(scheduler.sigmas.tolist())`. (d) per step `i`: for each CFG branch (cond, uncond) run `transformer.forward(spatial=latent, prompt=branch_embeds, timestep=t, text_encoder_layers=branch_list, spatial_rope=, prompt_rope=, spatial_sequence_length=h*w, prompt_sequence_length=branch_T)` → velocity; combine `v = v_uncond + guidance_scale*(v_cond - v_uncond)`; `latent = solver.step(step=i, latent=latent, velocity_pred=v)`. (e) `_decode_latents(latent, h, w)`.
  - **`_decode_latents`**: all-gather sp-sharded latent → `(1, h*w, 48)` → reshape `(1,48,h,w)` → unsqueeze T → `WanVAEDecoderAdapter.decode(latents_BCTHW, output_type="pt")` → **`unpatchify(out, patch_size=2)` then `clamp(-1,1)`** (`from diffusers.models.autoencoders.autoencoder_kl_wan import unpatchify`) → `VaeImageProcessor.postprocess(..., output_type=...)`.
  - Tracing optional for the first pass (run untraced; add `Tracer` after correctness).

- [ ] **Step 4: Run the smoke** → produces a `(1024,1024,3)` finite image, saves `fibo_smoke.png`. Iterate on wiring/shape/CCL until it runs. If DRAM OOM with all components resident, add `register_coresident_exclusions` (VAE ↔ transformer) as wan does. If a component's expected input layout mismatches, fix the handoff.
- [ ] **Step 5: Commit** — `feat(fibo-pipeline): BriaFiboPipeline end-to-end (30-step 1024x1024 smoke)`.

---

### Task 3: reduced-step latent PCC vs the reference pipeline

**Files:** Test `test_pipeline.py`.

**Interfaces:** Consumes `BriaFiboPipeline`. No src change (add a `return_latent` hook to `__call__` if needed to extract the pre-VAE latent — a small, justified addition).

- [ ] **Step 1: Write the reduced-step PCC test** (device + CPU reference). Run the tt pipeline and the diffusers `BriaFiboPipeline` with the same prompt + seed + `num_inference_steps=2`, compare the **pre-VAE latent** (add a `return_latent=True`/`output_type="latent"` path to `__call__` and use the reference's `output_type="latent"`), `assert_quality(ref_latent, tt_latent, pcc>=0.98)`. Seed both identically (the reference uses a torch generator; match the initial noise — construct the SAME initial latent and inject it into both, since RNG differs between host and device: build the noise on host with a fixed seed and pass it to both the tt pipeline (via a `latents=` arg) and the reference (via its `latents=` arg)).
```python
@pytest.mark.parametrize("mesh_device", [(2, 2)], indirect=["mesh_device"])
@pytest.mark.parametrize("device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D, "l1_small_size": 32768, "trace_region_size": 50000000}],
    indirect=["device_params"])
def test_fibo_pipeline_latent_pcc_vs_reference(*, mesh_device):
    import torch
    from diffusers import BriaFiboPipeline
    from models.tt_dit.utils.check import assert_quality
    ckpt = _fibo_local()
    steps, prompt = 2, "a luxury sports car"
    h = w = 1024 // 16
    torch.manual_seed(0); init = torch.randn(1, 48, h, w)     # shared initial latent
    ref = BriaFiboPipeline.from_pretrained(ckpt, torch_dtype=torch.float32)
    ref_lat = ref(prompt, num_inference_steps=steps, guidance_scale=5.0,
                  latents=init.clone(), output_type="latent").images  # exact kwarg per reference sig
    from ...pipelines.bria_fibo.pipeline_bria_fibo import BriaFiboPipeline as TTPipe, BriaFiboPipelineConfig
    pipe = TTPipe(device=mesh_device, config=BriaFiboPipelineConfig.default(mesh_shape=mesh_device.shape, checkpoint_name=ckpt))
    tt_lat = pipe(prompt, num_inference_steps=steps, guidance_scale=5.0, latents=init.clone(), output_type="latent")
    assert_quality(ref_lat, tt_lat, pcc=0.98)
```
> Inspect the reference `BriaFiboPipeline.__call__` signature for the exact `latents=`/`output_type="latent"` kwargs and the latent shape it expects; adapt `init`/extraction accordingly. If the 2-step CPU reference is prohibitively slow, drop to `steps=1`; if still too slow, fall back to comparing a single transformer forward (step 0, cond branch) through the pipeline's input construction vs the reference transformer — document the choice.

- [ ] **Step 2: Run → iterate** until PCC ≥ 0.98 (report the measured value; a documented lower floor is acceptable if bf16-over-steps warrants, but investigate < 0.95). Add the `latents=`/`output_type="latent"` plumbing to `__call__` as needed.
- [ ] **Step 3: Commit** — `test(fibo-pipeline): reduced-step latent PCC vs reference`.

---

### Task 4: Model doc + effort completion

**Files:** Modify `models/tt_dit/models/BriaFibo.md`.
- [ ] **Step 1** Add the sub-project 4 section (pipeline architecture, the unpadded-per-branch CFG design, `_decode_latents` unpatchify/clamp, how to run `test_pipeline.py`, the measured latent PCC + that a full 1024² image is produced) and mark sp4 Done — completing all four FIBO sub-projects in the doc.
- [ ] **Step 2: Commit** — `docs(fibo-pipeline): BriaFibo.md sub-project 4 section (FIBO bringup complete)`.

---

## Self-Review
**Spec coverage:** text_encoder wrapper + 37→46 (Task 1) ✓; pipeline __call__ (latent prep no-pack, rope, scheduler/shift, per-branch CFG denoise, VAE decode+unpatchify+clamp) (Task 2) ✓; mesh sp2/tp2/cfg1 (Task 2) ✓; reduced-step latent PCC vs reference + full image smoke (Tasks 2-3) ✓; doc (Task 4) ✓. Attention-mask avoidance via unpadded per-branch (Task 2 CFG) ✓. Deferred (spec §8): batched+masked CFG, tracing/perf — out of scope.
**Placeholder scan:** Task 2's `__call__` implementation and Task 3's reference kwargs are "mirror flux1 / inspect the reference signature" adaptation instructions (the pipeline is an integration over documented interfaces + a reference source) — the concrete component interfaces, mesh config, CFG algebra, and test code are given; this is the same reference-derived style as the transformer/VAE plans, gated by the smoke + PCC tests. No TBD/vague-requirement placeholders.
**Type consistency:** `SmolLM3TextEncoderWrapper.encode_prompt`, `build_text_encoder_layers`, `BriaFiboPipeline`/`BriaFiboPipelineConfig`, `__call__(prompt, ..., latents=, output_type=)`, `_decode_latents`, and the consumed component signatures are used consistently across tasks; `_fibo_local` shared across the test file.
**Note:** Tasks 2-3 are the integration crux — dispatch on opus. Task 1 sonnet, Task 4 sonnet.
