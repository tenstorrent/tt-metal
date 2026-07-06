# FIBO on TTNN — Sub-project 3: Wan 2.2 VAE decode + flow-match solver

**Date:** 2026-07-06
**Status:** Design approved (auto-approved per user workflow), ready for implementation plan
**Target hardware:** Blackhole Quietbox (4× P150)
**Branch:** `fibo-vae-solver` (stacked on `fibo-transformer`)
**Scope:** the VAE decode + the flow-match solver/scheduler wiring for FIBO. The encoder (sp1) and transformer (sp2) are done; the end-to-end pipeline is sub-project 4.

---

## Context: the larger FIBO effort
Third of four decomposed sub-projects for bringing up Bria **FIBO** (text→image, flow-matching MMDiT) in `models/tt_dit`, each PCC-gated vs the HF reference on Blackhole.
- sp1 SmolLM3 text encoder — **DONE** (`fibo-smollm3-encoder`).
- sp2 BriaFibo transformer — **DONE** (`fibo-transformer`).
- **sp3: Wan VAE decode + flow-match solver** ← this spec.
- sp4: pipeline + Blackhole bringup — TODO.

Reference: `diffusers` `AutoencoderKLWan` (VAE) and `FlowMatchEulerDiscreteScheduler`. Weights: `briaai/FIBO`, subfolders `vae` and `scheduler` (gated; access confirmed).

## 1. Goal
Decode FIBO's denoised latents to an image on Blackhole, reproducing the HF `AutoencoderKLWan` decode numerically, and provide the flow-match denoise-step machinery (Euler solver + dynamic-shift schedule) matching the HF `FlowMatchEulerDiscreteScheduler`. Both validated standalone vs the reference.

### In scope
- Implement the **Wan 2.2 residual VAE decoder** (`is_residual=True`) in tt_dit's Wan VAE so it can decode FIBO's latents (z_dim=48, decoder_base_dim=256, out_channels=12, 16× spatial), single-image (T=1).
- Reuse the flow-match **Euler solver** + **dynamic-shift** (`_calculate_shift`) with FIBO's scheduler config; add a FIBO-config validation.
- PCC tests vs the reference on real `briaai/FIBO` weights, on Blackhole.

### Out of scope (sub-project 4)
- The VAE **encode** path (not needed for text→image inference).
- Latent packing/unpacking between the transformer (in_channels 48) and the VAE — note: **z_dim = 48 = transformer in_channels directly**, so there is NO 2×2 packing (unlike Flux). The exact latent tensor layout handoff is finalized in the pipeline.
- The denoise loop, CFG, and stringing encoder→transformer→solver→VAE together.

## 2. Background — verified configs

**FIBO VAE (`briaai/FIBO/vae/config.json`):** `_class_name=AutoencoderKLWan`, `z_dim=48`, `base_dim=160`, `decoder_base_dim=256`, `dim_mult=[1,2,4,4]`, `num_res_blocks=2`, `is_residual=True`, `in_channels=12`, `out_channels=12`, `patch_size=2`, `scale_factor_spatial=16`, `scale_factor_temporal=4`, `temperal_downsample=[False,True,True]`, 48-element `latents_mean`/`latents_std`. This is the **Wan 2.2 high-compression (TI2V) VAE**, not Wan 2.1 (z_dim=16, 8× spatial).

**FIBO scheduler (`scheduler/scheduler_config.json`):** `FlowMatchEulerDiscreteScheduler`, `use_dynamic_shifting=True`, `base_shift=0.5`, `max_shift=1.15`, `base_image_seq_len=256`, `max_image_seq_len=4096`, `shift=3.0`, `time_shift_type=exponential`, `num_train_timesteps=1000`. These match tt_dit's `_calculate_shift` defaults exactly.

**tt_dit gap (the real work):** `models/tt_dit/models/vae/vae_wan2_1.py` has **hard `assert not is_residual`** in `WanDecoder3d.__init__` (~L1205), `WanDecoder.__init__` (~L1438), `WanEncoder3D.__init__` (~L1577), with the comment "Different codepath if is_residual. Not implemented yet." The Wan 2.2 residual path changes the up-block channel bookkeeping (no `in_dim // 2` halving at stages `i>0`; residual skip connections across upsample stages). Everything else FIBO needs (z_dim=48 → `aligned_channels(48)=64`, `decoder_base_dim`, `dim_mult`, `out_channels`, `latents_mean/std` read from HF config by `WanVAEDecoderAdapter`) is already config-driven. `patch_size`/`scale_factor_spatial` are pipeline-level (not in the VAE impl).

## 3. Design

### 3.1 Files
```
models/tt_dit/models/vae/vae_wan2_1.py         # MODIFY: add is_residual=True decoder path
models/tt_dit/tests/models/bria_fibo/test_vae.py      # NEW: Wan2.2 VAE decode PCC test
models/tt_dit/tests/models/bria_fibo/test_solver.py   # NEW: FIBO-config solver/shift validation
models/tt_dit/models/BriaFibo.md               # MODIFY: sub-project 3 section
```
(Extend the existing `vae_wan2_1.py` rather than a new file — the residual path is a variant the file already parameterizes and stubs; keep the Wan VAE in one place.)

### 3.2 Wan 2.2 residual decoder (the core work)
Reference: the diffusers `AutoencoderKLWan` decoder (the `is_residual`/WanResidualUpBlock path). In `vae_wan2_1.py`:
1. Remove the three `assert not is_residual` guards.
2. Implement the residual up-block path in `WanDecoder3d` (the `i>0` channel-dim branch currently stubbed at ~L1263): with `is_residual`, the up-blocks use residual skip connections and do **not** halve `in_dim` — mirror the diffusers `WanResidualUpBlock`/`WanUpBlock` residual variant (likely a new `WanResidualUpBlock`-style class or an `is_residual` branch in the existing up-block). Wire `decoder_base_dim` (256) as the decoder `dim` (distinct from encoder `base_dim`).
3. Extend `compute_decoder_dims`/`_BLOCKINGS` (the optimized-shape table) with entries for the TI2V 16×-spatial latent dims used by FIBO (e.g. 1024×1024 image → 64×64 latent → decode).
4. `out_channels=12` (patchified pixel space); `WanVAEDecoderAdapter` already reads z_dim/decoder_base_dim/is_residual/latents_mean/std from the HF `AutoencoderKLWan` config, so once the asserts/architecture are in, construction from FIBO's config works.
Keep the encoder path untouched (encode is out of scope; leave its `is_residual` assert or add a NotImplemented note — decode-only is fine for inference).

### 3.3 Solver (reuse)
No new solver code. Validate that the host `FlowMatchEulerDiscreteScheduler.from_pretrained(FIBO, subfolder="scheduler")` + `_calculate_shift(image_seq_len, scheduler)` + `EulerSolver.set_schedule(scheduler.sigmas)` + `EulerSolver.step` reproduce the reference sigma schedule and a denoise step for FIBO's config (dynamic shifting, exponential). The exponential `time_shift_type` is applied inside the host diffusers `set_timesteps`, transparent to the device-side `EulerSolver.step`.

### 3.4 Parallelization / precision
VAE decode uses `VaeHWParallelConfig` (height/width parallel). Bring up on a single Blackhole device first (`(1,1)`, height=(1,0)/width=(1,1)) for correctness, then optionally the mesh HW-parallel presets (the wan pipeline has BH presets). bf16 (fp32 forces HiFi4 on BH); the Wan VAE code already selects HiFi4 for fp32 on BH.

## 4. Public interface
Reuse `WanVAEDecoderAdapter(checkpoint_name="briaai/FIBO", ...).decode(latents, output_type=...)` — after the is_residual path is implemented, this constructs from FIBO's vae config and decodes z_dim=48 latents to an image. No new public class required; the deliverable is the residual-decoder capability + validation.

## 5. Testing & validation
- **VAE decode** (`test_vae.py`, modeled on `tests/models/wan2_2/test_vae_wan2_1.py::test_wan_decoder`): reference `AutoencoderKLWan.from_pretrained("briaai/FIBO", subfolder="vae")`; random (seeded) or reference latent `(1, 48, 1, H//16, W//16)` (T=1); de-normalize with the 48-element latents_mean/std; decode both; `assert_quality(pcc ≥ 0.99 bf16)`. Reduced resolution first (small H×W) then a production size (e.g. 1024→64×64 latent). Single Blackhole device, then HW-parallel mesh.
- **Solver** (`test_solver.py`): build FIBO's scheduler, compute `mu=_calculate_shift(seq_len)`, `set_timesteps(sigmas, mu=mu)`; assert `EulerSolver.step` + the sigma schedule match a diffusers `FlowMatchEulerDiscreteScheduler.step` reference (mirror `tests/unit/test_solvers.py`), for a representative FIBO seq_len.
- Env: `HF_HUB_OFFLINE=1` + pre-downloaded `briaai/FIBO/vae/*`; the offline invocation from [[tt-dit-test-env]].

**Definition of done:** FIBO's Wan 2.2 VAE decodes a latent to an image at PCC ≥ 0.99 vs the reference on Blackhole (T=1, reduced + a production resolution), and the flow-match solver/shift is validated against diffusers for FIBO's config.

## 6. Open items (resolve during implementation, not blockers)
- Pin the exact residual up-block channel bookkeeping against the diffusers `AutoencoderKLWan` source (the subtlest piece).
- Confirm the `out_channels=12` → image mapping (patchified pixels: 12 = 3 RGB × patch 2² ? verify the reference decode output shape and how it maps to an RGB image; the un-patchify to RGB may be pipeline-level).
- Blocking-table entries for the TI2V 16×-spatial dims (perf/shape correctness).
- Whether the mesh HW-parallel VAE is needed for sp3 or single-device suffices (defer mesh to sp4/perf if single-device validates).

## 7. Risks & mitigations
1. **Residual up-block fidelity** (main risk) — validate against the diffusers reference decode at PCC ≥ 0.99; bring up at reduced resolution first.
2. **Effort larger than "light reuse"** — the residual decoder is real architectural work. Fallback: sub-project 4's first end-to-end image can use the host torch `AutoencoderKLWan.decode` as a stopgap while the tt decoder is finished.
3. **VAE weight download** (~1 file, modest) — pre-download `vae/*` offline like the encoder/transformer.

## 8. Follow-on
Sub-project 4 (pipeline) wires the SmolLM3 encoder + BriaFibo transformer + this VAE + solver into the end-to-end flow-matching denoise loop with CFG, building the 37→46 text-layer list and the latent handoff.
