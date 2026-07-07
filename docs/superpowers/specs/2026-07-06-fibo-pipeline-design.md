# FIBO on TTNN — Sub-project 4: End-to-end pipeline (text → image)

**Date:** 2026-07-07
**Status:** Design approved (auto-approved per user workflow), ready for implementation plan
**Target hardware:** Blackhole Quietbox (4× P150), 2×2 mesh
**Branch:** `fibo-pipeline` (stacked on `fibo-vae-solver`)
**Scope:** the end-to-end FIBO text→image pipeline that orchestrates the three completed components. This is the capstone (final sub-project).

---

## Context: the larger FIBO effort
Fourth and final decomposed sub-project of the Bria **FIBO** port to `models/tt_dit`. The three model components are done and PCC-validated on Blackhole, on stacked branches:
- sp1 SmolLM3 text encoder — `SmolLM3TextEncoder.encode()` → `(prompt_embeds[B,T,4096], all_hidden_states)` (`fibo-smollm3-encoder`).
- sp2 BriaFibo transformer — `BriaFiboTransformer.forward(...)` + `BriaFiboCheckpoint` (`fibo-transformer`).
- sp3 Wan 2.2 VAE decoder + Euler flow-match solver — `WanVAEDecoderAdapter.decode`, `EulerSolver`, `_calculate_shift` (`fibo-vae-solver`).

This sub-project wires them into one pipeline, mirroring `models/tt_dit/pipelines/flux1/pipeline_flux1.py`. Reference: the diffusers `BriaFiboPipeline` (`python_env/lib/python3.12/site-packages/diffusers/pipelines/bria_fibo/pipeline_bria_fibo.py`).

## 1. Goal
Generate a 1024×1024 image from a text prompt end-to-end on Blackhole — tokenize → SmolLM3 encode → build per-block text-layer conditioning → CFG flow-match denoise loop (BriaFibo transformer + Euler solver + dynamic shift) → Wan VAE decode → image — reproducing the reference pipeline's latent trajectory (PCC-validated) and producing a coherent image.

### In scope
- A new `models/tt_dit/pipelines/bria_fibo/` package: `pipeline_bria_fibo.py` (+ `text_encoder.py` wrapper, `__init__.py`).
- Orchestration on the 2×2 mesh (sp=2, tp=2, cfg=1): submeshes, per-submesh transformer replica, tracer, solver; encoder + VAE on submesh 0.
- The net-new glue: the 37→46 `text_encoder_layers` build, CFG, the denoise loop, `_decode_latents` (unpatchify + clamp).
- Validation vs the reference pipeline (reduced-step latent PCC) + a full-resolution image smoke.

### Out of scope
- VAE encode (inference only). Batched/masked CFG efficiency (see §3.3 — we use unpadded per-branch forwards). Perf tuning / full tracing optimization / multi-image batching. The VLM-prompt-to-JSON preprocessing step.

## 2. Reference `__call__` mechanics (verified from diffusers source)
1. **Latent prep** (`do_patching=False` default): `height=H//16, width=W//16` (vae_scale_factor 16); latent `(B, C=48, H, W) → (B, H*W, 48)` — **NO 2×2 packing** (transformer in_channels = VAE z_dim = 48 directly). `latent_image_ids` = Flux-style `(h*w, 3)` grid; `txt_ids` = zeros.
2. **Prompt encode + 37→46 list:** SmolLM3 with `output_hidden_states=True` → 37 hidden states; `prompt_embeds = cat(hs[-1], hs[-2])` (4096). In `__call__`: `total = num_layers + num_single_layers = 46`; since 37 < 46, `text_encoder_layers = list(hs) + [hs[-1]] * (46-37)` (pad by repeating the last state 9×). The transformer indexes `text_encoder_layers[block_id]` 0..45.
3. **CFG:** reference batches `cat([negative, positive])` (doubling `prompt_embeds` AND each of the 46 layers) + a padding **attention mask** through `joint_attention_kwargs`; per step `noise = uncond + guidance_scale*(text - uncond)`. Defaults: `num_inference_steps=30`, `guidance_scale=5`, `negative_prompt=""`.
4. **Scheduler/shift:** `seq_len=(H//16)*(W//16)`; `mu = calculate_shift(seq_len, 256, 4096, 0.5, 1.15)`; `FlowMatchEulerDiscreteScheduler.set_timesteps(..., mu=mu)`; per-step Euler flow-match `step`.
5. **Transformer call:** `hidden_states=latent, timestep, encoder_hidden_states=prompt_embeds, text_encoder_layers=prompt_layers, txt_ids, img_ids` (+ attention_mask via joint_attention_kwargs). **Final:** `vae.decode` (internally `unpatchify(patch_size=2)` + `clamp(-1,1)` + latents denorm) → `VaeImageProcessor.postprocess`.

## 3. Design

### 3.1 File layout & mesh config
```
models/tt_dit/pipelines/bria_fibo/__init__.py
models/tt_dit/pipelines/bria_fibo/text_encoder.py    # SmolLM3 tokenize+encode+CFG wrapper
models/tt_dit/pipelines/bria_fibo/pipeline_bria_fibo.py
models/tt_dit/tests/models/bria_fibo/test_pipeline.py
models/tt_dit/models/BriaFibo.md                      # sp4 section
```
Mesh: `DiTParallelConfig.from_tuples(cfg=(1,0), sp=(2,0), tp=(2,1))` on the (2,2) mesh (mirrors flux1's BH `(2,2)` preset). One `CCLManager` + one `BriaFiboTransformer` replica + one `EulerSolver` on the (single) submesh; encoder + VAE on the submesh. No inter-stage eviction (flux1 pattern; FIBO fits — encoder 3B + transformer 8B + Wan VAE all resident). Add `register_coresident_exclusions` only if profiling shows DRAM pressure.

### 3.2 Reuse (from flux1 template + tt_dit)
- `pipelines/cfg.py` (`create_submeshes`, `CFGCombiner`, `distribute_cfg`), `Tracer`, `_calculate_shift`, `_latent_image_ids`, the `pos_embed` RoPE helper (FIBO's `EmbedND`, θ=10000, verified in sp2 tests).
- `EulerSolver` + host `FlowMatchEulerDiscreteScheduler` (validated for FIBO's config in sp3 `test_solver.py`).
- The three components: `SmolLM3TextEncoder.encode()`, `BriaFiboCheckpoint.build()`/`.forward()`, `WanVAEDecoderAdapter.decode()` (its internal latents denorm already matches the reference).

### 3.3 Net-new glue
1. **`text_encoder.py`:** a wrapper around `SmolLM3TextEncoder` that tokenizes with `AutoTokenizer` (handling the reference's empty-prompt special case — `bot_token_id=128000` for `""`), builds RoPE via the encoder's `create_rope_tensors`, calls `.encode()`, and returns `(prompt_embeds, all_hidden_states)` for a prompt.
2. **37→46 list build:** `text_encoder_layers = list(all_hidden_states) + [all_hidden_states[-1]] * (num_layers + num_single_layers - len(all_hidden_states))` (= +9 for SmolLM3's 37 → 46).
3. **CFG via unpadded per-branch forwards (key decision):** to avoid needing an attention mask (the tt transformer has none) and to avoid touching the validated transformer, run **batch=1** and encode uncond (`""`) and cond at their TRUE token lengths (no padding). Per denoise step, run the transformer **twice** — once per CFG branch (each with its own `prompt_embeds` + 46-layer list + `prompt_seq_len`) — then combine: `noise = uncond + guidance_scale*(cond - uncond)` (via `CFGCombiner`/`ttnn.lerp` or a direct op). No padding ⇒ no mask ⇒ faithful to the reference (which masks padding to the same effect). Cost: 2 forwards/step (acceptable for functional bringup; batched+masked is a perf follow-up).
4. **`_decode_latents`:** all-gather the sp-sharded final latent → `WanVAEDecoderAdapter.decode` → **`unpatchify(patch_size=2)` + `clamp(-1,1)`** (the adapter returns raw 12-ch patchified output; use `diffusers...autoencoder_kl_wan.unpatchify`) → `VaeImageProcessor.postprocess`.
5. **`__call__`** assembling steps 1-4 + latent prep (no pack) + rope + scheduler, mirroring `Flux1Pipeline.__call__`.

### 3.4 Precision
bf16 throughout (matching the components). The reference runs fp32/bf16 on host for the PCC oracle.

## 4. Public interface (indicative)
```python
class BriaFiboPipeline:
    def __init__(self, checkpoint="briaai/FIBO", *, mesh_device, ...): ...
    def __call__(self, prompt: str, *, negative_prompt: str = "", height=1024, width=1024,
                 num_inference_steps=30, guidance_scale=5.0, seed=0, output_type="pil"): -> image
```
Batch=1 (single prompt), mirroring flux1's batch restriction. Exact signature aligned to `Flux1Pipeline` during planning.

## 5. Testing & validation
`models/tt_dit/tests/models/bria_fibo/test_pipeline.py`:
- **Glue unit tests** (host/light): the 37→46 list build matches the reference rule; `text_encoder.py`'s `prompt_embeds` matches the reference `get_prompt_embeds` (reuse sp1's encoder, already PCC-validated); empty-prompt tokenization handling.
- **End-to-end latent PCC** (device): run the tt pipeline and the reference `BriaFiboPipeline` (diffusers, host) with the same prompt + seed + a REDUCED `num_inference_steps` (e.g. 2-4, since the CPU reference transformer is slow), compare the pre-VAE latent, PCC ≥ 0.99 (allow a documented lower floor if bf16 drift over steps warrants). This is the core end-to-end gate.
- **Full image smoke** (device): a full 30-step 1024×1024 generation — assert it runs, produces a `(1024,1024,3)` image with finite values, and save the PNG for visual inspection (no reference comparison needed at full steps since the per-step path is PCC-gated).
- Env: `HF_HUB_OFFLINE=1` + pre-downloaded `briaai/FIBO` (encoder/transformer/vae/tokenizer/scheduler); the offline invocation from [[tt-dit-test-env]].

**Definition of done:** the tt pipeline's latent trajectory matches the reference at PCC ≥ 0.99 (reduced steps) on the 2×2 Blackhole mesh, and a full 30-step run produces a coherent 1024×1024 image, on real `briaai/FIBO` weights.

## 6. Open items (resolve during implementation, not blockers)
- Confirm the reference's empty-prompt tokenization special case (`bot_token_id`) and replicate it in `text_encoder.py`.
- CPU reference speed: pick the largest `num_inference_steps` for the PCC gate that runs in reasonable time (start at 2, raise if feasible); the full 30-step run is tt-only (no reference).
- Whether tracing is needed for the first functional pass (flux1 traces; for bring-up an untraced loop is acceptable, add tracing after correctness) — decide during impl.
- DRAM coresidence on the 2×2 mesh (encoder 3B + transformer 8B + VAE) — start no-eviction; add exclusions only if OOM.

## 7. Risks & mitigations
1. **Attention-mask simplification** — mitigated by the unpadded per-branch batch=1 design (§3.3): no padding ⇒ the reference's mask is a no-op ⇒ faithful. Validated by the end-to-end latent PCC.
2. **CPU reference speed for the PCC gate** — use reduced steps; the full run is tt-only.
3. **Memory coresidence** on the 2×2 mesh — flux1's no-eviction pattern is the default; add exclusions if OOM.
4. **First multi-component integration** — the components are individually PCC-validated, so failures localize to the glue; the reduced-step latent PCC + per-piece unit checks isolate issues.

## 8. Follow-on
Completes the FIBO text→image bringup. Future perf work (out of scope): batched+masked CFG (needs transformer attention-mask support), full tracing, DRAM/coresidence tuning, and the VLM-prompt-to-JSON front-end.
