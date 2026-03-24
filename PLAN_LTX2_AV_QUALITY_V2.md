# Fix LTX-2 AudioVideo Pipeline Quality to Match Official Reference

This ExecPlan is a living document maintained in accordance with PLANS.md at the repository root.
The sections Progress, Surprises & Discoveries, Decision Log, and Key Measurements must be
kept up to date as work proceeds.

## Purpose

The TTNN LTX-2.3 AudioVideo pipeline currently produces "borderline garbage" — a barely recognizable cat and meaningless audio. After this work, running `generate_audio_video.py` with the same prompt and seed will produce video and audio quality matching the official `ltx_pipelines.ti2vid_two_stages` pipeline. The user should see a clearly recognizable scene with coherent audio.

**Root cause**: The TTNN pipeline has several critical divergences from the official reference that compound to destroy output quality. Each divergence is small in isolation but together they produce unusable output.

**How to verify**: Generate a video with both pipelines using the same prompt and seed, then visually compare frame quality and listen to audio.

## Acceptance Criteria

- Video shows recognizable scene matching the prompt (subjective visual quality)
- Audio is coherent and related to the scene (not noise/static)
- PSNR > 20 dB vs CPU reference pipeline output (same prompt, same seed, same steps)

## Reference Implementation

- Official TI2VID pipeline: `LTX-2/packages/ltx-pipelines/src/ltx_pipelines/ti2vid_two_stages.py`
- Pipeline params: `LTX-2/packages/ltx-pipelines/src/ltx_pipelines/utils/constants.py` — `LTX_2_3_PARAMS`
- Multi-modal guider: `LTX-2/packages/ltx-core/src/ltx_core/components/guiders.py` — `MultiModalGuider.calculate()`
- Denoising loop: `LTX-2/packages/ltx-pipelines/src/ltx_pipelines/utils/helpers.py` — `multi_modal_guider_denoising_func()`
- Default negative prompt: `LTX-2/packages/ltx-pipelines/src/ltx_pipelines/utils/constants.py` — `DEFAULT_NEGATIVE_PROMPT`
- TTNN pipeline: `models/tt_dit/demos/ltx/generate_audio_video.py`

## Context and Orientation

The TTNN pipeline runs on a WH LB 2x4 mesh (8 chips). Per-layer PCC is 0.999 vs reference — the transformer itself is numerically accurate. The quality problem is entirely in the **pipeline orchestration**: how we prepare inputs, apply guidance, and combine model predictions.

### Critical Divergences (ordered by impact)

**1. Wrong audio CFG scale (HIGH IMPACT)**
Our pipeline uses `guidance_scale=3.0` for both video and audio. The official LTX-2.3 defaults are:
- Video: `cfg_scale=3.0` (correct)
- Audio: `cfg_scale=7.0` (WRONG — we use 3.0)

Audio needs stronger guidance to produce coherent output. Using 3.0 instead of 7.0 means the audio barely follows the prompt, resulting in noise/static.

**2. Missing negative prompt (HIGH IMPACT)**
We encode `""` (empty string) as the negative prompt. The official pipeline uses a 700-character detailed negative prompt (`DEFAULT_NEGATIVE_PROMPT`) that steers the model away from artifacts, blurriness, distortion, robotic voice, etc. Without it, CFG has much less effect — the unconditional baseline is too weak.

**3. Missing STG guidance (MEDIUM IMPACT)**
The official pipeline runs a 3rd model pass per step with "Spatio-Temporal Guidance" — it perturbs block 28's self-attention to create a perturbed prediction, then applies:
```
pred += stg_scale * (cond - perturbed)  # stg_scale=1.0
```
This improves spatial coherence and temporal consistency. We skip this entirely.

**4. Missing modality guidance (MEDIUM IMPACT)**
The official pipeline runs a 4th model pass per step with A↔V cross-attention disabled, then applies:
```
pred += (modality_scale - 1) * (cond - isolated)  # modality_scale=3.0
```
This strengthens the audio-video alignment. We skip this entirely.

**5. Two-stage architecture (LOW IMPACT for now)**
The official pipeline runs stage 1 at half resolution (256x384 for 512x768 target), then upscales with a spatial upsampler + distilled LoRA in stage 2. We run one stage at full resolution. This affects fine detail but not fundamental quality — the cat should still be recognizable without stage 2.

### What Each Fix Costs in Compute

| Fix | Extra model passes/step | Time impact |
|-----|------------------------|-------------|
| Audio CFG scale | 0 (just change a number) | None |
| Negative prompt | 0 (just change the string) | None |
| STG guidance | +1 pass (48 layers) | +50% |
| Modality guidance | +1 pass (48 layers) | +50% |
| Total with all fixes | 4 passes vs current 2 | 2× time |

## Plan of Work (Milestones)

### Milestone 1: Fix the two free wins (audio CFG + negative prompt)

These cost zero extra compute and are likely the biggest quality improvements.

**What to edit:** `models/tt_dit/demos/ltx/generate_audio_video.py`

1. **Add separate video/audio CFG scales** — replace single `--guidance_scale` with `--video-cfg-scale` (default 3.0) and `--audio-cfg-scale` (default 7.0).

2. **Use official negative prompt** — import `DEFAULT_NEGATIVE_PROMPT` from `ltx_pipelines.utils.constants` and use it as the default for `--negative-prompt`.

3. **Apply different CFG scales per modality** — in the CFG block, use `video_cfg_scale` for video and `audio_cfg_scale` for audio.

**What to run:**
```bash
source python_env/bin/activate && export PYTHONPATH=$(pwd)
export HF_TOKEN=<your-hf-token>
python models/tt_dit/demos/ltx/generate_audio_video.py \
    --num_frames 33 --height 256 --width 256 --steps 5 --seed 10 \
    --output /tmp/av_m1_test.mp4 --no-cross-pe
```

**What to expect:** Cat should be clearly recognizable. Audio should sound less like static. If this alone fixes quality, stop here and verify at full resolution.

**What to do if it fails:** If still garbage, proceed to Milestone 2.

### Milestone 2: Add STG and modality guidance

**What to edit:** `models/tt_dit/demos/ltx/generate_audio_video.py`

1. **Add STG pass** — run a 3rd model forward with self-attention block 28 replaced by pass-through for both video and audio. This requires:
   - Adding a `perturbation_blocks` parameter to `LTXAudioVideoTransformerModel.inner_step`
   - Modifying the transformer block at index 28 to skip self-attention when perturbation is active
   - Computing `stg_denoised_video/audio` from this pass

2. **Add modality guidance pass** — run a 4th model forward with A↔V cross-attention disabled. This requires:
   - Adding a `skip_cross_attention` parameter to `LTXAudioVideoTransformerBlock.forward`
   - When active, skip the A→V and V→A cross-attention (just pass through residual)
   - Computing `mod_denoised_video/audio` from this pass

3. **Apply full MultiModalGuider formula:**
```python
# Video
v_pred = (v_denoised
    + (video_cfg_scale - 1) * (v_denoised - v_uncond)
    + video_stg_scale * (v_denoised - v_perturbed)
    + (video_modality_scale - 1) * (v_denoised - v_isolated))
# Rescale
v_factor = rescale * (v_denoised.std() / v_pred.std()) + (1 - rescale)
v_pred = v_pred * v_factor

# Audio (same formula with audio params)
a_pred = (a_denoised
    + (audio_cfg_scale - 1) * (a_denoised - a_uncond)
    + audio_stg_scale * (a_denoised - a_perturbed)
    + (audio_modality_scale - 1) * (a_denoised - a_isolated))
a_factor = rescale * (a_denoised.std() / a_pred.std()) + (1 - rescale)
a_pred = a_pred * a_factor
```

**Default parameters (from LTX_2_3_PARAMS):**
- Video: cfg=3.0, stg=1.0, rescale=0.7, modality=3.0, stg_blocks=[28]
- Audio: cfg=7.0, stg=1.0, rescale=0.7, modality=3.0, stg_blocks=[28]

**What to run:** Same as Milestone 1 but verify with 30-step full-res run.

**What to expect:** Significantly improved quality over Milestone 1, matching official pipeline output.

### Milestone 3: Generate CPU reference for PSNR comparison

Run the official TI2VID pipeline on CPU with the same parameters and compute PSNR against our output.

```bash
# This may need GPU or take very long on CPU
python LTX-2/packages/ltx-pipelines/src/ltx_pipelines/ti2vid_two_stages.py \
    --checkpoint-path ~/.cache/ltx-checkpoints/ltx-2.3-22b-dev.safetensors \
    --prompt "A cat playing piano in a cozy room with warm lighting" \
    --seed 42 --num-frames 33 --height 256 --width 256 --num-inference-steps 5 \
    --output-path /tmp/cpu_ref.mp4
```

If CPU reference is too slow, use the 1-layer PCC test infrastructure to compare latents at small scale.

## Progress

- [x] (2026-03-23 20:50Z) Milestone 1: Fix audio CFG scale (3.0→7.0) and negative prompt (DEFAULT_NEGATIVE_PROMPT)
- [x] (2026-03-23 20:50Z) Milestone 2: Add modality guidance (skip_cross_attn pass, modality_scale=3.0)
- [x] (2026-03-23 20:50Z) Milestone 2: Apply full MultiModalGuider formula with separate video/audio params
- [x] (2026-03-23 20:55Z) Milestone 1+2: Verified at 256x256, 5 steps — video+audio 1.38s, 3 passes/step
- [x] (2026-03-23 21:10Z) Full-res with CFG+modality (3 passes): 529.0s denoise, 17.6s/step
- [x] (2026-03-23 21:15Z) Milestone 2: Add STG guidance (skip_self_attn at stg_block per layer)
- [x] (2026-03-23 21:19Z) Verified 256x256 with all 4 passes: 40.5s (8.1s/step)
- [x] (2026-03-23 21:40Z) Full-res with all 4 guidance passes: 709.4s denoise (23.6s/step)
- [x] (2026-03-23 22:12Z) Milestone 3: Head-to-head comparison — audio diverges rapidly due to tile padding (34→64 = 88% zero padding). Full-res (126→128 = 1.6% padding) should be much better.
- [x] (2026-03-23 22:40Z) Fix av_ca gate timestep scaling (was using 1000×, reference uses 1×)
- [x] (2026-03-23 22:40Z) Add audio attention mask for padded K positions in SDPA
- [x] (2026-03-23 22:40Z) Zero-pad audio latent (not noise) for padded positions, zero after Euler step
- [x] (2026-03-23 23:04Z) Full-res regeneration with all fixes: 706.6s denoise
- [x] (2026-03-23 23:35Z) Run official CPU reference pipeline (33f 256x256 5 steps): produces recognizable output
- [x] (2026-03-23 23:41Z) TTNN vs CPU reference PSNR: 18.7 dB (was 23-24 for video-only)
- [x] (2026-03-24 01:04Z) 512x768 48-layer step-by-step comparison: audio vel PCC=0.88 at step 0, drops to 0.77 at step 1. Video starts at 0.98 but drops to 0.79 by step 1. Latent ranges diverge — ref audio grows, TT audio stays small.
- [x] (2026-03-24 03:10Z) Layer-by-layer PCC analysis complete. Audio PCC decay accelerates at layer 30+: 0.997@10L → 0.986@30L → 0.882@48L. Video stays 0.977-0.989. Per-layer PCC is 0.998 — the issue is NOT a single-op bug but accelerating cross-modal error feedback in deeper layers. Audio error compounds 3× faster than video due to smaller dimension (2048 vs 4096) and cross-attention amplification.
- [ ] Investigate: do deeper layer gate weights or FF weights have extreme values that amplify errors?
- [ ] Try: run with HiFi4 for all audio matmuls to see if precision improvement helps
- [ ] Check if reference runs audio in fp32 while we run in bf16 (model dtype mismatch)
- [ ] Check if there's an op ordering difference between reference and TTNN for the audio path

## Surprises & Discoveries

- Audio vocoder output sample rate is 48kHz (not 16kHz as initially assumed). Previous code passed 16000 to encode_video, causing 3× audio duration mismatch.
- The official LTX-2.3 pipeline uses 4 model passes per step: conditional, unconditional (CFG), perturbed (STG), isolated-modality. Our pipeline was doing only 2.
- Audio CFG scale default is 7.0 (not 3.0 like video). Audio needs much stronger guidance.
- DEFAULT_NEGATIVE_PROMPT is 700+ characters describing every kind of artifact to avoid. Empty negative makes CFG nearly ineffective.
- **CRITICAL: Audio token padding (34→64) causes rapid divergence.** Head-to-head comparison shows audio latent PCC drops to 0.659 after just 3 steps (1-layer model). The reference uses 34 tokens (unpadded); our padding to 64 inserts 30 zero tokens that participate in attention/FFN, diluting the signal and causing 2-3× scale amplification in audio latents. Video diverges more slowly (0.962 after 3 steps) since its token count (320) is already tile-aligned.

## Decision Log

- Decision: Fix audio CFG and negative prompt first (Milestone 1), because they are zero-cost and likely the biggest quality impact.
  Rationale: Audio cfg_scale 3.0 vs 7.0 is a 2.3× difference in guidance strength. Empty negative prompt vs 700-char detailed negative makes CFG nearly ineffective.
  Date: 2026-03-23

- Decision: Defer two-stage upscaling (requires spatial upsampler + distilled LoRA) to a future plan.
  Rationale: Stage 1 at full resolution should still produce recognizable content. Stage 2 improves fine detail but shouldn't be needed for basic quality.
  Date: 2026-03-23

## Constraints & Workarounds

- Hardware: WH LB 2x4 mesh (SP=2, TP=4), 8 chips
- STG and modality guidance add 2 extra model passes per step, doubling denoise time (~700s for 30 steps at full-res)
- Cross PE host fallback adds ~2.4× overhead when enabled; use `--no-cross-pe` for fast iteration
- Video VAE decode runs on CPU (~260s) — not a quality issue but affects turnaround

## Key Measurements

| Test | Metric | Value | Notes |
|------|--------|-------|-------|
| Current output | Visual | "borderline garbage" | Missing CFG, negative prompt, STG, modality guidance |
| Per-layer PCC | PCC | 0.999 | Transformer is accurate; problem is pipeline |
| Full-res CFG+modality only (3 passes) | Denoise | 529.0s | 30 steps, 17.6s/step |
| Full-res all guidance (4 passes) | Denoise | 709.4s | 30 steps, 23.6s/step |
| Full-res no guidance (2 passes, old) | Denoise | 353.5s | 30 steps, 11.8s/step |
