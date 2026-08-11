---
name: diffusion-qualitative-check
description: Perceptually and quantitatively check the output of a TTNN diffusion pipeline (image/video/audio) for non-degeneracy and quality. Use to gate an end-to-end diffusion generation: verify frames aren't black/NaN/frozen, audio isn't silent/NaN, latents track a reference, and (for conditioned generation) the output plausibly follows the prompt. Diffusion analog of $qualitative-check (which is text-LLM specific).
---

# Diffusion Qualitative Check

## Mission Context

If used as part of `$diffusion-model-bringup`, follow its contract. A per-step velocity PCC can be high while
the generated media is still garbage (or vice-versa) — for a generative model the END artifact is the real
gate. This skill defines the non-degeneracy + quality checks that gate `$diffusion-full-pipeline`.

## Checks

Load the saved artifact and assert:
- **Video**: value range sane (not all one value), **no NaN/Inf**, **no all-black/all-white frames**, and real
  **temporal motion** (mean per-pixel abs diff between consecutive frames > a small threshold; ~0 => frozen).
- **Audio**: **not silent** (rms above a floor), no NaN/Inf, amplitude within range (e.g. |x|≤1 after clamp),
  plausible envelope/spectral energy (not DC, not a constant tone).
- **Latents**: where a reference is feasible, latent-trajectory or final-latent PCC vs a diffusers CPU run at a
  tiny config; and per-step latent norms evolve (not stuck/exploding to NaN).
- **Prompt adherence** (conditioned models): at real resolution/steps, eyeball or use a CLIP/embedding score
  that the content follows the prompt. At tiny res/few steps, content is expected to be abstract — do NOT fail
  on low fidelity there; only fail on DEGENERACY.

## How To Approach It

- Separate **degeneracy** (a hard bug: NaN, black frames, silence, frozen video) from **low fidelity** (small
  config, few steps — expected, not a failure). Gate on the former; report the latter.
- Provide a small `scripts/` helper that ingests frames+wav and prints the stats + pass/fail, analogous to the
  LLM `check_degenerate_output.py`.
- If output is degenerate, bisect: is the DiT velocity sane (PCC vs ref)? the denoise trajectory? the VAE
  decode (feed the reference latents into the TT VAE)? the un-normalization (latents_mean/std)?

## Evidence To Leave

- The stats (frame range/motion, audio rms, latent PCC where available) + an explicit pass/fail, recorded in
  `doc/<stage>/work_log.md`, and the saved artifact paths so a human can eyeball. State the config
  (res/frames/steps) so fidelity is judged in context.
