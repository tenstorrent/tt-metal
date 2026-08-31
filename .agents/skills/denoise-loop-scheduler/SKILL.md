---
name: denoise-loop-scheduler
description: Implement and validate a diffusion sampler/scheduler and the iterative denoise loop on Tenstorrent — the diffusion counterpart of the LLM decode loop. Use when porting a rectified-flow / flow-matching / DDIM / EDM scheduler, building the multi-step denoise driver (optionally with classifier-free guidance and multiple modality schedules), and making it trace-safe. Schedulers are host-side tensor math; the per-step model call is the device work.
---

# Denoise Loop & Scheduler

## Mission Context

If used as part of `$diffusion-model-bringup`, follow that skill's contract. The scheduler advances the latent
across timesteps; the denoise loop calls the DiT once per step and updates the latents. There is no KV cache;
the "loop" is a fixed number of denoise steps, not autoregressive token generation.

## Your Part

Implement a self-contained scheduler (no dependency on the diffusers modeling code at runtime) under the
model's `pipelines/<model>/` package, and the denoise-loop driver in the pipeline. Validate the scheduler
against the diffusers reference and the full loop against a reference trajectory.

## How To Approach It

- Read the reference scheduler and record its EXACT conventions — small sign/scale differences are common and
  silently wrong:
  - velocity sign (`x0 = x_t +/- sigma * v`), timestep convention (`t = sigma` vs `t = 1 - sigma`, scale
    1000x vs [0,1]), the sigma grid construction (linspace bounds, terminal 0, shift, dedup), and whether the
    x0 sigma comes from the timestep vs the sigma grid (float32 round-trip differences matter).
  - `eta` (ancestral noise) — many "euler ancestral" configs actually run eta=0 (deterministic).
- A model may run **multiple schedules per request** (e.g. one per modality with different shift), held as
  separate scheduler instances by the pipeline.
- CFG: check whether the model is **guidance-distilled** — if so there is NO classifier-free guidance (single
  forward per step, no unconditional branch, no negative prompt). Don't add CFG that isn't there.
- The scheduler `step` writes every row it's given; the pipeline must hand it only the GENERATED rows so clean
  conditioning anchors persist across steps.
- Make the loop trace-safe (stable input buffers, tensor timesteps) via `$tt-enable-tracing` for perf.

## Evidence To Leave

- Scheduler parity vs the diffusers reference: schedule construction (sigmas/timesteps) and a full-trajectory
  `step` comparison — aim for **bit-exact** (host math) for both/all modality shifts.
- Self-consistency tests independent of diffusers (schedule invariants, scale_noise endpoints, determinism) so
  CI passes without the reference on PYTHONPATH.
- Full denoise-loop trajectory PCC vs a reference run (latent after N steps), and a degeneracy check (no
  NaN/frozen latents).
- Record in `doc/<stage>/work_log.md`.
