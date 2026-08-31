---
name: diffusion-full-pipeline
description: Assemble the validated components of a TTNN diffusion model (text encoder + DiT + scheduler + VAE decoders) into an end-to-end generation pipeline that produces a real artifact (image/video/audio), and validate it is non-degenerate. Use when wiring encode -> denoise loop -> VAE decode -> save/mux into one host-orchestrated pipeline, managing multichip device residency and mesh lifecycle. Diffusion analog of $full-model.
---

# Diffusion Full Pipeline

## Mission Context

If used as part of `$diffusion-model-bringup`, follow its contract. This stage ties the already-PCC-validated
components into one pipeline in `models/tt_dit/pipelines/<model>/pipeline_<model>.py` + a generate script + a
smoke test, and produces a real, non-degenerate generated artifact.

## Your Part

`__call__(prompt, height, width, num_frames/…, num_inference_steps, seed)`:
tokenize → encode (conditioning hidden states) → init latents from seeded noise → for each scheduler step
build the per-step host layout (position_ids/timestep_indices/adaln_indices/rope), run the DiT once, step the
scheduler on the GENERATED rows only → unpatchify + un-normalize (per-channel latents_mean/std) → VAE decode →
save frames/mp4 + wav (+ mux). No CFG if the model is guidance-distilled.

## Hard-won gotchas (read these — each cost real debugging)

- **MESH LIFECYCLE is the #1 trap.** Do NOT create+close a submesh per stage. Repeated
  `create_submesh` + `close_mesh_device` on a FABRIC_1D-enabled parent mesh **deadlocks the next collective**
  (looks like a multi-hour hang; the DiT forward or a later submesh `create` wedges the device → needs
  `tt-smi -r`). Run every stage on the ONE persistent parent mesh: the encoder/DiT are tensor-parallel over
  it; single-device VAE decoders run **replicated** on the full mesh (read back from device 0). This mirrors
  LTX (VAE on `self.mesh_device`) and SD35/Flux (carve submeshes ONCE at init, never close mid-run). Only free
  model WEIGHTS between stages, never the mesh.
- **Device residency / DRAM staging.** A large encoder + a large DiT + VAEs won't all fit. Build → run → free
  each heavy model's weights in sequence so exactly one is resident (log peak DRAM/ASIC). `from_pretrained`
  then just records paths/config; heavy device models are built inside `__call__`.
- **Degenerate small configs.** Video decoders with temporal chunking (`17n+5 → 5n+2`, token_drop) yield ZERO
  output chunks below a minimum (`num_frames` must be `17n+5, n≥1`, e.g. 22 — not 5). Validate the geometry
  before running: an empty `torch.cat` in decode means too few frames.
- **Degenerate timesteps.** At step 0 both modality schedulers start at σ=1 → t=0, so `torch.unique` collapses
  to `num_ts=1`; a size-1 timestep dimension can wedge a matmul/CCL op. Duplicate a lone timestep to 2
  (numerically identical) as a safety.
- **Per-step host prep only.** Build cos/sin, timestep tensors, and index tensors on host and upload each step;
  keep the device forward free of torch/from_torch/to_torch.
- **Instrument the loop** (log before/after each forward + sync + decode) so a hang is localized in seconds,
  not by staring at silence. Use hard `timeout` on every device run during bringup; a real hang wedges the
  device.

## Evidence To Leave

- A saved, NON-DEGENERATE artifact: frames not black/NaN, real frame-to-frame motion (mean abs diff > 0),
  audio not silent/NaN (rms > 0). Report the paths + these stats. Content quality at tiny res/steps will be
  low — that is acceptable for the first end-to-end; scale res/steps for real content.
- A few-step latent-trajectory PCC vs a diffusers CPU reference where feasible (else document why and rely on
  degeneracy + eyeball). Wall-clock breakdown (encode / DiT-load / denoise / decode) — weight loading usually
  dominates; flag it for `$optimize`/`$tt-enable-tracing`.
- `doc/<stage>/work_log.md` with configs, timings, artifact paths, and the mesh/staging decisions.
