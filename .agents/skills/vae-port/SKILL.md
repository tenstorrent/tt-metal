---
name: vae-port
description: Port a diffusion-model VAE encoder or decoder to TTNN, validated to reconstruction/output PCC against the diffusers reference. Use when bringing up a video VAE (3D-conv or ViT decoder), an audio VAE (1D-conv DAC/BigVGAN), or an image VAE (2D-conv) for a diffusion pipeline — including weight_norm folding, causal conv padding, GroupNorm-over-folded-time, tiling/temporal chunking, and anti-alias resampling. For text-to-* the DECODER (latents->output) is the priority; the encoder is only needed for conditioning and can be deferred.
---

# VAE Port

## Mission Context

If used as part of `$diffusion-model-bringup`, follow that skill's mission and reporting contract. A VAE turns
latents into pixels/audio (decode) or the reverse (encode). For a text-conditioned generation path, bring up
the **decoder first** — the encoder is only needed when the model is conditioned on an input image/frame/audio.

## Your Part

Implement the VAE under `models/tt_dit/models/vae/` or `models/audio_vae/`, e.g.
`decoder_<model>.py`, subclassing `models.tt_dit.layers.module.Module`, plus a state-dict adapter. Reuse the
existing tt_dit VAE stacks as templates (`vae_ltx.py`, `vae_wan2_1.py`, `vae_sd35.py`, `audio_vae/*`) and the
op library (`layers/conv3d.py`, `layers/conv2d.py`, `layers/audio_ops.py`, `layers/audio_resample.py`,
`layers/normalization.py` GroupNorm, `layers/{linear,feedforward}.py`).

## How To Approach It

- Read the diffusers VAE decode path and its config; identify conv types (1D/2D/3D, causal?), norm
  (GroupNorm groups; is the temporal axis folded into batch?), activation (SiLU/Snake/SnakeBeta), residual
  structure, up/downsampling, tiling, and the latent (de)normalization (per-channel mean/std) — which usually
  lives at the pipeline boundary, so compare the raw decoder output in the module test.
- Some "decoders" are actually a **ViT** (transformer) on latent tokens rather than a conv stack — reuse the
  model's DiT attention/rope/swiglu primitives, but watch for differences (learned residual gates vs AdaLN,
  bias flags, affine-free QK-norm, a different RoPE theta/coord convention).
- If `post_quant_conv` is 1x1x1, port it as a `Linear` — it needs no conv3d.
- Bring up smallest pieces first (a conv/resnet/AMP block, an up/downsample, one ViT block), each PCC-checked
  in isolation, then the full decoder. Keep tiling/chunking on host if needed for a first correct pass.

## Evidence To Leave

Default bar **PCC >= 0.99** (relative_rmse <= 0.05) for the full decoder vs the diffusers reference on REAL
weights (local partial-load), plus per-primitive micro-tests at **>= 0.999**. Record commands, PCC/RMSE,
precision recipe, op fallbacks, and the tiling path exercised in `doc/<stage>/work_log.md`. Skip-guard the
reference-dependent tests.

# VAE Port Knowledge

## Precision
- Default recipe: **bf16 weights+activations with fp32 accumulation (HiFi4, `fp32_dest_acc_en`,
  `packer_l1_acc`)** — matches diffusers fp16-autocast decode and clears the bars. Full-fp32 matmuls of wide
  dims overflow L1 on a single device (e.g. 2048->3072) — the bf16+fp32-acc recipe fixes this without editing
  shared layers. SDPA rejects fp32 inputs: cast Q/K/V to bf16 for the op, cast context back.

## weight_norm (audio and some conv VAEs)
- torch `weight_norm` stores `weight_g`/`weight_v`; the diffusers reference may have it removed but the raw
  checkpoint may not. **Fold at load time**: `w = weight_g * weight_v / ||weight_v||` over the norm dims
  (dim=0 → norm over dims (1,2) for Conv1d/ConvTranspose1d). Verify the fold matches `torch._weight_norm`.

## Ops and Blackhole coverage gaps
- 1D convs port via `audio_ops.Conv1dViaConv3d` / `_AlignedOutConv1d` / `ConvTranspose1dViaConv3d`;
  Snake/SnakeBeta via `ttnn.snake_beta` (fp32 works on Blackhole) or the decomposed `audio_ops.Snake`.
- **Single-device wide-channel gap**: `audio_resample.Activation1d`'s HEIGHT_SHARDED depthwise `ttnn.conv1d`
  anti-alias filter overflows L1 for wide channels on 1 device (LTX tests only run >=8-device meshes and skip
  it). Workaround for fixed kaiser-sinc taps: a bit-equivalent shift-and-add (strided slice + scalar mul/add).
  The regenerated kaiser-sinc taps match stored `*.filter` buffers exactly, so those buffers can be dropped.
- 3D-conv encoders: `ttnn.experimental.conv3d` supports zeros/replicate pad, NOT reflect — pre-pad explicitly.
  GroupNorm with the temporal axis folded into batch needs the folded layout preserved.
