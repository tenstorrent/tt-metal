---
name: adaln-conditioning
description: Implement and validate adaptive layer-norm (AdaLN / AdaLN-Zero) timestep-and-conditioning modulation on Tenstorrent for diffusion transformers. Use when a DiT block modulates its norms with shift/scale/gate vectors derived from a timestep (and optionally modality/class) embedding, including the per-row table-gather variant, the learned scale_shift_table variant, and the sinusoidal timestep embedding that feeds them.
---

# AdaLN / Timestep Conditioning

## Mission Context

A reference skill consumed by `$functional-dit-block`, `$denoise-loop-scheduler`, and
`$diffusion-model-bringup`. AdaLN turns a timestep (and sometimes modality) embedding into per-token
`shift/scale/gate` parameters that modulate a block's pre-norm activations and gate its sublayer outputs.

## Variants (identify which the model uses)

1. **Learned `scale_shift_table` + temb broadcast** (SD3/LTX/Wan style): a per-block Parameter
   `(n_coeff, 1, 1, D)` added to a broadcast timestep embedding, chunked into the coeffs. Bake the `+1` into
   the scale slots at load time so each modulation is a single `ttnn.addcmul`. Store the table sharded on D
   for tensor-parallel; `ttnn.chunk(dim=0)` is then a free tile-aligned slice.
2. **Per-(timestep, modality) table gathered per row** (e.g. MiniMax-H3): a per-block `adaln_proj` Linear maps
   `silu(temb)` to `n_params * D * n_modalities`; reshape row-major to `(num_rows, n_params*D)` and gather one
   row per sequence position with `ttnn.embedding(indices, table)` where `indices = timestep_index*n_modalities
   + modality_tag` (uint32). Then split into the params.

## How To Approach It

- **Timestep embedding**: reuse `tt_dit/layers/embeddings.py` `Timesteps` (sinusoidal) + `TimestepEmbedding`
  (2-Linear SiLU). Match `flip_sin_to_cos`/`downscale_freq_shift` and the SiLU precision to the reference.
- Run the modulation activation (`silu`) at the timestep embedding's precision (fp32 if the time embedder is
  fp32) BEFORE casting to the projection dtype — a rounding applied before the activation biases every block
  identically and accumulates over the denoising trajectory.
- Apply modulation as `ttnn.addcmul(shift, normed, add(scale, 1.0))` and gate as
  `ttnn.addcmul(residual, gate, sublayer_out)`.
- AdaLN-Zero: gate/output projections are zero-initialized so blocks start as identity — verify the zero-init
  is preserved through weight loading.

## Evidence To Leave

- An isolated modulation test: compare all gathered/broadcast params (shift/scale/gate for each norm) to the
  reference at **PCC >= 0.999**.
- Timestep-embedding parity vs the reference (sinusoidal + MLP) at PCC >= 0.999.
- For the output norm (`AdaLayerNormOut`), verify the per-timestep (not per-modality) gather if the model
  differs there.
