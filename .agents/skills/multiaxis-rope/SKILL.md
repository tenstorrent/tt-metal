---
name: multiaxis-rope
description: Implement and validate multi-axis (2D/3D/N-D) rotary position embedding on Tenstorrent for diffusion transformers and other models whose tokens carry more than one positional coordinate (e.g. video DiT with (t,h,w) grids, image DiT with (h,w)). Use when a model's RoPE spans several axes, uses partial rotary (rotates only some head channels), or uses the rotate_half vs interleaved convention, and you need a device implementation that matches the HF/diffusers reference.
---

# Multi-Axis RoPE

## Mission Context

A reference skill consumed by `$functional-dit-block` and `$diffusion-model-bringup`. Multi-axis RoPE assigns
each token a coordinate per axis (e.g. `(t,h,w)`), computes per-axis angles from a shared or per-axis
`inv_freq`, concatenates them, and rotates a subset of head-dim channels. Get the convention and channel
accounting exactly right or PCC collapses.

## How To Approach It

Read the reference `RotaryPosEmbed`/`_apply_rotary_emb` line by line and record, precisely:
- `inv_freq` construction (theta, how many freqs per axis, shared vs per-axis).
- How `position_ids (seq, n_axes)` map to angles: `freqs = pos[...,None] * inv_freq`, then the concat order of
  the per-axis blocks, and any final `cat(freqs, freqs)` doubling.
- **rotary_dim**: how many of `head_dim` channels are rotated (`2 * n_axes * freqs_per_axis`), and that the rest
  pass through unchanged.
- The convention: rotate_half/GPT-NeoX (`cat(-x[d/2:], x[:d/2])`, cos/sin duplicated) vs interleaved/LLaMA.

## Device implementation

- Build cos/sin on host (fp32) from `position_ids`, then ship to device. Shape them to broadcast over the
  head layout used by the attention module.
- For **rotate_half with partial rotary**, the half-split is at `rotary_dim/2`, which is frequently NOT
  tile-aligned (e.g. 48 for rotary_dim 96) — plain TILE-layout slicing breaks. Two robust options:
  1. A constant `(head_dim, head_dim)` matmul implementing rotate_half (±1 permutation over rotated channels,
     identity on pass-through), with cos/sin padded to head_dim (cos=1, sin=0 on pass-through). bf16-exact,
     tile-friendly. Best default for functional correctness.
  2. `ttnn.experimental.rotary_embedding_hf` — but only if its half-split (head_dim/2) matches the model's
     rotary_dim/2. It will NOT match partial rotary where rotary_dim < head_dim.
- `rotary_embedding_llama` needs the interleaved convention and a `trans_mat`; using it for a rotate_half
  checkpoint requires a per-head SPLIT->INTERLEAVED permute of the Q/K weights at load time.

## Evidence To Leave

- An isolated rope micro-test: apply device RoPE to random Q (and K) and compare to the reference
  `_apply_rotary_emb` at **PCC >= 0.999**. Cover non-trivial multi-axis `position_ids` (varied per-axis
  coordinates), and verify the pass-through channels are untouched.
- Record: rotary_dim vs head_dim, convention, and which device path was used (and why).
