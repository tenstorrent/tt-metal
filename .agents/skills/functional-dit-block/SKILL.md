---
name: functional-dit-block
description: Bring up a functionally correct TTNN implementation of a diffusion-transformer (DiT) block for a video/image diffusion model, validated to PCC against the diffusers reference. Use when producing the per-block TTNN module for a DiT (self-attention with QK-norm and multi-axis RoPE, AdaLN/timestep modulation, SwiGLU/GELU FFN), reading the diffusers block, comparing on a 1x1 mesh, and leaving compact correctness evidence. This is the diffusion analog of $functional-decoder (there is no KV-cache, no autoregressive decode).
---

# Functional DiT Block Bringup

## Mission Context

If this skill is used as part of `$diffusion-model-bringup`, follow that skill's mission, workspace, and
reporting contract. This stage turns a diffusers DiT transformer block into complete, working, tested TTNN
code. Diffusion transformers denoise a fixed latent sequence — there is **no KV cache, no prefill/decode
split, no causal mask**. Bring the block up on a single 1x1 mesh; parallelism is a later stage.

## Your Part

Implement the block(s) under the model's `tt_dit` package, e.g.
`models/tt_dit/models/transformers/<model>/transformer_<model>.py` (+ `attention_<model>.py`,
`rope_<model>.py`). Subclass `models.tt_dit.layers.module.Module`, implement `_prepare_torch_state`
(rename diffusers keys to the tt_dit layer names) and load via `load_torch_state_dict`. Reuse the tt_dit
primitives: `layers.linear.Linear` (fused `activation_fn="swiglu"`), `layers.normalization.RMSNorm`,
`layers.feedforward.FeedForward`, and `ttnn.transformer.scaled_dot_product_attention` (is_causal=False).

## How To Approach It

Read the diffusers block, attention, and rope classes line by line. Identify: the modulation scheme (adaLN
table vs per-block scale_shift_table), QK-norm placement (per-head vs whole-row), the RoPE convention and
which channels it rotates, the FFN activation, and bias flags. Bring up the smallest pieces first
(RoPE application, then the modulation gather, then attention, then the whole block), each PCC-checked in
isolation against the reference before assembling. Correctness first: bf16 / TILE / DRAM. Escalate matmul
fidelity (HiFi4, `fp32_dest_acc_en`) or specific modules to fp32 only when PCC needs it. Keep runtime free
of torch / `ttnn.from_torch` / `ttnn.to_torch` outside `from_state_dict`/setup and the test boundary. If a
correctness bug resists ordinary narrowing, use `$autofix`.

## Evidence To Leave

Default acceptance bar **PCC >= 0.99** (relative_rmse <= 0.05) per block kind, via `utils.check.assert_quality`.
Done means all recorded:
- HF-diffusers-vs-TTNN PCC for each block kind, on a 1x1 mesh, with SYNTHETIC weights (shared between ref
  and TT) for CI, plus at least one run with REAL weights (partial safetensors load of one block).
- Isolated micro-tests for the RoPE application and the modulation/adaLN gather.
- A skip-guard so reference-dependent tests SKIP (not error) when the diffusers model is unimportable.
- `doc/<stage>/README.md` + `work_log.md` recording commands, PCC/RMSE, precision choices, and quirks.
- Watcher note: run watcher-enabled if the environment permits; if the box emits a known watcher
  false-positive that reproduces on unrelated pre-existing tests, record that evidence rather than skipping asserts.

# DiT Block Knowledge

## Reference environment
- diffusers modeling code for a new model may not be in the installed `diffusers`. Do NOT upgrade the pinned
  diffusers (it breaks other tt_dit reference tests). Instead clone the diffusers source that has the model
  and run the reference with `PYTHONPATH=<diffusers>/src`, reusing the venv's torch/transformers.
- Reference block weights via **partial safetensors load**: read the `*.index.json` weight_map, collect the
  block-prefix keys, `safe_open` only those shards. Avoids materializing the whole (tens-of-GB) model.

## RoPE conventions (critical)
- **rotate_half / GPT-NeoX** convention (`rotate_half(x)=cat(-x[d/2:], x[:d/2])`, cos/sin duplicated) is NOT
  the interleaved (LLaMA) convention `ttnn.experimental.rotary_embedding_llama` expects. `rotary_embedding_hf`
  uses rotate_half but splits at head_dim/2 — if the model does **partial** rotary (rotates only k<head_dim
  channels), its half-split is at k/2, which `rotary_embedding_hf` won't match.
- Robust functional approach for partial rotate_half rope: apply as a constant `(head_dim, head_dim)` matmul
  (a permutation with ±1 columns implementing rotate_half over the rotated channels, identity on the
  pass-through channels) combined with cos/sin padded to head_dim (cos=1, sin=0 on pass-through channels).
  bf16-exact and tile-friendly (avoids non-tile-aligned slicing at k/2). Optimize to a fused op in $optimize.

## AdaLN modulation
- If each block owns an `adaln_proj` Linear producing a per-(timestep, modality) table gathered per row by an
  index tensor (not a learned scale_shift_table + temb broadcast): compute `adaln_proj(silu(temb))`, reshape
  row-major to `(num_rows, n_params*H)`, and gather per sequence position with **`ttnn.embedding(indices,
  table)`** (indices uint32). Then modulate with `ttnn.addcmul(shift, normed, add(scale, 1.0))` and gate with
  `ttnn.addcmul(residual, gate, sublayer_out)`. `silu` runs at temb precision (fp32 if time_embedder is fp32)
  before the projection cast — replicate for exactness.

## FFN / QK-norm gotchas
- SwiGLU value/gate ordering: diffusers `SwiGLU`/`FeedForward` pack `[value|gate]` (`out=value*silu(gate)`),
  which matches tt_dit `Linear(activation_fn="swiglu")` — no reorder needed. Confirm for other frameworks.
- Per-head QK-norm normalizes over head_dim independently per head; whole-row QK-norm normalizes the full row
  before the head split — they are NOT equivalent. Match the reference.
- The large FFN reduction dim benefits from HiFi4 (measurably lowers RMSE).

## Numerical artifact to expect
- With N(0, 0.1^2) small-init, per-head QK-RMSNorm makes attention logits tiny -> near-uniform softmax -> the
  ungated attn+ff output is a delicate average of large v-values that amplifies bf16 rounding (loosens
  relative-RMSE while PCC stays high). Gated DiT blocks hide this; an ungated refiner block may need a looser
  RMSE bound with PCC still >= 0.99. This is a small-init artifact, not a bug — document it.
