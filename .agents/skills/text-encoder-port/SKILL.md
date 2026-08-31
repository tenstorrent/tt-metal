---
name: text-encoder-port
description: Port a diffusion model's conditioning text (or text-vision) encoder to TTNN, validated to hidden-state PCC against the HF/transformers reference. Use when a diffusion pipeline conditions on a T5 / CLIP / Gemma / Qwen-VL encoder and you need its (possibly intermediate-layer) hidden states on device to feed the DiT's context embedder. Prefer reusing an existing tt_dit encoder (clip/t5/gemma/qwen25vl/qwen3vl) over writing a new one.
---

# Text Encoder Port

## Mission Context

If used as part of `$diffusion-model-bringup`, follow that skill's contract. The text encoder turns the prompt
into the conditioning hidden states the DiT cross-attends to (or, for single-stream DiTs, projects and packs
into the sequence). Many diffusion models take an INTERMEDIATE layer's hidden state, not the final one.

## Your Part

Reuse an existing `models/tt_dit/encoders/<name>/` implementation where possible; add a thin model-specific
wrapper + weight adapter + test under `tests/models/<model>/`. Only write a new encoder if none matches.

## How To Approach It

- Read how the diffusion pipeline calls the encoder: which submodule (often `model.language_model`, not the
  full CausalLM/VLM), which hidden-state layer index, tokenization (chat template? special tokens?), and the
  output shape/dtype fed to the DiT `context_embedder`.
- **Layer tap off-by-one**: HF `outputs.hidden_states[k]` is the output of `layers[k-1]` (index 0 is the
  embedding). If a tt_dit encoder taps by `layers[i]` output, use `activation_layers=(k-1,)` to reproduce
  `hidden_states[k]`, and verify it is the RAW pre-final-norm state. Truncate the model to the needed layers.
- **Config threading is the #1 correctness trap**: a config-driven encoder written for one checkpoint size may
  hardcode/derive params that differ for a larger one. For Qwen3-VL-32B the killers were **decoupled
  `head_dim` (128, not hidden/heads=80), `rope_theta` (5e6, not the 8B default), and `mrope_section`** — an
  un-threaded rope_theta alone collapsed PCC to ~14%. Thread head_dim/rope_theta/mrope_section (and any
  size-specific rope/attention params) explicitly, defaulting to the old expressions for back-compat.
- **Weight mapping**: the shipped diffusion checkpoint often prefixes the encoder (e.g.
  `model.language_model.*`) and includes a vision tower / lm_head to drop; strip/drop and partial-load the
  needed shards. Reuse the encoder's `_prepare_torch_state` q/k/v fusion (mind GQA + decoupled head_dim).
- Large encoders (tens of GB) need MULTICHIP (won't fit one ASIC): use the encoder's parallel config
  (TP + optional FSDP) on the available mesh; swap Wormhole-specific compute configs for
  `init_device_compute_kernel_config(arch(), ...)`.

## Evidence To Leave

- Real-weight PCC >= 0.99 vs `<HFModel>.model(input_ids, output_hidden_states=True).hidden_states[k]` on the
  target mesh, plus a weight-free random-weight CI variant (pcc ~0.98). Skip-guard the reference.
- Record the mesh/parallel config, peak DRAM/ASIC (OOM headroom), the layer-tap verification, and the
  config-threading fix in `doc/<stage>/work_log.md`. If you edit a SHARED encoder, keep it back-compatible
  with the other model that uses it and say how back-compat was verified.
