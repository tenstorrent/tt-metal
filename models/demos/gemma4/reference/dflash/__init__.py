# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Vendored HF/torch reference modeling for the Gemma4-31B DFlash drafter.

``dflash.py`` is a verbatim copy of ``github.com/z-lab/dflash`` (commit
``07ebd93db9f472af339b644bb70221ad8428328a``) ``dflash/model.py`` — the
upstream reference implementation of DFlash ("Block Diffusion for Flash
Speculative Decoding", arXiv:2602.06036). It is generic: the same
``Qwen3DFlashAttention`` / ``DFlashDraftModel`` classes are also used for the
Kimi-K2.6-DFlash checkpoint (see
``models/demos/deepseek_v3_d_p/reference/dflash_prefill/``) with different
config values.

Target/drafter pair for this branch (``ign/gemma4_31B_MTP_Dflash``):
  - target (verifier): ``google/gemma-4-31B-it``
  - drafter:           ``z-lab/gemma-4-31B-it-DFlash``

Drafter checkpoint config (``z-lab/gemma-4-31B-it-DFlash/config.json``):
  architecture ``DFlashDraftModel`` (the plain variant -- NOT ``DFlash2DraftModel``,
  which adds a dynamic causal-conv + learned candidate-selector this checkpoint
  does not ship), ``model_type: qwen3`` (the drafter's own 5 decoder layers reuse
  Qwen3-style blocks regardless of the target's own architecture -- same as Kimi),
  hidden_size=5376, num_hidden_layers=5, num_attention_heads=64,
  num_key_value_heads=8, head_dim=128, block_size=16 (the speculative block --
  4x Qwen3.6 MTP's usual K=4), mask_token_id=4, target_layer_ids=[1,12,23,35,46,57]
  (6 taps into the 60-layer target), final_logit_softcapping=30.0 (a
  Gemma-specific tanh softcap applied in ``compute_logits`` -- absent from the
  Kimi checkpoint), and 4 sliding-attention layers + 1 full-attention layer
  (Gemma-style local/global interleaving inside the drafter itself, handled by
  ``Qwen3DFlashAttention``'s ``_attention_mask`` -- also absent from Kimi's config).

Unlike Qwen3.6's MTP (chains ONE token at a time, K sequential forward passes)
or Kimi's DFlash usage, this drafter denoises a whole ``block_size``-token block
in ONE forward pass (see ``dflash_generate`` in ``dflash.py``): every masked
draft position attends to both the target's context (via ``target_hidden``,
fed through ``fc`` from the 6 tapped layers) and every other masked position in
the same block, so all 16 positions resolve together rather than sequentially.
"""
