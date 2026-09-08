# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Vendored HF/torch reference modeling for the Qwen3.6-27B DFlash drafter.

``dflash.py`` is a verbatim copy of ``github.com/z-lab/dflash`` (commit
``07ebd93db9f472af339b644bb70221ad8428328a``) ``dflash/model.py`` -- the upstream
reference implementation of DFlash ("Block Diffusion for Flash Speculative
Decoding", arXiv:2602.06036). It is generic: the same ``Qwen3DFlashAttention`` /
``DFlashDraftModel`` classes back the Gemma4 and Kimi-K2.6 DFlash checkpoints too,
with different config values.

Target/drafter pair here:
  - target (verifier): ``Qwen/Qwen3.6-27B``
  - drafter:           ``z-lab/Qwen3.6-27B-DFlash``

Drafter checkpoint facts (read from ``config.json`` and the ``model.safetensors``
header -- 58 tensors, 3.46 GB bf16 = 1.73 B params):
  architecture ``DFlashDraftModel`` (the plain variant -- NOT ``DFlash2DraftModel``,
  which adds a dynamic causal conv + learned candidate selector this checkpoint does
  not ship), ``model_type: qwen3``, hidden_size=5120 and intermediate_size=17408
  (both identical to the target's), num_hidden_layers=5, 32 Q / 8 KV heads
  (GQA 4:1), head_dim=128, rope_theta=1e7 with full rotary (no
  ``partial_rotary_factor``, unlike the target's 0.25), block_size=16,
  mask_token_id=248070, target_layer_ids=[1, 16, 31, 46, 61] (5 taps into the
  64-layer target), and NO ``final_logit_softcapping`` (unlike the Gemma4 drafter).

Two structural properties drive the TTNN port in ``tt/dflash/``:

1. **The drafter ships no ``embed_tokens`` and no ``lm_head``.** It borrows the
   target's: ``_raw_input_embeddings`` reads ``target.get_input_embeddings()`` and
   ``_output_head`` returns ``target.lm_head``. ``fc`` is ``[5120, 25600]`` --
   25600 = 5 x 5120 -- so it fuses the target's residual stream at the 5 tapped
   layers, EAGLE-3 style, rather than embedding tokens itself.

2. **One forward pass drafts the whole block; there is no iterative denoising
   loop.** ``DFlashDraftModel.forward`` takes Q from the 16 block positions only
   while K/V come from ``concat(target_hidden_ctx, block)``, so every masked slot
   resolves together. vLLM's ``DFlashProposer`` states the same invariant: "Only
   next_token_ids and mask tokens are query tokens, all other context is K/V".

Mask semantics matter and are asymmetric for this checkpoint. Our config supplies
no top-level ``is_causal``, so ``Qwen3DFlashAttention`` defaults to
``is_causal = (layer_type == "sliding_attention")``: the 4 sliding layers are
causal AND windowed (2048), while the single ``full_attention`` layer gets
``is_causal=False`` and ``sliding_window=None``, making ``_attention_mask`` return
no mask at all. The bidirectionality that makes this "block diffusion" therefore
lives entirely in layer 4. Do not substitute
``models/demos/deepseek_v3_d_p/reference/dflash_prefill/dflash.py`` for this file:
that is an older snapshot which hard-codes ``is_causal = False`` on every layer and
has no ``_attention_mask`` builder, which would silently make our sliding layers
bidirectional. ``tests/dflash/test_mask.py`` pins the distinction.

Note for anything that calls ``dflash_generate`` (the full draft/verify/accept
loop) rather than just ``DFlashDraftModel.forward``: it calls
``DynamicCache.activate_past_recording()`` unconditionally, which needs
``transformers`` 5.15.0, while this repo's ``python_env`` has 5.12.1 (the version
qwen36's own tests are validated against). Fixture capture only needs ``forward``.
"""
