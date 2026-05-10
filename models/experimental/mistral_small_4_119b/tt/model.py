# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Full multimodal E2E model (vision + text) for ``mistralai/Mistral-Small-4-119B-2603``.

**Reuse today**: import ``TtMistralSmall4VisionStack`` from ``.vision`` (alias of
``TtMistralVisionTransformer`` under ``tt_transformers`` multimodal ``mistral_24b``).

**Text stack progress (TTNN)** — see ``tt/text_backbone.py``, ``tt/mistral4_self_attention.py``,
and ``tt/mistral4_embed_tokens.py``:

- Done: **``embed_tokens``** host lookup + TTNN activations (:class:`TtMistral4EmbedTokensPrefill`;
  PCC ``tests/test_text_embed_tokens_prefill_pcc.py``). Table stays on host (~1 GiB bf16); device
  embedding / paging still open.
- Done: **embed → 1–2 decoder layers** on TTNN (:class:`TtMistral4DecoderSequence`; PCC
  ``tests/test_text_prefill_embed_decoder_stack_pcc.py`` for host-upload vs on-mesh RoPE table).
  Default stack uses **host** ``Mistral4MoE`` per layer (host RAM scales with ``N``); optional
  ``use_ttnn_moe=True`` runs routed + shared experts on TTNN (no host MoE weights).
- Done: **final norm + ``lm_head``** on host torch (:class:`TtMistral4FinalNormLmHeadPrefill`; PCC
  ``tests/test_text_output_head_prefill_pcc.py``); device logits still open.
- Done: **E2E prefill logits** ``embed → N layers → norm → lm_head`` (:class:`TtMistral4TextPrefillLogits`;
  PCC ``tests/test_text_prefill_e2e_logits_pcc.py``, ``N`` in ``{1, 2}``; logits PCC bar is lower for
  ``N=2`` because ``lm_head`` amplifies accumulated TTNN/bf16 drift on the full vocab tensor).
- Done: **E2E decode logits** (one new token after prefill, ``N`` in ``{1, 2}`` + :meth:`TtMistral4TextPrefillLogits.make_stack_kv_state`;
  PCC ``tests/test_text_prefill_e2e_decode_logits_pcc.py``; logits PCC bar lower for ``N=2``, same rationale as prefill E2E).
- Done (layer 0, prefill): input / post-attention RMSNorm stubs; MLA self-attention
  prefill on TTNN (RoPE, Llama-4 Q scaling, SDPA, ``o_proj``); cos/sin still from host rotary;
  :class:`TtMistral4DecoderLayerAttnPrefillBlock` (``layer_idx``) chains norm → attn → residual → post norm
  (PCC vs HF in ``tests/test_text_decoder_layer0_attn_prefill_block_pcc.py``).
  :class:`TtMistral4DecoderLayer` adds MoE + final residual: default **host** ``Mistral4MoE`` (PCC
  ``tests/test_text_decoder_layer0_full_layer_pcc.py``) or ``use_ttnn_moe=True`` for **TTNN** routed
  + shared MLP (hub FP8 MoE weights are resolved like HF via
  :func:`~models.experimental.mistral_small_4_119b.tt.mistral4_moe_mesh.mistral4_mlp_state_dict_bf16_match_hf`
  before ``ttnn.from_torch``).   With ``moe_hf_torch_routing=True``, PCC vs HF is
  ``tests/test_text_decoder_layer0_full_layer_ttnn_moe_pcc.py`` (HF ``gate`` / ``topk`` on host, expert matmuls on device).
  With ``moe_hf_torch_routing=False`` (demo MoE path: device ``topk``), see the same file's
  ``test_mistral_small_4_text_decoder_layer_full_ttnn_moe_device_routing_pcc_vs_hf`` (lower PCC bar than HF-routing).
  Random-init routed PCC stays strict in ``tests/test_mistral4_moe_mesh_routed_pcc.py``.
  Layer-0-only names remain as aliases.
- Next (still required for **no host fallback** end-to-end on e.g. four P150):

  1. **MLA attention**: :class:`~models.experimental.mistral_small_4_119b.tt.text_backbone.TtMistral4DecoderSequence`
     supports either one cos/sin upload per forward (``use_device_rotary_embedding_table=False``) or a
     bounded on-mesh RoPE table + ``ttnn.embedding`` (``True``; default in
     :class:`~models.experimental.mistral_small_4_119b.tt.mistral4_text_prefill.TtMistral4TextPrefillLogits`).
     Done (layer 0 bring-up): ``forward_prefill_with_kv`` / ``forward_decode_extend_kv`` on
     :class:`~models.experimental.mistral_small_4_119b.tt.mistral4_self_attention.TtMistral4SelfAttentionPrefill`
     wired through :class:`~models.experimental.mistral_small_4_119b.tt.text_backbone.TtMistral4DecoderLayerAttnPrefillBlock`
     / :class:`~models.experimental.mistral_small_4_119b.tt.text_backbone.TtMistral4DecoderSequence` with
     :class:`~models.experimental.mistral_small_4_119b.tt.mistral4_kv_state.Mistral4DecoderStackKvState` (PCC
     ``tests/test_text_decoder_layer0_self_attn_decode_kv_pcc.py``,
     ``tests/test_text_decoder_stack_decode_kv_pcc.py``).
     ``paged_update_cache`` / ``scaled_dot_product_attention_decode`` and optional causal SDPA when
     ``attention_mask`` is wired remain open.
  2. **MoE**: :class:`~models.experimental.mistral_small_4_119b.tt.mistral4_moe_mesh.TtMistral4MoeRoutedExpertParallelSkeleton`
     (device ``ttnn.linear`` router, ``ttnn.topk`` routing for ``n_group=1`` / ``topk_group=1``,
     ``ttnn`` expert matmuls, **replicated** expert tables on mesh, no combine reduce while replicated)
     + :class:`~models.experimental.mistral_small_4_119b.tt.mistral4_moe_mesh.TtMistral4SharedExpertsMlpTtnn``
     for HF ``shared_experts``. Routed PCC: ``tests/test_mistral4_moe_mesh_routed_pcc.py``. Integrated
     into :class:`~models.experimental.mistral_small_4_119b.tt.text_backbone.TtMistral4DecoderLayer` /
     :class:`~models.experimental.mistral_small_4_119b.tt.text_backbone.TtMistral4DecoderSequence` via
     ``use_ttnn_moe=True`` (``TT_CCL`` auto-built on multi-device when omitted); optional
     ``moe_hf_torch_routing`` for HF-identical routing during PCC bring-up. Tests:
     ``tests/test_text_decoder_layer0_full_layer_ttnn_moe_pcc.py``. **Sharding:** routed ``gate_up`` /
     ``down`` shard on expert dim when ``n_routed_experts % num_mesh_devices == 0`` (override via
     ``moe_shard_routed_experts``); combine uses ``tt_all_reduce`` (see ``constants.expert_index_ranges_per_mesh_device``).
  3. **Stack**: replicate per-layer builders for 36 layers; final RMSNorm; tied or untied
     ``lm_head`` (131072 × 4096) and ``embed_tokens`` placement / paging for memory.      **Smoke** (opt-in):
     ``MISTRAL4_E2E_36L_TTNN_MOE=1`` → ``tests/test_text_prefill_e2e_36l_ttnn_moe_smoke.py`` (``TtMistral4TextPrefillLogits``,
     ``use_ttnn_moe=True``, ``moe_hf_torch_routing=False``, short ``seq_len``). Depth: ``MISTRAL4_E2E_DEEP_N_LAYERS``
     (default 36); optional logits PCC for ``N in {1,2}``: ``MISTRAL4_E2E_DEEP_PCC=1`` (HF routing on TTNN MoE;
     asserts vs **host-MoE** ``TtMistral4TextPrefillLogits``, same floors as ``test_text_prefill_e2e_logits_pcc``).
     On **P150×4** DRAM OOM during MoE upload, lower depth or add compressed weights / staging later.
  4. **Multimodal merge**: image token positions + ``prepare_inputs_prefill`` parity with HF
     (``ModelArgs`` / tokenizer special tokens).
  5. **Generator**: wire into ``tt_transformers`` generator patterns once (1–4) hold.

**Checkpoint / host notes**: full HF ``from_pretrained`` for FP8 may need CUDA/XPU; on
CPU-only TT boxes use filtered safetensors + TTNN PCC tests (see
``tests/test_text_decoder_layer0_hf_real_weights_forward.py``).

Until the **full 36-layer** TTNN stack + generator are production-ready, do not subclass
``models.tt_transformers.tt.model.Transformer`` for this checkpoint: ``TransformerBlock`` assumes
dense MLP or Mixtral-style MoE only.

Model card: https://huggingface.co/mistralai/Mistral-Small-4-119B-2603
"""


def get_vision_stack_class():
    """Return the TTNN vision+MMP class reused for Small 4."""
    from models.experimental.mistral_small_4_119b.tt.vision import TtMistralSmall4VisionStack

    return TtMistralSmall4VisionStack


def create_full_model_unsupported_reason() -> str:
    return (
        "Mistral Small 4 119B text stack (Mistral4 MLA attention + 128-way MoE) is not "
        "implemented in TTNN yet. Use get_vision_stack_class() for the Pixtral+MMP path."
    )
