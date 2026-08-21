# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Mistral-Small-4-119B decoder-*layer* (block) CPU reference.

Mistral has no runnable HF reference model wired (see the adapter's ``reference_*`` comments), so it
validates the way GLM-5.1 does: by *composing* the CPU references this repo already owns. This module
assembles one full decoder layer exactly as ``TtPrefillBlock.forward`` does:

    attn_norm_out = rms_norm(x, attn_norm_weight)
    mla_out       = MLAReference(config, mla_weights).forward(attn_norm_out)   # dense MLA, no indexer
    x             = x + mla_out
    ffn_norm_out  = rms_norm(x, ffn_norm_weight)
    ffn_out       = dense_ffn(ffn_norm_out)  OR  mistral4_moe_reference(ffn_norm_out)
    out           = x + ffn_out

The MLA half reuses ``reference.mla_reference.create_mla_reference`` verbatim — the same truth that
already validates Mistral dense MLA in ``test_mla.test_mistral4_mla`` — driven by the config from
``mistral4_hf_config()``. ``rms_norm`` / ``dense_ffn`` are the GLM ones (identical math; a private
copy would be a second thing to keep in sync). The FFN-MoE half is the new
``mistral_small_4.moe.mistral4_moe_reference``, because Mistral's plain-softmax top-4 router cannot be
expressed by any existing gate in this repo. Nothing here re-implements MLA or router math.

Note that ``first_k_dense_replace = 0`` for this model: every one of its 36 layers is MoE. The
``ffn_weights`` (dense) path is kept only so the reference matches the GLM block-reference contract
and can serve a shrunken/synthetic dense layer.

KNOWN GAP — ``apply_llama4_attn_scale`` (default **False**)
----------------------------------------------------------
The real model multiplies its queries by a position-dependent scale
``1 + llama_4_scaling_beta * log(1 + floor(pos / original_max_position_embeddings))``
(``modeling_mistral4.get_llama_4_attn_scale``, applied in ``Mistral4Attention.forward``). **ttMLA has
no equivalent**, so the *device* never applies it. The flag therefore defaults to OFF: with it off the
reference matches what the device actually computes, which is what a block/transformer PCC test must
compare against. Turning it ON demonstrates the gap instead of hiding it.

Two traps this implementation is careful about:

* HF reads ``rope_parameters["original_max_position_embeddings"]`` — **8192** — even though the
  *parameter* of ``get_llama_4_attn_scale`` is named ``max_position_embeddings``. This model's
  ``config.max_position_embeddings`` is 1048576; reading that one makes the scale 1.0 everywhere and
  the bug invisible.
* ``Mistral4Small119BConfig.LLAMA4_SCALING_BETA`` (0.1) is recorded on the constants class but is
  *not* placed into the namespace ``mistral4_hf_config()`` returns, so a consumer reading only that
  namespace cannot see it. ``llama4_attn_scale_params`` looks in ``config.rope_scaling`` first
  (present if the config ever grows the key) and falls back to the constant.

The scale is exactly 1.0 for every position below 8192 — so it is inert for the ``<= 8192`` sequences
these tests run — and then steps at each 8192-token boundary: 1.0693 at pos 8192, 1.1099 at 16384,
1.2197 at 65536. Sequences longer than 8192 are where reference and device genuinely diverge.
"""

import torch
from transformers import DynamicCache

from models.common.utility_functions import hf_cache_layer_kv
from models.demos.deepseek_v3_d_p.reference.glm_5_1.block import dense_ffn, rms_norm
from models.demos.deepseek_v3_d_p.reference.mistral_small_4.moe import mistral4_moe_reference
from models.demos.deepseek_v3_d_p.reference.mistral_small_4_119b_config import Mistral4Small119BConfig
from models.demos.deepseek_v3_d_p.reference.mla_reference import create_mla_reference


def llama4_attn_scale(
    position_ids: torch.Tensor,
    beta: float,
    original_max_position_embeddings: int,
) -> torch.Tensor:
    """``modeling_mistral4.get_llama_4_attn_scale``, verbatim. ``position_ids`` [b, seq] -> [b, 1, seq, 1]."""
    scaling = 1 + beta * torch.log(1 + torch.floor(position_ids / original_max_position_embeddings))
    return scaling[:, None, :, None]


def llama4_attn_scale_params(config) -> tuple[float, int]:
    """``(beta, original_max_position_embeddings)`` for ``config``, with the two traps handled.

    ``original_max_position_embeddings`` comes from ``config.rope_scaling`` (8192), **never** from
    ``config.max_position_embeddings`` (1048576). ``beta`` comes from ``rope_scaling`` if present,
    else from ``Mistral4Small119BConfig.LLAMA4_SCALING_BETA``.
    """
    rope_scaling = getattr(config, "rope_scaling", None) or {}
    original_max = rope_scaling.get(
        "original_max_position_embeddings", Mistral4Small119BConfig.ROPE_SCALING_ORIGINAL_MAX_POSITION_EMBEDDINGS
    )
    beta = rope_scaling.get("llama_4_scaling_beta", Mistral4Small119BConfig.LLAMA4_SCALING_BETA)
    return float(beta), int(original_max)


def _register_llama4_q_scale_hook(mla_ref, config, position_ids: torch.Tensor):
    """Multiply the Q projection's output by the llama-4 position scale; returns the hook handle.

    Hooking the Q projection is exact rather than approximate: HF applies the scale to
    ``query_states = cat(q_pass, q_rot)``, and everything ``MLAReference`` does between the Q
    projection and ``query_states`` is linear in q — the nope/rope split, the absorbed
    ``q_nope @ kv_b1``, and the RoPE rotation. Scaling q by ``s(pos)`` up front therefore scales
    ``query_states`` by exactly ``s(pos)``. It cannot be folded into ``softmax_scale``, which is a
    single scalar, because ``s`` varies per query position.
    """
    attn = mla_ref.attention
    q_proj = attn.q_b_proj if attn.q_lora_rank is not None else attn.q_proj
    beta, original_max = llama4_attn_scale_params(config)
    # [b, 1, seq, 1] -> [b, seq, 1], broadcasting against the projection's [b, seq, heads*q_head_dim].
    scale = llama4_attn_scale(position_ids.float(), beta, original_max)[:, 0, :, :]

    def _hook(_module, _args, output):
        # fp32 multiply then cast back: the reference module runs in bf16, and this flag is a
        # demonstration path, so do not compound its own rounding onto the gap it is measuring.
        return (output.float() * scale).to(output.dtype)

    return q_proj.register_forward_hook(_hook)


def mistral4_decoder_layer_reference(
    config,
    mla_weights,
    attn_norm_weight: torch.Tensor,
    ffn_norm_weight: torch.Tensor,
    hidden_states: torch.Tensor,
    seq_len: int,
    *,
    ffn_weights: dict | None = None,
    moe_weights: dict | None = None,
    position_ids: torch.Tensor | None = None,
    apply_llama4_attn_scale: bool = False,
):
    """One Mistral-Small-4 decoder layer on CPU (dense MLA + norm/residual + FFN), matching TtPrefillBlock.forward.

    Args:
        config: Mistral HF-attribute config (``mistral4_hf_config()``). Not mutated: the reference's
            YaRN table is sized from the forward's own kv_seq_len, so ``config.max_seq_len`` (a
            device-path field the reference never reads) is left alone.
        mla_weights: flat ``self_attn`` weight dict (``q_a_proj.weight``, ``q_a_layernorm.weight``,
            ``q_b_proj.weight``, ``kv_a_proj_with_mqa.weight``, ``kv_a_layernorm.weight``,
            ``kv_b_proj.weight``, ``o_proj.weight``) — the same dict fed to ttMLA and to
            ``test_mla``'s ``create_mla_reference``.
        attn_norm_weight / ffn_norm_weight: the two RMSNorm gains [hidden].
        hidden_states: block input [1, seq, hidden] (pre-attn-norm).
        seq_len: sequence length.
        ffn_weights: dense-layer FFN weights ``{"gate_proj","up_proj","down_proj"}``, OR
        moe_weights: MoE-layer weights ``{"gate_weights","routed_expert_weights","shared_expert_weights"}``.
            Exactly one must be given. The MoE uses Mistral's own routing (128 routed experts, top-4,
            softmax scoring with NO correction bias, norm_topk_prob, routed_scaling_factor 1.0,
            n_group = topk_group = 1 so no group-limited routing) plus one shared expert that is added.
        position_ids: [1, seq] absolute positions; defaults to ``arange(seq_len)``. Passed explicitly
            because the vendored ``apply_rotary_pos_emb`` indexes ``cos[position_ids]`` and would
            mis-index on ``None``.
        apply_llama4_attn_scale: apply the real model's position-dependent Q scale. Default **False**
            so the reference matches the device, which has no equivalent — see the module docstring.

    Returns:
        ``(output [1, seq, hidden], kvpe_cache [1, 1, seq, kv_lora_rank + qk_rope_head_dim])`` — kvpe in
        the same layout ``test_mla`` compares against (``hf_cache_layer_kv(cache, 0)[0]``).
    """
    if (ffn_weights is None) == (moe_weights is None):
        raise ValueError("provide exactly one of ffn_weights (dense) or moe_weights (MoE)")

    if hidden_states.shape[1] != seq_len:
        raise ValueError(f"hidden_states has {hidden_states.shape[1]} tokens, seq_len says {seq_len}")
    if position_ids is None:
        position_ids = torch.arange(seq_len, dtype=torch.long).unsqueeze(0)

    x = hidden_states
    attn_norm_out = rms_norm(x, attn_norm_weight, config.rms_norm_eps)

    ref = create_mla_reference(
        config=config,
        state_dict={f"model.layers.0.self_attn.{k}": v for k, v in mla_weights.items()},
        layer_idx=0,
        module_path="model.layers.0.self_attn",
    )
    ref = ref.eval().to(torch.bfloat16)
    handle = _register_llama4_q_scale_hook(ref, config, position_ids) if apply_llama4_attn_scale else None
    try:
        cache = DynamicCache()
        with torch.no_grad():
            mla_out, _, cache = ref(
                hidden_states=attn_norm_out,
                position_ids=position_ids,
                past_key_value=cache,
                use_cache=True,
            )
    finally:
        if handle is not None:
            handle.remove()
    kvpe_cache = hf_cache_layer_kv(cache, 0)[0]
    x = x + mla_out

    ffn_norm_out = rms_norm(x, ffn_norm_weight, config.rms_norm_eps)
    if ffn_weights is not None:
        ffn_out = dense_ffn(ffn_norm_out, ffn_weights["gate_proj"], ffn_weights["up_proj"], ffn_weights["down_proj"])
    else:
        ffn_out = mistral4_moe_reference(
            ffn_norm_out,
            gate_weights=moe_weights["gate_weights"],
            routed_expert_weights=moe_weights["routed_expert_weights"],
            shared_expert_weights=moe_weights["shared_expert_weights"],
            emb_dim=config.hidden_size,
            num_experts_per_tok=config.num_experts_per_tok,
            n_group=config.n_group,
            topk_group=config.topk_group,
            norm_topk_prob=config.norm_topk_prob,
            routed_scaling_factor=config.routed_scaling_factor,
        )

    return x + ffn_out, kvpe_cache
