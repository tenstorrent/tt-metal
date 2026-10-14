#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Prefill goldens for google/gemma-4-26B-A4B-it, layer by layer (engine.py), from transformers' Gemma4 code.

Stored tensors (N = isl, i = layer; 30 layers: 5 sliding_attention then 1 full_attention, repeated; every layer has
the dense MLP and the MoE block side by side, 128 experts, top 8):

  hidden/embed.safetensors
    model.embed_tokens.output                  [1, N, 2816]       bf16  embeddings x sqrt(2816) (scale cast to the
                                                                        compute dtype: 53.07 -> 53.0 in bf16)
  hidden/layer_XX.safetensors
    model.layers.{i}.output                    [1, N, 2816]       bf16  all   (h + post_ff_norm(post_ff_norm_1(
                                                                        mlp(pre_ff_norm(h))) + post_ff_norm_2(
                                                                        experts(pre_ff_norm_2(h))))) x layer_scalar,
                                                                        h = x + post_attn_norm(attn(in_norm(x)))
    model.layers.{i}.self_attn.o_proj.output   [1, N, 2816]       bf16  all   --sublayers: attention after o_proj,
                                                                        before post_attention_layernorm
    model.layers.{i}.mlp.output                [1, N, 2816]       bf16  all   --sublayers: dense GeGLU MLP output,
                                                                        before post_feedforward_layernorm_1
    model.layers.{i}.experts.output            [N, 2816]          bf16  all   --sublayers: routed experts' weighted
                                                                        sum, before post_feedforward_layernorm_2
  cache/layer_XX.safetensors
    past_key_values.layers.{i}.keys            [1, 8, N, 256]     bf16  sliding  post k_norm + RoPE theta 1e4, every
                                                                        position (the device keeps the last 1024)
                                               [1, 2, N, 512]     bf16  full     post k_norm + proportional RoPE theta
                                                                        1e6: only dims [0:64], [256:320] rotate
    past_key_values.layers.{i}.values          [1, 8, N, 256]     bf16  sliding  v_norm (RMSNorm, no scale), no RoPE
                                               [1, 2, N, 512]     bf16  full     no v_proj (attention_k_eq_v): v_norm of
                                                                        the k_proj output
    model.layers.{i}.router.proj.output        [N, 128]           fp32  all   router logits: proj of
                                                                        rms(h) x scale x 2816^-0.5, h the residual
    model.layers.{i}.router.output             [N, 128]           fp32  all   router probabilities, softmax of logits
    model.layers.{i}.experts.top_k_index       [N, 8]             int32 all   the 8 chosen experts, sorted by id
    model.layers.{i}.experts.top_k_weights     [N, 8]             fp32  all   their weights, in top_k_index order:
                                                                        probs renormalized over the 8, then times
                                                                        router.per_expert_scale (need not sum to 1)
  hidden/final.safetensors
    model.last_hidden_state                    [1, N, 2816]       bf16  final norm
    logits                                     [1, 1, 262144]     fp32  last position: tied lm_head, then
                                                                        tanh(x / 30) * 30 (final_logit_softcapping)

fp32 router / routing tensors are the compute dtype's values widened, not recomputed: under bf16 compute they hold
bf16 numbers. The checkpoint already stores experts fused as gate_up_proj [128, 1408, 2816] and down_proj
[128, 2816, 704], so no conversion. As for 31B, the cache is a DynamicCache built without the config, so sliding
layers keep K/V for every position; the masks come from config.sliding_window either way.

    python generate_golden_gemma_4_26b_a4b_it.py --isl 10240
"""

from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from generate_golden_gemma_4_31b_it import NOTES as DENSE_NOTES  # noqa: E402
from generate_golden_gemma_4_31b_it import Gemma4Golden  # noqa: E402

NOTES = {
    **DENSE_NOTES,
    "model.embed_tokens.output": (
        "Token embeddings times sqrt(hidden_size), the scale cast to the compute dtype (exact in fp32; bf16 rounds "
        "it, 53.07 -> 53.0 for hidden 2816)."
    ),
    "model.layers.{i}.output": (
        "Decoder layer output: h + post_ff_norm(post_ff_norm_1(mlp(pre_ff_norm(h))) + post_ff_norm_2(experts("
        "pre_ff_norm_2(h)))) with h = x + post_attn_norm(attn(in_norm(x))), then times the layer's layer_scalar. "
        "The dense MLP and the MoE both read h. The next layer's input."
    ),
    "model.layers.{i}.mlp.output": "Dense GeGLU MLP output (gelu_pytorch_tanh), before post_feedforward_layernorm_1.",
    "model.layers.{i}.experts.output": (
        "Routed experts' output, tokens flattened: sum over the top-k of weight x GeGLU expert(pre_ff_norm_2(h)), "
        "before post_feedforward_layernorm_2."
    ),
    "model.layers.{i}.router.proj.output": (
        "Router logits, tokens flattened: proj(rms_norm(h) * router.scale * hidden_size^-0.5), rms_norm without a "
        "learned scale, h the residual after attention (not pre_feedforward_layernorm'd). Stored fp32."
    ),
    "model.layers.{i}.router.output": "Router probabilities: softmax of the router logits over all experts. Stored fp32.",
    "model.layers.{i}.experts.top_k_index": (
        "The top_k_experts experts each token is routed to (top-k of the probabilities), sorted by expert id, int32."
    ),
    "model.layers.{i}.experts.top_k_weights": (
        "The chosen experts' weights in top_k_index order: probabilities renormalized to sum 1 over the top-k, then "
        "times router.per_expert_scale of that expert (so they need not sum to 1). Stored fp32."
    ),
}


class Gemma4MoeGolden(Gemma4Golden):
    hf_repo = "google/gemma-4-26B-A4B-it"
    # Experts are called as experts(hidden, top_k_index, top_k_weights), which is what the engine's routing hook
    # reads. The router returns (probabilities, top_k_weights, top_k_index): its hook keeps the probabilities.
    recipe = dataclasses.replace(
        Gemma4Golden.recipe,
        modules=("router", "router.proj"),
        sublayers=("self_attn.o_proj", "mlp", "experts"),
        routing="experts",
        # A top-k: bf16 rounding of the stored scores alone could flip an expert at the cut.
        keep_fp32=("router.output", "router.proj.output", "experts.top_k_weights"),
        config=(
            *Gemma4Golden.recipe.config,
            "enable_moe_block",
            "num_experts",
            "top_k_experts",
            "moe_intermediate_size",
        ),
        notes=NOTES,
    )

    def tiny_config(self):
        config = super().tiny_config()
        text = config.text_config
        text.enable_moe_block = True
        text.num_experts = 8
        text.top_k_experts = 2
        text.moe_intermediate_size = 32
        return config


if __name__ == "__main__":
    sys.exit(Gemma4MoeGolden().main())
