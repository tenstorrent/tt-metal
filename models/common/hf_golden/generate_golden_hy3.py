#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Prefill goldens for tencent/Hy3, layer by layer (engine.py), from transformers' hy_v3 code.

Stored tensors (N = isl, i = layer; 80 layers, all full causal GQA attention (64 query heads, 8 KV heads of 128,
q / k RMSNorm before RoPE); dense MLP on layer 0, MoE (192 routed experts, top 8, 1 shared) on 1-79):

  hidden/embed.safetensors
    model.embed_tokens.output                  [1, N, 4096]       bf16  embeddings, no scale
  hidden/layer_XX.safetensors
    model.layers.{i}.output                    [1, N, 4096]       bf16  all   h + mlp(post_attn_norm(h)),
                                                                        h = x + attn(input_norm(x))
    model.layers.{i}.self_attn.output          [1, N, 4096]       bf16  all   --sublayers: attention after o_proj,
                                                                        before the residual
    model.layers.{i}.mlp.output                [1, N, 4096]       bf16  all   --sublayers: dense MLP / routed +
                                                                        shared experts, before the residual
  cache/layer_XX.safetensors
    past_key_values.layers.{i}.keys            [1, 8, N, 128]     bf16  all   k_proj -> k_norm -> RoPE (half split,
                                                                        theta 11158840)
    past_key_values.layers.{i}.values          [1, 8, N, 128]     bf16  all   v_proj, no norm, no RoPE
    model.layers.{i}.mlp.gate.output           [N, 192]           fp32  1-79  router logits x @ W.T in fp32, before
                                                                        sigmoid / correction bias
    model.layers.{i}.mlp.experts.top_k_index   [N, 8]             int32 1-79  chosen experts, ascending expert id
    model.layers.{i}.mlp.experts.top_k_weights [N, 8]             fp32  1-79  their weights, same order:
                                                                        normalized sigmoid, x 2.826
  hidden/final.safetensors
    model.last_hidden_state                    [1, N, 4096]       bf16  final norm
    logits                                     [1, 1, 120832]     fp32  last position, untied lm_head

What is model-specific:

  * Routing (HYV3TopKRouter): logits in fp32, sigmoid, top-8 of sigmoid + e_score_correction_bias (fp32, no group
    stage), weights from the uncorrected sigmoid renormalized and x router_scaling_factor (2.826). The shared expert is
    added unweighted, in the compute dtype: the checkpoint's config sets enable_moe_fp32_combine false.
  * Checkpoint names -> HF names: transformers' own hy_v3 renamings (mlp.router.gate -> mlp.gate, mlp.expert_bias ->
    mlp.e_score_correction_bias, mlp.shared_mlp -> mlp.shared_experts); routed experts ship per expert and the Loader
    fuses them. The MTP layer (layers.80) is never built into a stage nor read.
  * The checkpoint's config also carries enable_lm_head_fp32 true and enable_attention_fp32_softmax false, which
    transformers' hy_v3 does not read: lm_head runs in the compute dtype (logits stored fp32), and every attention
    softmax is fp32 anyway (sdpa / eager).

    python generate_golden_hy3.py --isl 10240
    python generate_golden_hy3.py --isl 56320 --attn-chunk 4096
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from engine import LayerwiseGolden, Recipe  # noqa: E402

NOTES = {
    "model.embed_tokens.output": "Token embeddings (no scale).",
    "model.layers.{i}.output": "Decoder layer output, residual included: the next layer's input.",
    "model.layers.{i}.self_attn.output": "Attention output after o_proj, before the residual add (--sublayers).",
    "model.layers.{i}.mlp.output": (
        "Dense SwiGLU MLP (layer 0), or routed experts (weighted) + the shared expert, before the residual add "
        "(--sublayers)."
    ),
    "past_key_values.layers.{i}.keys": (
        "Keys [1, num_key_value_heads, N, head_dim]: k_proj, then k_norm (RMSNorm per head), then RoPE (half-split "
        "rotation, rope_theta 11158840, no scaling)."
    ),
    "past_key_values.layers.{i}.values": "Values [1, num_key_value_heads, N, head_dim]: v_proj, no norm, no RoPE.",
    "model.layers.{i}.mlp.gate.output": (
        "Router logits [N, num_experts] = x @ gate.weight.T in fp32, before the sigmoid. Selection: top-k of "
        "sigmoid(logits) + e_score_correction_bias (fp32)."
    ),
    "model.layers.{i}.mlp.experts.top_k_index": "The num_experts_per_tok chosen experts per token, ascending by expert id. int32.",
    "model.layers.{i}.mlp.experts.top_k_weights": (
        "Weights of top_k_index, same order: sigmoid(logits) of the chosen experts (no bias) divided by their sum + "
        "1e-20, times router_scaling_factor (2.826). fp32; the shared expert is added unweighted."
    ),
    "model.last_hidden_state": "Final RMSNorm of the last layer's output.",
    "logits": "Last position only: lm_head (not tied to embed_tokens) in the compute dtype, stored fp32.",
}


class Hy3Golden(LayerwiseGolden):
    hf_repo = "tencent/Hy3"
    # No prefix: wikitext ppl over the same 10100 tokens at isl 10240 is 5.61 raw (top-1 62.0%), 5.81 after the BOS
    # token alone, 5.88 framed as its own reply by the chat template (no_think, its default: "<｜hy_begin_of_sentence:
    # opensource｜><｜reasoning_mode:opensource｜>reasoning_effort:no_think<｜hy_User:opensource｜>Continue this
    # WikiText-2 dump verbatim: ...<｜hy_Assistant:opensource｜><think:opensource></think:opensource>").
    prefix = ""
    recipe = Recipe(
        modules=("mlp.gate",),
        routing="mlp.experts",
        # Routing is a top-k: rounding the logits can flip an expert at the cut, so they are never stored rounded.
        keep_fp32=("mlp.gate.output", "mlp.experts.top_k_weights"),
        config=(
            "mlp_layer_types",
            "head_dim",
            "num_attention_heads",
            "num_key_value_heads",
            "qk_norm",
            "rope_parameters",
            "rms_norm_eps",
            "hidden_act",
            "intermediate_size",
            "moe_intermediate_size",
            "num_experts",
            "num_shared_experts",
            "num_experts_per_tok",
            "router_scaling_factor",
            "enable_moe_fp32_combine",
            "enable_lm_head_fp32",
            "enable_attention_fp32_softmax",
            "tie_word_embeddings",
            "max_position_embeddings",
        ),
        notes=NOTES,
    )

    def layer_types(self, text_config) -> list[str]:
        """'full_attention/<mlp>': every layer attends the same way; the MLP kind decides what a layer stores."""
        return [f"full_attention/{m}" for m in text_config.mlp_layer_types[: text_config.num_hidden_layers]]

    def convert(self, sd, prefix, module):
        """Checkpoint names -> HF names with transformers' hy_v3 renamings. Routed experts stay per expert: the Loader
        fuses them."""
        from transformers.conversion_mapping import get_checkpoint_conversion_mapping
        from transformers.core_model_loading import WeightRenaming, rename_source_key

        renamings = [t for t in get_checkpoint_conversion_mapping("hy_v3") if isinstance(t, WeightRenaming)]
        return {rename_source_key(prefix + k, renamings, [])[0].removeprefix(prefix): v for k, v in sd.items()}

    def tiny_init(self, model):
        super().tiny_init(model)
        g = torch.Generator().manual_seed(2)
        for name, t in [*model.named_parameters(), *model.named_buffers()]:
            if name.endswith("e_score_correction_bias"):  # zero from from_config: randomize so loading it is tested
                t.copy_(torch.randn(t.shape, generator=g) * 0.1)
            elif name.endswith(("norm.weight", "layernorm.weight")):  # O(1) norms, like real ones
                t.copy_(1 + torch.randn(t.shape, generator=g) * 0.1)

    def tiny_config(self):
        from transformers import HYV3Config

        return HYV3Config(
            vocab_size=512,
            hidden_size=64,
            intermediate_size=128,
            moe_intermediate_size=32,
            num_hidden_layers=4,  # dense, then MoE
            num_attention_heads=8,
            num_key_value_heads=2,
            head_dim=16,
            num_experts=8,
            num_experts_per_tok=2,
            num_shared_experts=1,
            router_scaling_factor=2.826,
            enable_moe_fp32_combine=False,  # as the checkpoint
            rope_parameters={"rope_type": "default", "rope_theta": 11158840.0},
            tie_word_embeddings=False,
            pad_token_id=None,
        )


if __name__ == "__main__":
    sys.exit(Hy3Golden().main())
