#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Prefill goldens for openai/gpt-oss-120b, layer by layer (engine.py), from transformers' GptOss code.

Stored tensors (N = isl, i = layer; 36 layers: sliding_attention (window 128) and full_attention, alternating, sliding
first):

  hidden/embed.safetensors
    model.embed_tokens.output                  [1, N, 2880]       bf16  embeddings, no scale
  hidden/layer_XX.safetensors
    model.layers.{i}.output                    [1, N, 2880]       bf16  all   h + mlp(post_attn_norm(h)),
                                                                        h = x + attn(input_norm(x))
    model.layers.{i}.self_attn.output          [1, N, 2880]       bf16  all   --sublayers: attention (with sinks)
                                                                        after o_proj + bias, before the residual
    model.layers.{i}.mlp.output                [1, N, 2880]       bf16  all   --sublayers: routed MoE output (no
                                                                        shared expert), before the residual
  cache/layer_XX.safetensors
    model.layers.{i}.mlp.router.output         [N, 128]           fp32  all   router logits x @ W.T + b, computed in
                                                                        the compute dtype, stored fp32
    model.layers.{i}.mlp.experts.top_k_index   [N, 4]             int32 all   chosen experts, ascending expert id
    model.layers.{i}.mlp.experts.top_k_weights [N, 4]             fp32  all   softmax over the 4 chosen logits,
                                                                        same order as top_k_index
    past_key_values.layers.{i}.keys            [1, 8, N, 64]      bf16  all   k_proj + bias, then YaRN RoPE (half
                                                                        split, cos/sin x 1.3466); every position,
                                                                        also on sliding layers
    past_key_values.layers.{i}.values          [1, 8, N, 64]      bf16  all   v_proj + bias, no RoPE
  hidden/final.safetensors
    model.last_hidden_state                    [1, N, 2880]       bf16  final norm
    logits                                     [1, 1, 201088]     fp32  last position, untied lm_head, no softcap

The routed experts are MXFP4 in the checkpoint (``*_blocks`` / ``*_scales``). The golden is the dequantized model:
convert() turns them into bf16 with transformers' own convert_moe_packed_tensors, exactly what
from_pretrained(..., quantization_config=Mxfp4Config(dequantize=True)) loads, and the Loader casts that to the compute
dtype (bf16 -> fp32 is exact). Activations are never quantized.

Attention sinks (self_attn.sinks, one logit per query head, a checkpoint parameter: nothing to store) join each
softmax as an extra column that is then dropped. Only the eager attention applies them; transformers' SDPA path
ignores them silently, so this script runs eager only.

The cache is a DynamicCache built without the config, so sliding layers keep K/V for every position rather than
the last window; the masks come from config.sliding_window either way, so the forward is unchanged.

    python generate_golden_gpt_oss_120b.py --isl 10240
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from engine import LayerwiseGolden, Recipe  # noqa: E402

NOTES = {
    "model.embed_tokens.output": "Token embeddings, no scale.",
    "model.layers.{i}.output": (
        "Decoder layer output: h + mlp(post_attention_layernorm(h)) with h = x + self_attn(input_layernorm(x)). "
        "The next layer's input."
    ),
    "model.layers.{i}.self_attn.output": (
        "Attention after o_proj (+ bias), before the residual. Eager attention with sinks: the per-head "
        "self_attn.sinks logit (a checkpoint parameter, not stored) is appended to each row of the scaled, masked "
        "QK^T, the row max subtracted, softmax taken, and the sink column dropped before multiplying V. "
        "On sliding_attention layers the query at position s sees keys s - sliding_window + 1 .. s (128 of them)."
    ),
    "model.layers.{i}.mlp.output": (
        "Routed MoE output, before the residual: sum over the 4 chosen experts of top_k_weights * expert(x), where "
        "expert(x) = (clamp(up, -7, 7) + 1) * g * sigmoid(1.702 g) @ down + down_bias, g = clamp(gate, max=7), and "
        "gate / up are the even / odd columns (interleaved) of x @ gate_up_proj + gate_up_proj_bias. Expert weights "
        "are the MXFP4 checkpoint dequantized by transformers (convert_moe_packed_tensors). No shared expert."
    ),
    "model.layers.{i}.mlp.router.output": (
        "Router logits [N, num_local_experts]: linear with bias on post_attention_layernorm output, computed in the "
        "compute dtype and stored fp32 (no upcast in HF; storing fp32 only avoids a second rounding)."
    ),
    "model.layers.{i}.mlp.experts.top_k_index": (
        "Experts chosen per token: torch.topk of the router logits (num_experts_per_tok of them), sorted ascending "
        "by expert id."
    ),
    "model.layers.{i}.mlp.experts.top_k_weights": (
        "Weights of top_k_index, same order: softmax over the chosen experts' logits only (not over all experts), "
        "in the compute dtype; no extra scaling."
    ),
    "past_key_values.layers.{i}.keys": (
        "K [1, num_key_value_heads, N, head_dim] for every position: k_proj (+ bias) then RoPE, half-split rotation "
        "over all 64 dims, YaRN (theta 150000, factor 32, original 4096, beta_fast 32, beta_slow 1): cos and sin are "
        "multiplied by attention_scaling 0.1 ln(32) + 1 = 1.3466 and cast to the compute dtype. Same on both layer "
        "kinds; on sliding_attention layers the query at s only sees keys s - 127 .. s."
    ),
    "past_key_values.layers.{i}.values": "V [1, num_key_value_heads, N, head_dim]: v_proj (+ bias), no RoPE.",
    "logits": "Last position only: lm_head (not tied to embed_tokens, no bias), no softcapping, fp32.",
}


class GptOssGolden(LayerwiseGolden):
    hf_repo = "openai/gpt-oss-120b"
    # GptOss declares _supports_sdpa = False: transformers' sdpa_attention_forward drops s_aux (the sinks) without a
    # word, and flex attention refuses sinks on CPU. Eager is the reference implementation, so it is the only choice.
    attn, attn_choices = "eager", ("eager",)
    recipe = Recipe(
        modules=("mlp.router",),
        routing="mlp.experts",
        # Routing is a top-k: rounding the logits to bf16 on store could reorder near-ties, so they stay as computed.
        keep_fp32=("mlp.router.output", "mlp.experts.top_k_weights"),
        config=(
            "layer_types",
            "sliding_window",
            "head_dim",
            "num_attention_heads",
            "num_key_value_heads",
            "attention_bias",
            "rope_parameters",
            "rms_norm_eps",
            "num_local_experts",
            "num_experts_per_tok",
            "intermediate_size",
            "swiglu_limit",
            "tie_word_embeddings",
            "quantization_config",
        ),
        notes=NOTES,
    )

    def make_cache(self, text_config):
        from transformers import DynamicCache

        return DynamicCache()  # no config: sliding layers keep full-length K/V too

    def convert(self, sd, prefix, module):
        """MXFP4 experts -> bf16, by the function Mxfp4Dequantize (from_pretrained's dequantize=True path) calls.
        It also transposes to GptOssExperts' [E, in, out] layout; gate/up stay interleaved, biases pass through.
        A checkpoint without ``_blocks`` (the unquantized selftest one) is returned untouched."""
        from transformers.integrations.mxfp4 import convert_moe_packed_tensors

        for k in [k for k in sd if k.endswith("_blocks")]:
            name = k.removesuffix("_blocks")
            sd[name] = convert_moe_packed_tensors(sd.pop(k), sd.pop(f"{name}_scales"))
        return sd

    def tiny_config(self):
        from transformers import GptOssConfig

        layer_types = ["sliding_attention", "full_attention"] * 2
        return GptOssConfig(
            vocab_size=512,
            hidden_size=128,
            intermediate_size=96,
            num_hidden_layers=len(layer_types),
            layer_types=layer_types,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=32,
            num_local_experts=8,
            num_experts_per_tok=4,
            sliding_window=16,  # < selftest isl: the window really cuts
            rope_parameters=dict(
                rope_type="yarn",
                rope_theta=150000.0,
                factor=32.0,
                original_max_position_embeddings=4096,
                beta_fast=32.0,
                beta_slow=1.0,
                truncate=False,
            ),
            tie_word_embeddings=False,
        )


if __name__ == "__main__":
    sys.exit(GptOssGolden().main())
