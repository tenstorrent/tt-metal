#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Prefill goldens for meta-models/Muse-Glimmer-30B, layer by layer (engine.py), from transformers' muse_glimmer code
(text backbone only; the vision tower is never built into a stage nor read).

REQUIRES transformers >= 5.17 (muse_glimmer is not in 5.12, which tt-metal's python_env pins; validated with 5.19.0).
Run it through a uv overlay that adds a newer transformers on top of python_env, without touching it:

    uv run --no-project --python python_env/bin/python --with transformers==5.19.0 \\
        python models/common/hf_golden/generate_golden_muse_glimmer_30b.py --isl 10240

Stored tensors (N = isl, i = layer; 52 dense layers in a repeating [sliding, sliding, sliding, full] pattern:
sliding_attention (window 2048, RoPE theta 5e5) on i % 4 != 3, full_attention with NO RoPE on i % 4 == 3; GQA
32 query heads, 2 KV heads of 128):

  hidden/embed.safetensors
    model.embed_tokens.output                  [1, N, 6656]       bf16  embeddings AFTER the weightless RMSNorm the
                                                                        embedding module applies (embed_norm)
  hidden/layer_XX.safetensors
    model.layers.{i}.output                    [1, N, 6656]       bf16  all   decoder output: h + post_ffn_norm(
                                                                        mlp(pre_ffn_norm(h))), h = x + post_attn_norm(
                                                                        attn(input_norm(x)))
    model.layers.{i}.self_attn.output          [1, N, 6656]       bf16  all   --sublayers: attention after the output
                                                                        gate and o_proj, before post_attention_norm
    model.layers.{i}.self_attn.gate_proj.output [1, N, 4096]      bf16  all   --sublayers: output-gate logits (32 heads
                                                                        x 128), before the sigmoid
    model.layers.{i}.post_attention_layernorm.output [1, N, 6656] bf16  all   --sublayers: the attention branch as added
                                                                        to the residual
    model.layers.{i}.mlp.output                [1, N, 6656]       bf16  all   --sublayers: SwiGLU MLP output, before
                                                                        post_feedforward_layernorm
    model.layers.{i}.post_feedforward_layernorm.output [1, N, 6656] bf16 all  --sublayers: the MLP branch as added to
                                                                        the residual
  cache/layer_XX.safetensors
    past_key_values.layers.{i}.keys            [1, 2, N, 128]     bf16  all   k_proj -> weightless RMSNorm (qk_norm) ->
                                                                        RoPE on sliding layers only (half split, theta
                                                                        5e5); full layers: no RoPE. Every position,
                                                                        also on sliding layers (the device keeps the
                                                                        last 2048)
    past_key_values.layers.{i}.values          [1, 2, N, 128]     bf16  all   v_proj, no norm, no RoPE
  hidden/final.safetensors
    model.last_hidden_state                    [1, N, 6656]       bf16  final RMSNorm (with weight, not centered)
    logits                                     [1, 1, 202048]     fp32  last position: 20 * tanh(lm_head(h) *
                                                                        0.19612 / 20), untied lm_head

What is model-specific, and why it is handled here:

  * Attention (MuseGlimmerTextAttention): q and k go through ONE weightless RMSNorm (qk_norm, eps rms_norm_eps), then
    q is multiplied by qk_scale_factor (3.87); the softmax scale is head_dim ** -0.5 on top. RoPE is per layer:
    layer_rope_theta[i] == 0 marks a NoPE layer, which gets position_embeddings=None (layer_kwargs mirrors
    MuseGlimmerTextModel.forward); in this checkpoint those are exactly the full_attention layers. The output is
    gated elementwise: attn * sigmoid(gate_proj(x)) before o_proj. No sinks, so sdpa computes it.
  * Norms: Gemma-style sandwich, input / post_attention / pre_feedforward / post_feedforward per layer, all
    "centered" RMSNorm (x * rsqrt(mean(x^2) + eps) * (1 + w), zero-initialized w), post norms with eps 1e-8. The
    final norm is a plain RMSNorm with weight (x * rsqrt(.) * w). The embedding output is normed too, by a weightless
    RMSNorm, so model.embed_tokens.output is that normed embedding.
  * Logits: lm_head, then x output_multiplier (0.19612), then Gemma's softcap T * tanh(x / T) with T = 20
    (final_logit_softcapping). The engine's default applies only the softcap, so logits() does both.
  * Weights ship bf16, names already HF's (model.language_model.*, lm_head); the checkpoint's vision tower, vision
    adapter and projection are never read.

    python generate_golden_muse_glimmer_30b.py --isl 10240
    python generate_golden_muse_glimmer_30b.py --isl 56320 --attn-chunk 4096
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from engine import LayerwiseGolden, Recipe  # noqa: E402

MIN_TRANSFORMERS = (5, 17)
UV = "uv run --no-project --python python_env/bin/python --with transformers==5.19.0 python"

NOTES = {
    "model.embed_tokens.output": (
        "Token embeddings after the embedding module's weightless RMSNorm (embed_norm, eps rms_norm_eps): "
        "e * rsqrt(mean(e^2) + eps), no scale. The first layer's input."
    ),
    "model.layers.{i}.output": (
        "Decoder layer output: h + post_feedforward_layernorm(mlp(pre_feedforward_layernorm(h))), with h = x + "
        "post_attention_layernorm(self_attn(input_layernorm(x))). All four norms are centered RMSNorm (x * (1 + w)). "
        "The next layer's input."
    ),
    "model.layers.{i}.self_attn.output": (
        "Attention output after the elementwise output gate and o_proj: o_proj(attn * sigmoid(gate_proj(x))), "
        "before post_attention_layernorm (--sublayers)."
    ),
    "model.layers.{i}.self_attn.gate_proj.output": (
        "Output-gate logits gate_proj(x) [1, N, num_heads * head_dim], before the sigmoid; x is the "
        "input_layernorm output (--sublayers)."
    ),
    "model.layers.{i}.post_attention_layernorm.output": (
        "post_attention_layernorm(self_attn.output): the attention branch exactly as added to the residual "
        "(--sublayers)."
    ),
    "model.layers.{i}.mlp.output": "SwiGLU MLP output down(silu(gate(x)) * up(x)), before post_feedforward_layernorm (--sublayers).",
    "model.layers.{i}.post_feedforward_layernorm.output": (
        "post_feedforward_layernorm(mlp.output): the MLP branch exactly as added to the residual (--sublayers)."
    ),
    "past_key_values.layers.{i}.keys": (
        "Keys [1, num_key_value_heads, N, head_dim]: k_proj, then the weightless qk_norm RMSNorm, then RoPE (half "
        "split, rope_theta) on sliding_attention layers only; full_attention layers (layer_rope_theta 0) are NoPE. "
        "Every position; a sliding layer attends to the last sliding_window (2048) of them. q (not stored) gets the "
        "same norm, then x qk_scale_factor."
    ),
    "past_key_values.layers.{i}.values": "Values [1, num_key_value_heads, N, head_dim]: v_proj, no norm, no RoPE.",
    "model.last_hidden_state": "Final RMSNorm (with weight, not centered) of the last layer's output.",
    "logits": (
        "Last position only: final_logit_softcapping * tanh(lm_head(h) * output_multiplier / "
        "final_logit_softcapping), untied lm_head, fp32."
    ),
}


def check_transformers() -> None:
    import transformers

    version = tuple(int(p) for p in transformers.__version__.split(".")[:2])
    if version < MIN_TRANSFORMERS:
        raise SystemExit(
            f"transformers {transformers.__version__} has no muse_glimmer: this script needs >= "
            f"{'.'.join(map(str, MIN_TRANSFORMERS))}. Run it through the overlay (never upgrade python_env):\n"
            f"    {UV} {Path(__file__).name} ..."
        )


class MuseGlimmerGolden(LayerwiseGolden):
    hf_repo = "meta-models/Muse-Glimmer-30B"
    # The BOS token, as the tokenizer adds it by default. Wikitext ppl over the same 10100 tokens at isl 10240 (logits
    # with output_multiplier and the softcap, as the model computes them): 7.02 BOS (top-1 55.9%), 7.11 raw, 7.15
    # framed as its own reply by the chat template ("<|begin_of_text|><|start|>system<|message|>... <|start|>user
    # <|message|>Continue this WikiText-2 dump verbatim: ...<|eot|><|start|>assistant to=user<|message|>").
    prefix = "<|begin_of_text|>"
    recipe = Recipe(
        sublayers=(
            "self_attn",
            "self_attn.gate_proj",
            "post_attention_layernorm",
            "mlp",
            "post_feedforward_layernorm",
        ),
        config=(
            "layer_types",
            "layer_rope_theta",
            "rope_parameters",
            "sliding_window",
            "head_dim",
            "num_attention_heads",
            "num_key_value_heads",
            "qk_scale_factor",
            "rms_norm_eps",
            "post_norm_eps",
            "hidden_activation",
            "intermediate_size",
            "output_multiplier",
            "final_logit_softcapping",
            "tie_word_embeddings",
            "max_position_embeddings",
        ),
        notes=NOTES,
    )

    def __init__(self):
        # The engine builds through AutoModelForCausalLM, which has no muse_glimmer entry (the checkpoint is
        # image-text-to-text). Register the checkpoint's class, as the GLM-5.3-Flash script does.
        check_transformers()
        from transformers import AutoModelForCausalLM, MuseGlimmerConfig, MuseGlimmerForConditionalGeneration

        AutoModelForCausalLM.register(MuseGlimmerConfig, MuseGlimmerForConditionalGeneration, exist_ok=True)

    def make_cache(self, text_config):
        """No config: transformers' DynamicCache(config) gives sliding layers a DynamicSlidingWindowLayer that keeps
        only the last sliding_window - 1 (2047) positions. Built without it, every layer keeps K/V for every position,
        as the other goldens store them; the masks come from the config either way, so the forward is unchanged."""
        from transformers import DynamicCache

        return DynamicCache()

    def layer_kwargs(self, i, kind, state):
        """MuseGlimmerTextModel.forward: NoPE layers (layer_rope_theta[i] == 0) get position_embeddings=None."""
        kwargs = super().layer_kwargs(i, kind, state)
        if not self.rope_theta[i]:
            kwargs["position_embeddings"] = None
        return kwargs

    def prepare(self, backbone, text_config, hidden, position_ids):
        self.rope_theta = list(text_config.layer_rope_theta)
        return super().prepare(backbone, text_config, hidden, position_ids)

    def logits(self, text_config, lm_head, hidden):
        """MuseGlimmerForConditionalGeneration.forward: x output_multiplier, then the tanh softcap."""
        logits = lm_head(hidden[:, -1:]) * text_config.output_multiplier
        cap = text_config.final_logit_softcapping
        return torch.tanh(logits / cap) * cap

    def tiny_init(self, model):
        super().tiny_init(model)
        g = torch.Generator().manual_seed(2)
        for name, t in model.named_parameters():
            if name.endswith("layernorm.weight"):  # centered norms: w ~ 0 like the real ones, but not exactly
                t.copy_(torch.randn(t.shape, generator=g) * 0.1)
            elif name.endswith("norm.weight"):  # the final, plain RMSNorm: w ~ 1
                t.copy_(1 + torch.randn(t.shape, generator=g) * 0.1)

    def tiny_config(self):
        from transformers import MuseGlimmerConfig

        n = 8  # two [sliding x3, full] blocks: RoPE and NoPE layers, both mask kinds
        types = ["sliding_attention"] * 3 + ["full_attention"]
        return MuseGlimmerConfig(
            text_config=dict(
                vocab_size=512,
                hidden_size=64,
                intermediate_size=128,
                num_hidden_layers=n,
                num_attention_heads=4,
                num_key_value_heads=2,
                head_dim=16,
                layer_types=types * (n // 4),
                layer_rope_theta=[500000.0, 500000.0, 500000.0, 0] * (n // 4),
                sliding_window=32,  # < selftest isl: the window really cuts
                qk_scale_factor=3.87,
                output_multiplier=0.196,
                final_logit_softcapping=20.0,
                tie_word_embeddings=False,
                pad_token_id=None,
            ),
            vision_config=dict(num_hidden_layers=1, hidden_size=32, intermediate_size=32, num_attention_heads=2),
        )


if __name__ == "__main__":
    sys.exit(MuseGlimmerGolden().main())
