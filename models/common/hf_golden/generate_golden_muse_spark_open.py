#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Prefill goldens for meta-models/Muse-Spark-Open-816B-A42B, layer by layer (engine.py), from transformers'
muse_spark code (text backbone only; the vision tower, vision adapter and audio encoder are never built into a stage
nor read).

REQUIRES transformers >= 5.20 (muse_spark is not in 5.12, which tt-metal's python_env pins; validated with 5.20.2).
Run it through a uv overlay that adds a newer transformers on top of python_env, without touching it:

    uv run --no-project --python python_env/bin/python --with transformers==5.20.2 \\
        python models/common/hf_golden/generate_golden_muse_spark_open.py --isl 56320 --attn-chunk 1024

Stored tensors (N = isl, i = layer; 62 layers, every one MoE, in a repeating [sliding, sliding, sliding, full]
pattern counted back from the last layer: sliding_attention (window 2048, RoPE theta 5e5) where (61 - i) % 4 != 0,
full_attention with NO RoPE where (61 - i) % 4 == 0, i.e. i = 1, 5, ..., 61; GQA 128 query heads, 16 KV heads of 64):

  hidden/embed.safetensors
    model.embed_tokens.output                       [1, N, 8192]   bf16  embeddings AFTER the weightless RMSNorm the
                                                                         embedding module applies (embed_norm)
  hidden/layer_XX.safetensors
    model.layers.{i}.output                         [1, N, 8192]   bf16  the residual stream after the layer
    model.layers.{i}.self_attn.output               [1, N, 8192]   bf16  --sublayers: attention after o_norm, the
                                                                         output gate and o_proj
    model.layers.{i}.self_attn.gate_proj.output     [1, N, 8192]   bf16  --sublayers: output-gate logits (128 heads
                                                                         x 64), before the sigmoid
    model.layers.{i}.post_attention_residual_gate.output
                                                    [1, N, 8192]   bf16  --sublayers: the residual stream between
                                                                         attention and MoE
    model.layers.{i}.mlp.pre_expert_norm.output     [N, 4096]      bf16  --sublayers: the experts' input (pooled to
                                                                         moe_hidden_size, centered RMSNorm)
    model.layers.{i}.mlp.experts.output             [N, 4096]      bf16  --sublayers: the routed experts' weighted sum,
                                                                         before post_expert_proj
    model.layers.{i}.mlp.output                     [1, N, 8192]   bf16  --sublayers: MoE output, before
                                                                         post_feedforward_layernorm
  cache/layer_XX.safetensors
    past_key_values.layers.{i}.keys                 [1, 16, N, 64] bf16  all   k_proj -> weightless RMSNorm (qk_norm)
                                                                         -> RoPE on sliding layers only (half split,
                                                                         theta 5e5); full layers: no RoPE. Every
                                                                         position, also on sliding layers (the device
                                                                         keeps the last 2048)
    past_key_values.layers.{i}.values               [1, 16, N, 64] bf16  all   v_proj, no norm, no RoPE
    model.layers.{i}.mlp.gate.output                [N, 256]       fp32  all   router logits x output_multiplier, before
                                                                         the sigmoid
    model.layers.{i}.mlp.experts.top_k_index        [N, 8]         int32 all   chosen experts, ascending
    model.layers.{i}.mlp.experts.top_k_weights      [N, 8]         fp32  all   their weights, same order
  hidden/final.safetensors
    model.last_hidden_state                         [1, N, 8192]   bf16  final RMSNorm (with weight, not centered)
    logits                                          [1, 1, 202048] fp32  last position: 20 * tanh(lm_head(h) *
                                                                         0.17678 / 20), untied lm_head

What is model-specific, and why it is handled here (the attention, embedding and logits are Muse-Glimmer's, see
generate_golden_muse_glimmer_30b.py; the rest is new in Spark):

  * Attention (MuseSparkTextAttention): q and k go through ONE weightless RMSNorm (qk_norm), then q is multiplied by
    qk_scale_factor (5.473); the softmax scale is head_dim ** -0.5 on top. RoPE is per layer: layer_rope_theta[i] == 0
    marks a NoPE layer, which gets position_embeddings=None (layer_kwargs mirrors MuseSparkTextModel.forward); in
    this checkpoint those are exactly the full_attention layers. The attention output goes through a weightless
    RMSNorm (o_norm) and is then gated elementwise: o_norm(attn) * sigmoid(gate_proj(x)) before o_proj. No sinks, so
    sdpa computes it.
  * Residual gates instead of plain adds, after attention and after the MoE: (alpha * residual + beta * branch) with
    beta = sigmoid(gate / 0.3), alpha = sqrt(max(sigmoid(-gate / 0.3) * (1 + beta), 1e-3)), per channel, fp32 (the
    gates are kept fp32, as from_pretrained keeps them: _keep_in_fp32_modules_strict).
  * Norms, three kinds: input / pre_feedforward / pre_expert are centered RMSNorm (x * rsqrt(.) * (1 + w));
    post_attention is a weightless RMSNorm (eps 1e-8); post_feedforward and each expert's post_expert_norm scale
    first, round to bf16, then normalize (eps 1e-8). The final norm is a plain RMSNorm with weight.
  * MoE (all layers): router = fp32 GEMM of the bf16 hidden with the fp32 weight, x output_multiplier, sigmoid;
    top-8 of sigmoid + e_score_correction_bias; weights renormalized (+1e-15) x routed_scaling_factor. The 256
    routed experts run in a 4096 latent: pre_expert_proj, pre_expert_norm, SwiGLU experts each followed by
    post_expert_norm and its routing weight, summed, post_expert_proj. No shared experts. Experts ship fused
    (gate_up_proj [256, 8192, 4096], down_proj), ~26 GB per layer in bf16; the experts implementation is
    transformers' default (metadata.json "experts_implementation").
  * Logits: lm_head, then x output_multiplier (0.17678), then Gemma's softcap T * tanh(x / T) with T = 20
    (final_logit_softcapping). The engine's default applies only the softcap, so logits() does both.
  * Weights ship bf16 (the -NVFP4 repo is a separate checkpoint), names already HF's (model.language_model.*,
    lm_head).

Memory at N = 56320 on an H100 (80 GB): ~27 GB of layer weights; attention in --attn-chunk query rows (128 heads:
1024 rows keep even sdpa's math fallback near 30 GB); the MoE runs expert by expert. No expert offload is needed.

    python generate_golden_muse_spark_open.py --selftest                        # first: tiny model, CPU, exact
    python generate_golden_muse_spark_open.py --isl 1024 --num-layers 4 --root /tmp/smoke   # RoPE and NoPE layers
    python generate_golden_muse_spark_open.py --isl 56320 --attn-chunk 1024
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from engine import LayerwiseGolden, Recipe  # noqa: E402

MIN_TRANSFORMERS = (5, 20)
UV = "uv run --no-project --python python_env/bin/python --with transformers==5.20.2 python"

NOTES = {
    "model.embed_tokens.output": (
        "Token embeddings after the embedding module's weightless RMSNorm (embed_norm, eps rms_norm_eps): "
        "e * (mean(e^2) + eps) ** -0.5, no scale. The first layer's input."
    ),
    "model.layers.{i}.output": (
        "Decoder layer output: g2(h, post_feedforward_layernorm(mlp(pre_feedforward_layernorm(h)))), with h = "
        "g1(x, post_attention_layernorm(self_attn(input_layernorm(x)))) and g1 / g2 the residual gates "
        "alpha * residual + beta * branch. The next layer's input."
    ),
    "model.layers.{i}.self_attn.output": (
        "Attention output o_proj(o_norm(attn) * sigmoid(gate_proj(x))), before post_attention_layernorm (--sublayers)."
    ),
    "model.layers.{i}.self_attn.gate_proj.output": (
        "Output-gate logits gate_proj(x) [1, N, num_heads * head_dim], before the sigmoid; x is the "
        "input_layernorm output (--sublayers)."
    ),
    "model.layers.{i}.post_attention_residual_gate.output": (
        "The residual stream after attention: alpha * x + beta * post_attention_layernorm(self_attn.output), "
        "beta = sigmoid(gate / T), alpha = sqrt(max(sigmoid(-gate / T) * (1 + beta), 1e-3)), T = "
        "residual_gate_temperature, fp32 then cast (--sublayers)."
    ),
    "model.layers.{i}.mlp.pre_expert_norm.output": (
        "The routed experts' input [N, moe_hidden_size]: centered RMSNorm of pre_expert_proj(x) (--sublayers)."
    ),
    "model.layers.{i}.mlp.experts.output": (
        "Sum over the chosen experts of top_k_weights * post_expert_norm(down(silu(gate(x)) * up(x))) "
        "[N, moe_hidden_size], before post_expert_proj (--sublayers)."
    ),
    "model.layers.{i}.mlp.output": (
        "MoE output post_expert_proj(experts.output) [1, N, hidden], before post_feedforward_layernorm "
        "(--sublayers). No shared experts."
    ),
    "past_key_values.layers.{i}.keys": (
        "Keys [1, num_key_value_heads, N, head_dim]: k_proj, then the weightless qk_norm RMSNorm, then RoPE (half "
        "split, rope_theta) on sliding_attention layers only; full_attention layers (layer_rope_theta 0) are NoPE. "
        "Every position; a sliding layer attends to the last sliding_window (2048) of them. q (not stored) gets the "
        "same norm, then x qk_scale_factor."
    ),
    "past_key_values.layers.{i}.values": "Values [1, num_key_value_heads, N, head_dim]: v_proj, no norm, no RoPE.",
    "model.layers.{i}.mlp.gate.output": (
        "Router logits [N, num_local_experts] = fp32(pre_feedforward_layernorm output) @ fp32(gate.weight).T x "
        "output_multiplier, before the sigmoid. Selection: top-8 of sigmoid(logits) + e_score_correction_bias."
    ),
    "model.layers.{i}.mlp.experts.top_k_index": "The 8 chosen experts per token, ascending by id. int32.",
    "model.layers.{i}.mlp.experts.top_k_weights": (
        "Weights of top_k_index, same order: sigmoid(logits) of the chosen (no bias) / (their sum + 1e-15) x "
        "routed_scaling_factor. fp32."
    ),
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
            f"transformers {transformers.__version__} has no muse_spark: this script needs >= "
            f"{'.'.join(map(str, MIN_TRANSFORMERS))}. Run it through the overlay (never upgrade python_env):\n"
            f"    {UV} {Path(__file__).name} ..."
        )


class MuseSparkOpenGolden(LayerwiseGolden):
    hf_repo = "meta-models/Muse-Spark-Open-816B-A42B"
    # The BOS token, as the tokenizer adds it by default: Muse-Glimmer-30B's measured best (same tokenizer and
    # template family), not yet measured on Spark.
    prefix = "<|begin_of_text|>"
    recipe = Recipe(
        modules=("mlp.gate",),
        sublayers=(
            "self_attn",
            "self_attn.gate_proj",
            "post_attention_residual_gate",
            "mlp",
            "mlp.pre_expert_norm",
            "mlp.experts",
        ),
        routing="mlp.experts",
        # Routing is a top-k: rounding the logits can flip an expert at the cut, so they are never stored rounded.
        keep_fp32=("mlp.gate.output", "mlp.experts.top_k_weights"),
        config=(
            "layer_types",
            "layer_rope_theta",
            "mlp_layer_types",
            "rope_parameters",
            "sliding_window",
            "head_dim",
            "num_attention_heads",
            "num_key_value_heads",
            "qk_scale_factor",
            "rms_norm_eps",
            "post_norm_eps",
            "hidden_act",
            "intermediate_size",
            "moe_hidden_size",
            "moe_intermediate_size",
            "num_local_experts",
            "num_experts_per_tok",
            "routed_scaling_factor",
            "residual_gate_temperature",
            "output_multiplier",
            "final_logit_softcapping",
            "tie_word_embeddings",
            "max_position_embeddings",
        ),
        notes=NOTES,
    )

    def __init__(self):
        # The engine builds through AutoModelForCausalLM, which has no muse_spark entry (the checkpoint is
        # image-text-to-text). Register the checkpoint's class, as the Muse-Glimmer script does.
        check_transformers()
        from transformers import AutoModelForCausalLM, MuseSparkConfig, MuseSparkForConditionalGeneration

        AutoModelForCausalLM.register(MuseSparkConfig, MuseSparkForConditionalGeneration, exist_ok=True)

    def make_cache(self, text_config):
        """No config: transformers' DynamicCache(config) gives sliding layers a DynamicSlidingWindowLayer that keeps
        only the last sliding_window - 1 (2047) positions. Built without it, every layer keeps K/V for every position,
        as the other goldens store them; the masks come from the config either way, so the forward is unchanged."""
        from transformers import DynamicCache

        return DynamicCache()

    def layer_kwargs(self, i, kind, state):
        """MuseSparkTextModel.forward: NoPE layers (layer_rope_theta[i] == 0) get position_embeddings=None."""
        kwargs = super().layer_kwargs(i, kind, state)
        if not self.rope_theta[i]:
            kwargs["position_embeddings"] = None
        return kwargs

    def prepare(self, backbone, text_config, hidden, position_ids):
        self.rope_theta = list(text_config.layer_rope_theta)
        return super().prepare(backbone, text_config, hidden, position_ids)

    def logits(self, text_config, lm_head, hidden):
        """MuseSparkForConditionalGeneration.forward: x output_multiplier, then the tanh softcap."""
        logits = lm_head(hidden[:, -1:]) * text_config.output_multiplier
        cap = text_config.final_logit_softcapping
        return torch.tanh(logits / cap) * cap

    def tiny_init(self, model):
        super().tiny_init(model)
        g = torch.Generator().manual_seed(2)
        for name, t in model.named_parameters():
            if name.endswith("language_model.norm.weight"):  # the final, plain RMSNorm: w ~ 1
                t.copy_(1 + torch.randn(t.shape, generator=g) * 0.1)
            elif name.endswith("norm.weight"):  # centered / input-scaled norms: w ~ 0 like the real ones, not exactly
                t.copy_(torch.randn(t.shape, generator=g) * 0.1)
            elif name.endswith("residual_gate.gate"):  # a spread of alpha / beta mixes
                t.copy_(torch.randn(t.shape, generator=g))
        for name, t in model.named_buffers():
            if name.endswith("e_score_correction_bias"):  # stored in the checkpoint: test that it is loaded
                t.copy_(torch.randn(t.shape, generator=g) * 0.1)

    def tiny_config(self):
        """The real config cut down: two [sliding x3, full] blocks (RoPE and NoPE layers, both mask kinds), MoE on
        every layer but the first (the real checkpoint is all MoE; the dense path is covered in case a config has
        one), vision and audio towers shrunk (built, never read)."""
        from transformers import MuseSparkConfig

        n = 8
        return MuseSparkConfig(
            text_config=dict(
                vocab_size=512,
                hidden_size=64,
                intermediate_size=128,
                num_hidden_layers=n,
                num_attention_heads=4,
                num_key_value_heads=2,
                head_dim=16,
                sliding_window=32,  # < selftest isl: the window really cuts
                moe_hidden_size=32,
                moe_intermediate_size=32,
                num_local_experts=8,
                num_experts_per_tok=4,
                routed_scaling_factor=2.5,  # 1.0 by default: != 1 so applying it is tested
                mlp_layer_types=["dense"] + ["sparse"] * (n - 1),
                tie_word_embeddings=False,
                pad_token_id=None,
                bos_token_id=None,
                eos_token_id=None,
            ),
            vision_config=dict(num_hidden_layers=1, hidden_size=32, intermediate_size=32, num_attention_heads=2),
            audio_config=dict(hidden_sizes=(8, 8, 8, 8), out_hidden_size=64),
            out_hidden_size=64,
            projector_hidden_size=32,
        )


if __name__ == "__main__":
    sys.exit(MuseSparkOpenGolden().main())
