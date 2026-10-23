#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Prefill goldens for mistralai/Mistral-Medium-3.5-128B, layer by layer (engine.py), from transformers' ministral3
code (text backbone of mistral3; the vision tower and projector are never built into a stage nor read).

Stored tensors (N = isl, i = layer; 88 dense layers, all full causal GQA attention (96 query heads, 8 KV heads of
128), SwiGLU MLP):

  hidden/embed.safetensors
    model.embed_tokens.output                  [1, N, 12288]      bf16  embeddings, no scale
  hidden/layer_XX.safetensors
    model.layers.{i}.output                    [1, N, 12288]      bf16  all   h + mlp(post_attn_norm(h)),
                                                                        h = x + attn(input_norm(x))
    model.layers.{i}.self_attn.output          [1, N, 12288]      bf16  all   --sublayers: attention after o_proj,
                                                                        before the residual
    model.layers.{i}.mlp.output                [1, N, 12288]      bf16  all   --sublayers: SwiGLU MLP, before the
                                                                        residual
  cache/layer_XX.safetensors
    past_key_values.layers.{i}.keys            [1, 8, N, 128]     bf16  all   k_proj -> RoPE (half split, theta 1e6,
                                                                        yarn), no norm
    past_key_values.layers.{i}.values          [1, 8, N, 128]     bf16  all   v_proj, no norm, no RoPE
  hidden/final.safetensors
    model.last_hidden_state                    [1, N, 12288]      bf16  final RMSNorm
    logits                                     [1, 1, 131072]     fp32  last position, untied lm_head

What is model-specific:

  * Weights: every decoder linear (q/k/v/o_proj, gate/up/down_proj) ships float8_e4m3fn with an fp32 per-tensor
    weight_scale_inv and a static per-tensor activation_scale; embeddings, norms and lm_head ship bf16. convert()
    dequantizes each weight with transformers' own routine (finegrained_fp8.Fp8Dequantize: fp32 product, cast to the
    compute dtype), which is what from_pretrained does wherever there is no FP8-capable GPU. Activations stay in the
    compute dtype: the activation scales, which an FP8 inference engine applies to quantize each linear's input, are
    not applied, and are recorded in metadata.json ("fp8_activation_scales").
  * RoPE: rope_theta 1e6 with yarn (factor 64, original context 4096, beta_fast 4, beta_slow 1, mscale 1,
    mscale_all_dim 0): inv_freq interpolated per dim, and cos / sin scaled by the yarn attention factor
    0.1 * ln(64) + 1 = 1.4159. A fixed rescale, the same at every ISL. The attention also multiplies q by the Llama-4
    position scale 1 + llama_4_scaling_beta * log(1 + floor(pos / max_position_embeddings)), which is 1 here (beta 0).
  * Checkpoint names are mistral3's (model.language_model.*, lm_head); the Mistral-native consolidated-*.safetensors
    copy in the same repo is never read.

    python generate_golden_mistral_medium_3_5.py --isl 10240
    python generate_golden_mistral_medium_3_5.py --isl 56320 --prefill-chunk 5120 --attn-chunk 4096
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from engine import LayerwiseGolden, Recipe  # noqa: E402

NOTES = {
    "model.embed_tokens.output": "Token embeddings (no scale).",
    "model.layers.{i}.output": "Decoder layer output, residual included: the next layer's input.",
    "model.layers.{i}.self_attn.output": "Attention output after o_proj, before the residual add (--sublayers).",
    "model.layers.{i}.mlp.output": "SwiGLU MLP output down(silu(gate(x)) * up(x)), before the residual add (--sublayers).",
    "past_key_values.layers.{i}.keys": (
        "Keys [1, num_key_value_heads, N, head_dim]: k_proj (FP8 weight, dequantized), then RoPE (half-split "
        "rotation, rope_theta 1e6, yarn: interpolated inv_freq, cos / sin x 1.4159), no norm."
    ),
    "past_key_values.layers.{i}.values": "Values [1, num_key_value_heads, N, head_dim]: v_proj, no norm, no RoPE.",
    "model.last_hidden_state": "Final RMSNorm of the last layer's output.",
    "logits": "Last position only: lm_head (bf16, not tied to embed_tokens) in the compute dtype, stored fp32.",
}


class MistralMedium35Golden(LayerwiseGolden):
    hf_repo = "mistralai/Mistral-Medium-3.5-128B"
    # The BOS token (the tokenizer adds none by itself). Wikitext ppl over the same 9600 tokens at isl 10240 (chunked
    # 5120): 6.271 BOS (top-1 58.6%), 6.408 raw, 6.445 framed as its own reply by the chat template (its default
    # 390-token Le Chat system prompt, [MODEL_SETTINGS]{"reasoning_effort": "none"}, then [INST]Continue this
    # WikiText-2 dump verbatim: ...[/INST]).
    prefix = "<s>"
    recipe = Recipe(
        config=(
            "head_dim",
            "num_attention_heads",
            "num_key_value_heads",
            "rope_parameters",
            "sliding_window",
            "rms_norm_eps",
            "hidden_act",
            "intermediate_size",
            "attention_bias",
            "tie_word_embeddings",
            "max_position_embeddings",
        ),
        notes=NOTES,
    )

    def __init__(self):
        # AutoModelForCausalLM has no mistral3 entry, and maps ministral3 to Ministral3ForCausalLM, whose names
        # (model.layers.*) are not the checkpoint's (model.language_model.layers.*); register the checkpoint's class.
        from transformers import AutoModelForCausalLM, Mistral3Config, Mistral3ForConditionalGeneration

        AutoModelForCausalLM.register(Mistral3Config, Mistral3ForConditionalGeneration, exist_ok=True)

    def generate(self, args):
        self.activation_scales = {}
        out = super().generate(args)
        meta = json.loads((out / "metadata.json").read_text())
        meta["fp8_weights"] = (
            "float8_e4m3fn x per-tensor weight_scale_inv, dequantized in fp32 by transformers' Fp8Dequantize and cast "
            "to the compute dtype; activations not quantized"
        )
        meta["fp8_activation_scales"] = self.activation_scales
        (out / "metadata.json").write_text(json.dumps(meta, indent=2, default=str))
        return out

    def convert(self, sd, prefix, module):
        """<name>.weight (float8_e4m3fn) with its <name>.weight_scale_inv -> the dequantized weight in the parameter's
        dtype, by transformers' Fp8Dequantize, as from_pretrained loads it without an FP8 GPU. <name>.activation_scale
        is recorded and dropped. bf16 tensors (and the unquantized --selftest checkpoint) pass through."""
        from transformers.integrations.finegrained_fp8 import Fp8Dequantize

        dq = Fp8Dequantize(None)
        for scale in [k for k in sd if k.endswith(".weight_scale_inv")]:
            weight = scale.removesuffix("_scale_inv")
            dtype = module.get_parameter(weight).dtype
            sd[weight] = dq._dequantize_one(sd[weight], sd.pop(scale), output_dtype=dtype)
        for act in [k for k in sd if k.endswith(".activation_scale")]:
            self.activation_scales[prefix + act] = sd.pop(act).item()
        return sd

    def selftest(self, args, compute, isl=64):
        """transformers 5.12 saves mistral3 under its legacy names (language_model.model.*, language_model.lm_head),
        not the checkpoint's (model.language_model.*, lm_head): read the tiny checkpoint by those."""
        self.checkpoint_prefix, self.lm_head_prefix = "language_model.model.", "language_model.lm_head."
        try:
            return super().selftest(args, compute, isl)
        finally:
            self.checkpoint_prefix = self.lm_head_prefix = None

    def tiny_init(self, model):
        super().tiny_init(model)
        g = torch.Generator().manual_seed(2)
        for name, t in model.named_parameters():
            if name.endswith("norm.weight"):  # O(1) norms, like real ones
                t.copy_(1 + torch.randn(t.shape, generator=g) * 0.1)

    def tiny_config(self):
        from transformers import Mistral3Config

        return Mistral3Config(
            text_config=dict(
                model_type="ministral3",
                vocab_size=512,
                hidden_size=64,
                intermediate_size=128,
                num_hidden_layers=4,
                num_attention_heads=8,
                num_key_value_heads=2,
                head_dim=8,
                max_position_embeddings=4096,
                # yarn with a short original context, so the interpolation and the attention factor both act
                rope_parameters={
                    "rope_type": "yarn",
                    "rope_theta": 1e6,
                    "factor": 64.0,
                    "original_max_position_embeddings": 32,
                    "beta_fast": 4.0,
                    "beta_slow": 1.0,
                    "mscale": 1.0,
                    "mscale_all_dim": 0.0,
                    "llama_4_scaling_beta": 0.1,  # nonzero, so the position scale is tested; the checkpoint has 0
                },
                tie_word_embeddings=False,
                pad_token_id=None,
            ),
            vision_config=dict(
                model_type="pixtral",
                hidden_size=32,
                intermediate_size=32,
                num_hidden_layers=1,
                num_attention_heads=2,
                head_dim=16,
            ),
            tie_word_embeddings=False,
        )


if __name__ == "__main__":
    sys.exit(MistralMedium35Golden().main())
