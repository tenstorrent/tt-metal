#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Prefill goldens for meta-llama/Llama-3.1-8B-Instruct, layer by layer (engine.py), from transformers' llama code.

Stored tensors (N = isl, i = layer; 32 dense layers, all full causal GQA attention (32 query heads, 8 KV heads of
128), SwiGLU MLP):

  hidden/embed.safetensors
    model.embed_tokens.output                  [1, N, 4096]       bf16  embeddings, no scale
  hidden/layer_XX.safetensors
    model.layers.{i}.output                    [1, N, 4096]       bf16  all   h + mlp(post_attn_norm(h)),
                                                                        h = x + attn(input_norm(x))
    model.layers.{i}.self_attn.output          [1, N, 4096]       bf16  all   --sublayers: attention after o_proj,
                                                                        before the residual
    model.layers.{i}.mlp.output                [1, N, 4096]       bf16  all   --sublayers: SwiGLU MLP, before the
                                                                        residual
  cache/layer_XX.safetensors
    past_key_values.layers.{i}.keys            [1, 8, N, 128]     bf16  all   k_proj -> RoPE (half split, theta 5e5,
                                                                        llama3 scaling), no norm
    past_key_values.layers.{i}.values          [1, 8, N, 128]     bf16  all   v_proj, no norm, no RoPE
  hidden/final.safetensors
    model.last_hidden_state                    [1, N, 4096]       bf16  final RMSNorm
    logits                                     [1, 1, 128256]     fp32  last position, untied lm_head

What is model-specific:

  * RoPE: rope_theta 500000 with llama3 frequency scaling (factor 8, low_freq_factor 1, high_freq_factor 4, original
    context 8192): wavelengths below 8192 / 4 keep their frequency, those above 8192 / 1 are divided by 8, and the
    band in between is interpolated. The scaling is a fixed rescale of inv_freq, the same at every ISL; there is no
    attention scaling (attention_factor 1).
  * Weights ship bf16 with HF names (model.*, lm_head); the original/ consolidated checkpoint is never read.

    python generate_golden_llama_3_1_8b_instruct.py --isl 10240
    python generate_golden_llama_3_1_8b_instruct.py --isl 56320 --prefill-chunk 5120 --attn-chunk 4096
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
    "model.layers.{i}.mlp.output": "SwiGLU MLP output down(silu(gate(x)) * up(x)), before the residual add (--sublayers).",
    "past_key_values.layers.{i}.keys": (
        "Keys [1, num_key_value_heads, N, head_dim]: k_proj, then RoPE (half-split rotation, rope_theta 500000, "
        "llama3 frequency scaling), no norm."
    ),
    "past_key_values.layers.{i}.values": "Values [1, num_key_value_heads, N, head_dim]: v_proj, no norm, no RoPE.",
    "model.last_hidden_state": "Final RMSNorm of the last layer's output.",
    "logits": "Last position only: lm_head (not tied to embed_tokens) in the compute dtype, stored fp32.",
}


class Llama31Golden(LayerwiseGolden):
    hf_repo = "meta-llama/Llama-3.1-8B-Instruct"
    # The BOS token, as the tokenizer adds it by default. Wikitext ppl over the same 10100 tokens at isl 10240: 8.358
    # BOS (top-1 53.6%), 8.428 raw, 8.623 framed as its own reply by the chat template ("<|begin_of_text|>
    # <|start_header_id|>system<|end_header_id|>\n\nCutting Knowledge Date: ...<|eot_id|><|start_header_id|>user
    # <|end_header_id|>\n\nContinue this WikiText-2 dump verbatim: ...<|eot_id|><|start_header_id|>assistant
    # <|end_header_id|>\n\n").
    prefix = "<|begin_of_text|>"
    recipe = Recipe(
        config=(
            "head_dim",
            "num_attention_heads",
            "num_key_value_heads",
            "rope_parameters",
            "rope_scaling",
            "rope_theta",
            "rms_norm_eps",
            "hidden_act",
            "intermediate_size",
            "attention_bias",
            "mlp_bias",
            "tie_word_embeddings",
            "max_position_embeddings",
        ),
        notes=NOTES,
    )

    def tiny_init(self, model):
        super().tiny_init(model)
        g = torch.Generator().manual_seed(2)
        for name, t in model.named_parameters():
            if name.endswith("norm.weight"):  # O(1) norms, like real ones
                t.copy_(1 + torch.randn(t.shape, generator=g) * 0.1)

    def tiny_config(self):
        from transformers import LlamaConfig

        return LlamaConfig(
            vocab_size=512,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=4,
            num_attention_heads=8,
            num_key_value_heads=2,
            head_dim=8,
            # llama3 scaling with a short original context, so every band of the rescale is exercised
            rope_parameters={
                "rope_type": "llama3",
                "rope_theta": 500000.0,
                "factor": 8.0,
                "low_freq_factor": 1.0,
                "high_freq_factor": 4.0,
                "original_max_position_embeddings": 32,
            },
            tie_word_embeddings=False,
            pad_token_id=None,
        )


if __name__ == "__main__":
    sys.exit(Llama31Golden().main())
