#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Prefill goldens for google/gemma-4-31B-it, layer by layer (engine.py), from transformers' Gemma4 code.

Stored tensors (N = isl, i = layer; 60 layers: 5 sliding_attention then 1 full_attention, repeated):

  hidden/embed.safetensors
    model.embed_tokens.output                  [1, N, 5376]       bf16  embeddings x sqrt(5376) (scale cast to the
                                                                        compute dtype: 73.32 -> 73.5 in bf16)
  hidden/layer_XX.safetensors
    model.layers.{i}.output                    [1, N, 5376]       bf16  all   x + post_ff_norm(mlp(pre_ff_norm(h))),
                                                                        h = x + post_attn_norm(attn(in_norm(x))),
                                                                        times the layer's layer_scalar
    model.layers.{i}.self_attn.o_proj.output   [1, N, 5376]       bf16  all   --sublayers: attention after o_proj,
                                                                        before post_attention_layernorm
    model.layers.{i}.mlp.output                [1, N, 5376]       bf16  all   --sublayers: GeGLU MLP output, before
                                                                        post_feedforward_layernorm
  cache/layer_XX.safetensors
    past_key_values.layers.{i}.keys            [1, 16, N, 256]    bf16  sliding  post k_norm + RoPE theta 1e4, every
                                                                        position (the device keeps the last 1024)
                                               [1, 4, N, 512]     bf16  full     post k_norm + proportional RoPE theta
                                                                        1e6: only dims [0:64], [256:320] rotate
    past_key_values.layers.{i}.values          [1, 16, N, 256]    bf16  sliding  v_norm (RMSNorm, no scale), no RoPE
                                               [1, 4, N, 512]     bf16  full     no v_proj (attention_k_eq_v): v_norm of
                                                                        the k_proj output
  hidden/final.safetensors
    model.last_hidden_state                    [1, N, 5376]       bf16  final norm
    logits                                     [1, 1, 262144]     fp32  last position: tied lm_head, then
                                                                        tanh(x / 30) * 30 (final_logit_softcapping)

The cache is a DynamicCache built without the config, so sliding layers keep K/V for every position rather than
the last window; the masks come from config.sliding_window either way, so the forward is unchanged.

    python generate_golden_gemma_4_31b_it.py --isl 10240
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from engine import LayerwiseGolden, Recipe, RunState  # noqa: E402

NOTES = {
    "model.embed_tokens.output": (
        "Token embeddings times sqrt(hidden_size), the scale cast to the compute dtype (exact in fp32; bf16 rounds "
        "it, 73.32 -> 73.5 for hidden 5376)."
    ),
    "model.layers.{i}.output": (
        "Decoder layer output: x + post_ff_norm(mlp(pre_ff_norm(h))) with h = x + post_attn_norm(attn(in_norm(x))), "
        "then times the layer's layer_scalar. The next layer's input."
    ),
    "model.layers.{i}.self_attn.o_proj.output": "Attention output after o_proj, before post_attention_layernorm.",
    "model.layers.{i}.mlp.output": "GeGLU MLP output (gelu_pytorch_tanh), before post_feedforward_layernorm.",
    "past_key_values.layers.{i}.keys": (
        "K for every position, post k_norm and RoPE (half-split). sliding_attention layers: [1, num_key_value_heads, "
        "N, head_dim], default RoPE, all dims rotated; the device only needs the last sliding_window of them. "
        "full_attention layers: [1, num_global_key_value_heads, N, global_head_dim], 'proportional' RoPE: only the "
        "first partial_rotary_factor of each half rotates (frequencies over global_head_dim), the rest pass through."
    ),
    "past_key_values.layers.{i}.values": (
        "V for every position, v_norm'd (RMSNorm, no learned scale), no RoPE. Same shapes as keys. On full_attention "
        "layers there is no v_proj (attention_k_eq_v): V is v_norm of the k_proj output, i.e. K's projection before "
        "k_norm and RoPE."
    ),
    "logits": "Last position only: lm_head (tied to embed_tokens) then the final_logit_softcapping tanh, fp32.",
}


class Gemma4Golden(LayerwiseGolden):
    hf_repo = "google/gemma-4-31B-it"
    recipe = Recipe(
        sublayers=("self_attn.o_proj", "mlp"),
        config=(
            "layer_types",
            "sliding_window",
            "head_dim",
            "global_head_dim",
            "num_attention_heads",
            "num_key_value_heads",
            "num_global_key_value_heads",
            "attention_k_eq_v",
            "rope_parameters",
            "rms_norm_eps",
            "hidden_activation",
            "intermediate_size",
            "final_logit_softcapping",
            "tie_word_embeddings",
        ),
        notes=NOTES,
    )

    def make_cache(self, text_config):
        from transformers import DynamicCache

        return DynamicCache()  # no config: sliding layers keep full-length K/V too

    def prepare(self, backbone, text_config, hidden, position_ids) -> RunState:
        state = super().prepare(backbone, text_config, hidden, position_ids)
        # The last layer of each kind publishes its K/V here for kv-shared layers (none in this checkpoint, but the
        # attention writes it regardless).
        state.carry["shared_kv_states"] = {}
        return state

    def tiny_config(self):
        from transformers import Gemma4Config

        layer_types = ["sliding_attention"] * 2 + ["full_attention"] + ["sliding_attention"] * 2 + ["full_attention"]
        return Gemma4Config(
            text_config=dict(
                vocab_size=512,
                hidden_size=128,
                intermediate_size=256,
                num_hidden_layers=len(layer_types),
                layer_types=layer_types,
                num_attention_heads=4,
                num_key_value_heads=2,
                head_dim=32,
                global_head_dim=64,
                num_global_key_value_heads=1,
                attention_k_eq_v=True,
                sliding_window=16,  # < selftest isl: the window really cuts
                hidden_size_per_layer_input=0,
                final_logit_softcapping=30.0,
                tie_word_embeddings=True,
            ),
            vision_config=None,
            audio_config=None,
        )


if __name__ == "__main__":
    sys.exit(Gemma4Golden().main())
