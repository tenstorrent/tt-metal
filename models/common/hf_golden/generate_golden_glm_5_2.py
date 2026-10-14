#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Prefill goldens for zai-org/GLM-5.2, layer by layer (engine.py), from transformers' glm_moe_dsa code.

Stored tensors (N = isl, i = layer, k = min(2048, N); 78 layers: MLA + DSA sparse attention everywhere, the indexer
"full" on 0-2, 6, 10, ... (21 layers) and "shared" on the rest; dense MLP on 0-2, MoE (256 experts, top 8, 1 shared)
on 3-77):

  hidden/embed.safetensors
    model.embed_tokens.output                              [1, N, 6144]   bf16   embeddings (no scale)
  hidden/layer_XX.safetensors
    model.layers.{i}.output                                [1, N, 6144]   bf16   all     decoder output, residual
                                                                                  included
    model.layers.{i}.self_attn.output                      [1, N, 6144]   bf16   all     --sublayers: after o_proj
    model.layers.{i}.mlp.output                            [1, N, 6144]   bf16   all     --sublayers: dense MLP /
                                                                                  routed + shared experts
  cache/layer_XX.safetensors
    model.layers.{i}.self_attn.kv_a_proj_with_mqa.output   [1, N, 576]    bf16   all     MLA latent [c_kv (512) |
                                                                                  k_pe (64)], raw: before
                                                                                  kv_a_layernorm and RoPE
    past_key_values.layers.{i}.indexer_keys                [1, 1, N, 128] bf16   all     indexer keys: wk -> k_norm
                                                                                  -> RoPE on the first 64 dims
    model.layers.{i}.self_attn.indexer.output              [1, N, k]      int64  full    top-k key indices per
                                                                                  query, descending score; row s:
                                                                                  first min(s+1, k) real (sets)
    model.layers.{i}.mlp.gate.output                       [N, 256]       fp32   3-77    router logits, before
                                                                                  sigmoid / correction bias
    model.layers.{i}.mlp.experts.top_k_index               [N, 8]         int32  3-77    chosen experts, ascending
    model.layers.{i}.mlp.experts.top_k_weights             [N, 8]         fp32   3-77    their weights, same order:
                                                                                  normalized, x 2.5
  hidden/final.safetensors
    model.last_hidden_state                                [1, N, 6144]   bf16   final norm
    logits                                                 [1, 1, 154880] fp32   last position, untied lm_head

The HF cache also holds K/V decompressed to 64 heads, which the device never stores, so only the indexer keys are
kept. A "shared" indexer layer reuses the previous full layer's top-k: the layer returns it and the next layer is
handed it (prev_topk_indices), as GlmMoeDsaModel.forward does.

    python generate_golden_glm_5_2.py --isl 10240
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from engine import LayerwiseGolden, Recipe, RunState  # noqa: E402

NOTES = {
    "model.layers.{i}.self_attn.kv_a_proj_with_mqa.output": (
        "MLA latent [c_kv (kv_lora_rank) | k_pe (qk_rope_head_dim)], the raw projection: before kv_a_layernorm on "
        "c_kv and before the (interleaved) RoPE on k_pe. Apply both to compare against a device latent cache that "
        "stores them normed / rotated."
    ),
    "model.layers.{i}.self_attn.indexer.output": (
        "DSA top-k key indices per query, k = min(index_topk, n_tokens), in descending score order. Only the first "
        "min(s+1, k) entries of row s are real; the rest are causally masked -inf ties in arbitrary order, so compare "
        "those first entries as sets. Present only on indexer_types == 'full' layers; a 'shared' layer reuses the "
        "previous full layer's. When n_tokens <= index_topk every key is selected and attention is effectively dense."
    ),
    "past_key_values.layers.{i}.indexer_keys": (
        "Indexer keys: wk -> k_norm -> RoPE on the first qk_rope_head_dim dims, as transformers implements it "
        "(half-split rotation; this transformers version does not read config.indexer_rope_interleave)."
    ),
    "model.layers.{i}.self_attn.output": "Attention output after o_proj, before the residual add (--sublayers).",
    "model.layers.{i}.mlp.output": "Dense MLP, or routed + shared experts, before the residual add (--sublayers).",
    "model.layers.{i}.mlp.gate.output": "Router logits [n_tokens, n_routed_experts], before sigmoid and e_score_correction_bias.",
    "model.layers.{i}.mlp.experts.top_k_index": (
        "Routed experts chosen per token (num_experts_per_tok of them), sorted ascending by expert id."
    ),
    "model.layers.{i}.mlp.experts.top_k_weights": (
        "Weights of top_k_index, same order: sigmoid scores normalized over the chosen experts (norm_topk_prob) and "
        "scaled by routed_scaling_factor; shared experts are added unweighted."
    ),
}


class GlmMoeDsaGolden(LayerwiseGolden):
    hf_repo = "zai-org/GLM-5.2"
    recipe = Recipe(
        modules=("self_attn.kv_a_proj_with_mqa", "self_attn.indexer", "mlp.gate"),
        routing="mlp.experts",
        cache=("indexer_keys",),
        # Routing is a top-k: rounding the logits can flip an expert at the cut, so they are never stored rounded.
        keep_fp32=("mlp.gate.output", "mlp.experts.top_k_weights"),
        config=(
            "indexer_types",
            "mlp_layer_types",
            "index_topk",
            "index_n_heads",
            "index_head_dim",
            "indexer_rope_interleave",
            "rope_interleave",
            "kv_lora_rank",
            "q_lora_rank",
            "qk_rope_head_dim",
            "qk_nope_head_dim",
            "v_head_dim",
            "n_routed_experts",
            "num_experts_per_tok",
            "n_shared_experts",
            "routed_scaling_factor",
            "norm_topk_prob",
            "scoring_func",
        ),
        notes=NOTES,
    )

    def layer_types(self, text_config) -> list[str]:
        """'<indexer>/<mlp>', e.g. 'full/dense', 'shared/sparse': both decide what a layer stores."""
        n = text_config.num_hidden_layers
        return [f"{a}/{m}" for a, m in zip(text_config.indexer_types[:n], text_config.mlp_layer_types[:n])]

    def prepare(self, backbone, text_config, hidden, position_ids) -> RunState:
        state = super().prepare(backbone, text_config, hidden, position_ids)  # one causal mask for every kind
        state.carry["prev_topk_indices"] = None
        return state

    def after_layer(self, i, output, state):
        hidden, state.carry["prev_topk_indices"] = output  # a shared indexer layer reuses this top-k
        return hidden

    def compare(self, name, got, ref, exact):
        if name.endswith("indexer.output"):  # masked -inf ties come in any order: compare each row as a set
            got, ref = got.sort(dim=-1).values, ref.sort(dim=-1).values
        return super().compare(name, got, ref, exact)

    def tiny_init(self, model):
        super().tiny_init(model)
        for name, b in model.named_buffers():
            if name.endswith("e_score_correction_bias"):  # zero from from_config: randomize so loading it is tested
                b.copy_(torch.randn(b.shape, generator=torch.Generator().manual_seed(2)) * 0.1 + 1.0)

    def tiny_config(self):
        from transformers import GlmMoeDsaConfig

        n = 8
        return GlmMoeDsaConfig(
            vocab_size=512,
            hidden_size=128,
            intermediate_size=256,
            moe_intermediate_size=64,
            num_hidden_layers=n,
            num_attention_heads=4,
            num_key_value_heads=4,
            q_lora_rank=64,
            kv_lora_rank=32,
            qk_nope_head_dim=32,
            qk_rope_head_dim=16,
            v_head_dim=32,
            index_head_dim=32,
            index_n_heads=4,
            index_topk=16,  # < selftest isl: DSA is really sparse
            n_routed_experts=8,
            num_experts_per_tok=2,
            n_group=1,
            topk_group=1,
            indexer_types=["full"] * 3 + ["full" if (i - 2) % 4 == 0 else "shared" for i in range(3, n)],
            mlp_layer_types=["dense"] * 3 + ["sparse"] * (n - 3),
        )


if __name__ == "__main__":
    sys.exit(GlmMoeDsaGolden().main())
