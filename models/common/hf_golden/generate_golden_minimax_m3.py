#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Prefill goldens for MiniMaxAI/MiniMax-M3, layer by layer (engine.py), from transformers' minimax_m3_vl code.

Stored tensors (N = isl, i = layer; 60 layers: 0-2 full_attention + dense MLP, 3-59 minimax_m3_sparse + MoE;
B = min(16, ceil(N / 128)) selected key blocks):

  hidden/embed.safetensors
    model.embed_tokens.output                   [1, N, 6144]       bf16  embed_tokens (no scaling)
  hidden/layer_XX.safetensors
    model.layers.{i}.output                     [1, N, 6144]       bf16  all     h + mlp(post_attn_norm(h)),
                                                                         h = x + attn(input_norm(x))
    model.layers.{i}.self_attn.output           [1, N, 6144]       bf16  all     --sublayers: attention after
                                                                         o_proj (no post-attention norm, no gate)
    model.layers.{i}.mlp.output                 [1, N, 6144]       bf16  all     --sublayers: dense SwiGLU-OAI MLP
                                                                         (0-2) / whole MoE block (3-59), before
                                                                         the residual add
    model.layers.{i}.mlp.shared_experts.output  [N, 6144]          bf16  3-59    --sublayers: shared expert
    model.layers.{i}.mlp.experts.output         [N, 6144]          bf16  3-59    --sublayers: weighted sum of the
                                                                         routed experts, BEFORE x routed_scaling_factor
                                                                         (mlp = 2.0 * experts + shared_experts)
  cache/layer_XX.safetensors
    past_key_values.layers.{i}.keys             [1, 4, N, 128]     bf16  all     post per-head k_norm (Gemma RMSNorm,
                                                                         x (1 + w)) + partial RoPE theta 5e6: only
                                                                         dims [0:64] rotate (half-split within them)
    past_key_values.layers.{i}.values           [1, 4, N, 128]     bf16  all     v_proj, no norm, no RoPE
    past_key_values.layers.{i}.idx_keys         [1, 1, N, 128]     bf16  3-59    lightning-indexer keys:
                                                                         index_k_proj -> index_k_norm -> RoPE [0:64]
    model.layers.{i}.self_attn.indexer.output   [1, N, B]          int64 3-59    selected key-block ids per query
                                                                         (blocks of 128 keys), descending score, the
                                                                         query's own block first, -1 = no block
    model.layers.{i}.mlp.gate.output            [N, 128]           fp32  3-59    router logits (bf16 values under
                                                                         bf16 compute), before sigmoid and bias
    model.layers.{i}.mlp.experts.top_k_index    [N, 4]             int32 3-59    chosen experts, ascending id
    model.layers.{i}.mlp.experts.top_k_weights  [N, 4]             fp32  3-59    sigmoid(logits) of the chosen,
                                                                         normalized to sum 1, same order
  hidden/final.safetensors
    model.last_hidden_state                     [1, N, 6144]       bf16  final norm
    logits                                      [1, 1, 200064]     fp32  last position, untied lm_head, no softcap

Every RMSNorm (layer norms, final norm, q/k norms, indexer q/k norms) is Gemma-style: fp32 normalize, x (1 + w).
The routed experts use SwiGLU-OAI: (clamp(up, +-7) + 1) * g * sigmoid(1.702 g), g = clamp(gate, max=7).

Sparse attention: on minimax_m3_sparse layers the lightning indexer scores each query against every past key
(4 heads, max over heads and over the 128 keys of a block), forces the query's own block in, keeps the top 16
blocks, and attention is restricted to those blocks (plus causality). Up to N = 16 * 128 = 2048 every causal
block is selected, so attention is dense; only longer prefills exercise the selection.

The router weight and e_score_correction_bias are fp32 in the checkpoint; from_pretrained (and so this script)
casts them to the compute dtype, so under bf16 compute the expert choice uses a bf16 router and a bf16 bias.
--compute-dtype float32 keeps them exact.

    python generate_golden_minimax_m3.py --isl 10240
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from engine import LayerwiseGolden, Recipe  # noqa: E402

NOTES = {
    "model.embed_tokens.output": "Token embeddings (embed_tokens, no scaling) of the first n_tokens ids.",
    "model.layers.{i}.output": (
        "Decoder layer output: h + mlp(post_attention_layernorm(h)) with h = x + self_attn(input_layernorm(x)). "
        "The next layer's input."
    ),
    "model.layers.{i}.self_attn.output": (
        "Attention output after o_proj (no post-attention norm, no output gate), before the residual add. On "
        "minimax_m3_sparse layers attention only sees the key blocks the indexer selected."
    ),
    "model.layers.{i}.mlp.output": (
        "MLP output before the residual add: the dense SwiGLU-OAI MLP on 'dense' layers, the whole MoE block "
        "(routed_scaling_factor * routed experts + shared expert) on 'sparse' layers."
    ),
    "model.layers.{i}.mlp.shared_experts.output": "Shared expert (SwiGLU-OAI MLP) output, tokens flattened.",
    "model.layers.{i}.mlp.experts.output": (
        "Routed experts: sum over the top-k of top_k_weights * expert(x), tokens flattened, BEFORE the "
        "routed_scaling_factor (2.0) is applied."
    ),
    "past_key_values.layers.{i}.keys": (
        "K for every position [1, num_key_value_heads, N, head_dim]: k_proj, per-head k_norm (Gemma RMSNorm, "
        "x (1 + w)), then RoPE on the first rotary_dim = 64 dims only (half-split rotation within those 64; "
        "theta 5e6), the last 64 dims pass through."
    ),
    "past_key_values.layers.{i}.values": "V for every position [1, num_key_value_heads, N, head_dim]: v_proj only.",
    "past_key_values.layers.{i}.idx_keys": (
        "Lightning-indexer keys [1, 1, N, index_head_dim] (minimax_m3_sparse layers): index_k_proj, index_k_norm "
        "(Gemma RMSNorm), RoPE on the first 64 dims like the attention keys. The state the indexer carries to "
        "the next token alongside K/V."
    ),
    "model.layers.{i}.self_attn.indexer.output": (
        "Key blocks each query may attend [1, N, min(index_topk_blocks, ceil(N / index_block_size))], int64: block "
        "score = max over index heads and over the block's keys of q_idx . k_idx (fp32, causal), the query's own "
        "block forced to +inf; top-k in descending score order, -1 where fewer blocks exist. Ties at equal score "
        "may come in any order: compare as sets."
    ),
    "model.layers.{i}.mlp.gate.output": (
        "Router logits [N, num_local_experts], stored fp32 but computed at the compute dtype (from_pretrained "
        "casts the fp32 checkpoint router to it), before sigmoid and e_score_correction_bias."
    ),
    "model.layers.{i}.mlp.experts.top_k_index": (
        "Routed experts per token (num_experts_per_tok), chosen by sigmoid(logits) + e_score_correction_bias, "
        "sorted ascending by expert id."
    ),
    "model.layers.{i}.mlp.experts.top_k_weights": (
        "Weights of top_k_index, same order: sigmoid(logits) of the chosen experts (no bias) normalized to sum 1. "
        "routed_scaling_factor is applied after the experts, not here."
    ),
    "logits": "Last position only: lm_head (untied), fp32.",
}

# Checkpoint (MiniMax / vLLM layout) -> transformers module names, within one decoder layer: what from_pretrained's
# "minimax_m3_vl" conversion mapping does (transformers/conversion_mapping.py).
RENAMES = (
    ("block_sparse_moe.experts.", "mlp.experts."),
    ("block_sparse_moe.shared_experts.", "mlp.shared_experts."),
    ("block_sparse_moe.gate.weight", "mlp.gate.weight"),
    ("block_sparse_moe.e_score_correction_bias", "mlp.gate.e_score_correction_bias"),
    ("self_attn.index_q_proj.", "self_attn.indexer.q_proj."),
    ("self_attn.index_k_proj.", "self_attn.indexer.k_proj."),
    ("self_attn.index_q_norm.", "self_attn.indexer.q_norm."),
    ("self_attn.index_k_norm.", "self_attn.indexer.k_norm."),
)
EXPERT_LEAVES = {".w1.weight": ".gate_proj.weight", ".w3.weight": ".up_proj.weight", ".w2.weight": ".down_proj.weight"}


class MiniMaxM3Golden(LayerwiseGolden):
    hf_repo = "MiniMaxAI/MiniMax-M3"
    checkpoint_prefix = "language_model.model."  # the VLM stores the text backbone under language_model.model
    lm_head_prefix = "language_model.lm_head."
    recipe = Recipe(
        modules=("self_attn.indexer", "mlp.gate"),
        sublayers=("self_attn", "mlp", "mlp.shared_experts", "mlp.experts"),
        routing="mlp.experts",
        cache=("keys", "values", "idx_keys"),
        # Routing is a top-k: rounding the logits can flip an expert at the cut, so they are never stored rounded.
        keep_fp32=("mlp.gate.output", "mlp.experts.top_k_weights"),
        config=(
            "layer_types",
            "mlp_layer_types",
            "head_dim",
            "num_attention_heads",
            "num_key_value_heads",
            "rope_parameters",
            "rotary_dim",
            "rms_norm_eps",
            "dense_intermediate_size",
            "intermediate_size",
            "shared_intermediate_size",
            "num_local_experts",
            "num_experts_per_tok",
            "routed_scaling_factor",
            "swiglu_alpha",
            "swiglu_limit",
            "index_n_heads",
            "index_head_dim",
            "index_block_size",
            "index_topk_blocks",
            "index_local_blocks",
            "tie_word_embeddings",
        ),
        notes=NOTES,
    )

    def __init__(self):
        # The engine (build, --selftest's reference) builds through AutoModelForCausalLM, which transformers maps
        # to the text-only MiniMaxM3VLForCausalLM. The checkpoint is the VLM; register it, as transformers itself
        # does for Gemma4ForConditionalGeneration, so the meta model and from_pretrained see the real layout.
        from transformers import AutoModelForCausalLM, MiniMaxM3SparseForConditionalGeneration, MiniMaxM3VLConfig

        AutoModelForCausalLM.register(MiniMaxM3VLConfig, MiniMaxM3SparseForConditionalGeneration, exist_ok=True)

    def convert(self, sd, prefix, module):
        out = {}
        for k, v in sd.items():
            for old, new in RENAMES:
                k = k.replace(old, new)
            if ".experts." in k and not k.startswith("mlp.shared_experts."):
                for old, new in EXPERT_LEAVES.items():
                    k = k.removesuffix(old) + new if k.endswith(old) else k
            out[k] = v
        # Per-expert gate/up/down are fused by the Loader (gate first, as the mapping's Concatenate(dim=1)); the
        # dense MLP and the shared expert hold one fused nn.Linear: Concatenate(dim=0) of gate_proj, up_proj.
        for mlp in ("mlp.", "mlp.shared_experts."):
            if f"{mlp}gate_proj.weight" in out:
                gate, up = out.pop(f"{mlp}gate_proj.weight"), out.pop(f"{mlp}up_proj.weight")
                out[f"{mlp}gate_up_proj.weight"] = torch.cat([gate, up], dim=0)
        return out

    def tiny_config(self):
        from transformers import MiniMaxM3VLConfig

        layer_types = ["full_attention"] * 2 + ["minimax_m3_sparse"] * 3
        return MiniMaxM3VLConfig(
            text_config=dict(
                vocab_size=512,
                hidden_size=128,
                intermediate_size=64,
                dense_intermediate_size=192,
                shared_intermediate_size=64,
                num_hidden_layers=len(layer_types),
                layer_types=layer_types,
                mlp_layer_types=["dense" if t == "full_attention" else "sparse" for t in layer_types],
                num_attention_heads=4,
                num_key_value_heads=2,
                head_dim=32,
                rope_parameters=dict(rope_type="default", rope_theta=5e6, partial_rotary_factor=0.5),
                num_local_experts=16,
                num_experts_per_tok=4,
                index_n_heads=2,
                index_head_dim=32,
                index_block_size=8,  # 64-token selftest: 8 key blocks, of which 3 are kept, so the selection cuts
                index_topk_blocks=3,
                index_local_blocks=1,
                bos_token_id=None,
                eos_token_id=None,
            ),
            vision_config=dict(
                hidden_size=32, intermediate_size=64, num_hidden_layers=1, num_attention_heads=2, image_size=28
            ),
            projector_hidden_size=64,
        )


if __name__ == "__main__":
    sys.exit(MiniMaxM3Golden().main())
