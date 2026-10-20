#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Prefill goldens for IFM/K2-Horizon-375B-A23B, layer by layer (engine.py), from the repo's own remote code
(modeling_k2_horizon.py at the pinned commit, trust_remote_code; a Qwen3-MoE derivative).

Stored tensors (N = isl, i = layer; 61 layers, all full causal GQA attention (48 query heads, 8 KV heads of 128, no
q / k norm, no sliding window, no attention gate); dense MLP on layers 0-2 (mlp_only_layers), MoE (192 routed
experts, top 8, 1 shared) on 3-60):

  hidden/embed.safetensors
    model.embed_tokens.output                  [1, N, 6144]       bf16  token embeddings (no scale)
  hidden/layer_XX.safetensors
    model.layers.{i}.output                    [1, N, 6144]       bf16  all   h + mlp(post_attn_norm(h)),
                                                                        h = x + attn(input_norm(x))
    model.layers.{i}.self_attn.output          [1, N, 6144]       bf16  all   --sublayers: attention after o_proj,
                                                                        before the residual
    model.layers.{i}.mlp.output                [1, N, 6144]       bf16  all   --sublayers: dense MLP / routed +
                                                                        shared experts, before the residual
  cache/layer_XX.safetensors
    past_key_values.layers.{i}.keys            [1, 8, N, 128]     bf16  all   k_proj -> partial RoPE (see below),
                                                                        theta 1e7
    past_key_values.layers.{i}.values          [1, 8, N, 128]     bf16  all   v_proj, no RoPE
    model.layers.{i}.mlp.gate.router_logits    [N, 192]           fp32  3-60  x @ gate.weight.T in bf16 (no bias),
                                                                        stored widened, before sigmoid
    model.layers.{i}.mlp.gate.top_k_index      [N, 8]             int32 3-60  chosen experts, ascending expert id
    model.layers.{i}.mlp.gate.top_k_weights    [N, 8]             fp32  3-60  their weights, same order:
                                                                        normalized sigmoid, x 2.5 (before the bf16 cast)
  hidden/final.safetensors
    model.last_hidden_state                    [1, N, 6144]       bf16  final norm
    logits                                     [1, 1, 250624]     fp32  last position, untied lm_head

What is model-specific:

  * Partial RoPE (rope_head_dim 64 of head_dim 128) applied through an interleave: the code rotates the 64 dims
    interleave(x)[:64] = (x[0], x[64], x[1], x[65], ..., x[31], x[95]) with rotate_half and 32 frequencies
    theta**(-2j/64). Per head that is: pairs (j, j + 64) for j < 32 rotate by position * freq_j; dims 32-63 and
    96-127 pass through. cos / sin are cast to the compute dtype first (bf16), as in Llama. Scaling 128**-0.5.
  * Routing (K2HorizonSparseMoeBlock): logits = F.linear(x, gate.weight) in the compute dtype (bf16, unlike most
    MoEs, which run the router in fp32), sigmoid in fp32, top-8 of sigmoid + gate.bias (a selection-only bias,
    bf16 in the checkpoint, never added to the logits nor the weights), weights from the unbiased sigmoid,
    renormalized and x router_scaling_factor (2.5), then cast to bf16 before scaling the expert outputs. The
    shared expert is added unweighted. The block keeps its top-k local: the hook recomputes it, verbatim, from the
    logits the block returns.
  * Weights are plain bf16 under the HF names (experts one module per expert, no fusing): convert is the identity.
    MoVA attention (routed value experts) exists in the code but is off (mova_num_experts 0).
  * The tokenizer prepends <|ifm|begin_of_text|> (id 0) when add_special_tokens is on; the token cache is made
    without it, so the BOS comes from the prefix.

    python generate_golden_k2_horizon.py --isl 10240 --prefill-chunk 5120
    python generate_golden_k2_horizon.py --isl 256 --num-layers 4 --root /tmp/smoke   # dense, then MoE
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
        "Dense SwiGLU MLP (layers 0-2), or routed experts (weighted) + the shared expert, before the residual add "
        "(--sublayers)."
    ),
    "past_key_values.layers.{i}.keys": (
        "Keys [1, num_key_value_heads, N, head_dim]: k_proj (no norm), then partial RoPE (rope_head_dim 64, "
        "rope_theta 1e7, no scaling): within each head, dim pairs (j, j + 64) for j < 32 are rotated by "
        "position * theta**(-2j/64) (x_j' = x_j cos - x_{j+64} sin, x_{j+64}' = x_{j+64} cos + x_j sin); dims 32-63 "
        "and 96-127 pass through. cos / sin in the compute dtype."
    ),
    "past_key_values.layers.{i}.values": "Values [1, num_key_value_heads, N, head_dim]: v_proj, no RoPE.",
    "model.layers.{i}.mlp.gate.router_logits": (
        "Router logits [N, num_experts] = x @ gate.weight.T in the compute dtype (bf16; the bias is not added), "
        "stored widened to fp32. Selection: top-k of sigmoid(logits in fp32) + gate.bias."
    ),
    "model.layers.{i}.mlp.gate.top_k_index": "The num_experts_per_tok chosen experts per token, ascending by expert id. int32.",
    "model.layers.{i}.mlp.gate.top_k_weights": (
        "Weights of top_k_index, same order: sigmoid(logits) of the chosen experts (no bias) divided by their sum, "
        "times router_scaling_factor (2.5), in fp32. The model casts them to the compute dtype before scaling the "
        "expert outputs; the shared expert is added unweighted."
    ),
    "model.last_hidden_state": "Final RMSNorm of the last layer's output.",
    "logits": "Last position only: lm_head (not tied to embed_tokens) in the compute dtype, stored fp32.",
}


class K2HorizonGolden(LayerwiseGolden):
    hf_repo = "IFM/K2-Horizon-375B-A23B"
    trust_remote_code = True
    # The BOS the tokenizer itself prepends. Wikitext ppl over the same 10100 tokens at isl 10240: 6.266 (top-1 58.0%)
    # with it, 6.278 raw, 6.325 framed by the chat template as its own reply ("<|ifm|begin_of_text|><|ifm|im_start|>user
    # \nContinue this WikiText-2 dump verbatim: ...<|ifm|im_end|><|ifm|im_start|>assistant\n<ifm|think>\n</ifm|think>").
    prefix = "<|ifm|begin_of_text|>"
    recipe = Recipe(
        # The logits are bf16 values (widening is exact); the weights are stored as computed, before their bf16 cast.
        keep_fp32=("mlp.gate.router_logits", "mlp.gate.top_k_weights"),
        config=(
            "mlp_only_layers",
            "decoder_sparse_step",
            "head_dim",
            "num_attention_heads",
            "num_key_value_heads",
            "query_key_norm",
            "attention_gate_func",
            "attention_bias",
            "rope_head_dim",
            "rope_parameters",
            "sliding_window",
            "rms_norm_eps",
            "layernorm_num_groups",
            "hidden_act",
            "intermediate_size",
            "moe_intermediate_size",
            "num_experts",
            "num_shared_experts",
            "num_experts_per_tok",
            "router_score_func",
            "moe_gate_bias",
            "norm_topk_prob",
            "router_scaling_factor",
            "mova_num_experts",
            "tie_word_embeddings",
            "max_position_embeddings",
        ),
        notes=NOTES,
    )

    def layer_types(self, text_config) -> list[str]:
        """'full_attention/<mlp>': every layer attends the same way; the MLP kind decides what a layer stores."""
        n, dense = text_config.num_hidden_layers, set(text_config.mlp_only_layers)
        step = text_config.decoder_sparse_step
        return [
            f"full_attention/{'moe' if i not in dense and (i + 1) % step == 0 else 'dense'}" for i in range(n)
        ]

    def register_hooks(self, i, layer, captured, sublayers):
        handles = super().register_hooks(i, layer, captured, sublayers)
        moe = layer.mlp
        if not hasattr(moe, "gate"):  # a dense layer
            return handles
        key = f"model.layers.{i}.mlp.gate"

        def routing(moe, _inputs, output):
            logits = output[1]
            # K2HorizonSparseMoeBlock.forward keeps its top-k local; these are its lines, verbatim.
            scores = torch.sigmoid(logits.to(torch.float32))
            choice = scores + moe.gate.bias.to(scores.dtype) if moe.gate.bias is not None else scores
            _, index = torch.topk(choice, moe.top_k, dim=-1)
            weights = torch.gather(scores, dim=-1, index=index)
            if moe.norm_topk_prob:
                weights = weights / weights.sum(dim=-1, keepdim=True)
            weights = weights * moe.router_scaling_factor
            index, order = index.sort(dim=-1)
            captured[f"{key}.router_logits"] = logits
            captured[f"{key}.top_k_index"] = index.to(torch.int32)
            captured[f"{key}.top_k_weights"] = weights.gather(-1, order)

        handles.append(moe.register_forward_hook(routing))
        return handles

    def tiny_init(self, model):
        super().tiny_init(model)
        g = torch.Generator().manual_seed(2)
        for name, t in model.named_parameters():
            if name.endswith("layernorm.weight") or name.endswith("norm.weight"):  # O(1) norms, like real ones
                t.copy_(1 + torch.randn(t.shape, generator=g) * 0.1)

    def tiny_config(self):
        from common import resolve_revision
        from transformers import AutoConfig

        revision = resolve_revision(self.hf_repo, "main")
        config = AutoConfig.from_pretrained(self.hf_repo, revision=revision, trust_remote_code=True)
        config.update(
            dict(
                vocab_size=512,
                hidden_size=64,
                intermediate_size=128,
                moe_intermediate_size=32,
                num_hidden_layers=4,  # dense, then MoE
                mlp_only_layers=[0],
                num_attention_heads=4,
                num_key_value_heads=2,
                head_dim=32,
                rope_head_dim=16,  # partial, as in the real model (half of head_dim)
                num_experts=8,
                num_experts_per_tok=2,
                num_shared_experts=1,
                max_position_embeddings=4096,
            )
        )
        return config


if __name__ == "__main__":
    sys.exit(K2HorizonGolden().main())
