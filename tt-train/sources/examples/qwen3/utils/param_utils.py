# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Parameter mapping utilities for Qwen3 HF ↔ TTML weight conversion.

Contains:
  - Weight permutation / inverse-permutation helpers (HF ↔ TTML layout)
  - HF → TTML parameter name mapping builders (single-device and distributed)
  - Distributed gradient extraction helper
"""

import torch
import ttnn
import ttml

# =====================================================================
# Parameter name mapping builders (distributed / TP — qwen3-specific)
# =====================================================================


def build_weight_mapping_distributed(config, root_prefix, tie_word_embeddings):
    """Build HF→TTML name mapping, shard types, and transform specs for distributed weight loading.
    Returns (mapping, shard_types, transforms).
    shard_types values: "col_w" | "row_w" | "col_b" | None (replicated)
    transforms values: ("unpermute_proj", num_heads) | ("unpermute_norm",)
    """
    mapping = {}
    shard_types = {}
    transforms = {}

    if tie_word_embeddings:
        mapping["model.embed_tokens.weight"] = f"{root_prefix}/lm_head/weight"
        shard_types["model.embed_tokens.weight"] = "col_w"
    else:
        mapping["model.embed_tokens.weight"] = f"{root_prefix}/model/embed_tokens"
        shard_types["model.embed_tokens.weight"] = "row_w"
        mapping["lm_head.weight"] = f"{root_prefix}/lm_head/weight"
        shard_types["lm_head.weight"] = "col_w"

    for i in range(config.num_hidden_layers):
        hp = f"model.layers.{i}"
        tp = f"{root_prefix}/model/layers/{i}"

        mapping[f"{hp}.self_attn.q_proj.weight"] = f"{tp}/self_attn/q_proj/weight"
        shard_types[f"{hp}.self_attn.q_proj.weight"] = "col_w"
        transforms[f"{hp}.self_attn.q_proj.weight"] = (
            "unpermute_proj",
            config.num_attention_heads,
        )

        # Fused KV: the HF k_proj maps to the single ttml kv_proj param; the
        # combine_kv_tp transform pulls v_proj and builds the per-shard-interleaved
        # [K_s0,V_s0,K_s1,V_s1,...] layout so a contiguous col_w shard lands
        # [K_local|V_local] on each device (see load_weights_from_hf_distributed).
        # v_proj is consumed by that transform, so it is NOT mapped separately.
        mapping[f"{hp}.self_attn.k_proj.weight"] = f"{tp}/self_attn/kv_proj/weight"
        shard_types[f"{hp}.self_attn.k_proj.weight"] = "col_w"
        transforms[f"{hp}.self_attn.k_proj.weight"] = (
            "combine_kv_tp",
            config.num_key_value_heads,
            f"{hp}.self_attn.v_proj.weight",
        )

        mapping[f"{hp}.self_attn.o_proj.weight"] = f"{tp}/self_attn/o_proj/weight"
        shard_types[f"{hp}.self_attn.o_proj.weight"] = "row_w"

        if config.attention_bias:
            mapping[f"{hp}.self_attn.q_proj.bias"] = f"{tp}/self_attn/q_proj/col_bias"
            shard_types[f"{hp}.self_attn.q_proj.bias"] = "col_b"
            transforms[f"{hp}.self_attn.q_proj.bias"] = (
                "unpermute_proj",
                config.num_attention_heads,
            )

            mapping[f"{hp}.self_attn.k_proj.bias"] = f"{tp}/self_attn/kv_proj/col_bias"
            shard_types[f"{hp}.self_attn.k_proj.bias"] = "col_b"
            transforms[f"{hp}.self_attn.k_proj.bias"] = (
                "combine_kv_tp",
                config.num_key_value_heads,
                f"{hp}.self_attn.v_proj.bias",
            )

            mapping[f"{hp}.self_attn.o_proj.bias"] = f"{tp}/self_attn/o_proj/row_bias"
            shard_types[f"{hp}.self_attn.o_proj.bias"] = None  # replicated

        mapping[f"{hp}.self_attn.q_norm.weight"] = f"{tp}/self_attn/q_norm/weight"
        shard_types[f"{hp}.self_attn.q_norm.weight"] = None
        transforms[f"{hp}.self_attn.q_norm.weight"] = ("unpermute_norm",)

        mapping[f"{hp}.self_attn.k_norm.weight"] = f"{tp}/self_attn/k_norm/weight"
        shard_types[f"{hp}.self_attn.k_norm.weight"] = None
        transforms[f"{hp}.self_attn.k_norm.weight"] = ("unpermute_norm",)

        mapping[f"{hp}.input_layernorm.weight"] = f"{tp}/input_layernorm/weight"
        shard_types[f"{hp}.input_layernorm.weight"] = None
        mapping[f"{hp}.post_attention_layernorm.weight"] = f"{tp}/post_attention_layernorm/weight"
        shard_types[f"{hp}.post_attention_layernorm.weight"] = None

        mapping[f"{hp}.mlp.gate_proj.weight"] = f"{tp}/mlp/gate_proj/weight"
        shard_types[f"{hp}.mlp.gate_proj.weight"] = "col_w"
        mapping[f"{hp}.mlp.up_proj.weight"] = f"{tp}/mlp/up_proj/weight"
        shard_types[f"{hp}.mlp.up_proj.weight"] = "col_w"
        mapping[f"{hp}.mlp.down_proj.weight"] = f"{tp}/mlp/down_proj/weight"
        shard_types[f"{hp}.mlp.down_proj.weight"] = "row_w"

    mapping["model.norm.weight"] = f"{root_prefix}/model/norm/weight"
    shard_types["model.norm.weight"] = None

    return mapping, shard_types, transforms


def _build_grad_mapping_single(config, root_prefix, tie_word_embeddings):
    """Build HF→TTML mapping for single-device model.
    Returns (mapping, inv_transforms, grad_strategies).
    grad_strategies is None for single-device (not needed).
    """
    mapping = {}
    inv_transforms = {}

    if tie_word_embeddings:
        mapping["model.embed_tokens.weight"] = f"{root_prefix}/fc/weight"
    else:
        mapping["model.embed_tokens.weight"] = f"{root_prefix}/tok_emb/weight"
        mapping["lm_head.weight"] = f"{root_prefix}/fc/weight"

    for i in range(config.num_hidden_layers):
        hp = f"model.layers.{i}"
        tp = f"{root_prefix}/blocks/{i}"

        kv_out = config.num_key_value_heads * config.head_dim

        mapping[f"{hp}.self_attn.q_proj.weight"] = f"{tp}/self_attn/q_proj/weight"
        inv_transforms[f"{hp}.self_attn.q_proj.weight"] = (
            "repermute_proj",
            config.num_attention_heads,
        )
        # Fused KV: both HF k_proj and v_proj map to the single ttml kv_proj grad
        # [2*kv_out, hidden] (K rows then V rows). split_kv selects the matching
        # half; the K half is then re-permuted back to HF layout (V is not).
        mapping[f"{hp}.self_attn.k_proj.weight"] = f"{tp}/self_attn/kv_proj/weight"
        inv_transforms[f"{hp}.self_attn.k_proj.weight"] = (
            "split_kv",
            "k",
            kv_out,
            config.num_key_value_heads,
        )
        mapping[f"{hp}.self_attn.v_proj.weight"] = f"{tp}/self_attn/kv_proj/weight"
        inv_transforms[f"{hp}.self_attn.v_proj.weight"] = (
            "split_kv",
            "v",
            kv_out,
            config.num_key_value_heads,
        )
        mapping[f"{hp}.self_attn.o_proj.weight"] = f"{tp}/self_attn/o_proj/weight"

        if config.attention_bias:
            mapping[f"{hp}.self_attn.q_proj.bias"] = f"{tp}/self_attn/q_proj/bias"
            inv_transforms[f"{hp}.self_attn.q_proj.bias"] = (
                "repermute_proj",
                config.num_attention_heads,
            )
            mapping[f"{hp}.self_attn.k_proj.bias"] = f"{tp}/self_attn/kv_proj/bias"
            inv_transforms[f"{hp}.self_attn.k_proj.bias"] = (
                "split_kv",
                "k",
                kv_out,
                config.num_key_value_heads,
            )
            mapping[f"{hp}.self_attn.v_proj.bias"] = f"{tp}/self_attn/kv_proj/bias"
            inv_transforms[f"{hp}.self_attn.v_proj.bias"] = (
                "split_kv",
                "v",
                kv_out,
                config.num_key_value_heads,
            )
            mapping[f"{hp}.self_attn.o_proj.bias"] = f"{tp}/self_attn/o_proj/bias"

        mapping[f"{hp}.self_attn.q_norm.weight"] = f"{tp}/self_attn/q_norm/weight"
        inv_transforms[f"{hp}.self_attn.q_norm.weight"] = ("repermute_norm",)
        mapping[f"{hp}.self_attn.k_norm.weight"] = f"{tp}/self_attn/k_norm/weight"
        inv_transforms[f"{hp}.self_attn.k_norm.weight"] = ("repermute_norm",)

        mapping[f"{hp}.input_layernorm.weight"] = f"{tp}/input_layernorm/weight"
        mapping[f"{hp}.post_attention_layernorm.weight"] = f"{tp}/post_attention_layernorm/weight"
        mapping[f"{hp}.mlp.gate_proj.weight"] = f"{tp}/mlp/gate_proj/weight"
        mapping[f"{hp}.mlp.up_proj.weight"] = f"{tp}/mlp/up_proj/weight"
        mapping[f"{hp}.mlp.down_proj.weight"] = f"{tp}/mlp/down_proj/weight"

    mapping["model.norm.weight"] = f"{root_prefix}/ln_fc/weight"
    return mapping, inv_transforms, None


def _build_grad_mapping_distributed(config, root_prefix, tie_word_embeddings):
    """Build HF→TTML mapping for distributed model.
    Returns (mapping, inv_transforms, grad_strategies).

    grad_strategies per-parameter:
      "concat_2"       – ColumnParallel weight, sharded dim 2
      "concat_3"       – ColumnParallel bias / RowParallel weight, sharded dim 3
      "replicated"     – identical on all devices, take first
      "sum_replicated" – replicated weight with partial per-device grads (QK-norm)
    """
    mapping = {}
    inv_transforms = {}
    gs = {}  # grad_strategies

    if tie_word_embeddings:
        mapping["model.embed_tokens.weight"] = f"{root_prefix}/lm_head/weight"
        gs["model.embed_tokens.weight"] = "concat_2"
    else:
        mapping["model.embed_tokens.weight"] = f"{root_prefix}/model/embed_tokens"
        gs["model.embed_tokens.weight"] = "concat_3"
        mapping["lm_head.weight"] = f"{root_prefix}/lm_head/weight"
        gs["lm_head.weight"] = "concat_2"

    for i in range(config.num_hidden_layers):
        hp = f"model.layers.{i}"
        tp = f"{root_prefix}/model/layers/{i}"

        mapping[f"{hp}.self_attn.q_proj.weight"] = f"{tp}/self_attn/q_proj/weight"
        inv_transforms[f"{hp}.self_attn.q_proj.weight"] = (
            "repermute_proj",
            config.num_attention_heads,
        )
        gs[f"{hp}.self_attn.q_proj.weight"] = "concat_2"

        # Fused KV: both HF k_proj and v_proj map to the single ttml kv_proj grad.
        # concat_2 reassembles it to the global per-shard-interleaved layout
        # [K_s0,V_s0,K_s1,V_s1,...]; split_kv_tp de-interleaves it back to the HF
        # K (then re-permuted) or V half. The tp size is threaded in at compare time.
        mapping[f"{hp}.self_attn.k_proj.weight"] = f"{tp}/self_attn/kv_proj/weight"
        inv_transforms[f"{hp}.self_attn.k_proj.weight"] = (
            "split_kv_tp",
            "k",
            config.num_key_value_heads,
        )
        gs[f"{hp}.self_attn.k_proj.weight"] = "concat_2"

        mapping[f"{hp}.self_attn.v_proj.weight"] = f"{tp}/self_attn/kv_proj/weight"
        inv_transforms[f"{hp}.self_attn.v_proj.weight"] = (
            "split_kv_tp",
            "v",
            config.num_key_value_heads,
        )
        gs[f"{hp}.self_attn.v_proj.weight"] = "concat_2"

        mapping[f"{hp}.self_attn.o_proj.weight"] = f"{tp}/self_attn/o_proj/weight"
        gs[f"{hp}.self_attn.o_proj.weight"] = "concat_3"

        if config.attention_bias:
            mapping[f"{hp}.self_attn.q_proj.bias"] = f"{tp}/self_attn/q_proj/col_bias"
            inv_transforms[f"{hp}.self_attn.q_proj.bias"] = (
                "repermute_proj",
                config.num_attention_heads,
            )
            gs[f"{hp}.self_attn.q_proj.bias"] = "concat_3"

            mapping[f"{hp}.self_attn.k_proj.bias"] = f"{tp}/self_attn/kv_proj/col_bias"
            inv_transforms[f"{hp}.self_attn.k_proj.bias"] = (
                "split_kv_tp",
                "k",
                config.num_key_value_heads,
            )
            gs[f"{hp}.self_attn.k_proj.bias"] = "concat_3"

            mapping[f"{hp}.self_attn.v_proj.bias"] = f"{tp}/self_attn/kv_proj/col_bias"
            inv_transforms[f"{hp}.self_attn.v_proj.bias"] = (
                "split_kv_tp",
                "v",
                config.num_key_value_heads,
            )
            gs[f"{hp}.self_attn.v_proj.bias"] = "concat_3"

            mapping[f"{hp}.self_attn.o_proj.bias"] = f"{tp}/self_attn/o_proj/row_bias"
            gs[f"{hp}.self_attn.o_proj.bias"] = "replicated"

        mapping[f"{hp}.self_attn.q_norm.weight"] = f"{tp}/self_attn/q_norm/weight"
        inv_transforms[f"{hp}.self_attn.q_norm.weight"] = ("repermute_norm",)
        gs[f"{hp}.self_attn.q_norm.weight"] = "sum_replicated"

        mapping[f"{hp}.self_attn.k_norm.weight"] = f"{tp}/self_attn/k_norm/weight"
        inv_transforms[f"{hp}.self_attn.k_norm.weight"] = ("repermute_norm",)
        gs[f"{hp}.self_attn.k_norm.weight"] = "sum_replicated"

        mapping[f"{hp}.input_layernorm.weight"] = f"{tp}/input_layernorm/weight"
        gs[f"{hp}.input_layernorm.weight"] = "replicated"
        mapping[f"{hp}.post_attention_layernorm.weight"] = f"{tp}/post_attention_layernorm/weight"
        gs[f"{hp}.post_attention_layernorm.weight"] = "replicated"

        mapping[f"{hp}.mlp.gate_proj.weight"] = f"{tp}/mlp/gate_proj/weight"
        gs[f"{hp}.mlp.gate_proj.weight"] = "concat_2"
        mapping[f"{hp}.mlp.up_proj.weight"] = f"{tp}/mlp/up_proj/weight"
        gs[f"{hp}.mlp.up_proj.weight"] = "concat_2"
        mapping[f"{hp}.mlp.down_proj.weight"] = f"{tp}/mlp/down_proj/weight"
        gs[f"{hp}.mlp.down_proj.weight"] = "concat_3"

    mapping["model.norm.weight"] = f"{root_prefix}/model/norm/weight"
    gs["model.norm.weight"] = "replicated"

    return mapping, inv_transforms, gs


# =====================================================================
# Distributed gradient extraction
# =====================================================================


def _extract_grad_distributed(tensor, device, strategy, dp_size=1):
    """Extract gradient from a distributed parameter and reconstruct full tensor.

    When dp_size > 1, gradients must already be synchronized (all-reduced) across
    DP groups so all replicas are identical.  The concat over the full 2-D mesh
    (dp × tp) includes dp_size duplicate copies of each TP shard; this function
    strips the duplicates.
    """
    if not tensor.is_grad_initialized():
        return None
    grad_tt = tensor.get_grad()
    if strategy == "concat_2":
        composer = ttml.core.distributed.concat_mesh_to_tensor_composer(device, 2)
        grad = ttnn.to_torch(grad_tt, mesh_composer=composer).to(torch.float32)
        if dp_size > 1:
            grad = grad[:, :, : grad.shape[2] // dp_size, :]
        return grad
    elif strategy == "concat_3":
        composer = ttml.core.distributed.concat_mesh_to_tensor_composer(device, 3)
        grad = ttnn.to_torch(grad_tt, mesh_composer=composer).to(torch.float32)
        if dp_size > 1:
            grad = grad[:, :, :, : grad.shape[3] // dp_size]
        return grad
    elif strategy == "sum_replicated":
        composer = ttml.core.distributed.concat_mesh_to_tensor_composer(device, 0)
        grad = ttnn.to_torch(grad_tt, mesh_composer=composer).to(torch.float32)
        grad = grad.sum(dim=0, keepdim=True)
        if dp_size > 1:
            grad = grad / dp_size
        return grad
    else:  # "replicated" – identical on all devices, take first
        composer = ttml.core.distributed.concat_mesh_to_tensor_composer(device, 0)
        grad = ttnn.to_torch(grad_tt, mesh_composer=composer).to(torch.float32)
        return grad[:1]
