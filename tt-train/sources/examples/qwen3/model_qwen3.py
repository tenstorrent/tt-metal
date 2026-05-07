# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Qwen3 HF weight loading and re-exports.

The model implementation lives in ``ttml.models.qwen3``.  This module provides
``load_weights_from_hf`` for loading HuggingFace checkpoints, and re-exports
symbols consumed by ``model_qwen3_distributed.py`` and ``model_factory.py``.
"""

import sys

import torch
from tqdm import tqdm

import ttml
from ttml.modules import AbstractModuleBase, LinearLayer, ModuleList, Parameter

# Re-export shared components so existing callers (model_qwen3_distributed,
# model_factory, etc.) continue to work with ``from model_qwen3 import ...``
from ttml.models.qwen3 import Qwen3, Qwen3Config, RMSNormFunction, ConcatLastDim  # noqa: F401

from utils.tensor_utils import (
    torch_to_ttml,
    make_weight,
    make_ones,
    weight_initializer,
    zeros_initializer,
)
from utils.param_utils import (  # noqa: F401 — re-exported for callers
    unpermute_proj_rows,
    unpermute_norm_weights,
    build_weight_mapping_single,
)

from utils.memory import memory_snapshot
from utils.checkpoint import checkpoint

# Backwards-compat alias: callers that created Qwen3ForCausalLM now get Qwen3
Qwen3ForCausalLM = Qwen3


def linear(x, weight, bias=None):
    return ttml.ops.linear.linear(x, weight, bias)


# =====================================================================
# Qwen3RMSNorm
# =====================================================================


class Qwen3RMSNorm(AbstractModuleBase):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.hidden_size = hidden_size
        self.weight = Parameter(make_ones((1, 1, 1, hidden_size)))

    def forward(self, hidden_states):
        # requires for 14B/32B backward, without throws an error ttml::metal::rmsnorm_bw
        # "Statically allocated circular buffers on core range [(x=0,y=0) - (x=4,y=3)]
        # grow to 1764640 B which is beyond max L1 size of 1499136 B"
        return RMSNormFunction.apply(hidden_states, self.weight.tensor, self.eps)
        # return ttml.ops.rmsnorm.rmsnorm(hidden_states, self.weight.tensor, self.eps)


# =====================================================================
# Qwen3Attention (single device)
# =====================================================================


class Qwen3Attention(AbstractModuleBase):
    def __init__(self, config: Qwen3Config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = config.head_dim
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.hidden_size = config.hidden_size

        q_out = self.num_heads * self.head_dim
        kv_out = self.num_kv_heads * self.head_dim

        w_init = weight_initializer()
        b_init = zeros_initializer() if config.attention_bias else None
        self.q_proj = LinearLayer(
            self.hidden_size, q_out, has_bias=config.attention_bias, weight_init=w_init, bias_init=b_init
        )
        self.k_proj = LinearLayer(
            self.hidden_size, kv_out, has_bias=config.attention_bias, weight_init=w_init, bias_init=b_init
        )
        self.v_proj = LinearLayer(
            self.hidden_size, kv_out, has_bias=config.attention_bias, weight_init=w_init, bias_init=b_init
        )
        self.o_proj = LinearLayer(
            q_out, self.hidden_size, has_bias=config.attention_bias, weight_init=w_init, bias_init=b_init
        )

        self.q_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)

        rope_scaling = ttml.ops.rope.RopeScalingParams()
        if config.rope_scaling_factor != 0.0 and config.rope_original_context_length != 0:
            rope_scaling.original_context_length = config.rope_original_context_length
            rope_scaling.scaling_factor = config.rope_scaling_factor
            rope_scaling.high_freq_factor = config.rope_high_freq_factor
            rope_scaling.low_freq_factor = config.rope_low_freq_factor

        self.rope_params = ttml.ops.rope.build_rope_params(
            sequence_length=config.max_position_embeddings,
            head_dim=self.head_dim,
            theta=config.rope_theta,
            rope_scaling_params=rope_scaling,
        )

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        past_key_values=None,
        position_offset=0,
    ):
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        q_shape = q.shape()
        k_shape = k.shape()
        B, S = q_shape[0], q_shape[2]

        q = ttml.ops.reshape.reshape(q, [B, 1, S * self.num_heads, self.head_dim])
        k = ttml.ops.reshape.reshape(k, [B, 1, S * self.num_kv_heads, self.head_dim])
        q = self.q_norm(q)
        k = self.k_norm(k)
        q = ttml.ops.reshape.reshape(q, q_shape)
        k = ttml.ops.reshape.reshape(k, k_shape)

        kvs = ConcatLastDim.apply(k, v)
        (
            query_heads,
            key_heads,
            value_heads,
        ) = ttml.ops.multi_head_utils.grouped_heads_creation(q, kvs, self.num_heads, self.num_kv_heads)

        query_heads = ttml.ops.rope.rope(query_heads, self.rope_params, position_offset)
        key_heads = ttml.ops.rope.rope(key_heads, self.rope_params, position_offset)

        # KV cache: append new K/V and use full history for attention
        if past_key_values is not None:
            key_heads, value_heads = past_key_values.update(self.layer_idx, key_heads, value_heads)

        q_seq = query_heads.shape()[2]
        k_seq = key_heads.shape()[2]
        sdpa_fn = (
            ttml.ops.attention.scaled_dot_product_attention
            if q_seq == k_seq
            else ttml.ops.attention.scaled_dot_product_attention_composite
        )
        attn = sdpa_fn(query_heads, key_heads, value_heads, attention_mask)

        attn_output = ttml.ops.multi_head_utils.heads_fusion(attn)
        return self.o_proj(attn_output)


# =====================================================================
# Qwen3MLP (single device)
# =====================================================================


class Qwen3MLP(AbstractModuleBase):
    def __init__(self, config: Qwen3Config):
        super().__init__()
        h = config.hidden_size
        inter = config.intermediate_size
        w_init = weight_initializer()
        self.gate_proj = LinearLayer(h, inter, has_bias=False, weight_init=w_init)
        self.up_proj = LinearLayer(h, inter, has_bias=False, weight_init=w_init)
        self.down_proj = LinearLayer(inter, h, has_bias=False, weight_init=w_init)

    def forward(self, x):
        gate = ttml.ops.unary.silu(self.gate_proj(x))
        up = self.up_proj(x)
        return self.down_proj(ttml.ops.binary.mul(gate, up))


# =====================================================================
# Qwen3DecoderLayer
# =====================================================================


class Qwen3DecoderLayer(AbstractModuleBase):
    def __init__(self, config: Qwen3Config, layer_idx: int):
        super().__init__()
        self.self_attn = Qwen3Attention(config, layer_idx)
        self.mlp = Qwen3MLP(config)
        self.input_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        past_key_values=None,
        position_offset=0,
    ):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states,
            attention_mask,
            past_key_values,
            position_offset=position_offset,
        )
        hidden_states = ttml.ops.binary.add(residual, hidden_states)

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = ttml.ops.binary.add(residual, hidden_states)
        return hidden_states


# =====================================================================
# Qwen3Model (backbone)
# =====================================================================


class Qwen3Model(AbstractModuleBase):
    def __init__(self, config: Qwen3Config, track_memory=0, use_checkpoint=False):
        super().__init__()
        self.config = config
        self.track_memory = track_memory
        self.use_checkpoint = use_checkpoint
        vocab_size_tiled = ((config.vocab_size + 31) // 32) * 32
        self.embed_tokens = Parameter(make_weight((1, 1, vocab_size_tiled, config.hidden_size)))
        self.layers = ModuleList([Qwen3DecoderLayer(config, i) for i in range(config.num_hidden_layers)])
        self.norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, input_ids, attention_mask=None, past_key_values=None):
        hidden_states = ttml.ops.embedding.embedding(input_ids, self.embed_tokens.tensor)
        if self.track_memory:
            hidden_states = memory_snapshot(hidden_states, "AFTER_EMBEDDING_FWD", "AFTER_EMBEDDING_BWD")
        position_offset = 0
        if past_key_values is not None:
            position_offset = past_key_values.get_seq_length()
        for i, layer in enumerate(self.layers):
            if self.use_checkpoint:
                hidden_states = checkpoint(
                    layer,
                    hidden_states,
                    attention_mask,
                    past_key_values,
                    position_offset,
                )
            else:
                hidden_states = layer(
                    hidden_states,
                    attention_mask,
                    past_key_values,
                    position_offset=position_offset,
                )
            if self.track_memory and (i + 1) % self.track_memory == 0:
                hidden_states = memory_snapshot(hidden_states, f"AFTER_LAYER_{i}_FWD", f"AFTER_LAYER_{i}_BWD")
        hidden_states = self.norm(hidden_states)
        return hidden_states


# =====================================================================
# Qwen3ForCausalLM
# =====================================================================


class Qwen3ForCausalLM(AbstractModuleBase):
    def __init__(
        self,
        config: Qwen3Config,
        tie_word_embeddings: bool = False,
        track_memory: int = 0,
        use_checkpoint=False,
    ):
        super().__init__()
        self.create_name("Qwen3ForCausalLM")
        self.config = config
        self.tie_word_embeddings = tie_word_embeddings
        self.track_memory = track_memory
        self.model = Qwen3Model(config, track_memory=track_memory, use_checkpoint=use_checkpoint)

        if tie_word_embeddings:
            self.lm_head_weight = None
        else:
            vocab_size_tiled = ((config.vocab_size + 31) // 32) * 32
            self.lm_head_weight = Parameter(make_weight((1, 1, vocab_size_tiled, config.hidden_size)))

    def forward(self, input_ids, attention_mask=None, past_key_values=None, **kwargs):
        hidden_states = self.model(input_ids, attention_mask, past_key_values)
        if self.track_memory:
            hidden_states = memory_snapshot(hidden_states, "AFTER_NORM_FWD", "AFTER_NORM_BWD")
        if self.tie_word_embeddings:
            logits = linear(hidden_states, self.model.embed_tokens.tensor, None)
        else:
            logits = linear(hidden_states, self.lm_head_weight.tensor, None)
        if self.track_memory:
            logits = memory_snapshot(logits, "AFTER_LM_HEAD_FWD", "AFTER_LM_HEAD_BWD")
        return logits


# =====================================================================
# Weight loading from HuggingFace
# =====================================================================


def load_weights_from_hf(
    ttml_model,
    hf_state_dict: dict,
    config: Qwen3Config,
    tie_word_embeddings: bool = False,
    verbose: bool = False,
) -> None:
    """Load HF weights into single-device ttml Qwen3 model."""
    ttml_params = ttml_model.parameters()

    if verbose:
        print("\n  TTML parameter names:")
        for name in sorted(ttml_params.keys()):
            shape = ttml_params[name].shape()
            print(f"    {name}: {list(shape)}")

    any_key = next(iter(ttml_params))
    root_prefix = any_key.split("/")[0]

    mapping, transforms = build_weight_mapping_single(config, root_prefix, tie_word_embeddings)

    ttml_shapes = {name: list(ttml_params[name].shape()) for name in ttml_params}

    def _prepare_and_transfer(hf_name, ttml_name):
        """CPU prep + host-side conversion + device transfer (pipelined)."""
        if hf_name not in hf_state_dict:
            return None
        if ttml_name not in ttml_shapes:
            return None

        weight = hf_state_dict[hf_name].float()

        if hf_name in transforms:
            tr = transforms[hf_name]
            if tr[0] == "unpermute_proj":
                weight = unpermute_proj_rows(weight, num_heads=tr[1])
            elif tr[0] == "unpermute_norm":
                weight = unpermute_norm_weights(weight)

        ttml_shape = ttml_shapes[ttml_name]

        if weight.dim() == 2:
            rows, cols = weight.shape
            tgt_rows, tgt_cols = ttml_shape[2], ttml_shape[3]
            if rows != tgt_rows or cols != tgt_cols:
                padded = torch.zeros(tgt_rows, tgt_cols, dtype=weight.dtype)
                padded[: min(rows, tgt_rows), : min(cols, tgt_cols)] = weight[
                    : min(rows, tgt_rows), : min(cols, tgt_cols)
                ]
                weight = padded
            weight = weight.unsqueeze(0).unsqueeze(0)
        elif weight.dim() == 1:
            dim = weight.shape[0]
            tgt_dim = ttml_shape[-1]
            if dim != tgt_dim:
                padded = torch.zeros(tgt_dim, dtype=weight.dtype)
                padded[: min(dim, tgt_dim)] = weight[: min(dim, tgt_dim)]
                weight = padded
            weight = weight.unsqueeze(0).unsqueeze(0).unsqueeze(0)

        return torch_to_ttml(weight)

    from concurrent.futures import ThreadPoolExecutor

    items = list(mapping.items())
    loaded = 0
    skipped = []

    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [
            (hf_name, ttml_name, pool.submit(_prepare_and_transfer, hf_name, ttml_name)) for hf_name, ttml_name in items
        ]

        for hf_name, ttml_name, future in tqdm(
            futures,
            total=len(items),
            desc="  Loading weights",
            unit="w",
            file=sys.stdout,
        ):
            new_tensor = future.result()
            if new_tensor is None:
                if ttml_name not in ttml_shapes:
                    print(f"  WARNING: ttml param '{ttml_name}' not found for HF '{hf_name}'")
                skipped.append(hf_name)
                continue
            ttml_params[ttml_name].assign(new_tensor)
            loaded += 1

    print(f"\n  Weight loading: {loaded} loaded, {len(skipped)} skipped")
    if skipped:
        print(f"  Skipped: {skipped}")
