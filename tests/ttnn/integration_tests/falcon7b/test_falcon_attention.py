# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0
import math
import torch
from torch.nn import functional as F
import transformers
import pytest
from loguru import logger

import ttnn
from ttnn.model_preprocessing import preprocess_model_parameters
from tests.ttnn.utils_for_testing import assert_with_pcc

torch.manual_seed(0)


def torch_functional_rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


### Torch Functional Implementation ###
def torch_functional_falcon_linear(hidden_states, parameters):
    hidden_states = hidden_states @ parameters.weight
    if parameters.get("bias", None):
        hidden_states = hidden_states + parameters.bias
    return hidden_states


def torch_functional_falcon_attention_split_heads(fused_qkv, multi_query, num_heads, head_dim):
    batch_size, seq_length, three_times_hidden_size = fused_qkv.shape
    if not multi_query:
        fused_qkv = fused_qkv.view(batch_size, seq_length, num_heads, 3, head_dim)
        return fused_qkv[..., 0, :], fused_qkv[..., 1, :], fused_qkv[..., 2, :]
    else:
        fused_qkv = fused_qkv.view(batch_size, seq_length, num_heads + 2, head_dim)
        return fused_qkv[..., :-2, :], fused_qkv[..., [-2], :], fused_qkv[..., [-1], :]


def torch_functional_apply_rotary_pos_emb(x, cos_cached, sin_cached, token_idx=None):
    seq_len = x.shape[-2]
    if token_idx is None:
        cos = cos_cached[:seq_len, ...]
        sin = sin_cached[:seq_len, ...]
    else:
        cos = cos_cached[token_idx : token_idx + 1, ...]
        sin = sin_cached[token_idx : token_idx + 1, ...]

    x_embed = (x * cos) + (torch_functional_rotate_half(x) * sin)
    return x_embed


def torch_functional_generate_sin_cos_rotary_embedding(
    seq_len,
    head_dim,
    rotary_embedding_cache,
    max_position_embeddings=2048,
    base=10000,
):
    max_cached_len = rotary_embedding_cache.get("max_len", 0)
    if seq_len <= max_cached_len:
        cos_emb, sin_emb = rotary_embedding_cache["embeddings"]
        return cos_emb[:seq_len], sin_emb[:seq_len]

    # Compute new embeddings for increased sequence length
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
    dtype = torch.get_default_dtype()
    t = torch.arange(seq_len, dtype=dtype)
    freqs = torch.outer(t, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    cos_emb, sin_emb = emb.cos(), emb.sin()

    # Update cache
    rotary_embedding_cache["max_len"] = max_position_embeddings
    rotary_embedding_cache["embeddings"] = (cos_emb, sin_emb)

    return cos_emb, sin_emb


def ttnn_functional_falcon_linear(hidden_states, parameters):
    hidden_states = hidden_states @ parameters.weight
    if parameters.get("bias", None):
        hidden_states = hidden_states + parameters.bias
    return hidden_states


def ttnn_functional_falcon_mlp(hidden_states, *, parameters):
    hidden_states = ttnn_functional_falcon_linear(hidden_states, parameters.dense_h_to_4h)
    hidden_states = ttnn.gelu(hidden_states)
    hidden_states = ttnn_functional_falcon_linear(hidden_states, parameters.dense_4h_to_h)


def torch_functional_falcon_attention(
    hidden_states, attention_mask, *, parameters, layer_past=None, num_heads=71, head_dim=64, multi_query=True
):
    assert multi_query
    num_kv_heads = 1

    fused_qkv = torch_functional_falcon_linear(hidden_states, parameters.query_key_value)
    query_layer, key_layer, value_layer = torch_functional_falcon_attention_split_heads(
        fused_qkv, multi_query, num_heads, head_dim
    )

    batch_size, query_length, _, _ = query_layer.shape

    query_layer = query_layer.transpose(1, 2).reshape(batch_size, num_heads, query_length, head_dim)
    key_layer = key_layer.transpose(1, 2).reshape(batch_size, num_kv_heads, query_length, head_dim)
    value_layer = value_layer.transpose(1, 2).reshape(batch_size, num_kv_heads, query_length, head_dim)

    kv_seq_len = key_layer.shape[-2]
    if layer_past is not None:
        kv_seq_len += layer_past[0].shape[-2]

    rotary_embedding_cache = dict()
    cos, sin = torch_functional_generate_sin_cos_rotary_embedding(
        query_length,
        head_dim,
        rotary_embedding_cache,
    )
    query_layer, key_layer = torch_functional_apply_rotary_pos_emb(
        query_layer, cos, sin
    ), torch_functional_apply_rotary_pos_emb(key_layer, cos, sin)

    if layer_past is not None:
        past_key, past_value = layer_past
        # concatenate along seq_length dimension:
        #  - key: [batch_size, self.num_heads, kv_length, head_dim]
        #  - value: [batch_size, self.num_heads, kv_length, head_dim]
        key_layer = torch.cat((past_key, key_layer), dim=-2)
        value_layer = torch.cat((past_value, value_layer), dim=-2)

    kv_length = key_layer.shape[-2]
    use_cache = False
    if use_cache:
        present = (key_layer, value_layer)
    else:
        present = None

    attention_scores = query_layer @ key_layer.transpose(-1, -2)
    attention_scores /= math.sqrt(head_dim)

    attention_scores = F.softmax(attention_scores + attention_mask, dim=-1, dtype=hidden_states.dtype)
    # It is unclear why neither dropout nor head_mask is applied here (while it is with alibi).
    attn_output = attention_scores @ value_layer

    attn_output = attn_output.view(batch_size, num_heads, query_length, head_dim)
    attn_output = attn_output.permute(0, 2, 1, 3)
    attn_output = attn_output.reshape(batch_size, query_length, num_heads * head_dim)

    attn_output = torch_functional_falcon_linear(attn_output, parameters.dense)

    return attn_output, present


@pytest.mark.parametrize("model_name", ["tiiuae/falcon-7b-instruct"])
@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize("sequence_length", [128])
def test_torch_functional_falcon_attention(model_name, batch_size, sequence_length):
    config = transformers.FalconConfig.from_pretrained(model_name)
    model = transformers.models.falcon.modeling_falcon.FalconAttention(config).eval()

    parameters = preprocess_model_parameters(
        initialize_model=lambda: model,
        convert_to_ttnn=lambda *_: False,
    )

    torch_hidden_states = (torch.rand(batch_size, sequence_length, config.hidden_size, dtype=torch.float32) * 2) - 1
    torch_attention_mask = torch.ones(1, sequence_length)
    torch_output, torch_present = model.forward(torch_hidden_states, alibi=None, attention_mask=torch_attention_mask)

    hidden_states = ttnn.from_torch(torch_hidden_states, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    attention_mask = ttnn.from_torch(torch_attention_mask, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)

    output, present = torch_functional_falcon_attention(
        hidden_states, attention_mask=attention_mask, parameters=parameters
    )

    assert_with_pcc(torch_output, output, 0.9999)


@pytest.mark.parametrize("model_name", ["tiiuae/falcon-7b-instruct"])
@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize("sequence_length", [128])
def test_torch_functional_falcon_attention(model_name, batch_size, sequence_length):
    config = transformers.FalconConfig.from_pretrained(model_name)
    model = transformers.models.falcon.modeling_falcon.FalconAttention(config).eval()

    torch_hidden_states = (torch.rand(batch_size, sequence_length, config.hidden_size, dtype=torch.float32) * 2) - 1
    torch_attention_mask = torch.ones(1, sequence_length)
    torch_output, torch_present = model.forward(torch_hidden_states, alibi=None, attention_mask=torch_attention_mask)

    parameters = preprocess_model_parameters(
        initialize_model=lambda: model,
        convert_to_ttnn=lambda *_: False,
    )

    output, present = torch_functional_falcon_attention(
        torch_hidden_states, attention_mask=torch_attention_mask, parameters=parameters
    )

    assert_with_pcc(torch_output, output, 0.9999)
