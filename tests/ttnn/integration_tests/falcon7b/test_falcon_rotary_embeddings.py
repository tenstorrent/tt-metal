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
from models.utility_functions import comp_pcc, divup
import ttnn

torch.manual_seed(0)


def torch_functional_rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def ttnn_functional_rotate_half(x):
    return ttnn.experimental.rotate_half(x)


def torch_functional_apply_rotary_pos_emb(x, cos_cached, sin_cached, token_idx=None):
    seq_len = x.shape[-2]
    if token_idx is None:
        cos = cos_cached[:, :, :seq_len, ...]
        sin = sin_cached[:, :, :seq_len, ...]
    else:
        cos = cos_cached[:, :, token_idx : token_idx + 1, ...]
        sin = sin_cached[:, :, token_idx : token_idx + 1, ...]

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


@pytest.mark.parametrize("shape", [[1, 1, 128, 64], [1, 71, 128, 64]])
def test_functional_rotary_embeddings(device, shape):
    x = torch.randn(shape).bfloat16().float()
    x_device = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    x_device_rotated = ttnn_functional_rotate_half(x_device)
    tt_output = ttnn.to_torch(x_device_rotated)

    pt_output = torch_functional_rotate_half(x)
    assert torch.equal(tt_output, pt_output)


# [1, 71, 128, 64], [32, 1, 32, 64], [32, 71, 32, 64]
@pytest.mark.parametrize(
    "batch_size, num_kv_heads, sequence_length, head_dim",
    ((1, 1, 128, 64),),
)
@pytest.mark.parametrize("cache_size", [2048])
def test_ttnn_functional_apply_rotary_embeddings(
    batch_size, num_kv_heads, sequence_length, head_dim, cache_size, device
):
    input_shape = [batch_size, num_kv_heads, sequence_length, head_dim]
    sin_cos_shape = [batch_size, num_kv_heads, cache_size, head_dim]
    x = torch.randn(input_shape).bfloat16().float()
    cos_cached = torch.randn(sin_cos_shape).bfloat16().float()
    sin_cached = torch.randn(sin_cos_shape).bfloat16().float()
    pt_out = torch_functional_apply_rotary_pos_emb(x, cos_cached, sin_cached)

    device_input = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    device_cost = ttnn.from_torch(cos_cached, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    device_sint = ttnn.from_torch(sin_cached, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    device_output = ttnn.experimental.rotary_embedding(device_input, device_cost, device_sint)
    torch_device_output = ttnn.to_torch(device_output)

    p, o = comp_pcc(pt_out, torch_device_output)
    logger.info(o)
    assert p


@pytest.mark.parametrize("model_name", ["tiiuae/falcon-7b-instruct"])
@pytest.mark.parametrize(
    "input_shape",
    [
        (1, 1, 128, 64),
    ],
)
def test_torch_functional_falcon_generate_rotary_embeddings(model_name, input_shape, device):
    # TODO: test reseting of cossin cache
    batch, num_kv_heads, query_length, head_dim = input_shape
    config = transformers.FalconConfig.from_pretrained(model_name)
    model = transformers.models.falcon.modeling_falcon.FalconRotaryEmbedding(head_dim).eval()
    value_layer = torch.rand(batch, num_kv_heads, query_length, head_dim, dtype=torch.float32)

    parameters = preprocess_model_parameters(
        initialize_model=lambda: model,
        convert_to_ttnn=lambda *_: False,
    )
    torch_cos, torch_sin = model.forward(value_layer, seq_len=query_length)
    functional_cos, functional_sin = torch_functional_generate_sin_cos_rotary_embedding(
        query_length, head_dim, rotary_embedding_cache={}
    )
    ttnn_cos, ttnn_sin = ttnn.from_torch(functional_cos, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16), ttnn.from_torch(
        functional_sin, layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.bfloat16
    )
    ttnn_cos_pt, ttnn_sin_pt = ttnn.to_torch(ttnn_cos), ttnn.to_torch(ttnn_sin)

    assert_with_pcc(torch_cos, functional_cos, 0.9999)
    assert_with_pcc(torch_sin, functional_sin, 0.9999)
    assert_with_pcc(torch_cos, ttnn_cos_pt, 0.9999)
    assert_with_pcc(torch_sin, ttnn_sin_pt, 0.9999)
