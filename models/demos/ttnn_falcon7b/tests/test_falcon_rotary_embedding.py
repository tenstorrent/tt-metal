# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
from torch.nn import functional as F
import transformers
import pytest
from loguru import logger

import ttnn
from ttnn.model_preprocessing import preprocess_model_parameters
from tests.ttnn.utils_for_testing import assert_with_pcc
from models.utility_functions import comp_pcc, divup
import tt_lib as ttl
import ttnn

from models.demos.ttnn_falcon7b.tt.falcon_rotary_embedding import TtFalconRotaryEmbedding
from models.demos.ttnn_falcon7b.tt.model_config import get_model_config, get_tt_cache_path
from models.demos.ttnn_falcon7b.tt.common import create_custom_preprocessor

torch.manual_seed(0)


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


@pytest.mark.parametrize("model_name", ["tiiuae/falcon-7b-instruct"])
@pytest.mark.parametrize(
    "input_shape",
    [
        (1, 1, 128, 64),
    ],
)
@pytest.mark.parametrize("model_config_str", ("BFLOAT16-DRAM", "BFLOAT16-L1"))
def test_falcon_rotary_embedding(model_name, input_shape, model_config_str, device):
    model_config = get_model_config(model_config_str)

    batch, num_kv_heads, query_length, head_dim = input_shape
    config = transformers.FalconConfig.from_pretrained(model_name)
    model = transformers.models.falcon.modeling_falcon.FalconRotaryEmbedding(config.head_dim).eval()
    value_layer = torch.rand(batch, num_kv_heads, query_length, head_dim, dtype=torch.float32)
    query_layer = torch.rand(batch, config.num_attention_heads, query_length, head_dim, dtype=torch.float32)
    key_layer = torch.rand(batch, num_kv_heads, query_length, head_dim, dtype=torch.float32)

    tt_cache_path = get_tt_cache_path(model_name)
    custom_preprocessor = (create_custom_preprocessor(model_config, tt_cache_path=tt_cache_path, device=device),)
    parameters = preprocess_model_parameters(
        initialize_model=lambda: model,
        custom_preprocessor=custom_preprocessor,
        device=device,
    )
    torch_cos, torch_sin = model.forward(value_layer, seq_len=query_length)
    q_embed, k_embed = transformers.models.falcon.modeling_falcon.apply_rotary_pos_emb(
        query_layer, key_layer, torch_cos, torch_sin, None
    )

    tt_model = TtFalconRotaryEmbedding(parameters, model_config=model_config)

    tt_query_layer = ttnn.from_torch(
        query_layer, device=device, layout=ttnn.TILE_LAYOUT, dtype=model_config["DEFAULT_DTYPE"]
    )
    tt_key_layer = ttnn.from_torch(
        key_layer, device=device, layout=ttnn.TILE_LAYOUT, dtype=model_config["DEFAULT_DTYPE"]
    )
    tt_q_embed = tt_model.forward(tt_query_layer)
    tt_k_embed = tt_model.forward(tt_key_layer)

    assert_with_pcc(q_embed, ttnn.to_torch(tt_q_embed), 0.9999)
    assert_with_pcc(k_embed, ttnn.to_torch(tt_k_embed), 0.9999)
