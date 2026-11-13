# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""The motion queries' cross-attention to the map vectors, with padded keys masked out."""

from types import SimpleNamespace

import pytest
import torch
from loguru import logger

import ttnn
from models.experimental.vadv1.common import init_dummy_weights
from models.experimental.vadv1.reference.decoder import MultiheadAttention
from models.experimental.vadv1.tt.tt_mha import TtMultiheadAttention
from tests.ttnn.utils_for_testing import assert_with_pcc
from ttnn.model_preprocessing import preprocess_linear_bias, preprocess_linear_weight

# The motion-map decoder's shapes: one query per (agent, mode) and every map vector plus a pad slot.
NUM_QUERIES = 300 * 6
NUM_KEYS = 100 + 1
EMBED_DIMS = 256
NUM_HEADS = 8


def _linear_params(weight, bias, device):
    return SimpleNamespace(
        weight=ttnn.to_device(preprocess_linear_weight(weight, dtype=ttnn.bfloat16), device),
        bias=ttnn.to_device(preprocess_linear_bias(bias, dtype=ttnn.bfloat16), device),
    )


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
def test_vadv1_mha_key_padding_mask(device, reset_seeds):
    torch_model = MultiheadAttention(EMBED_DIMS, NUM_HEADS)
    torch_model = init_dummy_weights(torch_model)

    query = torch.randn(1, NUM_QUERIES, EMBED_DIMS)
    key = torch.randn(NUM_KEYS, NUM_QUERIES, EMBED_DIMS)
    query_pos = torch.randn(1, NUM_QUERIES, EMBED_DIMS)
    key_pos = torch.randn(NUM_KEYS, NUM_QUERIES, EMBED_DIMS)
    # Most keys padded, the last (the pad slot) never, so every query keeps one key.
    key_padding_mask = torch.rand(NUM_QUERIES, NUM_KEYS) < 0.7
    key_padding_mask[:, -1] = False

    with torch.no_grad():
        torch_output = torch_model(
            query, key, key, query_pos=query_pos, key_pos=key_pos, key_padding_mask=key_padding_mask
        )
        # The reference's hand-written attention must agree with torch's own.
        expected = torch_model.attn(query + query_pos, key + key_pos, key, key_padding_mask=key_padding_mask)[0] + query
    assert torch.allclose(torch_output, expected, atol=1e-4), "reference MHA differs from torch.nn.MultiheadAttention"

    params = SimpleNamespace(
        in_proj=_linear_params(torch_model.attn.in_proj_weight, torch_model.attn.in_proj_bias, device),
        out_proj=_linear_params(torch_model.attn.out_proj.weight, torch_model.attn.out_proj.bias, device),
    )
    tt_model = TtMultiheadAttention(params, device, embed_dims=EMBED_DIMS, num_heads=NUM_HEADS)

    def to_device(tensor):
        return ttnn.from_torch(tensor, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    tt_output = tt_model(
        to_device(query),
        to_device(key),
        to_device(key),
        query_pos=to_device(query_pos),
        key_pos=to_device(key_pos),
        key_padding_mask=to_device(key_padding_mask.float()),
    )

    _, message = assert_with_pcc(torch_output, ttnn.to_torch(tt_output).float(), 0.99)
    logger.info(message)
