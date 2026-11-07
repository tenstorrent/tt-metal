# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
import torch
import ttnn
from loguru import logger
from models.experimental.vadv2.reference.decoder import MultiheadAttention
from models.experimental.vadv2.tt.tt_mha import TtMultiheadAttention
from tests.ttnn.utils_for_testing import assert_with_pcc
from ttnn.model_preprocessing import preprocess_linear_bias, preprocess_linear_weight


def _linear_params(weight, bias, device):
    return SimpleNamespace(
        weight=ttnn.to_device(preprocess_linear_weight(weight, dtype=ttnn.bfloat16), device),
        bias=ttnn.to_device(preprocess_linear_bias(bias, dtype=ttnn.bfloat16), device),
    )


@pytest.mark.parametrize("device_params", [{"l1_small_size": 20 * 1024}], indirect=True)
def test_vadv2_mha_key_padding_mask(device, reset_seeds):
    # The motion-map cross-attention shape: one query per agent mode, a few map keys plus the
    # fixed pad key, which is never masked.
    num_queries, num_keys, embed_dims = 1800, 4, 256
    torch_model = MultiheadAttention(embed_dims, 8).eval()
    query = torch.randn(1, num_queries, embed_dims)
    key = torch.randn(num_keys, num_queries, embed_dims)
    key_padding_mask = torch.rand(num_queries, num_keys) < 0.6
    key_padding_mask[:, -1] = False
    with torch.no_grad():
        expected = torch_model(query, key, key, key_padding_mask=key_padding_mask)
        unmasked = torch_model(query, key, key)

    params = SimpleNamespace(
        in_proj=_linear_params(torch_model.attn.in_proj_weight, torch_model.attn.in_proj_bias, device),
        out_proj=_linear_params(torch_model.attn.out_proj.weight, torch_model.attn.out_proj.bias, device),
    )
    tt_model = TtMultiheadAttention(params, device, embed_dims=embed_dims, num_heads=8)

    def to_tt(t):
        return ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    tt_key = to_tt(key)
    output = tt_model(to_tt(query), tt_key, tt_key, key_padding_mask=to_tt(key_padding_mask.float()))
    output = ttnn.to_torch(output).float()

    _, msg = assert_with_pcc(expected, output, 0.99)
    logger.info(f"masked: {msg}")
    assert not torch.allclose(expected, unmasked, atol=0.1)
