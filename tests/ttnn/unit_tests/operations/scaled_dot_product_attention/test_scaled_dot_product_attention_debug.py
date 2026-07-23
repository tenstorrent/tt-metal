# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
# Deterministic debug tests for scaled_dot_product_attention. DO NOT DELETE.

import math

import pytest
import torch

import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc
from ttnn.operations.scaled_dot_product_attention import scaled_dot_product_attention


def _ref(Q, K, V, attn_mask=None, scale=None):
    return torch.nn.functional.scaled_dot_product_attention(
        Q.float(),
        K.float(),
        V.float(),
        attn_mask=attn_mask.float() if attn_mask is not None else None,
        scale=scale,
    )


def _to_dev(t, device):
    return ttnn.from_torch(
        t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )


@pytest.mark.parametrize(
    "shape",
    [
        (1, 1, 32, 32),  # single tile, single KV block
        (1, 1, 128, 64),  # multi-tile self, 1 chunk
        (1, 1, 256, 64),  # multi-KV-chunk (Skvt=8 -> 2 chunks of 4)
        (1, 8, 256, 64),  # multi-head
    ],
)
def test_selfattn_none(device, shape):
    torch.manual_seed(0)
    B, H, S, D = shape
    Q = torch.randn(shape, dtype=torch.bfloat16)
    K = torch.randn(shape, dtype=torch.bfloat16)
    V = torch.randn(shape, dtype=torch.bfloat16)
    expected = _ref(Q, K, V)
    out = ttnn.to_torch(
        scaled_dot_product_attention(_to_dev(Q, device), _to_dev(K, device), _to_dev(V, device))
    ).float()
    assert list(out.shape) == list(shape)
    assert_with_pcc(expected, out, 0.99)
