# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Refinement 4 — causal masking (mask_mode=causal) tests.

Validates on-device triangular-mask generation + KV-loop truncation for
`is_causal=True` (no mask tensor). Covers self-attention across MHA/GQA/MQA,
multi-batch, multi-head, dtype, and explicit scale; the {causal, cross}
EXCLUSION; and the is_causal+attn_mask mutual-exclusion ValueError.

The device comes from the dir conftest (module-scoped fixture) — never opened
here. Do NOT delete this file.
"""

import pytest
import torch

import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc
from ttnn.operations._op_contract import ExcludedCell
from ttnn.operations.scaled_dot_product_attention import scaled_dot_product_attention

pytestmark = pytest.mark.use_module_device

PCC = {ttnn.bfloat16: 0.99, ttnn.float32: 0.999, ttnn.bfloat8_b: 0.99}


def _reference(Q, K, V, *, scale=None):
    Qf, Kf, Vf = Q.float(), K.float(), V.float()
    H_q, H_kv = Qf.shape[1], Kf.shape[1]
    if H_q != H_kv:
        r = H_q // H_kv
        Kf = Kf.repeat_interleave(r, dim=1)
        Vf = Vf.repeat_interleave(r, dim=1)
    return torch.nn.functional.scaled_dot_product_attention(Qf, Kf, Vf, is_causal=True, scale=scale)


# (Q_shape, K_shape, V_shape) — self-attention (S_q == S_kv) only; MHA/GQA/MQA,
# multi-batch, multi-head, single-tile → multi-chunk sequences.
SHAPES = [
    ((1, 1, 32, 32), (1, 1, 32, 32), (1, 1, 32, 32)),  # single tile (diagonal only)
    ((1, 1, 128, 64), (1, 1, 128, 64), (1, 1, 128, 64)),  # multi-chunk self
    ((1, 4, 256, 64), (1, 4, 256, 64), (1, 4, 256, 64)),  # multi-head, longer
    ((2, 4, 128, 64), (2, 4, 128, 64), (2, 4, 128, 64)),  # multi-batch
    ((1, 8, 128, 64), (1, 2, 128, 64), (1, 2, 128, 64)),  # GQA 4:1
    ((1, 8, 128, 64), (1, 1, 128, 64), (1, 1, 128, 64)),  # MQA
    ((1, 1, 512, 64), (1, 1, 512, 64), (1, 1, 512, 64)),  # many KV chunks (truncation)
]


@pytest.mark.parametrize("shapes", SHAPES)
@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32, ttnn.bfloat8_b])
@pytest.mark.parametrize("scale_mode", ["auto", "explicit"])
def test_causal(device, shapes, dtype, scale_mode):
    torch.manual_seed(42)
    q_shape, k_shape, v_shape = shapes
    torch_dtype = torch.float32 if dtype == ttnn.float32 else torch.bfloat16
    Q = torch.randn(q_shape, dtype=torch_dtype)
    K = torch.randn(k_shape, dtype=torch_dtype)
    V = torch.randn(v_shape, dtype=torch_dtype)
    scale = 0.125 if scale_mode == "explicit" else None

    expected = _reference(Q, K, V, scale=scale)

    to = lambda t: ttnn.from_torch(
        t, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    out = scaled_dot_product_attention(to(Q), to(K), to(V), is_causal=True, scale=scale)
    got = ttnn.to_torch(out).to(torch.float32)
    assert list(got.shape) == list(q_shape)
    assert_with_pcc(expected, got, PCC[dtype])


def test_causal_vs_custom_equivalence(device):
    """Native causal must match the equivalent additive upper-triangular mask."""
    torch.manual_seed(0)
    shape = (1, 2, 256, 64)
    Q = torch.randn(shape, dtype=torch.bfloat16)
    K = torch.randn(shape, dtype=torch.bfloat16)
    V = torch.randn(shape, dtype=torch.bfloat16)
    B, _H, S, _D = shape
    mask = torch.zeros(B, 1, S, S, dtype=torch.bfloat16)
    mask.masked_fill_(torch.triu(torch.ones(S, S, dtype=torch.bool), diagonal=1), float("-inf"))

    to = lambda t: ttnn.from_torch(
        t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    out_causal = ttnn.to_torch(scaled_dot_product_attention(to(Q), to(K), to(V), is_causal=True)).float()
    out_custom = ttnn.to_torch(scaled_dot_product_attention(to(Q), to(K), to(V), attn_mask=to(mask))).float()
    assert_with_pcc(out_custom, out_causal, 0.999)


def test_causal_cross_excluded(device, expect_error):
    """{mask_mode: causal, attention_kind: cross} (S_q != S_kv) is refused."""
    to = lambda t: ttnn.from_torch(
        t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    Q = torch.randn(1, 4, 64, 64)
    K = torch.randn(1, 4, 128, 64)
    V = torch.randn(1, 4, 128, 64)
    with expect_error(ExcludedCell, "unsupported combination"):
        scaled_dot_product_attention(to(Q), to(K), to(V), is_causal=True)


def test_causal_and_mask_mutually_exclusive(device, expect_error):
    """is_causal=True with attn_mask must raise ValueError."""
    to = lambda t: ttnn.from_torch(
        t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    Q = torch.randn(1, 1, 128, 64)
    K = torch.randn(1, 1, 128, 64)
    V = torch.randn(1, 1, 128, 64)
    M = torch.zeros(1, 1, 128, 128)
    with expect_error(ValueError, "mutually exclusive"):
        scaled_dot_product_attention(to(Q), to(K), to(V), is_causal=True, attn_mask=to(M))
