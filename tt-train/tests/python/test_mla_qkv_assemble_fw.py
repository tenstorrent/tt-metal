# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Forward parity coverage for the DeepSeek MLA QKV assemble op."""

from __future__ import annotations

import numpy as np
import pytest

import ttnn
import ttml
from ttml.models.deepseek.autograd_ops import autograd_concat, autograd_slice, split_heads


SEED = 2026


def _make_tensor(array: np.ndarray) -> ttml.autograd.Tensor:
    return ttml.autograd.Tensor.from_numpy(array, layout=ttnn.Layout.TILE, new_type=ttnn.DataType.BFLOAT16)


def _make_inputs(batch: int, seq_len: int, n_heads: int, qk_nope_dim: int, qk_rope_dim: int, v_dim: int):
    rng = np.random.default_rng(SEED)
    qk_head_dim = qk_nope_dim + qk_rope_dim
    q_pre = rng.standard_normal((batch, 1, seq_len, n_heads * qk_head_dim), dtype=np.float32)
    kv_up = rng.standard_normal((batch, 1, seq_len, n_heads * (qk_nope_dim + v_dim)), dtype=np.float32)
    k_pe = rng.standard_normal((batch, 1, seq_len, qk_rope_dim), dtype=np.float32)
    return _make_tensor(q_pre), _make_tensor(kv_up), _make_tensor(k_pe)


def _reference_assemble(q_pre, kv_up, k_pe, n_heads: int, qk_nope_dim: int, qk_rope_dim: int, v_dim: int):
    B, _, S, _ = list(q_pre.get_value().shape)
    qk_head_dim = qk_nope_dim + qk_rope_dim

    q = split_heads(q_pre, n_heads)
    kv_up = split_heads(kv_up, n_heads)
    k_nope = autograd_slice(kv_up, [0, 0, 0, 0], [B, n_heads, S, qk_nope_dim])
    v = autograd_slice(kv_up, [0, 0, 0, qk_nope_dim], [B, n_heads, S, qk_nope_dim + v_dim])
    k_pe = autograd_concat([k_pe] * n_heads, dim=1)
    k = autograd_concat([k_nope, k_pe], dim=3)

    assert list(q.get_value().shape) == [B, n_heads, S, qk_head_dim]
    assert list(k.get_value().shape) == [B, n_heads, S, qk_head_dim]
    assert list(v.get_value().shape) == [B, n_heads, S, v_dim]
    return q, k, v


def _to_numpy(tensor: ttml.autograd.Tensor) -> np.ndarray:
    return np.asarray(tensor.to_numpy(ttnn.DataType.FLOAT32), dtype=np.float32)


@pytest.fixture(autouse=True)
def reset_graph():
    yield
    ttml.autograd.AutoContext.get_instance().reset_graph()


def test_mla_qkv_assemble_fw_matches_reference_path():
    ctx = ttml.autograd.AutoContext.get_instance()
    ctx.open_device()

    try:
        device = ctx.get_device()
        ctx.set_gradient_mode(ttml.autograd.GradMode.DISABLED)

        batch = 2
        seq_len = 64
        n_heads = 2
        qk_nope_dim = 32
        qk_rope_dim = 32
        v_dim = 32

        q_pre, kv_up, k_pe = _make_inputs(batch, seq_len, n_heads, qk_nope_dim, qk_rope_dim, v_dim)
        q_ref, k_ref, v_ref = _reference_assemble(q_pre, kv_up, k_pe, n_heads, qk_nope_dim, qk_rope_dim, v_dim)
        ttnn.synchronize_device(device)
        expected = [_to_numpy(tensor) for tensor in (q_ref, k_ref, v_ref)]
        ctx.reset_graph()

        q_pre, kv_up, k_pe = _make_inputs(batch, seq_len, n_heads, qk_nope_dim, qk_rope_dim, v_dim)
        q, k, v = ttml.ops.mla.qkv_assemble_fw(q_pre, kv_up, k_pe, n_heads, qk_nope_dim, qk_rope_dim, v_dim)
        ttnn.synchronize_device(device)
        actual = [_to_numpy(tensor) for tensor in (q, k, v)]

        for name, actual_tensor, expected_tensor in zip(("q", "k", "v"), actual, expected):
            np.testing.assert_allclose(actual_tensor, expected_tensor, atol=0.0, rtol=0.0, err_msg=name)
    finally:
        ctx.close_device()
