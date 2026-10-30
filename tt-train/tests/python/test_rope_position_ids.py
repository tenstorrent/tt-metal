# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

import ttnn
import ttml

HEAD_COUNTS = (4, 2)  # Q heads, then K heads under GQA
MAX_SEQ_LEN = 256
STARTS = np.array([0, 5, 17, 100])


def _tensor(arr: np.ndarray, requires_grad: bool = False) -> ttml.autograd.Tensor:
    t = ttml.autograd.Tensor.from_numpy(arr.astype(np.float32), ttnn.Layout.TILE, ttnn.DataType.BFLOAT16)
    t.set_requires_grad(requires_grad)
    return t


def _to_np(t: ttml.autograd.Tensor) -> np.ndarray:
    return np.asarray(t.to_numpy(ttnn.DataType.FLOAT32), dtype=np.float32)


def _rope_and_grad(x_np, w_np, rope_fn):
    x = _tensor(x_np, requires_grad=True)
    out = rope_fn(x)
    loss = ttml.ops.unary.mean(ttml.ops.binary.mul(out, _tensor(w_np)))
    loss.backward(False)
    result = _to_np(out), _to_np(x.get_grad_tensor()) * x_np.size
    ttml.autograd.AutoContext.get_instance().reset_graph()
    return result


def _ids(starts: np.ndarray, seq_len: int) -> ttml.autograd.Tensor:
    ids_np = (starts[:, None] + np.arange(seq_len)[None, :]).astype(np.uint32)
    return ttml.autograd.Tensor.from_numpy(ids_np, ttnn.Layout.ROW_MAJOR, ttnn.DataType.UINT32)


def _assert_matches_scalar_rope(params, ids, starts, num_heads, seq_len, head_dim, rng):
    x_np = rng.standard_normal((len(starts), num_heads, seq_len, head_dim)).astype(np.float32)
    w_np = rng.standard_normal((len(starts), num_heads, seq_len, head_dim)).astype(np.float32)
    out, grad = _rope_and_grad(x_np, w_np, lambda x: ttml.ops.rope.rope(x, params, ids))

    for b, start in enumerate(starts):
        ref_out, ref_grad = _rope_and_grad(
            x_np[b : b + 1], w_np[b : b + 1], lambda x: ttml.ops.rope.rope(x, params, int(start))
        )
        np.testing.assert_allclose(out[b : b + 1], ref_out, atol=2e-2, rtol=2e-2)
        np.testing.assert_allclose(grad[b : b + 1], ref_grad, atol=5e-2, rtol=5e-2)


@pytest.mark.requires_device
@pytest.mark.parametrize("head_dim, seq_len", [(64, 1), (64, 32), (128, 1), (128, 5)])
def test_rope_position_ids_matches_per_row_scalar_rope(head_dim, seq_len):
    rng = np.random.default_rng(0)
    params = ttml.ops.rope.build_rope_params(MAX_SEQ_LEN, head_dim)
    shifted = STARTS + 1

    ids_tensor = _ids(STARTS, seq_len)
    ids = ttml.ops.rope.RotaryEmbeddingPositionIds(ids_tensor)
    # The second pass reads the cached rows.
    for num_heads in HEAD_COUNTS + HEAD_COUNTS:
        _assert_matches_scalar_rope(params, ids, STARTS, num_heads, seq_len, head_dim, rng)

    ids_tensor.set_value(_ids(shifted, seq_len).get_value())
    ids.update(ids_tensor)
    for num_heads in HEAD_COUNTS:
        _assert_matches_scalar_rope(params, ids, shifted, num_heads, seq_len, head_dim, rng)

    new_ids = ttml.ops.rope.RotaryEmbeddingPositionIds(_ids(shifted, seq_len))
    for num_heads in HEAD_COUNTS:
        _assert_matches_scalar_rope(params, new_ids, shifted, num_heads, seq_len, head_dim, rng)
