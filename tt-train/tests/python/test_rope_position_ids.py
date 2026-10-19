# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

import ttnn
import ttml

HEADS = 4
HEAD_DIM = 64
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


@pytest.mark.requires_device
@pytest.mark.parametrize("seq_len", [1, 32])
def test_rope_position_ids_matches_per_row_scalar_rope(seq_len):
    rng = np.random.default_rng(0)
    params = ttml.ops.rope.build_rope_params(MAX_SEQ_LEN, HEAD_DIM)
    batch = len(STARTS)
    x_np = rng.standard_normal((batch, HEADS, seq_len, HEAD_DIM)).astype(np.float32)
    w_np = rng.standard_normal((batch, HEADS, seq_len, HEAD_DIM)).astype(np.float32)

    ids_np = (STARTS[:, None] + np.arange(seq_len)[None, :]).astype(np.uint32)
    ids = ttml.autograd.Tensor.from_numpy(ids_np, ttnn.Layout.ROW_MAJOR, ttnn.DataType.UINT32)
    out, grad = _rope_and_grad(x_np, w_np, lambda x: ttml.ops.rope.rope(x, params, ids))

    for b, start in enumerate(STARTS):
        ref_out, ref_grad = _rope_and_grad(
            x_np[b : b + 1], w_np[b : b + 1], lambda x: ttml.ops.rope.rope(x, params, int(start))
        )
        np.testing.assert_allclose(out[b : b + 1], ref_out, atol=2e-2, rtol=2e-2)
        np.testing.assert_allclose(grad[b : b + 1], ref_grad, atol=5e-2, rtol=5e-2)
