# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Forward/backward parity for the ttml ops added for Qwen3.8.

``sum_over_dim``, ``cumsum``, ``softplus``, ``l2_norm``, ``transpose`` and
``shift_along_dim`` are new ``ttml::ops`` entries whose backwards are written by
hand in C++, so each is pinned against torch autograd on both the value and the
gradient.

Tolerances are loose because ttml tensors are stored and computed in bfloat16
(``PreferredPrecision::HALF``), so agreement with fp32 torch is limited by the
8-bit mantissa, not by the op.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

import ttnn
import ttml

from ttml.models.qwen38.autograd_ops import causal_mask, identity, tril_ones

U = ttml.ops.unary


@pytest.fixture(autouse=True)
def reset_graph():
    yield
    ttml.autograd.AutoContext.get_instance().reset_graph()


def _leaf(x_np):
    t = ttml.autograd.Tensor.from_numpy(
        np.ascontiguousarray(x_np, dtype=np.float32),
        layout=ttnn.Layout.TILE,
        new_type=ttnn.DataType.BFLOAT16,
    )
    t.set_requires_grad(True)
    return t


def _run(x_np, ttml_fn, torch_fn):
    """Push the same random-projection loss through both stacks.

    A plain ``sum`` loss would make many gradient bugs invisible, since the
    upstream gradient would be all ones.
    """
    leaf = _leaf(x_np)
    out = ttml_fn(leaf)
    out_np = out.to_numpy()

    w_np = np.random.RandomState(0).randn(*out_np.shape).astype(np.float32)
    w = ttml.autograd.Tensor.from_numpy(w_np, layout=ttnn.Layout.TILE, new_type=ttnn.DataType.BFLOAT16)
    ttml.ops.unary.mean(ttml.ops.binary.mul(out, w)).backward(False)

    xt = torch.tensor(x_np, dtype=torch.float32, requires_grad=True)
    ref = torch_fn(xt)
    (ref * torch.tensor(w_np)).mean().backward()

    return (out_np, ref.detach().numpy()), (leaf.get_grad_tensor().to_numpy(), xt.grad.numpy())


def _assert_close(pair, name, atol, rtol):
    got, ref = pair
    assert got.shape == ref.shape, f"{name}: shape {got.shape} != {ref.shape}"
    np.testing.assert_allclose(got, ref, atol=atol, rtol=rtol, err_msg=name)


@pytest.mark.requires_device
def test_transpose():
    x = np.random.randn(2, 3, 64, 32).astype(np.float32)
    fwd, bwd = _run(x, lambda t: U.transpose(t, 1, 2), lambda t: t.transpose(1, 2))
    _assert_close(fwd, "transpose forward", 1e-2, 1e-2)
    _assert_close(bwd, "transpose backward", 1e-2, 1e-2)


@pytest.mark.requires_device
@pytest.mark.parametrize("dim", [2, 3])
def test_sum_over_dim(dim):
    x = np.random.randn(1, 2, 64, 32).astype(np.float32)
    fwd, bwd = _run(x, lambda t: U.sum_over_dim(t, dim), lambda t: t.sum(dim=dim, keepdim=True))
    # A 32/64-deep bf16 reduction; compare on the scale of the result.
    _assert_close(fwd, f"sum_over_dim({dim}) forward", 2e-1, 2e-2)
    _assert_close(bwd, f"sum_over_dim({dim}) backward", 2e-2, 2e-2)


@pytest.mark.requires_device
@pytest.mark.parametrize("dim", [2, 3])
def test_cumsum(dim):
    x = np.random.randn(2, 1, 64, 32).astype(np.float32)
    fwd, bwd = _run(x, lambda t: U.cumsum(t, dim), lambda t: t.cumsum(dim))
    _assert_close(fwd, f"cumsum({dim}) forward", 2e-1, 2e-2)
    # The backward is a reverse cumsum (ttnn.cumsum(reverse_order=True)).
    _assert_close(bwd, f"cumsum({dim}) backward", 2e-1, 2e-2)


@pytest.mark.requires_device
def test_softplus():
    # Spread across the threshold so both the log1p and linear branches are hit.
    x = (np.random.randn(1, 1, 64, 64) * 3.0).astype(np.float32)
    fwd, bwd = _run(x, U.softplus, torch.nn.functional.softplus)
    _assert_close(fwd, "softplus forward", 3e-2, 3e-2)
    _assert_close(bwd, "softplus backward", 3e-2, 3e-2)


@pytest.mark.requires_device
def test_l2_norm():
    x = np.random.randn(1, 4, 64, 128).astype(np.float32)

    def torch_l2(t):
        return t * torch.rsqrt((t * t).sum(dim=-1, keepdim=True) + 1e-6)

    fwd, bwd = _run(x, U.l2_norm, torch_l2)
    _assert_close(fwd, "l2_norm forward", 2e-2, 2e-2)
    _assert_close(bwd, "l2_norm backward", 2e-2, 2e-2)


@pytest.mark.requires_device
@pytest.mark.parametrize("shift", [0, 1, 2, 3])
def test_shift_along_dim(shift):
    x = np.random.randn(1, 1, 64, 32).astype(np.float32)

    def torch_shift(t):
        if shift == 0:
            return t
        return torch.cat([torch.zeros_like(t[:, :, :shift]), t[:, :, :-shift]], dim=2)

    fwd, bwd = _run(x, lambda t: U.shift_along_dim(t, 2, shift), torch_shift)
    _assert_close(fwd, f"shift_along_dim({shift}) forward", 1e-2, 1e-2)
    _assert_close(bwd, f"shift_along_dim({shift}) backward", 1e-2, 1e-2)


@pytest.mark.requires_device
def test_causal_conv1d_from_shifts():
    """A kernel-4 causal depthwise conv composed of shift_along_dim + mul + add.

    This is how the DeltaNet's conv1d is built, so it is worth pinning directly
    against ``F.conv1d`` rather than only through the whole block.
    """
    kernel = 4
    channels = 32
    seq = 64
    x = np.random.randn(1, 1, seq, channels).astype(np.float32)
    w = np.random.randn(channels, kernel).astype(np.float32)

    leaf = _leaf(x)
    acc = None
    for j in range(kernel):
        w_j = ttml.autograd.Tensor.from_numpy(
            np.ascontiguousarray(w[:, j].reshape(1, 1, 1, channels)),
            ttnn.Layout.TILE,
            ttnn.DataType.BFLOAT16,
        )
        term = ttml.ops.binary.mul(U.shift_along_dim(leaf, 2, kernel - 1 - j), w_j)
        acc = term if acc is None else ttml.ops.binary.add(acc, term)
    got = acc.to_numpy()

    xt = torch.tensor(x).squeeze(1).transpose(1, 2)  # [1, C, T]
    ref = torch.nn.functional.conv1d(
        torch.nn.functional.pad(xt, (kernel - 1, 0)),
        torch.tensor(w).unsqueeze(1),  # [C, 1, K]
        groups=channels,
    )
    ref = ref.transpose(1, 2).unsqueeze(1).numpy()

    np.testing.assert_allclose(got, ref, atol=5e-2, rtol=5e-2, err_msg="causal conv1d")


@pytest.mark.requires_device
def test_mask_constants_match_numpy():
    """The constant masks the delta rule multiplies by must match the reference triangles."""
    chunk = 32
    batch = (2, 3)

    ref_strict = np.tril(np.ones((chunk, chunk), np.float32), k=-1)
    ref_diag = np.tril(np.ones((chunk, chunk), np.float32), k=0)

    for got, ref, name in [
        (causal_mask(chunk, batch, diagonal=0).to_numpy(), ref_strict, "keep(diagonal=0)"),
        (causal_mask(chunk, batch, diagonal=1).to_numpy(), ref_diag, "keep(diagonal=1)"),
        (tril_ones(chunk, batch).to_numpy(), ref_diag, "tril_ones"),
        (identity(chunk, batch).to_numpy(), np.eye(chunk, dtype=np.float32), "identity"),
    ]:
        assert got.shape == batch + (chunk, chunk), f"{name}: {got.shape}"
        for b0 in range(batch[0]):
            for b1 in range(batch[1]):
                np.testing.assert_array_equal(got[b0, b1], ref, err_msg=name)
