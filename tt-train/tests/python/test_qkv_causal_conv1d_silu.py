# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""``ttml.ops.conv.qkv_causal_conv1d_silu`` against a numpy reference.

One small case per mode; the full case matrix lives in the C++ test. Activations are ``[1, 1, T, Q+K+V]`` as the
model holds them, history is the three rows before the sequence, taps are ``[1, 1, 1, Q+K+V]``.
"""

import numpy as np
import pytest
import torch

import ttnn
import ttml

pytestmark = pytest.mark.requires_device

BF16 = ttnn.DataType.BFLOAT16
FP32 = ttnn.DataType.FLOAT32
# Per element, of max|expected|. The backward's floor is the bf16 rounding of its output, ~0.4%; the forward rounds
# the preactivation to bf16 before the sigmoid, so its outputs carry about two bf16 ulps of the preactivation.
REL_OF_MAX = 1e-2
FORWARD_REL_OF_MAX = 2e-2
MEAN_REL_OF_MAX = 1e-3


@pytest.fixture(autouse=True)
def _reset_graph():
    yield
    ttml.autograd.AutoContext.get_instance().reset_graph()


def _bf16(array):
    return torch.from_numpy(array).to(torch.bfloat16).to(torch.float32).numpy()


def _assert_close(actual, expected, what, rel_of_max=REL_OF_MAX):
    scale = np.abs(expected).max()
    err = np.abs(actual - expected)
    assert err.max() <= rel_of_max * scale, f"{what}: max abs err {err.max()} > {rel_of_max} of max {scale}"
    assert (
        err.mean() <= MEAN_REL_OF_MAX * scale
    ), f"{what}: mean abs err {err.mean()} > {MEAN_REL_OF_MAX} of max {scale}"


def _to_ttml(array, layout, requires_grad):
    tensor = ttml.autograd.Tensor.from_numpy(
        np.ascontiguousarray(array, dtype=np.float32), layout=layout, new_type=BF16
    )
    tensor.set_requires_grad(requires_grad)
    return tensor


def _reference(x, h, taps, grad_y):
    """y = silu(sum_j taps[j] * window[j:j+T]), window = [h; x]; returns y, grad_input, grad_taps."""
    T = x.shape[2]
    window = np.concatenate([h, x], axis=2).astype(np.float64)
    preact = sum(taps[j].astype(np.float64) * window[:, :, j : j + T, :] for j in range(4))
    sig = 1.0 / (1.0 + np.exp(-preact))
    y = preact * sig
    grad_preact = grad_y.astype(np.float64) * sig * (1.0 + preact * (1.0 - sig))
    grad_window = np.zeros_like(window)
    grad_taps = []
    for j in range(4):
        grad_window[:, :, j : j + T, :] += taps[j] * grad_preact
        grad_taps.append((grad_preact * window[:, :, j : j + T, :]).sum(axis=(0, 1, 2), keepdims=True))
    return y, grad_window[:, :, 3 : 3 + T, :], grad_taps


@pytest.mark.parametrize("taps_trainable", [False, True])
def test_qkv_causal_conv1d_silu(taps_trainable):
    T, Q, K, V = 64, 96, 64, 32
    C = Q + K + V
    rng = np.random.default_rng(0)
    x = _bf16(rng.uniform(-1, 1, (1, 1, T, C)).astype(np.float32))
    h = _bf16(rng.uniform(-1, 1, (1, 1, 3, C)).astype(np.float32))
    taps = [_bf16(rng.uniform(-1, 1, (1, 1, 1, C)).astype(np.float32)) for _ in range(4)]
    y, _, _ = _reference(x, h, taps, np.zeros_like(x))
    # loss = mean(q) + mean(k) + mean(v): every output's node runs, and d loss / d out = 1 / out.numel()
    grad_y = np.zeros_like(x)
    for lo, hi in ((0, Q), (Q, Q + K), (Q + K, C)):
        grad_y[..., lo:hi] = 1.0 / (T * (hi - lo))
    _, grad_input, grad_taps = _reference(x, h, taps, grad_y)

    x_ttml = _to_ttml(x, ttnn.Layout.TILE, True)
    h_ttml = _to_ttml(h, ttnn.Layout.ROW_MAJOR, False)
    taps_ttml = [_to_ttml(t, ttnn.Layout.TILE, taps_trainable) for t in taps]
    q, k, v = ttml.ops.conv.qkv_causal_conv1d_silu(x_ttml, h_ttml.get_value(), *taps_ttml, Q, K, V)

    assert q.shape() == [1, 1, T, Q]
    _assert_close(q.to_numpy(FP32), y[..., :Q], "q", FORWARD_REL_OF_MAX)
    _assert_close(k.to_numpy(FP32), y[..., Q : Q + K], "k", FORWARD_REL_OF_MAX)
    _assert_close(v.to_numpy(FP32), y[..., Q + K :], "v", FORWARD_REL_OF_MAX)

    loss = ttml.ops.unary.mean(q) + ttml.ops.unary.mean(k) + ttml.ops.unary.mean(v)
    loss.backward(False)

    assert x_ttml.get_grad_tensor().shape() == [1, 1, T, C]
    _assert_close(x_ttml.get_grad_tensor().to_numpy(FP32), grad_input, "grad_input")
    for j, tap in enumerate(taps_ttml):
        if taps_trainable:
            _assert_close(tap.get_grad_tensor().to_numpy(FP32).reshape(1, 1, 1, C), grad_taps[j], f"grad_tap{j}")
        else:
            assert not tap.is_grad_initialized(), "frozen tap received a gradient"
