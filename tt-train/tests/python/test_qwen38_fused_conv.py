# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Correctness of the fused Gated DeltaNet conv1d + SiLU against torch.

1. ``ttml.ops.metal.depthwise_conv1d_k4`` in its three modes: causal, anti-causal,
   and causal fused with the SiLU derivative.
2. :func:`fused_causal_conv1d_silu`: forward q/k/v and the input gradient.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

import ttnn
import ttml

from ttml.models.qwen38.fused_conv import fused_causal_conv1d_silu


@pytest.fixture(autouse=True)
def reset_graph():
    yield
    ttml.autograd.AutoContext.get_instance().reset_graph()


def _pcc(a, b):
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    if np.allclose(a, b):
        return 1.0
    return float(np.corrcoef(a, b)[0, 1])


def _assert_pcc(got, ref, name, threshold):
    got = np.asarray(got)
    ref = np.asarray(ref)
    assert got.shape == ref.shape, f"{name}: shape {got.shape} != {ref.shape}"
    pcc = _pcc(got, ref)
    print(f"  PCC {name}: {pcc:.6f}")
    assert pcc >= threshold, f"{name}: PCC {pcc:.6f} < {threshold}"


def _tensor(x_np, requires_grad=False):
    t = ttml.autograd.Tensor.from_numpy(
        np.ascontiguousarray(x_np, dtype=np.float32),
        layout=ttnn.Layout.TILE,
        new_type=ttnn.DataType.BFLOAT16,
    )
    t.set_requires_grad(requires_grad)
    return t


def _to_numpy(ttnn_tensor):
    return ttml.autograd.create_tensor(ttnn_tensor, requires_grad=False).to_numpy()


def _causal_conv(x, taps):
    """``u[t] = sum_j taps[j] * x[t + j - 3]``, zero before the sequence start. x: [T, C]."""
    seq = x.shape[0]
    padded = torch.nn.functional.pad(x, (0, 0, 3, 0))
    return sum(taps[j] * padded[j : j + seq] for j in range(4))


def _inputs(seq, channels, seed):
    rng = np.random.RandomState(seed)
    x = rng.randn(seq, channels).astype(np.float32)
    taps = [(rng.randn(channels) * 0.5).astype(np.float32) for _ in range(4)]
    return x, taps


@pytest.mark.requires_device
@pytest.mark.parametrize("seq,channels", [(64, 256), (256, 2560)])
def test_depthwise_conv1d_k4_vs_torch(seq, channels):
    x_np, taps_np = _inputs(seq, channels, seed=7)
    grad_np = np.random.RandomState(8).randn(seq, channels).astype(np.float32)

    x_rm = ttnn.untilize(_tensor(x_np[None, None]).get_value())
    taps = [_tensor(t.reshape(1, 1, 1, channels)).get_value() for t in taps_np]
    grad = _tensor(grad_np[None, None]).get_value()

    x_t = torch.tensor(x_np)
    taps_t = [torch.tensor(t) for t in taps_np]
    u_ref = _causal_conv(x_t, taps_t)
    # The anti-causal conv is the causal one run on the time-reversed sequence.
    anti_ref = _causal_conv(x_t.flip(0), taps_t).flip(0)
    sig = torch.sigmoid(u_ref)
    silu_grad_ref = torch.tensor(grad_np) * sig * (1 + u_ref * (1 - sig))

    causal = _to_numpy(ttml.ops.metal.depthwise_conv1d_k4(x_rm, *taps))[0, 0]
    anti = _to_numpy(ttml.ops.metal.depthwise_conv1d_k4(x_rm, *taps, anti_causal=True))[0, 0]
    fused = _to_numpy(ttml.ops.metal.depthwise_conv1d_k4(x_rm, *taps, silu_grad=grad))[0, 0]

    _assert_pcc(causal, u_ref.numpy(), "causal", 0.999)
    _assert_pcc(anti, anti_ref.numpy(), "anti-causal", 0.999)
    _assert_pcc(fused, silu_grad_ref.numpy(), "silu grad", 0.995)


@pytest.mark.requires_device
@pytest.mark.parametrize(
    "seq,widths",
    [
        (64, (64, 64, 128)),
        (256, (512, 512, 1536)),  # Qwen3.8 per-chip widths at TP=4
    ],
)
def test_fused_causal_conv1d_silu_vs_torch(seq, widths):
    channels = sum(widths)
    x_np, taps_np = _inputs(seq, channels, seed=11)

    qkv = _tensor(x_np[None, None], requires_grad=True)
    taps = [_tensor(t.reshape(1, 1, 1, channels)) for t in taps_np]
    q, k, v = fused_causal_conv1d_silu(qkv, taps, widths)
    out_np = np.concatenate([t.to_numpy() for t in (q, k, v)], axis=-1)[0, 0]

    w_np = np.random.RandomState(99).randn(seq, channels).astype(np.float32)
    bounds = np.cumsum((0,) + widths)
    loss = None
    for t, lo, hi in zip((q, k, v), bounds[:-1], bounds[1:]):
        w = _tensor(w_np[None, None, :, lo:hi])
        term = ttml.ops.unary.mean(ttml.ops.binary.mul(t, w))
        loss = term if loss is None else ttml.ops.binary.add(loss, term)
    loss.backward(False)

    x_t = torch.tensor(x_np, requires_grad=True)
    ref = torch.nn.functional.silu(_causal_conv(x_t, [torch.tensor(t) for t in taps_np]))
    loss_t = sum((ref[:, lo:hi] * torch.tensor(w_np[:, lo:hi])).mean() for lo, hi in zip(bounds[:-1], bounds[1:]))
    loss_t.backward()

    _assert_pcc(out_np, ref.detach().numpy(), "forward", 0.999)
    _assert_pcc(qkv.get_grad_tensor().to_numpy()[0, 0], x_t.grad.numpy(), "grad qkv", 0.99)
