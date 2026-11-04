# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Device micro-tests for the primitives the DFlash drafter relies on (TT_PORT_PLAN D2).

If ``test_dflash_drafter_parity`` fails, these say which building block is wrong:

* ``rope_half``: HF half-rotation RoPE from host cos/sin, forward and backward.
* head layout: ``grouped_heads_creation`` / ``heads_fusion`` match HF ``view``/``transpose``.
* ``dflash_kl_loss``: value and gradient vs an fp64 reference.
* frozen embedding gather: values, and no gradient/optimizer update for frozen weights.
* ``AdamWFullPrecision`` + ``clip_grad_norm``: two steps vs the torch AdamW formula.
"""

from __future__ import annotations

import numpy as np
import pytest
import ttnn

import ttml
from ttml.models.dflash.host import rope_cos_sin, signed_sin
from ttml.models.dflash.model import dflash_kl_loss, rope_half

pytestmark = pytest.mark.requires_device
RNG = np.random.default_rng(7)


@pytest.fixture
def ctx():
    c = ttml.autograd.AutoContext.get_instance()
    c.get_device()
    c.set_gradient_mode(ttml.autograd.GradMode.ENABLED)
    yield c
    c.reset_graph()
    c.close_device()


def _bf16(x: np.ndarray) -> np.ndarray:
    """Round-trip through bf16 so references see exactly what the device sees."""
    x = np.asarray(x, np.float32)
    return (x.view(np.uint32) + 0x7FFF + ((x.view(np.uint32) >> 16) & 1) & 0xFFFF0000).view(np.float32)


def _tensor(x: np.ndarray, requires_grad: bool, dtype=ttnn.DataType.BFLOAT16) -> ttml.autograd.Tensor:
    t = ttml.autograd.Tensor.from_numpy(np.ascontiguousarray(x, np.float32), layout=ttnn.Layout.TILE, new_type=dtype)
    t.set_requires_grad(requires_grad)
    return t


def _np(t) -> np.ndarray:
    return np.asarray(t.to_numpy(ttnn.DataType.FLOAT32), np.float32)


def _grad(t) -> np.ndarray:
    return _np(ttml.autograd.Tensor(t.get_grad(), False))


def _backward_with(out: ttml.autograd.Tensor, upstream: np.ndarray) -> None:
    """Backprop ``upstream`` as dL/d(out) via mean(out * upstream * numel)."""
    n = upstream.size
    loss = ttml.ops.unary.mean(ttml.ops.binary.mul(out, _tensor(upstream * n, False)))
    loss.backward(False)


def _rel(a, e) -> float:
    a, e = np.asarray(a, np.float64).ravel(), np.asarray(e, np.float64).ravel()
    return float(np.linalg.norm(a - e) / (np.linalg.norm(e) + 1e-30))


def _rotate_half(x):
    h = x.shape[-1] // 2
    return np.concatenate([-x[..., h:], x[..., :h]], axis=-1)


def test_rope_half_matches_hf(ctx):
    H, S, D = 4, 64, 128
    x = _bf16(RNG.standard_normal((1, H, S, D)))
    rope = {
        "rope_type": "deepseek_yarn",
        "rope_theta": 50000.0,
        "factor": 64.0,
        "mscale": 1.0,
        "mscale_all_dim": 1.0,
        "original_max_position_embeddings": 4096,
        "beta_fast": 32.0,
        "beta_slow": 1.0,
    }
    positions = RNG.integers(0, 4000, S)
    cos, sin = rope_cos_sin(positions, D, rope)
    cos_b, sin_b = _bf16(cos), _bf16(signed_sin(sin))

    xt = _tensor(x, True)
    out = rope_half(xt, _tensor(cos_b[None, None], False), _tensor(sin_b[None, None], False))
    ref = x * cos_b + _rotate_half(x) * _bf16(sin)  # HF formula with the same bf16 tables
    assert _rel(_np(out), ref) < 1e-2

    g = _bf16(RNG.standard_normal(x.shape))
    _backward_with(out, g)
    # d/dx of x*cos + rotate_half(x)*sin is g*cos + rotate_half^T(g*sin).
    gs = g * _bf16(sin)
    h = D // 2
    ref_grad = g * cos_b + np.concatenate([gs[..., h:], -gs[..., :h]], axis=-1)
    assert _rel(_grad(xt), ref_grad) < 2e-2


def test_grouped_heads_layout_matches_hf(ctx):
    S, Hq, Hkv, D = 64, 8, 2, 128
    q = _bf16(RNG.standard_normal((1, 1, S, Hq * D)))
    k = _bf16(RNG.standard_normal((1, 1, S, Hkv * D)))
    v = _bf16(RNG.standard_normal((1, 1, S, Hkv * D)))
    qh, kh, vh = ttml.ops.multi_head_utils.grouped_heads_creation(
        _tensor(q, False), _tensor(np.concatenate([k, v], -1), False), Hq, Hkv
    )

    def hf(x, h):  # view(bsz, q_len, -1, D).transpose(1, 2)
        return x.reshape(1, S, h, D).transpose(0, 2, 1, 3)

    np.testing.assert_array_equal(_np(qh), hf(q, Hq))
    np.testing.assert_array_equal(_np(kh), hf(k, Hkv))
    np.testing.assert_array_equal(_np(vh), hf(v, Hkv))
    fused = ttml.ops.multi_head_utils.heads_fusion(qh)
    np.testing.assert_array_equal(_np(fused), q)


def test_kl_loss_value_and_grad(ctx):
    N, V = 64, 512
    logits = _bf16(2 * RNG.standard_normal((1, 1, N, V)))
    teacher = _bf16(2 * RNG.standard_normal((1, 1, N, V)))
    w = (RNG.random(N) * (RNG.random(N) > 0.3)).astype(np.float32)
    w /= w.sum()

    def log_softmax(x):
        x = x.astype(np.float64)
        m = x.max(-1, keepdims=True)
        return x - m - np.log(np.exp(x - m).sum(-1, keepdims=True))

    log_p, log_q = log_softmax(teacher), log_softmax(logits)
    p, q = np.exp(log_p), np.exp(log_q)
    ref_value = float((w * (p * (log_p - log_q)).sum(-1)[0, 0]).sum())
    ref_grad = w[None, None, :, None] * (q - p)

    lt = _tensor(logits, True)
    loss, value = dflash_kl_loss(
        lt, _tensor(teacher, False), _tensor(w.reshape(1, 1, N, 1), False, ttnn.DataType.FLOAT32)
    )
    assert abs(value - ref_value) <= 2e-2 * abs(ref_value) + 1e-4, (value, ref_value)
    loss.backward(False)
    assert _rel(_grad(lt), ref_grad) < 5e-2


def test_frozen_embedding_gather(ctx):
    V, H, N = 512, 256, 64
    table = _bf16(RNG.standard_normal((V, H)))
    emb = ttml.modules.Embedding(V, H)
    from ttml.autograd.numpy_assign import assign_numpy  # noqa: PLC0415

    assign_numpy(emb.weight.tensor, table)
    emb.weight.tensor.set_requires_grad(False)
    ids = RNG.integers(0, V, N).astype(np.uint32)
    idx = ttml.autograd.Tensor.from_numpy(
        ids.reshape(1, 1, 1, N), layout=ttnn.Layout.ROW_MAJOR, new_type=ttnn.DataType.UINT32
    )
    out = emb(idx)
    np.testing.assert_array_equal(_np(out).reshape(N, H), table[ids])

    # Frozen: backward through it leaves no gradient on the table.
    lin = ttml.modules.LinearLayer(H, 32, False)
    _backward_with(lin(out), _bf16(RNG.standard_normal((1, 1, N, 32))))
    assert not emb.weight.tensor.is_grad_initialized()


def test_adamw_full_precision_and_clip_match_torch_formula(ctx):
    lin = ttml.modules.LinearLayer(64, 32, False)
    from ttml.autograd.numpy_assign import assign_numpy  # noqa: PLC0415

    w = _bf16(0.05 * RNG.standard_normal((32, 64)))
    assign_numpy(lin.weight.tensor, w)
    lr, wd, b1, b2, eps, max_norm = 1e-2, 0.01, 0.9, 0.999, 1e-8, 0.5
    opt = ttml.optimizers.AdamWFullPrecision(
        lin.parameters(), ttml.optimizers.AdamWFullPrecisionConfig.make(lr, b1, b2, eps, wd)
    )
    x = _bf16(RNG.standard_normal((1, 1, 32, 64)))
    ref_w = w.astype(np.float64)
    m = np.zeros_like(ref_w)
    v = np.zeros_like(ref_w)
    for t in range(1, 3):
        opt.zero_grad()
        upstream = _bf16(RNG.standard_normal((1, 1, 32, 32)))
        _backward_with(lin(_tensor(x, False)), upstream)
        g_dev = _grad(lin.weight.tensor).reshape(32, 64).astype(np.float64)
        g_ref = upstream.reshape(32, 32).T.astype(np.float64) @ x.reshape(32, 64)
        assert _rel(g_dev, g_ref) < 2e-2

        norm = float(_np(ttml.core.clip_grad_norm(lin.parameters(), max_norm, 2.0, False)).ravel()[0])
        assert abs(norm - np.linalg.norm(g_dev)) <= 1e-2 * np.linalg.norm(g_dev)
        g = g_dev * min(1.0, max_norm / (norm + 1e-6))
        opt.step()
        ctx.reset_graph()

        ref_w *= 1 - lr * wd
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * g * g
        ref_w -= lr * (m / (1 - b1**t)) / (np.sqrt(v / (1 - b2**t)) + eps)

    masters = opt.get_state_dict()["master_weights"]
    (master,) = list(masters.values())
    got = _np(master).reshape(32, 64)
    delta_dev, delta_ref = got - w, ref_w - w
    assert _rel(delta_dev, delta_ref) < 5e-2, _rel(delta_dev, delta_ref)
