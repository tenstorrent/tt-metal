# SPDX-FileCopyrightText: © 2026 Abror Shopulatov

# SPDX-License-Identifier: Apache-2.0
#
# Per-op checks for the FP32 reference (reference/chronos2_reference.py),
# recomputed with numpy float64 rather than by re-running the same torch code.
# Checkpoint-dependent tests need CHRONOS2_CHECKPOINT.

import os
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from models.experimental.chronos2.reference.chronos2_reference import rms_norm, rope_tables, rotate_half  # noqa: E402
from models.experimental.chronos2.tt.weights import fuse_group_attention, load_weights_fp32  # noqa: E402

WEIGHTS = os.environ.get("CHRONOS2_CHECKPOINT")
needs_weights = pytest.mark.skipif(
    not WEIGHTS or not Path(WEIGHTS).exists(), reason="set CHRONOS2_CHECKPOINT to a Chronos-2 checkpoint directory"
)


# --------------------------------------------------------------------------- #
# rms_norm                                                                     #
# --------------------------------------------------------------------------- #


def test_rms_norm_matches_float64_recomputation():
    rng = np.random.default_rng(1)
    x = (rng.normal(size=(3, 7, 16)) * 2.0).astype(np.float32)
    w = rng.normal(size=16).astype(np.float32)
    eps = 1e-6
    got = rms_norm(torch.from_numpy(x), torch.from_numpy(w), eps).numpy()
    x64, w64 = x.astype(np.float64), w.astype(np.float64)
    ref = w64 * x64 / np.sqrt((x64**2).mean(-1, keepdims=True) + eps)
    np.testing.assert_allclose(got, ref.astype(np.float32), rtol=1e-5, atol=1e-6)


# --------------------------------------------------------------------------- #
# RoPE helpers                                                                 #
# --------------------------------------------------------------------------- #


def test_rotate_half_neox_semantics():
    x = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    np.testing.assert_array_equal(rotate_half(x).numpy(), [[-3.0, -4.0, 1.0, 2.0]])


def test_rope_tables_match_float64_and_rotate_orthogonally():
    S, dk, theta = 11, 8, 10000.0
    cos, sin = rope_tables(S, dk, theta)
    inv = 1.0 / (theta ** (np.arange(0, dk, 2, dtype=np.float64) / dk))
    freqs = np.arange(S, dtype=np.float64)[:, None] * inv[None, :]
    emb = np.concatenate([freqs, freqs], -1)
    np.testing.assert_allclose(cos.numpy(), np.cos(emb).astype(np.float32), atol=1e-6)
    np.testing.assert_allclose(sin.numpy(), np.sin(emb).astype(np.float32), atol=1e-6)
    # q*cos + rotate_half(q)*sin must be a per-position rotation (norm-preserving)
    rng = np.random.default_rng(2)
    q = rng.normal(size=(S, dk)).astype(np.float32)
    qt = torch.from_numpy(q)
    rot = (qt * cos + rotate_half(qt) * sin).numpy()
    np.testing.assert_allclose(np.linalg.norm(rot, axis=-1), np.linalg.norm(q, axis=-1), rtol=1e-4)


# --------------------------------------------------------------------------- #
# fused group-attention linear                                                 #
# --------------------------------------------------------------------------- #


def test_group_attention_fusion_is_exact_up_to_fp32_rounding():
    # softmax over the single visible key of an independent row is exactly 1
    assert torch.softmax(torch.zeros(1, 1), dim=-1).item() == 1.0
    # therefore GroupSelfAttention(independent rows) == o(v(x)) == (W_o@W_v) x
    rng = np.random.default_rng(3)
    d, inner = 64, 48
    Wv = (rng.normal(size=(inner, d)) * 0.1).astype(np.float32)
    Wo = (rng.normal(size=(d, inner)) * 0.1).astype(np.float32)
    x = (rng.normal(size=(5, d)) * 0.5).astype(np.float32)
    unfused = (x @ Wv.T) @ Wo.T
    fused = x @ (Wo @ Wv).T
    np.testing.assert_allclose(fused, unfused, rtol=1e-4, atol=1e-6)


@needs_weights
def test_group_fusion_on_real_layer0_weights():
    raw = load_weights_fp32(Path(WEIGHTS))
    p = "encoder.block.0.layer.1"
    fused = fuse_group_attention(raw, p)
    rng = np.random.default_rng(4)
    x = (rng.normal(size=(7, 768)) * 0.3).astype(np.float32)
    unfused = (x @ raw[f"{p}.self_attention.v.weight"].T) @ raw[f"{p}.self_attention.o.weight"].T
    np.testing.assert_allclose(x @ fused.T, unfused, rtol=1e-4, atol=1e-5)


# --------------------------------------------------------------------------- #
# FP32 softmax with finfo.min additive mask                                    #
# --------------------------------------------------------------------------- #


def test_fp32_softmax_matches_float64_and_zeroes_masked_keys():
    rng = np.random.default_rng(5)
    B, H, S = 2, 3, 9
    scores = (rng.normal(size=(B, H, S, S)) * 5.0).astype(np.float32)
    mask = np.ones((B, S), np.float32)
    mask[:, :3] = 0.0
    add = (1.0 - mask)[:, None, None, :] * np.finfo(np.float32).min
    got = torch.softmax(torch.from_numpy(scores + add), dim=-1).numpy()
    vis = mask[:, None, None, :] > 0
    s64 = np.where(vis, scores.astype(np.float64), -np.inf)
    e = np.exp(s64 - s64.max(-1, keepdims=True))
    ref = (e / e.sum(-1, keepdims=True)).astype(np.float32)
    np.testing.assert_allclose(got, ref, rtol=1e-5, atol=1e-7)
    for b in range(B):  # masked keys get probability exactly 0 (exp underflow)
        assert np.all(got[b, ..., mask[b] == 0] == 0.0)
    np.testing.assert_allclose(got.sum(-1), 1.0, rtol=1e-6)


# --------------------------------------------------------------------------- #
# full-forward sanity                                                          #
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def port():
    if not WEIGHTS or not Path(WEIGHTS).exists():
        pytest.skip("set CHRONOS2_CHECKPOINT to a Chronos-2 checkpoint directory")
    from models.experimental.chronos2.reference.chronos2_reference import Chronos2ReferenceFP32

    return Chronos2ReferenceFP32(WEIGHTS)


@needs_weights
def test_port_forward_sanity_and_determinism(port):
    rng = np.random.default_rng(6)
    x = np.cumsum(rng.normal(size=(2, 200)).astype(np.float32), axis=-1) + 10.0
    x[1, :40] = 0.0  # masked head on row 1 (placeholder 0.0 + zero mask)
    m = np.ones_like(x)
    m[1, :40] = 0.0
    out1 = port.predict(x, m, 64)["quantiles"]
    out2 = port.predict(x, m, 64)["quantiles"]
    assert out1.shape == (2, 64, 21) and out1.dtype == np.float32
    assert np.isfinite(out1).all()
    assert np.array_equal(out1, out2)  # repeated-call bitwise determinism


@needs_weights
def test_port_constant_series_finite(port):
    x = np.full((1, 128), 7.0, np.float32)  # zero-variance edge case
    m = np.ones_like(x)
    out = port.predict(x, m, 16)["quantiles"]
    assert out.shape == (1, 16, 21) and np.isfinite(out).all()
