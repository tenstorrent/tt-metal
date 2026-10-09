# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Host-only checks of schedule / conditioning / RoPE / tokenization against the diffusers goldens.
Runs without a device (pytest models/experimental/qwen_image_2_1/tests/test_host_math.py)."""
import os

import numpy as np
import pytest
import torch

from models.experimental.qwen_image_2_1.common import rope, schedule, text, weights
from models.experimental.qwen_image_2_1.common.config import GOLDENS_DIR

requires_goldens = pytest.mark.skipif(not os.path.isdir(GOLDENS_DIR), reason="goldens missing")


def _pcc(a, b):
    a = a.float().flatten()
    b = b.float().flatten()
    return float(torch.corrcoef(torch.stack([a, b]))[0, 1])


@pytest.mark.parametrize("num_steps", [-1, 0, 1])
def test_schedule_rejects_fewer_than_two_steps(num_steps):
    with pytest.raises(ValueError, match="at least two steps"):  # allow-pytest.raises: host-only scheduler test.
        schedule.make_sigmas(num_steps, 4096)


@pytest.mark.parametrize("num_steps", [2, 40, 100])
def test_schedule_is_finite_and_decreasing(num_steps):
    sigmas = schedule.make_sigmas(num_steps, 4096)
    assert len(sigmas) == num_steps + 1
    assert np.isfinite(sigmas).all()
    assert np.isfinite(schedule.sigmas_to_timesteps(sigmas)).all()
    assert np.all(np.diff(sigmas) < 0)
    np.testing.assert_allclose(sigmas[[0, -2, -1]], [1.0, 0.02, 0.0])


@requires_goldens
def test_schedule_matches_reference():
    d = torch.load(os.path.join(GOLDENS_DIR, "denoise.pt"), weights_only=False)
    sig = schedule.make_sigmas(d["num_steps"], 4096)
    np.testing.assert_allclose(sig, d["sigmas"].numpy(), rtol=1e-5, atol=1e-6)
    ts = schedule.sigmas_to_timesteps(sig)
    np.testing.assert_allclose(ts, d["timesteps"].numpy(), rtol=1e-5, atol=1e-3)


@requires_goldens
def test_tokenization_and_drop_idx():
    t = torch.load(os.path.join(GOLDENS_DIR, "text_encoder.pt"), weights_only=False)
    ids = text.tokenize_prompt(t["prompt"])
    assert ids.tolist() == t["input_ids"].tolist()
    assert text.DROP_IDX == t["drop_idx"]
    assert t["prompt_embeds"].shape[1] == ids.shape[1] - text.DROP_IDX


def test_dit_rope_matches_diffusers_formula():
    # reference: diffusers QwenImage21Rope semantics re-derived here with torch.polar tables
    T, H, W = 20, 64, 64
    ang = rope.dit_angles(T, H, W)
    cis = rope.dit_freqs_cis_reference(T, H, W)
    assert cis.shape == (T + H * W, 64)
    # text token 5: all three axes at position 5
    f = rope._freqs(16, 10000)
    assert torch.allclose(ang[5, :8], 5 * f, atol=1e-9)
    # first image token: frame T, h = -32, w = -32
    assert torch.allclose(ang[T, :8], T * f, atol=1e-9)
    assert torch.allclose(ang[T, 8:36], -32 * rope._freqs(56, 10000), atol=1e-9)
    assert torch.allclose(ang[T, 36:], -32 * rope._freqs(56, 10000), atol=1e-9)
    # last image token: h = 31, w = 31
    assert torch.allclose(ang[-1, 8:36], 31 * rope._freqs(56, 10000), atol=1e-9)
    # adjacent-pair rotation equals complex multiply
    x = torch.randn(2, T + H * W, 128, dtype=torch.float64)
    cos, sin = rope.angles_to_cos_sin(ang, torch.float64)
    ref = torch.view_as_real(torch.view_as_complex(x.reshape(2, -1, 64, 2)) * cis[None]).flatten(2)
    out = rope.apply_rope_adjacent(x, cos, sin)
    assert torch.allclose(out, ref, atol=1e-12)


def test_te_rope_permutation_equivalence():
    # llama rotate_half on original layout == adjacent-pair rotation on permuted layout
    S, D = 34, 128
    perm = weights.interleave_pairs_permutation(D)
    x = torch.randn(S, D, dtype=torch.float64)
    inv = rope._freqs(D, 5e6)
    ang = torch.arange(S, dtype=torch.float64)[:, None] * inv[None]
    cos_h = torch.cos(torch.cat([ang, ang], -1))
    sin_h = torch.sin(torch.cat([ang, ang], -1))
    x1, x2 = x[:, : D // 2], x[:, D // 2 :]
    ref = x * cos_h + torch.cat([-x2, x1], -1) * sin_h
    cos, sin = rope.te_cos_sin(S, torch.float64)
    out = rope.apply_rope_adjacent(x[:, perm], cos, sin)
    assert torch.allclose(out, ref[:, perm], atol=1e-12)


@requires_goldens
def test_step_conditioning_matches_reference_modulation():
    """Reconstruct the modulation rows from the checkpoint and compare a block-0 sanity relation:
    the reference noise_pred at step 0 exists; here we only check shapes/dtypes and that the
    bf16-like path and fp32 path agree to bf16 precision."""
    ck = weights.transformer_ckpt()
    tc = schedule.TimeConditioning.from_ckpt(ck)
    sc32 = schedule.StepConditioning.make(tc, 1.0)
    sc16 = schedule.modulation_rows_bf16_like_reference(tc, 1.0)
    for a, b in zip(
        (sc32.one_plus_scale1, sc32.tanh_gate1, sc32.one_plus_scale2, sc32.tanh_gate2, sc32.one_plus_scale_out),
        (sc16.one_plus_scale1, sc16.tanh_gate1, sc16.one_plus_scale2, sc16.tanh_gate2, sc16.one_plus_scale_out),
    ):
        assert a.shape == (1, 4096)
        assert _pcc(a, b) > 0.999


def test_swiglu_interleave_layout():
    K, N = 64, 96
    g = torch.randn(N, K)
    u = torch.randn(N, K)
    w = weights.swiglu_interleave(g, u)
    assert w.shape == (K, 2 * N)
    assert torch.equal(w[:, 0:32], g.t()[:, 0:32])
    assert torch.equal(w[:, 32:64], u.t()[:, 0:32])
    assert torch.equal(w[:, 64:96], g.t()[:, 32:64])
