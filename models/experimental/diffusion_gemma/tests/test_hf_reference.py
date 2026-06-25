# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the HF reference adapter (#47468).

The real HF model is environment-gated, so these verify (a) the guard behaves,
and (b) the adapter + trajectory + PCC harness compose correctly using a mock
canvas-logits model that stands in for the HF / device model.
"""

import pytest
import torch

from models.experimental.diffusion_gemma.config import DiffusionConfig
from models.experimental.diffusion_gemma.reference import sampling as S
from models.experimental.diffusion_gemma.reference.hf_reference import (
    hf_reference_generate,
    is_hf_reference_available,
    load_hf_reference,
    make_logits_fn,
    run_reference_trajectory,
)
from models.experimental.diffusion_gemma.tests.trajectory_pcc import compare_trajectories


def _gen(seed=0):
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def _cfg(**kw):
    return DiffusionConfig(max_denoise_steps=8, entropy_stop_threshold=0.1, stable_steps_to_halt=1, **kw)


class _MockCanvasModel:
    """Deterministic canvas-logits model: fixed per-position logits, optional drift."""

    def __init__(self, batch, length, vocab, seed, drift=0.0):
        self._logits = torch.randn(batch, length, vocab, generator=_gen(seed))
        self._drift = drift

    def __call__(self, canvas, **kw):
        out = self._logits.clone()
        if self._drift:
            out[..., 0] += self._drift  # perturb one logit -> can flip argmax/accept
        return out


def test_guard_reports_unavailable_and_loader_raises(expect_error):
    # In this environment diffusion_gemma is not installed.
    if not is_hf_reference_available():
        with expect_error(ImportError, match="diffusion_gemma"):
            load_hf_reference("google/diffusiongemma-26B-A4B-it")
    else:  # pragma: no cover - only when a diffusion_gemma build is present
        pytest.skip("diffusion_gemma is available; loader guard not exercised")


def test_adapter_drives_trajectory_and_is_deterministic():
    """Two runs are token-for-token equal across EVERY decision class (sampled ids
    and renoised canvas included) only with **noise injection** — the R5 recipe
    the issue calls out (on-device RNG won't reproduce torch's RNG bit-exactly,
    and entropy-budget accept/renoise is noise-sensitive)."""
    batch, length, vocab = 1, 8, 32
    model = _MockCanvasModel(batch, length, vocab, seed=1)
    init = S.random_canvas((batch, length), vocab, generator=_gen(2))

    def gumbel_fn(step):
        return S.sample_gumbel_noise((batch, length, vocab), generator=_gen(1000 + step))

    def noise_tokens_fn(step):
        return torch.randint(0, vocab, (batch, length), generator=_gen(2000 + step))

    a = run_reference_trajectory(
        model, init.clone(), _cfg(), vocab, gumbel_noise_fn=gumbel_fn, noise_tokens_fn=noise_tokens_fn
    )
    b = run_reference_trajectory(
        model, init.clone(), _cfg(), vocab, gumbel_noise_fn=gumbel_fn, noise_tokens_fn=noise_tokens_fn
    )
    assert torch.equal(a.committed, b.committed)
    # Faithful candidate must match the oracle on EVERY decision class — argmax,
    # sampled ids, accept mask, renoised canvas, per-token entropy.
    assert compare_trajectories(a, b).passed


class _FakeRawHFModel:
    """Stands in for a raw DiffusionGemmaForBlockDiffusion: has generate() + a
    config with canvas_length, so the canvas-logits seam must reject it."""

    class _Cfg:
        canvas_length = 256

    def __init__(self):
        self.config = self._Cfg()

    def generate(self, input_ids, **kw):
        return {"sequences": input_ids, "kw": kw}


def test_canvas_logits_seam_rejects_raw_hf_model(expect_error):
    """make_logits_fn / run_reference_trajectory must NOT accept a raw HF model
    (its canvas is decoder_input_ids, not the first positional). Finding #2."""
    raw = _FakeRawHFModel()
    with expect_error(TypeError, match="canvas-logits callable"):
        make_logits_fn(raw)
    with expect_error(TypeError, match="canvas-logits callable"):
        run_reference_trajectory(raw, torch.zeros(1, 8, dtype=torch.long), _cfg(), 32)


def test_hf_reference_generate_delegates_to_model_generate():
    """The real-HF oracle seam calls model.generate(input_ids, ...) and forwards kwargs."""
    raw = _FakeRawHFModel()
    ids = torch.arange(8).view(1, 8)
    out = hf_reference_generate(raw, ids, max_new_tokens=256)
    assert out["sequences"] is ids
    assert out["kw"]["max_new_tokens"] == 256


def test_harness_rejects_drifted_candidate():
    """Drift must be caught on the DETERMINISTIC decision classes, not via RNG noise.
    Inject fixed Gumbel+renoise into all three runs so the only difference is the
    logit drift; a no-drift control must PASS (proving the test isn't vacuous), and
    the drifted run must fail with degraded argmax agreement."""
    batch, length, vocab = 1, 8, 32
    init = S.random_canvas((batch, length), vocab, generator=_gen(3))

    def gumbel_fn(step):
        return S.sample_gumbel_noise((batch, length, vocab), generator=_gen(500 + step))

    def noise_fn(step):
        return torch.randint(0, vocab, (batch, length), generator=_gen(600 + step))

    def run(drift):
        return run_reference_trajectory(
            _MockCanvasModel(batch, length, vocab, seed=5, drift=drift),
            init.clone(),
            _cfg(),
            vocab,
            gumbel_noise_fn=gumbel_fn,
            noise_tokens_fn=noise_fn,
        )

    ref = run(0.0)
    # no-drift control: same injected noise -> identical trajectory -> the harness PASSES
    assert compare_trajectories(ref, run(0.0)).passed
    # a strong logit drift flips argmax/accept deterministically -> the harness FAILS
    cmp = compare_trajectories(ref, run(5.0))
    assert not cmp.passed
    assert cmp.min_argmax_agreement < 1.0  # caught on the deterministic decision class, not RNG
