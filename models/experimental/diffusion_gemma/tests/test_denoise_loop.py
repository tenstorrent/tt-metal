# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""CPU unit tests for the reference per-block denoise trajectory (#47463/#47468).

Pure torch with a synthetic ``logits_fn`` oracle — no checkpoint / ttnn / HW.
"""

import torch

from models.experimental.diffusion_gemma.config import DiffusionConfig
from models.experimental.diffusion_gemma.reference import sampling as S
from models.experimental.diffusion_gemma.reference.denoise_loop import denoise_block


def _gen(seed=0):
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def _cfg(**kw):
    return DiffusionConfig(max_denoise_steps=8, entropy_stop_threshold=0.1, stable_steps_to_halt=1, **kw)


def _peaked_logits(batch, length, vocab, target):
    logits = torch.full((batch, length, vocab), -1e4)
    logits[..., target] = 1e4
    return logits


def test_halts_on_stable_low_entropy():
    batch, length, vocab = 1, 8, 32
    target = 7
    peaked = _peaked_logits(batch, length, vocab, target)  # constant, near-zero entropy
    init = S.random_canvas((batch, length), vocab, generator=_gen(1))

    traj = denoise_block(lambda canvas, step: peaked, init, _cfg(), vocab)

    assert traj.halted
    assert traj.num_steps <= 3  # stable+low-entropy detected as soon as prev exists
    assert torch.equal(traj.committed, torch.full((batch, length), target))  # commit = clean argmax
    assert len(traj.per_step) == traj.num_steps


def test_runs_to_cap_when_never_converges():
    batch, length, vocab = 1, 8, 64
    # constant near-uniform logits: argmax is stable but entropy stays high -> never halts
    flat = torch.zeros(batch, length, vocab)

    traj = denoise_block(
        lambda canvas, step: flat, S.random_canvas((batch, length), vocab, generator=_gen(2)), _cfg(), vocab
    )

    assert not traj.halted
    assert traj.num_steps == 8
    assert all(r.entropy_mean > 1.0 for r in traj.per_step)


def test_committed_equals_last_step_clean_argmax():
    batch, length, vocab = 2, 16, 48

    # logits depend on step so argmax shifts; never halts -> runs to cap
    def logits_fn(canvas, step):
        g = _gen(100 + step)
        return torch.randn(batch, length, vocab, generator=g)

    traj = denoise_block(logits_fn, S.random_canvas((batch, length), vocab, generator=_gen(3)), _cfg(), vocab)

    assert torch.equal(traj.committed, traj.per_step[-1].argmax)


def test_determinism_with_injected_noise():
    batch, length, vocab = 1, 12, 40

    def logits_fn(canvas, step):
        return torch.randn(batch, length, vocab, generator=_gen(200 + step))

    def gumbel_fn(step):
        return S.sample_gumbel_noise((batch, length, vocab), generator=_gen(300 + step))

    def noise_fn(step):
        return torch.randint(0, vocab, (batch, length), generator=_gen(400 + step))

    init = S.random_canvas((batch, length), vocab, generator=_gen(5))
    a = denoise_block(logits_fn, init.clone(), _cfg(), vocab, gumbel_noise_fn=gumbel_fn, noise_tokens_fn=noise_fn)
    b = denoise_block(logits_fn, init.clone(), _cfg(), vocab, gumbel_noise_fn=gumbel_fn, noise_tokens_fn=noise_fn)

    assert a.num_steps == b.num_steps and a.halted == b.halted
    assert torch.equal(a.committed, b.committed)
    for ra, rb in zip(a.per_step, b.per_step):
        assert torch.equal(ra.argmax, rb.argmax)
        assert ra.num_accepted == rb.num_accepted
