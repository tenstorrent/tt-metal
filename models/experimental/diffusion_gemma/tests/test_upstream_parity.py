# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Parity vs the vendored canonical transformers primitives (#47468 drift guard).

``reference/_upstream.py`` holds VERBATIM algorithm extractions from
transformers ``diffusion_gemma``. These tests assert our reconciled
``reference/`` primitives reproduce them bit-for-bit on random inputs, so the
oracle cannot silently diverge from the released model.
"""

import torch

from models.experimental.diffusion_gemma.config import DiffusionConfig
from models.experimental.diffusion_gemma.reference import _upstream as U
from models.experimental.diffusion_gemma.reference import sampling as S
from models.experimental.diffusion_gemma.reference.denoise_loop import denoise_block
from models.experimental.diffusion_gemma.reference.self_conditioning import SelfConditioning


def _gen(seed=0):
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def test_temperature_matches_upstream():
    t_min, t_max, N = 0.4, 0.8, 48
    for step in range(N):
        ours = S.temperature_at_step(step, N, t_max, t_min)  # t_start=t_max, t_end=t_min
        theirs = U.temperature_upstream(cur_step=N - step, t_min=t_min, t_max=t_max, max_denoising_steps=N)
        assert ours == theirs, f"step {step}: {ours} != {theirs}"


def test_token_entropy_matches_categorical():
    logits = torch.randn(3, 11, 97, generator=_gen(1))
    ours = S.token_entropy(logits)  # temperature 1.0
    theirs = torch.distributions.Categorical(logits=logits).entropy()
    assert torch.allclose(ours, theirs, atol=1e-5)


def test_entropy_accept_matches_upstream_accept_canvas():
    # The reference takes entropy; upstream takes logits. Compose token_entropy +
    # entropy_budget_accept and compare to accept_canvas_upstream(logits).
    for seed, bound in [(2, 0.05), (3, 0.1), (4, 0.5), (5, 2.0)]:
        logits = torch.randn(2, 64, 128, generator=_gen(seed))
        entropy = S.token_entropy(logits)
        ours = S.entropy_budget_accept(entropy, bound, min_accept=0)
        theirs = U.accept_canvas_upstream(logits, bound)
        assert torch.equal(ours, theirs), f"seed {seed} bound {bound}: mismatch"


def test_stopping_confidence_matches_upstream():
    logits = torch.randn(4, 32, 200, generator=_gen(6))
    entropy = S.token_entropy(logits)
    # straddle the boundary so the comparison is a real per-example check, not all-True/all-False
    thresh = float(entropy.mean(dim=-1).median())
    ours_per_example = entropy.mean(dim=-1) < thresh
    theirs_per_example = U.confident_upstream(logits, thresh)  # (B,)
    assert torch.equal(ours_per_example, theirs_per_example)


def test_stopping_criterion_halt_step_matches_upstream():
    """The reference denoise loop must halt on the SAME step as HF's
    StableAndConfidentStoppingCriteria (stability buffer + confidence). Build a
    batch-1 trajectory that is high-entropy (no halt) for a few steps then becomes
    peaked+stable, and check both fire together."""
    B, L, V, K = 1, 8, 32, 3  # peaked from step K
    target = 7
    flat = torch.zeros(B, L, V)
    peaked = torch.full((B, L, V), -1e4)
    peaked[..., target] = 1e4

    def logits_fn(canvas, step):
        return peaked if step >= K else flat

    cfg = DiffusionConfig(max_denoise_steps=12, entropy_stop_threshold=0.005, stable_steps_to_halt=1)
    traj = denoise_block(logits_fn, S.random_canvas((B, L), V, generator=_gen(1)), cfg, V)

    # Independently apply the vendored HF criterion to the same per-step argmax + temperature-scaled logits.
    crit = U.StableAndConfidentUpstream(
        stability_threshold=cfg.stable_steps_to_halt, confidence_threshold=cfg.entropy_stop_threshold
    )
    upstream_halt = None
    for i in range(cfg.max_denoise_steps):
        logits = logits_fn(None, i)
        T = S.temperature_at_step(i, cfg.max_denoise_steps, cfg.temperature_start, cfg.temperature_end)
        argmax = logits.argmax(dim=-1)
        if bool(crit(argmax, logits / T).all()):
            upstream_halt = i + 1  # 1-based step count, matching DenoiseTrajectory.num_steps
            break

    assert traj.halted and upstream_halt is not None
    assert traj.num_steps == upstream_halt, f"reference halted at {traj.num_steps}, upstream at {upstream_halt}"


def test_self_conditioning_matches_upstream():
    hidden, inter, vocab = 16, 40, 32
    torch.manual_seed(0)
    mod = SelfConditioning(hidden, intermediate_size=inter)
    embed_w = torch.randn(vocab, hidden, generator=_gen(7))
    emb = torch.randn(2, 5, hidden, generator=_gen(8))
    logits = torch.randn(2, 5, vocab, generator=_gen(9))

    ours = mod.condition(emb, logits, embed_w, enabled=True)
    theirs = U.self_conditioning_upstream(
        emb,
        logits,
        embed_w,
        pre_norm_w=mod.pre_norm.weight,
        gate_w=mod.gate_proj.weight,
        up_w=mod.up_proj.weight,
        down_w=mod.down_proj.weight,
    )
    assert torch.allclose(ours, theirs, atol=1e-6)

    # zero-signal (first step / disabled) path also matches
    ours0 = mod.condition(emb, None, embed_w, enabled=False)
    theirs0 = U.self_conditioning_upstream(
        emb,
        None,
        embed_w,
        pre_norm_w=mod.pre_norm.weight,
        gate_w=mod.gate_proj.weight,
        up_w=mod.up_proj.weight,
        down_w=mod.down_proj.weight,
    )
    assert torch.allclose(ours0, theirs0, atol=1e-6)
