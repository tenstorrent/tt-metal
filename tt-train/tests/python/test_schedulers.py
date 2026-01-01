# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for ttml LR schedulers.

All schedulers are tested via the pure-Python implementations in
ttml.common.schedulers.

Each scheduler has two test classes:
  - TestXMatchesPyTorch  — trajectory compared step-by-step against the
                            equivalent torch.optim.lr_scheduler class.
  - TestXStateDict        — get_state_dict / set_state_dict round-trip.
"""

import math

import pytest
import torch
import ttml
from ttml.common.schedulers import (
    CosineAnnealingScheduler,
    LambdaScheduler,
    LinearScheduler,
    SequentialScheduler,
    StepScheduler,
    linear_warmup_lambda,
)


BASE_LR = 0.1
N_STEPS = 60


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_opt(lr=BASE_LR, *group_lrs):
    cfg = ttml.optimizers.AdamWConfig.make(lr=lr, beta1=0.9, beta2=0.999, epsilon=1e-8, weight_decay=0.0)
    groups = [{"params": [], "lr": group_lr} for group_lr in group_lrs]
    return ttml.optimizers.AdamW(ttml.NamedParameters(), cfg, param_groups=groups)


def _step(opt, sched):
    sched.step()
    assert opt.get_group_lrs()[0] == pytest.approx(sched.get_last_lrs()[0], abs=1e-8)


def _make_torch_opt(lr=BASE_LR, *group_lrs):
    lrs = [lr, *group_lrs]
    params = [torch.nn.Parameter(torch.tensor(0.0)) for _ in lrs]
    return torch.optim.SGD([{"params": [p], "lr": group_lr} for p, group_lr in zip(params, lrs)], lr=lr)


def _roundtrip(make_sched_fn, n_steps=N_STEPS):
    """Generic state-dict round-trip helper.

    Steps n_steps//2, saves state, restores into a fresh scheduler, steps the
    remaining n_steps//2, and returns (resumed_lrs, reference_lrs_second_half).
    """
    half = n_steps // 2

    ref_opt, ref_sched = make_sched_fn()  # noqa: F841
    ref_lrs = []
    for _ in range(n_steps):
        _step(ref_opt, ref_sched)
        ref_lrs.append(ref_opt.get_group_lrs()[0])

    opt1, sched1 = make_sched_fn()  # noqa: F841
    for _ in range(half):
        _step(opt1, sched1)
    state = sched1.get_state_dict()

    opt2, sched2 = make_sched_fn()  # noqa: F841
    sched2.set_state_dict(state)
    # Restoring scheduler state must also restore the optimizer's live LR
    # (the constructor wrote the construction-time LR), so the first resumed
    # optimizer step runs at the checkpoint's LR — before any step() call.
    assert opt2.get_group_lrs()[0] == pytest.approx(sched1.get_last_lrs()[0], abs=1e-8)
    resumed_lrs = []
    for _ in range(n_steps - half):
        _step(opt2, sched2)
        resumed_lrs.append(opt2.get_group_lrs()[0])

    return resumed_lrs, ref_lrs[half:]


# ---------------------------------------------------------------------------
# CosineAnnealingScheduler
# ---------------------------------------------------------------------------

T_MAX = 20
ETA_MIN = 1e-4


def _make_cosine(base_lr=BASE_LR, T_max=T_MAX, eta_min=ETA_MIN):
    opt = _make_opt(base_lr)
    return opt, CosineAnnealingScheduler(opt, T_max, eta_min)


class TestCosineAnnealingMatchesPyTorch:
    def test_lr_trajectory(self):
        """Every step's LR must match PyTorch CosineAnnealingLR to within float32 precision."""
        opt, sched = _make_cosine()  # noqa: F841
        torch_opt = _make_torch_opt()
        torch_sched = torch.optim.lr_scheduler.CosineAnnealingLR(torch_opt, T_max=T_MAX, eta_min=ETA_MIN)

        # Step 0: construction-time LR must already match (torch performs an
        # initial step at construction).
        assert opt.get_group_lrs()[0] == pytest.approx(torch_opt.param_groups[0]["lr"], abs=1e-7), "step 0"

        for step in range(1, N_STEPS + 1):
            _step(opt, sched)
            torch_opt.step()
            torch_sched.step()
            assert opt.get_group_lrs()[0] == pytest.approx(torch_opt.param_groups[0]["lr"], abs=1e-5), f"step {step}"

    def test_lr_at_T_max_is_eta_min(self):
        """At step T_max the LR must equal eta_min."""
        opt, sched = _make_cosine()  # noqa: F841
        for _ in range(T_MAX):
            _step(opt, sched)
        assert opt.get_group_lrs()[0] == pytest.approx(ETA_MIN, abs=1e-6)

    def test_lr_at_2T_max_is_base_lr(self):
        """At step 2*T_max the LR must return to base_lr."""
        opt, sched = _make_cosine()  # noqa: F841
        for _ in range(2 * T_MAX):
            _step(opt, sched)
        assert opt.get_group_lrs()[0] == pytest.approx(BASE_LR, abs=1e-5)

    def test_get_last_lr_matches_current(self):
        opt, sched = _make_cosine()  # noqa: F841
        for _ in range(7):
            _step(opt, sched)
        assert sched.get_last_lrs()[0] == pytest.approx(opt.get_group_lrs()[0], abs=1e-8)

    def test_eta_min_zero_default(self):
        opt = _make_opt()  # noqa: F841
        sched = CosineAnnealingScheduler(opt, T_MAX)
        for _ in range(T_MAX):
            _step(opt, sched)
        assert opt.get_group_lrs()[0] == pytest.approx(0.0, abs=1e-6)


class TestCosineAnnealingStateDict:
    def test_roundtrip_continues_correctly(self):
        resumed, reference = _roundtrip(_make_cosine)
        assert resumed == pytest.approx(reference, abs=1e-7)

    def test_state_dict_contains_expected_keys(self):
        opt, sched = _make_cosine()  # noqa: F841
        for _ in range(5):
            _step(opt, sched)
        state = sched.get_state_dict()
        assert "m_last_step" in state
        assert "m_last_lrs" in state
        assert "m_T_max" in state
        assert "m_eta_min" in state

    def test_state_dict_step_count(self):
        opt, sched = _make_cosine()  # noqa: F841
        n = 13
        for _ in range(n):
            _step(opt, sched)
        assert sched.get_state_dict()["m_last_step"] == n

    def test_state_dict_persists_hyperparameters(self):
        opt, sched = _make_cosine()  # noqa: F841
        state = sched.get_state_dict()
        assert state["m_T_max"] == T_MAX
        assert state["m_eta_min"] == pytest.approx(ETA_MIN, abs=1e-12)

    def test_set_state_dict_restores_hyperparameters(self):
        # Construct destination with intentionally different hyperparameters,
        # then verify they get overwritten by ``set_state_dict``.
        src_opt, src = _make_cosine(T_max=T_MAX, eta_min=ETA_MIN)  # noqa: F841
        dst_opt = _make_opt()
        dst = CosineAnnealingScheduler(dst_opt, T_max=T_MAX * 2, eta_min=ETA_MIN * 10)
        dst.set_state_dict(src.get_state_dict())
        assert dst._T_max == T_MAX
        assert dst._eta_min == pytest.approx(ETA_MIN, abs=1e-12)

    def test_base_lr_persists_when_resumed_optimizer_has_decayed_lr(self):
        """Reproduces the bug where ``_base_lrs`` is not part of the saved state.

        Real-world resume sequence: the optimizer is loaded from a checkpoint
        with its CURRENT (already-decayed) LR, and the scheduler is then
        reconstructed on top of it. Without persisting ``_base_lrs`` in the
        state dict, the new scheduler captures the decayed LR as its base,
        and every subsequent step computes the cosine with the wrong
        amplitude ``0.5 * (decayed_lr - eta_min)`` instead of
        ``0.5 * (BASE_LR - eta_min)``.
        """
        src_opt, src = _make_cosine()  # noqa: F841
        for _ in range(5):
            _step(src_opt, src)
        decayed_lr = src_opt.get_group_lrs()[0]
        assert decayed_lr != pytest.approx(BASE_LR)  # sanity: the LR really did decay
        state = src.get_state_dict()

        # Mimic checkpoint resume: optimizer.get_group_lrs()[0] now returns the decayed value.
        dst_opt = _make_opt(decayed_lr)
        dst = CosineAnnealingScheduler(dst_opt, T_MAX, ETA_MIN)
        dst.set_state_dict(state)

        # Direct check: the scheduler must remember its ORIGINAL base lr,
        # not whatever happens to be on the optimizer right now.
        assert dst._base_lrs == pytest.approx([BASE_LR], abs=1e-7)

        # Behavioral check: the next step must produce the same LR as the source.
        src.step()
        dst.step()
        assert dst.get_last_lrs()[0] == pytest.approx(src.get_last_lrs()[0], abs=1e-7)

    def test_restore_before_first_step_is_noop(self):
        opt_a, sched_a = _make_cosine()  # noqa: F841
        opt_b, sched_b = _make_cosine()  # noqa: F841
        sched_b.set_state_dict(sched_a.get_state_dict())
        lrs_a, lrs_b = [], []
        for _ in range(10):
            _step(opt_a, sched_a)
            _step(opt_b, sched_b)
            lrs_a.append(opt_a.get_group_lrs()[0])
            lrs_b.append(opt_b.get_group_lrs()[0])
        assert lrs_a == pytest.approx(lrs_b, abs=1e-8)


# ---------------------------------------------------------------------------
# StepScheduler
# ---------------------------------------------------------------------------

STEP_SIZE = 5
GAMMA = 0.5


def _make_step(base_lr=BASE_LR, step_size=STEP_SIZE, gamma=GAMMA):
    opt = _make_opt(base_lr)
    return opt, StepScheduler(opt, step_size, gamma)


class TestStepSchedulerMatchesPyTorch:
    def test_lr_trajectory(self):
        """Must match torch.optim.lr_scheduler.StepLR step-by-step."""
        opt, sched = _make_step()  # noqa: F841
        torch_opt = _make_torch_opt()
        torch_sched = torch.optim.lr_scheduler.StepLR(torch_opt, step_size=STEP_SIZE, gamma=GAMMA)

        # Step 0: construction-time LR must already match.
        assert opt.get_group_lrs()[0] == pytest.approx(torch_opt.param_groups[0]["lr"], abs=1e-7), "step 0"

        for step in range(1, N_STEPS + 1):
            _step(opt, sched)
            torch_opt.step()
            torch_sched.step()
            assert opt.get_group_lrs()[0] == pytest.approx(torch_opt.param_groups[0]["lr"], abs=1e-6), f"step {step}"

    def test_lr_unchanged_before_boundary(self):
        """LR must not decay until step_size steps have been taken."""
        opt, sched = _make_step()  # noqa: F841
        for _ in range(STEP_SIZE - 1):
            _step(opt, sched)
        assert opt.get_group_lrs()[0] == pytest.approx(BASE_LR, abs=1e-7)

    def test_lr_decays_at_boundary(self):
        """LR must be base_lr * gamma exactly at step_size."""
        opt, sched = _make_step()  # noqa: F841
        for _ in range(STEP_SIZE):
            _step(opt, sched)
        assert opt.get_group_lrs()[0] == pytest.approx(BASE_LR * GAMMA, abs=1e-7)

    def test_get_last_lr_matches_current(self):
        opt, sched = _make_step()  # noqa: F841
        for _ in range(7):
            _step(opt, sched)
        assert sched.get_last_lrs()[0] == pytest.approx(opt.get_group_lrs()[0], abs=1e-8)


class TestStepSchedulerStateDict:
    def test_roundtrip_continues_correctly(self):
        resumed, reference = _roundtrip(_make_step)
        assert resumed == pytest.approx(reference, abs=1e-7)

    def test_state_dict_contains_expected_keys(self):
        opt, sched = _make_step()  # noqa: F841
        for _ in range(5):
            _step(opt, sched)
        state = sched.get_state_dict()
        assert "m_last_step" in state
        assert "m_last_lrs" in state
        assert "m_step_size" in state
        assert "m_gamma" in state

    def test_state_dict_step_count(self):
        opt, sched = _make_step()  # noqa: F841
        n = 17
        for _ in range(n):
            _step(opt, sched)
        assert sched.get_state_dict()["m_last_step"] == n

    def test_state_dict_persists_hyperparameters(self):
        opt, sched = _make_step()  # noqa: F841
        state = sched.get_state_dict()
        assert state["m_step_size"] == STEP_SIZE
        assert state["m_gamma"] == pytest.approx(GAMMA, abs=1e-12)

    def test_set_state_dict_restores_hyperparameters(self):
        src_opt, src = _make_step(step_size=STEP_SIZE, gamma=GAMMA)  # noqa: F841
        dst_opt = _make_opt()
        dst = StepScheduler(dst_opt, step_size=STEP_SIZE * 3, gamma=GAMMA * 0.5)
        dst.set_state_dict(src.get_state_dict())
        assert dst._step_size == STEP_SIZE
        assert dst._gamma == pytest.approx(GAMMA, abs=1e-12)

    def test_base_lr_persists_when_resumed_optimizer_has_decayed_lr(self):
        """Reproduces the ``_base_lrs``-not-saved bug for StepScheduler.

        StepScheduler computes ``base_lr * gamma**num_decays``; if ``base_lr``
        comes from a freshly-restored (decayed) optimizer instead of the
        original training value, the LR is silently mis-scaled.
        """
        # Step over a boundary so the LR really has dropped before save.
        src_opt, src = _make_step()  # noqa: F841
        for _ in range(STEP_SIZE + 1):  # past the first boundary
            _step(src_opt, src)
        decayed_lr = src_opt.get_group_lrs()[0]
        assert decayed_lr != pytest.approx(BASE_LR)  # sanity
        state = src.get_state_dict()

        dst_opt = _make_opt(decayed_lr)
        dst = StepScheduler(dst_opt, STEP_SIZE, GAMMA)
        dst.set_state_dict(state)

        assert dst._base_lrs == pytest.approx([BASE_LR], abs=1e-7)
        src.step()
        dst.step()
        assert dst.get_last_lrs()[0] == pytest.approx(src.get_last_lrs()[0], abs=1e-7)

    def test_restore_before_first_step_is_noop(self):
        opt_a, sched_a = _make_step()  # noqa: F841
        opt_b, sched_b = _make_step()  # noqa: F841
        sched_b.set_state_dict(sched_a.get_state_dict())
        lrs_a, lrs_b = [], []
        for _ in range(10):
            _step(opt_a, sched_a)
            _step(opt_b, sched_b)
            lrs_a.append(opt_a.get_group_lrs()[0])
            lrs_b.append(opt_b.get_group_lrs()[0])
        assert lrs_a == pytest.approx(lrs_b, abs=1e-8)


# ---------------------------------------------------------------------------
# LinearScheduler
# ---------------------------------------------------------------------------

START_FACTOR = 0.1
END_FACTOR = 1.0
TOTAL_STEPS = 30


def _make_linear(base_lr=BASE_LR, start_factor=START_FACTOR, end_factor=END_FACTOR, total_steps=TOTAL_STEPS):
    opt = _make_opt(base_lr)
    return opt, LinearScheduler(opt, start_factor, end_factor, total_steps)


class TestLinearSchedulerMatchesPyTorch:
    def test_constructor_applies_start_factor(self):
        """The pre-first-step LR must equal base_lr * start_factor, matching torch."""
        opt, sched = _make_linear()  # noqa: F841
        torch_opt = _make_torch_opt()
        torch.optim.lr_scheduler.LinearLR(
            torch_opt, start_factor=START_FACTOR, end_factor=END_FACTOR, total_iters=TOTAL_STEPS
        )
        assert opt.get_group_lrs()[0] == pytest.approx(BASE_LR * START_FACTOR, abs=1e-7)
        assert sched.get_last_lrs()[0] == pytest.approx(BASE_LR * START_FACTOR, abs=1e-7)
        assert opt.get_group_lrs()[0] == pytest.approx(torch_opt.param_groups[0]["lr"], abs=1e-7)

    def test_invalid_factors_rejected(self, expect_error):
        """Bounds match torch: 0 < start_factor <= 1 and 0 <= end_factor <= 1."""
        for start_factor, end_factor in [(0.0, 1.0), (-0.1, 1.0), (1.5, 1.0), (1.0, -0.1), (1.0, 1.5)]:
            with expect_error(ValueError, "multiplicative factor expected to be"):
                LinearScheduler(_make_opt(), start_factor, end_factor, TOTAL_STEPS)
        # Boundary values are legal: start_factor == 1, end_factor == 0 or 1.
        LinearScheduler(_make_opt(), 1.0, 0.0, TOTAL_STEPS)
        LinearScheduler(_make_opt(), 1.0, 1.0, TOTAL_STEPS)

    def test_lr_trajectory(self):
        """Must match torch.optim.lr_scheduler.LinearLR step-by-step."""
        opt, sched = _make_linear()  # noqa: F841
        torch_opt = _make_torch_opt()
        torch_sched = torch.optim.lr_scheduler.LinearLR(
            torch_opt, start_factor=START_FACTOR, end_factor=END_FACTOR, total_iters=TOTAL_STEPS
        )

        # Step 0: construction-time LR must already be base_lr * start_factor.
        # This is the assertion that the pre-fix implementation failed: it left
        # the optimizer at the full base_lr until the first step().
        assert opt.get_group_lrs()[0] == pytest.approx(torch_opt.param_groups[0]["lr"], abs=1e-7), "step 0"

        for step in range(1, N_STEPS + 1):
            _step(opt, sched)
            torch_opt.step()
            torch_sched.step()
            assert opt.get_group_lrs()[0] == pytest.approx(torch_opt.param_groups[0]["lr"], abs=1e-5), f"step {step}"

    def test_lr_at_total_steps_is_end_factor(self):
        """At total_steps the LR must equal base_lr * end_factor."""
        opt, sched = _make_linear()  # noqa: F841
        for _ in range(TOTAL_STEPS):
            _step(opt, sched)
        assert opt.get_group_lrs()[0] == pytest.approx(BASE_LR * END_FACTOR, abs=1e-6)

    def test_lr_clamps_after_total_steps(self):
        """Beyond total_steps the LR must stay at base_lr * end_factor."""
        opt, sched = _make_linear()  # noqa: F841
        for _ in range(TOTAL_STEPS + 20):
            _step(opt, sched)
        assert opt.get_group_lrs()[0] == pytest.approx(BASE_LR * END_FACTOR, abs=1e-6)

    def test_get_last_lr_matches_current(self):
        opt, sched = _make_linear()  # noqa: F841
        for _ in range(7):
            _step(opt, sched)
        assert sched.get_last_lrs()[0] == pytest.approx(opt.get_group_lrs()[0], abs=1e-8)


class TestLinearSchedulerStateDict:
    def test_roundtrip_continues_correctly(self):
        resumed, reference = _roundtrip(_make_linear)
        assert resumed == pytest.approx(reference, abs=1e-7)

    def test_state_dict_contains_expected_keys(self):
        opt, sched = _make_linear()  # noqa: F841
        for _ in range(5):
            _step(opt, sched)
        state = sched.get_state_dict()
        assert "m_last_step" in state
        assert "m_last_lrs" in state
        assert "m_start_factor" in state
        assert "m_end_factor" in state
        assert "m_total_steps" in state

    def test_state_dict_step_count(self):
        opt, sched = _make_linear()  # noqa: F841
        n = 11
        for _ in range(n):
            _step(opt, sched)
        assert sched.get_state_dict()["m_last_step"] == n

    def test_state_dict_persists_hyperparameters(self):
        opt, sched = _make_linear()  # noqa: F841
        state = sched.get_state_dict()
        assert state["m_start_factor"] == pytest.approx(START_FACTOR, abs=1e-12)
        assert state["m_end_factor"] == pytest.approx(END_FACTOR, abs=1e-12)
        assert state["m_total_steps"] == TOTAL_STEPS

    def test_set_state_dict_restores_hyperparameters(self):
        src_opt, src = _make_linear(  # noqa: F841
            start_factor=START_FACTOR, end_factor=END_FACTOR, total_steps=TOTAL_STEPS
        )
        dst_opt = _make_opt()
        dst = LinearScheduler(
            dst_opt,
            start_factor=START_FACTOR * 0.25,
            end_factor=END_FACTOR * 0.5,
            total_steps=TOTAL_STEPS * 2,
        )
        dst.set_state_dict(src.get_state_dict())
        assert dst._start_factor == pytest.approx(START_FACTOR, abs=1e-12)
        assert dst._end_factor == pytest.approx(END_FACTOR, abs=1e-12)
        assert dst._total_steps == TOTAL_STEPS

    def test_base_lr_persists_when_resumed_optimizer_has_decayed_lr(self):
        """Reproduces the ``_base_lrs``-not-saved bug for LinearScheduler.

        LinearScheduler computes ``base_lr * factor``; if ``base_lr`` comes
        from a freshly-restored (decayed) optimizer instead of the original
        training value, the LR is silently mis-scaled.
        """
        src_opt, src = _make_linear()  # noqa: F841
        for _ in range(5):
            _step(src_opt, src)
        decayed_lr = src_opt.get_group_lrs()[0]
        assert decayed_lr != pytest.approx(BASE_LR)
        state = src.get_state_dict()

        dst_opt = _make_opt(decayed_lr)
        dst = LinearScheduler(dst_opt, START_FACTOR, END_FACTOR, TOTAL_STEPS)
        dst.set_state_dict(state)

        assert dst._base_lrs == pytest.approx([BASE_LR], abs=1e-7)
        src.step()
        dst.step()
        assert dst.get_last_lrs()[0] == pytest.approx(src.get_last_lrs()[0], abs=1e-7)

    def test_restore_before_first_step_is_noop(self):
        opt_a, sched_a = _make_linear()  # noqa: F841
        opt_b, sched_b = _make_linear()  # noqa: F841
        sched_b.set_state_dict(sched_a.get_state_dict())
        lrs_a, lrs_b = [], []
        for _ in range(10):
            _step(opt_a, sched_a)
            _step(opt_b, sched_b)
            lrs_a.append(opt_a.get_group_lrs()[0])
            lrs_b.append(opt_b.get_group_lrs()[0])
        assert lrs_a == pytest.approx(lrs_b, abs=1e-8)


# ---------------------------------------------------------------------------
# LambdaScheduler
# ---------------------------------------------------------------------------

# Exponential decay: lr = base_lr * 0.95^step
_DECAY = 0.95
_LAMBDA = lambda step: _DECAY**step  # noqa: E731


def _make_lambda(base_lr=BASE_LR):
    opt = _make_opt(base_lr)
    return opt, LambdaScheduler(opt, _LAMBDA)


class TestLambdaSchedulerMatchesPyTorch:
    def test_constructor_applies_lambda_at_step_zero(self):
        """The pre-first-step LR must equal base_lr * lambda(0), matching torch."""
        opt = _make_opt()
        sched = LambdaScheduler(opt, lambda step: 0.5)
        assert opt.get_group_lrs()[0] == pytest.approx(BASE_LR * 0.5, abs=1e-7)
        assert sched.get_last_lrs()[0] == pytest.approx(BASE_LR * 0.5, abs=1e-7)

    def test_lr_trajectory(self):
        """Must match torch.optim.lr_scheduler.LambdaLR step-by-step."""
        opt, sched = _make_lambda()  # noqa: F841
        torch_opt = _make_torch_opt()
        torch_sched = torch.optim.lr_scheduler.LambdaLR(torch_opt, lr_lambda=_LAMBDA)

        # Step 0: construction-time LR must already be base_lr * lambda(0).
        # (_LAMBDA(0) == 1, so the real construction coverage is in
        # test_constructor_applies_lambda_at_step_zero with a 0.5 factor.)
        assert opt.get_group_lrs()[0] == pytest.approx(torch_opt.param_groups[0]["lr"], abs=1e-7), "step 0"

        for step in range(1, N_STEPS + 1):
            _step(opt, sched)
            torch_opt.step()
            torch_sched.step()
            assert opt.get_group_lrs()[0] == pytest.approx(torch_opt.param_groups[0]["lr"], abs=1e-6), f"step {step}"

    def test_lr_formula(self):
        """LR after N steps must equal base_lr * lambda(N) exactly."""
        opt, sched = _make_lambda()  # noqa: F841
        for n in range(1, 21):
            _step(opt, sched)
            expected = BASE_LR * (_DECAY**n)
            assert opt.get_group_lrs()[0] == pytest.approx(expected, abs=1e-6), f"step {n}"

    def test_constant_lambda_keeps_lr(self):
        """A lambda that always returns 1.0 must leave the LR unchanged."""
        opt = _make_opt()  # noqa: F841
        sched = LambdaScheduler(opt, lambda _: 1.0)
        for _ in range(10):
            _step(opt, sched)
        assert opt.get_group_lrs()[0] == pytest.approx(BASE_LR, abs=1e-7)

    def test_get_last_lr_matches_current(self):
        opt, sched = _make_lambda()  # noqa: F841
        for _ in range(7):
            _step(opt, sched)
        assert sched.get_last_lrs()[0] == pytest.approx(opt.get_group_lrs()[0], abs=1e-8)


class TestLambdaSchedulerStateDict:
    def test_roundtrip_continues_correctly(self):
        resumed, reference = _roundtrip(_make_lambda)
        assert resumed == pytest.approx(reference, abs=1e-7)

    def test_state_dict_contains_expected_keys(self):
        opt, sched = _make_lambda()  # noqa: F841
        for _ in range(5):
            _step(opt, sched)
        state = sched.get_state_dict()
        assert "m_last_step" in state
        assert "m_last_lrs" in state
        assert "m_lr_lambda" in state

    def test_state_dict_step_count(self):
        opt, sched = _make_lambda()  # noqa: F841
        n = 9
        for _ in range(n):
            _step(opt, sched)
        assert sched.get_state_dict()["m_last_step"] == n

    def test_state_dict_lambda_function_is_none(self):
        """Plain lambda → ``m_lr_lambda`` is saved as ``None`` (matches PyTorch)."""
        opt = _make_opt()  # noqa: F841
        sched = LambdaScheduler(opt, lambda step: 0.95**step)
        assert sched.get_state_dict()["m_lr_lambda"] is None

    def test_state_dict_def_function_is_none(self):
        """``def`` function → ``m_lr_lambda`` is saved as ``None`` (matches PyTorch)."""

        def lr_fn(step):
            return 0.9**step

        opt = _make_opt()  # noqa: F841
        sched = LambdaScheduler(opt, lr_fn)
        assert sched.get_state_dict()["m_lr_lambda"] is None

    def test_state_dict_callable_object_saves_dict(self):
        """Callable object → ``m_lr_lambda`` stores its ``__dict__`` (matches PyTorch)."""

        class _ExpDecay:
            def __init__(self, decay):
                self.decay = decay
                self.calls = 0

            def __call__(self, step):
                self.calls += 1
                return self.decay**step

        callable_obj = _ExpDecay(0.9)
        opt = _make_opt()  # noqa: F841
        sched = LambdaScheduler(opt, callable_obj)
        for _ in range(3):
            _step(opt, sched)

        saved = sched.get_state_dict()["m_lr_lambda"]
        # 4 calls: one at construction (lambda(0), matching PyTorch) + 3 steps.
        assert saved == {"decay": 0.9, "calls": 4}

    def test_state_dict_callable_dict_is_a_copy(self):
        """Mutating the callable after saving must not change the saved state."""

        class _Counter:
            def __init__(self):
                self.n = 0

            def __call__(self, step):
                self.n += 1
                return 1.0

        counter = _Counter()
        opt = _make_opt()  # noqa: F841
        sched = LambdaScheduler(opt, counter)
        _step(opt, sched)
        saved = sched.get_state_dict()["m_lr_lambda"]
        counter.n = 999
        # 2 calls: one at construction (lambda(0), matching PyTorch) + 1 step.
        assert saved == {"n": 2}

    def test_set_state_dict_restores_callable_object_state(self):
        """Round-trip must restore the callable's instance state."""

        class _Counter:
            def __init__(self, start=0):
                self.n = start

            def __call__(self, step):
                self.n += 1
                return 0.5

        src_opt = _make_opt()  # noqa: F841
        src = LambdaScheduler(src_opt, _Counter(start=0))
        for _ in range(4):
            _step(src_opt, src)
        # The source callable has been called 5 times: construction + 4 steps.
        assert src._lr_lambda.n == 5
        state = src.get_state_dict()

        dst_opt = _make_opt()
        dst = LambdaScheduler(dst_opt, _Counter(start=0))
        dst.set_state_dict(state)
        # Destination callable's ``n`` is restored to 5 and not called on load, so it is not one call ahead.
        assert dst._lr_lambda.n == 5
        assert dst_opt.get_group_lrs() == pytest.approx(src_opt.get_group_lrs())

    def test_set_state_dict_with_none_lambda_does_not_touch_callable(self):
        """If saved ``m_lr_lambda`` is ``None`` the destination callable is left alone."""

        def lr_fn(step):
            return 0.5

        # Functions have a writable ``__dict__``; pre-populate it so we can
        # detect any accidental mutation by ``set_state_dict``.
        lr_fn.tag = "untouched"  # type: ignore[attr-defined]

        src_opt = _make_opt()
        src = LambdaScheduler(src_opt, lr_fn)
        for _ in range(3):
            _step(src_opt, src)
        state = src.get_state_dict()
        assert state["m_lr_lambda"] is None

        # Reconstruct on the dst side with the same plain function. This must
        # not raise even though we cannot serialize the function itself.
        dst_opt = _make_opt()
        dst_lr_fn = lr_fn
        dst = LambdaScheduler(dst_opt, dst_lr_fn)
        dict_before = dict(dst._lr_lambda.__dict__)
        dst.set_state_dict(state)

        # Common state still resumed correctly.
        assert dst._last_step == src._last_step
        # Same callable object is still installed (no rebind).
        assert dst._lr_lambda is dst_lr_fn
        # Its ``__dict__`` was not mutated.
        assert dst._lr_lambda.__dict__ == dict_before
        assert dst._lr_lambda.tag == "untouched"  # type: ignore[attr-defined]

    def test_base_lr_persists_when_resumed_optimizer_has_decayed_lr(self):
        """Reproduces the ``_base_lrs``-not-saved bug for LambdaScheduler.

        LambdaScheduler computes ``base_lr * lr_lambda(step)``; if ``base_lr``
        comes from a freshly-restored (decayed) optimizer instead of the
        original training value, the LR is silently mis-scaled.
        """
        src_opt, src = _make_lambda()  # noqa: F841
        for _ in range(5):
            _step(src_opt, src)
        decayed_lr = src_opt.get_group_lrs()[0]
        assert decayed_lr != pytest.approx(BASE_LR)
        state = src.get_state_dict()

        dst_opt = _make_opt(decayed_lr)
        dst = LambdaScheduler(dst_opt, _LAMBDA)
        dst.set_state_dict(state)

        assert dst._base_lrs == pytest.approx([BASE_LR], abs=1e-7)
        src.step()
        dst.step()
        assert dst.get_last_lrs()[0] == pytest.approx(src.get_last_lrs()[0], abs=1e-7)

    def test_restore_before_first_step_is_noop(self):
        opt_a, sched_a = _make_lambda()  # noqa: F841
        opt_b, sched_b = _make_lambda()  # noqa: F841
        sched_b.set_state_dict(sched_a.get_state_dict())
        lrs_a, lrs_b = [], []
        for _ in range(10):
            _step(opt_a, sched_a)
            _step(opt_b, sched_b)
            lrs_a.append(opt_a.get_group_lrs()[0])
            lrs_b.append(opt_b.get_group_lrs()[0])
        assert lrs_a == pytest.approx(lrs_b, abs=1e-8)


# ---------------------------------------------------------------------------
# SequentialScheduler
# ---------------------------------------------------------------------------
#
# NOTE: ``SequentialScheduler`` follows the C++ ttml semantics where each
# ``milestones[i]`` is the *number of steps* allocated to child ``i``

CHILD0_STEPS = 10  # warmup length
CHILD1_STEPS = 20  # decay length
# start_factor == 0 is rejected (matches PyTorch's LinearLR); warmup ramps from
# base_lr / warmup_steps instead.
WARMUP_START = 1.0 / CHILD0_STEPS


def _warmup_factor(step, warmup_steps, start=None):
    """Linear ramp factor from ``start`` (default 1/warmup_steps) to 1.0."""
    if start is None:
        start = 1.0 / warmup_steps
    return start + (1.0 - start) * step / warmup_steps


def _make_sequential(base_lr=BASE_LR):
    """Linear warmup (WARMUP_START -> 1.0 over CHILD0_STEPS) then Linear decay
    (1.0 -> 0.1 over CHILD1_STEPS), chained via SequentialScheduler."""
    opt = _make_opt(base_lr)
    children = [
        LinearScheduler(opt, start_factor=WARMUP_START, end_factor=1.0, total_steps=CHILD0_STEPS),
        LinearScheduler(opt, start_factor=1.0, end_factor=0.1, total_steps=CHILD1_STEPS),
    ]
    return opt, SequentialScheduler(opt, children, milestones=[CHILD0_STEPS, CHILD1_STEPS])


class TestSequentialSchedulerMatchesPyTorch:
    """Head-to-head against torch.optim.lr_scheduler.SequentialLR.

    Milestone mapping: TTML milestones are per-child STEP COUNTS, while
    torch's are cumulative switch points with len == len(schedulers) - 1.
    So TTML [W, D] corresponds to torch [W].

    At the handoff step torch runs the incoming child at epoch 0 while TTML
    runs the outgoing child's final step; the trajectories coincide whenever
    each child starts at the factor the previous child ended on (the standard
    warmup -> decay construction used here).
    """

    def test_warmup_then_linear_decay_matches_sequential_lr(self):
        opt, sched = _make_sequential()  # noqa: F841
        torch_opt = _make_torch_opt()
        torch_children = [
            torch.optim.lr_scheduler.LinearLR(
                torch_opt, start_factor=WARMUP_START, end_factor=1.0, total_iters=CHILD0_STEPS
            ),
            torch.optim.lr_scheduler.LinearLR(torch_opt, start_factor=1.0, end_factor=0.1, total_iters=CHILD1_STEPS),
        ]
        torch_sched = torch.optim.lr_scheduler.SequentialLR(torch_opt, torch_children, milestones=[CHILD0_STEPS])

        # Step 0: SequentialLR resets to initial_lr and redoes the first
        # child's initial step at construction; TTML must land on the same LR.
        assert opt.get_group_lrs()[0] == pytest.approx(torch_opt.param_groups[0]["lr"], abs=1e-7), "step 0"

        # Both LinearLR children clamp after total_iters, so it is safe to
        # compare past the end of the chain (both sides hold base_lr * 0.1).
        for step in range(1, CHILD0_STEPS + CHILD1_STEPS + 6):
            _step(opt, sched)
            torch_opt.step()
            torch_sched.step()
            assert opt.get_group_lrs()[0] == pytest.approx(torch_opt.param_groups[0]["lr"], abs=1e-6), f"step {step}"

    def test_warmup_then_cosine_decay_matches_sequential_lr(self):
        warmup_steps = 5
        decay_steps = 20
        eta_min = 1e-4

        opt = _make_opt()
        children = [
            LinearScheduler(opt, start_factor=1.0 / warmup_steps, end_factor=1.0, total_steps=warmup_steps),
            CosineAnnealingScheduler(opt, T_max=decay_steps, eta_min=eta_min),
        ]
        sched = SequentialScheduler(opt, children, milestones=[warmup_steps, decay_steps])

        torch_opt = _make_torch_opt()
        torch_children = [
            torch.optim.lr_scheduler.LinearLR(
                torch_opt, start_factor=1.0 / warmup_steps, end_factor=1.0, total_iters=warmup_steps
            ),
            torch.optim.lr_scheduler.CosineAnnealingLR(torch_opt, T_max=decay_steps, eta_min=eta_min),
        ]
        torch_sched = torch.optim.lr_scheduler.SequentialLR(torch_opt, torch_children, milestones=[warmup_steps])

        assert opt.get_group_lrs()[0] == pytest.approx(torch_opt.param_groups[0]["lr"], abs=1e-7), "step 0"

        # Stop exactly at the end of the chain: past it TTML holds the final
        # LR (step() is a no-op) while torch keeps stepping the cosine child,
        # which cycles back up.
        for step in range(1, warmup_steps + decay_steps + 1):
            _step(opt, sched)
            torch_opt.step()
            torch_sched.step()
            assert opt.get_group_lrs()[0] == pytest.approx(torch_opt.param_groups[0]["lr"], abs=1e-6), f"step {step}"


class TestSequentialSchedulerBehavior:
    def test_warmup_then_decay_trajectory(self):
        """Closed-form check of the warmup-then-decay LR trajectory."""
        opt, sched = _make_sequential()  # noqa: F841
        # Construction already applies the warmup child's start factor.
        assert opt.get_group_lrs()[0] == pytest.approx(BASE_LR * WARMUP_START, abs=1e-7)
        for i in range(1, CHILD0_STEPS + 1):
            _step(opt, sched)
            expected = BASE_LR * _warmup_factor(i, CHILD0_STEPS)
            assert opt.get_group_lrs()[0] == pytest.approx(expected, abs=1e-7), f"warmup step {i}"
        for j in range(1, CHILD1_STEPS + 1):
            _step(opt, sched)
            expected = BASE_LR * (1.0 + (0.1 - 1.0) * j / CHILD1_STEPS)
            assert opt.get_group_lrs()[0] == pytest.approx(expected, abs=1e-7), f"decay step {j}"

    def test_linear_warmup_then_cosine_decay(self):
        """Common training schedule: linear warmup -> cosine annealing decay.

        - Warmup: ``warmup_steps`` linear steps from BASE_LR / warmup_steps to
                  BASE_LR.
        - Decay:  ``decay_steps`` cosine-annealing steps from BASE_LR down to
                  ``eta_min`` (where ``cos(pi) = -1`` lands the final LR exactly
                  at ``eta_min``).
        - Beyond the chain: ``step()`` becomes a no-op and the LR stays put.
        """
        warmup_steps = 5
        decay_steps = 20
        eta_min = 1e-4

        opt = _make_opt()  # opt.lr = BASE_LR at construction
        children = [
            LinearScheduler(opt, start_factor=1.0 / warmup_steps, end_factor=1.0, total_steps=warmup_steps),
            CosineAnnealingScheduler(opt, T_max=decay_steps, eta_min=eta_min),
        ]
        sched = SequentialScheduler(opt, children, milestones=[warmup_steps, decay_steps])

        # Construction applies the warmup child's start factor; the cosine
        # child (constructed second) must NOT have polluted the base LR.
        assert opt.get_group_lrs()[0] == pytest.approx(BASE_LR / warmup_steps, abs=1e-7)

        # Warmup phase.
        for i in range(1, warmup_steps + 1):
            _step(opt, sched)
            expected = BASE_LR * _warmup_factor(i, warmup_steps)
            assert opt.get_group_lrs()[0] == pytest.approx(expected, abs=1e-7), f"warmup step {i}"
        # End of warmup hands off cleanly at exactly BASE_LR.
        assert opt.get_group_lrs()[0] == pytest.approx(BASE_LR, abs=1e-7)

        # Decay phase: cosine annealing from BASE_LR -> eta_min.
        for j in range(1, decay_steps + 1):
            _step(opt, sched)
            expected = eta_min + 0.5 * (BASE_LR - eta_min) * (1.0 + math.cos(math.pi * j / decay_steps))
            assert opt.get_group_lrs()[0] == pytest.approx(expected, abs=1e-7), f"decay step {j}"
        # cos(pi) = -1 so the final LR is exactly ``eta_min``.
        assert opt.get_group_lrs()[0] == pytest.approx(eta_min, abs=1e-7)

        # Past the end of the chain: step() is a no-op and the LR is held.
        for _ in range(3):
            sched.step()
            assert opt.get_group_lrs()[0] == pytest.approx(eta_min, abs=1e-7)

    def test_step_is_noop_after_all_schedulers_exhausted(self):
        opt, sched = _make_sequential()  # noqa: F841
        total = CHILD0_STEPS + CHILD1_STEPS
        for _ in range(total):
            sched.step()
        # All children exhausted; further steps must be no-ops.
        final_lr = opt.get_group_lrs()[0]
        for _ in range(5):
            sched.step()
            assert opt.get_group_lrs()[0] == final_lr

    def test_get_last_lr_delegates_to_active_child(self):
        opt, sched = _make_sequential()  # noqa: F841
        sched.step()  # one step of warmup
        assert sched.get_last_lrs()[0] == pytest.approx(BASE_LR * _warmup_factor(1, CHILD0_STEPS), abs=1e-7)

    def test_get_last_lr_returns_stored_value_after_exhaustion(self):
        opt, sched = _make_sequential()  # noqa: F841
        for _ in range(CHILD0_STEPS + CHILD1_STEPS):
            sched.step()
        # End of decay: lr = base_lr * 0.1
        assert sched.get_last_lrs()[0] == pytest.approx(BASE_LR * 0.1, abs=1e-7)
        sched.step()  # no-op
        assert sched.get_last_lrs()[0] == pytest.approx(BASE_LR * 0.1, abs=1e-7)

    def test_requires_at_least_one_scheduler(self, expect_error):
        opt = _make_opt()
        with expect_error(ValueError, "requires at least one scheduler"):
            SequentialScheduler(opt, [], [])

    def test_milestones_length_must_match_schedulers(self, expect_error):
        opt = _make_opt()
        child = LinearScheduler(opt, 0.2, 1.0, 5)
        with expect_error(ValueError, r"must equal len\(milestones\)"):
            SequentialScheduler(opt, [child], [5, 10])

    def test_none_scheduler_rejected(self, expect_error):
        opt = _make_opt()
        with expect_error(ValueError, r"schedulers\[0\] is None"):
            SequentialScheduler(opt, [None], [5])


class TestSequentialSchedulerStateDict:
    def test_roundtrip_continues_correctly(self):
        # Round-trip across the whole warmup+decay schedule.
        resumed, reference = _roundtrip(_make_sequential, n_steps=CHILD0_STEPS + CHILD1_STEPS)
        assert resumed == pytest.approx(reference, abs=1e-7)

    def test_state_dict_contains_top_level_keys(self):
        opt, sched = _make_sequential()  # noqa: F841
        for _ in range(5):
            _step(opt, sched)
        state = sched.get_state_dict()
        assert "m_current_step_in_scheduler" in state
        assert "m_current_scheduler_index" in state
        assert "m_last_lrs" in state

    def test_state_dict_contains_per_child_keys(self):
        opt, sched = _make_sequential()  # noqa: F841
        state = sched.get_state_dict()
        # Every wrapped child saves its full state under the per-index prefix
        # ``scheduler_{i}/`` -- this is the behavior that fixes the C++ design
        # bug where only the currently-active child was saved.
        for child_idx in (0, 1):
            prefix = f"scheduler_{child_idx}/"
            assert f"{prefix}m_last_step" in state
            assert f"{prefix}m_last_lrs" in state
            assert f"{prefix}m_start_factor" in state
            assert f"{prefix}m_end_factor" in state
            assert f"{prefix}m_total_steps" in state

    def test_state_dict_persists_all_child_step_counts(self):
        """Even after child 0 is spent, its m_last_step is still saved."""
        opt, sched = _make_sequential()  # noqa: F841
        for _ in range(CHILD0_STEPS + 5):  # run all of child 0 + 5 steps of child 1
            sched.step()
        state = sched.get_state_dict()
        assert state["m_current_scheduler_index"] == 1
        assert state["m_current_step_in_scheduler"] == 5
        # Child 0 finished its full schedule and its end state is preserved.
        assert state["scheduler_0/m_last_step"] == CHILD0_STEPS
        # Child 1 has taken 5 steps so far.
        assert state["scheduler_1/m_last_step"] == 5

    def test_set_state_dict_restores_all_children_hyperparameters(self):
        """Destination chain built with WRONG hyperparameters has all of them
        overwritten by the round trip."""
        src_opt, src = _make_sequential()  # noqa: F841
        for _ in range(CHILD0_STEPS + 5):
            _step(src_opt, src)
        state = src.get_state_dict()

        # Destination has wrong hyperparameters in BOTH children.
        dst_opt = _make_opt()
        dst_children = [
            LinearScheduler(dst_opt, start_factor=0.5, end_factor=0.5, total_steps=999),
            LinearScheduler(dst_opt, start_factor=0.5, end_factor=0.5, total_steps=999),
        ]
        dst = SequentialScheduler(dst_opt, dst_children, milestones=[CHILD0_STEPS, CHILD1_STEPS])
        dst.set_state_dict(state)

        assert dst._schedulers[0]._start_factor == pytest.approx(WARMUP_START, abs=1e-12)
        assert dst._schedulers[0]._end_factor == pytest.approx(1.0, abs=1e-12)
        assert dst._schedulers[0]._total_steps == CHILD0_STEPS
        assert dst._schedulers[1]._start_factor == pytest.approx(1.0, abs=1e-12)
        assert dst._schedulers[1]._end_factor == pytest.approx(0.1, abs=1e-12)
        assert dst._schedulers[1]._total_steps == CHILD1_STEPS

        # And the per-child step counters are restored.
        assert dst._schedulers[0]._last_step == CHILD0_STEPS
        assert dst._schedulers[1]._last_step == 5

    def test_restore_before_first_step_is_noop(self):
        opt_a, sched_a = _make_sequential()  # noqa: F841
        opt_b, sched_b = _make_sequential()  # noqa: F841
        sched_b.set_state_dict(sched_a.get_state_dict())
        lrs_a, lrs_b = [], []
        for _ in range(CHILD0_STEPS + CHILD1_STEPS):
            _step(opt_a, sched_a)
            _step(opt_b, sched_b)
            lrs_a.append(opt_a.get_group_lrs()[0])
            lrs_b.append(opt_b.get_group_lrs()[0])
        assert lrs_a == pytest.approx(lrs_b, abs=1e-8)

    # ------------------------------------------------------------------
    # State-dict round trips through a Linear-warmup -> Cosine-decay chain.
    # These exercise the case where the active child at save-time can be
    # either child (warmup or decay), and where the chain has been fully
    # exhausted past the cosine phase.
    # ------------------------------------------------------------------

    def _make_warmup_cosine(self, warmup_steps=5, decay_steps=20, eta_min=1e-4):
        opt = _make_opt()
        children = [
            LinearScheduler(opt, start_factor=1.0 / warmup_steps, end_factor=1.0, total_steps=warmup_steps),
            CosineAnnealingScheduler(opt, T_max=decay_steps, eta_min=eta_min),
        ]
        return opt, SequentialScheduler(opt, children, milestones=[warmup_steps, decay_steps])

    def test_roundtrip_during_warmup_then_resumes_through_cosine_decay(self):
        """Save mid-warmup, restore into a chain with wrong cosine hyperparams,
        and verify the resumed run produces the same trajectory through the
        full warmup and cosine decay."""
        warmup_steps = 5
        decay_steps = 20
        eta_min = 1e-4

        src_opt, src = self._make_warmup_cosine(warmup_steps, decay_steps, eta_min)

        # Save mid-warmup (active scheduler = child 0 = Linear).
        mid_warmup = 3
        for _ in range(mid_warmup):
            _step(src_opt, src)
        state = src.get_state_dict()

        # Sanity-check what was saved.
        assert state["m_current_scheduler_index"] == 0
        assert state["m_current_step_in_scheduler"] == mid_warmup
        assert state["scheduler_0/m_last_step"] == mid_warmup
        # Child 1 (cosine) hasn't been stepped yet but its hyperparameters are still saved.
        assert state["scheduler_1/m_last_step"] == 0
        assert state["scheduler_1/m_T_max"] == decay_steps
        assert state["scheduler_1/m_eta_min"] == pytest.approx(eta_min, abs=1e-12)

        # Destination: cosine child has deliberately WRONG hyperparameters
        # that must be overwritten by the round trip.
        dst_opt = _make_opt()
        dst_children = [
            LinearScheduler(dst_opt, start_factor=1.0 / warmup_steps, end_factor=1.0, total_steps=warmup_steps),
            CosineAnnealingScheduler(dst_opt, T_max=999, eta_min=0.5),
        ]
        dst = SequentialScheduler(dst_opt, dst_children, milestones=[warmup_steps, decay_steps])
        dst.set_state_dict(state)

        # Cosine hyperparameters were restored even though we haven't entered the decay phase yet.
        assert dst._schedulers[1]._T_max == decay_steps
        assert dst._schedulers[1]._eta_min == pytest.approx(eta_min, abs=1e-12)

        # Resume both src and dst through end of warmup + full cosine decay;
        # every step's LR must match.
        remaining = (warmup_steps - mid_warmup) + decay_steps
        for k in range(remaining):
            _step(src_opt, src)
            _step(dst_opt, dst)
            assert dst_opt.get_group_lrs()[0] == pytest.approx(src_opt.get_group_lrs()[0], abs=1e-7), f"step {k}"

        # Both chains are now exhausted; final LR is eta_min on both sides.
        assert src.get_last_lrs()[0] == pytest.approx(eta_min, abs=1e-7)
        assert dst.get_last_lrs()[0] == pytest.approx(eta_min, abs=1e-7)

    def test_roundtrip_mid_cosine_decay_then_resumes_to_eta_min(self):
        """Save MIDWAY through the cosine decay phase (active child = cosine,
        warmup child is "spent"). Verify the resumed run finishes the decay
        with the same per-step trajectory as the source."""
        warmup_steps = 5
        decay_steps = 20
        eta_min = 1e-4
        cosine_steps_done = 8  # save after this many cosine steps

        src_opt, src = self._make_warmup_cosine(warmup_steps, decay_steps, eta_min)

        # Step through all of warmup + ``cosine_steps_done`` of cosine decay.
        for _ in range(warmup_steps + cosine_steps_done):
            _step(src_opt, src)
        state = src.get_state_dict()

        # Sanity-check what was saved: active child is the cosine scheduler.
        assert state["m_current_scheduler_index"] == 1
        assert state["m_current_step_in_scheduler"] == cosine_steps_done
        # Warmup child finished its full schedule and was preserved.
        assert state["scheduler_0/m_last_step"] == warmup_steps
        assert state["scheduler_0/m_total_steps"] == warmup_steps
        # Cosine child is partway through; its per-step state and hyperparameters were saved.
        assert state["scheduler_1/m_last_step"] == cosine_steps_done
        assert state["scheduler_1/m_T_max"] == decay_steps
        assert state["scheduler_1/m_eta_min"] == pytest.approx(eta_min, abs=1e-12)

        # Destination: BOTH children built with deliberately WRONG hyperparams.
        dst_opt = _make_opt()
        dst_children = [
            LinearScheduler(dst_opt, start_factor=0.5, end_factor=0.5, total_steps=999),
            CosineAnnealingScheduler(dst_opt, T_max=999, eta_min=0.5),
        ]
        dst = SequentialScheduler(dst_opt, dst_children, milestones=[warmup_steps, decay_steps])
        dst.set_state_dict(state)

        # Cosine hyperparameters and per-step counter restored.
        assert dst._schedulers[1]._T_max == decay_steps
        assert dst._schedulers[1]._eta_min == pytest.approx(eta_min, abs=1e-12)
        assert dst._schedulers[1]._last_step == cosine_steps_done
        # Warmup child's restored end-of-schedule state.
        assert dst._schedulers[0]._last_step == warmup_steps
        assert dst._schedulers[0]._total_steps == warmup_steps

        # Resume both src and dst through the remainder of cosine decay; every
        # step's LR must match.
        remaining = decay_steps - cosine_steps_done
        for k in range(remaining):
            _step(src_opt, src)
            _step(dst_opt, dst)
            assert dst_opt.get_group_lrs()[0] == pytest.approx(src_opt.get_group_lrs()[0], abs=1e-7), f"step {k}"

        # Both chains land at exactly eta_min at the end of decay.
        assert src.get_last_lrs()[0] == pytest.approx(eta_min, abs=1e-7)
        assert dst.get_last_lrs()[0] == pytest.approx(eta_min, abs=1e-7)

    def test_roundtrip_after_cosine_phase_preserves_exhausted_state(self):
        """Save AFTER the cosine annealing phase (chain fully exhausted) and
        verify the destination chain is also "spent" with all child state
        preserved."""
        warmup_steps = 5
        decay_steps = 20
        eta_min = 1e-4
        total_steps = warmup_steps + decay_steps

        src_opt, src = self._make_warmup_cosine(warmup_steps, decay_steps, eta_min)
        for _ in range(total_steps):
            _step(src_opt, src)

        # Chain is exhausted: it stays on the last child, which has taken all its
        # steps, and last_lr is eta_min.
        state = src.get_state_dict()
        assert state["m_current_scheduler_index"] == 1
        assert state["m_current_step_in_scheduler"] == decay_steps
        assert state["m_last_lrs"][0] == pytest.approx(eta_min, abs=1e-7)
        # Both children's end-of-schedule states were saved.
        assert state["scheduler_0/m_last_step"] == warmup_steps
        assert state["scheduler_1/m_last_step"] == decay_steps

        # Destination: BOTH children constructed with wrong hyperparameters.
        dst_opt = _make_opt()
        dst_children = [
            LinearScheduler(dst_opt, start_factor=0.5, end_factor=0.5, total_steps=999),
            CosineAnnealingScheduler(dst_opt, T_max=999, eta_min=0.5),
        ]
        dst = SequentialScheduler(dst_opt, dst_children, milestones=[warmup_steps, decay_steps])
        dst.set_state_dict(state)

        # Read back through the public state dict, which holds the chain's position and every child's state.
        restored = dst.get_state_dict()

        # Top-level: dst is exhausted too.
        assert restored["m_current_scheduler_index"] == 1
        assert restored["m_current_step_in_scheduler"] == decay_steps
        assert dst.get_last_lrs()[0] == pytest.approx(eta_min, abs=1e-7)

        # Linear child's full state preserved.
        assert restored["scheduler_0/m_start_factor"] == pytest.approx(1.0 / warmup_steps, abs=1e-12)
        assert restored["scheduler_0/m_end_factor"] == pytest.approx(1.0, abs=1e-12)
        assert restored["scheduler_0/m_total_steps"] == warmup_steps
        assert restored["scheduler_0/m_last_step"] == warmup_steps

        # Cosine child's full state preserved (despite dst constructing it with
        # wildly different hyperparameters).
        assert restored["scheduler_1/m_T_max"] == decay_steps
        assert restored["scheduler_1/m_eta_min"] == pytest.approx(eta_min, abs=1e-12)
        assert restored["scheduler_1/m_last_step"] == decay_steps

        # Further step() calls on dst are no-ops; get_last_lrs() stays put.
        final_lr = dst.get_last_lrs()[0]
        for _ in range(3):
            dst.step()
            assert dst.get_last_lrs()[0] == pytest.approx(final_lr, abs=1e-12)


# ---------------------------------------------------------------------------
# Optimizer initial_lr serialization
#
# Mirrors PyTorch, where ``param_group["initial_lr"]`` (recorded when the
# first scheduler attaches) rides along in ``optimizer.state_dict()`` so a
# scheduler attached after resuming a checkpoint still sees the original base
# LR rather than the decayed one. Unlike PyTorch's open param-group dicts,
# TTML always writes the key (falling back to the current LR when no
# scheduler has attached) and requires it when loading a state dict.
# ---------------------------------------------------------------------------


class TestOptimizerInitialLrStateDict:
    def test_state_dict_always_contains_initial_lr(self):
        # Before any scheduler attaches, the base LR is simply the current LR.
        opt = _make_opt()
        assert opt.get_state_dict()["param_groups.0.initial_lr"] == pytest.approx(BASE_LR, abs=1e-8)

    def test_state_dict_records_initial_lr_once_scheduler_attaches(self):
        opt = _make_opt()
        LinearScheduler(opt, 0.5, 1.0, 10)  # records initial_lr; LR becomes BASE_LR * 0.5
        state = opt.get_state_dict()
        assert state["param_groups.0.initial_lr"] == pytest.approx(BASE_LR, abs=1e-8)
        assert state["param_groups.0.lr"] == pytest.approx(BASE_LR * 0.5, abs=1e-8)

    def test_initial_lr_survives_optimizer_state_dict_round_trip(self):
        # Source run: scheduler attaches (records initial_lr), LR decays
        # mid-warmup, then the optimizer is checkpointed.
        src = _make_opt()
        sched = LinearScheduler(src, 0.5, 1.0, 10)
        for _ in range(3):
            sched.step()
        decayed_lr = src.get_group_lrs()[0]
        assert decayed_lr < BASE_LR
        state = src.get_state_dict()
        assert state["param_groups.0.initial_lr"] == pytest.approx(BASE_LR, abs=1e-8)

        # Resume into a fresh optimizer (deliberately different construction
        # LR) BEFORE any scheduler attaches — the usual restore order.
        dst = _make_opt(0.999)
        dst.set_state_dict(state)
        assert dst.get_group_lrs()[0] == pytest.approx(decayed_lr, abs=1e-8)
        assert dst.get_initial_group_lrs() == pytest.approx([BASE_LR], abs=1e-8)

        # A brand-new scheduler attached after resume uses the ORIGINAL base
        # LR: StepScheduler applies base_lr at construction, so the optimizer
        # LR snaps back to BASE_LR, not the decayed checkpoint LR.
        StepScheduler(dst, step_size=10, gamma=0.1)
        assert dst.get_group_lrs()[0] == pytest.approx(BASE_LR, abs=1e-8)


# ---------------------------------------------------------------------------
# Live LR is restored together with scheduler state
#
# Scheduler constructors write the construction-time LR to the optimizer, so
# in the resume order "load optimizer state -> construct scheduler -> load
# scheduler state" the checkpoint's live LR gets overwritten at construction.
# set_state_dict must push m_last_lrs back to the optimizer or the first
# resumed optimizer.step() runs at the wrong LR.
# ---------------------------------------------------------------------------


class TestLiveLrRestoredWithSchedulerState:
    def test_optimizer_state_loaded_before_scheduler_construction(self):
        # Source run: decay a few steps mid-warmup, checkpoint both.
        src = _make_opt()
        src_sched = LinearScheduler(src, 0.5, 1.0, 10)
        for _ in range(3):
            src_sched.step()
        live_lr = src.get_group_lrs()[0]
        opt_state = src.get_state_dict()
        sched_state = src_sched.get_state_dict()

        # Resume in the hazardous order: optimizer state FIRST, then scheduler
        # construction (which overwrites the live LR with base_lr * 0.5)...
        dst = _make_opt()
        dst.set_state_dict(opt_state)
        assert dst.get_group_lrs()[0] == pytest.approx(live_lr, abs=1e-8)
        dst_sched = LinearScheduler(dst, 0.5, 1.0, 10)
        assert dst.get_group_lrs()[0] == pytest.approx(BASE_LR * 0.5, abs=1e-8)

        # ...then scheduler state: the live LR must come back BEFORE any
        # step() call, so the first resumed optimizer step uses it.
        dst_sched.set_state_dict(sched_state)
        assert dst.get_group_lrs()[0] == pytest.approx(live_lr, abs=1e-8)

    def test_sequential_restores_active_childs_live_lr(self):
        # Chain: 10-step warmup then 20-step decay; checkpoint 3 steps into
        # the warmup (child 0 active).
        def make():
            opt = _make_opt()
            warmup = LinearScheduler(opt, 0.1, 1.0, 10)
            decay = LinearScheduler(opt, 1.0, 0.01, 20)
            return opt, SequentialScheduler(opt, [warmup, decay], [10, 20])

        src_opt, src_sched = make()
        for _ in range(3):
            src_sched.step()
        live_lr = src_opt.get_group_lrs()[0]
        state = src_sched.get_state_dict()

        # Restoring child states pushes each child's saved live LR to the
        # optimizer in turn (the LAST child's would win); the chain must
        # re-apply the ACTIVE child's live LR afterwards.
        dst_opt, dst_sched = make()
        dst_sched.set_state_dict(state)
        assert dst_opt.get_group_lrs()[0] == pytest.approx(live_lr, abs=1e-8)


# ---------------------------------------------------------------------------
# Multi-LR optimizers: MuonWithAdamW
#
# MuonWithAdamW holds two inner optimizers. Its param groups are Muon's
# followed by AdamW's, so a scheduler attached to the wrapper schedules every
# group of both from its own base LR (like a PyTorch scheduler over several
# param groups); one scheduler per inner optimizer gives them different
# schedules.
# ---------------------------------------------------------------------------

MUON_LR = 0.02
ADAMW_LR = 0.0003


def _make_muon_with_adamw():
    from ttml.common.muon_optimizer import MuonWithAdamW

    config = {"muon": {"lr": MUON_LR}, "adamw": {"lr": ADAMW_LR}}
    return MuonWithAdamW(config, ttml.NamedParameters())


class TestSchedulersOnMuonWithAdamWWrapper:
    def test_wrapper_groups_are_muon_then_adamw(self):
        opt = _make_muon_with_adamw()
        assert opt.get_group_lrs() == pytest.approx([MUON_LR, ADAMW_LR], abs=1e-8)
        assert [group["lr"] for group in opt.get_param_groups()] == pytest.approx([MUON_LR, ADAMW_LR], abs=1e-8)

    def test_wrapper_set_group_lrs_rejects_wrong_count(self, expect_error):
        opt = _make_muon_with_adamw()
        with expect_error(ValueError, "MuonWithAdamW has 2 param groups, but got 1 values for 'lr'"):
            opt.set_group_lrs([0.1])

    def test_one_scheduler_scales_both_inner_optimizers(self):
        opt = _make_muon_with_adamw()
        sched = StepScheduler(opt, step_size=1, gamma=0.5)
        assert opt.get_group_lrs() == pytest.approx([MUON_LR, ADAMW_LR], abs=1e-8)
        sched.step()
        assert opt.muon_optimizer().get_group_lrs() == pytest.approx([MUON_LR * 0.5], abs=1e-8)
        assert opt.adamw_optimizer().get_group_lrs() == pytest.approx([ADAMW_LR * 0.5], abs=1e-8)

    def test_sequential_scheduler_on_wrapper(self):
        opt = _make_muon_with_adamw()
        warmup = LinearScheduler(opt, 0.5, 1.0, 2)
        decay = StepScheduler(opt, step_size=1, gamma=0.1)
        sched = SequentialScheduler(opt, [warmup, decay], [2, 10])
        assert opt.get_group_lrs() == pytest.approx([MUON_LR * 0.5, ADAMW_LR * 0.5], abs=1e-8)
        for _ in range(3):
            sched.step()
        assert opt.get_group_lrs() == pytest.approx([MUON_LR * 0.1, ADAMW_LR * 0.1], abs=1e-8)

    def test_wrapper_initial_lrs_survive_state_dict_round_trip(self):
        src = _make_muon_with_adamw()
        sched = LinearScheduler(src, 0.5, 1.0, 10)
        for _ in range(3):
            sched.step()
        state = src.get_state_dict()

        # A scheduler attached to a resumed wrapper sees the inner optimizers' checkpointed bases.
        dst = _make_muon_with_adamw()
        dst.set_state_dict(state)
        assert dst.get_initial_group_lrs() == pytest.approx([MUON_LR, ADAMW_LR], abs=1e-8)
        StepScheduler(dst, step_size=10, gamma=0.1)
        assert dst.get_group_lrs() == pytest.approx([MUON_LR, ADAMW_LR], abs=1e-8)


class TestMuonWithAdamWPerInnerScheduling:
    """The sanctioned pattern: one scheduler per inner optimizer."""

    def test_inner_optimizers_accept_schedulers_independently(self):
        opt = _make_muon_with_adamw()
        muon_sched = StepScheduler(opt.muon_optimizer(), step_size=1, gamma=0.5)
        adamw_sched = StepScheduler(opt.adamw_optimizer(), step_size=1, gamma=0.1)

        assert opt.muon_optimizer().get_group_lrs()[0] == pytest.approx(MUON_LR, abs=1e-8)
        assert opt.adamw_optimizer().get_group_lrs()[0] == pytest.approx(ADAMW_LR, abs=1e-8)

        # Each scheduler drives only its own channel, from its own base.
        muon_sched.step()
        assert opt.muon_optimizer().get_group_lrs()[0] == pytest.approx(MUON_LR * 0.5, abs=1e-8)
        assert opt.adamw_optimizer().get_group_lrs()[0] == pytest.approx(ADAMW_LR, abs=1e-8)

        adamw_sched.step()
        assert opt.muon_optimizer().get_group_lrs()[0] == pytest.approx(MUON_LR * 0.5, abs=1e-8)
        assert opt.adamw_optimizer().get_group_lrs()[0] == pytest.approx(ADAMW_LR * 0.1, abs=1e-8)

    def test_inner_initial_lrs_survive_wrapper_state_dict_round_trip(self):
        # Source run: schedulers attach to both inners (recording each inner's
        # initial_lr), decay mid-warmup, then the WRAPPER is checkpointed.
        src = _make_muon_with_adamw()
        muon_sched = LinearScheduler(src.muon_optimizer(), 0.5, 1.0, 10)
        adamw_sched = LinearScheduler(src.adamw_optimizer(), 0.5, 1.0, 10)
        for _ in range(3):
            muon_sched.step()
            adamw_sched.step()
        assert src.muon_optimizer().get_group_lrs()[0] < MUON_LR
        assert src.adamw_optimizer().get_group_lrs()[0] < ADAMW_LR
        state = src.get_state_dict()
        assert state["muon"]["param_groups.0.initial_lr"] == pytest.approx(MUON_LR, abs=1e-8)
        assert state["adamw"]["param_groups.0.initial_lr"] == pytest.approx(ADAMW_LR, abs=1e-8)

        # Resume into a fresh wrapper, then attach brand-new schedulers to the
        # inners: each must see its ORIGINAL base LR, not the decayed one.
        # StepScheduler applies base_lr at construction, so both channels snap
        # back to their bases.
        dst = _make_muon_with_adamw()
        dst.set_state_dict(state)
        StepScheduler(dst.muon_optimizer(), step_size=10, gamma=0.1)
        StepScheduler(dst.adamw_optimizer(), step_size=10, gamma=0.1)
        assert dst.muon_optimizer().get_group_lrs()[0] == pytest.approx(MUON_LR, abs=1e-8)
        assert dst.adamw_optimizer().get_group_lrs()[0] == pytest.approx(ADAMW_LR, abs=1e-8)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


# ---------------------------------------------------------------------------
# Param groups: every group is scheduled from its own base LR.
# ---------------------------------------------------------------------------

GROUP_LRS = [0.1, 0.01]


def _torch_group_lrs(torch_opt):
    return [group["lr"] for group in torch_opt.param_groups]


_GROUPED_CASES = {
    # eta_min != 0 makes the cosine LR non-proportional to the base LR: each group must reach eta_min.
    "cosine": (
        lambda opt: CosineAnnealingScheduler(opt, T_max=10, eta_min=1e-3),
        lambda opt: torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=10, eta_min=1e-3),
    ),
    "step": (
        lambda opt: StepScheduler(opt, step_size=3, gamma=0.5),
        lambda opt: torch.optim.lr_scheduler.StepLR(opt, step_size=3, gamma=0.5),
    ),
    "linear": (
        lambda opt: LinearScheduler(opt, start_factor=0.25, end_factor=1.0, total_steps=8),
        lambda opt: torch.optim.lr_scheduler.LinearLR(opt, start_factor=0.25, end_factor=1.0, total_iters=8),
    ),
    "lambda": (
        lambda opt: LambdaScheduler(opt, lambda epoch: 1.0 / (epoch + 1)),
        lambda opt: torch.optim.lr_scheduler.LambdaLR(opt, lambda epoch: 1.0 / (epoch + 1)),
    ),
}


class TestParamGroupScheduling:
    @pytest.mark.parametrize("name", list(_GROUPED_CASES))
    def test_every_group_matches_pytorch(self, name):
        make_sched, make_torch_sched = _GROUPED_CASES[name]
        opt = _make_opt(*GROUP_LRS)
        sched = make_sched(opt)
        torch_opt = _make_torch_opt(*GROUP_LRS)
        torch_sched = make_torch_sched(torch_opt)

        assert opt.get_group_lrs() == pytest.approx(_torch_group_lrs(torch_opt), abs=1e-7), "step 0"
        for step in range(1, 16):
            sched.step()
            torch_opt.step()
            torch_sched.step()
            expected = _torch_group_lrs(torch_opt)
            assert opt.get_group_lrs() == pytest.approx(expected, abs=1e-6), f"step {step}"
            assert sched.get_last_lrs() == pytest.approx(expected, abs=1e-6), f"step {step}"

    def test_sequential_warmup_then_cosine_matches_pytorch(self):
        warmup_steps, decay_steps, eta_min = 4, 10, 1e-3
        opt = _make_opt(*GROUP_LRS)
        children = [
            LinearScheduler(opt, start_factor=0.25, end_factor=1.0, total_steps=warmup_steps),
            CosineAnnealingScheduler(opt, T_max=decay_steps, eta_min=eta_min),
        ]
        sched = SequentialScheduler(opt, children, milestones=[warmup_steps, decay_steps])

        torch_opt = _make_torch_opt(*GROUP_LRS)
        torch_children = [
            torch.optim.lr_scheduler.LinearLR(torch_opt, start_factor=0.25, end_factor=1.0, total_iters=warmup_steps),
            torch.optim.lr_scheduler.CosineAnnealingLR(torch_opt, T_max=decay_steps, eta_min=eta_min),
        ]
        torch_sched = torch.optim.lr_scheduler.SequentialLR(torch_opt, torch_children, milestones=[warmup_steps])

        assert opt.get_group_lrs() == pytest.approx(_torch_group_lrs(torch_opt), abs=1e-7), "step 0"
        for step in range(1, warmup_steps + decay_steps + 1):
            sched.step()
            torch_opt.step()
            torch_sched.step()
            assert opt.get_group_lrs() == pytest.approx(_torch_group_lrs(torch_opt), abs=1e-6), f"step {step}"

    def test_state_dict_restores_every_groups_base_and_live_lr(self):
        src_opt = _make_opt(*GROUP_LRS)
        src = CosineAnnealingScheduler(src_opt, T_max=10, eta_min=1e-3)
        for _ in range(4):
            src.step()
        state = src.get_state_dict()
        assert state["m_base_lrs"] == pytest.approx(GROUP_LRS)
        assert state["m_last_lrs"] == pytest.approx(src_opt.get_group_lrs())

        # Resume onto an optimizer whose LRs have already decayed: the saved bases are kept and the saved
        # live LRs are pushed back to the optimizer.
        dst_opt = _make_opt(*src_opt.get_group_lrs())
        dst = CosineAnnealingScheduler(dst_opt, T_max=10, eta_min=1e-3)
        dst.set_state_dict(state)
        assert dst_opt.get_group_lrs() == pytest.approx(src_opt.get_group_lrs(), abs=1e-7)
        for step in range(4):
            src.step()
            dst.step()
            assert dst_opt.get_group_lrs() == pytest.approx(src_opt.get_group_lrs(), abs=1e-7), f"step {step}"

    def test_set_group_lrs_rejects_wrong_count(self, expect_error):
        opt = _make_opt(*GROUP_LRS)
        with expect_error(ValueError, "has 2 param groups, but got 1 values for 'lr'"):
            opt.set_group_lrs([0.1])


def test_sequential_state_dict_with_out_of_range_index_rejected(expect_error):
    opt = _make_opt()
    children = [LinearScheduler(opt, 0.5, 1.0, 4), LinearScheduler(opt, 1.0, 0.1, 4)]
    sched = SequentialScheduler(opt, children, milestones=[4, 4])

    state = dict(sched.get_state_dict())
    state["m_current_scheduler_index"] = 2
    with expect_error(ValueError, "scheduler index 2, but the chain has 2 schedulers"):
        sched.set_state_dict(state)

    state = dict(sched.get_state_dict())
    state["m_current_step_in_scheduler"] = 5
    with expect_error(ValueError, "5 steps in scheduler 0, outside"):
        sched.set_state_dict(state)


def test_rejected_scheduler_state_leaves_scheduler_unchanged(expect_error):
    # A checkpoint from an optimizer with 2 groups, loaded into a scheduler on a 1-group optimizer.
    src = StepScheduler(_make_opt(*GROUP_LRS), step_size=1, gamma=0.5)
    src.step()
    two_group_state = src.get_state_dict()

    opt = _make_opt(0.3)
    dst = StepScheduler(opt, step_size=1, gamma=0.5)
    dst.step()
    with expect_error(
        ValueError, "Scheduler checkpoint has 2 base LRs and 2 last LRs, but the optimizer has 1 param groups"
    ):
        dst.set_state_dict(two_group_state)

    # Neither the base state nor the StepScheduler hyperparameters changed.
    assert dst.get_state_dict()["m_last_step"] == 1
    assert dst.get_state_dict()["m_base_lrs"] == pytest.approx([0.3])
    dst.step()
    assert opt.get_group_lrs() == pytest.approx([0.3 * 0.25])


def test_state_with_wrong_last_lrs_count_leaves_scheduler_unchanged(expect_error):
    src = StepScheduler(_make_opt(), step_size=2, gamma=0.1)
    state = dict(src.get_state_dict())
    state["m_last_lrs"] = [BASE_LR, BASE_LR]

    opt = _make_opt()
    dst = StepScheduler(opt, step_size=1, gamma=0.5)
    before = dst.get_state_dict()
    with expect_error(
        ValueError, "Scheduler checkpoint has 1 base LRs and 2 last LRs, but the optimizer has 1 param groups"
    ):
        dst.set_state_dict(state)
    assert dst.get_state_dict() == before


def test_lambda_restore_does_not_call_a_stateful_callable():
    """A callable whose result depends on how often it was called resumes exactly where it was saved."""

    class _Halving:
        def __init__(self):
            self.factor = 1.0

        def __call__(self, step):
            self.factor *= 0.5
            return self.factor

    src_opt = _make_opt()
    src = LambdaScheduler(src_opt, _Halving())
    for _ in range(3):
        src.step()
    state = src.get_state_dict()

    dst_opt = _make_opt()
    dst = LambdaScheduler(dst_opt, _Halving())
    dst.set_state_dict(state)
    assert dst_opt.get_group_lrs() == pytest.approx(src_opt.get_group_lrs())
    for _ in range(2):
        src.step()
        dst.step()
        assert dst_opt.get_group_lrs() == pytest.approx(src_opt.get_group_lrs())


def test_sequential_rejected_child_state_rolls_back(expect_error):
    """A later child rejecting its state undoes the earlier children's restore and the optimizer's LRs."""
    opt = _make_opt()
    children = [LinearScheduler(opt, 0.5, 1.0, 4), StepScheduler(opt, step_size=1, gamma=0.5)]
    sched = SequentialScheduler(opt, children, milestones=[4, 4])
    for _ in range(2):
        sched.step()
    other_opt = _make_opt()
    other = SequentialScheduler(
        other_opt,
        [LinearScheduler(other_opt, 0.5, 1.0, 4), StepScheduler(other_opt, step_size=1, gamma=0.5)],
        milestones=[4, 4],
    )
    for _ in range(6):
        other.step()
    state = dict(other.get_state_dict())
    del state["scheduler_1/m_gamma"]  # the second child cannot restore

    before = sched.get_state_dict()
    lrs_before = opt.get_group_lrs()
    with expect_error(KeyError, "m_gamma"):
        sched.set_state_dict(state)
    assert sched.get_state_dict() == before
    assert opt.get_group_lrs() == pytest.approx(lrs_before)


def test_lambda_restore_applies_lr_from_restored_callable_state():
    """On load the LR comes from the checkpoint and the callable state is restored, not taken from a fresh callable."""

    class _Scaled:
        def __init__(self):
            self.scale = 1.0

        def __call__(self, step):
            return self.scale

    src_opt = _make_opt()
    src = LambdaScheduler(src_opt, _Scaled())
    src._lr_lambda.scale = 0.25  # state the callable picked up during training
    src.step()
    assert src_opt.get_group_lrs()[0] == pytest.approx(BASE_LR * 0.25, abs=1e-8)
    state = src.get_state_dict()

    dst_opt = _make_opt()
    dst = LambdaScheduler(dst_opt, _Scaled())  # fresh callable: scale 1.0
    dst.set_state_dict(state)
    assert dst_opt.get_group_lrs()[0] == pytest.approx(BASE_LR * 0.25, abs=1e-8)
    assert dst._lr_lambda.scale == 0.25


@pytest.mark.parametrize("warmup_steps", [0, 1, 4])
def test_linear_warmup_lambda(warmup_steps):
    lr_lambda = linear_warmup_lambda(warmup_steps)
    for step in range(8):
        expected = 1.0 if warmup_steps == 0 else min(1.0, (step + 1) / warmup_steps)
        assert lr_lambda(step) == pytest.approx(expected)


def test_set_lr_before_attaching_does_not_fix_the_scheduler_base():
    """set_lr must not record the initial LRs: a scheduler attached afterwards takes the LR set_lr left
    as its base."""
    opt = _make_opt(1e-3)
    opt.set_lr(3e-4)
    LambdaScheduler(opt, lambda step: 1.0)
    assert opt.get_group_lrs() == pytest.approx([3e-4])


def test_set_lr_scales_every_group_keeping_their_ratios():
    """The deprecated single-LR API: get_lr is group 0's LR, and set_lr sets group 0 to lr with the other groups keeping
    their ratio to it."""
    opt = _make_opt(1e-3, 2e-3)
    assert opt.get_lr() == pytest.approx(1e-3)
    opt.set_lr(5e-4)
    assert opt.get_group_lrs() == pytest.approx([5e-4, 1e-3])
    assert opt.get_lr() == pytest.approx(5e-4)
    assert opt.get_initial_lr() == pytest.approx(5e-4)  # nothing recorded before, so the current LR is the base
