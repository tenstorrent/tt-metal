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
)


BASE_LR = 0.1
N_STEPS = 60


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_opt(lr=BASE_LR):
    params = ttml.NamedParameters()
    return ttml.optimizers.create_optimizer({"type": "AdamW", "lr": lr}, params)


def _step(opt, sched):
    sched.step()
    assert opt.get_lr() == pytest.approx(sched.get_last_lr(), abs=1e-8)


def _make_torch_opt(lr=BASE_LR):
    dummy = torch.nn.Parameter(torch.tensor(0.0))
    return torch.optim.SGD([dummy], lr=lr), dummy


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
        ref_lrs.append(ref_sched.get_current_lr())

    opt1, sched1 = make_sched_fn()  # noqa: F841
    for _ in range(half):
        _step(opt1, sched1)
    state = sched1.get_state_dict()

    opt2, sched2 = make_sched_fn()  # noqa: F841
    sched2.set_state_dict(state)
    resumed_lrs = []
    for _ in range(n_steps - half):
        _step(opt2, sched2)
        resumed_lrs.append(sched2.get_current_lr())

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
        torch_opt, _ = _make_torch_opt()
        torch_sched = torch.optim.lr_scheduler.CosineAnnealingLR(torch_opt, T_max=T_MAX, eta_min=ETA_MIN)

        for step in range(1, N_STEPS + 1):
            _step(opt, sched)
            torch_opt.step()
            torch_sched.step()
            assert sched.get_current_lr() == pytest.approx(torch_opt.param_groups[0]["lr"], abs=1e-5), f"step {step}"

    def test_lr_at_T_max_is_eta_min(self):
        """At step T_max the LR must equal eta_min."""
        opt, sched = _make_cosine()  # noqa: F841
        for _ in range(T_MAX):
            _step(opt, sched)
        assert sched.get_current_lr() == pytest.approx(ETA_MIN, abs=1e-6)

    def test_lr_at_2T_max_is_base_lr(self):
        """At step 2*T_max the LR must return to base_lr."""
        opt, sched = _make_cosine()  # noqa: F841
        for _ in range(2 * T_MAX):
            _step(opt, sched)
        assert sched.get_current_lr() == pytest.approx(BASE_LR, abs=1e-5)

    def test_get_last_lr_matches_current(self):
        opt, sched = _make_cosine()  # noqa: F841
        for _ in range(7):
            _step(opt, sched)
        assert sched.get_last_lr() == pytest.approx(sched.get_current_lr(), abs=1e-8)

    def test_eta_min_zero_default(self):
        opt = _make_opt()  # noqa: F841
        sched = CosineAnnealingScheduler(opt, T_MAX)
        for _ in range(T_MAX):
            _step(opt, sched)
        assert sched.get_current_lr() == pytest.approx(0.0, abs=1e-6)


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
        assert "m_last_lr" in state
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
        """Reproduces the bug where ``_base_lr`` is not part of the saved state.

        Real-world resume sequence: the optimizer is loaded from a checkpoint
        with its CURRENT (already-decayed) LR, and the scheduler is then
        reconstructed on top of it. Without persisting ``_base_lr`` in the
        state dict, the new scheduler captures the decayed LR as its base,
        and every subsequent step computes the cosine with the wrong
        amplitude ``0.5 * (decayed_lr - eta_min)`` instead of
        ``0.5 * (BASE_LR - eta_min)``.
        """
        src_opt, src = _make_cosine()  # noqa: F841
        for _ in range(5):
            _step(src_opt, src)
        decayed_lr = src_opt.get_lr()
        assert decayed_lr != pytest.approx(BASE_LR)  # sanity: the LR really did decay
        state = src.get_state_dict()

        # Mimic checkpoint resume: optimizer.get_lr() now returns the decayed value.
        dst_opt = _make_opt(decayed_lr)
        dst = CosineAnnealingScheduler(dst_opt, T_MAX, ETA_MIN)
        dst.set_state_dict(state)

        # Direct check: the scheduler must remember its ORIGINAL base lr,
        # not whatever happens to be on the optimizer right now.
        assert dst._base_lr == pytest.approx(BASE_LR, abs=1e-7)

        # Behavioral check: the next step must produce the same LR as the source.
        src.step()
        dst.step()
        assert dst.get_last_lr() == pytest.approx(src.get_last_lr(), abs=1e-7)

    def test_restore_before_first_step_is_noop(self):
        opt_a, sched_a = _make_cosine()  # noqa: F841
        opt_b, sched_b = _make_cosine()  # noqa: F841
        sched_b.set_state_dict(sched_a.get_state_dict())
        lrs_a, lrs_b = [], []
        for _ in range(10):
            _step(opt_a, sched_a)
            _step(opt_b, sched_b)
            lrs_a.append(sched_a.get_current_lr())
            lrs_b.append(sched_b.get_current_lr())
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
        torch_opt, _ = _make_torch_opt()
        torch_sched = torch.optim.lr_scheduler.StepLR(torch_opt, step_size=STEP_SIZE, gamma=GAMMA)

        for step in range(1, N_STEPS + 1):
            _step(opt, sched)
            torch_opt.step()
            torch_sched.step()
            assert sched.get_current_lr() == pytest.approx(torch_opt.param_groups[0]["lr"], abs=1e-6), f"step {step}"

    def test_lr_unchanged_before_boundary(self):
        """LR must not decay until step_size steps have been taken."""
        opt, sched = _make_step()  # noqa: F841
        for _ in range(STEP_SIZE - 1):
            _step(opt, sched)
        assert sched.get_current_lr() == pytest.approx(BASE_LR, abs=1e-7)

    def test_lr_decays_at_boundary(self):
        """LR must be base_lr * gamma exactly at step_size."""
        opt, sched = _make_step()  # noqa: F841
        for _ in range(STEP_SIZE):
            _step(opt, sched)
        assert sched.get_current_lr() == pytest.approx(BASE_LR * GAMMA, abs=1e-7)

    def test_get_last_lr_matches_current(self):
        opt, sched = _make_step()  # noqa: F841
        for _ in range(7):
            _step(opt, sched)
        assert sched.get_last_lr() == pytest.approx(sched.get_current_lr(), abs=1e-8)


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
        assert "m_last_lr" in state
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
        """Reproduces the ``_base_lr``-not-saved bug for StepScheduler.

        StepScheduler computes ``base_lr * gamma**num_decays``; if ``base_lr``
        comes from a freshly-restored (decayed) optimizer instead of the
        original training value, the LR is silently mis-scaled.
        """
        # Step over a boundary so the LR really has dropped before save.
        src_opt, src = _make_step()  # noqa: F841
        for _ in range(STEP_SIZE + 1):  # past the first boundary
            _step(src_opt, src)
        decayed_lr = src_opt.get_lr()
        assert decayed_lr != pytest.approx(BASE_LR)  # sanity
        state = src.get_state_dict()

        dst_opt = _make_opt(decayed_lr)
        dst = StepScheduler(dst_opt, STEP_SIZE, GAMMA)
        dst.set_state_dict(state)

        assert dst._base_lr == pytest.approx(BASE_LR, abs=1e-7)
        src.step()
        dst.step()
        assert dst.get_last_lr() == pytest.approx(src.get_last_lr(), abs=1e-7)

    def test_restore_before_first_step_is_noop(self):
        opt_a, sched_a = _make_step()  # noqa: F841
        opt_b, sched_b = _make_step()  # noqa: F841
        sched_b.set_state_dict(sched_a.get_state_dict())
        lrs_a, lrs_b = [], []
        for _ in range(10):
            _step(opt_a, sched_a)
            _step(opt_b, sched_b)
            lrs_a.append(sched_a.get_current_lr())
            lrs_b.append(sched_b.get_current_lr())
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
    def test_lr_trajectory(self):
        """Must match torch.optim.lr_scheduler.LinearLR step-by-step."""
        opt, sched = _make_linear()  # noqa: F841
        torch_opt, _ = _make_torch_opt()
        torch_sched = torch.optim.lr_scheduler.LinearLR(
            torch_opt, start_factor=START_FACTOR, end_factor=END_FACTOR, total_iters=TOTAL_STEPS
        )

        for step in range(1, N_STEPS + 1):
            _step(opt, sched)
            torch_opt.step()
            torch_sched.step()
            assert sched.get_current_lr() == pytest.approx(torch_opt.param_groups[0]["lr"], abs=1e-5), f"step {step}"

    def test_lr_at_total_steps_is_end_factor(self):
        """At total_steps the LR must equal base_lr * end_factor."""
        opt, sched = _make_linear()  # noqa: F841
        for _ in range(TOTAL_STEPS):
            _step(opt, sched)
        assert sched.get_current_lr() == pytest.approx(BASE_LR * END_FACTOR, abs=1e-6)

    def test_lr_clamps_after_total_steps(self):
        """Beyond total_steps the LR must stay at base_lr * end_factor."""
        opt, sched = _make_linear()  # noqa: F841
        for _ in range(TOTAL_STEPS + 20):
            _step(opt, sched)
        assert sched.get_current_lr() == pytest.approx(BASE_LR * END_FACTOR, abs=1e-6)

    def test_get_last_lr_matches_current(self):
        opt, sched = _make_linear()  # noqa: F841
        for _ in range(7):
            _step(opt, sched)
        assert sched.get_last_lr() == pytest.approx(sched.get_current_lr(), abs=1e-8)


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
        assert "m_last_lr" in state
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
        """Reproduces the ``_base_lr``-not-saved bug for LinearScheduler.

        LinearScheduler computes ``base_lr * factor``; if ``base_lr`` comes
        from a freshly-restored (decayed) optimizer instead of the original
        training value, the LR is silently mis-scaled.
        """
        src_opt, src = _make_linear()  # noqa: F841
        for _ in range(5):
            _step(src_opt, src)
        decayed_lr = src_opt.get_lr()
        assert decayed_lr != pytest.approx(BASE_LR)
        state = src.get_state_dict()

        dst_opt = _make_opt(decayed_lr)
        dst = LinearScheduler(dst_opt, START_FACTOR, END_FACTOR, TOTAL_STEPS)
        dst.set_state_dict(state)

        assert dst._base_lr == pytest.approx(BASE_LR, abs=1e-7)
        src.step()
        dst.step()
        assert dst.get_last_lr() == pytest.approx(src.get_last_lr(), abs=1e-7)

    def test_restore_before_first_step_is_noop(self):
        opt_a, sched_a = _make_linear()  # noqa: F841
        opt_b, sched_b = _make_linear()  # noqa: F841
        sched_b.set_state_dict(sched_a.get_state_dict())
        lrs_a, lrs_b = [], []
        for _ in range(10):
            _step(opt_a, sched_a)
            _step(opt_b, sched_b)
            lrs_a.append(sched_a.get_current_lr())
            lrs_b.append(sched_b.get_current_lr())
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
    def test_lr_trajectory(self):
        """Must match torch.optim.lr_scheduler.LambdaLR step-by-step."""
        opt, sched = _make_lambda()  # noqa: F841
        torch_opt, _ = _make_torch_opt()
        torch_sched = torch.optim.lr_scheduler.LambdaLR(torch_opt, lr_lambda=_LAMBDA)

        for step in range(1, N_STEPS + 1):
            _step(opt, sched)
            torch_opt.step()
            torch_sched.step()
            assert sched.get_current_lr() == pytest.approx(torch_opt.param_groups[0]["lr"], abs=1e-6), f"step {step}"

    def test_lr_formula(self):
        """LR after N steps must equal base_lr * lambda(N) exactly."""
        opt, sched = _make_lambda()  # noqa: F841
        for n in range(1, 21):
            _step(opt, sched)
            expected = BASE_LR * (_DECAY**n)
            assert sched.get_current_lr() == pytest.approx(expected, abs=1e-6), f"step {n}"

    def test_constant_lambda_keeps_lr(self):
        """A lambda that always returns 1.0 must leave the LR unchanged."""
        opt = _make_opt()  # noqa: F841
        sched = LambdaScheduler(opt, lambda _: 1.0)
        for _ in range(10):
            _step(opt, sched)
        assert sched.get_current_lr() == pytest.approx(BASE_LR, abs=1e-7)

    def test_get_last_lr_matches_current(self):
        opt, sched = _make_lambda()  # noqa: F841
        for _ in range(7):
            _step(opt, sched)
        assert sched.get_last_lr() == pytest.approx(sched.get_current_lr(), abs=1e-8)


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
        assert "m_last_lr" in state
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
        assert saved == {"decay": 0.9, "calls": 3}

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
        assert saved == {"n": 1}

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
        # The source callable has been called 4 times.
        assert src._lr_lambda.n == 4
        state = src.get_state_dict()

        dst_opt = _make_opt()
        dst = LambdaScheduler(dst_opt, _Counter(start=0))
        dst.set_state_dict(state)
        # Destination callable's ``n`` must be restored to 4.
        assert dst._lr_lambda.n == 4

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
        """Reproduces the ``_base_lr``-not-saved bug for LambdaScheduler.

        LambdaScheduler computes ``base_lr * lr_lambda(step)``; if ``base_lr``
        comes from a freshly-restored (decayed) optimizer instead of the
        original training value, the LR is silently mis-scaled.
        """
        src_opt, src = _make_lambda()  # noqa: F841
        for _ in range(5):
            _step(src_opt, src)
        decayed_lr = src_opt.get_lr()
        assert decayed_lr != pytest.approx(BASE_LR)
        state = src.get_state_dict()

        dst_opt = _make_opt(decayed_lr)
        dst = LambdaScheduler(dst_opt, _LAMBDA)
        dst.set_state_dict(state)

        assert dst._base_lr == pytest.approx(BASE_LR, abs=1e-7)
        src.step()
        dst.step()
        assert dst.get_last_lr() == pytest.approx(src.get_last_lr(), abs=1e-7)

    def test_restore_before_first_step_is_noop(self):
        opt_a, sched_a = _make_lambda()  # noqa: F841
        opt_b, sched_b = _make_lambda()  # noqa: F841
        sched_b.set_state_dict(sched_a.get_state_dict())
        lrs_a, lrs_b = [], []
        for _ in range(10):
            _step(opt_a, sched_a)
            _step(opt_b, sched_b)
            lrs_a.append(sched_a.get_current_lr())
            lrs_b.append(sched_b.get_current_lr())
        assert lrs_a == pytest.approx(lrs_b, abs=1e-8)


# ---------------------------------------------------------------------------
# SequentialScheduler
# ---------------------------------------------------------------------------
#
# NOTE: ``SequentialScheduler`` follows the C++ ttml semantics where each
# ``milestones[i]`` is the *number of steps* allocated to child ``i``

CHILD0_STEPS = 10  # warmup length
CHILD1_STEPS = 20  # decay length


def _make_sequential(base_lr=BASE_LR):
    """Linear warmup (0.0 -> 1.0 over CHILD0_STEPS) then Linear decay
    (1.0 -> 0.1 over CHILD1_STEPS), chained via SequentialScheduler."""
    opt = _make_opt(base_lr)
    children = [
        LinearScheduler(opt, start_factor=0.0, end_factor=1.0, total_steps=CHILD0_STEPS),
        LinearScheduler(opt, start_factor=1.0, end_factor=0.1, total_steps=CHILD1_STEPS),
    ]
    return opt, SequentialScheduler(opt, children, milestones=[CHILD0_STEPS, CHILD1_STEPS])


class TestSequentialSchedulerBehavior:
    def test_warmup_then_decay_trajectory(self):
        """Closed-form check of the warmup-then-decay LR trajectory."""
        opt, sched = _make_sequential()  # noqa: F841
        for i in range(1, CHILD0_STEPS + 1):
            _step(opt, sched)
            expected = BASE_LR * (i / CHILD0_STEPS)
            assert sched.get_current_lr() == pytest.approx(expected, abs=1e-7), f"warmup step {i}"
        for j in range(1, CHILD1_STEPS + 1):
            _step(opt, sched)
            expected = BASE_LR * (1.0 + (0.1 - 1.0) * j / CHILD1_STEPS)
            assert sched.get_current_lr() == pytest.approx(expected, abs=1e-7), f"decay step {j}"

    def test_linear_warmup_then_cosine_decay(self):
        """Common training schedule: linear warmup -> cosine annealing decay.

        - Warmup: ``warmup_steps`` linear steps from 0 to BASE_LR.
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
            LinearScheduler(opt, start_factor=0.0, end_factor=1.0, total_steps=warmup_steps),
            CosineAnnealingScheduler(opt, T_max=decay_steps, eta_min=eta_min),
        ]
        sched = SequentialScheduler(opt, children, milestones=[warmup_steps, decay_steps])

        # Warmup phase.
        for i in range(1, warmup_steps + 1):
            _step(opt, sched)
            expected = BASE_LR * (i / warmup_steps)
            assert sched.get_current_lr() == pytest.approx(expected, abs=1e-7), f"warmup step {i}"
        # End of warmup hands off cleanly at exactly BASE_LR.
        assert opt.get_lr() == pytest.approx(BASE_LR, abs=1e-7)

        # Decay phase: cosine annealing from BASE_LR -> eta_min.
        for j in range(1, decay_steps + 1):
            _step(opt, sched)
            expected = eta_min + 0.5 * (BASE_LR - eta_min) * (1.0 + math.cos(math.pi * j / decay_steps))
            assert sched.get_current_lr() == pytest.approx(expected, abs=1e-7), f"decay step {j}"
        # cos(pi) = -1 so the final LR is exactly ``eta_min``.
        assert opt.get_lr() == pytest.approx(eta_min, abs=1e-7)

        # Past the end of the chain: step() is a no-op and the LR is held.
        for _ in range(3):
            sched.step()
            assert opt.get_lr() == pytest.approx(eta_min, abs=1e-7)

    def test_step_is_noop_after_all_schedulers_exhausted(self):
        opt, sched = _make_sequential()  # noqa: F841
        total = CHILD0_STEPS + CHILD1_STEPS
        for _ in range(total):
            sched.step()
        # All children exhausted; further steps must be no-ops.
        final_lr = opt.get_lr()
        for _ in range(5):
            sched.step()
            assert opt.get_lr() == final_lr

    def test_get_last_lr_delegates_to_active_child(self):
        opt, sched = _make_sequential()  # noqa: F841
        sched.step()  # one step of warmup
        assert sched.get_last_lr() == pytest.approx(BASE_LR * (1 / CHILD0_STEPS), abs=1e-7)

    def test_get_last_lr_returns_stored_value_after_exhaustion(self):
        opt, sched = _make_sequential()  # noqa: F841
        for _ in range(CHILD0_STEPS + CHILD1_STEPS):
            sched.step()
        # End of decay: lr = base_lr * 0.1
        assert sched.get_last_lr() == pytest.approx(BASE_LR * 0.1, abs=1e-7)
        sched.step()  # no-op
        assert sched.get_last_lr() == pytest.approx(BASE_LR * 0.1, abs=1e-7)

    def test_requires_at_least_one_scheduler(self):
        opt = _make_opt()
        with pytest.raises(ValueError):
            SequentialScheduler(opt, [], [])

    def test_milestones_length_must_match_schedulers(self):
        opt = _make_opt()
        child = LinearScheduler(opt, 0.0, 1.0, 5)
        with pytest.raises(ValueError):
            SequentialScheduler(opt, [child], [5, 10])

    def test_none_scheduler_rejected(self):
        opt = _make_opt()
        with pytest.raises(ValueError):
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
        assert "m_last_lr" in state

    def test_state_dict_contains_per_child_keys(self):
        opt, sched = _make_sequential()  # noqa: F841
        state = sched.get_state_dict()
        # Every wrapped child saves its full state under the per-index prefix
        # ``scheduler_{i}/`` -- this is the behavior that fixes the C++ design
        # bug where only the currently-active child was saved.
        for child_idx in (0, 1):
            prefix = f"scheduler_{child_idx}/"
            assert f"{prefix}m_last_step" in state
            assert f"{prefix}m_last_lr" in state
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

        assert dst._schedulers[0]._start_factor == pytest.approx(0.0, abs=1e-12)
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
            lrs_a.append(sched_a.get_current_lr())
            lrs_b.append(sched_b.get_current_lr())
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
            LinearScheduler(opt, start_factor=0.0, end_factor=1.0, total_steps=warmup_steps),
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
            LinearScheduler(dst_opt, start_factor=0.0, end_factor=1.0, total_steps=warmup_steps),
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
            assert dst.get_current_lr() == pytest.approx(src.get_current_lr(), abs=1e-7), f"step {k}"

        # Both chains are now exhausted; final LR is eta_min on both sides.
        assert src.get_last_lr() == pytest.approx(eta_min, abs=1e-7)
        assert dst.get_last_lr() == pytest.approx(eta_min, abs=1e-7)

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
            assert dst.get_current_lr() == pytest.approx(src.get_current_lr(), abs=1e-7), f"step {k}"

        # Both chains land at exactly eta_min at the end of decay.
        assert src.get_last_lr() == pytest.approx(eta_min, abs=1e-7)
        assert dst.get_last_lr() == pytest.approx(eta_min, abs=1e-7)

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

        # Chain is exhausted: index advanced past the last child, step-in-
        # scheduler reset to 0, last_lr is eta_min.
        state = src.get_state_dict()
        assert state["m_current_scheduler_index"] == 2
        assert state["m_current_step_in_scheduler"] == 0
        assert state["m_last_lr"] == pytest.approx(eta_min, abs=1e-7)
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

        # Top-level: dst is exhausted too.
        assert dst._current_scheduler_index == 2
        assert dst._current_step_in_scheduler == 0
        assert dst.get_last_lr() == pytest.approx(eta_min, abs=1e-7)

        # Linear child's full state preserved.
        assert dst._schedulers[0]._start_factor == pytest.approx(0.0, abs=1e-12)
        assert dst._schedulers[0]._end_factor == pytest.approx(1.0, abs=1e-12)
        assert dst._schedulers[0]._total_steps == warmup_steps
        assert dst._schedulers[0]._last_step == warmup_steps

        # Cosine child's full state preserved (despite dst constructing it with
        # wildly different hyperparameters).
        assert dst._schedulers[1]._T_max == decay_steps
        assert dst._schedulers[1]._eta_min == pytest.approx(eta_min, abs=1e-12)
        assert dst._schedulers[1]._last_step == decay_steps

        # Further step() calls on dst are no-ops; get_last_lr() stays put.
        final_lr = dst.get_last_lr()
        for _ in range(3):
            dst.step()
            assert dst.get_last_lr() == pytest.approx(final_lr, abs=1e-12)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
