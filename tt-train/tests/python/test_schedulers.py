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

import pytest
import torch
import ttml
from ttml.common.schedulers import CosineAnnealingScheduler, LambdaScheduler, LinearScheduler, StepScheduler


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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
