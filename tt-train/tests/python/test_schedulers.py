# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for ttml LR schedulers."""

import pytest
import torch
import ttml


BASE_LR = 0.1
ETA_MIN = 1e-4
T_MAX = 20
N_STEPS = 50  # intentionally more than one cosine cycle


def _make_ttml_scheduler(base_lr=BASE_LR, T_max=T_MAX, eta_min=ETA_MIN):
    params = ttml.NamedParameters()
    opt = ttml.optimizers.create_optimizer({"type": "AdamW", "lr": base_lr}, params)
    scheduler = ttml.schedulers.CosineAnnealingScheduler(opt, T_max, eta_min)
    return opt, scheduler


def _make_torch_scheduler(base_lr=BASE_LR, T_max=T_MAX, eta_min=ETA_MIN):
    dummy = torch.nn.Parameter(torch.tensor(0.0))
    opt = torch.optim.SGD([dummy], lr=base_lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=T_max, eta_min=eta_min)
    return opt, scheduler


class TestCosineAnnealingMatchesPyTorch:
    def test_lr_trajectory(self):
        """Every step's LR must match PyTorch CosineAnnealingLR to within float32 precision."""
        opt, sched = _make_ttml_scheduler()  # noqa: F841 — opt must outlive sched (raw C++ pointer)
        torch_opt, torch_sched = _make_torch_scheduler()

        for step in range(1, N_STEPS + 1):
            sched.step()
            torch_sched.step()

            our_lr = sched.get_current_lr()
            ref_lr = torch_opt.param_groups[0]["lr"]

            assert our_lr == pytest.approx(
                ref_lr, abs=1e-5
            ), f"step {step}: our LR {our_lr:.8f} != PyTorch LR {ref_lr:.8f}"

    def test_lr_at_T_max_is_eta_min(self):
        """At step T_max the LR must equal eta_min."""
        opt, sched = _make_ttml_scheduler()  # noqa: F841
        for _ in range(T_MAX):
            sched.step()
        assert sched.get_current_lr() == pytest.approx(ETA_MIN, abs=1e-6)

    def test_lr_at_2T_max_is_base_lr(self):
        """At step 2*T_max the LR must return to base_lr (second cycle peak)."""
        opt, sched = _make_ttml_scheduler()  # noqa: F841
        for _ in range(2 * T_MAX):
            sched.step()
        assert sched.get_current_lr() == pytest.approx(BASE_LR, abs=1e-5)

    def test_get_last_lr_matches_current(self):
        """get_last_lr() must agree with get_current_lr() immediately after step()."""
        opt, sched = _make_ttml_scheduler()  # noqa: F841
        for _ in range(7):
            sched.step()
        assert sched.get_last_lr() == pytest.approx(sched.get_current_lr(), abs=1e-8)

    def test_eta_min_zero_default(self):
        """Default eta_min=0: LR at T_max must reach 0."""
        params = ttml.NamedParameters()
        opt = ttml.optimizers.create_optimizer({"type": "AdamW", "lr": BASE_LR}, params)
        sched = ttml.schedulers.CosineAnnealingScheduler(opt, T_MAX)
        for _ in range(T_MAX):
            sched.step()
        assert sched.get_current_lr() == pytest.approx(0.0, abs=1e-6)


class TestCosineAnnealingStateDict:
    def test_roundtrip_continues_correctly(self):
        """Step N/2, save state, restore into a fresh scheduler, step N/2 more.

        The resumed LRs must exactly match those of an uninterrupted reference run.
        """
        half = N_STEPS // 2

        # Uninterrupted reference
        ref_opt, ref_sched = _make_ttml_scheduler()  # noqa: F841
        ref_lrs = []
        for _ in range(N_STEPS):
            ref_sched.step()
            ref_lrs.append(ref_sched.get_current_lr())

        # First half
        opt1, sched1 = _make_ttml_scheduler()  # noqa: F841
        for _ in range(half):
            sched1.step()

        state = sched1.get_state_dict()

        # Restore into a fresh scheduler with identical hyperparameters
        opt2, sched2 = _make_ttml_scheduler()  # noqa: F841
        sched2.set_state_dict(state)

        resumed_lrs = []
        for _ in range(N_STEPS - half):
            sched2.step()
            resumed_lrs.append(sched2.get_current_lr())

        assert resumed_lrs == pytest.approx(ref_lrs[half:], abs=1e-7)

    def test_state_dict_contains_expected_keys(self):
        """State dict must contain the step counter and last LR."""
        opt, sched = _make_ttml_scheduler()  # noqa: F841
        for _ in range(5):
            sched.step()
        state = sched.get_state_dict()
        assert "m_last_step" in state
        assert "m_last_lr" in state

    def test_state_dict_step_count(self):
        """m_last_step in the state dict must reflect the number of step() calls."""
        opt, sched = _make_ttml_scheduler()  # noqa: F841
        n = 13
        for _ in range(n):
            sched.step()
        state = sched.get_state_dict()
        assert state["m_last_step"] == n

    def test_restore_before_first_step(self):
        """Restoring a state dict from a freshly constructed (zero-step) scheduler is a no-op."""
        opt_a, sched_a = _make_ttml_scheduler()  # noqa: F841
        state = sched_a.get_state_dict()

        opt_b, sched_b = _make_ttml_scheduler()  # noqa: F841
        sched_b.set_state_dict(state)

        lrs_a, lrs_b = [], []
        for _ in range(10):
            sched_a.step()
            sched_b.step()
            lrs_a.append(sched_a.get_current_lr())
            lrs_b.append(sched_b.get_current_lr())

        assert lrs_a == pytest.approx(lrs_b, abs=1e-8)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
