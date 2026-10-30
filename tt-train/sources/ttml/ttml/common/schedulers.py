# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Learning rate and optimizer parameter schedulers."""

import math
import types
from typing import Optional

from ttml.common.config import SpeedrunSchedulerConfig


class SpeedrunScheduler:
    """Linear warmup -> optional hold -> linear decay; optional beta1 warmup."""

    def __init__(self, cfg: SpeedrunSchedulerConfig):
        self.cfg = cfg

    def lr_at(self, step: int) -> float:
        s = step
        w = max(0, self.cfg.warmup_steps)
        h = max(0, self.cfg.hold_steps)
        T = max(1, self.cfg.total_steps)
        peak = self.cfg.max_lr
        min_lr = self.cfg.min_lr

        if s < w:
            # Linear warmup to lr_max over exactly w steps.
            return peak * ((s + 1) / w)
        elif s <= w + h:
            # hold at lr_max
            return peak
        else:
            # linear decay from lr_max at (w+h) to min_lr at T
            s2 = min(s, T)
            frac = (s2 - (w + h)) / max(1, (T - (w + h)))
            return peak + (min_lr - peak) * frac

    def beta1_at(self, step: int) -> Optional[float]:
        if self.cfg.beta1_start is None or self.cfg.beta1_end is None or self.cfg.beta1_warmup_steps <= 0:
            return None
        s = min(step, self.cfg.beta1_warmup_steps)
        t = s / float(self.cfg.beta1_warmup_steps)
        return (1.0 - t) * self.cfg.beta1_start + t * self.cfg.beta1_end


# ---------------------------------------------------------------------------
# Stateful LR schedulers
# ---------------------------------------------------------------------------


class _SchedulerBase:
    """Base class for stateful LR schedulers (mirrors the C++ ``LRSchedulerBase``).

    Schedulers set the LR of each optimizer param group as training goes on.
    Subclasses implement ``step``, called once per training step.

    ``optimizer`` must expose ``get_initial_group_lrs()``, ``get_group_lrs()``
    and ``set_group_lrs()`` (every subclass of ``ttml.optimizers.OptimizerBase``
    inherits them from the C++ base; Python-defined optimizers implement them).
    The base LRs come from ``get_initial_group_lrs()``, which mirrors PyTorch's
    ``param_group["initial_lr"]``: the optimizer records its LRs the first time
    a scheduler reads them, so a scheduler constructed after an earlier one has
    already scaled the LRs at construction (e.g. a chained warmup + decay pair)
    still sees the original base LRs.
    """

    def __init__(self, optimizer):
        self._optimizer = optimizer
        self._base_lrs = list(optimizer.get_initial_group_lrs())
        self._last_lrs = list(self._base_lrs)
        self._last_step = 0

    def step(self):
        """Advance the schedule by one step and apply the new LR of every param group."""
        raise NotImplementedError

    def get_last_lrs(self) -> list:
        return list(self._last_lrs)

    def get_state_dict(self) -> dict:
        return {
            "m_last_step": self._last_step,
            "m_base_lrs": list(self._base_lrs),
            "m_last_lrs": list(self._last_lrs),
        }

    def set_state_dict(self, state: dict):
        """Restore the step count, base LRs and the subclass's own state, then write the saved last LRs back.

        Raises without changing the scheduler if ``state`` is rejected.
        """
        last_step, base_lrs, last_lrs = state["m_last_step"], list(state["m_base_lrs"]), list(state["m_last_lrs"])
        num_groups = len(self._optimizer.get_group_lrs())
        if len(base_lrs) != num_groups or len(last_lrs) != num_groups:
            raise ValueError(
                f"Scheduler checkpoint has {len(base_lrs)} base LRs and {len(last_lrs)} last LRs, "
                f"but the optimizer has {num_groups} param groups"
            )
        self._restore_own_state(state)
        # The saved LRs are written back rather than recomputed, so a LambdaScheduler's callable is not called on load.
        self._optimizer.set_group_lrs(last_lrs)
        self._last_step, self._base_lrs, self._last_lrs = last_step, base_lrs, last_lrs

    def _restore_own_state(self, state: dict):
        """Read and assign the subclass's own state.

        ``set_state_dict`` calls it after reading the base state and before assigning it, so it must leave the
        scheduler unchanged if it raises.
        """


class _ClosedFormScheduler(_SchedulerBase):
    """Scheduler whose LR for each group is a function of that group's base LR and the step count
    (mirrors the C++ ``ClosedFormLRScheduler``). Subclasses implement ``_compute_lr`` for ``self._last_step``.
    """

    def _compute_lr(self, base_lr: float) -> float:
        """LR of one param group at ``self._last_step``, given the group's base LR."""
        raise NotImplementedError

    def _apply_lrs(self):
        """Set every group's LR to ``_compute_lr`` of its base LR and record them as the last LRs.

        step() calls it after advancing the step, and constructors call it once, mirroring PyTorch's
        construction-time initial step (each scheduler applies its step-0 LRs at construction).
        """
        lrs = [self._compute_lr(base_lr) for base_lr in self._base_lrs]
        self._optimizer.set_group_lrs(lrs)
        self._last_lrs = lrs

    def step(self):
        self._last_step += 1
        self._apply_lrs()


class CosineAnnealingScheduler(_ClosedFormScheduler):
    def __init__(self, optimizer, T_max: int, eta_min: float = 0.0):
        if T_max <= 0:
            raise ValueError(f"T_max = {T_max} must be greater than zero.")
        super().__init__(optimizer)
        self._T_max = T_max
        self._eta_min = eta_min
        # At step 0 the cosine factor is 1, so every group's LR is (re)set to its base LR.
        self._apply_lrs()

    def _compute_lr(self, base_lr: float) -> float:
        return self._eta_min + 0.5 * (base_lr - self._eta_min) * (
            1.0 + math.cos(math.pi * self._last_step / self._T_max)
        )

    def get_state_dict(self) -> dict:
        state = super().get_state_dict()
        state["m_T_max"] = self._T_max
        state["m_eta_min"] = self._eta_min
        return state

    def _restore_own_state(self, state: dict):
        self._T_max, self._eta_min = state["m_T_max"], state["m_eta_min"]


class StepScheduler(_ClosedFormScheduler):
    def __init__(self, optimizer, step_size: int, gamma: float = 0.1):
        if step_size <= 0:
            raise ValueError(f"step_size = {step_size} must be greater than zero.")
        if gamma <= 0.0:
            raise ValueError(f"gamma = {gamma} must be greater than zero.")
        super().__init__(optimizer)
        self._step_size = step_size
        self._gamma = gamma
        # At step 0 the factor is gamma**0 == 1, so every group's LR is (re)set to its base LR.
        self._apply_lrs()

    def _compute_lr(self, base_lr: float) -> float:
        num_decays = self._last_step // self._step_size
        return base_lr * (self._gamma**num_decays)

    def get_state_dict(self) -> dict:
        state = super().get_state_dict()
        state["m_step_size"] = self._step_size
        state["m_gamma"] = self._gamma
        return state

    def _restore_own_state(self, state: dict):
        self._step_size, self._gamma = state["m_step_size"], state["m_gamma"]


class LinearScheduler(_ClosedFormScheduler):
    def __init__(self, optimizer, start_factor: float, end_factor: float, total_steps: int):
        if total_steps <= 0:
            raise ValueError(f"total_steps = {total_steps} must be greater than zero.")
        # start_factor == 0 is rejected because the factor is applied at construction,
        # so it would make the first optimizer step a no-op (LR = 0).
        if start_factor > 1.0 or start_factor <= 0:
            raise ValueError("Starting multiplicative factor expected to be greater than 0 and less or equal to 1.")
        if end_factor > 1.0 or end_factor < 0:
            raise ValueError("Ending multiplicative factor expected to be between 0 and 1.")
        super().__init__(optimizer)
        self._start_factor = start_factor
        self._end_factor = end_factor
        self._total_steps = total_steps
        self._apply_lrs()

    def _compute_lr(self, base_lr: float) -> float:
        progress = min(self._last_step / self._total_steps, 1.0)
        factor = self._start_factor + (self._end_factor - self._start_factor) * progress
        return base_lr * factor

    def get_state_dict(self) -> dict:
        state = super().get_state_dict()
        state["m_start_factor"] = self._start_factor
        state["m_end_factor"] = self._end_factor
        state["m_total_steps"] = self._total_steps
        return state

    def _restore_own_state(self, state: dict):
        self._start_factor, self._end_factor, self._total_steps = (
            state["m_start_factor"],
            state["m_end_factor"],
            state["m_total_steps"],
        )


def linear_warmup_lambda(warmup_steps: int):
    """``step -> factor`` that ramps linearly to 1 over ``warmup_steps`` (``1 / warmup_steps`` at step 0), then stays
    at 1. No warmup if ``warmup_steps <= 0``."""

    def lr_lambda(step: int) -> float:
        if warmup_steps <= 0:
            return 1.0
        return min(1.0, (step + 1) / warmup_steps)

    return lr_lambda


class LambdaScheduler(_ClosedFormScheduler):
    def __init__(self, optimizer, lr_lambda):
        super().__init__(optimizer)
        self._lr_lambda = lr_lambda
        self._apply_lrs()

    def _compute_lr(self, base_lr: float) -> float:
        return base_lr * self._lr_lambda(self._last_step)

    # The ``lr_lambda`` itself is never pickled. If it is a plain function or
    # ``lambda`` (``types.FunctionType``) the state stores ``None`` and the
    # caller must reconstruct the scheduler with the same callable before
    # restoring state. If it is a callable *object* (e.g. an instance of a
    # class with ``__call__``), its ``__dict__`` is saved so that any per-
    # instance state is preserved across save/load.
    def get_state_dict(self) -> dict:
        state = super().get_state_dict()
        if isinstance(self._lr_lambda, types.FunctionType):
            state["m_lr_lambda"] = None
        else:
            state["m_lr_lambda"] = self._lr_lambda.__dict__.copy()
        return state

    def _restore_own_state(self, state: dict):
        lr_lambda_state = state.get("m_lr_lambda")
        if lr_lambda_state is not None:
            self._lr_lambda.__dict__.update(lr_lambda_state)


class SequentialScheduler(_SchedulerBase):
    """Chain multiple LR schedulers, each running for a fixed number of steps.

    ``milestones[i]`` is the **number of steps** for child ``schedulers[i]``
    (matches the C++ ``ttml::schedulers::SequentialScheduler`` semantics and
    differs from PyTorch's ``SequentialLR``, whose milestones are cumulative
    epoch indices).

    Once a child has been stepped ``milestones[i]`` times, the chain advances
    to the next child. After every child has been exhausted, ``step()``
    becomes a no-op.

    On save/load, **every** child's state is persisted (matching PyTorch's
    ``SequentialLR.state_dict`` and the C++ ``SequentialScheduler::get_state_dict``)
    under per-child key prefixes ``scheduler_{i}/``.
    """

    def __init__(self, optimizer, schedulers, milestones):
        if len(schedulers) == 0:
            raise ValueError("SequentialScheduler requires at least one scheduler.")
        if len(schedulers) != len(milestones):
            raise ValueError(f"len(schedulers) ({len(schedulers)}) must equal len(milestones) ({len(milestones)}).")
        for i, s in enumerate(schedulers):
            if s is None:
                raise ValueError(f"schedulers[{i}] is None.")

        super().__init__(optimizer)
        self._schedulers = list(schedulers)
        self._milestones = list(milestones)
        self._current_scheduler_index = 0
        self._current_step_in_scheduler = 0
        # The children were constructed back-to-back on the same optimizer, so
        # the optimizer currently holds the LAST child's construction-time LRs.
        # Only the first child is active; redo its initial step.
        self._apply_lrs()

    def _apply_lrs(self):
        """Write the current child's last LRs to the optimizer and record them.

        Called by the constructor, right after the current child computed its LRs, so its last LRs are up to date.
        """
        self._last_lrs = self._schedulers[self._current_scheduler_index].get_last_lrs()
        self._optimizer.set_group_lrs(self._last_lrs)

    def step(self):
        # Move on to the next child only when it is about to take its first step, so
        # ``_current_scheduler_index`` always points at the child that wrote the
        # optimizer's current LRs.
        if self._current_step_in_scheduler >= self._milestones[self._current_scheduler_index]:
            if self._current_scheduler_index + 1 >= len(self._schedulers):
                return  # every child has run all its steps
            self._current_scheduler_index += 1
            self._current_step_in_scheduler = 0

        current = self._schedulers[self._current_scheduler_index]
        current.step()
        self._current_step_in_scheduler += 1
        self._last_lrs = list(current.get_last_lrs())

    def get_state_dict(self) -> dict:
        # Top-level keys mirror the C++ ``SequentialScheduler``.
        state = super().get_state_dict()
        state["m_current_step_in_scheduler"] = self._current_step_in_scheduler
        state["m_current_scheduler_index"] = self._current_scheduler_index
        for i, child in enumerate(self._schedulers):
            prefix = f"scheduler_{i}/"
            for k, v in child.get_state_dict().items():
                state[prefix + k] = v
        return state

    def _restore_own_state(self, state: dict):
        # NOTE: assumes the destination ``SequentialScheduler`` was constructed
        # with the same number of children as the source. A size mismatch will
        # cause the child's ``set_state_dict`` to raise ``KeyError`` because
        # expected keys won't be present in the per-child sub-dict.
        index = state["m_current_scheduler_index"]
        step_in_scheduler = state["m_current_step_in_scheduler"]
        if not 0 <= index < len(self._schedulers):
            raise ValueError(
                f"SequentialScheduler state has scheduler index {index}, but the chain has "
                f"{len(self._schedulers)} schedulers"
            )
        if not 0 <= step_in_scheduler <= self._milestones[index]:
            raise ValueError(
                f"SequentialScheduler state has {step_in_scheduler} steps in scheduler {index}, "
                f"outside [0, {self._milestones[index]}]"
            )
        # A later child can still reject its state after earlier ones were restored (and wrote their LRs to the
        # optimizer), so snapshot everything and roll back on failure.
        snapshots = [child.get_state_dict() for child in self._schedulers]
        live_lrs = list(self._optimizer.get_group_lrs())
        try:
            for i, child in enumerate(self._schedulers):
                prefix = f"scheduler_{i}/"
                child_state = {k[len(prefix) :]: v for k, v in state.items() if k.startswith(prefix)}
                child.set_state_dict(child_state)
        except Exception:
            for child, snapshot in zip(self._schedulers, snapshots):
                child.set_state_dict(snapshot)
            self._optimizer.set_group_lrs(live_lrs)
            raise
        # Only after the children accepted their states, so a rejected state dict leaves the chain's position unchanged.
        self._current_scheduler_index = index
        self._current_step_in_scheduler = step_in_scheduler
