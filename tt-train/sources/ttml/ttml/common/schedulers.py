# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Learning rate and optimizer parameter schedulers."""
import math
import types
from typing import Optional

from ttml.common.config import SchedulerConfig


class SpeedrunScheduler:
    """Linear warmup -> optional hold -> linear decay; optional beta1 warmup."""

    def __init__(self, cfg: SchedulerConfig):
        self.cfg = cfg

    def lr_at(self, step: int) -> float:
        s = step
        w = max(0, self.cfg.warmup_steps)
        h = max(0, self.cfg.hold_steps)
        T = max(1, self.cfg.total_steps)
        peak = self.cfg.max_lr
        min_lr = self.cfg.min_lr

        if s <= w:
            # linear warmup 0 -> lr_max
            return peak * (s / max(1, w))
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


class OptimParamSetter:
    """Helper to set optimizer parameters during training."""

    def __init__(self, optim):
        self.optim = optim

    def set_lr(self, lr: float):
        self.optim.set_lr(float(lr))


# ---------------------------------------------------------------------------
# Stateful LR schedulers
# ---------------------------------------------------------------------------


class _SchedulerBase:
    def __init__(self, optimizer):
        self._optimizer = optimizer
        self._base_lr = optimizer.get_lr()
        self._last_step = 0
        self._last_lr = self._base_lr

    def step(self):
        raise NotImplementedError

    def get_last_lr(self) -> float:
        return self._last_lr

    def get_current_lr(self) -> float:
        return self._optimizer.get_lr()

    def get_state_dict(self) -> dict:
        return {"m_last_step": self._last_step, "m_last_lr": self._last_lr}

    def set_state_dict(self, state: dict):
        self._last_step = state["m_last_step"]
        self._last_lr = state["m_last_lr"]


class CosineAnnealingScheduler(_SchedulerBase):
    def __init__(self, optimizer, T_max: int, eta_min: float = 0.0):
        if T_max <= 0:
            raise ValueError(f"T_max = {T_max} must be greater than zero.")
        super().__init__(optimizer)
        self._T_max = T_max
        self._eta_min = eta_min

    def step(self):
        self._last_step += 1
        new_lr = self._eta_min + 0.5 * (self._base_lr - self._eta_min) * (
            1.0 + math.cos(math.pi * self._last_step / self._T_max)
        )
        self._optimizer.set_lr(new_lr)
        self._last_lr = new_lr

    def get_state_dict(self) -> dict:
        state = super().get_state_dict()
        state["m_T_max"] = self._T_max
        state["m_eta_min"] = self._eta_min
        return state

    def set_state_dict(self, state: dict):
        super().set_state_dict(state)
        self._T_max = state["m_T_max"]
        self._eta_min = state["m_eta_min"]


class StepScheduler(_SchedulerBase):
    def __init__(self, optimizer, step_size: int, gamma: float = 0.1):
        if step_size <= 0:
            raise ValueError(f"step_size = {step_size} must be greater than zero.")
        if gamma <= 0.0:
            raise ValueError(f"gamma = {gamma} must be greater than zero.")
        super().__init__(optimizer)
        self._step_size = step_size
        self._gamma = gamma

    def step(self):
        self._last_step += 1
        num_decays = self._last_step // self._step_size
        new_lr = self._base_lr * (self._gamma**num_decays)
        self._optimizer.set_lr(new_lr)
        self._last_lr = new_lr

    def get_state_dict(self) -> dict:
        state = super().get_state_dict()
        state["m_step_size"] = self._step_size
        state["m_gamma"] = self._gamma
        return state

    def set_state_dict(self, state: dict):
        super().set_state_dict(state)
        self._step_size = state["m_step_size"]
        self._gamma = state["m_gamma"]


class LinearScheduler(_SchedulerBase):
    def __init__(self, optimizer, start_factor: float, end_factor: float, total_steps: int):
        if total_steps <= 0:
            raise ValueError(f"total_steps = {total_steps} must be greater than zero.")
        super().__init__(optimizer)
        self._start_factor = start_factor
        self._end_factor = end_factor
        self._total_steps = total_steps

    def step(self):
        self._last_step += 1
        progress = min(self._last_step / self._total_steps, 1.0)
        factor = self._start_factor + (self._end_factor - self._start_factor) * progress
        new_lr = self._base_lr * factor
        self._optimizer.set_lr(new_lr)
        self._last_lr = new_lr

    def get_state_dict(self) -> dict:
        state = super().get_state_dict()
        state["m_start_factor"] = self._start_factor
        state["m_end_factor"] = self._end_factor
        state["m_total_steps"] = self._total_steps
        return state

    def set_state_dict(self, state: dict):
        super().set_state_dict(state)
        self._start_factor = state["m_start_factor"]
        self._end_factor = state["m_end_factor"]
        self._total_steps = state["m_total_steps"]


class LambdaScheduler(_SchedulerBase):
    def __init__(self, optimizer, lr_lambda):
        super().__init__(optimizer)
        self._lr_lambda = lr_lambda

    def step(self):
        self._last_step += 1
        new_lr = self._base_lr * self._lr_lambda(self._last_step)
        self._optimizer.set_lr(new_lr)
        self._last_lr = new_lr

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

    def set_state_dict(self, state: dict):
        super().set_state_dict(state)
        lr_lambda_state = state.get("m_lr_lambda")
        if lr_lambda_state is not None:
            self._lr_lambda.__dict__.update(lr_lambda_state)
