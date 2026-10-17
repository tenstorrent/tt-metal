# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Dual Block Cache (DBCache) step-skipping bookkeeping for DiT denoising loops.

This is a framework-agnostic port of the decision logic in
`cache-dit <https://github.com/vipshop/cache-dit>`_ (``DBCacheConfig`` / ``CachedContext``).
Tensor work (computing residuals, the L1 diff and applying cached residuals) is done by the
model / pipeline; this module only decides *whether* a step may reuse the cached residual.

Terminology (per transformer forward):

* ``Fn`` blocks: the first ``Fn_compute_blocks`` transformer blocks are always computed. The
  change they make to the hidden state (the *Fn residual*) is compared against the Fn residual of
  the previous computed step with a relative mean-L1 metric.
* ``Mn`` blocks: the middle blocks. On a cached step they are skipped and the *Bn residual*
  (their total contribution saved on the last computed step) is added instead.
* ``Bn`` blocks: the last ``Bn_compute_blocks`` blocks are always computed.

A *branch* is one of the two classifier-free-guidance passes (``0`` = conditional,
``1`` = unconditional). Branches keep separate cached-step accounting (cache-dit's
``enable_separate_cfg=True``) but share the step counter.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Literal

Decision = Literal["compute", "cache", "dynamic"]


@dataclass(frozen=True)
class DBCacheConfig:
    """DBCache knobs. Field names follow cache-dit's ``DBCacheConfig`` for easy cross-reference.

    Attributes:
        Fn_compute_blocks: Number of leading blocks always computed (>= 1).
        Bn_compute_blocks: Number of trailing blocks always computed (>= 0).
        residual_diff_threshold: Cache the step when the relative L1 diff of the Fn residual
            against the previous computed step is below this value. ``<= 0`` disables caching.
        max_warmup_steps: Steps ``[0, max_warmup_steps)`` (stepping by ``warmup_interval``) never
            cache.
        warmup_interval: Stride of the warmup step set, e.g. ``2`` means only steps ``0, 2, 4, ...``
            of the warmup window are forced to compute.
        max_cached_steps: Cap on cached steps per branch per denoising run. ``-1`` is unlimited.
        max_continuous_cached_steps: Cap on consecutive cached steps per branch. ``-1`` is
            unlimited.
        max_accumulated_residual_diff_threshold: Stop caching once the sum of recorded diffs
            exceeds this. ``None`` disables the check.
        cfg_diff_compute_separate: If ``False`` the unconditional branch reuses the conditional
            branch's diff value instead of computing its own.
        steps_computation_mask: Optional per-step mask (``1`` = must compute, ``0`` = may cache).
            Overrides warmup once set, see :meth:`DBCacheContext.warmup_steps`.
        steps_computation_policy: ``"dynamic"`` gates masked-``0`` steps with the residual diff,
            ``"static"`` caches them unconditionally.
        taylorseer_order: Number of Taylor derivatives used to forecast the cached residual
            (`TaylorSeer <https://arxiv.org/abs/2503.06923>`_, cache-dit's ``TaylorSeerCalibratorConfig``).
            ``0`` re-applies the last computed residual unchanged; ``1``/``2`` extrapolate it from
            the trajectory of previous computed steps.
    """

    Fn_compute_blocks: int = 1
    Bn_compute_blocks: int = 0
    residual_diff_threshold: float = 0.08
    max_warmup_steps: int = 4
    warmup_interval: int = 1
    max_cached_steps: int = -1
    max_continuous_cached_steps: int = 2
    max_accumulated_residual_diff_threshold: float | None = None
    cfg_diff_compute_separate: bool = True
    steps_computation_mask: tuple[int, ...] | None = None
    steps_computation_policy: Literal["dynamic", "static"] = "dynamic"
    taylorseer_order: int = 0

    def __post_init__(self) -> None:
        if self.taylorseer_order < 0:
            msg = "taylorseer_order must be >= 0"
            raise ValueError(msg)
        if self.Fn_compute_blocks < 1:
            msg = "Fn_compute_blocks must be >= 1"
            raise ValueError(msg)
        if self.Bn_compute_blocks < 0:
            msg = "Bn_compute_blocks must be >= 0"
            raise ValueError(msg)
        if self.warmup_interval < 1:
            msg = "warmup_interval must be >= 1"
            raise ValueError(msg)
        if self.steps_computation_policy not in ("dynamic", "static"):
            msg = f"steps_computation_policy must be 'dynamic' or 'static', got {self.steps_computation_policy!r}"
            raise ValueError(msg)
        if self.steps_computation_mask is not None:
            object.__setattr__(self, "steps_computation_mask", tuple(int(m) for m in self.steps_computation_mask))

    def replace(self, **changes: object) -> DBCacheConfig:
        return dataclasses.replace(self, **changes)

    @property
    def enabled(self) -> bool:
        """Whether any step can be cached at all."""
        if self.steps_computation_mask is not None and self.steps_computation_policy == "static":
            return True
        return self.residual_diff_threshold > 0.0

    def strify(self) -> str:
        s = (
            f"F{self.Fn_compute_blocks}B{self.Bn_compute_blocks}_"
            f"W{self.max_warmup_steps}I{self.warmup_interval}"
            f"M{max(0, self.max_cached_steps)}MC{max(0, self.max_continuous_cached_steps)}_"
            f"R{self.residual_diff_threshold}"
        )
        if self.steps_computation_mask is not None:
            s += f"_SCM{''.join(map(str, self.steps_computation_mask))}_{self.steps_computation_policy}"
        if self.taylorseer_order > 0:
            s += f"_T1O{self.taylorseer_order}"
        return s


def steps_mask(*, compute_bins: list[int], cache_bins: list[int]) -> tuple[int, ...]:
    """Build a LeMiCa/EasyCache style computation mask from alternating compute/cache run lengths.

    ``compute_bins=[6, 1, 1]``, ``cache_bins=[1, 2, 5]`` gives ``1111110 1 00 1 00000``.
    """
    if len(compute_bins) != len(cache_bins):
        msg = "compute_bins and cache_bins must have the same length"
        raise ValueError(msg)
    mask: list[int] = []
    for c, k in zip(compute_bins, cache_bins, strict=True):
        mask.extend([1] * c)
        mask.extend([0] * k)
    return tuple(mask)


@dataclass
class _BranchState:
    cached_steps: list[int] = dataclasses.field(default_factory=list)
    continuous_cached_steps: int = 0
    residual_diffs: dict[int, float] = dataclasses.field(default_factory=dict)
    accumulated_residual_diff: float = 0.0
    has_fn_buffer: bool = False
    has_bn_buffer: bool = False


class DBCacheContext:
    """Per-transformer cache decision state for one denoising run.

    Call :meth:`reset` at the start of every denoising run (cache-dit's ``refresh_context``),
    :meth:`mark_step_begin` once per denoising step, then per branch :meth:`gate` and, if it
    returns ``"dynamic"``, :meth:`decide` with the measured relative L1 diff.
    """

    def __init__(self, config: DBCacheConfig, *, name: str = "dbcache", num_branches: int = 2) -> None:
        self.config = config
        self.name = name
        self.num_branches = num_branches
        self.reset()

    # ------------------------------------------------------------------ lifecycle
    def reset(self) -> None:
        self.executed_steps = 0
        self._branches = [_BranchState() for _ in range(self.num_branches)]
        # Set on a branch when the current step was decided from the other branch's diff.
        self._step_decisions: dict[int, bool] = {}

    def mark_step_begin(self) -> None:
        self.executed_steps += 1
        self._step_decisions = {}

    @property
    def current_step(self) -> int:
        return self.executed_steps - 1

    def branch(self, branch: int) -> _BranchState:
        return self._branches[branch]

    # ------------------------------------------------------------------ config-derived
    @property
    def warmup_steps(self) -> list[int]:
        cfg = self.config
        max_warmup = cfg.max_warmup_steps
        if cfg.steps_computation_mask is not None:
            leading = 0
            for m in cfg.steps_computation_mask:
                if m != 1:
                    break
                leading += 1
            max_warmup = min(max_warmup, leading)
        return list(range(0, max_warmup, cfg.warmup_interval))

    def is_in_warmup(self) -> bool:
        return self.current_step in self.warmup_steps

    def is_in_full_compute_step(self) -> bool:
        mask = self.config.steps_computation_mask
        if mask is None:
            return False
        step = self.current_step
        return step < len(mask) and mask[step] == 1

    # ------------------------------------------------------------------ decisions
    def gate(self, branch: int) -> Decision:
        """Cheap, diff-free part of cache-dit's ``can_cache``.

        Returns ``"compute"`` or ``"cache"`` when the decision is already settled, or
        ``"dynamic"`` when the residual diff must be measured and passed to :meth:`decide`.
        """
        cfg = self.config
        st = self._branches[branch]

        if self.is_in_warmup():
            return "compute"

        if cfg.steps_computation_mask is not None:
            if self.is_in_full_compute_step():
                return "compute"
            if cfg.steps_computation_policy == "static":
                return "cache" if st.has_bn_buffer else "compute"

        if cfg.max_cached_steps >= 0 and len(st.cached_steps) >= cfg.max_cached_steps:
            return "compute"

        if cfg.max_continuous_cached_steps >= 0 and st.continuous_cached_steps >= cfg.max_continuous_cached_steps:
            # cache-dit resets the continuous counter when the cap forces a compute step.
            st.continuous_cached_steps = 0
            return "compute"

        thr = cfg.max_accumulated_residual_diff_threshold
        if thr is not None and thr > 0.0 and st.accumulated_residual_diff >= thr:
            return "compute"

        if cfg.residual_diff_threshold <= 0.0:
            return "compute"

        if not st.has_fn_buffer or not st.has_bn_buffer:
            return "compute"

        if branch != 0 and not cfg.cfg_diff_compute_separate and 0 in self._step_decisions:
            # Reuse the conditional branch's decision instead of measuring a second diff.
            return "cache" if self._step_decisions[0] else "compute"

        return "dynamic"

    def decide(self, branch: int, diff: float) -> bool:
        """Resolve a ``"dynamic"`` gate with the measured relative L1 diff. Returns ``True`` to cache."""
        st = self._branches[branch]
        if self.current_step not in st.residual_diffs:
            st.residual_diffs[self.current_step] = diff
            if diff > 0.0:
                st.accumulated_residual_diff += diff
        use_cache = diff < self.config.residual_diff_threshold
        self._step_decisions[branch] = use_cache
        return use_cache

    def add_cached_step(self, branch: int) -> None:
        st = self._branches[branch]
        step = self.current_step
        if st.cached_steps and step - st.cached_steps[-1] == 1:
            st.continuous_cached_steps += 2 if st.continuous_cached_steps == 0 else 1
        else:
            st.continuous_cached_steps += 1
        st.cached_steps.append(step)
        self._step_decisions[branch] = True

    def add_computed_step(self, branch: int) -> None:
        """Record that ``branch`` recomputed the middle blocks this step (its buffers are now valid)."""
        st = self._branches[branch]
        st.has_fn_buffer = True
        st.has_bn_buffer = True
        self._step_decisions.setdefault(branch, False)

    # ------------------------------------------------------------------ reporting
    def summary(self) -> dict[str, object]:
        return {
            "name": self.name,
            "config": self.config.strify(),
            "executed_steps": self.executed_steps,
            "cached_steps": [list(b.cached_steps) for b in self._branches],
            "residual_diffs": [dict(b.residual_diffs) for b in self._branches],
        }

    def num_cached_steps(self, branch: int | None = None) -> int:
        if branch is None:
            return sum(len(b.cached_steps) for b in self._branches)
        return len(self._branches[branch].cached_steps)
