# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
import math
from typing import List, Optional

import torch

from ...format_config import DataFormat
from ..spec import StimuliSpec
from ..utils import (
    _get_dtype_for_format,
    _get_integer_bounds,
    _in_intervals,
    _split_size_across_intervals,
    integer_face_bounds_or_constant,
)

# Tiny offset so we never hit exact 0 or 1 in the Gaussian CDF, which
# would otherwise produce ±infinity.
_GAUSSIAN_LINSPACE_EPS = 1e-6


# ─────────────────────────────────────────────────────────────────────────────
# Shared base for the three linspace-style strategies
# ─────────────────────────────────────────────────────────────────────────────


class _LinspaceStrategy:
    """Generate values with a deterministic sweep across the full tensor.

    Used for linspace-style distributions that bypass the face loop and
    produce one continuous pattern over the whole tensor.
    """

    short_circuit = True

    def generate_face(
        self,
        spec: StimuliSpec,
        stimuli_format: DataFormat,
        face_r_dim: int,
        size: int,
        generator: Optional[torch.Generator],
    ) -> torch.Tensor:
        if stimuli_format.is_integer():
            return self._integer_face(spec, stimuli_format, size)
        dtype = _get_dtype_for_format(stimuli_format)
        return self._float(spec, size, dtype)

    def generate_full_tensor(
        self,
        spec: StimuliSpec,
        stimuli_format: DataFormat,
        num_elements: int,
        input_dimensions: Optional[List[int]],
        generator: Optional[torch.Generator],
    ) -> torch.Tensor:
        dtype = _get_dtype_for_format(stimuli_format)
        if stimuli_format.is_integer():
            int_min, int_max = _get_integer_bounds(stimuli_format)
            raw = self._float(spec, num_elements, torch.float32)
            return raw.round().clamp(int_min, int_max).to(dtype=dtype)
        return self._float(spec, num_elements, dtype)

    def _integer_face(
        self,
        spec: StimuliSpec,
        stimuli_format: DataFormat,
        size: int,
    ) -> torch.Tensor:
        """Face-mode integer path — clamps to SPEC bounds (not format bounds)."""
        dtype = _get_dtype_for_format(stimuli_format)
        bounds = integer_face_bounds_or_constant(spec, stimuli_format, size, dtype)
        if isinstance(bounds, torch.Tensor):
            return bounds
        low, high = bounds
        raw = self._float(spec, size, torch.float32)
        return raw.round().clamp(low, high).to(dtype=dtype)

    @staticmethod
    def _float(spec: StimuliSpec, size: int, dtype: torch.dtype) -> torch.Tensor:
        raise NotImplementedError("subclasses must override _float")


# ─────────────────────────────────────────────────────────────────────────────
# Ramp — continuous linear sweep
# ─────────────────────────────────────────────────────────────────────────────


class RampStrategy(_LinspaceStrategy):
    @staticmethod
    def _float(spec: StimuliSpec, size: int, dtype: torch.dtype) -> torch.Tensor:
        if spec.intervals:
            counts = _split_size_across_intervals(spec.intervals, size)
            segments = []
            for (lo, hi), n in zip(spec.intervals, counts):
                if n > 0:
                    segments.append(torch.linspace(lo, hi, n, dtype=torch.float32))
            return torch.cat(segments).to(dtype=dtype)
        return torch.linspace(spec.low, spec.high, size, dtype=torch.float32).to(
            dtype=dtype
        )


# ─────────────────────────────────────────────────────────────────────────────
# Gaussian linspace — inverse-CDF sweep through a Normal distribution
# ─────────────────────────────────────────────────────────────────────────────


class GaussianLinspaceStrategy(_LinspaceStrategy):
    @staticmethod
    def _float(spec: StimuliSpec, size: int, dtype: torch.dtype) -> torch.Tensor:
        if spec.intervals:
            raise ValueError("intervals not supported for gaussian_linspace yet")
        p = torch.linspace(_GAUSSIAN_LINSPACE_EPS, 1.0 - _GAUSSIAN_LINSPACE_EPS, size)
        values = spec.mean + spec.std * math.sqrt(2.0) * torch.erfinv(2.0 * p - 1.0)
        return values.to(dtype=dtype)


# ─────────────────────────────────────────────────────────────────────────────
# Log-uniform linspace — deterministic log-spaced sweep
# ─────────────────────────────────────────────────────────────────────────────


class LogUniformLinspaceStrategy(_LinspaceStrategy):
    @staticmethod
    def _float(spec: StimuliSpec, size: int, dtype: torch.dtype) -> torch.Tensor:
        if spec.intervals:
            for lo, hi in spec.intervals:
                if lo <= 0 or hi <= 0:
                    raise ValueError(
                        f"log_uniform_linspace intervals require strictly positive "
                        f"bounds, got ({lo}, {hi})"
                    )
            counts = _split_size_across_intervals(spec.intervals, size)
            segments = []
            for (lo, hi), n in zip(spec.intervals, counts):
                if n > 0:
                    log_lo = math.log(lo)
                    log_hi = math.log(hi)
                    segments.append(
                        torch.exp(
                            torch.linspace(log_lo, log_hi, n, dtype=torch.float32)
                        )
                    )
            return torch.cat(segments).to(dtype=dtype)
        if spec.low <= 0 or spec.high <= 0:
            raise ValueError(
                f"log_uniform_linspace requires strictly positive low and high, "
                f"got low={spec.low}, high={spec.high}"
            )
        log_low = math.log(spec.low)
        log_high = math.log(spec.high)
        return torch.exp(
            torch.linspace(log_low, log_high, size, dtype=torch.float32)
        ).to(dtype=dtype)


# ─────────────────────────────────────────────────────────────────────────────
# Sequential — arithmetic progression with three modes
# ─────────────────────────────────────────────────────────────────────────────


class SequentialStrategy:
    short_circuit = True

    def generate_face(
        self,
        spec: StimuliSpec,
        stimuli_format: DataFormat,
        face_r_dim: int,
        size: int,
        generator: Optional[torch.Generator],
    ) -> torch.Tensor:
        dtype = _get_dtype_for_format(stimuli_format)
        return self._core(spec, size, dtype, stimuli_format)

    def generate_full_tensor(
        self,
        spec: StimuliSpec,
        stimuli_format: DataFormat,
        num_elements: int,
        input_dimensions: Optional[List[int]],
        generator: Optional[torch.Generator],
    ) -> torch.Tensor:
        dtype = _get_dtype_for_format(stimuli_format)
        return self._core(spec, num_elements, dtype, stimuli_format)

    @staticmethod
    def _core(
        spec: StimuliSpec,
        size: int,
        dtype: torch.dtype,
        stimuli_format: DataFormat,
    ) -> torch.Tensor:
        """Three modes:

        1. Default (no low/high/step/intervals set): 1, 2, 3, …, size.
        2. Intervals set: base sequence 1, 1+step, 1+2*step, …; values outside
           the union are zeroed.
        3. Custom low/high/step (no intervals): start at *low*, increment by
           *step*, zero-fill positions beyond *high*.
        """
        is_int = stimuli_format is not None and stimuli_format.is_integer()

        # Mode 1: default
        if (
            spec.low == 0.0
            and spec.high == 1.0
            and spec.std == 1.0
            and spec.intervals is None
        ):
            if is_int:
                int_min, int_max = _get_integer_bounds(stimuli_format)
                result = torch.arange(1, size + 1, dtype=torch.int64)
                return result.clamp(min=int_min, max=int_max).to(dtype)
            return torch.arange(1, size + 1, dtype=dtype)

        # Mode 2: intervals set — ignore low/high, use step + mask
        if spec.intervals is not None:
            step = spec.std
            idx = torch.arange(size, dtype=torch.float32)
            vals = 1.0 + step * idx

            if is_int:
                int_min, int_max = _get_integer_bounds(stimuli_format)
                vals = vals.round().clamp(int_min, int_max)

            mask = _in_intervals(vals, spec.intervals)
            vals = vals * mask.float()
            return vals.to(dtype)

        # Mode 3: custom low/high/step, no intervals
        low = spec.low
        step = spec.std
        high = spec.high

        idx = torch.arange(size, dtype=torch.float32)
        vals = low + step * idx

        if high == 1.0 and low != 0.0:
            # Special case: unbounded ramp — skip the zero-fill cutoff entirely.
            pass
        elif step > 0:
            vals = torch.where(vals <= high, vals, torch.zeros_like(vals))
        elif step < 0:
            vals = torch.where(vals >= high, vals, torch.zeros_like(vals))
        # step == 0: every element equals low

        if is_int:
            int_min, int_max = _get_integer_bounds(stimuli_format)
            vals = vals.round().clamp(int_min, int_max)

        return vals.to(dtype)


# ─────────────────────────────────────────────────────────────────────────────
# Saw — per-face linspace (deterministic, restarts every face)
# ─────────────────────────────────────────────────────────────────────────────


class SawStrategy:
    """Generate a saw pattern within each face.

    Each face gets its own `torch.linspace(spec.low, spec.high, size)`, so the
    pattern restarts at every face.
    """

    short_circuit = False

    def generate_face(
        self,
        spec: StimuliSpec,
        stimuli_format: DataFormat,
        face_r_dim: int,
        size: int,
        generator: Optional[torch.Generator],
    ) -> torch.Tensor:
        if stimuli_format.is_integer():
            return self._generate_integer_face(spec, stimuli_format, size)
        return self._generate_float_face(spec, stimuli_format, size)

    def generate_full_tensor(
        self,
        spec: StimuliSpec,
        stimuli_format: DataFormat,
        num_elements: int,
        input_dimensions: Optional[List[int]],
        generator: Optional[torch.Generator],
    ) -> torch.Tensor:
        raise NotImplementedError(
            "distribution='saw' is per-face only; use generate_face, "
            "not generate_full_tensor"
        )

    @staticmethod
    def _generate_float_face(
        spec: StimuliSpec, stimuli_format: DataFormat, size: int
    ) -> torch.Tensor:
        dtype = _get_dtype_for_format(stimuli_format)
        if spec.intervals:
            counts = _split_size_across_intervals(spec.intervals, size)
            segments = []
            for (lo, hi), n in zip(spec.intervals, counts):
                if n > 0:
                    segments.append(torch.linspace(lo, hi, n, dtype=dtype))
            return torch.cat(segments)
        return torch.linspace(spec.low, spec.high, size, dtype=dtype)

    @staticmethod
    def _generate_integer_face(
        spec: StimuliSpec, stimuli_format: DataFormat, size: int
    ) -> torch.Tensor:
        # Note: SAW on integer formats does not support intervals (matches
        # legacy behavior — the original _generate_integer_face's SAW branch
        # ignored spec.intervals).
        dtype = _get_dtype_for_format(stimuli_format)
        bounds = integer_face_bounds_or_constant(spec, stimuli_format, size, dtype)
        if isinstance(bounds, torch.Tensor):
            return bounds
        low, high = bounds
        return (
            torch.linspace(spec.low, spec.high, size, dtype=torch.float32)
            .round()
            .clamp(low, high)
            .to(dtype=dtype)
        )
