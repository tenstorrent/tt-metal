# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Random face-mode distribution strategies: uniform, gaussian, log_uniform."""

import math
from typing import List, Optional

import torch

from ...format_config import DataFormat
from ..spec import StimuliSpec
from ..utils import (
    _get_dtype_for_format,
    _get_integer_bounds,
    _sample_gaussian_intervals,
    _sample_integer_gaussian_intervals,
    _sample_integer_intervals,
    _sample_log_uniform_intervals,
    _sample_uniform_intervals,
    integer_face_bounds_or_constant,
)

# ─────────────────────────────────────────────────────────────────────────────
# Uniform
# ─────────────────────────────────────────────────────────────────────────────


def _clamp_to_bounds(
    values: torch.Tensor, low: float, high: float, dtype: torch.dtype
) -> torch.Tensor:
    """Keep *values* inside [low, high] as the format represents them.

    The fp32 draw is rounded to *dtype* last, and rounding can carry a value just past a
    bound that is not itself representable (0.99 in bfloat16 is 0.98828125, so a draw of
    0.9899 would land on 0.9921875). The bounds are rounded the same way and clamped to,
    so no stimulus leaves the interval the spec asked for.
    """
    if dtype.is_floating_point:
        lo = torch.tensor(low, dtype=dtype)
        hi = torch.tensor(high, dtype=dtype)
        return values.clamp(min=min(lo, hi), max=max(lo, hi))
    return values


class UniformStrategy:
    """Uniform random sampling. Per-face only."""

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
            return self._generate_integer_face(spec, stimuli_format, size, generator)
        return self._generate_float_face(spec, stimuli_format, size, generator)

    def generate_full_tensor(
        self,
        spec: StimuliSpec,
        stimuli_format: DataFormat,
        num_elements: int,
        input_dimensions: Optional[List[int]],
        generator: Optional[torch.Generator],
    ) -> torch.Tensor:
        raise NotImplementedError(
            "distribution='uniform' is per-face only; use generate_face, "
            "not generate_full_tensor"
        )

    def _generate_float_face(self, spec, stimuli_format, size, generator):
        dtype = _get_dtype_for_format(stimuli_format)
        if spec.intervals:
            return _sample_uniform_intervals(spec.intervals, size, dtype, generator)
        # Draw in fp32 and cast last, as the interval sampler does. torch.rand in
        # bfloat16 carries only 8 random bits (256 distinct values in [0, 1), 2048 for
        # float16), so a 16-bit uniform drawn in its own dtype put every stimulus on a
        # 1/256 grid whatever the tensor size: Exp over [-100, 80] saw 234 distinct inputs
        # at a spacing of 0.5-0.78. Drawn in fp32 and rounded, the same draw reaches the
        # format's full resolution across the range. Clamped to the rounded bounds so the
        # cast cannot step outside the requested [low, high].
        raw = torch.rand(size, dtype=torch.float32, generator=generator)
        values = (raw * (spec.high - spec.low) + spec.low).to(dtype)
        return _clamp_to_bounds(values, spec.low, spec.high, dtype)

    def _generate_integer_face(self, spec, stimuli_format, size, generator):
        dtype = _get_dtype_for_format(stimuli_format)
        bounds = integer_face_bounds_or_constant(spec, stimuli_format, size, dtype)
        if isinstance(bounds, torch.Tensor):
            return bounds
        low, high = bounds
        if spec.intervals:
            int_min, int_max = _get_integer_bounds(stimuli_format)
            return _sample_integer_intervals(
                spec.intervals, int_min, int_max, size, dtype, generator
            )
        return torch.randint(
            low=low, high=high + 1, size=(size,), dtype=dtype, generator=generator
        )


# ─────────────────────────────────────────────────────────────────────────────
# Gaussian
# ─────────────────────────────────────────────────────────────────────────────


class GaussianStrategy:
    """Normal-distribution sampling with optional interval truncation."""

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
            return self._generate_integer_face(spec, stimuli_format, size, generator)
        return self._generate_float_face(spec, stimuli_format, size, generator)

    def generate_full_tensor(
        self,
        spec: StimuliSpec,
        stimuli_format: DataFormat,
        num_elements: int,
        input_dimensions: Optional[List[int]],
        generator: Optional[torch.Generator],
    ) -> torch.Tensor:
        raise NotImplementedError(
            "distribution='gaussian' is per-face only; use generate_face, "
            "not generate_full_tensor"
        )

    def _generate_float_face(self, spec, stimuli_format, size, generator):
        dtype = _get_dtype_for_format(stimuli_format)
        if spec.intervals:
            return _sample_gaussian_intervals(spec, size, dtype, generator)
        raw = torch.randn(size, dtype=dtype, generator=generator)
        return raw * spec.std + spec.mean

    def _generate_integer_face(self, spec, stimuli_format, size, generator):
        dtype = _get_dtype_for_format(stimuli_format)
        bounds = integer_face_bounds_or_constant(spec, stimuli_format, size, dtype)
        if isinstance(bounds, torch.Tensor):
            return bounds
        low, high = bounds
        if spec.intervals:
            int_min, int_max = _get_integer_bounds(stimuli_format)
            return _sample_integer_gaussian_intervals(
                spec, int_min, int_max, size, dtype, generator
            )
        raw = (
            torch.randn(size, dtype=torch.float32, generator=generator) * spec.std
            + spec.mean
        )
        return raw.round().clamp(low, high).to(dtype=dtype)


# ─────────────────────────────────────────────────────────────────────────────
# Log-uniform
# ─────────────────────────────────────────────────────────────────────────────


class LogUniformStrategy:
    """Log-uniform random sampling. Requires strictly positive bounds."""

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
            return self._generate_integer_face(spec, stimuli_format, size, generator)
        return self._generate_float_face(spec, stimuli_format, size, generator)

    def generate_full_tensor(
        self,
        spec: StimuliSpec,
        stimuli_format: DataFormat,
        num_elements: int,
        input_dimensions: Optional[List[int]],
        generator: Optional[torch.Generator],
    ) -> torch.Tensor:
        raise NotImplementedError(
            "distribution='log_uniform' is per-face only; use generate_face, "
            "not generate_full_tensor"
        )

    def _generate_float_face(self, spec, stimuli_format, size, generator):
        dtype = _get_dtype_for_format(stimuli_format)
        if spec.intervals:
            return _sample_log_uniform_intervals(spec.intervals, size, dtype, generator)
        if spec.low <= 0 or spec.high <= 0:
            raise ValueError(
                f"log_uniform requires strictly positive low and high, "
                f"got low={spec.low}, high={spec.high}"
            )
        log_low = math.log(spec.low)
        log_high = math.log(spec.high)
        # fp32 draw, cast last: see UniformStrategy._generate_float_face.
        raw = torch.rand(size, dtype=torch.float32, generator=generator)
        values = torch.exp(raw * (log_high - log_low) + log_low).to(dtype=dtype)
        return _clamp_to_bounds(values, spec.low, spec.high, dtype)

    def _generate_integer_face(self, spec, stimuli_format, size, generator):
        if spec.low <= 0 or spec.high <= 0:
            raise ValueError(
                f"log_uniform requires strictly positive low and high bounds; "
                f"got low={spec.low}, high={spec.high}.  "
                f"For integer formats use a positive range such as "
                f"StimuliSpec.log_uniform(low=1, high=1000)."
            )
        dtype = _get_dtype_for_format(stimuli_format)
        bounds = integer_face_bounds_or_constant(spec, stimuli_format, size, dtype)
        if isinstance(bounds, torch.Tensor):
            return bounds
        low, high = bounds
        log_low = math.log(spec.low)
        log_high = math.log(spec.high)
        raw_u = torch.rand(size, dtype=torch.float32, generator=generator)
        raw = torch.exp(raw_u * (log_high - log_low) + log_low)
        return raw.round().clamp(low, high).to(dtype=dtype)
