# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
import functools
from typing import List, Optional

import torch

from ...format_config import DataFormat
from ...tile_constants import FACE_C_DIM
from ..spec import StimuliSpec
from ..utils import _get_dtype_for_format, _get_integer_bounds

# ─────────────────────────────────────────────────────────────────────────────
# Face-identity (per-face block)
# ─────────────────────────────────────────────────────────────────────────────


class FaceIdentityStrategy:
    """Per-face identity block: *spec.value* on the face diagonal, zero elsewhere."""

    short_circuit = False

    def generate_face(
        self,
        spec: StimuliSpec,
        stimuli_format: DataFormat,
        face_r_dim: int,
        size: int,
        generator: Optional[torch.Generator],
    ) -> torch.Tensor:
        dtype = _get_dtype_for_format(stimuli_format)
        diag_val = spec.value
        if stimuli_format.is_integer():
            int_min, int_max = _get_integer_bounds(stimuli_format)
            diag_val = max(int_min, min(int(round(diag_val)), int_max))
        face = torch.zeros(face_r_dim, FACE_C_DIM, dtype=dtype)
        face.diagonal()[:] = diag_val
        return face.reshape(-1)

    def generate_full_tensor(
        self,
        spec: StimuliSpec,
        stimuli_format: DataFormat,
        num_elements: int,
        input_dimensions: Optional[List[int]],
        generator: Optional[torch.Generator],
    ) -> torch.Tensor:
        raise NotImplementedError(
            "distribution='face_identity' is per-face only; use generate_face, "
            "not generate_full_tensor"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Custom (explicit per-face values + zero-fill remainder)
# ─────────────────────────────────────────────────────────────────────────────


class CustomStrategy:
    """Explicit values at the head of each face; remainder zero-filled."""

    short_circuit = False

    def generate_face(
        self,
        spec: StimuliSpec,
        stimuli_format: DataFormat,
        face_r_dim: int,
        size: int,
        generator: Optional[torch.Generator],
    ) -> torch.Tensor:
        if spec.values is None or len(spec.values) == 0:
            raise ValueError("distribution='custom' requires a non-empty 'values' list")
        if len(spec.values) > size:
            raise ValueError(
                f"custom values list has {len(spec.values)} elements "
                f"but face has only {size} "
                f"({face_r_dim} rows × {FACE_C_DIM} cols)"
            )
        dtype = _get_dtype_for_format(stimuli_format)
        if stimuli_format.is_integer():
            int_min, int_max = _get_integer_bounds(stimuli_format)
            vals = [max(int_min, min(int(round(v)), int_max)) for v in spec.values]
        else:
            vals = list(spec.values)
        tensor = torch.zeros(size, dtype=dtype)
        tensor[: len(vals)] = torch.tensor(vals, dtype=dtype)
        return tensor

    def generate_full_tensor(
        self,
        spec: StimuliSpec,
        stimuli_format: DataFormat,
        num_elements: int,
        input_dimensions: Optional[List[int]],
        generator: Optional[torch.Generator],
    ) -> torch.Tensor:
        raise NotImplementedError(
            "distribution='custom' is per-face only; use generate_face, "
            "not generate_full_tensor"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Identity (tensor-level identity matrix)
# ─────────────────────────────────────────────────────────────────────────────


class IdentityStrategy:
    """Tensor-level identity matrix: *spec.value* on the diagonal, zero elsewhere."""

    short_circuit = True

    def generate_face(
        self,
        spec: StimuliSpec,
        stimuli_format: DataFormat,
        face_r_dim: int,
        size: int,
        generator: Optional[torch.Generator],
    ) -> torch.Tensor:
        raise ValueError(
            "distribution='identity' is a tensor-level operation and cannot "
            "be used in a per-face context (e.g. inside face_specs). "
            "Use distribution='face_identity' for per-face identity blocks."
        )

    def generate_full_tensor(
        self,
        spec: StimuliSpec,
        stimuli_format: DataFormat,
        num_elements: int,
        input_dimensions: Optional[List[int]],
        generator: Optional[torch.Generator],
    ) -> torch.Tensor:
        if input_dimensions is None or len(input_dimensions) != 2:
            raise ValueError(
                "distribution='identity' requires input_dimensions=[rows, cols]"
            )
        rows, cols = input_dimensions
        dtype = _get_dtype_for_format(stimuli_format)
        diag_val = spec.value
        if stimuli_format.is_integer():
            int_min, int_max = _get_integer_bounds(stimuli_format)
            diag_val = max(int_min, min(int(round(diag_val)), int_max))
        tensor = torch.zeros(rows, cols, dtype=dtype)
        tensor.diagonal()[:] = diag_val
        return tensor.reshape(-1)


# ─────────────────────────────────────────────────────────────────────────────
# ULP sweep (exhaustive 1-ULP enumeration of representable values)
# ─────────────────────────────────────────────────────────────────────────────


@functools.lru_cache(maxsize=32)
def _enumerate_representable(
    stimuli_format: DataFormat,
    low: float,
    high: float,
    max_elements: int = 2**16,
) -> torch.Tensor:
    """Return all finite representable values in [low, high] for a 16-bit float format.

    Iterates every 2^16 bit pattern, reinterprets as the target dtype, keeps
    finite values inside [low, high], then sorts, deduplicates (-0.0 == +0.0),
    and clips to max_elements.

    Cached: repeated calls with the same inputs return the same saved tensor.
    Do not modify the returned tensor in place.
    """
    if stimuli_format == DataFormat.Float16_b:
        dtype = torch.bfloat16
    elif stimuli_format == DataFormat.Float16:
        dtype = torch.float16
    else:
        raise ValueError(
            f"ULP_SWEEP only supports Float16_b and Float16 formats, "
            f"got {stimuli_format.name!r}"
        )

    all_bits = torch.arange(0, 2**16, dtype=torch.int16)
    all_vals = all_bits.view(dtype).to(torch.float32)

    mask = torch.isfinite(all_vals) & (all_vals >= low) & (all_vals <= high)
    vals = all_vals[mask]

    vals, _ = torch.sort(vals)

    if vals.numel() > 1:
        unique_mask = torch.cat([torch.tensor([True]), vals[1:] != vals[:-1]])
        vals = vals[unique_mask]

    if vals.numel() > max_elements:
        vals = vals[:max_elements]

    return vals.to(dtype)


class UlpSweepStrategy:
    """Exhaustive 1-ULP sweep — every representable value in [low, high].

    Only Float16_b and Float16 are supported. Padded with zeros to fill the
    requested tensor length.
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
        raise ValueError(
            "distribution='ulp_sweep' is a tensor-level operation and cannot "
            "be used in a per-face context."
        )

    def generate_full_tensor(
        self,
        spec: StimuliSpec,
        stimuli_format: DataFormat,
        num_elements: int,
        input_dimensions: Optional[List[int]],
        generator: Optional[torch.Generator],
    ) -> torch.Tensor:
        # Clone the cached tensor — the slice in the `n >= num_elements` branch
        # below returns a view, which could leak the cache to a caller that
        # then mutates it. Cloning here makes the downstream slice safe.
        vals = _enumerate_representable(stimuli_format, spec.low, spec.high).clone()
        n = vals.numel()
        if n >= num_elements:
            return vals[:num_elements]
        padding = torch.zeros(num_elements - n, dtype=vals.dtype)
        return torch.cat([vals, padding])
