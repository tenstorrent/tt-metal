# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
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


def _enumerate_fp32_in_range(
    low: float, high: float, max_elements: int, offset: int = 0
) -> torch.Tensor:
    """Give back the float32 numbers in [low, high], smallest to largest — up to
    max_elements of them, skipping the first `offset`.

    There are 4 billion float32 numbers, so we can't just list them all. The
    trick: adding 1 to a float's bit pattern (read as an integer) gives the very
    next float. So we take low's bits, count up one integer at a time to high's
    bits, and turn each back into a float. (A small fix makes negatives and
    crossing zero work.) `offset` starts the count further along, so sweeping a
    big range in chunks never repeats numbers already covered.
    """
    INT_MIN = -(2**31)

    def _bits(x: float) -> int:
        return int(torch.tensor([x], dtype=torch.float32).view(torch.int32).item())

    def _to_key(bits: int) -> int:
        # Monotonic total order over float32 (an involution): a larger key means
        # a larger float value, including across the sign boundary.
        return bits if bits >= 0 else INT_MIN - bits

    base_lo = _to_key(_bits(low))
    base_hi = _to_key(_bits(high))
    if base_lo > base_hi:
        base_lo, base_hi = base_hi, base_lo

    lo_key = base_lo + offset
    if lo_key > base_hi:
        return torch.empty(0, dtype=torch.float32)  # offset past the range end
    hi_key = min(base_hi, lo_key + max_elements - 1)

    keys = torch.arange(lo_key, hi_key + 1, dtype=torch.int64)
    bits = torch.where(keys < 0, INT_MIN - keys, keys).to(torch.int32)
    return bits.view(torch.float32)


def _enumerate_representable(
    stimuli_format: DataFormat,
    low: float,
    high: float,
    max_elements: int = 2**16,
    offset: int = 0,
) -> torch.Tensor:
    """Return the numbers a float format can represent in [low, high] — sorted,
    duplicates removed, capped at max_elements, skipping the first `offset`.

    The 16-bit formats are small, so we list all 2^16 possible values and keep
    the ones in range. float32 has far too many for that, so we walk only the
    range instead (see _enumerate_fp32_in_range).

    `offset` lets a big range be covered in chunks across several calls
    (offset = 0, max_elements, 2*max_elements, ...).
    """
    if stimuli_format in (DataFormat.Float16_b, DataFormat.Float16):
        dtype = (
            torch.bfloat16 if stimuli_format == DataFormat.Float16_b else torch.float16
        )
        all_bits = torch.arange(0, 2**16, dtype=torch.int16)
        all_vals = all_bits.view(dtype).to(torch.float32)
        # 16-bit enumerates the whole domain up front, so the offset is applied
        # when slicing the sorted in-range values below.
        slice_start = offset
    elif stimuli_format == DataFormat.Float32:
        dtype = torch.float32
        # float32 applies the offset inside the walk (jumps straight to it), so
        # the slice below starts at 0.
        all_vals = _enumerate_fp32_in_range(low, high, max_elements, offset)
        slice_start = 0
    else:
        raise ValueError(
            f"ULP_SWEEP supports Float16_b, Float16, and Float32 formats, "
            f"got {stimuli_format.name!r}"
        )

    mask = torch.isfinite(all_vals) & (all_vals >= low) & (all_vals <= high)
    vals = all_vals[mask]

    vals, _ = torch.sort(vals)

    if vals.numel() > 1:
        unique_mask = torch.cat([torch.tensor([True]), vals[1:] != vals[:-1]])
        vals = vals[unique_mask]

    vals = vals[slice_start : slice_start + max_elements]

    return vals.to(dtype)


def ulp_sweep_value_count(stimuli_format: DataFormat, low: float, high: float) -> int:
    """How many numbers the format can represent in [low, high].

    For float32 we get this by subtracting the two endpoints' integer bit
    patterns — no values are actually built — so a batched sweep can size itself
    without enumerating millions of them. The 16-bit formats are small, so we
    just count them by listing.
    """
    if stimuli_format == DataFormat.Float32:
        INT_MIN = -(2**31)

        def _key(x: float) -> int:
            b = int(torch.tensor([x], dtype=torch.float32).view(torch.int32).item())
            return b if b >= 0 else INT_MIN - b

        lo, hi = _key(low), _key(high)
        if lo > hi:
            lo, hi = hi, lo
        return hi - lo + 1
    return int(_enumerate_representable(stimuli_format, low, high).numel())


class UlpSweepStrategy:
    """Exhaustive 1-ULP sweep — every representable value in [low, high].

    Float16_b, Float16, and Float32 are supported (float32 only over a range,
    not its full domain). Padded with zeros to fill the requested tensor length.
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
        # Grab exactly num_elements values, starting spec.offset into the range
        # (this is how a big range gets swept in batches).
        vals = _enumerate_representable(
            stimuli_format, spec.low, spec.high, num_elements, spec.offset
        )
        n = vals.numel()
        if n >= num_elements:
            return vals[:num_elements]
        padding = torch.zeros(num_elements - n, dtype=vals.dtype)
        return torch.cat([vals, padding])
