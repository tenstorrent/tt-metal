# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
import math
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


#: How far, in units of the stride, each strided float32 sample moves along its own
#: stride-wide cell from the one before: 1/phi, whose multiples mod 1 are as evenly
#: spread as any sequence's. See `_enumerate_fp32_in_range`.
_CELL_PHASE_STEP = (math.sqrt(5) - 1) / 2


def _enumerate_fp32_in_range(
    low: float, high: float, max_elements: int, *, offset: int = 0, stride: int = 1
) -> torch.Tensor:
    """Give back the float32 numbers in [low, high], smallest to largest — up to
    max_elements of them, skipping the first `offset`, and one from each run of
    `stride` consecutive ones.

    There are 4 billion float32 numbers, so we can't just list them all. The
    trick: adding 1 to a float's bit pattern (read as an integer) gives the very
    next float. So we take low's bits, count up `stride` integers at a time (one, by
    default) to high's bits, and turn each back into a float. (A small fix makes
    negatives and crossing zero work.) `offset` starts the count further along, so
    sweeping a big range in chunks never repeats numbers already covered; `stride`
    spreads one tensor's worth over a range far wider than it, down to the whole
    format. Keyword-only, both: the two are ints side by side, and a swap runs.

    With `stride` > 1, sample `i` is taken at a different place inside its own run of
    `stride` values, `(i * stride / phi) mod stride` along it, rather than at the
    run's start. A fixed position is a fixed set of low bits: a stride of 2**16 from
    -inf lands only on bfloat16 values, already swept as Float16_b, and one of
    2**16 + 1 ties the low half to the high, so a whole binade (128 consecutive
    samples) sits at one point of each bfloat16 cell. The golden-ratio step puts
    consecutive samples far apart in their cells, and on a power-of-two stride never
    twice at the same place within `stride` samples.
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
    # `stride` spreads the sample over the whole range instead of taking the first
    # max_elements consecutive values, which for float32 is a microscopic slice of one
    # binade. Every binade holds the same number of representable values, so striding
    # the total order gives each one an equal share of the sample.
    count = min(max_elements, (base_hi - lo_key) // stride + 1)
    index = torch.arange(count, dtype=torch.int64)
    keys = lo_key + index * stride
    if stride > 1:
        # Integer arithmetic on a step rounded to the nearest odd number of values: odd
        # is coprime with a power-of-two stride, so the phase only returns to 0 after
        # `stride` samples. The last cell may then reach past `high`.
        phase_step = int(stride * _CELL_PHASE_STEP) | 1
        keys = keys + (index * phase_step) % stride
        keys = keys[keys <= base_hi]
    bits = torch.where(keys < 0, INT_MIN - keys, keys).to(torch.int32)
    return bits.view(torch.float32)


def _enumerate_representable(
    stimuli_format: DataFormat,
    low: float,
    high: float,
    max_elements: int = 2**16,
    *,
    stride: int = 1,
    offset: int = 0,
) -> torch.Tensor:
    """Return the numbers a float format can represent in [low, high] — sorted,
    duplicates removed, capped at max_elements, skipping the first `offset`.

    The 16-bit formats are small, so we list all 2^16 possible values and keep
    the ones in range. float32 has far too many for that, so we walk only the
    range instead (see _enumerate_fp32_in_range).

    `offset` lets a big range be covered in chunks across several calls
    (offset = 0, max_elements, 2*max_elements, ...).

    `stride` takes one value from each run of `stride` consecutive ones instead of
    every value, so a range with more values than one tensor holds is sampled across
    its whole width rather than only at its start: the run's first value on a 16-bit
    format, a different place in each run on float32 (see _enumerate_fp32_in_range).
    """
    if stimuli_format in (DataFormat.Float16_b, DataFormat.Float16):
        dtype = (
            torch.bfloat16 if stimuli_format == DataFormat.Float16_b else torch.float16
        )
        all_bits = torch.arange(0, 2**16, dtype=torch.int16)
        all_vals = all_bits.view(dtype).to(torch.float32)
    elif stimuli_format == DataFormat.Float32:
        dtype = torch.float32
        # float32 applies the offset and the stride inside the walk (jumps straight to
        # them), so the slicing below is the 16-bit formats' alone.
        all_vals = _enumerate_fp32_in_range(
            low, high, max_elements, offset=offset, stride=stride
        )
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

    if stimuli_format != DataFormat.Float32:
        # The 16-bit formats enumerate their whole domain first, so they skip and stride
        # here -- the offset first, as the float32 walk does: `offset` counts in-range
        # values, not strided samples. (No in-cell phase: no sweep strides them.)
        vals = vals[offset::stride]
    vals = vals[:max_elements]

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
    """Exhaustive 1-ULP sweep — every representable value in [low, high], or every
    ``spec.stride``-th of them.

    Float16_b and Float16 are enumerated whole. Float32 is walked: a range as given, or
    its full domain with a stride wide enough to fit one tensor. Padded with zeros to
    fill the requested tensor length.
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
            stimuli_format,
            spec.low,
            spec.high,
            num_elements,
            stride=spec.stride,
            offset=spec.offset,
        )
        n = vals.numel()
        if n >= num_elements:
            return vals[:num_elements]
        padding = torch.zeros(num_elements - n, dtype=vals.dtype)
        return torch.cat([vals, padding])
