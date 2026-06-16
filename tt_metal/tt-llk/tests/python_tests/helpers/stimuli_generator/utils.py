# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
import math
import warnings
from typing import Dict, List, Optional, Tuple

import torch

from ..format_config import MX_FORMAT_MAX_NORMAL, MX_FORMAT_MIN_MAGNITUDE, DataFormat
from ..llk_params import format_dict
from ..tile_constants import (
    DEFAULT_TILE_C_DIM,
    DEFAULT_TILE_R_DIM,
    FACE_C_DIM,
    MAX_FACE_R_DIM,
    MAX_NUM_FACES,
    validate_tile_dimensions,
)
from .spec import StimuliSpec

# ─────────────────────────────────────────────────────────────────────────────
# Format helpers
# ─────────────────────────────────────────────────────────────────────────────


def _get_dtype_for_format(stimuli_format: DataFormat) -> torch.dtype:
    """Return the torch dtype to use for *stimuli_format*."""
    if stimuli_format in (DataFormat.Bfp8_b, DataFormat.Bfp4_b, DataFormat.Bfp2_b):
        return torch.bfloat16
    if stimuli_format == DataFormat.Tf32:
        return torch.float32
    return format_dict[stimuli_format]


_INTEGER_FORMAT_DTYPE: Dict[DataFormat, torch.dtype] = {
    DataFormat.Int8: torch.int8,
    DataFormat.UInt8: torch.uint8,
    DataFormat.Int16: torch.int16,
    DataFormat.UInt16: torch.uint16,
    DataFormat.Int32: torch.int32,
    DataFormat.UInt32: torch.uint32,
}


def _get_integer_bounds(stimuli_format: DataFormat) -> tuple[int, int]:
    """Return the valid integer range (min, max) inclusive for *stimuli_format*.

    For signed formats we exclude the most negative value (min + 1) because our
    integer paths use sign-magnitude encoding, and the INT_MIN bit pattern cannot
    be represented correctly in that scheme.
    """
    if stimuli_format not in _INTEGER_FORMAT_DTYPE:
        raise ValueError(f"Unsupported integer format: {stimuli_format}")

    dtype = _INTEGER_FORMAT_DTYPE[stimuli_format]
    info = torch.iinfo(dtype)
    lo = info.min + 1 if info.min < 0 else info.min
    return lo, info.max


def integer_face_bounds_or_constant(
    spec: StimuliSpec,
    stimuli_format: DataFormat,
    size: int,
    dtype: torch.dtype,
):
    """Return integer bounds for this spec, or a constant tensor if the range
    collapses after clamping.

    Args:
        spec: Stimuli spec with the requested low/high range.
        stimuli_format: Target integer data format.
        size: Output size for a constant fallback tensor.
        dtype: Torch dtype for the fallback tensor.

    Returns:
        Either (low, high) inclusive integer bounds, or a constant tensor
        of length size when the valid range is empty or has one value.
    """
    int_min, int_max = _get_integer_bounds(stimuli_format)
    low = max(math.ceil(spec.low), int_min)
    high = min(math.floor(spec.high), int_max)

    if high < low:
        fallback = max(int_min, min(round((spec.low + spec.high) / 2), int_max))
        warnings.warn(
            f"No integer exists in [{spec.low}, {spec.high}] for "
            f"{stimuli_format.name} (representable range [{int_min}, {int_max}]). "
            f"Returning constant tensor filled with {fallback}.",
            stacklevel=3,
        )
        return torch.full((size,), fallback, dtype=dtype)

    if high == low:
        return torch.full((size,), low, dtype=dtype)

    return (low, high)


# ─────────────────────────────────────────────────────────────────────────────
# Interval helpers
# ─────────────────────────────────────────────────────────────────────────────


def _in_intervals(
    values: torch.Tensor, intervals: List[Tuple[float, float]]
) -> torch.Tensor:
    """Return a boolean mask: True where the value falls inside any [lo, hi]."""
    mask = torch.zeros(values.shape, dtype=torch.bool)
    for lo, hi in intervals:
        mask |= (values >= lo) & (values <= hi)
    return mask


def _split_size_across_intervals(
    intervals: List[Tuple[float, float]], size: int
) -> List[int]:
    """Split *size* elements across intervals proportional to interval length.

    Rounding residuals are distributed to the intervals with the largest
    fractional parts so the total is exactly *size*.
    """
    if not intervals:
        raise ValueError("intervals must be non-empty for piecewise linspace")
    lengths = [abs(hi - lo) for lo, hi in intervals]
    total_length = sum(lengths)
    if total_length <= 0:
        raise ValueError(
            "Total interval length must be positive for piecewise linspace"
        )
    raw = [size * l / total_length for l in lengths]
    counts = [int(r) for r in raw]
    remainder = size - sum(counts)
    fracs = sorted(range(len(counts)), key=lambda i: raw[i] - counts[i], reverse=True)
    for j in range(remainder):
        counts[fracs[j]] += 1
    return counts


def _sample_uniform_intervals(
    intervals: List[Tuple[float, float]],
    size: int,
    dtype: torch.dtype,
    generator: Optional[torch.Generator],
) -> torch.Tensor:
    """Sample uniformly from a union of [low, high] intervals.

    Each interval is selected with probability proportional to its length.
    """
    if not intervals:
        raise ValueError("intervals must be a non-empty list")
    lows = torch.tensor([lo for lo, _ in intervals], dtype=torch.float32)
    highs = torch.tensor([hi for _, hi in intervals], dtype=torch.float32)
    lengths = torch.clamp(highs - lows, min=0.0)
    total = lengths.sum()
    if total <= 0:
        raise ValueError("Total interval length must be positive")
    cdf = (lengths / total).cumsum(dim=0)
    u = torch.rand(size, dtype=torch.float32, generator=generator)
    idx = torch.searchsorted(cdf, u, right=False)
    u_inner = torch.rand(size, dtype=torch.float32, generator=generator)
    lo_sel = lows[idx]
    hi_sel = highs[idx]
    return (lo_sel + u_inner * (hi_sel - lo_sel)).to(dtype)


def _sample_log_uniform_intervals(
    intervals: List[Tuple[float, float]],
    size: int,
    dtype: torch.dtype,
    generator: Optional[torch.Generator],
) -> torch.Tensor:
    """Sample log-uniformly from a union of strictly positive intervals.

    Each interval is selected with probability proportional to its
    log-space length (log(hi) - log(lo)).
    """
    if not intervals:
        raise ValueError("intervals must be a non-empty list")
    for lo, hi in intervals:
        if lo <= 0 or hi <= 0:
            raise ValueError(
                f"log_uniform intervals require strictly positive bounds, "
                f"got ({lo}, {hi})"
            )
    log_lows = torch.tensor([math.log(lo) for lo, _ in intervals], dtype=torch.float32)
    log_highs = torch.tensor([math.log(hi) for _, hi in intervals], dtype=torch.float32)
    lengths = torch.clamp(log_highs - log_lows, min=0.0)
    total = lengths.sum()
    if total <= 0:
        raise ValueError("Total log-space interval length must be positive")
    cdf = (lengths / total).cumsum(dim=0)
    u = torch.rand(size, dtype=torch.float32, generator=generator)
    idx = torch.searchsorted(cdf, u, right=False)
    u_inner = torch.rand(size, dtype=torch.float32, generator=generator)
    ll = log_lows[idx]
    lh = log_highs[idx]
    return torch.exp(ll + u_inner * (lh - ll)).to(dtype)


def _sample_integer_intervals(
    intervals: List[Tuple[float, float]],
    int_min: int,
    int_max: int,
    size: int,
    dtype: torch.dtype,
    generator: Optional[torch.Generator],
) -> torch.Tensor:
    """Sample uniformly from a union of integer intervals (inclusive).

    Each interval is selected with probability proportional to the
    number of integer points it contains.
    """
    clamped = []
    counts = []
    for lo_f, hi_f in intervals:
        lo_i = max(math.ceil(lo_f), int_min)
        hi_i = min(math.floor(hi_f), int_max)
        if hi_i >= lo_i:
            clamped.append((lo_i, hi_i))
            counts.append(hi_i - lo_i + 1)
    if not clamped:
        fallback = max(int_min, min((int_min + int_max) // 2, int_max))
        warnings.warn(
            f"No valid integer exists in any interval after clamping to "
            f"[{int_min}, {int_max}]. Returning constant tensor filled with {fallback}.",
            stacklevel=3,
        )
        return torch.full((size,), fallback, dtype=dtype)
    lengths = torch.tensor(counts, dtype=torch.float32)
    cdf = (lengths / lengths.sum()).cumsum(dim=0)
    u = torch.rand(size, dtype=torch.float32, generator=generator)
    idx = torch.searchsorted(cdf, u, right=False)
    lows = torch.tensor([lo for lo, _ in clamped], dtype=torch.int64)
    ranges = torch.tensor(counts, dtype=torch.int64)
    lo_sel = lows[idx]
    range_sel = ranges[idx]
    u_inner = torch.rand(size, dtype=torch.float32, generator=generator)
    offsets = (u_inner * range_sel.to(torch.float32)).to(torch.int64)
    return (lo_sel + offsets).to(dtype)


def _sample_gaussian_intervals(
    spec: StimuliSpec,
    size: int,
    dtype: torch.dtype,
    generator: Optional[torch.Generator],
) -> torch.Tensor:
    """Sample from a truncated Gaussian over a union of intervals via rejection."""
    _MAX_ATTEMPTS = 100
    result = torch.empty(0, dtype=torch.float32)
    remaining = size
    for _ in range(_MAX_ATTEMPTS):
        batch = max(remaining * 4, 64)
        candidates = (
            torch.randn(batch, dtype=torch.float32, generator=generator) * spec.std
            + spec.mean
        )
        accepted = candidates[_in_intervals(candidates, spec.intervals)]
        result = torch.cat([result, accepted])
        if result.numel() >= size:
            return result[:size].to(dtype)
        remaining = size - result.numel()
    raise ValueError(
        f"Gaussian rejection sampling failed to fill {size} elements after "
        f"{_MAX_ATTEMPTS} attempts. The intervals {spec.intervals} may be "
        f"incompatible with mean={spec.mean}, std={spec.std}."
    )


def _sample_integer_gaussian_intervals(
    spec: StimuliSpec,
    int_min: int,
    int_max: int,
    size: int,
    dtype: torch.dtype,
    generator: Optional[torch.Generator],
) -> torch.Tensor:
    """Truncated integer Gaussian over a union of intervals via rejection.

    When intervals are set, spec.low/spec.high are ignored — the domain is
    spec.intervals intersected with the format's representable range.
    """
    clamped = []
    for lo_f, hi_f in spec.intervals:
        lo_i = max(math.ceil(lo_f), int_min)
        hi_i = min(math.floor(hi_f), int_max)
        if hi_i >= lo_i:
            clamped.append((lo_i, hi_i))
    if not clamped:
        raise ValueError(
            f"No valid integer interval remains after clamping "
            f"{spec.intervals} to [{int_min}, {int_max}]."
        )

    def _in_int_intervals(vals: torch.Tensor) -> torch.Tensor:
        mask = torch.zeros(vals.shape, dtype=torch.bool)
        for lo_i, hi_i in clamped:
            mask |= (vals >= lo_i) & (vals <= hi_i)
        return mask

    _MAX_ATTEMPTS = 100
    result = torch.empty(0, dtype=torch.int64)
    remaining = size
    for _ in range(_MAX_ATTEMPTS):
        batch = max(remaining * 4, 64)
        raw = (
            torch.randn(batch, dtype=torch.float32, generator=generator) * spec.std
            + spec.mean
        )
        rounded = raw.round().clamp(int_min, int_max).to(torch.int64)
        accepted = rounded[_in_int_intervals(rounded)]
        result = torch.cat([result, accepted])
        if result.numel() >= size:
            return result[:size].to(dtype)
        remaining = size - result.numel()
    raise ValueError(
        f"Integer Gaussian rejection sampling failed to fill {size} elements "
        f"after {_MAX_ATTEMPTS} attempts. The intervals {spec.intervals} "
        f"(clamped to {clamped}) may be incompatible with "
        f"mean={spec.mean}, std={spec.std}."
    )


# ─────────────────────────────────────────────────────────────────────────────
# Tile / face count
# ─────────────────────────────────────────────────────────────────────────────


def calculate_tile_and_face_counts(
    input_dimensions_A: list,
    input_dimensions_B: list,
    face_r_dim: int,
    num_faces: int,
) -> tuple[int, int, int]:
    """
    Calculate tile counts and faces to generate based on input dimensions and face configuration.
    Uses 32x32 tiles in the full-face case; falls back to a single-tile
    partial-face layout when face_r_dim < MAX_FACE_R_DIM.

    Args:
        input_dimensions_A: [height, width] in elements for input A
        input_dimensions_B: [height, width] in elements for input B
        face_r_dim: Number of rows in a face (typically 16 for full faces)
        num_faces: Number of faces to generate for partial face case

    Returns:
        tuple: (tile_cnt_A, tile_cnt_B, faces_to_generate)
    """
    if not (
        face_r_dim == MAX_FACE_R_DIM
        or (face_r_dim < MAX_FACE_R_DIM and face_r_dim == input_dimensions_A[0])
    ):
        raise ValueError(
            f"face_r_dim must be {MAX_FACE_R_DIM} (full face) or < {MAX_FACE_R_DIM} "
            f"and equal to input_dimensions_A[0] (partial single-tile case); "
            f"got face_r_dim={face_r_dim}, input_dimensions_A={input_dimensions_A}"
        )

    # Handle partial faces
    if face_r_dim < MAX_FACE_R_DIM:
        # Partial face case: generate exactly num_faces worth of data
        tile_cnt_A, tile_cnt_B = 1, 1
        faces_to_generate = num_faces  # Generate exactly the right number of faces
    else:
        # Full tile case - always use 32x32 tiles
        tile_cnt_A = (
            input_dimensions_A[0]
            // DEFAULT_TILE_R_DIM
            * input_dimensions_A[1]
            // DEFAULT_TILE_C_DIM
        )
        tile_cnt_B = (
            input_dimensions_B[0]
            // DEFAULT_TILE_R_DIM
            * input_dimensions_B[1]
            // DEFAULT_TILE_C_DIM
        )
        faces_to_generate = MAX_NUM_FACES

    return tile_cnt_A, tile_cnt_B, faces_to_generate


def calculate_tile_and_face_counts_w_tile_dimensions(
    input_dimensions_A: list,
    input_dimensions_B: list,
    face_r_dim: int,
    num_faces: int,
    tile_dimensions: list,
) -> tuple[int, int, int]:
    """
    Calculate tile counts and faces to generate for variable tile dimensions (dense mode).

    Args:
        input_dimensions_A: [height, width] in elements for input A
        input_dimensions_B: [height, width] in elements for input B
        face_r_dim: Number of rows in a face (1, 2, 4, 8, or 16)
        num_faces: Number of faces per tile (1, 2, or 4)
        tile_dimensions: [rows, cols] for tile size

    Returns:
        tuple: (tile_cnt_A, tile_cnt_B, faces_to_generate)
    """
    validate_tile_dimensions(tile_dimensions)
    tile_r_dim, tile_c_dim = tile_dimensions

    # Calculate tile counts based on actual tile dimensions
    tile_cnt_A = (input_dimensions_A[0] // tile_r_dim) * (
        input_dimensions_A[1] // tile_c_dim
    )
    tile_cnt_B = (input_dimensions_B[0] // tile_r_dim) * (
        input_dimensions_B[1] // tile_c_dim
    )
    # Always generate all faces to fill the tile densely
    faces_to_generate = num_faces

    return tile_cnt_A, tile_cnt_B, faces_to_generate


# ─────────────────────────────────────────────────────────────────────────────
# MX clamping
# ─────────────────────────────────────────────────────────────────────────────


def _clamp_mx_tensors(
    srcA_tensor: torch.Tensor,
    srcB_tensor: torch.Tensor,
    stimuli_format_A: DataFormat,
    stimuli_format_B: DataFormat,
    output_format: DataFormat = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Clamp tensors for MX format compatibility.

    Args:
        srcA_tensor: Source A tensor
        srcB_tensor: Source B tensor
        stimuli_format_A: Data format for source A
        stimuli_format_B: Data format for source B
        output_format: Optional output format for range constraints

    Returns:
        tuple: (clamped_srcA_tensor, clamped_srcB_tensor)
    """
    # Clamp inputs if both are different MX formats (use more restrictive MxFp8P)
    if stimuli_format_A.is_mx_format() and stimuli_format_B.is_mx_format():
        if stimuli_format_A != stimuli_format_B:
            srcA_tensor = torch.clamp(
                srcA_tensor,
                -MX_FORMAT_MAX_NORMAL[DataFormat.MxFp8P],
                MX_FORMAT_MAX_NORMAL[DataFormat.MxFp8P],
            )
            srcB_tensor = torch.clamp(
                srcB_tensor,
                -MX_FORMAT_MAX_NORMAL[DataFormat.MxFp8P],
                MX_FORMAT_MAX_NORMAL[DataFormat.MxFp8P],
            )

    # Clamp inputs based on output format to prevent excessive rounding errors
    if output_format == DataFormat.MxFp8P:
        srcA_tensor = torch.clamp(
            srcA_tensor,
            -MX_FORMAT_MAX_NORMAL[DataFormat.MxFp8P],
            MX_FORMAT_MAX_NORMAL[DataFormat.MxFp8P],
        )
        srcB_tensor = torch.clamp(
            srcB_tensor,
            -MX_FORMAT_MAX_NORMAL[DataFormat.MxFp8P],
            MX_FORMAT_MAX_NORMAL[DataFormat.MxFp8P],
        )
    elif output_format == DataFormat.MxFp8R:
        srcA_tensor = torch.clamp(
            srcA_tensor,
            -MX_FORMAT_MAX_NORMAL[DataFormat.MxFp8R],
            MX_FORMAT_MAX_NORMAL[DataFormat.MxFp8R],
        )
        srcB_tensor = torch.clamp(
            srcB_tensor,
            -MX_FORMAT_MAX_NORMAL[DataFormat.MxFp8R],
            MX_FORMAT_MAX_NORMAL[DataFormat.MxFp8R],
        )

    return srcA_tensor, srcB_tensor


# ─────────────────────────────────────────────────────────────────────────────
# Matmul / L1 layout
# ─────────────────────────────────────────────────────────────────────────────


def _mask_tile(
    tile: torch.Tensor,
    num_faces: int,
    is_matrix_B: bool,
    face_r_dim: int = MAX_FACE_R_DIM,
) -> torch.Tensor:
    masked = tile.clone()
    if num_faces == 1:
        # Keep only f0
        masked[:MAX_FACE_R_DIM, FACE_C_DIM:] = 0  # Zero f1
        masked[face_r_dim:, :] = 0  # Zero f2, f3 and part of f0
    elif num_faces == 2:
        if is_matrix_B:
            # matrix B (In1/SrcA): keep partial f0, f2
            if face_r_dim < MAX_FACE_R_DIM:
                masked[face_r_dim:MAX_FACE_R_DIM, :FACE_C_DIM] = 0  # Zero part of f0
                masked[MAX_FACE_R_DIM + face_r_dim :, :FACE_C_DIM] = (
                    0  # Zero part of f2
                )
            masked[:MAX_FACE_R_DIM, FACE_C_DIM:] = 0  # Zero f1
            masked[MAX_FACE_R_DIM:, FACE_C_DIM:] = 0  # Zero f3
        else:
            # matrix A (In0/SrcB): keep f0, f1
            masked[face_r_dim:, :] = 0  # Zero part of f0 and f1
    return masked


def generate_face_matmul_data(
    num_faces: int,
    stimuli_format: DataFormat,
    input_dimensions=[DEFAULT_TILE_R_DIM, DEFAULT_TILE_C_DIM],
    is_matrix_A=True,  # True for matrix A (SrcB), False for matrix B (SrcA)
    face_r_dim=MAX_FACE_R_DIM,
) -> torch.Tensor:

    # Validate num_faces
    if num_faces not in [1, 2, MAX_NUM_FACES]:
        raise ValueError(f"num_faces must be 1, 2, or {MAX_NUM_FACES}, got {num_faces}")

    # Validate input_dimensions
    rows, cols = input_dimensions
    if rows % DEFAULT_TILE_R_DIM != 0 or cols % DEFAULT_TILE_C_DIM != 0:
        raise ValueError(
            f"Input dimensions must be multiples of {DEFAULT_TILE_R_DIM}, "
            f"got {input_dimensions}"
        )

    rt, ct = rows // DEFAULT_TILE_R_DIM, cols // DEFAULT_TILE_C_DIM
    dtype = format_dict[stimuli_format]

    out = torch.rand(rt, ct, DEFAULT_TILE_R_DIM, DEFAULT_TILE_C_DIM, dtype=dtype)
    mask = _mask_tile(
        torch.ones(DEFAULT_TILE_R_DIM, DEFAULT_TILE_C_DIM, dtype=dtype),
        num_faces,
        not is_matrix_A,
        face_r_dim,
    )
    out *= mask
    out = out.permute(0, 2, 1, 3).reshape(rows, cols)

    return out


def convert_to_l1_view(
    tilized_tensor: torch.Tensor,
    input_dimensions: list,
    tile_dimensions: list = None,
) -> torch.Tensor:
    """
    Convert a tilized tensor to its L1 memory view by condensing data based on tile dimensions.

    This function extracts only the data that corresponds to the specified tile dimensions
    and places it at the beginning of each tile, with the remaining space zeroed out.
    The full tile size (1024 elements) is preserved.

    Tilized format: faces are stored sequentially [f0 (256), f1 (256), f2 (256), f3 (256)]
    Within each face, data is stored row-major (16 rows × 16 cols).

    Face layout in a 32×32 tile:
    - f0: rows 0-15, cols 0-15  (top-left)
    - f1: rows 0-15, cols 16-31 (top-right)
    - f2: rows 16-31, cols 0-15 (bottom-left)
    - f3: rows 16-31, cols 16-31 (bottom-right)

    Examples:
    - tile_dimensions=[32, 32]: full tile, no change [f0, f1, f2, f3]
    - tile_dimensions=[16, 32]: top half [f0, f1, 0, 0]
    - tile_dimensions=[32, 16]: left half [f0, f2, 0, 0]
    - tile_dimensions=[16, 16]: top-left only [f0, 0, 0, 0]
    - tile_dimensions=[8, 32]: first 8 rows [f0_rows0-7, f1_rows0-7, 0, ...]

    Args:
        tilized_tensor: Input tensor in tilized format (faces stored sequentially per tile)
        input_dimensions: [rows, cols] of the full input matrix
        tile_dimensions: [rows, cols] to keep per tile (default [32, 32])
                        rows must be one of: 1, 2, 4, 8, 16, 32
                        cols must be one of: 16, 32

    Returns:
        Tensor with condensed data at the beginning (face by face), zeros at the end
    """
    if tile_dimensions is None:
        tile_dimensions = [32, 32]

    tile_rows, tile_cols = tile_dimensions

    valid_rows = {1, 2, 4, 8, 16, 32}
    valid_cols = {16, 32}

    if tile_rows not in valid_rows:
        raise ValueError(
            f"tile_dimensions[0] (rows) must be one of {sorted(valid_rows)}, got {tile_rows}"
        )
    if tile_cols not in valid_cols:
        raise ValueError(
            f"tile_dimensions[1] (cols) must be one of {sorted(valid_cols)}, got {tile_cols}"
        )

    rows, cols = input_dimensions
    if rows % 32 != 0 or cols % 32 != 0:
        raise ValueError(
            f"Input dimensions must be multiples of 32, got {input_dimensions}"
        )

    # If using full tile dimensions, no conversion needed
    if tile_rows == 32 and tile_cols == 32:
        return tilized_tensor.flatten()

    # Calculate number of tiles
    tile_cnt = (rows // 32) * (cols // 32)
    face_rows = 16
    face_cols = 16

    # Reshape to [num_tiles, 4, 16, 16] for easier face/row manipulation
    # Face order in tilized format: [f0, f1, f2, f3]
    tensor_by_tiles = tilized_tensor.flatten().view(tile_cnt, 4, face_rows, face_cols)

    # Which faces to keep and how many rows from each top-face:
    #   tile_rows <= 16 → only top faces (f0, f1), take tile_rows from each
    #   tile_rows == 32 → all faces, all 16 rows from each
    #   tile_cols == 16 → only left faces (f0, f2)
    #   tile_cols == 32 → both left and right faces
    use_bottom_faces = tile_rows == 32
    use_right_faces = tile_cols == 32
    rows_per_face = tile_rows if tile_rows <= 16 else 16

    # Collect the kept parts of each face across all tiles, then join them in
    # face order into one condensed block per tile.
    segments = [tensor_by_tiles[:, 0, :rows_per_face, :].reshape(tile_cnt, -1)]
    if use_right_faces:
        segments.append(tensor_by_tiles[:, 1, :rows_per_face, :].reshape(tile_cnt, -1))
    if use_bottom_faces:
        segments.append(tensor_by_tiles[:, 2, :, :].reshape(tile_cnt, -1))
    if use_bottom_faces and use_right_faces:
        segments.append(tensor_by_tiles[:, 3, :, :].reshape(tile_cnt, -1))

    condensed = torch.cat(segments, dim=1)  # [tile_cnt, condensed_len]

    # Place condensed data at the head of each tile; remainder stays zero.
    tile_elems = 4 * face_rows * face_cols
    output = torch.zeros(tile_cnt, tile_elems, dtype=tilized_tensor.dtype)
    output[:, : condensed.shape[1]] = condensed

    return output.flatten()


# ─────────────────────────────────────────────────────────────────────────────
# Magnitude helpers
# ─────────────────────────────────────────────────────────────────────────────


def format_elem_max(data_format: DataFormat) -> float:
    """
    Return the maximum representable element magnitude for *data_format*.

    Falls back to `torch.finfo(format_dict[data_format]).max` for non-MX formats.
    """
    if data_format.is_mx_format():
        return MX_FORMAT_MAX_NORMAL[data_format]

    return float(torch.finfo(format_dict[data_format]).max)


def _format_elem_min_magnitude(data_format: DataFormat) -> float:
    """
    Return the minimum stimulus magnitude that avoids denormals / sub-range values
    in *data_format*.

    For MX formats uses the element type's documented minimum magnitude; for other
    formats returns `max(1e-6, finfo(torch_dtype).tiny * 100)`.
    """
    if data_format.is_mx_format():
        return MX_FORMAT_MIN_MAGNITUDE[data_format]

    return max(1e-6, float(torch.finfo(format_dict[data_format]).tiny) * 100)


def compute_safe_input_magnitude_range(
    input_format: DataFormat,
    output_format: DataFormat,
    *,
    input_magnitude_cap: float,
    output_magnitude_cap: float,
    bfloat16_precision_cap: float = 1e4,
) -> tuple[float, float]:
    """
    Combine caller-supplied caps with format-aware rules to produce a
    (min_magnitude, max_magnitude) range for stimuli feeding a tensix op.

    The caller pre-computes both caps from the op's magnitude relation:
      - *input_magnitude_cap*:  max |x| implied by the input format's range
      - *output_magnitude_cap*: max |x| implied by the output format's range
        (e.g. `sqrt(output_format_max)` for a squaring op, `output_format_max / 2`
        for summing two operands, etc.)

    The returned max magnitude is `min(input_magnitude_cap, output_magnitude_cap)`,
    further clamped to *bfloat16_precision_cap* for non-MX bfloat16 inputs to keep
    precision reasonable. Set `bfloat16_precision_cap=math.inf` to disable this.

    The returned min magnitude respects MX minimum magnitudes when either input or
    output is an MX format.
    """
    max_magnitude = min(input_magnitude_cap, output_magnitude_cap)

    input_torch_format = format_dict[input_format]
    if input_torch_format == torch.bfloat16 and not input_format.is_mx_format():
        max_magnitude = min(max_magnitude, bfloat16_precision_cap)

    min_magnitude = _format_elem_min_magnitude(input_format)
    if output_format.is_mx_format():
        min_magnitude = max(min_magnitude, _format_elem_min_magnitude(output_format))

    return min_magnitude, max_magnitude


def apply_log_uniform_magnitudes(
    magnitude_source: torch.Tensor,
    *,
    min_magnitude: float,
    max_magnitude: float,
    cast_to_format: DataFormat,
    sign_source: torch.Tensor = None,
    alternate_sign_every_n: int = None,
) -> torch.Tensor:
    """
    Remap the values in *magnitude_source* to log-uniform magnitudes in
    [min_magnitude, max_magnitude] and apply signs, returning a tensor cast to
    `format_dict[cast_to_format]`.

    Sign selection (mutually exclusive):
      - *sign_source* provided: normalized to [0, 1]; values < 0.5 map to -1, else +1
      - `alternate_sign_every_n=n`: element i gets -1 when (i % n == 0), else +1
      - neither provided: all +1

    The output is clamped to [-max_magnitude, max_magnitude] before casting.
    """
    if sign_source is not None and alternate_sign_every_n is not None:
        raise ValueError(
            "Provide either sign_source or alternate_sign_every_n, not both"
        )

    src_float = magnitude_source.to(torch.float32)
    src_min = src_float.min()
    src_max = src_float.max()
    normalized = (
        (src_float - src_min) / (src_max - src_min)
        if src_max > src_min
        else torch.zeros_like(src_float)
    )

    log_min = torch.log(torch.tensor(min_magnitude, dtype=torch.float32))
    log_max = torch.log(torch.tensor(max_magnitude, dtype=torch.float32))
    magnitudes = torch.exp(log_min + normalized * (log_max - log_min))

    if sign_source is not None:
        sign_float = sign_source.to(torch.float32)
        sign_min = sign_float.min()
        sign_max = sign_float.max()
        sign_normalized = (
            (sign_float - sign_min) / (sign_max - sign_min)
            if sign_max > sign_min
            else torch.zeros_like(sign_float)
        )
        signs = torch.where(sign_normalized < 0.5, -1.0, 1.0)
    elif alternate_sign_every_n is not None:
        signs = torch.where(
            torch.arange(magnitude_source.numel()) % alternate_sign_every_n == 0,
            torch.tensor(-1.0),
            torch.tensor(1.0),
        )
    else:
        signs = torch.ones_like(magnitudes)

    values = signs * magnitudes
    values = torch.clamp(values, -max_magnitude, max_magnitude)
    return values.to(format_dict[cast_to_format])
