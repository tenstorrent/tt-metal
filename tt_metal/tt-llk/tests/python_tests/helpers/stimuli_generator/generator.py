# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
import math
from typing import Optional

import torch

from ..bfp_format_utils import bfp2b_to_float16b, bfp4b_to_float16b
from ..format_config import MX_FORMAT_MAX_NORMAL, DataFormat
from ..llk_params import format_dict
from ..tile_constants import (
    DEFAULT_TILE_C_DIM,
    DEFAULT_TILE_R_DIM,
    FACE_C_DIM,
    MAX_FACE_R_DIM,
    MAX_NUM_FACES,
    get_tile_params,
)
from .spec import DistributionKind, StimuliSpec
from .strategies import lookup_strategy
from .strategies.structured import _enumerate_representable
from .utils import (
    _clamp_mx_tensors,
    _get_dtype_for_format,
    calculate_tile_and_face_counts,
    calculate_tile_and_face_counts_w_tile_dimensions,
)

# ─────────────────────────────────────────────────────────────────────────────
# Public: single-face generator
# ─────────────────────────────────────────────────────────────────────────────


def generate_face(
    spec: StimuliSpec,
    stimuli_format: DataFormat = DataFormat.Float16_b,
    face_r_dim: int = MAX_FACE_R_DIM,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Generate a single face tensor of shape (face_r_dim * 16,) using *spec*.

    Dispatches to the strategy registered for *spec.distribution*. Callable
    distributions bypass the registry — the callable is invoked directly and
    its return value validated.

    Args:
        spec: Generation specification (distribution, bounds, seed, etc.).
        stimuli_format: Target hardware data format.
        face_r_dim: Rows per face (1-16, default 16).
        generator: External RNG state; when supplied, ``spec.seed`` is ignored.

    Returns:
        1-D tensor with face_r_dim * 16 elements.
    """
    size = face_r_dim * FACE_C_DIM
    dtype = _get_dtype_for_format(stimuli_format)

    if generator is None and spec.seed is not None:
        generator = torch.Generator()
        generator.manual_seed(spec.seed)

    # Callable distributions bypass the strategy registry.
    if callable(spec.distribution):
        result = spec.distribution(size, dtype, generator)
        if not isinstance(result, torch.Tensor):
            raise TypeError(
                f"Custom distribution callable must return a torch.Tensor, "
                f"got {type(result).__name__}"
            )
        if result.ndim != 1:
            raise ValueError(
                f"Custom distribution callable must return a 1-D tensor; "
                f"got {result.ndim}-D tensor with shape {tuple(result.shape)}"
            )
        if len(result) != size:
            raise ValueError(
                f"Custom distribution callable returned {len(result)} elements "
                f"but {size} were expected "
                f"({face_r_dim} rows × {FACE_C_DIM} cols)"
            )
        return result.to(dtype=dtype)

    strategy = lookup_strategy(spec.distribution)
    return strategy.generate_face(spec, stimuli_format, face_r_dim, size, generator)


# ─────────────────────────────────────────────────────────────────────────────
# Private: full operand tensor
# ─────────────────────────────────────────────────────────────────────────────


def _make_generator(spec: StimuliSpec) -> Optional[torch.Generator]:
    """Create a torch.Generator for the given spec.

    Args:
        spec: Generation specification (distribution, bounds, seed, etc.).

    Returns:
        torch.Generator or None if no seed is set.
    """
    if spec.seed is not None:
        gen = torch.Generator()
        gen.manual_seed(spec.seed)
        return gen
    return None


def _run_face_loop(
    spec: StimuliSpec,
    stimuli_format: DataFormat,
    face_r_dim: int,
    num_elements: int,
    generator: Optional[torch.Generator],
) -> torch.Tensor:
    """Generate the tensor face by face, using any per-face overrides and
    zeroing faces listed in masked_faces.
    """
    elements_per_face = face_r_dim * FACE_C_DIM
    faces_needed = math.ceil(num_elements / elements_per_face)

    face_tensors: list[torch.Tensor] = []
    for face_idx in range(faces_needed):
        face_spec = spec
        if spec.face_specs and face_idx < len(spec.face_specs):
            override = spec.face_specs[face_idx]
            if override is not None:
                face_spec = override

        face_tensor = generate_face(
            spec=face_spec,
            stimuli_format=stimuli_format,
            face_r_dim=face_r_dim,
            generator=generator,
        )

        if spec.masked_faces and face_idx in spec.masked_faces:
            face_tensor = torch.zeros_like(face_tensor)

        face_tensors.append(face_tensor)

    return torch.cat(face_tensors)[:num_elements]


def _generate_source_tensor(
    stimuli_format: DataFormat,
    num_elements: int,
    face_r_dim: int,
    spec: StimuliSpec,
    input_dimensions: Optional[list] = None,
) -> torch.Tensor:
    """Generate a full operand tensor of *num_elements* using *spec*.

    Callable distributions and face-based strategies build the tensor one
    face at a time. Short-circuit strategies generate the whole tensor in
    one call. For Bfp4_b, apply the final pack/unpack quantization here.

    Args:
        stimuli_format: Hardware data format for the tensor.
        num_elements: Total number of elements to generate.
        face_r_dim: Number of rows per face.
        spec: Stimuli specification describing how values should be generated.
        input_dimensions: Optional [rows, cols] shape for tensor-level
            distributions that need matrix dimensions.

    Returns:
        A 1-D tensor of length *num_elements*.
    """
    if callable(spec.distribution):
        gen = _make_generator(spec)
        tensor = _run_face_loop(spec, stimuli_format, face_r_dim, num_elements, gen)
        if stimuli_format == DataFormat.Bfp4_b:
            tensor = bfp4b_to_float16b(tensor)
        elif stimuli_format == DataFormat.Bfp2_b:
            tensor = bfp2b_to_float16b(tensor)
        return tensor

    strategy = lookup_strategy(spec.distribution)

    if spec.masked_faces and strategy.short_circuit:
        raise ValueError(
            f"masked_faces cannot be used with distribution={spec.distribution!r} "
            f"because it bypasses the face loop.  Use a per-face distribution "
            f"(uniform, gaussian, saw, …) instead."
        )

    if strategy.short_circuit:
        tensor = strategy.generate_full_tensor(
            spec, stimuli_format, num_elements, input_dimensions, None
        )
    else:
        gen = _make_generator(spec)
        tensor = _run_face_loop(spec, stimuli_format, face_r_dim, num_elements, gen)

    if stimuli_format == DataFormat.Bfp4_b:
        tensor = bfp4b_to_float16b(tensor)
    elif stimuli_format == DataFormat.Bfp2_b:
        tensor = bfp2b_to_float16b(tensor)

    return tensor


# ─────────────────────────────────────────────────────────────────────────────
# Built-in defaults for omitted specs
# ─────────────────────────────────────────────────────────────────────────────


def _default_bfp8b_face(
    size: int, dtype: torch.dtype, gen: Optional[torch.Generator] = None
) -> torch.Tensor:
    integer_part = torch.randint(0, 3, (size,), generator=gen)
    fraction = torch.randint(0, 16, (size,), generator=gen).to(torch.bfloat16) / 16.0
    return integer_part.to(torch.bfloat16) + fraction


def _default_bfp4b_face(
    size: int, dtype: torch.dtype, gen: Optional[torch.Generator] = None
) -> torch.Tensor:
    integer_part = torch.randint(0, 3, (size,), generator=gen)
    fraction = torch.randint(0, 8, (size,), generator=gen).to(torch.bfloat16) / 8.0
    return integer_part.to(torch.bfloat16) + fraction


def _default_bfp2b_face(
    size: int, dtype: torch.dtype, gen: Optional[torch.Generator] = None
) -> torch.Tensor:
    integer_part = torch.randint(0, 3, (size,), generator=gen)
    fraction = torch.randint(0, 4, (size,), generator=gen).to(torch.bfloat16) / 4.0
    return integer_part.to(torch.bfloat16) + fraction


def default_spec_for_format(stimuli_format: DataFormat) -> StimuliSpec:
    """Return the built-in default StimuliSpec for a given data format.

    Defaults are chosen to give reasonable value ranges and avoid overflows
    (e.g. positive ranges for floats, half-range for integers).
    """
    if stimuli_format == DataFormat.MxFp8R:
        return StimuliSpec.gaussian(
            mean=0.1, std=0.05 * MX_FORMAT_MAX_NORMAL[DataFormat.MxFp8R]
        )
    if stimuli_format == DataFormat.MxFp8P:
        return StimuliSpec.gaussian(
            mean=0.1, std=0.05 * MX_FORMAT_MAX_NORMAL[DataFormat.MxFp8P]
        )
    if stimuli_format == DataFormat.Bfp8_b:
        return StimuliSpec(distribution=_default_bfp8b_face)
    if stimuli_format == DataFormat.Bfp4_b:
        return StimuliSpec(distribution=_default_bfp4b_face)
    if stimuli_format == DataFormat.Bfp2_b:
        return StimuliSpec(distribution=_default_bfp2b_face)
    if stimuli_format.is_integer():
        if stimuli_format == DataFormat.UInt32:
            return StimuliSpec.uniform(low=0.0, high=float(2**32 - 2))
        dtype = format_dict[stimuli_format]
        v1_type_max = torch.iinfo(dtype).max // 2
        return StimuliSpec.uniform(low=0.0, high=float(v1_type_max - 1))
    return StimuliSpec.uniform(low=0.1, high=1.1)


# ─────────────────────────────────────────────────────────────────────────────
# Public: top-level stimuli generator
# ─────────────────────────────────────────────────────────────────────────────


def generate_stimuli(
    stimuli_format_A: DataFormat = DataFormat.Float16_b,
    input_dimensions_A: Optional[list] = None,
    stimuli_format_B: DataFormat = DataFormat.Float16_b,
    input_dimensions_B: Optional[list] = None,
    spec_A: Optional[StimuliSpec] = None,
    spec_B: Optional[StimuliSpec] = None,
    tile_dimensions: Optional[list] = None,
    face_r_dim: int = MAX_FACE_R_DIM,
    num_faces: int = MAX_NUM_FACES,
    output_format: Optional[DataFormat] = None,
) -> tuple[torch.Tensor, int, torch.Tensor, int]:
    """Generate test stimuli for two operands.

    When tile_dimensions is provided, operates in dense mode with derived face layout;
    otherwise uses the standard 32x32 tile path.

    Args:
        stimuli_format_A: Hardware data format for operand A.
        input_dimensions_A: [height, width] in elements (default [32, 32]).
        stimuli_format_B: Hardware data format for operand B.
        input_dimensions_B: [height, width] in elements (default [32, 32]).
        spec_A: Generation spec for operand A (default: format-aware built-in spec).
        spec_B: Generation spec for operand B (default: format-aware built-in spec).
        tile_dimensions: [rows, cols] tile size for dense mode (default None = standard path).
        face_r_dim: Rows per face, 1-16 (ignored in dense mode, default 16).
        num_faces: Faces per tile for partial-face case (ignored in dense mode, default 4).
        output_format: Clamp outputs for mixed MX-format pairs when set.

    Returns:
        (srcA_tensor, tile_cnt_A, srcB_tensor, tile_cnt_B).
    """
    _spec_B_originally_none = spec_B is None

    if input_dimensions_A is None:
        input_dimensions_A = [DEFAULT_TILE_R_DIM, DEFAULT_TILE_C_DIM]
    if input_dimensions_B is None:
        input_dimensions_B = [DEFAULT_TILE_R_DIM, DEFAULT_TILE_C_DIM]
    if spec_A is None:
        spec_A = default_spec_for_format(stimuli_format_A)
    if spec_B is None:
        spec_B = default_spec_for_format(stimuli_format_B)

    # ULP_SWEEP: auto-size input_dimensions_A from the count of representable values,
    # ignoring any caller-supplied value. Mirror B if the caller didn't specify spec_B.
    if spec_A.distribution == DistributionKind.ULP_SWEEP:
        _ulp_vals = _enumerate_representable(stimuli_format_A, spec_A.low, spec_A.high)
        _num_ulp = _ulp_vals.numel()
        _tile_size = DEFAULT_TILE_R_DIM * DEFAULT_TILE_C_DIM
        _num_tiles = max(1, math.ceil(_num_ulp / _tile_size))
        # Align to 16 tiles when above the DestSync.Half limit (8) so the layout
        # satisfies both DestSync.Half (needs num_tiles % 8 == 0) and DestSync.Full
        # (needs num_tiles % 16 == 0 when > 16). LCM(8, 16) = 16.
        if _num_tiles > 8:
            _num_tiles = math.ceil(_num_tiles / 16) * 16
        input_dimensions_A = [DEFAULT_TILE_R_DIM, DEFAULT_TILE_C_DIM * _num_tiles]
        if _spec_B_originally_none:
            input_dimensions_B = list(input_dimensions_A)

    if tile_dimensions is not None:
        if face_r_dim != MAX_FACE_R_DIM:
            raise ValueError(
                f"tile_dimensions and face_r_dim are mutually exclusive: "
                f"when tile_dimensions is provided, face_r_dim is derived "
                f"automatically. Got tile_dimensions={tile_dimensions}, "
                f"face_r_dim={face_r_dim}."
            )
        if num_faces != MAX_NUM_FACES:
            raise ValueError(
                f"tile_dimensions and num_faces are mutually exclusive: "
                f"when tile_dimensions is provided, num_faces is derived "
                f"automatically.  Got tile_dimensions={tile_dimensions}, "
                f"num_faces={num_faces}."
            )
        # Dense mode: derive face layout from tile_dimensions
        face_r_dim, num_faces_r_dim, num_faces_c_dim = get_tile_params(tile_dimensions)
        num_faces = num_faces_r_dim * num_faces_c_dim
        tile_cnt_A, tile_cnt_B, _ = calculate_tile_and_face_counts_w_tile_dimensions(
            input_dimensions_A,
            input_dimensions_B,
            face_r_dim,
            num_faces,
            tile_dimensions,
        )
    else:
        # Standard 32×32 tile path
        tile_cnt_A, tile_cnt_B, _ = calculate_tile_and_face_counts(
            input_dimensions_A, input_dimensions_B, face_r_dim, num_faces
        )

    num_elements_A = input_dimensions_A[0] * input_dimensions_A[1]
    num_elements_B = input_dimensions_B[0] * input_dimensions_B[1]

    srcA_tensor = _generate_source_tensor(
        stimuli_format=stimuli_format_A,
        num_elements=num_elements_A,
        face_r_dim=face_r_dim,
        spec=spec_A,
        input_dimensions=input_dimensions_A,
    )
    srcB_tensor = _generate_source_tensor(
        stimuli_format=stimuli_format_B,
        num_elements=num_elements_B,
        face_r_dim=face_r_dim,
        spec=spec_B,
        input_dimensions=input_dimensions_B,
    )

    srcA_tensor, srcB_tensor = _clamp_mx_tensors(
        srcA_tensor, srcB_tensor, stimuli_format_A, stimuli_format_B, output_format
    )

    return srcA_tensor, tile_cnt_A, srcB_tensor, tile_cnt_B
