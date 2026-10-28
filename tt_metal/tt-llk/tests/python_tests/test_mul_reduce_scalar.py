# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Fused multiply + reduce-to-scalar LLK test (experimental, Blackhole only).

Exercises the experimental ``mul_reduce_scalar`` LLKs
(``llk_math_mul_reduce_scalar.h`` / ``llk_unpack_mul_reduce_scalar.h``):

    result = sum_over_all_tiles_and_elements(A * B)

stored in element ``[0]`` of the output tile. Per the op's contract (Compute
API doc + on-silicon gtest), only element ``[0]`` is defined — the packer's
other lanes are unspecified — so the test validates the reduced scalar alone.

Kernel B is held at 1.0 (matching the on-silicon gtest and the
``fuser_config/fpu_reduce_scalar.yaml`` recipe), so the multiply reduces to
``sum(A)``. Coverage: bf16 (num_tiles up to 8, the DEST half-sync capacity)
plus native fp32 DEST (up to 4 tiles), for HiFi2/HiFi4, across the full 32x32
tile (num_faces=4) and the 16x32 (num_faces=2) / 16x16 (num_faces=1) "tiny
tiles" (mirroring the on-silicon MulReduceScalarTinyTile gtest suite), plus
1x32 to 8x32 tiles with short faces, which the gtest does not cover.

The element-wise op is a parameter. For ELWADD and ELWSUB, B ~ U[0, 0.25],
so the sum stays positive and away from cancellation.
"""

import pytest
import torch
from helpers.chip_architecture import ChipArchitecture, get_chip_architecture
from helpers.data_format_inference import dest_acc_for
from helpers.format_config import DataFormat, InputOutputFormat
from helpers.llk_params import (
    DestAccumulation,
    MathFidelity,
    MathOperation,
    format_dict,
)
from helpers.param_config import parametrize
from helpers.stimuli_config import StimuliConfig
from helpers.stimuli_generator import StimuliSpec, generate_stimuli
from helpers.test_config import TestConfig
from helpers.test_variant_parameters import (
    MATH_FIDELITY,
    MATH_OP,
    NUM_FACES_C_DIM,
    NUM_FACES_R_DIM,
    TEST_FACE_DIMS,
    TILE_COUNT,
)
from helpers.tile_constants import MAX_TILE_ELEMENTS
from helpers.tile_shape import construct_tile_shape
from helpers.utils import tolerances

# Inputs are always bf16; only the DEST/output precision varies.
FORMATS = [
    InputOutputFormat(DataFormat.Float16_b, DataFormat.Float16_b),
    InputOutputFormat(DataFormat.Float16_b, DataFormat.Float32),
]

# Full 32x32 tile (4 faces) plus the tiny tiles: 16x32 (2 faces, num_faces=2)
# and 16x16 (1 face, num_faces=1), and 1x32 to 8x32 tiles with short faces.
# The reduce collapses every element to [0] regardless of tile geometry.
TILE_DIMENSIONS = [[32, 32], [16, 32], [16, 16], [1, 32], [2, 32], [4, 32], [8, 32]]

# Stale-row poison for SrcA and SrcB (see mul_reduce_scalar_test.cpp).
POISON_TILES = 2
SRCA_POISON_VALUE = 100.0
SRCB_POISON_VALUE = 25.0


GOLDEN_OPS = {
    MathOperation.Elwmul: torch.mul,
    MathOperation.Elwadd: torch.add,
    MathOperation.Elwsub: torch.sub,
}


def _num_tiles_for_format(formats):
    """DEST half-sync holds 8 bf16 tiles or 4 fp32 tiles; every multiply-phase
    product must be resident before the reduce phase consumes it. The cap is in
    DEST tile-slots, so it applies to tiny tiles too (matching the gtest)."""
    return (
        [1, 2, 3, 4]
        if dest_acc_for(formats.output_format) == DestAccumulation.Yes
        else [1, 2, 3, 7, 8]
    )


@parametrize(
    formats=FORMATS,
    math_fidelity=[MathFidelity.HiFi2, MathFidelity.HiFi4],
    num_tiles=_num_tiles_for_format,
    tile_dimensions=TILE_DIMENSIONS,
    math_op=list(GOLDEN_OPS),
)
def test_mul_reduce_scalar(formats, math_fidelity, num_tiles, tile_dimensions, math_op):
    if get_chip_architecture() != ChipArchitecture.BLACKHOLE:
        pytest.skip("mul_reduce_scalar is a Blackhole-only experimental LLK")
    if math_op != MathOperation.Elwmul and tile_dimensions[0] == 8:
        pytest.skip(
            "The ELWADD/ELWSUB MOP reads the stale rows 8-15 of an 8-row face (#60970, remove with #62117)"
        )

    tile_shape = construct_tile_shape(tile_dimensions)
    elements_per_tile = tile_shape.total_tile_size()
    dest_acc = dest_acc_for(formats.output_format)
    input_dimensions = [num_tiles * tile_dimensions[0], tile_dimensions[1]]

    # A ~ U[0, 1] mirrors the on-silicon gtest and keeps the accumulated sum
    # well inside bf16's dynamic range for the larger tile counts. Passing
    # tile_dimensions puts the generator in dense mode (real tiny-tile layout).
    src_A, tile_cnt_A, src_B, tile_cnt_B = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=input_dimensions,
        stimuli_format_B=formats.input_format,
        input_dimensions_B=input_dimensions,
        tile_dimensions=tile_dimensions,
        spec_A=StimuliSpec.uniform(low=0.0, high=1.0),
        spec_B=StimuliSpec.uniform(low=0.0, high=0.25),
    )
    # ELWMUL: B == 1.0 everywhere (matching the gtest and fpu_reduce_scalar.yaml):
    # A * B == A, so the fused op reduces to sum(A) over all tiles/elements.
    # ELWADD and ELWSUB keep B ~ U[0, 0.25] (see the module docstring).
    if math_op == MathOperation.Elwmul:
        src_B = torch.ones(
            tile_cnt_B * elements_per_tile, dtype=format_dict[formats.input_format]
        )

    # Golden mirrors the on-silicon reference (test_mul_reduce_scalar.cpp): the
    # element-wise result summed over every element of every tile, in fp32.
    golden_scalar = float(
        GOLDEN_OPS[math_op](src_A.to(torch.float32), src_B.to(torch.float32))
        .sum()
        .item()
    )

    # The SrcA poison, then the SrcB poison, follow the input tiles in buffer_A as full tiles.
    poison = torch.cat(
        [
            torch.full((POISON_TILES * MAX_TILE_ELEMENTS,), value, dtype=src_A.dtype)
            for value in (SRCA_POISON_VALUE, SRCB_POISON_VALUE)
        ]
    )

    configuration = TestConfig(
        "sources/mul_reduce_scalar_test.cpp",
        formats,
        templates=[
            MATH_FIDELITY(math_fidelity),
            MATH_OP(mathop=math_op),
        ],
        runtimes=[
            TILE_COUNT(num_tiles),
            NUM_FACES_R_DIM(tile_shape.num_faces_r_dim, tile_shape.num_faces_r_dim),
            NUM_FACES_C_DIM(tile_shape.num_faces_c_dim, tile_shape.num_faces_c_dim),
            TEST_FACE_DIMS(face_r_dim=tile_shape.face_r_dim),
        ],
        variant_stimuli=StimuliConfig(
            torch.cat([src_A, poison]),
            formats.input_format,
            src_B,
            formats.input_format,
            formats.output_format,
            tile_count_A=tile_cnt_A + len(poison) // elements_per_tile,
            tile_count_B=tile_cnt_B,
            tile_count_res=1,
            num_faces=tile_shape.total_num_faces(),
            face_r_dim=tile_shape.face_r_dim,
            tile_dimensions=tile_dimensions,
            use_dense_tile_dimensions=True,
            sfpu=False,
        ),
        dest_acc=dest_acc,
    )

    res_from_L1 = configuration.run().result

    assert (
        len(res_from_L1) == elements_per_tile
    ), f"Expected one {elements_per_tile}-element output tile, got {len(res_from_L1)}"

    # The reduced scalar lives in element [0]; every other lane is unspecified.
    device_scalar = float(res_from_L1[0])
    tol = tolerances[formats.output_format]
    assert abs(device_scalar - golden_scalar) <= tol.atol + tol.rtol * abs(
        golden_scalar
    ), (
        f"mul_reduce_scalar mismatch: device={device_scalar} golden={golden_scalar} "
        f"(op={math_op.name}, num_tiles={num_tiles}, tile={tile_dimensions}, fidelity={math_fidelity.name})"
    )
