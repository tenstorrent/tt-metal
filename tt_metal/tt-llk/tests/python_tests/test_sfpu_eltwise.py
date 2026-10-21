# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
llk::sfpu elementwise face selection.

``eltwise_unary`` / ``eltwise_binary`` / ``eltwise_ternary`` (``llk_lib/sfpu/eltwise.h``) own both loops a hand-written SFPU functor writes
today: the face walk that ``_llk_math_eltwise_unary_sfpu_params_`` drives from a
runtime ``VectorMode``, and the ``ITERATIONS`` loop over dest vectors inside each face.
DEST is a 2x2 face grid, so the TensorShape the call walks picks a sub-grid of it,
which is exactly what the vector modes expressed:

    num_faces 2x2 -> faces 0,1,2,3   (VectorMode::RC)
    num_faces 1x2 -> faces 0,1       (VectorMode::R)
    num_faces 2x1 -> faces 0,2       (VectorMode::C)

The 2x1 case is the interesting one: faces 0 and 2 are two face-advances apart, and the
walk reaches face 2 by advancing past the skipped face 1 rather than by any notion of a
stride. A walk that visited only the active faces would land on face 1 instead, so this
case is what pins the behaviour down.

The tile is a full 32x32 through unpack and pack in every variant; only the shape changes.
The op is negation, so a face-selection error shows up as a sign flip -- far larger than
the format tolerance the result is compared at. The comparison is tolerance-based because
the L1 -> DEST -> L1 round trip is lossy when the dest is narrower than the L1 format.
"""

import pytest
import torch
from helpers.format_config import DataFormat
from helpers.llk_params import DestAccumulation, format_dict
from helpers.param_config import input_output_formats, parametrize
from helpers.stimuli_config import StimuliConfig
from helpers.stimuli_generator import generate_stimuli
from helpers.test_config import TestConfig
from helpers.test_variant_parameters import (
    NUM_BLOCKS,
    NUM_FACES_C_DIM,
    NUM_FACES_R_DIM,
    NUM_TILES_IN_BLOCK,
    SFPU_ELTWISE_ARITY,
    SFPU_WALK_FACE_R_DIM,
)
from helpers.tile_constants import FACE_C_DIM, MAX_FACE_R_DIM
from helpers.utils import passed_test

FACE_ELEMENTS = MAX_FACE_R_DIM * FACE_C_DIM
DEST_FACES_R = 2
DEST_FACES_C = 2
# One sfpi vector is 4 face rows by 8 columns of one parity; the walk covers whole 4-row groups.
DEST_VECTOR_ROWS = 4


def selected_faces(num_faces_r_dim: int, num_faces_c_dim: int) -> list[int]:
    """Faces of the 2x2 DEST grid that a shape of this many faces selects."""
    return [
        face
        for face in range(DEST_FACES_R * DEST_FACES_C)
        if (face // DEST_FACES_C) < num_faces_r_dim
        and (face % DEST_FACES_C) < num_faces_c_dim
    ]


@parametrize(
    formats=input_output_formats([DataFormat.Float16_b, DataFormat.Float32], same=True),
    dest_acc=[DestAccumulation.No, DestAccumulation.Yes],
    # Rows per face the walk is handed. Below 16 the walk covers ceil(face_r_dim / 4) groups of 4
    # rows -- even and odd columns each -- so 1 and 2 rows pin the odd-column half of the walk.
    walk_face_r_dim=[16, 8, 4, 2, 1],
    # (num_faces_r_dim, num_faces_c_dim) -> the sub-grid unary should touch. Short faces are only
    # in the math TensorShape coverage table as 1x1 / 1x2 grids.
    face_grid=lambda walk_face_r_dim: (
        [(2, 2), (1, 2), (2, 1), (1, 1)] if walk_face_r_dim == 16 else [(1, 2), (1, 1)]
    ),
    # Inputs of the call. Every operand is the same tile and the functor reduces to negation,
    # so the golden is unchanged while the walk addresses one, two or three inputs.
    arity=[1, 2, 3],
)
def test_sfpu_eltwise_face_selection(
    formats, dest_acc, walk_face_r_dim, face_grid, arity
):
    num_faces_r_dim, num_faces_c_dim = face_grid

    # Two tiles, so the Dest tile index is 0 for one and 1 for the other: this is what
    # exercises the Dest base unary programs from the tile index, not just tile 0.
    input_dimensions = [64, 32]
    src_A, tile_cnt_A, src_B, tile_cnt_B = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=input_dimensions,
        stimuli_format_B=formats.input_format,
        input_dimensions_B=input_dimensions,
    )

    configuration = TestConfig(
        "sources/sfpu_eltwise_test.cpp",
        formats,
        dest_acc=dest_acc,
        # The face dims must be TEMPLATES, not runtimes: the kernel builds a constexpr
        # TensorShape from them to pass as a non-type template argument, so a runtime
        # struct field would leave the shape non-constant.
        templates=[
            NUM_FACES_R_DIM(num_faces_r_dim, num_faces_r_dim),
            NUM_FACES_C_DIM(num_faces_c_dim, num_faces_c_dim),
            SFPU_ELTWISE_ARITY(arity),
            SFPU_WALK_FACE_R_DIM(walk_face_r_dim),
        ],
        runtimes=[
            NUM_BLOCKS(1),
            NUM_TILES_IN_BLOCK(2),
        ],
        variant_stimuli=StimuliConfig(
            src_A,
            formats.input_format,
            src_B,
            formats.input_format,
            formats.output_format,
            tile_count_A=tile_cnt_A,
            tile_count_B=tile_cnt_B,
            tile_count_res=tile_cnt_A,
        ),
    )

    res_from_L1 = configuration.run().result

    torch_format = format_dict[formats.output_format]
    res_tensor = torch.tensor(res_from_L1, dtype=torch_format)
    assert (
        res_tensor.numel() == src_A.numel()
    ), f"Result length {res_tensor.numel()} does not match source length {src_A.numel()}"

    # Golden: the selected faces are negated in place, the rest pass through untouched.
    # Faces are contiguous in the tilized layout, FACE_ELEMENTS apart.
    golden = src_A.to(torch_format).clone()
    touched = selected_faces(num_faces_r_dim, num_faces_c_dim)
    tile_count = src_A.numel() // (FACE_ELEMENTS * DEST_FACES_R * DEST_FACES_C)
    # Within a face, the walk negates whole 4-row groups: the first ceil(face_r_dim / 4) * 4 rows.
    walked_rows = -(-walk_face_r_dim // DEST_VECTOR_ROWS) * DEST_VECTOR_ROWS
    walked_elements = walked_rows * FACE_C_DIM
    for tile in range(tile_count):
        for face in touched:
            lo = (tile * DEST_FACES_R * DEST_FACES_C + face) * FACE_ELEMENTS
            golden[lo : lo + walked_elements] = -golden[lo : lo + walked_elements]

    # Compare at the format's tolerance, not exactly: the L1 -> DEST -> L1 round trip is
    # lossy whenever the dest is narrower than the L1 format (Float32 through a 16-bit
    # dest), so exact equality would fail on precision and mask what this test is for.
    if not passed_test(golden, res_tensor, formats.output_format):
        untouched = [f for f in range(DEST_FACES_R * DEST_FACES_C) if f not in touched]
        faces_per_tile = DEST_FACES_R * DEST_FACES_C
        wrong = [
            (tile, face)
            for tile in range(tile_count)
            for face in range(faces_per_tile)
            if not passed_test(
                golden[
                    (tile * faces_per_tile + face)
                    * FACE_ELEMENTS : (tile * faces_per_tile + face + 1)
                    * FACE_ELEMENTS
                ],
                res_tensor[
                    (tile * faces_per_tile + face)
                    * FACE_ELEMENTS : (tile * faces_per_tile + face + 1)
                    * FACE_ELEMENTS
                ],
                formats.output_format,
                print_errors=False,
            )
        ]
        pytest.fail(
            f"shape {num_faces_r_dim}x{num_faces_c_dim} (face_r_dim {walk_face_r_dim}) should negate "
            f"rows 0-{walked_rows - 1} of faces {touched} "
            f"and leave {untouched} untouched in each of {tile_count} tiles; "
            f"(tile, face) pairs {wrong} disagree with the golden"
        )
