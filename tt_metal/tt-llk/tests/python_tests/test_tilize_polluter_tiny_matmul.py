# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Cross-op tile-geometry transition test: REGULAR `unpack_tilize` -> TINY matmul.

Phase 3b Direction B. `test_matmul_unpack_tilize` only exercises the tilize->matmul
transition with `num_faces == 4`; this closes the `num_faces < 4` (tiny / partial-face
matmul) corner of that cell.

A coupled pipeline cannot mix tile geometries, so the tiny matmul reads its own
independent, pre-tilized operands (in0 = tiny [face_r,32] SrcB, in1 = regular 32x32
SrcA), exactly as `test_unpack_matmul`'s tiny path does. Run 0 is a regular 32x32
`unpack_tilize` polluter (output discarded to scratch) that leaves the unpacker in a
regular-tilize state; `_llk_unpack_tilize_uninit_` then restores the baseline (gated by
`do_restore`) before the tiny matmul.

Because the tiny-matmul run re-runs `_llk_unpack_hw_configure_` for its operands, this
primarily validates that the regular-tilize -> uninit -> tiny-matmul SEQUENCE produces a
correct result; `do_restore=False` is the negative control.
"""

from dataclasses import dataclass

import torch
from conftest import skip_for_blackhole
from helpers.format_config import DataFormat
from helpers.golden_generators import MatmulGolden, get_golden_generator
from helpers.llk_params import (
    DestAccumulation,
    MathFidelity,
    StochasticRounding,
    Transpose,
    format_dict,
)
from helpers.matmul_sweep import (
    FaceLayoutConfig,
    calculate_matmul_output_faces,
    generate_tile_dims,
)
from helpers.param_config import input_output_formats, parametrize
from helpers.stimuli_config import StimuliConfig
from helpers.stimuli_generator import convert_to_l1_view, generate_face_matmul_data
from helpers.test_config import TestConfig
from helpers.test_variant_parameters import (
    CRK_TILE_DIMM,
    DEST_INDEX,
    IN_TILE_DIMS,
    MATH_FIDELITY,
    NUM_FACES,
    PARTIAL_FACE,
    STOCHASTIC_ROUNDING,
    THROTTLE_LEVEL,
    TILE_COUNT,
    UNPACK_TRANS_FACES,
    UNPACK_TRANS_WITHIN_FACE,
    TemplateParameter,
)
from helpers.tilize_untilize import tilize_block
from helpers.utils import passed_test

# Tiny in0 (SrcB) is 2 horizontal faces; in1 (SrcA) is a regular 4-face 32x32 tile.
TINY_NUM_FACES_IN0 = 2
REGULAR_NUM_FACES_IN1 = 4


@dataclass
class DO_RESTORE(TemplateParameter):
    """Whether `_llk_unpack_tilize_uninit_` runs between the polluter and the matmul."""

    do_restore: bool = True

    def convert_to_cpp(self) -> str:
        return f"constexpr bool DO_RESTORE = {str(self.do_restore).lower()};"


def _tiny_matmul_layout(in0_tile_r_dim: int):
    """Build (tile_dims, face_layout) for a single-output-tile tiny matmul:
    in0 = [in0_tile_r_dim, 32] (SrcB), in1 = [32, 32] (SrcA)."""
    input1_dims = [32, 32]
    tile_dims = generate_tile_dims(
        ([32, 32], input1_dims), in0_tile_r_dim=in0_tile_r_dim
    )
    output_num_faces = calculate_matmul_output_faces(
        num_faces_in0=TINY_NUM_FACES_IN0,
        num_faces_in1=REGULAR_NUM_FACES_IN1,
        is_in0_horizontal=True,
    )
    face = FaceLayoutConfig(
        num_faces_in0=TINY_NUM_FACES_IN0,
        num_faces_in1=REGULAR_NUM_FACES_IN1,
        num_faces=output_num_faces,
        unpack_transpose_faces=Transpose.No,
        unpack_transpose_within_face=Transpose.No,
        partial_face_in0=True,  # SrcB
        partial_face_in1=False,  # SrcA
        partial_face_math=in0_tile_r_dim < 16,
        partial_face_pack=True,
    )
    return tile_dims, face


@skip_for_blackhole
@parametrize(
    formats=input_output_formats(
        [
            DataFormat.Float16_b,
            DataFormat.Float16,
            DataFormat.Float32,
        ],
        same=False,
    ),
    dest_acc=[DestAccumulation.No, DestAccumulation.Yes],
    math_fidelity=[
        MathFidelity.LoFi,
        MathFidelity.HiFi2,
        MathFidelity.HiFi3,
        MathFidelity.HiFi4,
    ],
    in0_tile_r_dim=[8, 4, 2, 1],
    do_restore=[True, False],
)
def test_tilize_polluter_tiny_matmul(
    formats,
    dest_acc,
    math_fidelity,
    in0_tile_r_dim,
    do_restore,
):
    tile_dims, face = _tiny_matmul_layout(in0_tile_r_dim)
    in0_dimensions = tile_dims.in0_dimensions
    in1_dimensions = tile_dims.in1_dimensions

    # Independent tiny/regular matmul operands (run 1 reads these tilized).
    in0 = generate_face_matmul_data(
        num_faces=face.num_faces_in0,
        stimuli_format=formats.input_format,
        input_dimensions=in0_dimensions,
        is_matrix_A=True,  # SrcB uses f0,f1
        face_r_dim=(in0_tile_r_dim if in0_tile_r_dim < 16 else 16),
    )
    in1 = generate_face_matmul_data(
        num_faces=face.num_faces_in1,
        stimuli_format=formats.input_format,
        input_dimensions=in1_dimensions,
        is_matrix_A=False,  # SrcA uses f0,f2
    )

    generate_golden = get_golden_generator(MatmulGolden)
    golden_tensor = generate_golden(
        in0,
        in1,
        formats.output_format,
        math_fidelity,
        input_A_dimensions=in0_dimensions,
        input_B_dimensions=in1_dimensions,
        tilize=True,
        input_A_format=formats.input_format,
        input_B_format=formats.input_format,
    )
    golden_tensor = convert_to_l1_view(
        golden_tensor,
        (in0_dimensions[0], in1_dimensions[1]),
        tile_dimensions=[tile_dims.in0_tile_r_dim, tile_dims.in1_tile_c_dim],
    )

    tilized_in0 = tilize_block(
        in0, dimensions=in0_dimensions, stimuli_format=formats.input_format
    )
    tilized_in1 = tilize_block(
        in1, dimensions=in1_dimensions, stimuli_format=formats.input_format
    )
    tilized_in0_l1_view = convert_to_l1_view(
        tilized_in0,
        in0_dimensions,
        tile_dimensions=[tile_dims.in0_tile_r_dim, tile_dims.in0_tile_c_dim],
    )
    tilized_in1_l1_view = convert_to_l1_view(
        tilized_in1,
        in1_dimensions,
        tile_dimensions=[tile_dims.in1_tile_r_dim, tile_dims.in1_tile_c_dim],
    )

    L1_to_L1_iterations = 2
    configuration = TestConfig(
        "sources/tilize_polluter_tiny_matmul_test.cpp",
        formats,
        templates=[
            STOCHASTIC_ROUNDING(StochasticRounding.No),
            MATH_FIDELITY(math_fidelity),
            THROTTLE_LEVEL(0),
            DO_RESTORE(do_restore=do_restore),
        ],
        runtimes=[
            TILE_COUNT(tile_dims.tile_cnt),
            NUM_FACES(face.num_faces, face.num_faces_in0, face.num_faces_in1),
            UNPACK_TRANS_FACES(Transpose.No),
            UNPACK_TRANS_WITHIN_FACE(Transpose.No),
            PARTIAL_FACE(
                partial_a=face.partial_face_in0,
                partial_face_pack=face.partial_face_pack,
                partial_b=face.partial_face_in1,
                partial_face_math=face.partial_face_math,
            ),
            CRK_TILE_DIMM(tile_dims.ct_dim, tile_dims.rt_dim, tile_dims.kt_dim),
            IN_TILE_DIMS(
                tile_dims.in0_tile_r_dim,
                tile_dims.in0_tile_c_dim,
                tile_dims.in1_tile_r_dim,
                tile_dims.in1_tile_c_dim,
            ),
            DEST_INDEX(0),
        ],
        variant_stimuli=StimuliConfig(
            tilized_in0_l1_view.flatten(),
            formats.input_format,
            tilized_in1_l1_view.flatten(),
            formats.input_format,
            formats.output_format,
            tile_count_A=tile_dims.tile_cnt_in0,
            tile_count_B=tile_dims.tile_cnt_in1,
            tile_count_res=tile_dims.tile_cnt,
        ),
        dest_acc=dest_acc,
        L1_to_L1_iterations=L1_to_L1_iterations,
    )

    res_from_L1 = configuration.run().result

    assert len(res_from_L1) == len(
        golden_tensor
    ), "Result tensor and golden tensor are not of the same length"

    torch_format = format_dict[formats.output_format]
    res_tensor = torch.tensor(res_from_L1, dtype=torch_format)

    # num_faces < 4: compare only the active faces of each output tile.
    num_elements_per_tile = tile_dims.in0_tile_r_dim * tile_dims.in1_tile_c_dim
    TILE_R_DIM, TILE_C_DIM = 32, 32
    for i in range(tile_dims.output_tile_cnt):
        start = i * (TILE_R_DIM * TILE_C_DIM)
        assert passed_test(
            golden_tensor[start : start + num_elements_per_tile],
            res_tensor[start : start + num_elements_per_tile],
            formats.output_format,
        ), f"Assert on tile {i}/{tile_dims.output_tile_cnt} against golden failed"
