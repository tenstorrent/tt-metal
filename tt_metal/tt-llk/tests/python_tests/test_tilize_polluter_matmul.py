# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Cross-op tile-geometry transition test: `unpack_tilize` (polluter) -> regular matmul.

This is the "tiny tile unpack tilize + regular matmul" case from the unpacker-stride
coverage roadmap (Phase 3b). A coupled pipeline (tilize output feeding the matmul)
cannot mix tile geometries, so this uses a **state-leak** design instead:

    run 0: `unpack_tilize` at geometry G0 (the polluter; output discarded to scratch).
           G0 = tiny when face_r_dim < 16, regular when face_r_dim == 16.
    transition (only when do_restore=True): `_llk_unpack_tilize_uninit_` (PR C1) +
           `_llk_unpack_hw_configure_` re-establishing the regular 32x32 operand baseline
           (a tiny->regular geometry change needs a full reconfigure, not a stride-only one).
    run 1: a REGULAR 32x32 `_llk_unpack_AB_matmul_` on independent, pre-tilized
           operands read directly from buffer_A[0] / buffer_B[0].

`_llk_unpack_AB_matmul_init_` only reprograms x-end and the ZW *counter*; it does NOT
reset the SrcA Y-stride / Z-stride / Tile_x_dim that `unpack_tilize` mutates (those are
restored by `_llk_unpack_tilize_uninit_`). So a tiny tilize that is not properly uninit'd
leaks its strides into the matmul, corrupting the result.

The `do_restore` toggle makes this a controlled experiment:
    do_restore=True  -> transition restores the baseline; matmul must match golden.
    do_restore=False -> no restore; on a correct (PR) build this always exposes the leak
                        and corrupts the matmul, for every polluter geometry including the
                        regular G0 (face_r_dim=16). Even when the regular polluter's
                        Y-stride already matches, `unpack_tilize` leaves `tilize_mode` set
                        and `Tile_x_dim` covering the whole tile row, which
                        `_llk_unpack_AB_matmul_init_` does not reset — so divergence is
                        guaranteed and (16, False) is a valid negative-control point.
"""

from dataclasses import dataclass

import torch
from conftest import skip_for_blackhole
from helpers.format_config import DataFormat
from helpers.golden_generators import MatmulGolden, get_golden_generator
from helpers.llk_params import DestAccumulation, MathFidelity, format_dict
from helpers.param_config import input_output_formats, parametrize
from helpers.stimuli_config import StimuliConfig
from helpers.stimuli_generator import convert_to_l1_view, generate_face_matmul_data
from helpers.test_config import TestConfig
from helpers.test_variant_parameters import (
    MATH_FIDELITY,
    NUM_FACES,
    TEST_FACE_DIMS,
    TemplateParameter,
    generate_input_dim,
)
from helpers.tilize_untilize import tilize_block
from helpers.utils import passed_test

# The matmul victim is always a regular 32x32 (4-face) tile.
MATMUL_NUM_FACES = 4
REGULAR_FACE_R_DIM = 16


@dataclass
class DO_RESTORE(TemplateParameter):
    """Whether the run-0 -> run-1 transition restores the tilize-mutated operand baseline.

    True  -> `_llk_unpack_tilize_uninit_` + `_llk_unpack_hw_configure_` (regular baseline)
             before the matmul, so it reads correct operands.
    False -> nothing; the (possibly tiny) tilize state leaks straight into the matmul.
    """

    do_restore: bool = True

    def convert_to_cpp(self) -> str:
        return f"constexpr bool DO_RESTORE = {str(self.do_restore).lower()};"


@skip_for_blackhole
@parametrize(
    # Same format for both runs so skipping the restore (do_restore=False) does not
    # introduce a data-format mismatch — isolating the unpacker-stride leak.
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
    # Polluter (run-0) tilize face row count: 16 == regular (control), <16 == tiny.
    polluter_face_r_dim=[16, 4, 1],
    do_restore=[True, False],
)
def test_tilize_polluter_matmul(
    formats,
    dest_acc,
    math_fidelity,
    polluter_face_r_dim,
    do_restore,
):
    torch_format = format_dict[formats.output_format]
    mm_dimensions = [32, 32]

    # Independent, regular 32x32 matmul operands (run 1 reads these tilized).
    in0 = generate_face_matmul_data(
        num_faces=MATMUL_NUM_FACES,
        stimuli_format=formats.input_format,
        input_dimensions=mm_dimensions,
        is_matrix_A=True,
        face_r_dim=REGULAR_FACE_R_DIM,
    )
    in1 = generate_face_matmul_data(
        num_faces=MATMUL_NUM_FACES,
        stimuli_format=formats.input_format,
        input_dimensions=mm_dimensions,
        is_matrix_A=False,
    )

    generate_golden = get_golden_generator(MatmulGolden)
    golden_tensor = generate_golden(
        in0,
        in1,
        formats.output_format,
        math_fidelity,
        input_A_dimensions=mm_dimensions,
        input_B_dimensions=mm_dimensions,
        tilize=True,
        input_A_format=formats.input_format,
        input_B_format=formats.input_format,
    )
    golden_tensor = convert_to_l1_view(
        golden_tensor,
        (mm_dimensions[0], mm_dimensions[1]),
        tile_dimensions=mm_dimensions,
    )

    # Pre-tilize the operands and lay them out exactly as L1 expects them; run-1's
    # matmul reads them directly, run-0's tilize reads the same bytes as raw row-major
    # input (its output is discarded, so the garbage does not matter).
    tilized_in0 = tilize_block(
        in0, dimensions=mm_dimensions, stimuli_format=formats.input_format
    )
    tilized_in1 = tilize_block(
        in1, dimensions=mm_dimensions, stimuli_format=formats.input_format
    )
    tilized_in0_l1_view = convert_to_l1_view(
        tilized_in0, mm_dimensions, tile_dimensions=mm_dimensions
    )
    tilized_in1_l1_view = convert_to_l1_view(
        tilized_in1, mm_dimensions, tile_dimensions=mm_dimensions
    )

    polluter_num_faces = 4 if polluter_face_r_dim == REGULAR_FACE_R_DIM else 2

    L1_to_L1_iterations = 2
    configuration = TestConfig(
        "sources/tilize_polluter_matmul_test.cpp",
        formats,
        templates=[
            generate_input_dim(mm_dimensions, mm_dimensions),
            MATH_FIDELITY(math_fidelity),
            DO_RESTORE(do_restore=do_restore),
        ],
        runtimes=[
            NUM_FACES(polluter_num_faces),
            TEST_FACE_DIMS(face_r_dim=polluter_face_r_dim, face_c_dim=16),
        ],
        variant_stimuli=StimuliConfig(
            tilized_in0_l1_view.flatten(),
            formats.input_format,
            tilized_in1_l1_view.flatten(),
            formats.input_format,
            formats.output_format,
            tile_count_A=1,
            tile_count_B=1,
            tile_count_res=1,
        ),
        dest_acc=dest_acc,
        L1_to_L1_iterations=L1_to_L1_iterations,
    )

    res_from_L1 = configuration.run().result

    assert len(res_from_L1) == len(
        golden_tensor
    ), "Result tensor and golden tensor are not of the same length"

    res_tensor = torch.tensor(res_from_L1, dtype=torch_format)

    result_matches = passed_test(
        golden_tensor,
        res_tensor,
        formats.output_format,
        L1_to_L1_iterations=L1_to_L1_iterations,
        # do_restore=False is a negative control that *expects* a mismatch, so
        # suppress the (otherwise alarming) ERROR-level diff dump for it. The
        # do_restore=True path keeps it on so an unexpected divergence is shown.
        print_errors=do_restore,
    )

    if do_restore:
        # Correct path: uninit + FACE_ROW_MAJOR reconfig on BOTH operands re-establishes
        # the regular 32x32 baseline, so the matmul reads its operands correctly.
        assert (
            result_matches
        ), "restore (uninit + reconfig) failed: matmul diverged from golden"
    else:
        # Negative control: run-1 performs NO hw_configure and relies entirely on the
        # restore that we skipped here, so the polluter tilize state (tilize_mode, mutated
        # Tile_x_dim / Y-stride, and — for tiny polluters — a <4-face descriptor) leaks into
        # the regular matmul and MUST corrupt the result. This proves the restore is
        # load-bearing (and that _llk_unpack_AB_matmul_init_ does not reset that state).
        assert (
            not result_matches
        ), "expected a state leak without restore, but the matmul matched golden"
