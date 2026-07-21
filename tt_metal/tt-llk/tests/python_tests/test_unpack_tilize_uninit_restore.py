# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Cross-op unpacker-state restore test for `_llk_unpack_tilize_uninit_`.

This exercises a gap left by the existing tilize tests: every other call site of
`_llk_unpack_tilize_uninit_` (matmul_unpack_tilize, the fuser tilize node, the
C++ sweep) restores with ``num_faces=4`` / ``face_r_dim=16``. The PR threads the
operand's ``num_faces``/``face_r_dim`` through uninit so it restores the SrcA
tile-descriptor + ``Tile_x_dim_cntx0`` back to the *operand* baseline programmed
by ``configure_unpack_AB`` (not a hardcoded 16x16, 4-face baseline).

The C++ source runs two ops back-to-back on the SAME operand format:
    1. ``unpack_tilize`` of operand A (with the parameterized ``num_faces``)
    2. ``_llk_unpack_tilize_uninit_`` (the restore under test)
    3. a plain ``_llk_unpack_A_`` datacopy of the tilized tile, with NO
       data-format reconfig in between.

Because there is no reconfig, the uninit restore is the only thing that resets
the unpacker state between the two ops. If it leaves the tile-descriptor Z-dim
(num_faces), the Y-dim, ``tilize_mode``, or ``Tile_x_dim_cntx0`` in a
tilize-specific / wrong-faces state, the second datacopy reads corrupted data
and the result diverges from ``TilizeGolden(src_A, num_faces)``.

``num_faces ∈ {1, 2}`` specifically covers the tile-descriptor Z-dim restore
that no existing tilize test reaches.
"""

import pytest
import torch
from helpers.chip_architecture import ChipArchitecture, get_chip_architecture
from helpers.format_config import DataFormat
from helpers.golden_generators import TilizeGolden, get_golden_generator
from helpers.llk_params import DestAccumulation, format_dict
from helpers.param_config import input_output_formats, parametrize
from helpers.stimuli_config import StimuliConfig
from helpers.stimuli_generator import generate_stimuli
from helpers.test_config import TestConfig
from helpers.test_variant_parameters import NUM_FACES, TEST_FACE_DIMS
from helpers.utils import passed_test


@parametrize(
    # Same input/output format for both ops so the second op needs NO data-format
    # reconfig — isolating the uninit restore as the only state reset.
    formats=input_output_formats(
        [
            DataFormat.Float16_b,
            DataFormat.Float16,
            DataFormat.Float32,
        ],
        same=True,
    ),
    dest_acc=[DestAccumulation.Yes, DestAccumulation.No],
    # num_faces=1,2 are the new coverage; 4 is the existing-baseline control.
    num_faces=[4, 2, 1],
)
def test_unpack_tilize_uninit_restore(
    formats,
    dest_acc,
    num_faces,
):
    # BH unpack_tilize does not support num_faces=1 (LLK asserts num_faces in {2, 4}).
    # WH supports num_faces=1.
    if num_faces == 1 and get_chip_architecture() == ChipArchitecture.BLACKHOLE:
        pytest.skip("BH unpack_tilize does not support num_faces=1")

    torch_format = format_dict[formats.output_format]
    input_dimensions = [32, 32]

    src_A, tile_cnt_A, src_B, tile_cnt_B = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=input_dimensions,
        stimuli_format_B=formats.input_format,
        input_dimensions_B=input_dimensions,
    )

    # Golden: the run-1 datacopy is an identity copy of the run-0 tilized tile,
    # so a correct restore yields exactly the tilized operand A.
    tilize_function = get_golden_generator(TilizeGolden)
    golden_tensor = tilize_function(
        src_A,
        input_dimensions,
        formats.output_format,
        num_faces,
    ).to(torch_format)

    L1_to_L1_iterations = 2
    configuration = TestConfig(
        "sources/unpack_tilize_uninit_restore_test.cpp",
        formats,
        templates=[],
        runtimes=[
            NUM_FACES(num_faces),
            # Normal (non-tiny) tile: face_r_dim=16. The tiny-tile face_r_dim<16
            # coverage lives in test_unpack_tilize_uninit_restore_tiny.py.
            TEST_FACE_DIMS(face_r_dim=16, face_c_dim=16),
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
            num_faces=num_faces,
            write_full_tiles=True,  # tilize input needs full tiles in L1
        ),
        dest_acc=dest_acc,
        L1_to_L1_iterations=L1_to_L1_iterations,
    )

    res_from_L1 = configuration.run().result

    assert len(res_from_L1) == len(
        golden_tensor
    ), "Result tensor and golden tensor are not of the same length"

    res_tensor = torch.tensor(res_from_L1, dtype=torch_format)

    assert passed_test(
        golden_tensor,
        res_tensor,
        formats.output_format,
        L1_to_L1_iterations=L1_to_L1_iterations,
    ), "Assert against golden failed"
