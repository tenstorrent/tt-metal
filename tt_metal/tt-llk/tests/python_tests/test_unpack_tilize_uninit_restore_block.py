# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Multi-tile-block cross-op restore test for `_llk_unpack_tilize_uninit_` (Phase 6).

Same restore design as `test_unpack_tilize_uninit_restore` (tilize ->
`_llk_unpack_tilize_uninit_` -> plain datacopy, NO data-format reconfig in
between so uninit is the only state reset), but run 0 tilizes a BLOCK of
``BLOCK_CT_DIM > 1`` column tiles in one init/execute sweep.

`_llk_unpack_tilize_init_(ct_dim=BLOCK_CT_DIM)` programs the tilize
``shift_amount`` / per-column addressing from the block width, so the op leaves
SrcA in a block-tilize state. This pins that the PR's uninit still restores the
canonical single-tile operand baseline: run 1 datacopies EACH tilized tile of
the block back, and the full block must equal ``TilizeGolden(src_A_block)``. A
residual block-tilize state that uninit fails to clear corrupts the run-1
datacopy and diverges from the golden.
"""

import torch
from helpers.format_config import DataFormat
from helpers.golden_generators import TilizeGolden, get_golden_generator
from helpers.llk_params import DestAccumulation, format_dict
from helpers.param_config import input_output_formats, parametrize
from helpers.stimuli_config import StimuliConfig
from helpers.stimuli_generator import generate_stimuli
from helpers.test_config import TestConfig
from helpers.test_variant_parameters import (
    NUM_FACES,
    TEST_FACE_DIMS,
    TILE_COUNT,
    generate_input_dim,
)
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
    num_faces=[4, 2],
    # The new axis: tilize a block of >1 column tiles before uninit.
    block_ct_dim=[2, 4],
)
def test_unpack_tilize_uninit_restore_block(
    formats,
    dest_acc,
    num_faces,
    block_ct_dim,
):
    torch_format = format_dict[formats.output_format]
    # A horizontal block of `block_ct_dim` 32x32 tiles.
    input_dimensions = [32, 32 * block_ct_dim]

    src_A, tile_cnt_A, src_B, tile_cnt_B = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=input_dimensions,
        stimuli_format_B=formats.input_format,
        input_dimensions_B=input_dimensions,
    )

    # Golden: the run-1 datacopy is an identity copy of the run-0 tilized block,
    # so a correct restore yields exactly the tilized operand-A block.
    tilize_function = get_golden_generator(TilizeGolden)
    golden_tensor = tilize_function(
        src_A,
        input_dimensions,
        formats.output_format,
        num_faces,
    ).to(torch_format)

    L1_to_L1_iterations = 2
    configuration = TestConfig(
        "sources/unpack_tilize_uninit_restore_block_test.cpp",
        formats,
        templates=[],
        runtimes=[
            NUM_FACES(num_faces),
            TEST_FACE_DIMS(face_r_dim=16, face_c_dim=16),
            TILE_COUNT(tile_cnt_A),
            generate_input_dim(
                input_dimensions,
                input_dimensions,
                block_ct_dim=block_ct_dim,
                block_rt_dim=1,
            ),
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
