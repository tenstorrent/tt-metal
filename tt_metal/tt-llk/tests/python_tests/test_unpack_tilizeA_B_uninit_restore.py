# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Teardown test for `_llk_unpack_tilizeA_B_uninit_`.

Runs tilizeA_B init, then uninit, then a plain ``_llk_unpack_A_`` datacopy of
operand A with no data-format reconfig in between, so uninit is the only thing
that puts the unpacker back at the ``configure_unpack_AB`` baseline. The datacopy
is an identity copy, so the result must equal operand A.

Only ``face_r_dim < 16`` is interesting: that is where the operand baseline and a
fixed 16x16 restore disagree. Wormhole restored ``Tile_x_dim_cntx0`` from a GPR
holding ``256 | 256<<16``, and Blackhole wrote that register even though its init
never touches it. Either one leaves a full-tile per-row datum count behind and
corrupts the datacopy.
"""

import torch
from helpers.format_config import DataFormat
from helpers.golden_generators import DataCopyGolden, get_golden_generator
from helpers.llk_params import DestAccumulation, format_dict
from helpers.param_config import input_output_formats, parametrize
from helpers.stimuli_config import StimuliConfig
from helpers.stimuli_generator import generate_stimuli
from helpers.test_config import TestConfig
from helpers.test_variant_parameters import NUM_FACES, TEST_FACE_DIMS
from helpers.utils import passed_test

# Tiny tiles are always 2 horizontal faces ([face_r_dim, 32] => f0 | f1).
TINY_NUM_FACES = 2


@parametrize(
    # Same input/output format so the datacopy needs no reconfig, which is what
    # isolates the teardown as the only state reset.
    formats=input_output_formats(
        [
            DataFormat.Float16_b,
            DataFormat.Float16,
        ],
        same=True,
    ),
    dest_acc=[DestAccumulation.Yes, DestAccumulation.No],
    face_r_dim=[8, 4, 2, 1],
)
def test_unpack_tilizeA_B_uninit_restore(
    formats,
    dest_acc,
    face_r_dim,
):
    torch_format = format_dict[formats.output_format]
    input_dimensions = [face_r_dim, 32]

    src_A, tile_cnt_A, src_B, tile_cnt_B = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=input_dimensions,
        stimuli_format_B=formats.input_format,
        input_dimensions_B=input_dimensions,
        face_r_dim=face_r_dim,
        num_faces=TINY_NUM_FACES,
    )

    generate_golden = get_golden_generator(DataCopyGolden)
    golden_tensor = generate_golden(
        src_A,
        formats.output_format,
        TINY_NUM_FACES,
        input_dimensions,
        face_r_dim=face_r_dim,
    )

    configuration = TestConfig(
        "sources/unpack_tilizeA_B_uninit_restore_test.cpp",
        formats,
        templates=[],
        runtimes=[
            NUM_FACES(TINY_NUM_FACES),
            TEST_FACE_DIMS(face_r_dim=face_r_dim, face_c_dim=16),
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
            num_faces=TINY_NUM_FACES,
            face_r_dim=face_r_dim,
        ),
        dest_acc=dest_acc,
    )

    res_from_L1 = configuration.run().result

    assert len(res_from_L1) == len(
        golden_tensor
    ), "Result tensor and golden tensor are not of the same length"

    res_tensor = torch.tensor(res_from_L1, dtype=torch_format)

    assert passed_test(
        golden_tensor, res_tensor, formats.output_format
    ), "Assert against golden failed"
