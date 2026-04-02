# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import torch
from conftest import skip_for_wormhole
from helpers.format_config import DataFormat
from helpers.llk_params import DestAccumulation, DestSync, format_dict
from helpers.param_config import (
    input_output_formats,
    parametrize,
)
from helpers.stimuli_config import StimuliConfig
from helpers.stimuli_generator import generate_stimuli
from helpers.test_config import TestConfig
from helpers.test_variant_parameters import (
    DEST_SYNC,
    IN_TILE_DIMS,
    NUM_FACES,
    generate_input_dim,
)
from helpers.utils import passed_test


def generate_specific_stimuli(input_dimensions, r_dim, data_format):
    """Generate specific stimuli pattern for pack untilize test."""
    valid_datum = 1.0
    invalid_datum = -1.0
    row_offset = 2 * input_dimensions[1]
    face_offset = 16
    num_faces = (input_dimensions[0] * input_dimensions[1]) // (16 * 16)
    src_A = []
    for i in range(num_faces):
        for j in range(r_dim):
            for k in range(16):
                src_A.append(valid_datum * (row_offset * j + face_offset * i + k))
        for j in range(16 - r_dim):
            for k in range(16):
                src_A.append(invalid_datum * (row_offset * j + face_offset * i + k))
    src_A = torch.tensor(src_A, dtype=format_dict[data_format])
    return src_A


def generate_golden_output(src_A, input_dimensions, r_dim, data_format):
    """Generate expected golden output for pack untilize test."""
    golden = []
    row_offset = 16
    block_offset = 16 * 16
    num_blocks = (input_dimensions[1] // 16) * 2
    for i in range(r_dim):
        for j in range(num_blocks):
            for k in range(16):
                golden.append(src_A[i * row_offset + j * block_offset + k])
    golden_tensor = torch.tensor(golden, dtype=format_dict[data_format])
    return golden_tensor


def print_stimuli_and_golden(src_A, golden, input_dimensions, r_dim):
    """Print stimuli and golden output for debugging."""
    print("SrcA")
    for i in range(0, len(src_A), 16):
        row = (i // 16) % 16
        face = (i // 256) % 4
        tile = i // 1024
        print(f"T{tile}F{face}R{row}\t", src_A[i : i + 16])
    print("Golden")
    for i in range(0, len(golden), 16):
        segment = (i // 16) % ((input_dimensions[1] * 2) // 16)
        row = i // (input_dimensions[1] * 2)
        print(f"R{row}S{segment}\t", golden[i : i + 16])


@skip_for_wormhole
@parametrize(
    formats=input_output_formats([DataFormat.Float16_b]),
    dest_acc=[DestAccumulation.No],
    input_dimensions=[[32, 32], [32, 64], [32, 128], [32, 256]],
    r_dim=[1, 2, 4, 8, 16],
    dest_sync=[DestSync.Half],
)
def test_pack_untilize(
    formats, dest_acc, input_dimensions, r_dim, dest_sync, workers_tensix_coordinates
):

    src_A, tile_cnt_A, src_B, tile_cnt_B = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=input_dimensions,
        stimuli_format_B=formats.input_format,
        input_dimensions_B=input_dimensions,
        sfpu=False,
    )

    # src_A = generate_specific_stimuli(input_dimensions, r_dim, formats.input_format)
    golden_tensor = generate_golden_output(
        src_A, input_dimensions, r_dim, formats.input_format
    )
    # print_stimuli_and_golden(src_A, golden_tensor, input_dimensions, r_dim)

    configuration = TestConfig(
        "sources/dense_pack_untilize_test.cpp",
        formats,
        templates=[
            generate_input_dim(
                input_dimensions,
                input_dimensions,
            ),
            DEST_SYNC(dest_sync),
        ],
        runtimes=[NUM_FACES(4), IN_TILE_DIMS(r_dim, 32, 32, 32)],
        variant_stimuli=StimuliConfig(
            src_A,
            formats.input_format,
            src_B,
            formats.input_format,
            formats.output_format,
            tile_count_A=tile_cnt_A,
            tile_count_B=tile_cnt_B,
            tile_count_res=tile_cnt_A,
            sfpu=False,
        ),
        dest_acc=dest_acc,
        unpack_to_dest=False,
    )

    res_from_L1 = configuration.run(workers_tensix_coordinates).result
    # Since input and output shapes are identical we always specify r_dim of 16 for stimulus
    # because stimulus must have 4 faces and 16 is only valid r_dim for 4 faces
    # In output for smaller actual r_dims lower portion is unused and therefore truncated
    res_from_L1 = res_from_L1[0 : r_dim * input_dimensions[1] * 2]

    assert len(res_from_L1) == len(
        golden_tensor
    ), "Result tensor and golden tensor are not of the same length"

    res_tensor = torch.tensor(res_from_L1, dtype=format_dict[formats.output_format])

    assert passed_test(
        golden_tensor, res_tensor, formats.output_format
    ), "Assert against golden failed"
