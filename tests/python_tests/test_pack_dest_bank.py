# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import torch
from conftest import skip_for_wormhole
from helpers.chip_architecture import ChipArchitecture, get_chip_architecture
from helpers.format_config import DataFormat
from helpers.llk_params import DestAccumulation, L1Accumulation, Tilize, format_dict
from helpers.param_config import (
    input_output_formats,
    parametrize,
)
from helpers.stimuli_config import StimuliConfig
from helpers.stimuli_generator import generate_stimuli
from helpers.test_config import TestConfig
from helpers.test_variant_parameters import (
    DEST_INDEX,
    L1_ACC,
    NUM_FACES,
    TILE_COUNT,
    TILIZE,
    generate_input_dim,
)
from helpers.utils import passed_test


def get_valid_tilize_datacopy(formats):
    """
    Get valid tilize options for _llk_math_eltwise_unary_datacopy_

    - Blackhole and Wormhole have differing APIs:
        - Blackhole: Has tilize argument (SW workaround for HW bug)
        - Wormhole: No tilize argument (No SW workaround needed)
    - Tilize cannot be enabled if input format is Bfp8_b (HW limitation)

    Therefore we only test tilization on Blackhole
    """

    chip_arch = get_chip_architecture()

    if chip_arch == ChipArchitecture.WORMHOLE:
        return [Tilize.No]

    if formats.input_format == DataFormat.Bfp8_b:
        return [Tilize.No]

    return [Tilize.No, Tilize.Yes]


def get_valid_num_faces_datacopy(tilize):
    """
    Get valid num_faces options for _llk_math_eltwise_unary_datacopy_

    - Number of faces must be 4 when tilization is enabled (SW limitation)
    - Otherwise num_faces can be 1, 2, or 4
    """

    if tilize == Tilize.Yes:
        return [4]

    return [1, 2, 4]


@skip_for_wormhole
@parametrize(
    formats=input_output_formats(
        [
            # DataFormat.Float32,
            DataFormat.Float16,
            DataFormat.Float16_b,
            # DataFormat.Bfp8_b,
        ]
    ),
    dest_acc=[DestAccumulation.No, DestAccumulation.Yes],
    l1_acc=[L1Accumulation.No],
    num_faces=4,
    tilize=[Tilize.No],
    dest_index=0,
    input_dimensions=[[32, 32], [32, 64], [128, 32], [128, 64]],
)
def test_pack_dest_bank(
    formats,
    dest_acc,
    l1_acc,
    num_faces,
    tilize,
    dest_index,
    input_dimensions,
    workers_tensix_coordinates,
):

    src_A, tile_cnt_A, src_B, tile_cnt_B = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=input_dimensions,
        stimuli_format_B=formats.input_format,
        input_dimensions_B=input_dimensions,
    )

    torch_format = format_dict[formats.output_format]
    golden_tensor = src_A.to(torch_format)

    unpack_to_dest = (
        False
        if tilize == Tilize.Yes and formats.input_format == DataFormat.Float32
        else formats.input_format.is_32_bit() and dest_acc == DestAccumulation.Yes
    )

    configuration = TestConfig(
        "sources/pack_dest_bank_test.cpp",
        formats,
        templates=[
            generate_input_dim(input_dimensions, input_dimensions),
            TILIZE(tilize),
        ],
        runtimes=[
            DEST_INDEX(dest_index),
            TILE_COUNT(tile_cnt_A),
            NUM_FACES(num_faces),
            L1_ACC(l1_acc),
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
        ),
        dest_acc=dest_acc,
        l1_acc=l1_acc,
        unpack_to_dest=unpack_to_dest,
    )

    res_from_L1 = configuration.run(workers_tensix_coordinates).result

    assert len(res_from_L1) == len(golden_tensor)

    res_tensor = torch.tensor(res_from_L1, dtype=torch_format)

    assert passed_test(golden_tensor, res_tensor, formats.output_format)
