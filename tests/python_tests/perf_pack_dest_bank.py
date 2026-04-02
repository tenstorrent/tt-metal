# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from conftest import skip_for_wormhole
from helpers.chip_architecture import ChipArchitecture, get_chip_architecture
from helpers.format_config import DataFormat
from helpers.llk_params import DestAccumulation, L1Accumulation, PerfRunType, Tilize
from helpers.param_config import (
    input_output_formats,
    parametrize,
)
from helpers.perf import PerfConfig
from helpers.stimuli_config import StimuliConfig
from helpers.test_variant_parameters import (
    DEST_INDEX,
    L1_ACC,
    LOOP_FACTOR,
    NUM_FACES,
    TILE_COUNT,
    TILIZE,
)


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
@pytest.mark.perf
@parametrize(
    formats=input_output_formats(
        [
            DataFormat.Float16_b,
            # DataFormat.Float16,
            # DataFormat.Float32,
            # DataFormat.Bfp8_b,  # Can be enabled if needed
        ]
    ),
    dest_acc=[DestAccumulation.No],
    l1_acc=[L1Accumulation.No, L1Accumulation.Yes],
    num_faces=4,
    tilize=[Tilize.No],
    dest_index=0,
    loop_factor=[1, 16, 64],
)
def test_perf_pack_dest_bank(
    perf_report,
    formats,
    dest_acc,
    l1_acc,
    num_faces,
    tilize,
    dest_index,
    loop_factor,
    workers_tensix_coordinates,
):
    # Test packing 8 tiles
    tile_cnt = 8

    src_A = torch.ones(tile_cnt * 1024, dtype=torch.bfloat16)

    # src_B is not used in this test but needed for StimuliConfig
    src_B = torch.zeros_like(src_A)
    tile_cnt_B = tile_cnt

    unpack_to_dest = (
        False
        if tilize == Tilize.Yes and formats.input_format == DataFormat.Float32
        else formats.input_format.is_32_bit() and dest_acc == DestAccumulation.Yes
    )

    configuration = PerfConfig(
        "sources/pack_dest_bank_perf.cpp",
        formats,
        run_types=[
            PerfRunType.PACK_ISOLATE,
        ],
        templates=[
            TILIZE(tilize),
            LOOP_FACTOR(loop_factor),
        ],
        runtimes=[
            DEST_INDEX(dest_index),
            TILE_COUNT(tile_cnt),
            NUM_FACES(num_faces),
            L1_ACC(l1_acc),
        ],
        variant_stimuli=StimuliConfig(
            src_A,
            formats.input_format,
            src_B,
            formats.input_format,
            formats.output_format,
            tile_count_A=tile_cnt,
            tile_count_B=tile_cnt_B,
            tile_count_res=tile_cnt,
            num_faces=num_faces,
        ),
        dest_acc=dest_acc,
        l1_acc=l1_acc,
        unpack_to_dest=unpack_to_dest,
    )

    configuration.run(perf_report, location=workers_tensix_coordinates)
