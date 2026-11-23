# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from typing import List

import pytest
from helpers.format_config import DataFormat
from helpers.golden_generators import DataCopyGolden, get_golden_generator
from helpers.llk_params import DestAccumulation, DestSyncMechanism, PerfRunType
from helpers.param_config import (
    flat_axes,
    input_output_formats,
    max_dest_tiles,
    parametrize,
    quasar_dest_sync_modes,
    runtime,
)
from helpers.perf.core import create_test_or_perf_config
from helpers.stimuli_config import StimuliConfig
from helpers.stimuli_generator import generate_stimuli
from helpers.test_variant_parameters import (
    DEST_SYNC,
    DEST_SYNC_MECHANISM,
    LOOP_FACTOR,
    NUM_BLOCKS,
    NUM_FACES,
    NUM_FACES_C_DIM,
    NUM_FACES_R_DIM,
    NUM_TILES_IN_BLOCK,
    TEST_FACE_DIMS,
    TILE_COUNT,
    generate_input_dim,
)
from helpers.tile_shape import construct_tile_shape
from helpers.utils import passed_test, result_tensor

DEST_SYNC_MECHANISM_FORMATS = input_output_formats([DataFormat.Float32], same=True)
TILE_DIMENSIONS = (32, 32)
NUM_SECTIONS = 8


def section_tile_counts(dest_sync) -> List[int]:
    full = max_dest_tiles(DestAccumulation.Yes, dest_sync)
    return sorted({1, 2, full})


def generate_dest_sync_mechanism_combinations(*, is_perf=False):
    combinations = []
    for fmt in DEST_SYNC_MECHANISM_FORMATS:
        for dest_sync in quasar_dest_sync_modes(is_perf=is_perf):
            for mechanism in DestSyncMechanism:
                for tiles_per_section in section_tile_counts(dest_sync):
                    combinations.append(
                        (
                            fmt,
                            DestAccumulation.Yes,
                            dest_sync,
                            mechanism,
                            runtime(tiles_per_section),
                        )
                    )
    return combinations


DEST_SYNC_MECHANISM_FLAT_NAMES = (
    "formats",
    "dest_acc",
    "dest_sync",
    "dest_sync_mechanism",
    "tiles_per_section",
)
ALL_DEST_SYNC_MECHANISM_COMBINATIONS = generate_dest_sync_mechanism_combinations()
PERF_DEST_SYNC_MECHANISM_COMBINATIONS = generate_dest_sync_mechanism_combinations(
    is_perf=True
)

DEST_SYNC_MECHANISM_PERF_RUN_TYPES = [
    [
        PerfRunType.L1_TO_L1,
        PerfRunType.UNPACK_ISOLATE,
        PerfRunType.PACK_ISOLATE,
    ]
]


@pytest.mark.quasar
@parametrize(
    **flat_axes(DEST_SYNC_MECHANISM_FLAT_NAMES, ALL_DEST_SYNC_MECHANISM_COMBINATIONS),
    run_types=[[PerfRunType.L1_TO_L1]],
    loop_factor=[1],
)
def test_dest_sync_mechanism_quasar(
    formats,
    dest_acc,
    dest_sync,
    dest_sync_mechanism,
    tiles_per_section,
    run_types,
    loop_factor,
    *,
    is_perf=False,
    perf_report=None,
):
    if is_perf and perf_report is None:
        raise ValueError("perf_report must be provided when is_perf=True")

    tile_shape = construct_tile_shape(TILE_DIMENSIONS)
    num_faces = tile_shape.total_num_faces()
    input_dimensions = [TILE_DIMENSIONS[0] * NUM_SECTIONS * tiles_per_section, 32]

    src_A, tile_cnt_A, src_B, _ = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=input_dimensions,
        stimuli_format_B=formats.input_format,
        input_dimensions_B=input_dimensions,
        tile_dimensions=TILE_DIMENSIONS,
    )

    generate_golden = get_golden_generator(DataCopyGolden)
    golden_tensor = generate_golden(
        src_A,
        formats.output_format,
        num_faces=num_faces,
        face_r_dim=tile_shape.face_r_dim,
        input_dimensions=input_dimensions,
        input_format=formats.input_format,
        tile_shape=tile_shape,
    )

    test_config_kwargs = {
        "test_name": "sources/quasar/dest_sync_mechanism_quasar_test.cpp",
        "formats": formats,
        "templates": [
            DEST_SYNC(dest_sync),
            DEST_SYNC_MECHANISM(dest_sync_mechanism),
        ],
        "runtimes": [
            TILE_COUNT(tile_cnt_A),
            NUM_FACES(num_faces),
            TEST_FACE_DIMS(tile_shape.face_r_dim),
            NUM_FACES_R_DIM(tile_shape.num_faces_r_dim),
            NUM_FACES_C_DIM(tile_shape.num_faces_c_dim),
            NUM_BLOCKS(NUM_SECTIONS),
            NUM_TILES_IN_BLOCK(tiles_per_section),
            LOOP_FACTOR(loop_factor),
            generate_input_dim(
                input_dimensions, input_dimensions, tile_dimensions=TILE_DIMENSIONS
            ),
        ],
        "variant_stimuli": StimuliConfig(
            src_A,
            formats.input_format,
            src_B,
            formats.input_format,
            formats.output_format,
            tile_count_A=tile_cnt_A,
            tile_count_B=tile_cnt_A,
            tile_count_res=tile_cnt_A,
            num_faces=num_faces,
            face_r_dim=tile_shape.face_r_dim,
            tile_dimensions=TILE_DIMENSIONS,
            use_dense_tile_dimensions=True,
        ),
        "unpack_to_dest": True,
        "dest_acc": dest_acc,
    }

    configuration = create_test_or_perf_config(
        is_perf=is_perf,
        run_types=run_types,
        test_config_kwargs=test_config_kwargs,
    )
    if is_perf:
        configuration.run(perf_report)
        return

    res_from_L1 = configuration.run().result
    res_tensor = result_tensor(res_from_L1, formats.output_format, golden_tensor)

    assert passed_test(
        golden_tensor, res_tensor, formats.output_format
    ), "Assert against golden failed"
