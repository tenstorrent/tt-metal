# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
from helpers.chip_architecture import ChipArchitecture, get_chip_architecture
from helpers.format_config import DataFormat
from helpers.llk_params import (
    PerfRunType,
)
from helpers.param_config import (
    input_output_formats,
    parametrize,
)
from helpers.perf.core import PerfConfig
from helpers.stimuli_config import StimuliConfig
from helpers.test_variant_parameters import (
    LOOP_FACTOR,
    NUM_FACES,
    PACK_UNTILIZE_NARROW_ROW,
    TILE_COUNT,
    generate_input_dim,
)
from helpers.tile_constants import DEFAULT_TILE_C_DIM, get_tile_params


def _run_perf_pack_untilize(
    perf_report,
    formats,
    full_rt_dim,
    full_ct_dim,
    tile_dimensions=(32, 32),
    narrow_row=False,
    row_num_datums=DEFAULT_TILE_C_DIM,
):
    """One pack-untilize perf variant; every variant goes through this single PerfConfig so
    the module emits one column schema (narrow_row / row_num_datums / num_faces included).
    """
    if (
        formats.input_format == DataFormat.Fp8_e4m3
        or formats.output_format == DataFormat.Fp8_e4m3
    ) and get_chip_architecture() != ChipArchitecture.BLACKHOLE:
        pytest.skip(
            "Pack Untilize does not support Fp8_e4m3 format on non-BLACKHOLE architectures"
        )

    if formats.output_format == DataFormat.Bfp8_b:
        pytest.skip("Pack Untilize does not support Bfp8_b output")

    if (formats.input_format == DataFormat.Int32) ^ (
        formats.output_format == DataFormat.Int32
    ):
        pytest.skip("Pack Untilize does not support mixing Int32 with other formats")

    max_block_dim = 4 if formats.input_format.is_32_bit() else 8

    # fixme: handle format outlier case properly
    if (
        formats.input_format == DataFormat.Float16_b
        or formats.input_format == DataFormat.Bfp8_b
    ) and formats.output_format == DataFormat.Float16:
        max_block_dim = 4

    # Find the maximum block size that divides full_ct_dim and is <= max_block_dim
    block_ct_dim = 1
    for candidate in range(min(full_ct_dim, max_block_dim), 0, -1):
        if full_ct_dim % candidate == 0:
            block_ct_dim = candidate
            break

    tile_r_dim, tile_c_dim = tile_dimensions
    _, num_faces_r_dim, num_faces_c_dim = get_tile_params(tile_dimensions)
    num_faces = num_faces_r_dim * num_faces_c_dim
    tile_count = full_rt_dim * full_ct_dim
    dimensions = [full_rt_dim * tile_r_dim, full_ct_dim * tile_c_dim]

    configuration = PerfConfig(
        "sources/pack_untilize_perf.cpp",
        formats,
        run_types=[
            PerfRunType.L1_TO_L1,
            PerfRunType.PACK_ISOLATE,
            PerfRunType.L1_CONGESTION,
        ],
        templates=[
            generate_input_dim(
                dimensions, dimensions, block_ct_dim, tile_dimensions=tile_dimensions
            ),
            PACK_UNTILIZE_NARROW_ROW(narrow_row, row_num_datums),
        ],
        runtimes=[TILE_COUNT(tile_count), LOOP_FACTOR(32), NUM_FACES(num_faces)],
        variant_stimuli=StimuliConfig(
            None,
            formats.input_format,
            None,
            formats.input_format,
            formats.output_format,
            tile_count_A=tile_count,
            tile_count_B=tile_count,
            tile_count_res=tile_count,
        ),
        unpack_to_dest=formats.input_format.is_32_bit(),
    )

    configuration.run(perf_report)


@pytest.mark.perf
@parametrize(
    formats=input_output_formats(
        [
            DataFormat.Float16_b,
            DataFormat.Float16,
            DataFormat.Float32,
            DataFormat.Int32,
            DataFormat.Bfp8_b,
            DataFormat.Fp8_e4m3,
        ]
    ),
    full_rt_dim=[1, 2, 3, 4, 5, 6, 7, 8],
    full_ct_dim=[1, 2, 3, 4, 5, 6, 7, 8],
)
def test_perf_pack_untilize(
    perf_report,
    formats,
    full_rt_dim,
    full_ct_dim,
):
    _run_perf_pack_untilize(perf_report, formats, full_rt_dim, full_ct_dim)


# Narrow rows (each tile contributes row_num_datums datums per output row): one tile per block
# (the sharded-transpose shape) up to a full 8-tile block.
@pytest.mark.perf
@parametrize(
    formats=input_output_formats([DataFormat.Float16_b]),
    full_ct_dim=[1, 2, 4, 8],
    row_num_datums=[8],
)
def test_perf_pack_untilize_narrow_row(
    perf_report,
    formats,
    full_ct_dim,
    row_num_datums,
):
    _run_perf_pack_untilize(
        perf_report,
        formats,
        1,
        full_ct_dim,
        narrow_row=True,
        row_num_datums=row_num_datums,
    )


# Single-face 16x16 tiles (num_faces == 1): one packer walks all 16 rows of each tile.
@pytest.mark.perf
@parametrize(
    formats=input_output_formats([DataFormat.Float16_b]),
    full_ct_dim=[4, 8],
)
def test_perf_pack_untilize_single_face_tile(
    perf_report,
    formats,
    full_ct_dim,
):
    _run_perf_pack_untilize(
        perf_report, formats, 1, full_ct_dim, tile_dimensions=(16, 16)
    )
