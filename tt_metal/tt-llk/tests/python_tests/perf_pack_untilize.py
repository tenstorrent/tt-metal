# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
from helpers.format_config import DataFormat
from helpers.llk_params import (
    DestAccumulation,
    DestSync,
    PerfRunType,
)
from helpers.param_config import (
    input_output_formats,
    parametrize,
)
from helpers.perf import PerfConfig
from helpers.stimuli_config import StimuliConfig
from helpers.test_variant_parameters import (
    LOOP_FACTOR,
    TILE_COUNT,
    generate_input_dim,
)

_PACK_UNTILIZE_INPUT_DIMENSIONS = [
    [rt * 32, ct * 32] for rt in range(1, 9) for ct in range(1, 9)
]


@pytest.mark.perf
@parametrize(
    formats=input_output_formats(
        [
            DataFormat.Float16_b,
            DataFormat.Float16,
            DataFormat.Float32,
            DataFormat.Int32,
            DataFormat.Bfp8_b,
        ]
    ),
    dest_acc=[DestAccumulation.No],
    input_dimensions=_PACK_UNTILIZE_INPUT_DIMENSIONS,
    dest_sync=[DestSync.Half],
    tile_dst_ct_offset=[0],
)
def test_perf_pack_untilize(
    perf_report,
    formats,
    dest_acc,
    input_dimensions,
    dest_sync,
    tile_dst_ct_offset,
):
    if formats.output_format == DataFormat.Bfp8_b:
        pytest.skip("Pack Untilize does not support Bfp8_b output")

    if (formats.input_format == DataFormat.Int32) ^ (
        formats.output_format == DataFormat.Int32
    ):
        pytest.skip("Pack Untilize does not support mixing Int32 with other formats")

    full_rt_dim = input_dimensions[0] // 32
    full_ct_dim = input_dimensions[1] // 32

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

    tile_count = full_rt_dim * full_ct_dim

    configuration = PerfConfig(
        "sources/pack_untilize_perf.cpp",
        formats,
        run_types=[
            PerfRunType.L1_TO_L1,
            PerfRunType.PACK_ISOLATE,
            PerfRunType.L1_CONGESTION,
        ],
        templates=[
            generate_input_dim(input_dimensions, input_dimensions, block_ct_dim)
        ],
        runtimes=[TILE_COUNT(tile_count), LOOP_FACTOR()],
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
        dest_acc=dest_acc,
    )

    configuration.run(perf_report)
