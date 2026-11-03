# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
from helpers.chip_architecture import ChipArchitecture, get_chip_architecture
from helpers.format_config import DataFormat
from helpers.llk_params import PerfRunType
from helpers.param_config import input_output_formats, parametrize
from test_unpack_tilize import (
    UNPACK_TILIZE_NUM_FACES,
    unpack_tilize,
    unpack_tilize_input_dimensions,
)

_PERF_RUN_TYPES = [
    PerfRunType.L1_TO_L1,
    PerfRunType.UNPACK_ISOLATE,
    PerfRunType.PACK_ISOLATE,
    PerfRunType.L1_CONGESTION,
]
PERF_LOOP_FACTOR = 256


@pytest.mark.perf
@parametrize(
    formats=input_output_formats(
        [
            DataFormat.Float16_b,
            DataFormat.Float32,
            DataFormat.Bfp8_b,
            DataFormat.Fp8_e4m3,
        ]
    ),
    num_faces=UNPACK_TILIZE_NUM_FACES,
    rt_dim=[1, 2, 3, 4, 5, 6, 7, 8],
    ct_dim=[1, 2, 3, 4, 5, 6, 7, 8],
    run_types=[_PERF_RUN_TYPES],
    loop_factor=[PERF_LOOP_FACTOR],
    is_perf=[True],
)
def test_perf_unpack_tilize_float(
    perf_report,
    formats,
    num_faces,
    rt_dim,
    ct_dim,
    run_types,
    loop_factor,
    is_perf,
):
    if (
        formats.input_format == DataFormat.Fp8_e4m3
        or formats.output_format == DataFormat.Fp8_e4m3
    ) and get_chip_architecture() != ChipArchitecture.BLACKHOLE:
        pytest.skip(
            "Unpack Tilize does not support Fp8_e4m3 format on non-BLACKHOLE architectures"
        )

    if formats.input_format == DataFormat.Bfp8_b:
        pytest.skip("Bfp8_b input not supported for unpack_tilize")

    if formats.output_format == DataFormat.Bfp8_b and num_faces != 4:
        pytest.skip("Bfp8_b output format only works with num_faces=4")

    unpack_tilize(
        formats,
        num_faces=num_faces,
        input_dimensions=unpack_tilize_input_dimensions(rt_dim, ct_dim),
        run_types=run_types,
        loop_factor=loop_factor,
        is_perf=is_perf,
        perf_report=perf_report,
    )


@pytest.mark.perf
@parametrize(
    formats=input_output_formats([DataFormat.Int32]),
    num_faces=UNPACK_TILIZE_NUM_FACES,
    rt_dim=[1, 2],
    ct_dim=[1, 2],
    run_types=[_PERF_RUN_TYPES],
    loop_factor=[PERF_LOOP_FACTOR],
    is_perf=[True],
)
def test_perf_unpack_tilize_int(
    perf_report,
    formats,
    num_faces,
    rt_dim,
    ct_dim,
    run_types,
    loop_factor,
    is_perf,
):
    unpack_tilize(
        formats,
        unpack_to_dest=True,
        num_faces=num_faces,
        input_dimensions=unpack_tilize_input_dimensions(rt_dim, ct_dim),
        run_types=run_types,
        loop_factor=loop_factor,
        is_perf=is_perf,
        perf_report=perf_report,
    )
