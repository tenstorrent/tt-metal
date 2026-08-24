# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
from helpers.chip_architecture import ChipArchitecture, get_chip_architecture
from helpers.format_config import DataFormat
from helpers.golden_generators import FACES_PER_TILE, TILE_DIM
from helpers.llk_params import DestAccumulation, DestSync, PerfRunType
from helpers.param_config import (
    generate_perf_input_dimensions,
    input_output_formats,
    parametrize,
)
from helpers.perf.core import PerfConfig
from helpers.stimuli_config import StimuliConfig
from helpers.test_variant_parameters import (
    LOOP_FACTOR,
    NUM_FACES,
    TILE_COUNT,
    generate_input_dim,
)


def _unpack_tilize_rt_ct(dest_acc):
    return [
        (dims[0] // TILE_DIM, dims[1] // TILE_DIM)
        for dims in generate_perf_input_dimensions(dest_acc, DestSync.Half)
    ]


@pytest.mark.perf
@parametrize(
    formats=input_output_formats(
        [
            DataFormat.Float16_b,
            DataFormat.Float16,
            DataFormat.Float32,
            DataFormat.Bfp8_b,
            DataFormat.Fp8_e4m3,
        ]
    ),
    rt_ct=_unpack_tilize_rt_ct(DestAccumulation.No),
    num_faces=[2, 4],
)
def test_perf_unpack_tilize_float(
    perf_report,
    formats,
    rt_ct,
    num_faces,
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

    if formats.output_format == DataFormat.Bfp8_b and num_faces != FACES_PER_TILE:
        pytest.skip("Bfp8_b output format only works with num_faces=4")

    rt_dim, ct_dim = rt_ct
    _perf_unpack_tilize(
        perf_report,
        formats,
        rt_dim,
        ct_dim,
        num_faces,
    )


@pytest.mark.perf
@parametrize(
    formats=input_output_formats([DataFormat.Int32]),
    rt_ct=_unpack_tilize_rt_ct(DestAccumulation.Yes),
    num_faces=[2, 4],
)
def test_perf_unpack_tilize_int(
    perf_report,
    formats,
    rt_ct,
    num_faces,
):
    rt_dim, ct_dim = rt_ct
    _perf_unpack_tilize(
        perf_report,
        formats,
        rt_dim,
        ct_dim,
        num_faces,
        dest_acc=DestAccumulation.Yes,
    )


def _perf_unpack_tilize(
    perf_report,
    formats,
    rt_dim,
    ct_dim,
    num_faces,
    dest_acc=DestAccumulation.No,
):
    tile_count = rt_dim * ct_dim
    dimensions = [rt_dim * TILE_DIM, ct_dim * TILE_DIM]

    configuration = PerfConfig(
        "sources/unpack_tilize_perf.cpp",
        formats,
        run_types=[
            PerfRunType.L1_TO_L1,
            PerfRunType.UNPACK_ISOLATE,
            PerfRunType.PACK_ISOLATE,
            PerfRunType.L1_CONGESTION,
        ],
        templates=[],
        runtimes=[
            generate_input_dim(dimensions, dimensions),
            TILE_COUNT(tile_count),
            LOOP_FACTOR(256),
            NUM_FACES(num_faces),
        ],
        variant_stimuli=StimuliConfig(
            None,
            formats.input_format,
            None,
            formats.input_format,
            formats.output_format,
            tile_count_A=tile_count,
            tile_count_B=tile_count,
            tile_count_res=tile_count,
            num_faces=num_faces,
        ),
        unpack_to_dest=formats.input_format == DataFormat.Int32,
        dest_acc=dest_acc,
    )

    configuration.run(perf_report)
