# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
from helpers.chip_architecture import ChipArchitecture, get_chip_architecture
from helpers.dest_params import (
    UnpackPath,
    dest_acc_modes,
    dest_sync_modes,
    unpack_to_dest_modes,
)
from helpers.format_config import DataFormat
from helpers.llk_params import PerfRunType
from helpers.param_config import input_output_formats, parametrize
from helpers.perf.core import PerfConfig
from helpers.stimuli_config import StimuliConfig
from helpers.test_variant_parameters import (
    DEST_SYNC,
    LOOP_FACTOR,
    TILE_COUNT,
    generate_input_dim,
)


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
    dest_acc=dest_acc_modes,
    dest_sync=lambda: dest_sync_modes(is_perf=True),
    unpack_to_dest=lambda formats, dest_acc: unpack_to_dest_modes(
        formats, dest_acc, path=UnpackPath.ForceFalse
    ),
    rt_dim=[1, 2, 3, 4, 5, 6, 7, 8],
    ct_dim=[1, 2, 3, 4, 5, 6, 7, 8],
)
def test_perf_unpack_tilize_float(
    perf_report,
    formats,
    dest_acc,
    dest_sync,
    unpack_to_dest,
    rt_dim,
    ct_dim,
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

    _perf_unpack_tilize(
        perf_report,
        formats,
        dest_acc,
        dest_sync,
        unpack_to_dest,
        rt_dim,
        ct_dim,
    )


@pytest.mark.perf
@parametrize(
    formats=input_output_formats([DataFormat.Int32]),
    dest_acc=dest_acc_modes,
    dest_sync=lambda: dest_sync_modes(is_perf=True),
    unpack_to_dest=lambda formats, dest_acc: unpack_to_dest_modes(
        formats, dest_acc, path=UnpackPath.Int32Dest
    ),
    rt_dim=[1, 2],
    ct_dim=[1, 2],
)
def test_perf_unpack_tilize_int(
    perf_report,
    formats,
    dest_acc,
    dest_sync,
    unpack_to_dest,
    rt_dim,
    ct_dim,
):
    _perf_unpack_tilize(
        perf_report,
        formats,
        dest_acc,
        dest_sync,
        unpack_to_dest,
        rt_dim,
        ct_dim,
    )


def _perf_unpack_tilize(
    perf_report,
    formats,
    dest_acc,
    dest_sync,
    unpack_to_dest,
    rt_dim,
    ct_dim,
):
    tile_count = rt_dim * ct_dim
    dimensions = [rt_dim * 32, ct_dim * 32]

    configuration = PerfConfig(
        "sources/unpack_tilize_perf.cpp",
        formats,
        run_types=[
            PerfRunType.L1_TO_L1,
            PerfRunType.UNPACK_ISOLATE,
            PerfRunType.PACK_ISOLATE,
            PerfRunType.L1_CONGESTION,
        ],
        templates=[
            DEST_SYNC(dest_sync),
        ],
        runtimes=[
            generate_input_dim(dimensions, dimensions),
            TILE_COUNT(tile_count),
            LOOP_FACTOR(256),
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
        ),
        unpack_to_dest=unpack_to_dest,
        dest_acc=dest_acc,
    )

    configuration.run(perf_report)
