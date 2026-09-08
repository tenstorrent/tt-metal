# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
from helpers.format_config import DataFormat, is_dest_acc_needed
from helpers.llk_params import (
    DestAccumulation,
    DestSync,
    MathFidelity,
    PerfRunType,
    Transpose,
)
from helpers.matmul_sweep import (
    generate_matmul_dimension_combinations,
    generate_tile_dims,
)
from helpers.param_config import input_output_formats, parametrize
from helpers.perf.core import PerfConfig
from helpers.stimuli_config import StimuliConfig
from helpers.test_variant_parameters import (
    CRK_TILE_DIMM,
    DEST_SYNC,
    LOOP_FACTOR,
    MATH_FIDELITY,
    NUM_FACES,
    THROTTLE_LEVEL,
    TILE_COUNT,
    UNPACK_TRANS_FACES,
)

# Important K dimensions to test
KT_DIMS = [1, 2, 3, 4, 8, 32]


def _matmul_dest_bank_tiles(formats, dest_acc):
    if is_dest_acc_needed(formats) or dest_acc == DestAccumulation.Yes:
        return 4
    return 8


@pytest.mark.perf
@parametrize(
    formats=input_output_formats(
        [
            DataFormat.Float16_b,
            DataFormat.Float16,
            DataFormat.Float32,
            DataFormat.Bfp8_b,
        ]
    ),
    dest_acc=[DestAccumulation.No, DestAccumulation.Yes],
    dest_sync=[DestSync.Half],
    unpack_to_dest=[False],
    dimensions=lambda formats, dest_acc: generate_matmul_dimension_combinations(
        _matmul_dest_bank_tiles(formats, dest_acc), kt_dims=KT_DIMS
    ),
    math_fidelity=[
        MathFidelity.LoFi,
        MathFidelity.HiFi2,
        MathFidelity.HiFi3,
        MathFidelity.HiFi4,
    ],
)
def test_perf_matmul(
    perf_report,
    formats,
    dest_acc,
    dest_sync,
    unpack_to_dest,
    dimensions,
    math_fidelity,
):
    if is_dest_acc_needed(formats) and dest_acc == DestAccumulation.No:
        pytest.skip("Dest accumulation must be enabled for this format")

    run_types = [
        PerfRunType.L1_TO_L1,
        PerfRunType.UNPACK_ISOLATE,
        PerfRunType.MATH_ISOLATE,
        PerfRunType.PACK_ISOLATE,
        PerfRunType.L1_CONGESTION,
    ]

    dims = generate_tile_dims(dimensions)

    variant_tile_count = dims.rt_dim * dims.ct_dim * dims.kt_dim

    configuration = PerfConfig(
        "sources/matmul_perf.cpp",
        formats,
        run_types,
        templates=[
            MATH_FIDELITY(math_fidelity),
            DEST_SYNC(dest_sync),
            THROTTLE_LEVEL(),
        ],
        runtimes=[
            UNPACK_TRANS_FACES(Transpose.No),
            NUM_FACES(),
            LOOP_FACTOR(64),
            TILE_COUNT(variant_tile_count),
            CRK_TILE_DIMM(dims.ct_dim, dims.rt_dim, dims.kt_dim),
        ],
        variant_stimuli=StimuliConfig(
            None,
            formats.input_format,
            None,
            formats.input_format,
            formats.output_format,
            tile_count_A=variant_tile_count,
            tile_count_B=variant_tile_count,
            tile_count_res=variant_tile_count,
        ),
        dest_acc=dest_acc,
        unpack_to_dest=unpack_to_dest,
    )

    configuration.run(perf_report)
