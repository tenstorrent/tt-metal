# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
from conftest import skip_for_blackhole
from helpers.format_config import DataFormat
from helpers.llk_params import (
    DestAccumulation,
    MathFidelity,
    MathOperation,
    PerfRunType,
)
from helpers.param_config import input_output_formats, parametrize
from helpers.perf import PerfConfig
from helpers.stimuli_config import StimuliConfig
from helpers.test_variant_parameters import (
    DEST_SYNC,
    MATH_FIDELITY,
    MATH_OP,
    SRCA_REUSE_COUNT,
    TILE_COUNT,
)


@skip_for_blackhole
@pytest.mark.perf
@parametrize(
    formats=input_output_formats([DataFormat.Float16_b]),
    mathop=[MathOperation.Elwsub, MathOperation.Elwadd, MathOperation.Elwmul],
    dest_acc=[DestAccumulation.No],
    srca_reuse_count=[2, 4, 8],
    math_fidelity=[
        MathFidelity.LoFi,
    ],
    input_dimensions=[
        [128, 32],
        [32, 128],
        [64, 128],
    ],
)
def test_perf_col_tile_sdpa(
    perf_report,
    formats,
    mathop,
    dest_acc,
    math_fidelity,
    input_dimensions,
    srca_reuse_count,
    workers_tensix_coordinates,
):

    # MathFidelity is only used for Elwmul
    if mathop != MathOperation.Elwmul and math_fidelity != MathFidelity.LoFi:
        pytest.skip("Fidelity does not affect Elwadd and Elwsub operations")

    tile_count = input_dimensions[0] * input_dimensions[1] // 1024

    configuration = PerfConfig(
        "sources/unpack_a_bcast_eltwise_perf.cpp",
        formats,
        # For now only L1_TO_L1 and PACK_ISOLATE are supported because of custom usage of dvalid signals
        run_types=[PerfRunType.L1_TO_L1, PerfRunType.PACK_ISOLATE],
        templates=[
            MATH_FIDELITY(math_fidelity),
            MATH_OP(mathop=mathop),
            DEST_SYNC(),
        ],
        runtimes=[TILE_COUNT(tile_count), SRCA_REUSE_COUNT(srca_reuse_count)],
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
        dest_acc=dest_acc,
    )

    configuration.run(
        perf_report,
        run_count=10,
        location=workers_tensix_coordinates,
    )
