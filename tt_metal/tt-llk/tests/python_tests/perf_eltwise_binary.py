# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
from helpers.constraints import (
    get_valid_dest_accumulation_modes,
    get_valid_math_fidelities,
)
from helpers.format_config import DataFormat
from helpers.llk_params import (
    BroadcastType,
    MathFidelity,
    MathOperation,
    PerfRunType,
    Transpose,
)
from helpers.param_config import input_output_formats, parametrize
from helpers.perf import PerfConfig
from helpers.stimuli_config import StimuliConfig
from helpers.test_variant_parameters import (
    BROADCAST_TYPE,
    MATH_FIDELITY,
    MATH_OP,
    TILE_COUNT,
)


@pytest.mark.perf
@parametrize(
    cpp_source=["sources/eltwise_binary_fpu_perf.cpp"],
    formats=input_output_formats(
        [DataFormat.Bfp8_b, DataFormat.Float16, DataFormat.Float16_b]
    ),
    math_op=[MathOperation.Elwadd, MathOperation.Elwsub, MathOperation.Elwmul],
    broadcast_type=[BroadcastType.None_],
    transpose_srca=[Transpose.No],
    input_dimensions=[[512, 32]],
    tile_dimensions=[[32, 32]],
    math_fidelity=lambda formats, math_op: get_valid_math_fidelities(
        formats, math_op, PERF_RUN=True
    ),
    dest_acc=lambda formats: get_valid_dest_accumulation_modes(formats),
)
def test_perf_eltwise_binary(
    perf_report,
    cpp_source,
    formats,
    math_op,
    broadcast_type,
    transpose_srca,
    input_dimensions,
    tile_dimensions,
    math_fidelity,
    dest_acc,
):
    if math_op != MathOperation.Elwmul and math_fidelity != MathFidelity.LoFi:
        pytest.skip("Fidelity does not affect Elwadd and Elwsub operations")

    tile_rows, tile_cols = tile_dimensions
    tile_count = (input_dimensions[0] // tile_rows) * (input_dimensions[1] // tile_cols)

    configuration = PerfConfig(
        cpp_source,
        formats,
        run_types=[
            PerfRunType.L1_TO_L1,
            PerfRunType.UNPACK_ISOLATE,
            PerfRunType.MATH_ISOLATE,
            PerfRunType.PACK_ISOLATE,
            PerfRunType.L1_CONGESTION,
        ],
        templates=[
            BROADCAST_TYPE(broadcast_type),
            MATH_FIDELITY(math_fidelity),
            MATH_OP(mathop=math_op),
        ],
        runtimes=[TILE_COUNT(tile_count)],
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

    configuration.run(perf_report)
