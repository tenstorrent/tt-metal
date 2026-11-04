# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
from helpers.format_config import DataFormat
from helpers.llk_params import (
    BroadcastType,
    DestAccumulation,
    MathFidelity,
    MathOperation,
    PerfRunType,
)
from helpers.param_config import input_output_formats, parametrize
from helpers.stimuli_config import StimuliConfig
from test_eltwise_bcast_col_custom import create_eltwise_bcast_col_custom_config


@pytest.mark.perf
@parametrize(
    formats=input_output_formats([DataFormat.Float16_b]),
    mathop=[MathOperation.Elwsub],
    dest_acc=[DestAccumulation.No],
    math_fidelity=[MathFidelity.LoFi],
    broadcast_type=[BroadcastType.Column],
    ct_dim=[1, 8],
    loop_factor=[64],
)
def test_perf_eltwise_bcast_col_custom(
    perf_report,
    formats,
    mathop,
    dest_acc,
    math_fidelity,
    broadcast_type,
    ct_dim,
    loop_factor,
):
    configuration = create_eltwise_bcast_col_custom_config(
        formats,
        mathop,
        dest_acc,
        math_fidelity,
        broadcast_type,
        ct_dim,
        ct_dim,
        StimuliConfig(
            None,
            formats.input_format,
            None,
            formats.input_format,
            formats.output_format,
            tile_count_A=ct_dim,
            tile_count_B=1,
            tile_count_res=ct_dim,
        ),
        loop_factor=loop_factor,
        is_perf=True,
        run_types=[
            PerfRunType.L1_TO_L1,
            PerfRunType.UNPACK_ISOLATE,
            PerfRunType.MATH_ISOLATE,
            PerfRunType.PACK_ISOLATE,
        ],
    )

    configuration.run(perf_report)
