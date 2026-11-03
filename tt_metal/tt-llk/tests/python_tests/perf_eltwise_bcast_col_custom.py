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
from helpers.tile_constants import get_tile_params
from test_eltwise_bcast_col_custom import (
    BCAST_COL_CT_DIMS,
    BCAST_COL_TILE_ROWS,
    create_eltwise_bcast_col_custom_config,
)


@pytest.mark.perf
@parametrize(
    formats=input_output_formats([DataFormat.Float16_b]),
    mathop=[MathOperation.Elwsub],
    dest_acc=[DestAccumulation.No],
    math_fidelity=[MathFidelity.LoFi],
    broadcast_type=[BroadcastType.Column],
    tile_rows=BCAST_COL_TILE_ROWS,
    ct_dim=BCAST_COL_CT_DIMS,
    loop_factor=[64],
)
def test_perf_eltwise_bcast_col_custom(
    perf_report,
    formats,
    mathop,
    dest_acc,
    math_fidelity,
    broadcast_type,
    tile_rows,
    ct_dim,
    loop_factor,
):
    tile_dims = [tile_rows, 32]
    face_r_dim, num_faces_r_dim, num_faces_c_dim = get_tile_params(tile_dims)
    num_faces = num_faces_r_dim * num_faces_c_dim
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
            num_faces=num_faces,
            face_r_dim=face_r_dim,
            tile_dimensions=tile_dims,
            use_dense_tile_dimensions=True,
        ),
        loop_factor=loop_factor,
        is_perf=True,
        num_faces=num_faces,
        face_r_dim=face_r_dim,
        run_types=[
            PerfRunType.L1_TO_L1,
            PerfRunType.UNPACK_ISOLATE,
            PerfRunType.MATH_ISOLATE,
            PerfRunType.PACK_ISOLATE,
        ],
    )

    configuration.run(perf_report)
