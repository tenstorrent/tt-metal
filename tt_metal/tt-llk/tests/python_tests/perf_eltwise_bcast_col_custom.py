# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
from helpers.format_config import DataFormat
from helpers.llk_params import (
    BroadcastType,
    DestAccumulation,
    DestSync,
    MathFidelity,
    MathOperation,
)
from helpers.param_config import (
    get_num_blocks_and_num_tiles_in_block,
    input_output_formats,
    parametrize,
)
from helpers.perf.core import ALL_PERF_RUN_TYPES
from helpers.stimuli_config import StimuliConfig
from helpers.tile_constants import get_tile_params
from test_eltwise_bcast_col_custom import create_eltwise_bcast_col_custom_config


@pytest.mark.perf
@parametrize(
    formats=input_output_formats([DataFormat.Float16_b]),
    mathop=[MathOperation.Elwsub],
    dest_acc=[DestAccumulation.No],
    math_fidelity=[MathFidelity.LoFi],
    broadcast_type=[BroadcastType.Column],
    # Perf measures full 32x32 tiles; the functional sweep also runs 16x32 tiny tiles.
    tile_rows=[32],
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
    tile_rows,
    ct_dim,
    loop_factor,
):
    # Derived exactly as the functional test derives them, so perf emits the same geometry.
    tile_dims = [tile_rows, 32]
    face_r_dim, num_faces_r_dim, num_faces_c_dim = get_tile_params(tile_dims)
    num_blocks, num_tiles_in_block = get_num_blocks_and_num_tiles_in_block(
        DestSync.Half,
        dest_acc,
        formats,
        [tile_rows, ct_dim * 32],
        tile_dimensions=tile_dims,
    )

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
        num_faces=num_faces_r_dim * num_faces_c_dim,
        face_r_dim=face_r_dim,
        num_blocks=num_blocks,
        num_tiles_in_block=num_tiles_in_block,
        loop_factor=loop_factor,
        is_perf=True,
        run_types=ALL_PERF_RUN_TYPES,
    )

    configuration.run(perf_report)
