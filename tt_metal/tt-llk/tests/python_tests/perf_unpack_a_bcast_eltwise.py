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
from test_unpack_a_bcast_eltwise import run_unpack_a_bcast_eltwise

_PERF_RUN_TYPES = [PerfRunType.L1_TO_L1, PerfRunType.PACK_ISOLATE]


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
        [64, 128],
    ],
    run_types=[_PERF_RUN_TYPES],
    is_perf=[True],
)
def test_perf_col_tile_sdpa(
    perf_report,
    formats,
    mathop,
    dest_acc,
    math_fidelity,
    input_dimensions,
    srca_reuse_count,
    run_types,
    is_perf,
):

    # MathFidelity is only used for Elwmul
    if mathop != MathOperation.Elwmul and math_fidelity != MathFidelity.LoFi:
        pytest.skip("Fidelity does not affect Elwadd and Elwsub operations")

    run_unpack_a_bcast_eltwise(
        formats,
        mathop,
        dest_acc=dest_acc,
        math_fidelity=math_fidelity,
        input_dimensions=input_dimensions,
        srca_reuse_count=srca_reuse_count,
        run_types=run_types,
        is_perf=is_perf,
        perf_report=perf_report,
    )
