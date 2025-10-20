# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
from conftest import skip_for_blackhole
from helpers.format_config import DataFormat
from helpers.llk_params import DestAccumulation, MathFidelity, MathOperation
from helpers.param_config import input_output_formats, parametrize
from helpers.perf import PerfRunType, perf_benchmark, update_report


@skip_for_blackhole
@pytest.mark.perf
@parametrize(
    test_name="unpack_a_bcast_eltwise_perf",
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
    test_name,
    formats,
    mathop,
    dest_acc,
    math_fidelity,
    input_dimensions,
    srca_reuse_count,
):

    # MathFidelity is only used for Elwmul
    if mathop != MathOperation.Elwmul and math_fidelity != MathFidelity.LoFi:
        pytest.skip("Fidelity does not affect Elwadd and Elwsub operations")

    tile_cnt = input_dimensions[0] * input_dimensions[1] // 1024

    test_config = {
        "formats": formats,
        "testname": test_name,
        "dest_acc": dest_acc,
        "input_A_dimensions": input_dimensions,
        "input_B_dimensions": input_dimensions,
        "mathop": mathop,
        "math_fidelity": math_fidelity,
        "tile_cnt": tile_cnt,
        "srca_reuse_count": srca_reuse_count,
    }

    # For now only L1_TO_L1 and PACK_ISOLATE are supported because of custom usage of dvalid signals
    results = perf_benchmark(
        test_config, [PerfRunType.L1_TO_L1, PerfRunType.PACK_ISOLATE], 10
    )
    update_report(perf_report, test_config, results)
