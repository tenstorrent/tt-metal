# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
from helpers.constraints import (
    get_valid_dest_accumulation_modes,
    get_valid_math_fidelities,
)
from helpers.format_config import DataFormat
from helpers.llk_params import MathOperation
from helpers.param_config import input_output_formats, parametrize
from helpers.perf import ALL_RUN_TYPES, perf_benchmark, update_report


@pytest.mark.perf
@parametrize(
    test_name="eltwise_binary_fpu_perf",
    formats=input_output_formats(
        [DataFormat.Bfp8_b, DataFormat.Float16, DataFormat.Float16_b]
    ),
    mathop=[MathOperation.Elwadd, MathOperation.Elwsub, MathOperation.Elwmul],
    tile_count=16,
    math_fidelity=lambda formats, mathop: get_valid_math_fidelities(
        formats, mathop, PERF_RUN=True
    ),
    dest_acc=lambda formats: get_valid_dest_accumulation_modes(formats),
)
def test_perf_eltwise_binary_fpu(
    perf_report, test_name, formats, mathop, tile_count, math_fidelity, dest_acc
):

    test_config = {
        "testname": test_name,
        "mathop": mathop,
        "formats": formats,
        "math_fidelity": math_fidelity,
        "tile_cnt": tile_count,
        "dest_acc": dest_acc,
    }

    results = perf_benchmark(test_config, ALL_RUN_TYPES)
    update_report(perf_report, test_config, results)
