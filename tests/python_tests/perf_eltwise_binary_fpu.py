# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest

from helpers.format_arg_mapping import (
    DestAccumulation,
    MathFidelity,
    MathOperation,
)
from helpers.format_config import DataFormat
from helpers.param_config import (
    clean_params,
    generate_param_ids,
    generate_params,
    input_output_formats,
)
from helpers.perf import ALL_RUN_TYPES, perf_benchmark, write_to_report

# SUPPORTED FORMATS FOR TEST
supported_formats = [DataFormat.Bfp8_b, DataFormat.Float16, DataFormat.Float16_b]

test_formats = input_output_formats(supported_formats)
all_params = generate_params(
    ["eltwise_binary_fpu_perf"],
    test_formats,
    dest_acc=[DestAccumulation.No, DestAccumulation.Yes],
    mathop=[MathOperation.Elwadd, MathOperation.Elwsub, MathOperation.Elwmul],
    math_fidelity=[
        MathFidelity.LoFi,
        MathFidelity.HiFi2,
        MathFidelity.HiFi3,
        MathFidelity.HiFi4,
    ],
)
param_ids = generate_param_ids(all_params)


@pytest.mark.perf
@pytest.mark.parametrize(
    "testname, formats, dest_acc, mathop, math_fidelity",
    clean_params(all_params),
    ids=param_ids,
)
def test_perf_eltwise_binary_fpu(testname, formats, dest_acc, mathop, math_fidelity):

    # MathFidelity is only used for Elwmul
    if mathop != MathOperation.Elwmul and math_fidelity != MathFidelity.LoFi:
        pytest.skip("Fidelity does not affect Elwadd and Elwsub operations")

    test_config = {
        "testname": testname,
        "tile_cnt": 16,
        "formats": formats,
        "dest_acc": dest_acc,
        "mathop": mathop,
        "math_fidelity": math_fidelity,
    }

    results = perf_benchmark(test_config, ALL_RUN_TYPES)
    write_to_report(test_config, ALL_RUN_TYPES, results)
