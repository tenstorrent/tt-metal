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
    input_output_formats,
    parametrize,
)
from helpers.perf import (
    ALL_RUN_TYPES,
    PerfReport,
    delete_benchmark_dir,
    dump_report,
    dump_scatter,
    perf_benchmark,
    update_report,
)

TEST_NAME = "eltwise_binary_fpu_perf"

report = PerfReport()


@pytest.fixture(scope="module")
def report_fixture():
    delete_benchmark_dir(TEST_NAME)
    yield
    dump_report(TEST_NAME, report)
    dump_scatter(TEST_NAME, report)


@pytest.mark.perf
@parametrize(
    test_name=TEST_NAME,
    formats=input_output_formats(
        [DataFormat.Bfp8_b, DataFormat.Float16, DataFormat.Float16_b]
    ),
    mathop=[MathOperation.Elwadd, MathOperation.Elwsub, MathOperation.Elwmul],
    tile_count=16,
    math_fidelity=[
        MathFidelity.LoFi,
        MathFidelity.HiFi2,
        MathFidelity.HiFi3,
        MathFidelity.HiFi4,
    ],
    dest_acc=[DestAccumulation.No, DestAccumulation.Yes],
)
def test_perf_eltwise_binary_fpu(
    report_fixture, test_name, formats, mathop, tile_count, math_fidelity, dest_acc
):

    # MathFidelity is only used for Elwmul
    if mathop != MathOperation.Elwmul and math_fidelity != MathFidelity.LoFi:
        pytest.skip("Fidelity does not affect Elwadd and Elwsub operations")

    test_config = {
        "testname": test_name,
        "mathop": mathop,
        "formats": formats,
        "math_fidelity": math_fidelity,
        "tile_cnt": tile_count,
        "dest_acc": dest_acc,
    }

    results = perf_benchmark(test_config, ALL_RUN_TYPES)
    update_report(report, test_config, results)
