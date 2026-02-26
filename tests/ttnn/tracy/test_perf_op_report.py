# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import pandas as pd

from tracy.process_model_log import (
    post_process_ops_log,
    run_device_profiler,
    get_latest_ops_log_filename,
    get_profiler_folder,
)
from models.common.utility_functions import skip_for_blackhole
from tracy.compare_ops_logs import compare_ops_logs
from tracy.common import generate_logs_folder, PROFILER_CPP_DEVICE_PERF_REPORT, PROFILER_DEFAULT_OP_SUPPORT_COUNT
import numpy


@pytest.fixture(scope="class")
def run_test(request):
    assert "command" in request.param, "Bad test setup, command not found in test setup dict"
    assert "name" in request.param, "Bad test setup, name not found in test setup dict"
    op_support_count = (
        PROFILER_DEFAULT_OP_SUPPORT_COUNT
        if "op_support_count" not in request.param
        else request.param["op_support_count"]
    )
    sum_profiling = False if "sum_profiling" not in request.param else request.param["sum_profiling"]
    capture_perf_counters_groups = request.param.get("capture_perf_counters_groups")
    run_device_profiler(
        request.param["command"],
        request.param["name"],
        capture_perf_counters_groups=capture_perf_counters_groups,
        sum_profiling=sum_profiling,
        op_support_count=op_support_count,
    )
    return request.param


@pytest.fixture(scope="class")
def do_postproc(request, run_test):
    columns = post_process_ops_log(run_test["name"])
    return columns, run_test


@pytest.fixture(scope="class")
def run_test_do_post_proc(request, do_postproc):
    return do_postproc


@pytest.fixture(scope="class")
def run_test_do_cpp_and_python_post_procs(request):
    assert "command" in request.param, "Bad test setup, command not found in test setup dict"
    assert "name" in request.param, "Bad test setup, name not found in test setup dict"
    run_device_profiler(request.param["command"], request.param["name"], python_post_process=True)
    return request


@pytest.fixture(scope="class")
def run_test_do_cpp_post_proc(request):
    assert "command" in request.param, "Bad test setup, command not found in test setup dict"
    assert "name" in request.param, "Bad test setup, name not found in test setup dict"
    assert "op_support_count" in request.param, "Bad test setup, op_support_count not found in test setup dict"
    op_support_count = request.param["op_support_count"]
    sum_profiling = "sum_profiling" in request.param and request.param["sum_profiling"] == True
    is_command_binary_exe = "is_binary_exe" in request.param and request.param["is_binary_exe"] == True
    run_device_profiler(
        request.param["command"],
        request.param["name"],
        python_post_process=False,
        sum_profiling=sum_profiling,
        op_support_count=op_support_count,
        is_command_binary_exe=is_command_binary_exe,
    )
    return request


def verify_equal(received, expected, column):
    ret = None
    if expected != received:
        ret = f"Bad column value on perf report, expected {column} to be {expected} but received {received}"
    return ret


def verify_float(received, expected, column):
    ret = None
    if type(received) != numpy.float64:
        ret = f"Bad column value on perf report, expected {column} to be a numpy.float64 but received {type(received)}"
    if numpy.isnan(received):
        ret = f"Bad column value on perf report, expected {column} to be a valid numpy.float64 but received nan"
    return ret


def verify_columns(received_columns, expected_columns, verify_func):
    failures = []
    for column, limit in expected_columns.items():
        assert column in received_columns, f"Bad test results: column {column} does not exist in op perf report csv"
        verification_res = verify_func(received_columns[column], limit, column)
        if verification_res is not None:
            failures.append(verification_res)
    assert len(failures) == 0, "\n" + "\n".join(failures)


def get_first_op_columns(columns):
    firstOpIndex = 0
    return {column: columns[column][firstOpIndex] for column in columns}


matmul_test = {
    "name": "Matmul",
    "command": "pytest tests/tt_eager/python_api_testing/sweep_tests/pytests/tt_dnn/test_matmul.py::test_run_matmul_test[BFLOAT16-input_shapes0]",
}


@skip_for_blackhole()
@pytest.mark.parametrize("run_test", [pytest.param(matmul_test, id=matmul_test["name"])], indirect=True)
class TestSingleOp:
    def test_core_count(self, run_test_do_post_proc):
        res, request = run_test_do_post_proc
        received_columns = get_first_op_columns(res)
        expected_columns = {"CORE COUNT": 1}
        verify_columns(received_columns, expected_columns, verify_equal)

    def test_performance_models(self, run_test_do_post_proc):
        res, request = run_test_do_post_proc
        received_columns = get_first_op_columns(res)
        expected_columns = {
            "PM IDEAL [ns]": 7,
            "PM COMPUTE [ns]": 1,
            "PM BANDWIDTH [ns]": 7,
            "PM REQ I BW": "[292.5714416503906; 292.5714416503906]",
            "PM REQ O BW": "[292.5714416503906]",
        }
        verify_columns(received_columns, expected_columns, verify_equal)


matmul_test_tensor_io = {
    "name": "Matmul_tensor_io",
    "command": 'pytest "tests/ttnn/unit_tests/operations/matmul/test_matmul.py::test_matmul_padding[dram_sharded-2_faces_padded-input_a_value=4.0-input_b_value=2.0]"',
}


@skip_for_blackhole()
@pytest.mark.parametrize("run_test", [pytest.param(matmul_test_tensor_io, id=matmul_test["name"])], indirect=True)
class TestTensorIO:
    def test_tensor_io(self, run_test_do_post_proc):
        res, request = run_test_do_post_proc
        received_columns = get_first_op_columns(res)
        expected_columns = {
            "INPUT_0_W_PAD[LOGICAL]": "1[1]",
            "INPUT_0_Z_PAD[LOGICAL]": "1[1]",
            "INPUT_0_Y_PAD[LOGICAL]": "32[1]",
            "INPUT_0_X_PAD[LOGICAL]": "96[65]",
            "INPUT_1_W_PAD[LOGICAL]": "1[1]",
            "INPUT_1_Z_PAD[LOGICAL]": "1[1]",
            "INPUT_1_Y_PAD[LOGICAL]": "96[65]",
            "INPUT_1_X_PAD[LOGICAL]": "32[16]",
            "OUTPUT_0_W_PAD[LOGICAL]": "1[1]",
            "OUTPUT_0_Z_PAD[LOGICAL]": "1[1]",
            "OUTPUT_0_Y_PAD[LOGICAL]": "32[1]",
            "OUTPUT_0_X_PAD[LOGICAL]": "32[16]",
        }
        verify_columns(received_columns, expected_columns, verify_equal)


cpp_post_proc_test = {
    "name": "Ops",
    "command": 'pytest "tests/ttnn/tracy/test_trace_runs.py::test_with_ops"',
}


@pytest.mark.parametrize(
    "run_test_do_cpp_and_python_post_procs",
    [pytest.param(cpp_post_proc_test, id=cpp_post_proc_test["name"])],
    indirect=True,
)
class TestCppPostProc:
    def test_cpp_post_proc(self, run_test_do_cpp_and_python_post_procs):
        request = run_test_do_cpp_and_python_post_procs
        python_ops_perf_report = get_latest_ops_log_filename(request.param["name"])
        cpp_ops_perf_report = (
            generate_logs_folder(get_profiler_folder(request.param["name"])) / PROFILER_CPP_DEVICE_PERF_REPORT
        )
        compare_ops_logs(python_ops_perf_report=python_ops_perf_report, cpp_ops_perf_report=cpp_ops_perf_report)


matmul_test_perf_counters = {
    "name": "Matmul_perf_counters",
    "command": "pytest tests/ttnn/unit_tests/operations/matmul/test_matmul.py::test_padded_2d_matmul[tile_count=1375-side=width]",
    "capture_perf_counters_groups": ["fpu"],
}


@skip_for_blackhole()
@pytest.mark.parametrize(
    "run_test",
    [pytest.param(matmul_test_perf_counters, id=matmul_test_perf_counters["name"])],
    indirect=True,
)
class TestPerfCountersSingleOp:
    def test_performance_counter_columns(self, run_test_do_post_proc):
        res, request = run_test_do_post_proc
        received_columns = get_first_op_columns(res)
        expected_columns = {
            "SFPU Util Min (%)": 0.0,
            "SFPU Util Median (%)": 0.0,
            "SFPU Util Max (%)": 0.0,
            "Avg SFPU util on full grid (%)": 0.0,
            "FPU Util Min (%)": 0.0,
            "FPU Util Median (%)": 0.0,
            "FPU Util Max (%)": 0.0,
            "Avg FPU util on full grid (%)": 0.0,
            "MATH Util Min (%)": 0.0,
            "MATH Util Median (%)": 0.0,
            "MATH Util Max (%)": 0.0,
            "Avg Math util on full grid (%)": 0.0,
        }
        # Just check presence of float columns
        verify_columns(received_columns, expected_columns, verify_float)


op_support_count_tests = [
    {
        "name": "Op_Support_Count_10",
        "command": "build/programming_examples/profiler/test_multi_op 100",
        "op_support_count": 10,
        # Number of ops we expect to detect is 43 because that is the minimum number of ops that will be reported for any program with at least 43 ops
        "expected_op_count": 43,
        "is_binary_exe": True,
    },
    {
        "name": "Op_Support_Count_100",
        "command": "build/programming_examples/profiler/test_multi_op 10000",
        "op_support_count": 100,
        "expected_op_count": 100,
        "is_binary_exe": True,
    },
    {
        "name": "Op_Support_Count_1000",
        "command": "build/programming_examples/profiler/test_multi_op 10000",
        "op_support_count": 1000,
        "expected_op_count": 1000,
        "is_binary_exe": True,
    },
    {
        "name": "Op_Support_Count_5000",
        "command": "build/programming_examples/profiler/test_multi_op 10000",
        "op_support_count": 5000,
        "expected_op_count": 5000,
        "is_binary_exe": True,
    },
    {
        "name": "Op_Support_Count_10000",
        "command": "build/programming_examples/profiler/test_multi_op 10000",
        "op_support_count": 10000,
        "expected_op_count": 10000,
        "is_binary_exe": True,
    },
]


@pytest.mark.parametrize(
    "run_test_do_cpp_post_proc", [pytest.param(test, id=test["name"]) for test in op_support_count_tests], indirect=True
)
class TestOpSupportCount:
    def test_op_support_count(self, run_test_do_cpp_post_proc):
        request = run_test_do_cpp_post_proc
        cpp_ops_perf_report = (
            generate_logs_folder(get_profiler_folder(request.param["name"])) / PROFILER_CPP_DEVICE_PERF_REPORT
        )

        df = pd.read_csv(cpp_ops_perf_report)

        # Count unique combinations of (GLOBAL CALL COUNT, METAL TRACE ID, METAL TRACE REPLAY SESSION ID)
        actual_count = df.groupby(
            ["GLOBAL CALL COUNT", "METAL TRACE ID", "METAL TRACE REPLAY SESSION ID"], dropna=False
        ).ngroups
        expected_count = request.param["expected_op_count"]

        assert (
            actual_count == expected_count
        ), f"Expected to detect {expected_count} ops, but detected {actual_count} ops"


op_support_count_with_sum_profiling_enabled_test = {
    "name": "Op_Support_Count_200_With_Sum_Profiling_Enabled",
    "command": 'pytest "tests/ttnn/tracy/test_trace_runs.py::test_with_ops_single_core[100-5]"',
    "op_support_count": 200,
    # Number of ops we expect to detect is higher than the op support count value because BRISC, NCRISC, and TRISC1 use the extra space reserved for accumulation zones to record ops instead
    "expected_op_count": 266,
    "sum_profiling": True,
}


@pytest.mark.parametrize(
    "run_test",
    [
        pytest.param(
            op_support_count_with_sum_profiling_enabled_test,
            id=op_support_count_with_sum_profiling_enabled_test["name"],
        )
    ],
    indirect=True,
)
class TestOpSupportCountWithSumProfilingEnabled:
    def test_op_support_count_with_sum_profiling_enabled(self, run_test_do_post_proc):
        res, request = run_test_do_post_proc

        # Count unique combinations of (GLOBAL CALL COUNT, METAL TRACE ID, METAL TRACE REPLAY SESSION ID)
        res = res[res["DEVICE FW DURATION [ns]"].notna()]
        actual_count = res.groupby(
            ["GLOBAL CALL COUNT", "METAL TRACE ID", "METAL TRACE REPLAY SESSION ID"], dropna=False
        ).ngroups
        expected_count = request["expected_op_count"]

        assert (
            actual_count == expected_count
        ), f"Expected to detect {expected_count} ops, but detected {actual_count} ops"

        for _, row in res.iterrows():
            assert (
                row["DEVICE COMPUTE CB WAIT FRONT [ns]"] != 0
            ), f"DEVICE COMPUTE CB WAIT FRONT [ns] is 0 for op (GLOBAL CALL COUNT={row['GLOBAL CALL COUNT']}, METAL TRACE ID={row['METAL TRACE ID']}, METAL TRACE REPLAY SESSION ID={row['METAL TRACE REPLAY SESSION ID']})"
            assert (
                row["DEVICE COMPUTE CB RESERVE BACK [ns]"] != 0
            ), f"DEVICE COMPUTE CB RESERVE BACK [ns] is 0 for op (GLOBAL CALL COUNT={row['GLOBAL CALL COUNT']}, METAL TRACE ID={row['METAL TRACE ID']}, METAL TRACE REPLAY SESSION ID={row['METAL TRACE REPLAY SESSION ID']})"
