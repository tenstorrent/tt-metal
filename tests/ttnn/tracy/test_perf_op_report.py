# SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import os

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
from tracy.common import (
    generate_logs_folder,
    PROFILER_CPP_DEVICE_PERF_REPORT,
    PROFILER_DEVICE_SIDE_LOG,
    PROFILER_DEFAULT_OP_SUPPORT_COUNT,
)
from tracy import process_ops_logs
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


various_ops_test = {
    "name": "VariousOps",
    "command": "pytest tests/ttnn/tracy/test_various_ops_profile.py::test_various_ops_profile",
}

EXPECTED_DEVICE_OP_COUNT = 5
SUB_DEVICE_AVAILABLE_CORE_COUNTS = {0: 16, 1: 5}
SUB_DEVICE_0_MATMUL_CORE_COUNT = 4
SUB_DEVICE_0_BINARY_CORE_COUNT = 16
SUB_DEVICE_1_BINARY_CORE_COUNT = 5


@skip_for_blackhole()
@pytest.mark.parametrize("run_test", [pytest.param(various_ops_test, id=various_ops_test["name"])], indirect=True)
class TestVariousOpsProfile:
    def test_sub_device_id_column_present(self, run_test_do_post_proc):
        res, _request = run_test_do_post_proc
        assert "SUB DEVICE ID" in res.columns
        assert "SUB DEVICE MANAGER ID" not in res.columns

    def test_sub_device_ids_are_not_all_zero(self, run_test_do_post_proc):
        res, _request = run_test_do_post_proc
        op_rows = res[res["OP TYPE"] != "signpost"]
        sub_device_ids = pd.to_numeric(op_rows["SUB DEVICE ID"], errors="coerce").dropna()
        assert not sub_device_ids.empty, "Expected SUB DEVICE ID values in perf report"
        assert sub_device_ids.nunique() >= 2, (
            f"Expected ops on multiple sub-devices, but only saw SUB DEVICE ID values: "
            f"{sorted(sub_device_ids.unique())}"
        )

    def test_various_op_codes_reported(self, run_test_do_post_proc):
        res, _request = run_test_do_post_proc
        op_rows = res[res["OP TYPE"] != "signpost"]
        op_codes = [str(code).lower() for code in op_rows["OP CODE"].dropna() if str(code).strip()]

        assert (
            len(op_codes) >= EXPECTED_DEVICE_OP_COUNT
        ), f"Expected at least {EXPECTED_DEVICE_OP_COUNT} device ops, found {len(op_codes)}: {op_codes}"
        assert any("matmul" in code for code in op_codes), f"Expected a matmul op in OP CODE column, found: {op_codes}"
        binary_op_count = sum(1 for code in op_codes if "binary" in code)
        assert (
            binary_op_count >= 3
        ), f"Expected at least 3 binary ops (add/multiply/subtract), found {binary_op_count} in: {op_codes}"

    def test_device_durations_populated(self, run_test_do_post_proc):
        res, _request = run_test_do_post_proc
        device_rows = res[res["DEVICE FW DURATION [ns]"].notna()]
        assert len(device_rows) >= EXPECTED_DEVICE_OP_COUNT, (
            f"Expected device durations for at least {EXPECTED_DEVICE_OP_COUNT} ops, "
            f"but only found {len(device_rows)}"
        )
        positive_durations = pd.to_numeric(device_rows["DEVICE FW DURATION [ns]"], errors="coerce").fillna(0) > 0
        assert (
            positive_durations.sum() >= EXPECTED_DEVICE_OP_COUNT
        ), "Expected non-zero DEVICE FW DURATION [ns] for profiled ops"

    def test_core_counts_match_sub_device(self, run_test_do_post_proc):
        res, _request = run_test_do_post_proc
        op_rows = res[res["OP TYPE"] != "signpost"].copy()
        assert "CORE COUNT" in op_rows.columns
        assert "AVAILABLE WORKER CORE COUNT" in op_rows.columns

        op_rows["SUB DEVICE ID"] = pd.to_numeric(op_rows["SUB DEVICE ID"], errors="coerce")
        op_rows["CORE COUNT"] = pd.to_numeric(op_rows["CORE COUNT"], errors="coerce")
        op_rows["AVAILABLE WORKER CORE COUNT"] = pd.to_numeric(op_rows["AVAILABLE WORKER CORE COUNT"], errors="coerce")

        failures = []
        for _, row in op_rows.iterrows():
            sub_device_id = int(row["SUB DEVICE ID"])
            op_code = str(row["OP CODE"]).lower()
            core_count = int(row["CORE COUNT"])
            available_cores = int(row["AVAILABLE WORKER CORE COUNT"])
            expected_available = SUB_DEVICE_AVAILABLE_CORE_COUNTS[sub_device_id]

            if available_cores != expected_available:
                failures.append(
                    f"{row['OP CODE']} on sub-device {sub_device_id}: "
                    f"expected AVAILABLE WORKER CORE COUNT={expected_available}, got {available_cores}"
                )

            if "matmul" in op_code:
                expected_core_count = SUB_DEVICE_0_MATMUL_CORE_COUNT
            elif sub_device_id == 0:
                expected_core_count = SUB_DEVICE_0_BINARY_CORE_COUNT
            else:
                expected_core_count = SUB_DEVICE_1_BINARY_CORE_COUNT

            if core_count != expected_core_count:
                failures.append(
                    f"{row['OP CODE']} on sub-device {sub_device_id}: "
                    f"expected CORE COUNT={expected_core_count}, got {core_count}"
                )

            if core_count > available_cores:
                failures.append(
                    f"{row['OP CODE']} on sub-device {sub_device_id}: "
                    f"CORE COUNT ({core_count}) exceeds AVAILABLE WORKER CORE COUNT ({available_cores})"
                )

        assert not failures, "Core count mismatches:\n" + "\n".join(failures)

    def test_sub_device_id_matches_device_csv_when_present(self, run_test_do_post_proc):
        res, request = run_test_do_post_proc
        device_csv = generate_logs_folder(get_profiler_folder(request["name"])) / PROFILER_DEVICE_SIDE_LOG
        lookup = process_ops_logs.build_sub_device_id_lookup_from_device_csv(device_csv)
        if not lookup:
            pytest.skip("Device CSV has no sub_device_id metadata for this run")

        mismatches = []
        for _, row in res.iterrows():
            if str(row.get("SUB DEVICE ID", "")).strip() in ("", "nan"):
                continue

            trace_id = -1
            if pd.notna(row.get("METAL TRACE ID")) and str(row.get("METAL TRACE ID", "")).strip() != "":
                trace_id = int(row["METAL TRACE ID"])

            trace_id_counter = -1
            if (
                pd.notna(row.get("METAL TRACE REPLAY SESSION ID"))
                and str(row.get("METAL TRACE REPLAY SESSION ID", "")).strip() != ""
            ):
                trace_id_counter = int(row["METAL TRACE REPLAY SESSION ID"])

            key = (int(row["DEVICE ID"]), int(row["GLOBAL CALL COUNT"]), trace_id, trace_id_counter)
            if key not in lookup:
                continue

            if int(row["SUB DEVICE ID"]) != lookup[key]:
                mismatches.append(f"{key}: report={row['SUB DEVICE ID']} device_csv={lookup[key]}")

        assert not mismatches, "SUB DEVICE ID mismatches vs device CSV:\n" + "\n".join(mismatches[:10])


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
    "capture_perf_counters_groups": ["all"],
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
            "Unpacker0 Write Efficiency Min (%)": 0.0,
            "Unpacker0 Write Efficiency Median (%)": 0.0,
            "Unpacker0 Write Efficiency Max (%)": 0.0,
            "Unpacker0 Write Efficiency Avg (%)": 0.0,
            "Unpacker1 Write Efficiency Min (%)": 0.0,
            "Unpacker1 Write Efficiency Median (%)": 0.0,
            "Unpacker1 Write Efficiency Max (%)": 0.0,
            "Unpacker1 Write Efficiency Avg (%)": 0.0,
            "Unpacker Write Efficiency Min (%)": 0.0,
            "Unpacker Write Efficiency Median (%)": 0.0,
            "Unpacker Write Efficiency Max (%)": 0.0,
            "Unpacker Write Efficiency Avg (%)": 0.0,
            "Packer Efficiency Min (%)": 0.0,
            "Packer Efficiency Median (%)": 0.0,
            "Packer Efficiency Max (%)": 0.0,
            "Packer Efficiency Avg (%)": 0.0,
            "FPU Execution Efficiency Min (%)": 0.0,
            "FPU Execution Efficiency Median (%)": 0.0,
            "FPU Execution Efficiency Max (%)": 0.0,
            "FPU Execution Efficiency Avg (%)": 0.0,
            "Math Pipeline Utilization Min (%)": 0.0,
            "Math Pipeline Utilization Median (%)": 0.0,
            "Math Pipeline Utilization Max (%)": 0.0,
            "Math Pipeline Utilization Avg (%)": 0.0,
            "Math-to-Pack Handoff Efficiency Min (%)": 0.0,
            "Math-to-Pack Handoff Efficiency Median (%)": 0.0,
            "Math-to-Pack Handoff Efficiency Max (%)": 0.0,
            "Math-to-Pack Handoff Efficiency Avg (%)": 0.0,
            "Unpacker-to-Math Data Flow Min (%)": 0.0,
            "Unpacker-to-Math Data Flow Median (%)": 0.0,
            "Unpacker-to-Math Data Flow Max (%)": 0.0,
            "Unpacker-to-Math Data Flow Avg (%)": 0.0,
            # INSTRN_THREAD thread stall rates
            "Thread 0 Stall Rate Min (%)": 0.0,
            "Thread 0 Stall Rate Avg (%)": 0.0,
            "Thread 1 Stall Rate Min (%)": 0.0,
            "Thread 1 Stall Rate Avg (%)": 0.0,
            "Thread 2 Stall Rate Min (%)": 0.0,
            "Thread 2 Stall Rate Avg (%)": 0.0,
            # INSTRN_THREAD pipeline waits
            "SrcA Valid Wait Min (%)": 0.0,
            "SrcB Valid Wait Min (%)": 0.0,
            "Math Idle Wait T1 Min (%)": 0.0,
            "Pack Idle Wait T2 Min (%)": 0.0,
            "Unpack Idle Wait T0 Min (%)": 0.0,
            # INSTRN_THREAD semaphore waits
            "Semaphore Zero Wait T0 Min (%)": 0.0,
            "Semaphore Zero Wait T1 Min (%)": 0.0,
            "Semaphore Zero Wait T2 Min (%)": 0.0,
            # TDMA_UNPACK data hazard stalls
            "Data Hazard Stall Rate Min (%)": 0.0,
            # L1 Bank 0 metrics
            "L1 Unpacker Port Util Min (%)": 0.0,
            "L1 TDMA Bundle Util Min (%)": 0.0,
            "NOC Ring 0 Outgoing Util Min (%)": 0.0,
            "NOC Ring 0 Incoming Util Min (%)": 0.0,
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


slow_dispatch_test = {
    "name": "SlowDispatch",
    "command": "pytest tests/ttnn/tracy/test_profiler_sync.py::test_with_ops",
}


@pytest.fixture(scope="class")
def run_slow_dispatch_test(request):
    assert "command" in request.param, "Bad test setup, command not found in test setup dict"
    assert "name" in request.param, "Bad test setup, name not found in test setup dict"
    prev_value = os.environ.get("TT_METAL_SLOW_DISPATCH_MODE")
    os.environ["TT_METAL_SLOW_DISPATCH_MODE"] = "1"
    try:
        run_device_profiler(request.param["command"], request.param["name"])
    finally:
        if prev_value is None:
            os.environ.pop("TT_METAL_SLOW_DISPATCH_MODE", None)
        else:
            os.environ["TT_METAL_SLOW_DISPATCH_MODE"] = prev_value
    return request.param


@pytest.mark.parametrize(
    "run_slow_dispatch_test",
    [pytest.param(slow_dispatch_test, id=slow_dispatch_test["name"])],
    indirect=True,
)
class TestSlowDispatch:
    def test_slow_dispatch_profiling(self, run_slow_dispatch_test):
        name = run_slow_dispatch_test["name"]
        filename = get_latest_ops_log_filename(name)
        df = pd.read_csv(filename)
        assert len(df) > 0, "Expected at least one op in the slow dispatch profiler output"


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
