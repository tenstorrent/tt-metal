# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

from tt_metal.tools.profiler.process_model_log import post_process_ops_log, run_device_profiler
from models.utility_functions import skip_for_blackhole


@pytest.fixture(scope="class")
def run_test(request):
    assert "command" in request.param, "Bad test setup, command not found in test setup dict"
    assert "name" in request.param, "Bad test setup, name not found in test setup dict"
    run_device_profiler(request.param["command"], request.param["name"])
    return request.param


@pytest.fixture(scope="class")
def do_postproc(request, run_test):
    columns = post_process_ops_log(run_test["name"])
    return columns, run_test


@pytest.fixture(scope="class", autouse=True)
def run_test_do_post_proc(request, do_postproc):
    return do_postproc


def verify_equal(received, expected, column):
    ret = None
    if expected != received:
        ret = f"Bad column value on perf report, expected {column} to be {expected} but received {received}"
    return ret


def verify_columns(received_columns, expected_columns, verify_func):
    failures = []
    for column, limit in expected_columns.items():
        assert column in received_columns, f"Bad test results: column {column} does not exist in op perf report csv"
        verification_res = verify_func(received_columns[column], limit, column)
        if verification_res is not None:
            failures.append(verification_res)
    assert len(failures) == 0, "\n" + "\n".join(failures)


matmul_test = {
    "name": "Matmul",
    "command": "pytest tests/tt_eager/python_api_testing/sweep_tests/pytests/tt_dnn/test_matmul.py::test_run_matmul_test[BFLOAT16-input_shapes0]",
}


@skip_for_blackhole()
@pytest.mark.parametrize("run_test", [pytest.param(matmul_test, id=matmul_test["name"])], indirect=True)
class TestSingleOp:
    def get_first_op_columns(cls, columns):
        firstOpIndex = 0
        return {column: columns[column][firstOpIndex] for column in columns}

    def test_core_count(self, run_test_do_post_proc):
        res, request = run_test_do_post_proc
        received_columns = self.get_first_op_columns(res)
        expected_columns = {"CORE COUNT": 1}
        verify_columns(received_columns, expected_columns, verify_equal)

    def test_performance_models(self, run_test_do_post_proc):
        res, request = run_test_do_post_proc
        received_columns = self.get_first_op_columns(res)
        expected_columns = {
            "PM IDEAL [ns]": 7,
            "PM COMPUTE [ns]": 1,
            "PM BANDWIDTH [ns]": 7,
            "PM REQ I BW": "[292.5714416503906; 292.5714416503906]",
            "PM REQ O BW": "[292.5714416503906]",
        }
        verify_columns(received_columns, expected_columns, verify_equal)
