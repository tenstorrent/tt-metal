# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import os
import copy
import pathlib
import sys
import subprocess as sp
from enum import Enum, auto
from functools import partial, wraps
from collections import namedtuple
from operator import ne, truth

from loguru import logger

from models.utility_functions import is_wormhole_b0, is_grayskull, is_blackhole


class TestSuiteType(Enum):
    BUILD_KERNELS_FOR_RISCV = auto()
    LLRT = auto()
    TT_METAL = auto()
    PROGRAMMING_EXAMPLE = auto()
    TT_EAGER = auto()
    UNKNOWN = auto()


TestEntry = namedtuple("TestEntry", ["test_name", "executable_name", "extra_params"], defaults=[""])


def void_for_whb0(x):
    return (not is_wormhole_b0()) and x or None


def void_for_gs(x):
    return (not is_grayskull()) and x or None


def void_for_bh(x):
    return (not is_blackhole()) and x or None


def filter_empty(fn):
    @wraps(fn)
    def __filter_empty():
        return list(filter(truth, fn()))

    return __filter_empty


def generate_test_entry_id(test_entry):
    if isinstance(test_entry, TestEntry):
        return f"{test_entry.test_name}-{test_entry.extra_params}"


def namespace_to_test_suite_type(namespace: str) -> TestSuiteType:
    test_suite_types_str_list = tuple(map(lambda type_: type_.name, TestSuiteType))

    namespace_as_test_suite_str = namespace.upper()

    assert namespace_as_test_suite_str in test_suite_types_str_list, f"{namespace} is not a recognized test type"

    return TestSuiteType[namespace_as_test_suite_str]


def is_test_suite_type_that_uses_silicon(test_suite_type: TestSuiteType) -> bool:
    return test_suite_type in (TestSuiteType.LLRT, TestSuiteType.TT_METAL)


def run_process_and_get_result(command, extra_env={}, capture_output=True):
    full_env = copy.deepcopy(os.environ)
    full_env.update(extra_env)

    result = sp.run(command, shell=True, capture_output=capture_output, env=full_env)

    return result


def get_git_home_dir_str():
    result = run_process_and_get_result("git rev-parse --show-toplevel")
    git_home_dir_str = result.stdout.decode("utf-8").replace("\n", "")

    git_home_dir = pathlib.Path(git_home_dir_str)

    assert git_home_dir.exists(), f"{git_home_dir} doesn't exist"
    assert git_home_dir.is_dir(), f"{git_home_dir} is not a directory"

    return git_home_dir_str


def get_env_dict_for_fw_tests(tt_arch):
    return {
        "TT_METAL_HOME": get_git_home_dir_str(),
        "ARCH_NAME": tt_arch,
    }


def default_build_full_path_to_test(namespace, executable_name, extra_params):
    return pathlib.Path(f"{get_git_home_dir_str()}/build/test/{namespace}/{executable_name}")


def build_executable_command_for_test(namespace: str, test_entry: TestEntry, timeout, tt_arch, build_full_path_to_test):
    assert namespace in (
        "build_kernels_for_riscv",
        "llrt",
        "tt_metal",
        "programming_example",
        "tt_eager",
    )

    test_name = test_entry.test_name
    executable_name = test_entry.executable_name
    extra_params = test_entry.extra_params

    full_path_to_test = build_full_path_to_test(namespace, executable_name, extra_params)

    assert (
        full_path_to_test.exists()
    ), f"Path to {test_name} does not exist - did you build it? Should be {full_path_to_test}"
    assert not full_path_to_test.is_dir()

    if namespace in ("build_kernels_for_riscv"):
        logger.warning(f"tt-arch should be injected as a cmdline param for build_kernels_for_riscv eventually")
        return f"timeout {timeout} {full_path_to_test} {extra_params}"
    elif namespace in ("llrt",):
        return f"timeout {timeout} {full_path_to_test} --arch {tt_arch} {extra_params}"
    else:
        logger.warning(f"tt-arch not implemented for {namespace}-type tests")
        return f"timeout {timeout} {full_path_to_test} {extra_params}"


class SpecificReturnCodes(Enum):
    PASSED_RETURN_CODE = 0
    TIMEOUT_RETURN_CODE = 124


def completed_process_failed(completed_process: sp.CompletedProcess):
    assert isinstance(completed_process, sp.CompletedProcess)

    does_not_match_return_code_of_process = partial(ne, completed_process.returncode)

    raw_specific_return_codes = map(lambda specific_code: specific_code.value, SpecificReturnCodes)

    does_not_match_any_specific_code = all(map(does_not_match_return_code_of_process, raw_specific_return_codes))

    return does_not_match_any_specific_code


def completed_process_timed_out(completed_process: sp.CompletedProcess):
    assert isinstance(completed_process, sp.CompletedProcess)

    return completed_process.returncode == SpecificReturnCodes.TIMEOUT_RETURN_CODE.value


def completed_process_passed(completed_process: sp.CompletedProcess):
    assert isinstance(completed_process, sp.CompletedProcess)

    return completed_process.returncode == SpecificReturnCodes.PASSED_RETURN_CODE.value


def get_result_str_from_completed_process_(completed_process: sp.CompletedProcess):
    assert isinstance(completed_process, sp.CompletedProcess)
    PASSED_STR = "\033[92m {}\033[00m".format("PASSED")
    FAILED_STR = "\033[91m {}\033[00m".format("FAILED")
    TIMEOUT_STR = "\033[93m {}\033[00m".format("TIMEOUT")

    if completed_process_passed(completed_process):
        return PASSED_STR
    elif completed_process_timed_out(completed_process):
        return TIMEOUT_STR
    elif completed_process_failed(completed_process):
        return FAILED_STR
    else:
        raise Exception("Unhandled return code for test process")


def report_tests(test_report):
    assert isinstance(test_report, dict)

    print("Printing test report for tt-metal regression")
    for test_entry, completed_process in test_report.items():
        assert isinstance(completed_process, sp.CompletedProcess)
        result_str = get_result_str_from_completed_process_(completed_process)
        extra_params_str = "/".join(test_entry.extra_params.split(" "))
        print(f"  {test_entry.test_name}-[{extra_params_str}]: {result_str}")


def run_single_test(
    namespace: str,
    test_entry: TestEntry,
    timeout,
    tt_arch="grayskull",
    capture_output=False,
    build_full_path_to_test=default_build_full_path_to_test,
):
    command = build_executable_command_for_test(
        namespace,
        test_entry,
        timeout=timeout,
        tt_arch=tt_arch,
        build_full_path_to_test=build_full_path_to_test,
    )

    env_for_fw_test = get_env_dict_for_fw_tests(tt_arch)

    completed_process = run_process_and_get_result(command, capture_output=capture_output, extra_env=env_for_fw_test)

    test_suite_type = namespace_to_test_suite_type(namespace)

    uses_tensix = is_test_suite_type_that_uses_silicon(test_suite_type)

    reset_tensix = completed_process_failed(completed_process) and uses_tensix

    if reset_tensix:
        logger.warning("Detected error on test that uses silicon - resetting")
        if tt_arch == "grayskull":
            run_process_and_get_result("tt-smi -tr all")
        elif tt_arch == "wormhole_b0":
            run_process_and_get_result("tt-smi -wr all wait")
        else:
            raise Exception(f"Unrecognized arch for tensix-reset: {tt_arch}")
        logger.warning("Silicon reset complete - returning status of FAILURE for this test")

    return completed_process


run_single_test_capture_output = partial(run_single_test, capture_output=True)


def test_report_has_all_passed(test_report):
    assert isinstance(test_report, dict)

    get_passed_values = lambda: test_report.values()

    is_completed_process = lambda value_: isinstance(value_, sp.CompletedProcess)

    is_all_completed_processes = all(map(is_completed_process, get_passed_values()))

    all_passed = all(map(completed_process_passed, get_passed_values()))

    return all_passed


def error_out_if_test_report_has_failures(test_report):
    all_passed = test_report_has_all_passed(test_report)

    if not all_passed:
        sys.exit(1)
