import copy
import os
import json
import subprocess
import pytest
import regex as re
from argparse import ArgumentParser

DEFAULT_TEST_CONFIG = "tests/tt_metal/tt_fabric/common/fabric_test_config.json"
FABRIC_TESTS_DIR = "test/tt_metal/perf_microbenchmark/routing"
TOP_LEVEL_KEY = "fabric_test_config"

TEST_GROUP_CHOICES = ["sanity", "benchmark", "all"]
BOARD_TYPE_CHOICES = ["n300", "t3k", "glx8", "glx16", "glx32"]

# env vars
TT_METAL_HOME = os.getenv("TT_METAL_HOME")
ARCH_NAME = os.getenv("ARCH_NAME")

NUM_FAILED_TESTS = 0


def read_test_config(test_config_rel_path):
    test_config_abs_path = os.path.join(TT_METAL_HOME, test_config_rel_path)
    assert os.path.exists(test_config_abs_path)
    with open(test_config_abs_path, "r") as file:
        data = json.load(file)

    # check for the top level key
    assert TOP_LEVEL_KEY in data

    return data


def parse_test_group(test_config, group_name):
    for group in test_config[TOP_LEVEL_KEY]:
        # TODO: check for required keys
        if group_name == group["test_group_name"]:
            return group["tests"]


def convert_to_param(key, value):
    param_str = "--" + key + " "
    if not isinstance(value, bool):
        param_str += str(value) + " "

    return param_str


def run_test(test_file, params):
    global NUM_FAILED_TESTS
    full_env = copy.deepcopy(os.environ)
    command = "TT_METAL_SLOW_DISPATCH_MODE=1" + " "
    command += test_file
    command += params
    print("Running test: ", command)
    # TODO: add timeouts
    result = subprocess.run(command, shell=True, capture_output=True, env=full_env)
    if result.returncode:
        print("Test failed")
        print(result.stdout.decode("utf-8"))
        print(result.stderr.decode("utf-8"))
        NUM_FAILED_TESTS += 1

    return result.returncode, result.stdout.decode("utf-8")


def parse_perf_from_output(perf_type, output):
    if "bandwidth" == perf_type:
        pattern = r"(Total TX BW = )([0-9\.]*)( B/cycle)"
        result = re.findall(pattern, output)
        return float(result[0][1])
    else:
        return None


def compare_perf(perf_type, expectation, tolerance, output):
    global NUM_FAILED_TESTS
    test_perf = parse_perf_from_output(perf_type, output)
    if "bandwidth" == perf_type:
        if test_perf >= (1 - tolerance) * expectation:
            print(
                f"Test performance within tolerance. Expected: {expectation}, Observed: {test_perf}, Tolerance: {tolerance}"
            )
            return True
        else:
            print(
                f"Test performance outside tolerance. Expected: {expectation}, Observed: {test_perf}, Tolerance: {tolerance}"
            )
            NUM_FAILED_TESTS += 1
            return False


def parse_and_run_tests(test_config, group_name, board_type):
    test_group = parse_test_group(test_config, group_name)
    benchmark = False
    print("Running test group: ", group_name)
    for test in test_group:
        print("Running test name: ", test["test_name"])
        # TODO: check for required keys
        test_file = os.path.join(TT_METAL_HOME, "build", FABRIC_TESTS_DIR, test["test_file"])
        test_file += "_" + ARCH_NAME
        if "benchmark" == group_name:
            assert "perf_expectation" in test
            benchmark = True
        param_str = convert_to_param("board_type", board_type)
        for key, value in test["test_params"].items():
            param_str += convert_to_param(key, value)
        if "paramterize" in test:
            # TODO: match the size of expectations if benchmarking
            for key, values in test["paramterize"].items():
                for i in range(len(values)):
                    add_param = convert_to_param(key, values[i])
                    return_code, output = run_test(test_file, param_str + add_param)
                    # TODO: check only for tests that ran successfully
                    if benchmark and not return_code:
                        outcome = compare_perf(
                            test["perf_type"], test["perf_expectation"][i], test["perf_tolerance"], output
                        )


def main():
    parser = ArgumentParser()
    parser.add_argument("--test-group", type=str, choices=TEST_GROUP_CHOICES, default="sanity")
    parser.add_argument("--board-type", type=str, choices=BOARD_TYPE_CHOICES, default="glx32")
    parser.add_argument("--test-config", type=str, default=DEFAULT_TEST_CONFIG)
    args = parser.parse_args()

    test_config = read_test_config(args.test_config)
    test_groups = [x for x in TEST_GROUP_CHOICES if x is not "all"]
    print(test_groups)
    if "all" == args.test_group:
        for group in test_groups:
            parse_and_run_tests(test_config, group, args.board_type)
    else:
        parse_and_run_tests(test_config, args.test_group, args.board_type)

    if NUM_FAILED_TESTS > 0:
        return 1
    else:
        return 0


if __name__ == "__main__":
    main()
