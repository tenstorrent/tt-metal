# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import os
import json
import time

from loguru import logger
from models.utility_functions import comp_pcc, comp_equal, divup, roundup
from typing import Tuple


def construct_pcc_assert_message(message, expected_pytorch_result, actual_pytorch_result):
    messages = []
    messages.append(message)
    # messages.append("Expected")
    # messages.append(str(expected_pytorch_result))
    # messages.append("Actual")
    # messages.append(str(actual_pytorch_result))
    messages = [str(m) for m in messages]
    return "\n".join(messages)


def assert_with_pcc(expected_pytorch_result, actual_pytorch_result, pcc=0.9999):
    assert list(expected_pytorch_result.shape) == list(
        actual_pytorch_result.shape
    ), f"list(expected_pytorch_result.shape)={list(expected_pytorch_result.shape)} vs list(actual_pytorch_result.shape)={list(actual_pytorch_result.shape)}"
    pcc_passed, pcc_message = comp_pcc(expected_pytorch_result, actual_pytorch_result, pcc)
    assert pcc_passed, construct_pcc_assert_message(pcc_message, expected_pytorch_result, actual_pytorch_result)
    return pcc_passed, pcc_message


def assert_equal(expected_pytorch_result, actual_pytorch_result):
    assert list(expected_pytorch_result.shape) == list(
        actual_pytorch_result.shape
    ), f"list(expected_pytorch_result.shape)={list(expected_pytorch_result.shape)} vs list(actual_pytorch_result.shape)={list(actual_pytorch_result.shape)}"
    equal_passed, equal_message = comp_equal(expected_pytorch_result, actual_pytorch_result)
    return equal_passed, equal_message


def check_with_pcc(expected_pytorch_result, actual_pytorch_result, pcc=0.9999):
    if expected_pytorch_result.shape != actual_pytorch_result.shape:
        return (
            False,
            f"list(expected_pytorch_result.shape)={list(expected_pytorch_result.shape)} vs list(actual_pytorch_result.shape)={list(actual_pytorch_result.shape)}",
        )
    pcc_passed, pcc_message = comp_pcc(expected_pytorch_result, actual_pytorch_result, pcc)
    return pcc_passed, construct_pcc_assert_message(pcc_message, expected_pytorch_result, actual_pytorch_result)


def check_with_pcc_list(expected_pytorch_result, actual_pytorch_result, pcc=0.9999):
    if len(expected_pytorch_result) != len(actual_pytorch_result):
        return (
            False,
            f"len(expected_pytorch_result)={len(expected_pytorch_result)} vs len(actual_pytorch_result)={len(actual_pytorch_result)}",
        )

    pcc_passed = []
    pcc_message = ""
    for i in range(len(expected_pytorch_result)):
        pcc_passed_, pcc_message_ = check_with_pcc(expected_pytorch_result[i], actual_pytorch_result[i], pcc)
        pcc_passed.append(pcc_passed_)
        pcc_message += pcc_message_ + ", "

    if all(pcc_passed):
        passed = True
    else:
        passed = False

    pcc_message = pcc_message[:-2]
    return passed, construct_pcc_assert_message(pcc_message, expected_pytorch_result, actual_pytorch_result)


def check_with_pcc_without_tensor_printout(expected_pytorch_result, actual_pytorch_result, pcc=0.9999):
    if expected_pytorch_result.shape != actual_pytorch_result.shape:
        return (
            False,
            f"list(expected_pytorch_result.shape)={list(expected_pytorch_result.shape)} vs list(actual_pytorch_result.shape)={list(actual_pytorch_result.shape)}",
        )
    pcc_passed, pcc_message = comp_pcc(expected_pytorch_result, actual_pytorch_result, pcc)
    return pcc_passed, pcc_message


def set_slow_dispatch_mode(set_var):
    prev_value = os.environ.pop("TT_METAL_SLOW_DISPATCH_MODE", None)

    if set_var != "" and set_var is not None:
        os.environ["TT_METAL_SLOW_DISPATCH_MODE"] = set_var
        logger.info("Setting slow dispatch mode")
    else:
        logger.info("Setting fast dispatch mode")

    return prev_value


def update_process_id():
    print(f"Debugging PID: {os.getpid()}")
    cwd = os.getcwd()
    launch_json_path = os.path.join(cwd, ".vscode", "launch.json")
    with open(launch_json_path, "r") as f:
        launch_data = json.load(f)

    for config in launch_data.get("configurations", []):
        if config.get("name") == "C++: Attach to Python":
            config["processId"] = str(os.getpid())
            break

    with open(launch_json_path, "w") as f:
        json.dump(launch_data, f, indent=4)


def get_per_core_size_and_num_cores(
    size: int, num_cores_choices: Tuple[int, ...], min_per_core_size: int = 32, max_per_core_size: int = None
) -> Tuple[int, int]:
    if max_per_core_size is None:
        max_per_core_size = size

    for num_cores in num_cores_choices:
        per_core_size = roundup(divup(size, num_cores), 32)  # Divide, round up, then round up to nearest 32
        if per_core_size > min_per_core_size and per_core_size < max_per_core_size:
            # Actual num_cores might be less after we round up to nearest 32
            num_cores_actual = divup(size, per_core_size)  # Divide and round up
            yield per_core_size, num_cores_actual


def start_measuring_time() -> int:
    return time.time_ns()


def stop_measuring_time(start_time) -> int:
    return time.time_ns() - start_time
