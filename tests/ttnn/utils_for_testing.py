# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import os
import json

import torch

from models.utility_functions import comp_pcc


def torch_random(shape, low, high, dtype):
    return torch.zeros(shape, dtype=dtype).uniform_(low, high)


def print_comparison(message, expected_pytorch_result, actual_pytorch_result):
    messages = []
    messages.append(message)
    messages.append("Expected")
    messages.append(str(expected_pytorch_result))
    messages.append("Actual")
    messages.append(str(actual_pytorch_result))
    return "\n".join(messages)


def assert_with_pcc(expected_pytorch_result, actual_pytorch_result, pcc=0.99):
    assert list(expected_pytorch_result.shape) == list(
        actual_pytorch_result.shape
    ), f"list(expected_pytorch_result.shape)={list(expected_pytorch_result.shape)} vs list(actual_pytorch_result.shape)={list(actual_pytorch_result.shape)}"
    pcc_passed, pcc_message = comp_pcc(expected_pytorch_result, actual_pytorch_result, pcc)
    assert pcc_passed, print_comparison(pcc_message, expected_pytorch_result, actual_pytorch_result)


def check_with_pcc(expected_pytorch_result, actual_pytorch_result, pcc=0.99):
    return (
        expected_pytorch_result.shape == actual_pytorch_result.shape,
        f"list(expected_pytorch_result.shape)={list(expected_pytorch_result.shape)} vs list(actual_pytorch_result.shape)={list(actual_pytorch_result.shape)}",
    )
    pcc_passed, pcc_message = comp_pcc(expected_pytorch_result, actual_pytorch_result, pcc)
    return pcc_passed, pcc_message


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
