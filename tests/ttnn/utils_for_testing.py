# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from models.utility_functions import comp_pcc
import os
import json


def print_comparison(message, expected_pytorch_result, actual_pytorch_result):
    messages = []
    messages.append(message)
    messages.append("Expected")
    # TODO: handle other dimensions
    if len(expected_pytorch_result.shape) == 4:
        messages.append(str(expected_pytorch_result[0:5, 0:5, 0:5, 0:5]))
    messages.append("Actual")
    if len(actual_pytorch_result.shape) == 4:
        messages.append(str(actual_pytorch_result[0:5, 0:5, 0:5, 0:5]))
    return "\n".join(messages)


def assert_with_pcc(expected_pytorch_result, actual_pytorch_result, pcc=0.99):
    pcc_passed, pcc_message = comp_pcc(expected_pytorch_result, actual_pytorch_result, pcc)
    assert pcc_passed, print_comparison(pcc_message, expected_pytorch_result, actual_pytorch_result)


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


#    input("Press Enter to continue once the debugger is attached...")
