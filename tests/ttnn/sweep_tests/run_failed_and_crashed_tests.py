# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import argparse

import ttnn


from tests.ttnn.sweep_tests.sweep import run_failed_and_crashed_tests


def convert_string_to_list(string):
    if string is None:
        output = []
    else:
        output = string.split(",")
        output = [element.strip() for element in output]
    return set(output)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--include", type=str)
    parser.add_argument("--exclude", type=str)
    parser.add_argument("--stepwise", action="store_true")

    include = parser.parse_args().include
    exclude = parser.parse_args().exclude
    stepwise = parser.parse_args().stepwise

    include = convert_string_to_list(include)
    exclude = convert_string_to_list(exclude)
    if include and exclude:
        raise ValueError("Cannot specify both include and exclude")

    device = ttnn.open_device(device_id=0)
    run_failed_and_crashed_tests(device=device, stepwise=stepwise, include=include, exclude=exclude)
    ttnn.close_device(device)


if __name__ == "__main__":
    main()
