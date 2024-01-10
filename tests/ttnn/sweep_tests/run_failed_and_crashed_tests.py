# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import argparse

import ttnn


from tests.ttnn.sweep_tests.sweep import run_failed_and_crashed_tests


def parse_exclude_string(exclude):
    if exclude is None:
        exclude = []
    else:
        exclude = exclude.split(",")
        exclude = [test_name.strip() for test_name in exclude]
    return set(exclude)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exclude", type=str)
    parser.add_argument("--stepwise", action="store_true")

    exclude = parser.parse_args().exclude
    stepwise = parser.parse_args().stepwise

    exclude = parse_exclude_string(exclude)

    device = ttnn.open(0)
    run_failed_and_crashed_tests(device=device, stepwise=stepwise, exclude=exclude)
    ttnn.close(device)


if __name__ == "__main__":
    main()
