# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import argparse

import ttnn


from tests.ttnn.sweep_tests.sweep import run_failed_and_crashed_tests


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exclude", type=str)

    exclude = parser.parse_args().exclude
    exclude = exclude.split(",")
    exclude = [test_name.strip() for test_name in exclude]

    device = ttnn.open(0)
    run_failed_and_crashed_tests(device=device, exclude=exclude)
    ttnn.close(device)


if __name__ == "__main__":
    main()
