# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import argparse

import ttnn


from sweeps import run_sweeps, print_report


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

    include = parser.parse_args().include

    include = convert_string_to_list(include)

    device = ttnn.open_device(device_id=0)
    table_names = run_sweeps(device=device, include=include)
    ttnn.close_device(device)
    print_report(table_names=table_names)


if __name__ == "__main__":
    main()
