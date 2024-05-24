# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import argparse

import ttnn


from sweeps import run_sweeps, print_report


def convert_string_to_set(string):
    return set([element.strip() for element in string.split(",")]) if string else set()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--include", type=str, help="Comma separated list of sweep names to include eg) --include sweep1.py,sweep2.py"
    )
    parser.add_argument(
        "--collect-only", action="store_true", help="Print the sweeps that will be run but do not run them"
    )

    args = parser.parse_args()
    include = convert_string_to_set(args.include)
    device = None
    if not args.collect_only:
        device = ttnn.open_device(device_id=0)
        print("Running sweeps...")
    else:
        print("Collecting sweeps to run...")

    table_names = run_sweeps(device=device, include=include)
    if not args.collect_only:
        ttnn.close_device(device)
        print_report(table_names=table_names)


if __name__ == "__main__":
    main()
