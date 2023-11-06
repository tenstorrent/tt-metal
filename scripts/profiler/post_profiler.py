#!/usr/bin/env python3

# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import csv
import os
import argparse

from tt_metal.tools.profiler.common import PROFILER_LOGS_DIR, PROFILER_DEVICE_SIDE_LOG


def make_parser():
    description = (
        f"""
Steps to dump profiler result in profile_data.csv:

Run the test_eltwise_unary.cpp file with required op enabled with required configuration.

    $ make tests

run the compiled file using ./build/test/tt_metal/test_eltwise_unary
Once test is completed successfully. It will dump a {TT_METAL_HOME}/generated/profiler/.logs/profile_log_device.csv file.
Run the command,

    $ python """
        + __file__
        + """ --path {{path to csv}}

It will dump profiler data into a new tt-metal/profile_data.csv file.

"""
    )
    parser = argparse.ArgumentParser(description=description, formatter_class=argparse.RawTextHelpFormatter)
    ogpath = PROFILER_LOGS_DIR / PROFILER_DEVICE_SIDE_LOG
    parser.add_argument("--path", help="path containing the profile dumps", default=ogpath, type=str)
    return parser


def get_args():
    parser = make_parser()
    return parser.parse_args()


def main():
    args = get_args()
    file = open(args.path, mode="r")
    csvFile = csv.reader(file)
    target_column = 3
    next(csvFile)
    next(csvFile)
    sum_TRISC_0_start = 0
    sum_TRISC_1_start = 0
    sum_TRISC_2_start = 0
    sum_TRISC_0_end = 0
    sum_TRISC_1_end = 0
    sum_TRISC_2_end = 0
    for rows in csvFile:
        if rows[4] == " 9997" and rows[3] == " TRISC_0":
            sum_TRISC_0_start += int(rows[-1])
        elif rows[4] == " 9998" and rows[3] == " TRISC_0":
            sum_TRISC_0_end += int(rows[-1])
        elif rows[4] == " 9997" and rows[3] == " TRISC_1":
            sum_TRISC_1_start += int(rows[-1])
        elif rows[4] == " 9998" and rows[3] == " TRISC_1":
            sum_TRISC_1_end += int(rows[-1])
        elif rows[4] == " 9997" and rows[3] == " TRISC_2":
            sum_TRISC_2_start += int(rows[-1])
        elif rows[4] == " 9998" and rows[3] == " TRISC_2":
            sum_TRISC_2_end += int(rows[-1])
    dif_t0 = sum_TRISC_0_end - sum_TRISC_0_start
    dif_t1 = sum_TRISC_1_end - sum_TRISC_1_start
    dif_t2 = sum_TRISC_2_end - sum_TRISC_2_start

    no_of_iteration = 1000
    avg_0 = round(dif_t0 / no_of_iteration)
    avg_1 = round(dif_t1 / no_of_iteration)
    avg_2 = round(dif_t2 / no_of_iteration)
    throughput = round(1 / (avg_1 / (1.202 * 1000000000)))
    print("TRISC 0 = ", avg_0)
    print("TRISC 1 = ", avg_1)
    print("TRISC 2 = ", avg_2)
    print("Throughput = ", throughput)

    fieldnames = [
        "Shape",
        "Cores",
        "Repetitions",
        "TRISC 0",
        "TRISC 1",
        "TRISC 2",
        "Clock speed (MHz)",
        "Board",
        "Throughput (mean) samples/s",
        "Arch",
    ]
    data_to_write = [
        {
            "Shape": "1x1x32x32",
            "Cores": 1,
            "Repetitions": no_of_iteration,
            "TRISC 0": avg_0,
            "TRISC 1": avg_1,
            "TRISC 2": avg_2,
            "Clock speed (MHz)": 1202,
            "Board": "e150",
            "Throughput (mean) samples/s": throughput,
            "Arch": "GS",
        }
    ]
    with open("profile_data.csv", "a", newline="") as data_dump:
        csv_writer = csv.DictWriter(data_dump, fieldnames=fieldnames)
        if data_dump.tell() == 0:
            csv_writer.writeheader()
        csv_writer.writerows(data_to_write)
    file.close()
    return


if __name__ == "__main__":
    main()
