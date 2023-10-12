#!/usr/bin/env python3

# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import sys
import os
sys.path.insert(0,os.environ['TT_METAL_HOME']+'/tt_metal/tools/profiler/')

import pandas as pd
import numpy as np
import glob
import argparse
import os
import csv
import datetime
from profile_unary import shapes, arch, no_iterations, config

from tt_metal.tools.profiler.process_device_log import import_device_profile_log

# we marked the kernel as 5, 6 time stamp markers
# we just want to find the diff of 5, 6 for TRISC 0, 1, 2 respectively and write it out
if os.environ.get("ARCH_NAME", "wormhole_b0") == "wormhole_b0":
    SCALE = 80
    Wcore, Hcore = 9, 8
else:
    SCALE = 120
    Wcore, Hcore = 12, 9


class TRISC:
    def __init__(self, start_exec_end: np.array):
        self.start_exec_end = start_exec_end

    @property
    def start(self):
        return self.start_exec_end[0]

    @property
    def exec(self):
        self.exec = self.start_exec_end[1]

    @property
    def end(self):
        self.end = self.start_exec_end[2]


def parse_device_log_for_opprofiler(filename):
    data = import_device_profile_log(filename)
    times = np.zeros((3,), dtype=float)
    scale = 0
    START_TAG = 9997
    END_TAG = 9998
    for core,dd in data['devices'][0]['cores'].items():
        core_x_col, core_y_col = core
        scale += 1

        trisc_0 = dict(dd['riscs']['TRISC_0']['timeseries'])
        trisc_1 = dict(dd['riscs']['TRISC_1']['timeseries'])
        trisc_2 = dict(dd['riscs']['TRISC_2']['timeseries'])

        end = np.array( [trisc_0[END_TAG], trisc_1[END_TAG], trisc_2[END_TAG]] )
        start= np.array( [trisc_0[START_TAG], trisc_1[START_TAG], trisc_2[START_TAG]] )

        try:
            times = np.vstack([times, (end - start).astype(float)])
        except Exception as _:
            pass
    lct = times[times[:, 1] == times.max(axis=0)[1]]
    return TRISC(lct[0])


def main(args):
    times = list(map(parse_device_log_for_opprofiler, glob.glob(f"{args.path}/**/*_log_device.csv", recursive=True)))
    paths = glob.glob(f"{args.path}/**/*_log_device.csv", recursive=True)
    result = times[0].start_exec_end
    for idx in range(1, len(times)):
        # if len(times) < 0 not times[idx]: continue
        result = np.vstack([result, times[idx].start_exec_end])
    print("value|\tTRISC 0,\t TRISC 1,\t TRISC 2")
    print("mean|\t", result.mean(axis=0))
    print("std|\t", result.var(axis=0) ** 0.5)

    fieldnames = [
        "operator",
        "shape",
        "cores",
        "Repetitions",
        "TRISC 0",
        "std (TRISC 0)",
        "TRISC 1",
        "std (TRISC 1)",
        "TRISC 2",
        "std (TRISC 2)",
        "Clock speed (MHz)",
        "Board",
        "Throughput (mean) samples/s",
        "Arch",
        "Config",
        "Date",
    ]

    def as_path(path):
        return path.split(os.path.sep)[1]

    op_name = as_path(paths[0])

    record1 = {
        "operator": op_name,
        "shape": shapes,
        "cores": 1,
        "Repetitions": no_iterations,
        "TRISC 0": round(result.mean(axis=0)[0]),
        "std (TRISC 0)": round(result.var(axis=0)[0] ** 0.5),
        "TRISC 1": round(result.mean(axis=0)[1]),
        "std (TRISC 1)": round(result.var(axis=0)[1] ** 0.5),
        "TRISC 2": round(result.mean(axis=0)[2]),
        "std (TRISC 2)": round(result.var(axis=0)[2] ** 0.5),
        "Clock speed (MHz)": 1202,
        "Board": "e120",
        "Throughput (mean) samples/s": round(1 / ((result.mean(axis=0)[1]) / (1202000000))),
        "Arch": arch,
        "Config": config,
        "Date": datetime.date.today().strftime("%m/%d/%Y"),
    }

    csv_file_path = "profiler_data.csv"

    with open(csv_file_path, mode="a", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        if csv_file.tell() == 0:
            writer.writeheader()
        writer.writerow(record1)
    return csv_file_path


def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", help="path containing the profile dumps", type=str)
    return parser


def get_args():
    parser = make_parser()
    return parser.parse_args()


def run_post_process(args: argparse.Namespace):
    folder_path = args.path
    assert os.path.exists(folder_path) and os.path.isdir(folder_path)
    folder_contents = os.listdir(folder_path)
    parser = make_parser()

    for folder_name in folder_contents:
        folder_full_path = os.path.join(folder_path, folder_name)
        args = parser.parse_args(["--path", folder_full_path])
        op_name = folder_name.split("-")[0]

        try:
            main(args)
        except IndexError as e:
            print(f"An error occurred while processing {folder_full_path}: {e}")
    return


if __name__ == "__main__":
    args = get_args()
    run_post_process(args)
