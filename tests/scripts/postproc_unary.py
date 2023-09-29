#!/usr/bin/env python3

import pandas as pd
import numpy as np
import glob
import argparse
import os
import csv
import datetime
from profile_unary import shapes, arch, no_iterations, config

from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor


def run_io_tasks_in_parallel(tasks):
    with ThreadPoolExecutor() as executor:
        running_tasks = [executor.submit(task) for task in tasks]
        for running_task in running_tasks:
            running_task.result()


def run_cpu_tasks_in_parallel(tasks):
    data = []
    with ProcessPoolExecutor(max_workers=100) as executor:
        running_tasks = [executor.submit(task) for task in tasks]
        for running_task in running_tasks:
            data.append(running_task.result)
    return data


# we marked the kernel as 5, 6 time stamp markers
# we just want to find the diff of 5, 6 for TRISC 0, 1, 2 respectively and write it out
from tests.tt_eager.python_api_testing.sweep_tests.common import is_wormhole_b0

if is_wormhole_b0():
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


from itertools import product


def parse_device_log(filename):
    dd = pd.read_csv(filename, header=1)
    core_x_col = dd.columns[1]
    core_y_col = dd.columns[2]
    timer_id_col = dd.columns[-2]
    time_col = dd.columns[-1]
    times = np.zeros((3,), dtype=float)
    scale = 0
    for core_x, core_y in product(range(Wcore), range(Hcore)):
        scale += 1
        dd_ = dd[dd[core_x_col] == core_x]
        d = dd_[dd_[core_y_col] == core_y]
        start = d[d[timer_id_col] == 5].to_numpy()
        end = d[d[timer_id_col] == 6].to_numpy()
        try:
            times = np.vstack([times, (end[:, -1] - start[:, -1]).astype(float)])
        except Exception as _:
            pass
    lct = times[times[:, 1] == times.max(axis=0)[1]]
    return TRISC(lct[0])


def main(args):
    # times = run_cpu_tasks_in_parallel(map(parse_device_log, glob.glob(f"{args.path}/**/*_log_device.csv", recursive=True)))
    times = list(map(parse_device_log, glob.glob(f"{args.path}/**/*_log_device.csv", recursive=True)))
    result = times[0].start_exec_end
    for idx in range(1, len(times)):
        # if len(times) < 0 not times[idx]: continue
        result = np.vstack([result, times[idx].start_exec_end])
    print(result)
    print(result.shape)
    print(result.mean(axis=0))
    print(result.var(axis=0) ** 0.5)

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

    person1 = {
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
        writer.writerow(person1)

def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path",help="path containing the profile dumps", type=str)
    return parser

def get_args():
    parser = make_parser()
    return parser.parse_args()

# print( parse_device_log("./tt::tt_metal::EltwiseUnary/720/profile_log_device.csv") )
if __name__ == "__main__":
    folder_path = get_args().path

    if os.path.exists(folder_path) and os.path.isdir(folder_path):
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
