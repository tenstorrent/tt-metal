# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import csv
import re
import time
from os import listdir
from os.path import isfile, join

import git
from loguru import logger

today = time.strftime("%Y_%m_%d")


def merge_perf_files(fname, perf_fname, expected_cols):
    mypath = "./"
    csvfiles = [
        f
        for f in listdir(mypath)
        if isfile(join(mypath, f)) and re.match(f"{perf_fname}_.*_{today}.csv", f) is not None
    ]

    repo = git.Repo(search_parent_directories=True)

    merge_res = open(fname, "w")
    if not repo.head.is_detached:
        merge_res.write(f"branch: {repo.active_branch} \n")
    merge_res.write(f"hash: {repo.head.object.hexsha} \n")
    cols = ", ".join(expected_cols)
    merge_res.write(f"{cols} \n")

    csvfiles.sort()
    for csvfile in csvfiles:
        row_name = csvfile.replace("perf_", "")
        row_name = row_name.replace(f"{today}", "")
        row_name = row_name.replace(".csv", "")

        f = open(f"./{csvfile}", "r")
        f.readline()
        row = f.readline().strip().strip("\n")
        merge_res.write(f"{row}\n")

    merge_res.close()


def process_perf_results(fname, expected_cols):
    with open(fname) as file:
        merge_res = csv.reader(file, skipinitialspace=True)
        logger.info(next(merge_res)[0].strip())
        logger.info(next(merge_res)[0].strip())
        cols = next(merge_res)
        cols = [c.strip() for c in cols]
        assert len(expected_cols) == len(cols), "Mismatch between expected and actual columns in perf CSV"
        for expected_c, c in zip(expected_cols, cols):
            assert expected_c == c, f"Expected column {expected_c} but instead got {c}"
        merge_res = list(merge_res)
    return cols, merge_res


def check_perf_results(
    fname,
    expected_cols,
    check_cols,
    table_width=110,
    float_precision=4,
):
    red_color_code = "\033[91m"
    green_color_code = "\033[92m"
    reset_color_code = "\033[0m"
    bold_code = "\033[1m"
    underline_code = "\033[4m"
    fail_label = "[FAIL]"
    pass_label = "[PASS]"

    cols, merge_res = process_perf_results(fname, expected_cols)
    visited_models = []
    slow_measured = {col: [] for col in check_cols}

    logger.info(f"Checking perf results for {fname}...")
    print("=" * table_width)
    print(f"{bold_code}PERFORMANCE RESULTS{reset_color_code}")
    print("=" * table_width)

    for models_info in merge_res:
        dict_info = {name: value for name, value in zip(cols, models_info)}
        model_name = f"{dict_info['Model']}_{dict_info['Setting']}"
        visited_models.append(model_name)

        print(f"\n{bold_code}{underline_code}Model: {dict_info['Model']}{reset_color_code}")
        print(f"{bold_code}Setting: {dict_info['Setting']}{reset_color_code}")
        print("-" * 60)

        highlight = False
        for col in check_cols:
            model_expected_col = float(dict_info[f"Expected {col}"])
            model_measured_col = float(dict_info[col])
            if model_measured_col > model_expected_col:
                slow_measured[col].append((model_name, model_measured_col, model_expected_col))
                print(
                    f"  FAIL {col}: {red_color_code}{model_measured_col:.{float_precision}f} > {model_expected_col:.{float_precision}f}{reset_color_code}"
                )
                highlight = True
            else:
                print(
                    f"  PASS {col}: {model_measured_col:.{float_precision}f} <= {model_expected_col:.{float_precision}f}"
                )

        if highlight:
            print(f"  {red_color_code}This model has performance issues{reset_color_code}")
        else:
            print(f"  {green_color_code}All performance targets met{reset_color_code}")

    print("\n" + "=" * table_width)
    print(f"{bold_code}SUMMARY{reset_color_code}")
    print("=" * table_width)

    all_passed = True
    for col in slow_measured:
        if slow_measured[col]:
            all_passed = False
            print(f"{red_color_code}{fail_label} The following models are too slow for '{col}':{reset_color_code}")
            for model_name, measured, expected in slow_measured[col]:
                print(
                    f"  - {model_name}: measured={measured:.{float_precision}f}, expected={expected:.{float_precision}f}"
                )
        else:
            print(f"{green_color_code}{pass_label} All models meet expected '{col}' performance.{reset_color_code}")

    print("=" * table_width)
    if all_passed:
        print(f"{green_color_code}All performance checks passed!{reset_color_code}")
    else:
        print(f"{red_color_code}Some performance checks failed!{reset_color_code}")
    print("=" * table_width)

    if any(len(slow) > 0 for slow in slow_measured.values()):
        raise Exception("Some model(s) are too slow - see above for details on slow models")


def prep_perf_report(
    model_name: str,
    batch_size: int,
    inference_and_compile_time: float,
    inference_time: float,
    expected_compile_time: float,
    expected_inference_time: float,
    comments: str,
    inference_time_cpu: float = None,
):
    today = time.strftime("%Y_%m_%d")

    def write_dict_to_file(csv_path, dict_res):
        columns = ", ".join([str(d) for d in dict_res.keys()])
        values = ", ".join([d for d in dict_res.values()])

        with open(csv_path, "w") as csvfile:
            csvfile.write(columns)
            csvfile.write("\n")
            csvfile.write(values)

    compile_time = inference_and_compile_time - inference_time
    device_throughput = "{:.4f}".format(batch_size * (1 / inference_time))
    cpu_throughput = batch_size * (1 / inference_time_cpu) if inference_time_cpu else "unknown"
    cpu_throughput = "{:.4f}".format(cpu_throughput) if not isinstance(cpu_throughput, str) else cpu_throughput
    dict_res = {
        "Model": model_name,
        "Setting": comments,
        "Batch": str(batch_size),
        "First Run (sec)": "{:.2f}".format(inference_and_compile_time),
        "Second Run (sec)": "{:.2f}".format(inference_time),
        "Compile Time (sec)": "{:.2f}".format(compile_time),
        "Expected Compile Time (sec)": "{:.2f}".format(expected_compile_time),
        "Inference Time (sec)": "{:.4f}".format(inference_time),
        "Expected Inference Time (sec)": "{:.4f}".format(expected_inference_time),
        "Throughput (batch*inf/sec)": device_throughput,
        "Inference Time CPU (sec)": "{:.4f}".format(inference_time_cpu) if inference_time_cpu else "unknown",
        "Throughput CPU (batch*inf/sec)": cpu_throughput,
    }

    model_name = model_name.replace("/", "_")
    csv_file = f"perf_{model_name}_{comments}_{today}.csv"
    write_dict_to_file(csv_file, dict_res)
