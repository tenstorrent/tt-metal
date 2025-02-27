import argparse
import importlib
import os
import sys
import json
import pathlib
import csv
from collections import defaultdict

# from framework.serialize import serialize, deserialize_vector
# from framework.sweeps_logger import sweeps_logger as logger
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression


SWEEPS_DIR = pathlib.Path(__file__).parent
VECTORS_FOLDER = "vectors_export"
RESULTS_FOLDER = "results_export"
TILE_SIZE = 32

DATUM_SIZE = {"DataType.BFLOAT16": 2}


def relative_root_mean_squared_error(y_actual, y_predicted):
    rmse = np.sqrt(np.mean((y_actual - y_predicted) ** 2))
    return rmse / np.mean(y_actual)


def root_mean_squared_relative_error(y_actual, y_predicted):
    relative_errors = (y_actual - y_predicted) / y_actual
    return np.sqrt(np.mean(relative_errors**2))


def get_vectors(suite: str, path):
    with open(path) as file:
        vectors = json.load(file)
    return {k: v for k, v in vectors[suite].items() if v["validity"] == "VectorValidity.VALID"}


def add_tile_and_data_size(vectors):  # -> dict[dict]
    for v in vectors.values():
        v["tile_size"] = (
            int(v["shape"][1:-1].split(",")[2]) * int(v["shape"][1:-1].split(",")[3]) / (TILE_SIZE * TILE_SIZE)
        )
        v["data_size"] = v["tile_size"] * DATUM_SIZE[v["dtype"]] * TILE_SIZE * TILE_SIZE
    return vectors


def get_suites(path):
    with open(path) as file:
        sweep_results = json.load(file)
    suite_name_list = [sweep["suite_name"] for sweep in sweep_results]
    return set(suite_name_list)


def get_perfs(suite: str, path):  # -> dict[float]
    with open(path) as file:
        sweep_results = json.load(file)
    if suite is None:
        suite_name_list = [sweep["suite_name"] for sweep in sweep_results]
        suite = set(suite_name_list)
        # print(f" detected suite names: {suite}")
        sweep_results = [
            sweep for sweep in sweep_results if sweep["status"] == "TestStatus.PASS" and sweep["suite_name"] in suite
        ]
    else:
        sweep_results = [
            sweep for sweep in sweep_results if sweep["status"] == "TestStatus.PASS" and sweep["suite_name"] == suite
        ]
    perfs = {}
    for sweep in sweep_results:
        perfs[sweep["vector_id"]] = float(sweep["device_perf"]["DEVICE KERNEL DURATION [ns]"])
    return perfs


def get_perfs_by_field(vectors, perfs, field_x: str):
    perfs_for_field = defaultdict(list)
    for vec_id, vec in vectors.items():
        perfs_for_field[vec[field_x]].append(perfs[vec_id])
    perfs_for_field = {k: np.mean(v) for k, v in perfs_for_field.items()}
    return perfs_for_field


def get_stats(reg, x, y):
    return {
        "r^2": reg.score(x, y),
        "RRMSE": relative_root_mean_squared_error(y, reg.predict(x)),
        "RMSRE": root_mean_squared_relative_error(y, reg.predict(x)),
        "num_points": len(x),
    }


def fit(vectors, perfs, field_x: str):
    x = []
    y = []

    for field, runs in get_perfs_by_field(vectors, perfs, field_x).items():
        x.append(float(field))
        y.append(runs)

    x = np.asarray(x).reshape(-1, 1)
    y = np.asarray(y)
    reg = LinearRegression().fit(x, y)
    return reg, get_stats(reg, x, y)


def process_single_module(module_name: str):
    full_path = os.path.join(SWEEPS_DIR, RESULTS_FOLDER, module_name + ".json")
    # print(f"Using result file: {full_path}")

    # TODO find a way to fix deserialization and use extract_plot_parameters if possible
    test_module = importlib.import_module("sweeps." + module_name)
    if "extract_plot_parameters" not in dir(test_module):
        return
    for suite in test_module.parameters:
        for i, p in enumerate(test_module.parameters[suite]):
            if i == 0:
                print(p)
            print(test_module.extract_plot_parameters(p))

    for suite in get_suites(full_path):
        # print(f"suite: {suite}")
        perfs = get_perfs(suite, full_path)
        vecs = get_vectors(suite, os.path.join(SWEEPS_DIR, VECTORS_FOLDER, module_name + ".json"))
        vecs = {k: vecs[k] for k in vecs if k in perfs}

        v = {}
        for perf_id in perfs:
            if perf_id not in vecs:
                continue
            p = vecs[perf_id]
            # print(p)

            # TODO manually edit this part to filter data points and extract parameters
            # filter some unwanted parameters...
            if not (
                vecs[perf_id]["input_a_dtype"] == "DataType.BFLOAT16"
                and vecs[perf_id]["input_b_dtype"] == "DataType.BFLOAT16"
                and vecs[perf_id]["output_dtype"] == "DataType.BFLOAT16"
            ):
                continue
            # height sharded
            output_memory_layout = json.loads(p["output_memory_config"]["data"])["memory_layout"]
            if output_memory_layout != 2:
                continue
            shard_spec = eval(p["height_sharded_specs"])
            if shard_spec[0] == (2,):
                continue

            input_shapes = [shard_spec[1], eval(p["k_size"]), eval(p["n_size"])]
            # print(f"{p}, type {type(p)}")
            # print(f"{test_module.extract_plot_parameters(p)}")
            print(f"{input_shapes[0]},{input_shapes[1]},{input_shapes[2]},{shard_spec[3]},{perfs[perf_id]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Perf Sweep Result Analyzer",
        description="Analyze perf sweep results",
    )
    parser.add_argument("--module-name", required=False, help="Test Module Name, or all tests if omitted")
    args = parser.parse_args(sys.argv[1:])

    results = []
    # print(f"SWEEPS_DIR: {SWEEPS_DIR}")
    # print(f"RESULTS_FOLDER: {RESULTS_FOLDER}")
    modules = []
    if args.module_name is None:
        modules = [file_name.removesuffix(".json") for file_name in os.listdir(SWEEPS_DIR / RESULTS_FOLDER)]
    else:
        modules = [args.module_name]

    for module in modules:
        process_single_module(module)
