# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import subprocess
import time
import os
import json
import pandas as pd
import argparse
import copy

TARGET_JSON_PATH = "models/demos/llama3_subdevices/tests/decoder_perf_targets_4u.json"


def reset_device():
    print("BE SURE YOU ARE RESETTING THE CORRECT DEVICE")
    print("Resetting devices...")
    try:
        result = subprocess.run(["tt-smi", "-r", "/opt/tt_metal_infra/scripts/reset.json"])
        if result.returncode != 0:
            print(f"[ERROR] Device reset failed with code {result.returncode}")
    except Exception as e:
        print(f"[EXCEPTION] Exception during device reset: {e}")
    else:
        print("Device reset completed.\n")


def run_test(test_name: str):
    print(f"Running test: {test_name}")
    try:
        env = os.environ.copy()
        env["FAKE_DEVICE"] = "TG"
        env["TT_METAL_ENABLE_ERISC_IRAM"] = "1"
        env["TT_METAL_KERNELS_EARLY_RETURN"] = "1"

        command = ["pytest", test_name]
        result = subprocess.run(command, env=env)

        if result.returncode != 0:
            print(f"[ERROR] Test {test_name} failed with return code {result.returncode}.")
        else:
            print(f"Completed test: {test_name}\n")
    except Exception as e:
        print(f"[EXCEPTION] Exception while running {test_name}: {e}")


def generate_perf_json(decoder_csv="out.csv", dispatch_csv="out-non-dispatched.csv"):
    print("Updating performance targets from latest CSVs...")
    df_main = pd.read_csv(decoder_csv)
    df_dispatch = pd.read_csv(dispatch_csv)

    def is_collective(op):
        return any(keyword in op for keyword in ["AllGather", "ReduceScatter", "AllReduce", "Matmul_RS"])

    decoder_ops = set(df_main[df_main["metric_type"].str.startswith("mid_")]["op_code_with_id"].unique())
    model_tail_ops = set(df_main[df_main["metric_type"].str.startswith("model_tail")]["op_code_with_id"].unique())

    decoder_df = df_main[df_main["op_code_with_id"].isin(decoder_ops)]
    model_tail_df = df_main[df_main["op_code_with_id"].isin(model_tail_ops)]

    def aggregate_metrics(df, metrics):
        df_filtered = df[df["metric_type"].isin(metrics)]
        pivot = df_filtered.pivot_table(
            index=["run_number", "op_code_with_id"], columns="metric_type", values="value"
        ).reset_index()
        return pivot.groupby("op_code_with_id").agg(["mean", "min", "max"])

    decoder_agg = aggregate_metrics(
        decoder_df, ["mid_compile_avg", "mid_trace_avg", "mid_dispatch_avg", "mid_first_to_last_start_avg"]
    )

    decoder_ops_combined = {}
    for op in decoder_agg.index:
        collective = is_collective(op)
        kernel_metric = "mid_trace_avg" if collective else "mid_compile_avg"

        try:
            decoder_ops_combined[op] = {
                "op_name": op,
                "kernel_duration": decoder_agg.loc[op][(kernel_metric, "mean")],
                "op_to_op": decoder_agg.loc[op][("mid_dispatch_avg", "mean")],
                "first_to_last_start": decoder_agg.loc[op][("mid_first_to_last_start_avg", "mean")],
                "non-overlapped-dispatch-time": df_dispatch[
                    (df_dispatch["metric_type"] == "mid_dispatch_avg") & (df_dispatch["op_code_with_id"] == op)
                ]["value"].mean(),
            }
        except KeyError:
            print(f"[WARN] Missing data for decoder op: {op}, skipping.")

    tail_agg = aggregate_metrics(
        model_tail_df, ["model_tail_compile_avg", "model_tail_trace_avg", "model_tail_dispatch_avg"]
    )

    tail_dispatch = (
        df_dispatch[df_dispatch["metric_type"] == "model_tail_dispatch_avg"].groupby("op_code_with_id")["value"].mean()
    )

    model_tail_combined = {}
    for op in tail_agg.index:
        collective = is_collective(op)
        kernel_metric = "model_tail_trace_avg" if collective else "model_tail_compile_avg"

        try:
            model_tail_combined[op] = {
                "op_name": op,
                "kernel_duration": tail_agg.loc[op][(kernel_metric, "mean")],
                "op_to_op": tail_agg.loc[op][("model_tail_dispatch_avg", "mean")],
                "non-overlapped-dispatch-time": tail_dispatch.get(op, 0),
            }
        except KeyError:
            print(f"[WARN] Missing data for model tail op: {op}, skipping.")

    # Load and update existing file
    if os.path.exists(TARGET_JSON_PATH):
        with open(TARGET_JSON_PATH, "r") as f:
            existing_data = json.load(f)
    else:
        print(f"[WARN] {TARGET_JSON_PATH} not found. Creating a new one.")
        existing_data = {"decoder": {}, "model_tail": {}}

    updated_data = copy.deepcopy(existing_data)

    for op, new_vals in decoder_ops_combined.items():
        updated_data["decoder"].setdefault(op, {})
        for key, val in new_vals.items():
            updated_data["decoder"][op][key] = val  # Overwrite only performance values

    for op, new_vals in model_tail_combined.items():
        updated_data["model_tail"].setdefault(op, {})
        for key, val in new_vals.items():
            updated_data["model_tail"][op][key] = val

    with open(TARGET_JSON_PATH, "w") as f:
        json.dump(updated_data, f, indent=4)

    print(f"Updated performance targets written to {TARGET_JSON_PATH}")


def automate_perf_collection(num_runs: int = 5):
    decoder_test = "models/demos/llama3_subdevices/tests/test_decoder_device_perf.py::test_llama_TG_perf_device"
    dispatch_test = "models/demos/llama3_subdevices/tests/test_decoder_device_perf.py::test_llama_TG_perf_device_non_overlapped_dispatch"

    for i in range(1, num_runs + 1):
        print(f"\n===== Run {i}/{num_runs} =====")
        reset_device()
        print("Sleeping 20 seconds for hardware to reset...")
        time.sleep(20)
        run_test(decoder_test)

        reset_device()
        print("Sleeping 20 seconds for hardware to reset...")
        time.sleep(20)
        run_test(dispatch_test)

    generate_perf_json()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Automate LLaMA performance data collection")
    parser.add_argument("--runs", type=int, default=5, help="Number of times to repeat each test")
    args = parser.parse_args()

    automate_perf_collection(num_runs=args.runs)
