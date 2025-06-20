# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import subprocess
import time
import os
import json
import pandas as pd
import argparse

# Update this to the location of the RESET file
RESET_SCRIPT_PATH = "/proj_sw/user_dev/yalrawwash/reset/reset-1.json"


def reset_device():
    """Reset devices using tt-smi."""
    print("BE SURE YOU ARE RESETTING THE CORRECT DEVICE")
    print("Resetting devices...")
    try:
        result = subprocess.run(["tt-smi", "-r", RESET_SCRIPT_PATH])
        if result.returncode != 0:
            print(f"[ERROR] Device reset failed with code {result.returncode}")
    except Exception as e:
        print(f"[EXCEPTION] Exception during device reset: {e}")
    else:
        print("Device reset completed.\n")


def run_test(test_name: str):
    """Run a specific pytest test with live output."""
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


def generate_perf_json(
    decoder_csv="out.csv", dispatch_csv="out-non-dispatched.csv", output_json="aggregated_perf_targets.json"
):
    """Load CSVs and generate aggregated performance target JSON."""
    print("Generating aggregated performance target JSON...")
    df_main = pd.read_csv(decoder_csv)
    df_dispatch = pd.read_csv(dispatch_csv)

    def is_collective(op):
        return any(keyword in op for keyword in ["AllGather", "ReduceScatter", "AllReduce", "Matmul_RS"])

    decoder_ops = set(df_main[df_main["metric_type"].str.startswith("mid_")]["op_code_with_id"].unique())
    model_tail_ops = set(df_main[df_main["metric_type"].str.startswith("model_tail")]["op_code_with_id"].unique())

    decoder_df = df_main[df_main["op_code_with_id"].isin(decoder_ops)]
    model_tail_df = df_main[df_main["op_code_with_id"].isin(model_tail_ops)]

    def aggregate_metrics(df, kernel_metric, dispatch_metric, ftls_metric):
        df_filtered = df[df["metric_type"].isin([kernel_metric, dispatch_metric, ftls_metric])]
        pivot = df_filtered.pivot_table(
            index=["run_number", "op_code_with_id"], columns="metric_type", values="value"
        ).reset_index()
        return pivot.groupby("op_code_with_id").mean()

    decoder_kernel_trace = aggregate_metrics(
        decoder_df, "mid_trace_avg", "mid_dispatch_avg", "mid_first_to_last_start_avg"
    )
    decoder_kernel_compile = aggregate_metrics(
        decoder_df, "mid_compile_avg", "mid_dispatch_avg", "mid_first_to_last_start_avg"
    )
    decoder_ftls = aggregate_metrics(decoder_df, "mid_trace_avg", "mid_dispatch_avg", "mid_first_to_last_start_avg")

    decoder_ops_combined = {}
    for op in sorted(decoder_kernel_trace.index.union(decoder_kernel_compile.index)):
        kernel_src = decoder_kernel_trace if is_collective(op) else decoder_kernel_compile
        kernel_duration = kernel_src.at[op, "mid_trace_avg" if is_collective(op) else "mid_compile_avg"]
        dispatch_duration = decoder_ftls.at[op, "mid_dispatch_avg"]
        ftls_duration = decoder_ftls.at[op, "mid_first_to_last_start_avg"]

        non_overlap_dispatch = df_dispatch[
            (df_dispatch["metric_type"] == "mid_dispatch_avg") & (df_dispatch["op_code_with_id"] == op)
        ]["value"].mean()

        decoder_ops_combined[op] = {
            "op_name": op,
            "kernel_duration": kernel_duration,
            "op_to_op": dispatch_duration,
            "first_to_last_start": ftls_duration,
            "non-overlapped-dispatch-time": non_overlap_dispatch,
            "kernel_duration_relative_margin": 0.05,
            "op_to_op_duration_relative_margin": 0.2,
            "first_to_last_start_relative_margin": 0.2,
            "dispatch_duration_relative_margin": 0.2,
        }

    model_tail_avg = model_tail_df[
        model_tail_df["metric_type"].isin(["model_tail_compile_avg", "model_tail_trace_avg", "model_tail_dispatch_avg"])
    ]
    pivot_tail = model_tail_avg.pivot_table(
        index=["run_number", "op_code_with_id"], columns="metric_type", values="value"
    ).reset_index()
    tail_grouped = pivot_tail.groupby("op_code_with_id").mean()
    tail_dispatch = (
        df_dispatch[df_dispatch["metric_type"] == "model_tail_dispatch_avg"].groupby("op_code_with_id")["value"].mean()
    )

    model_tail_combined = {}
    for op in sorted(tail_grouped.index):
        kernel_duration = tail_grouped.at[op, "model_tail_trace_avg" if is_collective(op) else "model_tail_compile_avg"]
        dispatch_duration = tail_grouped.at[op, "model_tail_dispatch_avg"]
        non_overlap_dispatch = tail_dispatch.get(op, 0)

        model_tail_combined[op] = {
            "op_name": op,
            "kernel_duration": kernel_duration,
            "op_to_op": dispatch_duration,
            "non-overlapped-dispatch-time": non_overlap_dispatch,
            "kernel_duration_relative_margin": 0.2,
            "op_to_op_duration_relative_margin": 0.2,
            "dispatch_duration_relative_margin": 0.5,
        }

    output_json_data = {"decoder": decoder_ops_combined, "model_tail": model_tail_combined}

    with open(output_json, "w") as f:
        json.dump(output_json_data, f, indent=4)

    print(f"Saved aggregated metrics to {output_json}")


def automate_perf_collection(
    num_runs: int = 5,
    generate_json: bool = True,
):
    """Main automation loop for running perf data collection and generating JSON."""
    decoder_test = "models/demos/llama3_subdevices/tests/test_decoder_device_perf.py::test_llama_TG_perf_device"
    dispatch_test = "models/demos/llama3_subdevices/tests/test_decoder_device_perf.py::test_llama_TG_perf_device_non_overlapped_dispatch"

    for i in range(1, num_runs + 1):
        print(f"\n===== Run {i}/{num_runs} =====")

        reset_device()
        print(f"Sleeping 20 seconds to allow hardware reset...")
        time.sleep(20)

        run_test(decoder_test)

        reset_device()
        print(f"Sleeping 20 seconds to allow hardware reset...")
        time.sleep(20)

        run_test(dispatch_test)

    if generate_json:
        generate_perf_json()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Automate LLaMA performance data collection")
    parser.add_argument("--runs", type=int, default=5, help="Number of times to repeat each test")
    parser.add_argument("--no-json", action="store_true", help="Skip generating JSON")
    args = parser.parse_args()

    automate_perf_collection(num_runs=args.runs, generate_json=not args.no_json)
