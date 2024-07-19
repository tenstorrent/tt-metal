# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pandas as pd
import argparse

DRAM_INTERLEAVED = "{'buffer_type': 'DRAM'; 'memory_layout': 'INTERLEAVED'}"


def assign_op_id_cross_device(df, num_chips):
    df.loc[:, "OP ID CROSS DEVICE"] = df["GLOBAL CALL COUNT"].apply(lambda x: x // num_chips)

    return df


def group_by_op_all_devices(df):
    # Split the DataFrame into AllGather and the rest
    df1 = df[df["OP CODE"] != "tt::tt_metal::AllGather"]
    df2 = df[df["OP CODE"] == "tt::tt_metal::AllGather"]

    # Perform the groupby operation separately on each DataFrame
    # We want to aggregate AllGather by the min and the rest by the max duration
    grouped_df1 = df1.groupby("OP ID CROSS DEVICE").max()
    grouped_df2 = df2.groupby("OP ID CROSS DEVICE").min()

    # Concatenate the results
    grouped_df = pd.concat([grouped_df1, grouped_df2])

    # Sort by global call id again
    grouped_df = grouped_df.sort_values(by="GLOBAL CALL COUNT", ascending=True)

    grouped_df = grouped_df.reset_index()

    return grouped_df


def analyze_perf_csv(
    csv_file, layers, show_all, out_file=None, num_chips=1, seq=32, group=False, remove_warmup_runs=False
):
    df = pd.read_csv(csv_file)
    # find index of the first non-warmup run
    # and remove all warmup runs
    # index of the first non-warmup run is latest occurrence of "tt::tt_metal::Embeddings" - 7
    if remove_warmup_runs:
        warmup_run_idx = df[df["OP CODE"] == "tt::tt_metal::Embeddings"].index[-1] - 7
        df = df.iloc[warmup_run_idx:]
        # save to intermed file for further analysis
        df.to_csv(out_file.split(".")[0] + ".warmup_removed.csv", index=False, sep=",")
    df = df[df["DEVICE FW DURATION [ns]"] != "-"]
    df["DEVICE FW DURATION [ns]"] = df["DEVICE FW DURATION [ns]"].fillna(0).astype(int)

    df = assign_op_id_cross_device(df, num_chips)
    sorted_df = df
    sum_duration = df["DEVICE FW DURATION [ns]"].sum()

    matmul_rows = sorted_df[sorted_df["OP CODE"].str.contains("Matmul")]

    matmul_rows.loc[:, "bytes"] = (
        matmul_rows["INPUT_1_W"] * matmul_rows["INPUT_1_Z"] * matmul_rows["INPUT_1_Y"] * matmul_rows["INPUT_1_X"]
    )
    matmul_rows.loc[:, "flops"] = 2 * matmul_rows["INPUT_0_Y"] * matmul_rows["INPUT_0_X"] * matmul_rows["OUTPUT_0_X"]
    matmul_rows["GB/s"] = matmul_rows["bytes"] / matmul_rows["DEVICE FW DURATION [ns]"]
    matmul_rows["TFLOP/s"] = matmul_rows["flops"] / matmul_rows["DEVICE FW DURATION [ns]"] / 1000
    matmul_rows["% DRAM (240)"] = 100 * matmul_rows["GB/s"] / 240  # Peak expected WH bandwidth
    matmul_rows["% FPU (82)"] = 100 * matmul_rows["TFLOP/s"] / 82  # Peak theoretical FP16 FPU performance
    matmul_rows["% TIME"] = 100 * matmul_rows["DEVICE FW DURATION [ns]"] / sum_duration
    # matmul_rows["% TIME SUM"] = matmul_rows["% TIME"].cumsum()
    matmul_sum_duration = matmul_rows["DEVICE FW DURATION [ns]"].sum()

    # shorten some column names
    matmul_rows.rename(columns={"DEVICE FW DURATION [ns]": "DURATION [ns]"}, inplace=True)
    sorted_df.rename(columns={"DEVICE FW DURATION [ns]": "DURATION [ns]"}, inplace=True)

    data_type = {"BFLOAT16": 2, "BFLOAT8_B": 1}
    sorted_df[["INPUT_0_DATATYPE", "INPUT_1_DATATYPE", "OUTPUT_0_DATATYPE"]] = (
        sorted_df[["INPUT_0_DATATYPE", "INPUT_1_DATATYPE", "OUTPUT_0_DATATYPE"]]
        .applymap(data_type.get)
        .fillna(0)
        .astype(int)
    )

    sorted_df.loc[:, "TOTAL_BYTES"] = (
        (sorted_df["INPUT_0_MEMORY"].str.contains(DRAM_INTERLEAVED, na=False)).fillna(0).astype(int)
        * (
            sorted_df["INPUT_0_W"]
            * sorted_df["INPUT_0_Z"]
            * sorted_df["INPUT_0_Y"]
            * sorted_df["INPUT_0_X"]
            * sorted_df["INPUT_0_DATATYPE"]
        )
        + (sorted_df["INPUT_1_MEMORY"].str.contains(DRAM_INTERLEAVED, na=False)).fillna(0).astype(int)
        * (
            sorted_df["INPUT_1_W"]
            * sorted_df["INPUT_1_Z"]
            * sorted_df["INPUT_1_Y"]
            * sorted_df["INPUT_1_X"]
            * sorted_df["INPUT_1_DATATYPE"]
        )
        + (sorted_df["OUTPUT_0_MEMORY"].str.contains(DRAM_INTERLEAVED, na=False)).fillna(0).astype(int)
        * (
            sorted_df["OUTPUT_0_W"]
            * sorted_df["OUTPUT_0_Z"]
            * sorted_df["OUTPUT_0_Y"]
            * sorted_df["OUTPUT_0_X"]
            * sorted_df["OUTPUT_0_DATATYPE"]
        )
    )

    sorted_df.loc[:, "flops"] = 2 * sorted_df["INPUT_0_Y"] * sorted_df["INPUT_0_X"] * sorted_df["OUTPUT_0_X"]
    sorted_df["DRAM BW GB/s"] = sorted_df["TOTAL_BYTES"] * (1000**3) / (sorted_df["DURATION [ns]"] * (1024**3))
    sorted_df["TFLOP/s"] = sorted_df["flops"] / sorted_df["DURATION [ns]"] / 1000
    sorted_df["% DRAM (240)"] = 100 * sorted_df["DRAM BW GB/s"] / 240  # Peak expected WH bandwidth
    sorted_df["% FPU (82)"] = 100 * sorted_df["TFLOP/s"] / 82  # Peak theoretical FP16 FPU performance
    sorted_df["TOTAL MBs"] = sorted_df["TOTAL_BYTES"] / 1024 / 1024
    selected_columns = [
        "GLOBAL CALL COUNT",
        "OP ID CROSS DEVICE",
        "OP CODE",
        "% TIME",
        # "% TIME SUM",
        "% DRAM (240)",
        "% FPU (82)",
        "DURATION [ns]",
        "GB/s",
        "TFLOP/s",
        "CORE COUNT",
        "INPUT_0_Y",
        "INPUT_0_X",
        "INPUT_1_Y",
        "INPUT_1_X",
        "OUTPUT_0_Y",
        "OUTPUT_0_X",
    ]

    if show_all:
        selected_columns = [
            "GLOBAL CALL COUNT",
            "OP ID CROSS DEVICE",
            "OP CODE",
            "% TIME",
            # "% TIME SUM",
            "DURATION [ns]",
            "% DEV CB WAIT",
            # "DURATION SUM [ns]",
            "DRAM BW GB/s",
            "% DRAM (240)",
            "CORE COUNT",
            "INPUT_0_Y",
            "INPUT_0_X",
            "INPUT_1_Y",
            "INPUT_1_X",
            "OUTPUT_0_Y",
            "OUTPUT_0_X",
            "TOTAL MBs",
            "DEVICE COMPUTE CB WAIT FRONT [ns]",
        ]
        sorted_df["% TIME"] = 100 * sorted_df["DURATION [ns]"] / sum_duration
        sorted_df["% DEV CB WAIT"] = (
            100 * sorted_df["DEVICE COMPUTE CB WAIT FRONT [ns]"].fillna(0).astype(int) / sorted_df["DURATION [ns]"]
        )

        # trim all floats to 6 decimal places
        sorted_df = sorted_df.round(6)
        selected_df = sorted_df[selected_columns]

        # save to file for further analysis
        if out_file:
            selected_df.to_csv(out_file, index=False, sep="\t")

        if group:
            grouped_df = group_by_op_all_devices(selected_df)

            sum_duration = grouped_df["DURATION [ns]"].sum()
            grouped_df["% TIME"] = 100 * grouped_df["DURATION [ns]"] / sum_duration
            grouped_df["% DEV CB WAIT"] = (
                100
                * grouped_df["DEVICE COMPUTE CB WAIT FRONT [ns]"].fillna(0).astype(int)
                / grouped_df["DURATION [ns]"]
            )

            grouped_matmul_rows = group_by_op_all_devices(matmul_rows)
            matmul_sum_duration = grouped_matmul_rows["DURATION [ns]"].sum()

            if out_file:
                grouped_df.to_csv(out_file.split(".")[0] + ".grouped.csv", index=False, sep="\t")

    if layers:
        tokens_per_sec_user = 1000000000 / sum_duration / layers
        tokens_per_sec = seq * tokens_per_sec_user
        print(f"Layer ms: {sum_duration / 1000000:.8f} ({matmul_sum_duration / sum_duration:.2%} matmul)")
        print(f"Inferences/sec: {tokens_per_sec_user:.8f}")
        print(f"Tokens/sec: {tokens_per_sec:.8f}")


def main():
    parser = argparse.ArgumentParser(description="Analyze perf CSV file")
    parser.add_argument(
        "-a", "--all", action="store_true", help="List ops in the CSV file - by default only matmul ops are shown."
    )
    parser.add_argument("-l", "--layers", type=int, help="Number of layers to extrapolate perf results up to.")
    parser.add_argument(
        "csv_file", type=str, help="Path to the perf CSV file from tt-metal for a single decoder layer."
    )
    parser.add_argument(
        "-o", "--out_file", required=False, type=str, help="Path to the output file for further analysis."
    )

    parser.add_argument(
        "--num-chips", type=int, default=1, help="Number of chips running model in parallel mode. Default is 1."
    )

    parser.add_argument("--remove-warmup", action="store_true", help="Remove warmup runs from the CSV file.")
    parser.add_argument("--seq", type=int, default=32, help="Sequence length or number of users.")

    parser.add_argument("-g", "--group", action="store_true", help="Group by ops running in parallel on all devices.")

    args = parser.parse_args()

    analyze_perf_csv(
        args.csv_file,
        layers=args.layers,
        show_all=args.all,
        out_file=args.out_file,
        num_chips=args.num_chips,
        seq=args.seq,
        group=args.group,
        remove_warmup_runs=args.remove_warmup,
    )


if __name__ == "__main__":
    main()
