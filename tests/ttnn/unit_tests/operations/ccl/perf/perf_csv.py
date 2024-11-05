# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pandas as pd
import os
import re


def perf_report(original_file_path):
    df = pd.read_csv(original_file_path)
    filtered_df = df[
        [
            "ATTRIBUTES",
            "INPUT_0_W",
            "INPUT_0_Z",
            "INPUT_0_Y",
            "INPUT_0_X",
            "OUTPUT_0_W",
            "OUTPUT_0_Z",
            "OUTPUT_0_Y",
            "OUTPUT_0_X",
            "INPUT_0_LAYOUT",
            "INPUT_0_DATATYPE",
            "OP CODE",
            "HOST DURATION [ns]",
            "DEVICE FW START CYCLE",
            "DEVICE FW END CYCLE",
            "OP TO OP LATENCY [ns]",
            "DEVICE FW DURATION [ns]",
            "DEVICE KERNEL DURATION [ns]",
            "DEVICE ERISC KERNEL DURATION [ns]",
        ]
    ].copy()

    fields_to_remove = [
        "receiver_device_id",
        "ring_index",
        "sender_device_id",
        "user_defined_num_buffers_per_channel",
        "user_defined_num_workers",
        "output_mem_config",
        "buffer_type",
        "shard_spec",
    ]

    def clean_attributes(attributes):
        attributes_list = attributes.split(";")
        filtered_attributes = [
            attr.strip() for attr in attributes_list if not any(field in attr for field in fields_to_remove)
        ]
        return "; ".join(filtered_attributes).strip("; ")

    filtered_df["ATTRIBUTES"] = filtered_df["ATTRIBUTES"].apply(clean_attributes)
    filtered_df["ATTRIBUTES_BACKUP"] = filtered_df["ATTRIBUTES"]

    filtered_df["Input Shape"] = filtered_df.apply(
        lambda row: f"[{row['INPUT_0_W']}, {row['INPUT_0_Z']}, {row['INPUT_0_Y']}, {row['INPUT_0_X']}]", axis=1
    )
    filtered_df["Output Shape"] = filtered_df.apply(
        lambda row: f"[{row['OUTPUT_0_W']}, {row['OUTPUT_0_Z']}, {row['OUTPUT_0_Y']}, {row['OUTPUT_0_X']}]", axis=1
    )
    filtered_df["Cycles Count"] = filtered_df["DEVICE FW END CYCLE"] - filtered_df["DEVICE FW START CYCLE"]

    def split_attributes(attributes):
        attr_dict = {key: value for key, value in re.findall(r"'([^']+)':\s*'([^']+)'", attributes)}
        attr_dict["topology"] = attr_dict.get("topology", "").split("::")[-1]
        if "ring_size" not in attr_dict:
            raise KeyError("Missing 'ring_size' attribute")
        attr_dict["n_chips"] = int(attr_dict["ring_size"])
        return pd.Series(
            {
                "dim": attr_dict.get("dim"),
                "num_links": attr_dict.get("num_links"),
                "topology": attr_dict["topology"],
                "n_chips": attr_dict["n_chips"],
            }
        )

    split_attrs_df = filtered_df["ATTRIBUTES"].apply(split_attributes)
    filtered_df = pd.concat([filtered_df, split_attrs_df], axis=1)
    filtered_df.drop(
        columns=[
            "ATTRIBUTES",
            "INPUT_0_W",
            "INPUT_0_Z",
            "INPUT_0_Y",
            "INPUT_0_X",
            "OUTPUT_0_W",
            "OUTPUT_0_Z",
            "OUTPUT_0_Y",
            "OUTPUT_0_X",
        ],
        inplace=True,
    )
    filtered_df.rename(columns={"INPUT_0_LAYOUT": "Layout", "INPUT_0_DATATYPE": "Data Type"}, inplace=True)

    numeric_columns = [
        "HOST DURATION [ns]",
        "Cycles Count",
        "OP TO OP LATENCY [ns]",
        "DEVICE FW DURATION [ns]",
        "DEVICE KERNEL DURATION [ns]",
        "DEVICE ERISC KERNEL DURATION [ns]",
    ]
    filtered_df[numeric_columns] = filtered_df[numeric_columns].apply(pd.to_numeric, errors="coerce")

    def calculate_min_avg_max_by_common_runs(df):
        group_columns = [
            "ATTRIBUTES_BACKUP",
            "Input Shape",
            "Output Shape",
            "dim",
            "num_links",
            "topology",
            "Layout",
            "Data Type",
            "OP CODE",
        ]
        results = []

        for name, group in df.groupby(group_columns):
            n_chips = group["n_chips"].iloc[0]
            group_excluded = group.iloc[2 * n_chips :]
            if not group_excluded.empty:
                min_values = group_excluded[numeric_columns].min().round(2)
                avg_values = group_excluded[numeric_columns].mean().round(2)
                max_values = group_excluded[numeric_columns].max().round(2)

                op_bw_values, link_bw_values = [], []

                for _, row in group_excluded.iterrows():
                    op_bw, link_bw = calculate_bandwidth(row)
                    op_bw_values.append(op_bw)
                    link_bw_values.append(link_bw)

                op_bw_min, op_bw_avg, op_bw_max = (
                    round(min(op_bw_values), 2),
                    round(sum(op_bw_values) / len(op_bw_values), 2),
                    round(max(op_bw_values), 2),
                )
                link_bw_min, link_bw_avg, link_bw_max = (
                    round(min(link_bw_values), 2),
                    round(sum(link_bw_values) / len(link_bw_values), 2),
                    round(max(link_bw_values), 2),
                )

                result_row = {
                    **{col: f"{min_values[col]} - {avg_values[col]} - {max_values[col]}" for col in numeric_columns},
                    "Op BW [GB/s]": f"{op_bw_min} - {op_bw_avg} - {op_bw_max}",
                    "Link BW [GB/s]": f"{link_bw_min} - {link_bw_avg} - {link_bw_max}",
                    **{key: value for key, value in zip(group_columns, name)},
                    "n_chips": n_chips,
                }
                results.append(result_row)

        return pd.DataFrame(results)

    def calculate_bandwidth(row):
        element_size = 2
        longest_device_fw_time = row["DEVICE FW DURATION [ns]"]
        longest_erisc_fw_time = row["DEVICE ERISC KERNEL DURATION [ns]"]

        input_tensor_volume = (
            int(row["Input Shape"].split(",")[0][1:])
            * int(row["Input Shape"].split(",")[1])
            * int(row["Input Shape"].split(",")[2])
            * int(row["Input Shape"].split(",")[3][:-1])
            * element_size
        )

        output_tensor_volume = (
            int(row["Output Shape"].split(",")[0][1:])
            * int(row["Output Shape"].split(",")[1])
            * int(row["Output Shape"].split(",")[2])
            * int(row["Output Shape"].split(",")[3][:-1])
            * element_size
        )

        op_bw, link_bw = None, None
        n_chips = row["n_chips"]
        if row["topology"] == "Ring":
            if row["OP CODE"] == "AllGather":
                op_bw = (output_tensor_volume * (n_chips - 1) / n_chips) / longest_device_fw_time
                link_bw = (output_tensor_volume * (n_chips - 1) / n_chips) / longest_erisc_fw_time
            elif row["OP CODE"] == "ReduceScatter":
                op_bw = (input_tensor_volume / n_chips) / longest_device_fw_time
                link_bw = (input_tensor_volume * (n_chips - 1) / n_chips) / longest_erisc_fw_time
        elif row["topology"] == "Linear":
            if row["OP CODE"] == "AllGather":
                op_bw = input_tensor_volume * n_chips / longest_device_fw_time
                link_bw = input_tensor_volume * (n_chips - 1) / longest_erisc_fw_time
            elif row["OP CODE"] == "ReduceScatter":
                op_bw = input_tensor_volume / longest_device_fw_time
                link_bw = input_tensor_volume * (n_chips - 1) / n_chips / longest_erisc_fw_time
        return round(op_bw, 2), round(link_bw, 2)

    average_values_by_common_run = calculate_min_avg_max_by_common_runs(filtered_df)

    final_order = [
        "Input Shape",
        "OP CODE",
        "dim",
        "num_links",
        "topology",
        "Layout",
        "Data Type",
        "HOST DURATION [ns]",
        "Cycles Count",
        "OP TO OP LATENCY [ns]",
        "DEVICE FW DURATION [ns]",
        "DEVICE KERNEL DURATION [ns]",
        "Op BW [GB/s]",
        "Link BW [GB/s]",
    ]
    average_values_by_common_run = average_values_by_common_run[final_order]

    base, ext = os.path.splitext(original_file_path)
    modified_file_path = f"{base}_modified{ext}"
    filtered_df.to_csv(modified_file_path, index=False)
    print(f"Filtered CSV created successfully at: {modified_file_path}")

    averages_file_path = f"{base}_averages.csv"
    average_values_by_common_run.to_csv(averages_file_path, index=False)
    print(f"Averages CSV created successfully at: {averages_file_path}")

    return average_values_by_common_run
