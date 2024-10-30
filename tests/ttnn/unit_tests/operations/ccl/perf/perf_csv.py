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
            "INPUT_0_LAYOUT",
            "INPUT_0_DATATYPE",
            "HOST DURATION [ns]",
            "DEVICE FW START CYCLE",
            "DEVICE FW END CYCLE",
            "OP TO OP LATENCY [ns]",
            "DEVICE FW DURATION [ns]",
            "DEVICE KERNEL DURATION [ns]",
        ]
    ].copy()

    fields_to_remove = [
        "receiver_device_id",
        "ring_index",
        "ring_size",
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
    filtered_df["Cycles Count"] = filtered_df["DEVICE FW END CYCLE"] - filtered_df["DEVICE FW START CYCLE"]

    def split_attributes(attributes):
        attr_dict = {key: value for key, value in re.findall(r"'([^']+)':\s*'([^']+)'", attributes)}
        attr_dict["topology"] = attr_dict.get("topology", "").split("::")[-1]
        return pd.Series(
            {"dim": attr_dict.get("dim"), "num_links": attr_dict.get("num_links"), "topology": attr_dict["topology"]}
        )

    split_attrs_df = filtered_df["ATTRIBUTES"].apply(split_attributes)
    filtered_df = pd.concat([filtered_df, split_attrs_df], axis=1)
    filtered_df.drop(columns=["ATTRIBUTES", "INPUT_0_W", "INPUT_0_Z", "INPUT_0_Y", "INPUT_0_X"], inplace=True)

    filtered_df.rename(columns={"INPUT_0_LAYOUT": "Layout", "INPUT_0_DATATYPE": "Data Type"}, inplace=True)
    new_order = [
        "ATTRIBUTES_BACKUP",
        "Input Shape",
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
    ]
    filtered_df = filtered_df[new_order]

    numeric_columns = [
        "HOST DURATION [ns]",
        "Cycles Count",
        "OP TO OP LATENCY [ns]",
        "DEVICE FW DURATION [ns]",
        "DEVICE KERNEL DURATION [ns]",
    ]
    filtered_df[numeric_columns] = filtered_df[numeric_columns].apply(pd.to_numeric, errors="coerce")

    def calculate_min_avg_max_by_common_runs(df):
        group_columns = ["ATTRIBUTES_BACKUP", "Input Shape", "dim", "num_links", "topology", "Layout", "Data Type"]
        results = []

        for name, group in df.groupby(group_columns):
            group_excluded = group.iloc[4:]
            if not group_excluded.empty:
                min_values = group_excluded[numeric_columns].min().round(2)
                avg_values = group_excluded[numeric_columns].mean().round(2)
                max_values = group_excluded[numeric_columns].max().round(2)

                result_row = {
                    **{col: f"{min_values[col]} - {avg_values[col]} - {max_values[col]}" for col in numeric_columns},
                    **{key: value for key, value in zip(group_columns, name)},
                }
                results.append(result_row)

        return pd.DataFrame(results)

    average_values_by_common_run = calculate_min_avg_max_by_common_runs(filtered_df)

    print_order = ["Input Shape", "dim", "num_links", "topology", "Layout", "Data Type"] + numeric_columns
    average_values_by_common_run = average_values_by_common_run[print_order]

    filtered_df.drop(columns=["ATTRIBUTES_BACKUP"], inplace=True)

    base, ext = os.path.splitext(original_file_path)
    modified_file_path = f"{base}_modified{ext}"
    filtered_df.to_csv(modified_file_path, index=False)
    print(f"Filtered CSV created successfully at: {modified_file_path}")

    averages_file_path = f"{base}_averages.csv"
    average_values_by_common_run.to_csv(averages_file_path, index=False)
    print(f"Averages CSV created successfully at: {averages_file_path}")

    return average_values_by_common_run
