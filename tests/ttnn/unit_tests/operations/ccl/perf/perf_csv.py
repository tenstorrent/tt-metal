# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import pandas as pd
import os
from tabulate import tabulate
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
            "PM BANDWIDTH [ns]",
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
        filtered_attributes = []
        for attr in attributes_list:
            attr = attr.strip()
            if not any(field in attr for field in fields_to_remove):
                filtered_attributes.append(attr)
        return "; ".join(filtered_attributes).strip("; ")

    filtered_df["ATTRIBUTES"] = filtered_df["ATTRIBUTES"].apply(clean_attributes)

    filtered_df["Input Shape"] = filtered_df.apply(
        lambda row: f"[{row['INPUT_0_W']}, {row['INPUT_0_Z']}, {row['INPUT_0_Y']}, {row['INPUT_0_X']}]", axis=1
    )

    filtered_df["Cycles Count"] = filtered_df["DEVICE FW END CYCLE"] - filtered_df["DEVICE FW START CYCLE"]

    def split_attributes(attributes):
        attr_dict = {}
        matches = re.findall(r"'([^']+)':\s*'([^']+)'", attributes)
        for key, value in matches:
            attr_dict[key.strip()] = value.strip()

        topology = attr_dict.get("topology", None)
        if topology:
            topology = topology.split("::")[-1]
        attr_dict["topology"] = topology

        return pd.Series(
            {
                "dim": attr_dict.get("dim", None),
                "num_links": attr_dict.get("num_links", None),
                "topology": attr_dict.get("topology", None),
            }
        )

    split_attrs_df = filtered_df["ATTRIBUTES"].apply(split_attributes)

    filtered_df = pd.concat([filtered_df, split_attrs_df], axis=1)

    filtered_df.drop(columns=["ATTRIBUTES", "INPUT_0_W", "INPUT_0_Z", "INPUT_0_Y", "INPUT_0_X"], inplace=True)

    filtered_df.rename(columns={"INPUT_0_LAYOUT": "Layout", "INPUT_0_DATATYPE": "Data Type"}, inplace=True)

    new_order = [
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
        "PM BANDWIDTH [ns]",
    ]

    filtered_df = filtered_df[new_order]

    numeric_columns = [
        "HOST DURATION [ns]",
        "Cycles Count",
        "OP TO OP LATENCY [ns]",
        "DEVICE FW DURATION [ns]",
        "DEVICE KERNEL DURATION [ns]",
        "PM BANDWIDTH [ns]",
    ]

    filtered_df[numeric_columns] = filtered_df[numeric_columns].apply(pd.to_numeric, errors="coerce")

    avg_values = filtered_df[numeric_columns].iloc[4:].mean().round(2).tolist()

    avg_row = pd.Series(["Average", None, None, None, None, None] + avg_values, index=filtered_df.columns)

    filtered_df = pd.concat([filtered_df, avg_row.to_frame().T], ignore_index=True)

    avg_df = avg_row.to_frame().T
    avg_df.columns = filtered_df.columns

    base, ext = os.path.splitext(original_file_path)
    modified_file_path = f"{base}_modified{ext}"

    filtered_df.to_csv(modified_file_path, index=False)

    print(f"Filtered CSV created successfully at: {modified_file_path}")

    return avg_df
