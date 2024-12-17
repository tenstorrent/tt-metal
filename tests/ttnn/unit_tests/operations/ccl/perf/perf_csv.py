# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pandas as pd
import os
import re
import time
import shutil


def perf_report(file_path):
    df = pd.read_csv(file_path)

    df = df.dropna(subset=["DEVICE ERISC KERNEL DURATION [ns]"])
    df = df[df["OP TO OP LATENCY [ns]"] != 0]
    df = df[df["METAL TRACE ID"].notna() & (df["METAL TRACE ID"] != "")]

    def remove_keys_from_attributes(attributes):
        attributes = attributes.replace(";", ",").replace("'", '"')

        keys_to_remove = ["receiver_device_id", "ring_index", "sender_device_id"]

        try:
            attributes_dict = eval(attributes)

            attributes_dict["topology"] = attributes_dict.get("topology", "").split("::")[-1]

            if "ring_size" not in attributes_dict:
                raise KeyError("Missing 'ring_size' attribute")

            attributes_dict["n_chips"] = int(attributes_dict["ring_size"])

            for key in keys_to_remove:
                if key in attributes_dict:
                    del attributes_dict[key]

            modified_attributes = str(attributes_dict).replace(",", ";").replace('"', "'")
            return modified_attributes
        except Exception as e:
            print(f"Error processing attributes: {e}")
            return attributes

    df["ATTRIBUTES"] = df["ATTRIBUTES"].apply(remove_keys_from_attributes)

    def safe_parse_attributes(attributes):
        attributes = attributes.replace(";", ",")

        try:
            attr_dict = eval(attributes)
            return attr_dict
        except Exception as e:
            print(f"Error processing attributes: {e}")
            return {}

    df["topology"] = df["ATTRIBUTES"].apply(
        lambda x: safe_parse_attributes(x).get("topology", "") if isinstance(safe_parse_attributes(x), dict) else ""
    )

    df["dim"] = df["ATTRIBUTES"].apply(
        lambda x: safe_parse_attributes(x).get("dim", safe_parse_attributes(x).get("scatter_dim", ""))
        if isinstance(safe_parse_attributes(x), dict)
        else ""
    )

    df["num_links"] = df["ATTRIBUTES"].apply(
        lambda x: safe_parse_attributes(x).get("num_links", "") if isinstance(safe_parse_attributes(x), dict) else ""
    )

    df["output_mem_config"] = df["ATTRIBUTES"].apply(
        lambda x: ", ".join(
            [
                match.split("::")[1]
                for match in re.findall(
                    r"(BufferType::\w+|TensorMemoryLayout::\w+)",
                    str(safe_parse_attributes(x).get("output_mem_config", "")),
                )
            ]
        )
        if isinstance(safe_parse_attributes(x), dict)
        else ""
    )

    df["n_chips"] = df["ATTRIBUTES"].apply(
        lambda x: int(safe_parse_attributes(x).get("ring_size", ""))
        if isinstance(safe_parse_attributes(x), dict)
        else 0
    )

    group_columns = [
        "ATTRIBUTES",
        "INPUT_0_W",
        "INPUT_0_Z",
        "INPUT_0_Y",
        "INPUT_0_X",
        "INPUT_0_LAYOUT",
        "INPUT_0_DATATYPE",
        "OUTPUT_0_W",
        "OUTPUT_0_Z",
        "OUTPUT_0_Y",
        "OUTPUT_0_X",
        "OUTPUT_0_LAYOUT",
        "OUTPUT_0_DATATYPE",
    ]

    grouped = df.groupby(group_columns)

    numeric_columns = [
        "HOST DURATION [ns]",
        "Cycles Count",
        "OP TO OP LATENCY [ns]",
        "DEVICE FW DURATION [ns]",
        "DEVICE KERNEL DURATION [ns]",
        "DEVICE ERISC KERNEL DURATION [ns]",
        "Op BW [GB/s]",
        "Link BW [GB/s]",
    ]

    averages_data = []

    def calculate_bandwidth(row):
        dtype = row["Data Type"]
        dtype_sizes = {
            "BFLOAT16": 2,
            "BFLOAT8_B": 1.0625,  # ((1024 + 64) / 1024)
            "BFLOAT4_B": 0.5625,  # ((1024/2 + 64) / 1024)
            "FLOAT32": 4,
            "UINT8": 1,
            "UINT16": 2,
            "INT32": 4,
            "UINT32": 4,
        }
        element_size = dtype_sizes.get(dtype)

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
                op_bw = input_tensor_volume / longest_device_fw_time
                link_bw = input_tensor_volume / longest_erisc_fw_time
        elif row["topology"] == "Linear":
            if row["OP CODE"] == "AllGather":
                op_bw = input_tensor_volume * n_chips / longest_device_fw_time
                link_bw = input_tensor_volume * (n_chips - 1) / longest_erisc_fw_time
            elif row["OP CODE"] == "ReduceScatter":
                op_bw = input_tensor_volume / longest_device_fw_time
                link_bw = input_tensor_volume / longest_erisc_fw_time
        return round(op_bw, 2), round(link_bw, 2)

    utilization_data = []

    for i, (group, group_df) in enumerate(grouped, start=1):
        group_df = group_df.iloc[2 * group_df["n_chips"].iloc[0] :]

        group_df = group_df.sort_values(by=["DEVICE ID", "OP TO OP LATENCY [ns]"]).reset_index(drop=True)
        group_df = group_df.groupby("DEVICE ID").apply(lambda x: x.iloc[4:-4]).reset_index(drop=True)

        group_df.rename(columns={"INPUT_0_LAYOUT": "Layout", "INPUT_0_DATATYPE": "Data Type"}, inplace=True)

        group_df["Input Shape"] = group_df.apply(
            lambda row: f"[{int(row['INPUT_0_W'])}, {int(row['INPUT_0_Z'])}, {int(row['INPUT_0_Y'])}, {int(row['INPUT_0_X'])}]",
            axis=1,
        )
        group_df["Output Shape"] = group_df.apply(
            lambda row: f"[{int(row['OUTPUT_0_W'])}, {int(row['OUTPUT_0_Z'])}, {int(row['OUTPUT_0_Y'])}, {int(row['OUTPUT_0_X'])}]",
            axis=1,
        )
        group_df["Cycles Count"] = group_df["DEVICE FW END CYCLE"] - group_df["DEVICE FW START CYCLE"]
        group_df[["Op BW [GB/s]", "Link BW [GB/s]"]] = group_df.apply(calculate_bandwidth, axis=1, result_type="expand")

        group_file_path = file_path.replace(".csv", f"_group_{i}.csv")

        group_df.to_csv(group_file_path, index=False)

        group_data = {
            "Input Shape": group_df["Input Shape"].iloc[0],
            "OP CODE": group_df["OP CODE"].iloc[0],
            "dim": group_df["dim"].iloc[0] if "dim" in group_df else "",
            "num_links": group_df["num_links"].iloc[0] if "num_links" in group_df else "",
            "output_mem_config": group_df["output_mem_config"].iloc[0] if "output_mem_config" in group_df else "",
            "topology": group_df["topology"].iloc[0],
            "Layout": group_df["Layout"].iloc[0] if "Layout" in group_df else "",
            "Data Type": group_df["Data Type"].iloc[0] if "Data Type" in group_df else "",
        }

        for column in numeric_columns:
            min_val = round(group_df[column].min(), 2)
            largest_vals = group_df[column].nlargest(3)
            max_val = round(largest_vals.iloc[-1], 2)
            if min_val == max_val:
                avg_val = min_val
            else:
                avg_val = round(group_df[column][~group_df[column].isin(largest_vals.head(2))].mean(), 2)

            group_data[column] = f"{min_val} - {avg_val} - {max_val}"

        link_bw_str = group_data["Link BW [GB/s]"]
        link_bw_values = [float(x.strip()) for x in link_bw_str.split(" - ")]

        link_bw_min, link_bw_avg, link_bw_max = link_bw_values

        link_bw = 12.5 if group_df["topology"].iloc[0] == "Linear" else 25.0  # GB/s
        num_links = int(group_df["num_links"].iloc[0]) if "num_links" in group_df else 0

        theoretical_peak = num_links * link_bw
        min_utilization = round((link_bw_min / theoretical_peak) * 100, 2)
        avg_utilization = round((link_bw_avg / theoretical_peak) * 100, 2)
        max_utilization = round((link_bw_max / theoretical_peak) * 100, 2)

        utilization_summary = {
            "Input Shape": group_df["Input Shape"].iloc[0],
            "OP CODE": group_df["OP CODE"].iloc[0],
            "dim": group_df["dim"].iloc[0] if "dim" in group_df else "",
            "num_links": group_df["num_links"].iloc[0] if "num_links" in group_df else "",
            "output_mem_config": group_df["output_mem_config"].iloc[0] if "output_mem_config" in group_df else "",
            "topology": group_df["topology"].iloc[0],
            "Layout": group_df["Layout"].iloc[0] if "Layout" in group_df else "",
            "Data Type": group_df["Data Type"].iloc[0] if "Data Type" in group_df else "",
            "Link BW [GB/s]": f"{group_df['Link BW [GB/s]'].min()} - {group_df['Link BW [GB/s]'].mean():.2f} - {group_df['Link BW [GB/s]'].max()}",
            "Theoretical Peak [GB/s]": f"{theoretical_peak}",
            "Utilization [%]": f"{min_utilization} - {avg_utilization} - {max_utilization}",
        }
        utilization_data.append(utilization_summary)
        averages_data.append(group_data)

    utilization_df = pd.DataFrame(utilization_data)
    averages_df = pd.DataFrame(averages_data)
    op_code = averages_df.iloc[0]["OP CODE"]

    today = time.strftime("%Y_%m_%d")
    if op_code == "AllGather":
        ccl_perf_file_path = f"CCL_all_gather_Perf_{today}.csv"
    elif op_code == "ReduceScatter":
        ccl_perf_file_path = f"CCL_reduce_scatter_Perf_{today}.csv"
    else:
        ccl_perf_file_path = f"CCL_Perf_{today}.csv"
    utilization_file_path = file_path.replace(".csv", f"_utilization_{i}.csv")

    shutil.copy(file_path, ccl_perf_file_path)

    logs_dir = "generated/profiler/.logs"
    os.makedirs(logs_dir, exist_ok=True)
    shutil.copy(ccl_perf_file_path, logs_dir)

    averages_df.to_csv(ccl_perf_file_path, index=False)
    utilization_df.to_csv(utilization_file_path, index=False)

    print(f"CCL Perf report CSV saved to: {ccl_perf_file_path}")
    print(f"CCL Perf Utilization report CSV saved to: {utilization_file_path}")

    return averages_df, utilization_df
