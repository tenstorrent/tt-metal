import pandas as pd
from collections import defaultdict
import os


# List of CSV files to process
csv_files = [
    "output_log_pow_test.csv",
]


# Folder path
folder = "/home/ubuntu/tt-metal/binary_ng_BH/binary_ng_BH_full"
folder_reshaped = "/home/ubuntu/tt-metal/binary_ng_BH/binary_ng_BH_full_reshaped"


# Define all dtype combinations to track
dtype_combos = [
    "ttnn.bfloat16 - ttnn.bfloat16",
    "ttnn.bfloat16 - ttnn.float32",
    "ttnn.bfloat16 - ttnn.bfloat8_b",
    "ttnn.bfloat16 - ttnn.bfloat4_b",
    "ttnn.float32 - ttnn.float32",
    "ttnn.float32 - ttnn.bfloat16",
    "ttnn.float32 - ttnn.bfloat8_b",
    "ttnn.float32 - ttnn.bfloat4_b",
    "ttnn.bfloat8_b - ttnn.bfloat8_b",
    "ttnn.bfloat8_b - ttnn.float32",
    "ttnn.bfloat8_b - ttnn.bfloat16",
    "ttnn.bfloat8_b - ttnn.bfloat4_b",
    "ttnn.bfloat4_b - ttnn.float32",
    "ttnn.bfloat4_b - ttnn.bfloat4_b",
    "ttnn.bfloat4_b - ttnn.bfloat16",
    "ttnn.bfloat4_b - ttnn.bfloat8_b",
    "ttnn.bfloat16 - none",
    "ttnn.float32 - none",
    "ttnn.bfloat8_b - none",
    "ttnn.bfloat4_b - none",
]


for csv_file in csv_files:
    input_csv = os.path.join(folder, csv_file)
    df = pd.read_csv(input_csv, header=None)

    result_map = defaultdict(dict)

    for _, row in df.iterrows():
        a_mem, b_mem = row[2], row[3]
        dtype_a, dtype_b = row[0], row[1]
        status = "PASS" if row[4] == "True" or row[4] == True else "FAIL"
        value = row[5]
        key = f"{dtype_a} - {dtype_b}"
        result_map[(a_mem, b_mem)][key] = (status, value)

    output_rows = []
    for (a_mem, b_mem), dtype_results in result_map.items():
        row_data = [a_mem, b_mem]
        for dtype_combo in dtype_combos:
            if dtype_combo in dtype_results:
                status, value = dtype_results[dtype_combo]
            else:
                status, value = "MISSING", ""
            row_data.extend([dtype_combo, status, value])
        output_rows.append(row_data)

    headers = ["a_mem_config", "b_mem_config"]
    for dtype_combo in dtype_combos:
        headers.extend([f"{dtype_combo}", "pass/fail", "value"])

    output_df = pd.DataFrame(output_rows, columns=headers)

    output_name = f"reshaped_{csv_file}"
    output_path = os.path.join(folder_reshaped, output_name)
    output_df.to_csv(output_path, index=False)

    print(f"✅ Reshaped CSV saved to: {output_path}")
