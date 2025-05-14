import pandas as pd
from collections import defaultdict
import os
import glob

ARCH = os.getenv("ARCH_NAME")
WH_BH = "WH" if ARCH == "wormhole_b0" else "BH"

# Folder path
folder = f"/home/ubuntu/tt-metal/binary_ng_{WH_BH}/binary_ng_{WH_BH}_full"
folder_reshaped = f"/home/ubuntu/tt-metal/binary_ng_{WH_BH}/binary_ng_{WH_BH}_full_reshaped"
os.makedirs(folder_reshaped, exist_ok=True)

# Get all CSV files from the folder
csv_files = [os.path.basename(f) for f in glob.glob(os.path.join(folder, "*.csv"))]

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
