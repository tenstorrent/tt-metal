import pandas as pd
from collections import defaultdict
import os

# List of CSV files to process
csv_files = [
    "output_log_hypot_bcast.csv",
    "output_log_xlogy_bcast.csv",
    "output_log_minimum_bcast.csv",
    "output_log_maximum_bcast.csv",
    "output_log_atan2_bcast.csv",
    "output_log_nextafter_bcast.csv",
    "output_log_addalpha_bcast.csv",
    "output_log_subalpha_bcast.csv",
    "output_log_isclose_bcast.csv",
    "output_log_remainder_bcast.csv",
    "output_log_fmod_bcast.csv",
    "output_log_div_bcast.csv",
    "output_log_div_no_nan_bcast.csv",
    "output_log_scatter_bcast.csv",
    "output_log_outer_bcast.csv",
    "output_log_gcd_bcast.csv",
    "output_log_lcm_bcast.csv",
]

ARCH = os.getenv("ARCH_NAME")
WH_BH = "WH" if ARCH == "wormhole_b0" else "BH"
# Folder path
folder = f"/home/ubuntu/tt-metal/binary_ng_{WH_BH}/binary_ng_{WH_BH}_full"
folder_reshaped = f"/home/ubuntu/tt-metal/binary_ng_{WH_BH}/binary_ng_{WH_BH}_full_reshaped"

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
    "ttnn.int32 - ttnn.int32",
]

for csv_file in csv_files:
    input_csv = os.path.join(folder, csv_file)
    df = pd.read_csv(input_csv, header=None)

    result_map = defaultdict(dict)

    for _, row in df.iterrows():
        input_shape_a = row[0]
        input_shape_b = row[1]
        dtype_a, dtype_b = row[2], row[3]
        a_mem, b_mem = row[4], row[5]
        status = "PASS" if row[6] == "True" or row[6] == True else "FAIL"
        value = row[7]
        key = f"{dtype_a} - {dtype_b}"

        result_map[(input_shape_a, input_shape_b, a_mem, b_mem)][key] = (status, value)

    output_rows = []
    for (input_shape_a, input_shape_b, a_mem, b_mem), dtype_results in result_map.items():
        row_data = [input_shape_a, input_shape_b, a_mem, b_mem]
        for dtype_combo in dtype_combos:
            if dtype_combo in dtype_results:
                status, value = dtype_results[dtype_combo]
            else:
                status, value = "MISSING", ""
            row_data.extend([dtype_combo, status, value])
        output_rows.append(row_data)

    headers = ["input_shape_a", "input_shape_b", "a_mem_config", "b_mem_config"]
    for dtype_combo in dtype_combos:
        headers.extend([f"{dtype_combo}", "pass/fail", "value"])

    output_df = pd.DataFrame(output_rows, columns=headers)

    output_name = f"reshaped_{csv_file}"
    os.makedirs(folder_reshaped, exist_ok=True)
    output_path = os.path.join(folder_reshaped, output_name)
    output_df.to_csv(output_path, index=False)

    print(f"✅ Reshaped CSV saved to: {output_path}")
