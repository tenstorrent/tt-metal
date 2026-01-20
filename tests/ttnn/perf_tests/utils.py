# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pandas as pd
import re


def extract_ops_between_signposts(csv_path, op_name):
    """
    Extract operation durations between signpost pairs from a Tracy CSV file.

    Parses a Tracy profiler CSV file and extracts operation durations that occur between
    "test-name-start" and "test-name-end" signpost markers. The test name is extracted
    from the signpost OP CODE.

    Args:
        csv_path (str): Path to the Tracy profiler CSV file.
        op_name (str): Name of the operation to extract (e.g., "Pool2D", "Conv2dDeviceOperation").

    Returns:
        dict: Dictionary mapping test names to lists of durations in nanoseconds.
              Format: {test_name: [duration_ns, ...]}

    Example:
        results = extract_ops_between_signposts("ops_perf_results.csv", "Pool2D")
        # Returns: {"resnet50_maxpool_hs": [12345.0, 12456.0], "vgg16_maxpool": [23456.0]}
    """
    df = pd.read_csv(csv_path)
    results = {}
    current_region = None

    for _, row in df.iterrows():
        if row["OP TYPE"] == "signpost":
            op_code = row["OP CODE"]
            if op_code.endswith("-start"):
                current_region = op_code[:-6]  # Remove "-start" suffix to get test_name
            elif op_code.endswith("-end"):
                current_region = None
        elif current_region:
            op_code = row["OP CODE"]
            # Match op_name as a complete word, even if part of a larger string (e.g.,
            # "MeshDeviceOperationAdapter<ttnn::operations::conv::conv2d::Conv2dDeviceOperation>"
            # which is the current format possibly prone to changes)
            if pd.notna(op_code) and re.search(rf"\b{re.escape(op_name)}(?:<|$|\W)", op_code):
                if current_region not in results:
                    results[current_region] = []
                results[current_region].append(float(row["DEVICE KERNEL DURATION [ns]"]))

    return results
