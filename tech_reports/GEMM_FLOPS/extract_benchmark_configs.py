# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

"""
Extract best configurations from test_benchmark.py and create data for plotting scripts.
This script reads the optimized configurations and generates the data needed for
OOB vs best config and tracing vs non-tracing comparisons.
"""

import sys
import os

sys.path.append("tests/ttnn/unit_tests/benchmarks")

# Import the configurations from test_benchmark.py
from test_benchmark import (
    matmul_shapes_n150_bfloat16,
    matmul_shapes_n150_bfloat8_b,
    matmul_shapes_n150_bfloat4_b,
    matmul_shapes_p150_bfloat16,
    matmul_shapes_p150_bfloat8_b,
    matmul_shapes_p150_bfloat4_b,
)

import pandas as pd


def create_best_config_data():
    """Create dataframes with the best configurations from test_benchmark.py"""

    # Define the configuration mappings
    configs = {
        "n150": {
            "BFLOAT16_HiFi4": matmul_shapes_n150_bfloat16,
            "BFLOAT8_B_HiFi2": matmul_shapes_n150_bfloat8_b,
            "BFLOAT4_B_LoFi": matmul_shapes_n150_bfloat4_b,
        },
        "p150": {
            "BFLOAT16_HiFi4": matmul_shapes_p150_bfloat16,
            "BFLOAT8_B_HiFi2": matmul_shapes_p150_bfloat8_b,
            "BFLOAT4_B_LoFi": matmul_shapes_p150_bfloat4_b,
        },
    }

    # Grid sizes for scaling
    grid_sizes = {"n150": (8, 8), "p150": (13, 10)}  # x, y  # x, y

    best_config_rows = []

    for device, device_configs in configs.items():
        grid_x, grid_y = grid_sizes[device]

        for dtype_fidelity, shapes in device_configs.items():
            for shape_config in shapes:
                (
                    m_base,
                    k_base,
                    n_base,
                    in0_sharded,
                    out_sharded,
                    in0_block_w_div,
                    num_out_blocks_h,
                    num_out_blocks_w,
                ) = shape_config

                # Scale by grid size (as done in the benchmark)
                m = m_base * grid_y
                k = k_base * grid_x
                n = n_base * grid_x

                # Calculate matrix elements
                matrix_elements = m * k * n

                # Create entries for both traced and non-traced
                for use_trace in [True, False]:
                    row = {
                        "source": device,
                        "dtype_fidelity": dtype_fidelity,
                        "dtype": f"DataType.{dtype_fidelity.split('_')[0]}",
                        "math_fidelity": f"MathFidelity.{dtype_fidelity.split('_')[-1]}",
                        "use_trace": use_trace,
                        "m": m,
                        "k": k,
                        "n": n,
                        "matrix_elements": matrix_elements,
                        "in0_sharded": in0_sharded,
                        "out_sharded": out_sharded,
                        "in0_block_w_div": in0_block_w_div,
                        "num_out_blocks_h": num_out_blocks_h,
                        "num_out_blocks_w": num_out_blocks_w,
                        # Storage types based on sharding
                        "in0_storage_type": "L1" if in0_sharded else "DRAM",
                        "in1_storage_type": "DRAM",  # Always DRAM
                        "out_storage_type": "L1" if out_sharded else "DRAM",
                        "config_type": "best_config",  # Mark as best config
                    }
                    best_config_rows.append(row)

    return pd.DataFrame(best_config_rows)


def create_oob_config_data():
    """Create dataframes with OOB (out-of-box) configurations"""

    # OOB configs: DRAM,DRAM,DRAM with minimal settings (1,1,1,1,1)
    oob_shapes = [
        (64, 64, 64),
        (64, 128, 128),
        (64, 128, 256),
        (128, 128, 128),
        (128, 128, 256),
        (128, 256, 256),
        (256, 256, 256),
        (256, 256, 384),
        (256, 384, 384),
        (384, 384, 384),
        (384, 384, 512),
        (384, 512, 512),
        (512, 512, 512),
    ]

    # Grid sizes for scaling
    grid_sizes = {"n150": (8, 8), "p150": (13, 10)}  # x, y  # x, y

    dtype_fidelities = ["BFLOAT16_HiFi2", "BFLOAT8_B_LoFi", "BFLOAT4_B_LoFi"]

    oob_config_rows = []

    for device in ["n150", "p150"]:
        grid_x, grid_y = grid_sizes[device]

        for dtype_fidelity in dtype_fidelities:
            for m_base, k_base, n_base in oob_shapes:
                # Scale by grid size
                m = m_base * grid_y
                k = k_base * grid_x
                n = n_base * grid_x

                # Calculate matrix elements
                matrix_elements = m * k * n

                # Create entries for both traced and non-traced
                for use_trace in [True, False]:
                    row = {
                        "source": device,
                        "dtype_fidelity": dtype_fidelity,
                        "dtype": f"DataType.{dtype_fidelity.split('_')[0]}",
                        "math_fidelity": f"MathFidelity.{dtype_fidelity.split('_')[-1]}",
                        "use_trace": use_trace,
                        "m": m,
                        "k": k,
                        "n": n,
                        "matrix_elements": matrix_elements,
                        "in0_sharded": False,  # OOB: no sharding
                        "out_sharded": False,  # OOB: no sharding
                        "in0_block_w_div": 1,  # OOB: minimal settings
                        "num_out_blocks_h": 1,  # OOB: minimal settings
                        "num_out_blocks_w": 1,  # OOB: minimal settings
                        # Storage types: all DRAM for OOB
                        "in0_storage_type": "DRAM",
                        "in1_storage_type": "DRAM",
                        "out_storage_type": "DRAM",
                        "config_type": "oob",  # Mark as OOB config
                    }
                    oob_config_rows.append(row)

    return pd.DataFrame(oob_config_rows)


if __name__ == "__main__":
    print("Extracting best configurations from test_benchmark.py...")

    # Create best config data
    best_df = create_best_config_data()
    print(f"Created {len(best_df)} best configuration entries")

    # Create OOB config data
    oob_df = create_oob_config_data()
    print(f"Created {len(oob_df)} OOB configuration entries")

    # Save to files for the plotting scripts to use
    best_df.to_csv("tech_reports/GEMM_FLOPS/best_configs.csv", index=False)
    oob_df.to_csv("tech_reports/GEMM_FLOPS/oob_configs.csv", index=False)

    print("Saved best_configs.csv and oob_configs.csv")
    print("\nBest config sample:")
    print(best_df.head())
    print("\nOOB config sample:")
    print(oob_df.head())
