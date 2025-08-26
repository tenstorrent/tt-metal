# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from config_utils import load_sweep_data, get_best_config_with_storage_precedence

# Filter by specific dtype-fidelity pairs with their plot colors
dtype_configs = [
    ("BFLOAT4_B_LoFi", "#1f77b4"),
    ("BFLOAT8_B_HiFi2", "#2ca02c"),
    ("BFLOAT16_HiFi4", "#ff7f0e"),
]

# Load sweep data
df = load_sweep_data()

# Generate plots for each source
for source in ["n150", "p150"]:
    plt.figure(figsize=(16, 12))

    for dtype_fidelity, color in dtype_configs:
        traced_perf = []
        nontraced_perf = []
        df_slice = df[(df["source"] == source) & (df["dtype_fidelity"] == dtype_fidelity)]

        for matrix_size in sorted(df_slice["matrix_elements"].unique()):
            size_group = df_slice[df_slice["matrix_elements"] == matrix_size]

            traced_group = size_group[size_group["use_trace"] == True]
            best_traced = get_best_config_with_storage_precedence(traced_group)
            if not best_traced.empty:
                traced_perf.append((matrix_size, best_traced["tflops"]))

            nontraced_group = size_group[size_group["use_trace"] == False]
            best_nontraced = get_best_config_with_storage_precedence(nontraced_group)
            if not best_nontraced.empty:
                nontraced_perf.append((matrix_size, best_nontraced["tflops"]))

        traced_x, traced_y = zip(*traced_perf) if traced_perf else ([], [])
        nontraced_x, nontraced_y = zip(*nontraced_perf) if nontraced_perf else ([], [])

        plt.plot(traced_x, traced_y, color=color, marker=">", linestyle="-", label=f"{dtype_fidelity} (Traced)")
        plt.scatter(traced_x, traced_y, color=color, marker=">", s=60)
        plt.plot(
            nontraced_x, nontraced_y, color=color, marker="<", linestyle="--", label=f"{dtype_fidelity} (Non-traced)"
        )
        plt.scatter(nontraced_x, nontraced_y, color=color, marker="<", s=60)

        for x, y_trace, y_nontrace in zip(traced_x, traced_y, nontraced_y):
            improvement = ((y_trace - y_nontrace) / y_nontrace) * 100
            plt.annotate(
                f"{improvement:+.1f}%",
                (x, max(y_trace, y_nontrace)),
                xytext=(0, 15),
                textcoords="offset points",
                ha="center",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", alpha=0.8),
            )

    plt.xscale("log")
    plt.grid(True, which="both", linestyle="--", alpha=0.7)
    plt.xlabel("Matrix Elements")
    plt.ylabel("TFLOPs")
    plt.title(
        f"Traced vs Non-traced Performance ({source.upper()})\nTraced execution shows consistent performance improvements"
    )
    plt.legend(bbox_to_anchor=(0.5, -0.04), loc="upper center", ncol=3)
    plt.tight_layout()
    plt.savefig(f"tech_reports/GEMM_FLOPS/images/trace_comparison_{source}.png", bbox_inches="tight")
    plt.close()

print("Tracing comparison plots saved!")
