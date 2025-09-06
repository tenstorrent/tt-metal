# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors
from config_utils import load_sweep_data, get_best_config_with_storage_precedence

# Filter by specific dtype-fidelity pairs with their plot colors
dtype_configs = [
    ("BFLOAT4_B_LoFi", "#1f77b4"),
    ("BFLOAT8_B_HiFi2", "#2ca02c"),
    ("BFLOAT16_HiFi4", "#ff7f0e"),
]

# Load sweep data
df = load_sweep_data()

# Increase width for better horizontal ratio
plt.figure(figsize=(16, 12))

for dtype_fidelity, color in dtype_configs:
    # --- p150 ---
    selected_rows_p150 = []
    group_p150 = df[(df["dtype_fidelity"] == dtype_fidelity) & (df["source"] == "p150")]
    for matrix_elements in sorted(group_p150["matrix_elements"].unique()):
        size_group = group_p150[group_p150["matrix_elements"] == matrix_elements]
        best_config = get_best_config_with_storage_precedence(size_group)
        if not best_config.empty:
            selected_rows_p150.append(best_config)

    if not selected_rows_p150:
        continue

    selected_df_p150 = pd.DataFrame(selected_rows_p150).sort_values("matrix_elements")
    p150_x_cutoff = selected_df_p150["matrix_elements"].max()

    # --- n150, cutoff at p150 max x ---
    selected_rows_n150 = []
    group_n150 = df[(df["dtype_fidelity"] == dtype_fidelity) & (df["source"] == "n150")]
    for matrix_elements in sorted(group_n150["matrix_elements"].unique()):
        if matrix_elements > p150_x_cutoff:
            continue
        size_group = group_n150[group_n150["matrix_elements"] == matrix_elements]
        best_config = get_best_config_with_storage_precedence(size_group)
        if not best_config.empty:
            selected_rows_n150.append(best_config)

    # Plot p150 (solid line, > marker)
    plt.plot(
        selected_df_p150["matrix_elements"],
        selected_df_p150["tflops"],
        color=color,
        alpha=0.7,
        linewidth=1.5,
        marker=">",
        linestyle="-",
        label=f"{dtype_fidelity} (p150)",
    )
    plt.scatter(
        selected_df_p150["matrix_elements"],
        selected_df_p150["tflops"],
        color=color,
        marker=">",
        s=60,
        alpha=0.8,
        zorder=10,
    )

    # Plot n150 (dashed line, < marker)
    if selected_rows_n150:
        selected_df_n150 = pd.DataFrame(selected_rows_n150).sort_values("matrix_elements")
        plt.plot(
            selected_df_n150["matrix_elements"],
            selected_df_n150["tflops"],
            color=color,
            alpha=0.7,
            linewidth=1.5,
            marker="<",
            linestyle="--",
            label=f"{dtype_fidelity} (n150)",
        )
        plt.scatter(
            selected_df_n150["matrix_elements"],
            selected_df_n150["tflops"],
            color=color,
            marker="<",
            s=60,
            alpha=0.8,
            zorder=10,
        )

plt.xscale("log")
plt.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)
plt.xlabel("Matrix Elements")
plt.ylabel("TFLOPs")
plt.title("TFLOPs vs Matrix Size (n150 vs p150)")
plt.legend(title="DType_Fidelity (Source)", loc="upper center", bbox_to_anchor=(0.5, -0.04), ncol=3, fontsize=9)
plt.tight_layout()
plt.savefig("tech_reports/GEMM_FLOPS/images/flops_vs_matrix_elements_comparison.png", bbox_inches="tight")
plt.close()

print("Performance scatter plot saved!")
