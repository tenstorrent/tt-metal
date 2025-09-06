# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from config_utils import load_sweep_data, get_default_oob_config, get_best_config_vs_default

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
        df_slice = df[(df["source"] == source) & (df["dtype_fidelity"] == dtype_fidelity)]

        default_perf = []
        best_perf = []

        for matrix_size in sorted(df_slice["matrix_elements"].unique()):
            size_group = df_slice[df_slice["matrix_elements"] == matrix_size]

            default_config = get_default_oob_config(size_group)
            if not default_config.empty:
                default_perf.append((matrix_size, default_config["tflops"]))

                best_config = get_best_config_vs_default(size_group, default_config["tflops"])
                if not best_config.empty:
                    best_perf.append((matrix_size, best_config["tflops"]))

        default_x, default_y = zip(*default_perf) if default_perf else ([], [])
        best_x, best_y = zip(*best_perf) if best_perf else ([], [])

        plt.plot(best_x, best_y, color=color, marker=">", linestyle="-", label=f"{dtype_fidelity} (Best)")
        plt.scatter(best_x, best_y, color=color, marker=">", s=60)
        plt.plot(default_x, default_y, color=color, marker="<", linestyle="--", label=f"{dtype_fidelity} (Default)")
        plt.scatter(default_x, default_y, color=color, marker="<", s=60)

        for x, y_best, y_default in zip(best_x, best_y, default_y):
            if y_best != y_default:
                improvement = ((y_best - y_default) / y_default) * 100
                plt.annotate(
                    f"{improvement:+.1f}%",
                    (x, max(y_best, y_default)),
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
        f"Best vs Default Performance ({source.upper()})\nOptimized configs outperform default by 5-35% on average"
    )
    plt.legend(bbox_to_anchor=(0.5, -0.04), loc="upper center", ncol=3)

    plt.tight_layout()
    plt.savefig(f"tech_reports/GEMM_FLOPS/images/oob_comparison_{source}.png", bbox_inches="tight")
    plt.close()

print("OOB comparison plots saved!")
