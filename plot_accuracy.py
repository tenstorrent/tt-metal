import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.patches as mpatches
from matplotlib.ticker import ScalarFormatter, PercentFormatter


import pandas as pd
import seaborn as sns
import numpy as np

import math
import sys
import os.path

plt.rcParams["svg.fonttype"] = "none"  # Make text editing in SVG easier


def load_csv(filename):
    return pd.read_csv(filename, sep=",", index_col=False, skipinitialspace=True)


def plot_accuracy_op(data, dest, opname, op_dtype, ax_fun=None):
    data["max_rel_error"] *= 100
    data["mean_rel_error"] *= 100

    # data = data[(data["base_x"] == 1.0) | (data["base_x"] == 2.0) ]
    # print(f"DATA = \n{data}")

    data_mlt = pd.melt(
        data,
        id_vars=["base_x", "operation"],
        value_vars=["max_rel_error", "mean_rel_error"],
        var_name="type",
        value_name="val",
    )

    # We want 'max' and 'meman' instead of 'max_real_error' and 'mean_rel_error'
    data_mlt["type"] = data_mlt["type"].replace({"max_rel_error": "max", "mean_rel_error": "mean"})

    # print(f"DATA = \n{data_mlt}")

    fig, ax = plt.subplots(figsize=(25, 15))

    data_max = data_mlt[data_mlt["type"] == "max"]
    data_mean = data_mlt[data_mlt["type"] == "mean"]

    data_max["operation"] = data_max["operation"] + " (max)"
    data_mean["operation"] = data_mean["operation"] + " (mean)"

    plt.axhline(y=0, color="k", linewidth=2)

    sns.lineplot(data=data_max, x="base_x", y="val", hue="operation", ax=ax)

    sns.lineplot(data=data_mean, x="base_x", y="val", hue="operation", ax=ax, linestyle="--")

    ax.set_xscale("symlog", base=2)
    ax.set_yscale("asinh")
    ax.set_xlabel("X")

    def symlog_format(x, pos):
        if abs(x) < 10:
            return f"int(x)" if x.is_integer() else f"{x:.1f}"
        else:
            return f"{x:.2f}"

    ax.xaxis.set_major_locator(ticker.SymmetricalLogLocator(linthresh=1, base=2))

    ax.set_title(f"Relative error of ttnn.{opname} against torch implementation\non {op_dtype}")

    plt.gca().xaxis.set_major_formatter(ScalarFormatter(useOffset=False, useMathText=False))
    plt.gca().yaxis.set_major_formatter(ScalarFormatter(useOffset=False, useMathText=False))

    custom_ticks = [0, 0.5, 1, 5, 10, 50, 100]
    plt.gca().set_yticks(custom_ticks)
    plt.gca().set_yticklabels([f"{x}%" for x in custom_ticks])  # For decimal form

    ax.set_ylabel("Error vs. torch [%]\n(lower is better)")
    # ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"$2^{{{int(np.log2(x))}}}$" if x > 0 else "0"))

    if ax_fun is not None:
        ax_fun(ax)

    plt.legend(ncol=2)

    plt.savefig(f"{dest}.pdf", bbox_inches="tight", pad_inches=0.0)
    plt.savefig(f"{dest}.png", bbox_inches="tight", pad_inches=0.0)

    plt.close()


def plot_values(data, dest, opname, op_dtype, ax_fun=None):
    # Reference values (torch.<OP>) are present for each entry.
    # e.g.
    # | ttnn.exp(1)        | torch.exp(1)
    # | ttnn.exp-approx(1) | torch.exp(1)
    # | ttnn.exp(2)        | torch.exp(2)
    # | ttnn.exp-approx(2) | torch.exp(2)
    # But what we want is:
    # | ttnn.exp(1)
    # | ttnn.exp-approx(1)
    # | torch.exp(1)
    # | ttnn.exp(2)
    # | ttnn.exp-approx(2)
    # | torch.approx(2)

    # To achieve this, we extra 'base_yref' columns and then concatenate it
    # We remove duplicate values with groupby/min (probably not necessary)
    data_ref = data.groupby(["base_x"], sort=False).min(numeric_only=True).reset_index()
    data_ref["base_y"] = data_ref["base_yref"]
    data_ref["operation"] = "torch"
    data_ref["max_rel_error"] = 0.0
    data_ref["mean_rel_error"] = 0.0

    data = pd.concat([data, data_ref], axis=0).reset_index()
    data = data[(data["base_x"].notna()) & (np.isfinite(data["base_x"]))]

    # print(f"DATA[melt] =\n{data_mlt}")
    fig, ax = plt.subplots(figsize=(25, 15))

    sns.lineplot(data=data, x="base_x", y="base_y", hue="operation", ax=ax)

    # ax.set_xscale('symlog')
    # ax.set_yscale('symlog', linthresh=10)

    # ax.set_xlim(-10e2, 10e2)
    # ax.set_ylim(-0.5, )

    # ax.legend([f"ttnn.{opname}(x)", f"torch.{opname}(x)"])

    ax.set_xlabel("X")
    ax.set_ylabel(f"{opname}(x)  [{op_dtype}]")

    if ax_fun is not None:
        ax_fun(ax)

    plt.savefig(f"{dest}.png", bbox_inches="tight", pad_inches=0.0)
    plt.savefig(f"{dest}.pdf", bbox_inches="tight", pad_inches=0.0)

    plt.close()

    pass


def main():
    sns.set(
        style="ticks",
        rc={
            "axes.grid": True,
            "axes.edgecolor": "black",
            "axes.titlesize": 60,
            "axes.labelsize": 60,
            "xtick.labelsize": 60,
            "ytick.labelsize": 60,
            "font.size": 60,
            "legend.fontsize": 30,
            "lines.linewidth": 4,
            "axes.linewidth": 4,
            "axes.edgecolor": "black",
            "font.serif": ["Latin Modern Math"],
        },
    )

    data_exp_csv = load_csv("accuracy-exp-float32.csv")
    data_log_csv = load_csv("accuracy-log.csv")
    data_exp_approx_csv = load_csv("accuracy-exp-approx-float32.csv")
    data_exp_torch_f32 = load_csv("accuracy-exp-float32-torch.csv")
    data_exp_torch_bf16 = load_csv("accuracy-exp-bfloat16-torch.csv")

    data_exp_bf16 = load_csv("accuracy-exp-bfloat16.csv")
    data_exp_approx_bf16 = load_csv("accuracy-exp-approx-bfloat16.csv")
    data_log_bf16 = load_csv("accuracy-log-bfloat16.csv")
    data_silu_bf16 = load_csv("accuracy-silu-bfloat16.csv")
    data_tan_bf16 = load_csv("accuracy-tan-bfloat16.csv")

    data_exp_csv["operation"] = "exp"
    data_exp_approx_csv["operation"] = "exp-approx"
    data_exp_torch_f32["operation"] = "torch.exp"
    data_log_csv["operation"] = "log"

    data_exp_bf16["operation"] = "exp"
    data_exp_approx_bf16["operation"] = "exp-approx"
    data_exp_torch_bf16["operation"] = "torch.exp"
    data_log_bf16["operation"] = "log"
    data_silu_bf16["operation"] = "silu"

    data_all_exp = pd.concat([data_exp_csv, data_exp_approx_csv])
    data_all_exp_bf16 = pd.concat([data_exp_bf16, data_exp_approx_bf16])

    # plot_accuracy(data_exp_csv, "accuracy-exp", "exp")
    # plot_accuracy(data_log_csv, "accuracy-log", "log")
    # plot_accuracy(data_exp_approx_csv, "accuracy-exp-approx", "exp[approx]")

    if True:
        plot_values(
            data_all_exp_bf16,
            "exp[bf16]",
            "exp",
            "bfloat16",
            ax_fun=lambda ax: (ax.set_xlim(-4.0, 8.0), ax.set_ylim(-1, math.exp(8.05))),
        )

        plot_values(
            data_log_bf16,
            "log[bf16]",
            "log",
            "bfloat16",
            ax_fun=lambda ax: (ax.set_xlim(-4.0, 8.0), ax.set_ylim(-101, 8)),
        )

        plot_values(
            data_silu_bf16,
            "silu[bf16]",
            "silu",
            "bfloat16",
            ax_fun=lambda ax: (
                ax.set_xlim(-8.0, 4.0),
                ax.set_ylim(-1.0, 4.0),
                ax.axvline(x=-5.0, color="k", linestyle="--", label="x=-5", linewidth=3),
                plt.text(-5 + 0.2, 3, "x=-5", fontsize=40),
            ),
        )

        plot_values(
            data_tan_bf16,
            "tan[bf16]",
            "tan",
            "bfloat16",
            ax_fun=lambda ax: (
                ax.set_xlim(255.0, 257.0),
                ax.set_ylim(-0, 30),
            ),
        )

    if True:
        plot_accuracy_op(data_all_exp, "accuracy-all-exp-float32", "exp", "float32")
        plot_accuracy_op(data_log_csv, "accuracy-all-log-float32", "log", "float32")

        plot_accuracy_op(data_all_exp_bf16, "accuracy-all-exp-bfloat16", "exp", "bfloat16")
        plot_accuracy_op(
            data_silu_bf16, "accuracy-silu-bfloat16", "silu", "bfloat16", ax_fun=lambda ax: ax.set_xlim(-128, 32)
        )

        def override_tan_zoom_plot(ax):
            xmax = 24
            ax.set_xlim(0, xmax)
            ax.set_xscale("linear")
            kpi = math.pi / 2
            k = 0
            while kpi < xmax:
                ax.axvline(x=kpi, color="k", linewidth=3, linestyle="--")
                plt.text(kpi + 0.1, 50, f"{2*k+1}pi/2", fontsize=30)

                k += 1
                kpi = k * math.pi + math.pi / 2.0

        plot_accuracy_op(
            data_tan_bf16,
            "accuracy-all-tan-bfloat16",
            "tan",
            "bfloat16",
        )

    if False:

        def ax_fun_torch_f32(ax):
            ax.set_yscale("log")
            custom_ticks = [0, 1e-9, 1e-8, 1e-6, 1e-5, 1e-4, 1e-3]
            plt.gca().set_yticks(custom_ticks)
            ax.set_ylim(-1e8, 1e-3)
            ax.set_ylabel("Erorr vs. math.exp [%]")
            ax.set_title("Relative error of torch.exp vs. math.exp\n on float32")
            # plt.gca().set_yticklabels([f"{x/100:.0%}" for x in custom_ticks])  # For decimal form

        def ax_fun_torch_bf16(ax):
            ax.set_yscale("log")
            custom_ticks = [0, 1e-9, 1e-8, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2]
            plt.gca().set_yticks(custom_ticks)
            ax.set_ylim(-1e8, 1e-2)
            ax.set_ylabel("Erorr vs. math.exp [%]")
            ax.set_title("Relative error of torch.exp vs. math.exp\n on bfloat16")
            # ax.set_ylim()

        plot_accuracy_op(
            data_exp_torch_f32, "accuracy-torch-exp-float32", "exp-torch", "float32", ax_fun=ax_fun_torch_f32
        )

        plot_accuracy_op(
            data_exp_torch_f32, "accuracy-torch-exp-bfloat16", "exp-torch", "bfloat16", ax_fun=ax_fun_torch_bf16
        )


main()
