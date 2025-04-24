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


def plot_accuracy_op(data, dest, opname, op_dtype, plot_override_fun=None):
    data = data.copy()
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

    color_palette = sns.color_palette()
    sns.lineplot(data=data_max, x="base_x", y="val", hue="operation", ax=ax, palette=color_palette)

    sns.lineplot(
        data=data_mean,
        x="base_x",
        y="val",
        hue="operation",
        ax=ax,
        linestyle="--",
        palette=sns.color_palette(color_palette.as_hex(), desat=0.7),
    )

    ax.set_xscale("symlog", base=10)
    ax.set_yscale("asinh")
    ax.set_xlabel("X")

    def symlog_format(x, pos):
        if abs(x) < 10:
            return f"int(x)" if x.is_integer() else f"{x:.1f}"
        else:
            return f"{x:.2f}"

    # If extreme values are low, don't use scientific notation
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()

    # If extreme values are low, don't use scientific notation
    if xmax < 100:
        plt.gca().xaxis.set_major_formatter(ScalarFormatter(useOffset=False, useMathText=False))

    custom_ticks = [0, 0.5, 1, 5, 10, 50, 100]
    if ymax > 100:
        custom_ticks += [int(10**i) for i in range(3, int(min(10, math.log10(ymax))) + 1)]
        pass
    print(f"YMAX = {ymax}")
    print(f"CUSTOM TICKS = {custom_ticks}")

    ax.xaxis.set_major_locator(ticker.SymmetricalLogLocator(linthresh=1, base=10))

    ax.set_title(f"Relative error of ttnn.{opname} against torch implementation\[{op_dtype}]")

    plt.gca().yaxis.set_major_formatter(ScalarFormatter(useOffset=False, useMathText=False))

    plt.gca().set_yticks(custom_ticks)
    plt.gca().set_yticklabels([f"{x}%" for x in custom_ticks])  # For decimal form

    ax.set_ylabel("Error vs. torch [%]\n(lower is better)")

    def xticks_formatter(x, pos):
        if abs(x) >= 100:
            # Convert to power of 10 notation with LaTeX superscript
            exponent = int(math.log10(abs(x)))
            sign = "-" if x < 0 else ""
            return f"${sign}10^{exponent}$"
        else:
            return f"{x:.1f}" if not x.is_integer() else f"{int(x)}"

    ax.xaxis.set_major_formatter(ticker.FuncFormatter(xticks_formatter))

    # ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"$2^{{{int(np.log2(x))}}}$" if x > 0 else "0"))

    if plot_override_fun is not None:
        plot_override_fun(ax)

    plt.legend(ncol=2)

    plt.savefig(f"{dest}.pdf", bbox_inches="tight", pad_inches=0.0)
    plt.savefig(f"{dest}.png", bbox_inches="tight", pad_inches=0.0)

    plt.close()


def plot_values(data, dest, opname, op_dtype, plot_override_fun=None):
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

    if plot_override_fun is not None:
        plot_override_fun(ax)

    plt.savefig(f"{dest}.png", bbox_inches="tight", pad_inches=0.0)
    plt.savefig(f"{dest}.svg", bbox_inches="tight", pad_inches=0.0)

    plt.close()


class PlotExp:
    def override_accuracy(self, ax):
        pass

    def override_accuracy_zoom(self, ax):
        pass

    def override_values(self, ax):
        ax.set_xlim(-4.0, 8.0)
        ax.set_ylim(-1, math.exp(8.05))
        pass


class PlotLog:
    def __init__(self, base):
        self.base = base

    def override_accuracy(self, ax):
        ax.set_xscale("log", base=10)

        custom_ticks = [10e-30, 10e-17, 10e-4, 1, 10e9, 10e22, 10e35]

        base_str = f"{self.base}" if self.base != math.e else "e"

        # Found experimentally
        for k in [-64, -32, -16, 16, 32, 64]:
            xval = math.pow(self.base, k)
            ax.axvline(x=xval, color="k", linestyle="--")
            plt.text(xval * 1.05, 70, f"x={base_str}**{k}", fontsize=24)

        plt.gca().set_xticks(custom_ticks)

    def override_accuracy_zoom(self, ax):
        ax.set_xlim(1e-1, 100)
        ax.set_xscale("log", base=10)

        custom_yticks = [0, 0.5, 1, 5, 10, 50, 100, 500]

        custom_xticks = [1e-1, 0.5, 1.0, 2.0, 5, 10, 20, 100]
        plt.gca().set_xticks(custom_xticks)
        plt.gca().set_xticklabels([f"{x:g}" for x in custom_xticks])  # For decimal form

        plt.gca().set_yticks(custom_yticks)
        plt.gca().set_yticklabels([f"{x:g}%" for x in custom_yticks])  # For decimal form

        # custom_yticks = [0, 1, 5, 10, 50, 100, 500]
        # plt.gca().set_yticks(custom_yticks)
        # plt.gca().set_yticklabels([f"{x}%" for x in custom_yticks])  # For decimal form

    def override_values(self, ax):
        ax.set_xlim(-4.0, 8.0)
        ax.set_ylim(-101, 8)
        pass


class PlotSiLU:
    def override_accuracy(self, ax):
        ax.set_xlim(-128, 32)

    def override_accuracy_zoom(self, ax):
        pass

    def override_values(self, ax):
        ax.set_xlim(-8.0, 4.0),
        ax.set_ylim(-1.0, 4.0)
        ax.axvline(x=-5.0, color="k", linestyle="--", label="x=-5", linewidth=3)
        plt.text(-5 + 0.2, 3, "x=-5", fontsize=40)
        pass


class PlotCos:
    def override_accuracy(self, ax):
        ax.set_ylim(0, 100)
        pass

    def override_accuracy_zoom(self, ax):
        pass

    def override_values(self, ax):
        pass


class PlotTan:
    def override_accuracy(self, ax):
        pass

    def override_accuracy_zoom(self, ax):
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

    def override_values(self, ax):
        ax.set_xlim(255.0, 257.0)
        ax.set_ylim(-0, 30)
        pass


all_override_plot_funs = {
    # Exponential functions
    "exp": PlotExp(),
    "tanh": PlotExp(),  # reuse exp settings
    "cosh": PlotExp(),  # reuse exp settings
    "sinh": PlotExp(),  # reuse exp settings
    # Logarithmic functions
    "log": PlotLog(math.e),
    "log10": PlotLog(10),
    "log2": PlotLog(2),
    "log1p": None,
    # Activation functions
    "silu": PlotSiLU(),  # reuse silu settings
    "logit": PlotSiLU(),  # reuse silu settings
    "gelu": PlotSiLU(),  # reuse silu settings
    "swish": PlotSiLU(),  # reuse silu settings
    "mish": PlotSiLU(),  # reuse silu settings
    "elu": PlotSiLU(),  # reuse silu settings
    "selu": PlotSiLU(),  # reuse silu settings
    "softplus": PlotSiLU(),  # reuse silu settings
    "softsign": PlotSiLU(),  # reuse silu settings
    # Trigonometric functions
    "tan": PlotTan(),  # reuse tan settings
    "atan": None,
    "sin": PlotCos(),
    "cos": PlotCos(),
    # Miscellaneous functions
    "sqrt": None,
    "rsqrt": None,
    "rsqrt_approx": None,
    "digamma": None,
    "lgamma": None,
    "tanhshrink": None,
}


def plot_all_ops(accuracy_dir, ops_list, dest_dir):
    # This is not great because we have to define each parameter manually,
    # but this works for now.

    # Store all operations results in dictionary
    # This is inefficient but is more convenient for concatenating
    # results such as 'exp' and 'exp_approx'
    all_op_data = {}

    # Aggregate data
    for op, dtype, samples in ops_list:

        def should_regenerate(input_csv, output_file):
            script_path = __file__

            # Check if output files exist and get their modification times
            output_mtime = 0
            if os.path.exists(output_file):
                output_mtime = os.path.getmtime(output_file)

            input_mtime = 0
            if os.path.exists(input_csv):
                input_mtime = os.path.getmtime(input_csv)
            else:
                return False  # No data for given operation

            script_mtime = os.path.getmtime(script_path)

            # Regenerate if:
            # - Outputs don't exist (output_time == 0), or
            # - Input CSV is newer than outputs, or
            # - Script is newer than outputs
            return output_mtime == 0 or input_mtime > output_mtime or script_mtime > output_mtime

        input_csv = f"{accuracy_dir}/{op}-{dtype}-[{samples}].csv"
        output_file = f"{dest_dir}/{op}-{dtype}-[{samples}].pdf"

        if should_regenerate(input_csv, output_file):
            data = load_csv(input_csv)

            all_op_data[(op, dtype, samples)] = data

    # Concatenate exp and exp_approx (semi-generic)
    # (write both into all_op_data[exp] and remove all_op_data[exp_approx])
    for opkey in list(all_op_data.keys()):
        op, dtype, samples = opkey
        if op == "exp_approx":
            exp_approx_data = all_op_data[opkey]

            exp_data = None
            if ("exp", dtype, samples) in all_op_data:
                exp_data = all_op_data[("exp", dtype, samples)]

            all_op_data[("exp", dtype, samples)] = pd.concat([exp_data, exp_approx_data])
            del all_op_data[opkey]

    # Plot operations
    for op, dtype, samples in all_op_data.keys():
        print(f"Plotting {op} {dtype} {samples}\r")
        data = all_op_data[(op, dtype, samples)]

        # Each operation may have specific properties
        # To highlight these, custom functions have been defined
        override_plot_fun = None
        if all_override_plot_funs[op] is not None:
            override_plot_fun = all_override_plot_funs[op].override_accuracy

        plot_accuracy_op(data, f"{dest_dir}/{op}-{dtype}", op, dtype, override_plot_fun)


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

    all_operations = [
        ("exp", "bfloat16", 4),
        ("exp_approx", "bfloat16", 4),
        ("log", "bfloat16", 4),
        ("tanh", "bfloat16", 4),
        ("cosh", "bfloat16", 4),
        ("sinh", "bfloat16", 4),
        ("log10", "bfloat16", 4),
        ("log2", "bfloat16", 4),
        ("log1p", "bfloat16", 4),
        ("silu", "bfloat16", 4),
        ("gelu", "bfloat16", 4),
        ("logit", "bfloat16", 4),
        ("swish", "bfloat16", 4),
        ("mish", "bfloat16", 4),
        ("elu", "bfloat16", 4),
        ("selu", "bfloat16", 4),
        ("softplus", "bfloat16", 4),
        ("softsign", "bfloat16", 4),
        ("tan", "bfloat16", 4),
        ("atan", "bfloat16", 4),
        ("sin", "bfloat16", 4),
        ("cos", "bfloat16", 4),
        ("sqrt", "bfloat16", 4),
        ("rsqrt", "bfloat16", 4),
        ("rsqrt_approx", "bfloat16", 4),
        ("digamma", "bfloat16", 4),
        ("lgamma", "bfloat16", 4),
        ("tanhshrink", "bfloat16", 4),
    ]

    accuracy_dir = "accuracy_results"
    dest_dir = "accuracy_results/plots"

    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    plot_all_ops(accuracy_dir, all_operations, dest_dir)

    return

    # Unused
    data_all_exp = pd.concat([data_exp_csv, data_exp_approx_csv])
    data_all_exp_bf16 = pd.concat([data_exp_bf16, data_exp_approx_bf16])

    # plot_accuracy(data_exp_csv, "accuracy-exp", "exp")
    # plot_accuracy(data_log_csv, "accuracy-log", "log")
    # plot_accuracy(data_exp_approx_csv, "accuracy-exp-approx", "exp[approx]")

    if False:
        plot_values(
            data_all_exp_bf16,
            "exp[bf16]",
            "exp",
            "bfloat16",
            plot_override_fun=lambda ax: (ax.set_xlim(-4.0, 8.0), ax.set_ylim(-1, math.exp(8.05))),
        )

        plot_values(
            data_log_bf16,
            "log[bf16]",
            "log",
            "bfloat16",
            plot_override_fun=lambda ax: (ax.set_xlim(-4.0, 8.0), ax.set_ylim(-101, 8)),
        )

        plot_values(
            data_silu_bf16,
            "silu[bf16]",
            "silu",
            "bfloat16",
            plot_override_fun=lambda ax: (
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
            plot_override_fun=lambda ax: (
                ax.set_xlim(255.0, 257.0),
                ax.set_ylim(-0, 30),
            ),
        )

    if True:
        plot_accuracy_op(data_all_exp, "accuracy-all-exp-float32", "exp", "float32")
        plot_accuracy_op(data_log_csv, "accuracy-all-log-float32", "log", "float32")

        plot_accuracy_op(data_all_exp_bf16, "accuracy-all-exp-bfloat16", "exp", "bfloat16")
        plot_accuracy_op(
            data_silu_bf16,
            "accuracy-silu-bfloat16",
            "silu",
            "bfloat16",
            plot_override_fun=lambda ax: ax.set_xlim(-128, 32),
        )

        def override_log_plot(ax):
            ax.set_xscale("log", base=10)

            custom_ticks = [10e-30, 10e-17, 10e-4, 1, 10e9, 10e22, 10e35]

            # Found experimentally

            for k in [-64, -32, -16, 16, 32, 64]:
                xval = math.exp(k)
                ax.axvline(x=xval, color="k", linestyle="--")
                plt.text(xval * 1.05, 70, f"x=exp({k})", fontsize=24)

            plt.gca().set_xticks(custom_ticks)

        def override_log_zoom_plot(ax):
            ax.set_xlim(1e-1, 100)
            ax.set_xscale("log", base=10)

            custom_yticks = [0, 0.5, 1, 5, 10, 50, 100, 500]

            custom_xticks = [1e-1, 0.5, 1.0, 2.0, 5, 10, 20, 100]
            plt.gca().set_xticks(custom_xticks)
            plt.gca().set_xticklabels([f"{x:g}" for x in custom_xticks])  # For decimal form

            plt.gca().set_yticks(custom_yticks)
            plt.gca().set_yticklabels([f"{x:g}%" for x in custom_yticks])  # For decimal form

            # custom_yticks = [0, 1, 5, 10, 50, 100, 500]
            # plt.gca().set_yticks(custom_yticks)
            # plt.gca().set_yticklabels([f"{x}%" for x in custom_yticks])  # For decimal form

        plot_accuracy_op(data_log_bf16, "accuracy-log-bfloat16", "log", "bfloat16", plot_override_fun=override_log_plot)

        plot_accuracy_op(
            data_log_bf16, "accuracy-log-bfloat16[1-10]", "log", "bfloat16", plot_override_fun=override_log_zoom_plot
        )

    if False:

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
            data_exp_torch_f32, "accuracy-torch-exp-float32", "exp-torch", "float32", plot_override_fun=ax_fun_torch_f32
        )

        plot_accuracy_op(
            data_exp_torch_f32,
            "accuracy-torch-exp-bfloat16",
            "exp-torch",
            "bfloat16",
            plot_override_fun=ax_fun_torch_bf16,
        )


main()
