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


def plot_accuracy_op(data, dest, opname, op_dtype, plot_override_fun=None, plot_mean=True):
    data = data.copy()
    data["max_rel_error"] *= 100
    data["mean_rel_error"] *= 100

    print(f"plot_accuracy_op: {opname} {op_dtype} {plot_mean}")
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

    data_max = data_mlt[data_mlt["type"] == "max"].copy()
    data_mean = data_mlt[data_mlt["type"] == "mean"].copy()

    data_max["operation"] += " (max)"
    data_mean["operation"] += " (mean)"

    plt.axhline(y=0, color="k", linewidth=2)

    ncolors = len(data_max["operation"].unique())  # Not necessary, but remove warning
    color_palette = sns.color_palette("deep", ncolors)
    sns.lineplot(data=data_max, x="base_x", y="val", hue="operation", ax=ax, palette=color_palette)

    if plot_mean:
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

    ax.xaxis.set_major_locator(ticker.SymmetricalLogLocator(linthresh=1, base=10))
    ax.set_ylim(0, ymax)

    ax.set_title(f"Relative error of ttnn.{opname} against torch implementation\n[{op_dtype}]")

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

    # plt.savefig(f"{dest}.svg", bbox_inches="tight", pad_inches=0.0)
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
        ax.axvline(x=math.e, color="k", linestyle="--")
        ax.set_xscale("symlog", base=2)
        # custom_ticks = [-10, -2, -1,1, 2, 10]
        # ax.set_xticks(custom_ticks)
        # ax.set_xticklabels([f"{x:g}" for x in custom_ticks])

        pass

    def override_values(self, ax):
        ax.set_xlim(-4.0, 8.0)
        ax.set_ylim(-1, math.exp(8.05))
        pass


class PlotTanh:
    def override_accuracy(self, ax):
        pass

    def override_accuracy_zoom(self, ax):
        ax.set_xlim(-10, 10.0)
        ax.set_ylim(0, 100)

        custom_ticks = [-10, -2, -1, 0, 1, 2, 10]
        ax.set_xticks(custom_ticks)
        ax.set_xticklabels([f"{x:g}" for x in custom_ticks])

        pass

    def override_values(self, ax):
        pass


class PlotLog:
    def __init__(self, base):
        self.base = base

    def override_accuracy(self, ax):
        ax.set_xscale("log", base=10)

        custom_ticks = [10e-30, 10e-17, 10e-4, 1, 10e9, 10e22, 10e35]

        base_str = f"{self.base}" if self.base != math.e else "e"

        xmin, xmax = ax.get_xlim()

        # Found experimentally
        for k in [-64, -32, -16, 16, 32, 64]:
            xval = math.pow(self.base, k)

            if xval < xmin or xval > xmax:
                continue

            ax.axvline(x=xval, color="k", linestyle="--")
            plt.text(xval * 1.05, 70, f"$x={base_str}^{{{k}}}$", fontsize=30)

        plt.gca().set_xticks(custom_ticks)

    def override_accuracy_zoom(self, ax):
        ax.set_xlim(1e-1, 100)
        ax.set_ylim(
            0,
        )

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


class PlotLog1p:
    def override_accuracy(self, ax):
        ax.set_xlim(
            -1,
        )
        pass

    def override_accuracy_zoom(self, ax):
        ax.set_xlim(-1, 100)
        ax.set_ylim(0, 100)

    def override_values(self, ax):
        pass


class PlotSiLU:
    def override_accuracy(self, ax):
        ax.set_xlim(-128, 32)

    def override_accuracy_zoom(self, ax):
        ax.set_xlim(-4, 4)

        custom_ticks = [-4, -2, -1, 0, 1, 2, 4]
        ax.set_xticks(custom_ticks)
        ax.set_xticklabels([f"{x:g}" for x in custom_ticks])

    def override_values(self, ax):
        ax.set_xlim(-8.0, 4.0),
        ax.set_ylim(-1.0, 4.0)
        ax.axvline(x=-5.0, color="k", linestyle="--", label="x=-5", linewidth=3)
        plt.text(-5 + 0.2, 3, "x=-5", fontsize=40)
        pass


class PlotLogit:
    def override_accuracy(self, ax):
        ax.set_xlim(-0.1, 1)
        ax.set_xscale("linear")

        custom_ticks = [0, 0.1, 0.5, 1]
        ax.set_xticks(custom_ticks)
        ax.set_xticklabels([f"{x:g}" for x in custom_ticks])

    def override_accuracy_zoom(self, ax):
        self.override_accuracy(ax)
        ax.set_ylim(0, 500)

        pass

    def override_values(self, ax):
        ax.set_xlim(-0.1, 1)
        pass


class PlotMish:
    def override_accuracy(self, ax):
        pass

    def override_accuracy_zoom(self, ax):
        ax.set_xlim(-10, 10)
        ax.set_ylim(0, 500)
        ax.set_xscale("symlog", base=2)

        custom_ticks = [-8, -4, -2, -1, 0, 1, 2, 4, 8]
        ax.set_xticks(custom_ticks)
        ax.set_xticklabels([f"{x:g}" for x in custom_ticks])

    def override_values(self, ax):
        pass


class PlotCos:
    def __init__(self, is_sin=False):
        self.is_sin = is_sin

    def override_accuracy(self, ax):
        ax.set_ylim(0, 100)
        pass

    def override_accuracy_zoom(self, ax):
        ax.set_xlim(-3 * math.pi, 3 * math.pi)

        if self.is_sin:
            phasej = 1

        custom_ticks = [i for i in range(-6, 7)]
        if not self.is_sin:
            ax.set_xticks([j * math.pi / 2 for j in custom_ticks])
            ax.set_xticklabels([f"$\\frac{{{j}\\pi}}{{2}}$" for j in custom_ticks])
        else:
            ax.set_xticks([j * math.pi / 4 for j in custom_ticks])
            ax.set_xticklabels([f"$\\frac{{{3*j}\\pi}}{{4}}$" for j in custom_ticks])

        ax.set_ylim(0, 100)

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


class PlotSqrt:
    def override_accuracy(self, ax):
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)

    def override_accuracy_zoom(self, ax):
        self.override_accuracy(ax)
        ax.set_xscale("symlog", base=2)

        ax.set_xlim(0, 100)
        custom_ticks = [0, 1, 2, 4, 8, 16, 32, 64]
        ax.set_xticks(custom_ticks)
        ax.set_xticklabels([f"{x:g}" for x in custom_ticks])

    def override_values(self, ax):
        pass


all_override_plot_funs = {
    # Exponential functions
    "exp": PlotExp(),
    "tanh": PlotTanh(),  # reuse exp settings
    "cosh": PlotExp(),  # reuse exp settings
    "sinh": PlotMish(),  # reuse exp settings
    # Logarithmic functions
    "log": PlotLog(math.e),
    "log10": PlotLog(10),
    "log2": PlotLog(2),
    "log1p": PlotLog1p(),
    # Activation functions
    "silu": PlotSiLU(),  # reuse silu settings
    "logit": PlotLogit(),  # reuse silu settings
    "gelu": PlotSiLU(),  # reuse silu settings
    "swish": PlotSiLU(),  # reuse silu settings
    "mish": PlotMish(),  # reuse silu settings
    "elu": PlotSiLU(),  # reuse silu settings
    "selu": PlotSiLU(),  # reuse silu settings
    "softplus": PlotSiLU(),  # reuse silu settings
    "softsign": PlotSiLU(),  # reuse silu settings
    # Trigonometric functions
    "tan": PlotTan(),  # reuse tan settings
    "atan": None,
    "sin": PlotCos(is_sin=True),
    "cos": PlotCos(),
    # Miscellaneous functions
    "sqrt": PlotSqrt(),
    "rsqrt": None,
    "rsqrt_approx": None,
    "digamma": None,
    "lgamma": None,
    "tanhshrink": None,
}


def plot_all_ops(accuracy_dir, ops_list, dest_dir, highres=False):
    # This is not great because we have to define each parameter manually,
    # but this works for now.
    print(f"plot_all_ops: {ops_list}")
    # Store all operations results in dictionary
    # This is inefficient but is more convenient for concatenating
    # results such as 'exp' and 'exp_approx'
    all_op_data = {}

    # Aggregate data
    for op, dtype, group_size in ops_list:

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

        input_csv = f"{accuracy_dir}/{op}-{dtype}-[{group_size}].csv"
        output_file = f"{dest_dir}/{op}-{dtype}-[{group_size}].pdf"

        if should_regenerate(input_csv, output_file):
            data = load_csv(input_csv)

            all_op_data[(op, dtype, group_size)] = data

    # Concatenate exp and exp_approx (semi-generic)
    # (write both into all_op_data[exp] and remove all_op_data[exp_approx])
    for opkey in list(all_op_data.keys()):
        op, dtype, group_size = opkey

        if op == "exp_approx":
            exp_approx_data = all_op_data[opkey]

            exp_data = None
            if ("exp", dtype, group_size) in all_op_data:
                exp_data = all_op_data[("exp", dtype, group_size)]

            all_op_data[("exp", dtype, group_size)] = pd.concat([exp_data, exp_approx_data])
            del all_op_data[opkey]

    # Plot operations
    for op, dtype, group_size in all_op_data.keys():
        print(f"Plotting {op} {dtype} {group_size}\r")
        data = all_op_data[(op, dtype, group_size)]

        print(f"Plotting {op} {dtype} {group_size}", end="\r")

        # Each operation may have specific properties
        # To highlight these, custom functions have been defined
        override_plot_fun = None
        dest_file = f"{dest_dir}/{op}-{dtype}"

        plot_mean = True
        if highres:
            dest_file += "-zoom"
            plot_mean = False

        if all_override_plot_funs[op] is not None:
            override_plot_fun = all_override_plot_funs[op].override_accuracy
            if highres:
                override_plot_fun = all_override_plot_funs[op].override_accuracy_zoom

        plot_accuracy_op(data, dest_file, op, dtype, override_plot_fun, plot_mean=plot_mean)


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
        # ("exp", "bfloat16", 32),
        # ("exp_approx", "bfloat16", 32),
        # ("log", "bfloat16", 32),
        # ("tanh", "bfloat16", 32),
        # ("cosh", "bfloat16", 32),
        ("sinh", "bfloat16", 32),
        # ("log10", "bfloat16", 32),
        # ("log2", "bfloat16", 32),
        # ("log1p", "bfloat16", 32),
        # ("silu", "bfloat16", 32),
        # ("gelu", "bfloat16", 32),
        # ("logit", "bfloat16", 32),
        # ("swish", "bfloat16", 32),
        ("mish", "bfloat16", 32),
        # ("elu", "bfloat16", 32),
        # ("selu", "bfloat16", 32),
        # ("softplus", "bfloat16", 32),
        # ("softsign", "bfloat16", 32),
        # ("tan", "bfloat16", 32),
        # ("atan", "bfloat16", 32),
        # ("sin", "bfloat16", 32),
        # ("cos", "bfloat16", 32),
        # ("sqrt", "bfloat16", 32),
        # ("rsqrt", "bfloat16", 32),
        # ("rsqrt_approx", "bfloat16", 32),
        # ("digamma", "bfloat16", 32),
        # ("lgamma", "bfloat16", 32),
        # ("tanhshrink", "bfloat16", 32),
    ]

    highres_operations = [
        # ("exp", "bfloat16", 1),
        # ("exp_approx", "bfloat16", 1),
        # ("tanh", "bfloat16", 1),
        # ("cosh", "bfloat16", 1),
        ("sinh", "bfloat16", 1),
        # ("log", "bfloat16", 1),
        # ("log10", "bfloat16", 1),
        # ("log2", "bfloat16", 1),
        # ("log1p", "bfloat16", 1),
        # ("tan", "bfloat16", 1),
        # ("cos", "bfloat16", 1),
        # ("sin", "bfloat16", 1),
        # ("silu", "bfloat16", 1),
        # ("gelu", "bfloat16", 1),
        # ("logit", "bfloat16", 1),
        # ("swish", "bfloat16", 1),
        ("mish", "bfloat16", 1),
        # ("elu", "bfloat16", 1),
        # ("sqrt", "bfloat16", 1),
        # ("rsqrt_approx", "bfloat16", 1),
        # ("rsqrt", "bfloat16", 1),
    ]

    accuracy_dir = "accuracy_results"
    dest_dir = "accuracy_results/plots"

    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    plot_all_ops(accuracy_dir, all_operations, dest_dir)
    plot_all_ops(accuracy_dir, highres_operations, dest_dir, highres=True)

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


main()
