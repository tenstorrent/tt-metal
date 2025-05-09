import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.patches as mpatches
from matplotlib.ticker import ScalarFormatter, PercentFormatter


import pandas as pd
import seaborn as sns
import numpy as np

import json
import math
import sys
import os.path
import re
import time
import multiprocessing as mp


RED = "\033[91m"
GREEN = "\033[92m"
BLUE = "\033[94m"
YELLOW = "\033[93m"
RESET = "\033[0m"

plt.rcParams["svg.fonttype"] = "none"  # Make text editing in SVG easier


def load_csv(filename):
    return pd.read_csv(filename, sep=",", index_col=False, skipinitialspace=True)


def create_directory(file_path):
    dir_path = os.path.dirname(file_path)
    os.makedirs(dir_path, exist_ok=True)


def preprocess_operation(data):
    # Somewhat hacky way to have differenciate series on the same legend
    data["operation_mean"] = data["operation"] + " (mean)"
    data["operation_max"] = data["operation"] + " (max)"

    return data


def parse_plot_config(plot_config_path):
    with open(plot_config_path, "r") as f:
        plot_config = json.load(f)

    for group in plot_config["groups"]:
        for plot_entry in group["plots"]:
            insert_default_params(group, plot_entry)

    return plot_config


def hash_plot_entry(entry):
    import hashlib

    entry_str = json.dumps(entry, sort_keys=True)
    hash_obj = hashlib.sha256(entry_str.encode())
    return hash_obj.hexdigest()


def generate_plot_config_hashes(plot_config):
    # Dictionary to store hashes for each plot configuration
    plot_config_hashes = {}

    # Process each plot entry in the configuration
    for group in plot_config["groups"]:
        for plot_entry in group["plots"]:
            if "data" in plot_entry:
                del plot_entry["data"]

            hash_value = hash_plot_entry(plot_entry)
            plot_id = plot_entry["id"]

            # Store the hash with its corresponding plot entry
            plot_config_hashes[plot_id] = hash_value

    return plot_config_hashes


def load_plot_config_hashes(input_path):
    if not os.path.exists(input_path):
        plot_config_hashes = {}
        plot_config_hashes["last_modified"] = 0
        return plot_config_hashes

    # Get time stamp of last modification of hash file
    hash_last_modified = os.path.getmtime(input_path)

    # Load as json file
    with open(input_path, "r") as f:
        plot_config_hashes = json.load(f)

    plot_config_hashes["last_modified"] = hash_last_modified

    return plot_config_hashes


def save_plot_config_hashes(plot_config_hashes, output_path):
    # Save as json file
    with open(output_path, "w") as f:
        json.dump(plot_config_hashes, f)


def try_plot(plot_entry):
    try:
        plot(plot_entry)
    except Exception as e:
        print(f"Error plotting {RED}{plot_entry['id']}: {e}{RESET}")


def plot(plot_entry):
    output_path = plot_entry["output"]
    data = plot_entry["data"]

    # Read parameters

    plot_params = plot_entry["plot_params"]

    id = plot_entry["id"]
    title = plot_params["title"] if "title" in plot_params else None
    short_name = plot_entry["name"] if "name" in plot_params else id

    if title is not None:
        title = title.format(short_name)

    xbase = plot_params["xbase"] if "xbase" in plot_params else 10
    xscale = plot_params["xscale"] if "xscale" in plot_params else "symlog"

    yscale = plot_params["yscale"] if "yscale" in plot_params else "asinh"
    ybase = plot_params["ybase"] if "ybase" in plot_params else 10

    [xmin, xmax] = plot_params["xlim"] if "xlim" in plot_params else [None, None]
    [ymin, ymax] = plot_params["ylim"] if "ylim" in plot_params else [None, None]

    xticks = plot_params["xticks"] if "xticks" in plot_params else None
    yticks = plot_params["yticks"] if "yticks" in plot_params else None

    palette_offset = plot_params["palette_offset"] if "palette_offset" in plot_params else 0

    xname = plot_entry["xname"]
    ynames = plot_entry["ynames"]
    hseries = plot_entry["hue"] if "hue" in plot_entry else None

    xlabel = plot_params["xlabel"] if "xlabel" in plot_params else xname
    ylabel = plot_params["ylabel"] if "ylabel" in plot_params else ynames[0]

    yticksformat = plot_params["yticksformat"] if "yticksformat" in plot_params else None

    fig, ax = plt.subplots(figsize=(25, 15))

    # color_palette = sns.color_palette("deep", len(ynames))
    ncolors = len(data[hseries].unique())
    color_palette = sns.color_palette("deep", ncolors + palette_offset)[palette_offset:]

    for y in ynames:
        data = plot_entry["data"]
        d2 = data.copy()

        [yname, ysuffix, linestyle] = y
        d2["operation"] += ysuffix
        if hseries is not None:
            ax = sns.lineplot(data=d2, x=xname, y=yname, ax=ax, hue=hseries, linestyle=linestyle, palette=color_palette)
        else:
            ax = sns.lineplot(data=d2, x=xname, y=yname, ax=ax, label=yname, linestyle=linestyle, palette=color_palette)

    print(f"{output_path} - XSCALE = {xscale}, base = {xbase}")

    if xscale == "linear":
        ax.set_xscale("linear")
    else:
        ax.set_xscale(xscale, base=xbase)

    if yscale == "asinh":
        ax.set_yscale(yscale, linear_width=0.01)
    elif yscale == "linear":
        ax.set_yscale("linear")
    else:
        ax.set_yscale(yscale, base=ybase)

    if title is not None:
        ax.set_title(title)

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if "vertical_lines" in plot_params:
        for vertical_line in plot_params["vertical_lines"]:
            ax.axvline(x=vertical_line[0], color="k", linestyle="--")
            ax.text(vertical_line[0], ax.get_ylim()[1], vertical_line[1], fontsize=14)

    if yticks is not None:
        # print(f"yticks = {yticks}")
        ax.set_yticks(yticks)

        pass

    if yticksformat == "percent":
        # plt.gca().set_yticklabels([f"{100*x}%" for x in ax.get_yticks()])
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1))
        pass

    if xticks is not None:
        ax.set_xticks(xticks)

    plt.savefig(output_path, bbox_inches="tight", pad_inches=0.0)
    plt.close()

    pass


def insert_default_params(plot_parent_config, plot_group_config):
    for default_param in plot_parent_config["default_params"]:
        if default_param not in plot_group_config:
            plot_group_config[default_param] = plot_parent_config["default_params"][default_param]

    if "default_params" in plot_parent_config:
        default_plot_params = plot_parent_config["default_params"]["plot_params"]
        for default_plot_param in default_plot_params:
            if default_plot_param not in plot_group_config["plot_params"]:
                plot_group_config["plot_params"][default_plot_param] = default_plot_params[default_plot_param]


def plot_all(plot_config, base_output_dir, last_hashes, current_hashes, force_replot=False):
    plot_args = []

    for plot_group in plot_config["groups"]:
        for plot_entry in plot_group["plots"]:
            plot_id = plot_entry["id"]

            do_replot = True

            if plot_id in current_hashes:
                # Compute new hash and check against previous one
                hash_value = current_hashes[plot_id]
                last_hash = last_hashes[plot_id] if plot_id in last_hashes else None

                if hash_value == last_hash:
                    print(f"{BLUE}Skipping {plot_id} because hash value is the same{RESET}")
                    do_replot = False

            last_modified = last_hashes["last_modified"]
            for file in plot_entry["files"]:
                if os.path.getmtime(file) > last_modified:
                    do_replot = True
                    break

            if force_replot:
                do_replot = True

            if do_replot:
                plot_args.append(plot_entry)

    # For each plot entry, import data
    for plot_entry in plot_args:
        data_series = plot_entry["files"]
        list_all_data = []

        # TODO: Cache data
        for series in data_series:
            data_op = load_csv(series)
            list_all_data.append(data_op)

        data = pd.concat(list_all_data, axis=0)

        # Remove data with NaN or infinity (might speedup plotting)
        data = data[(data["base_x"].notna()) & (np.isfinite(data["base_x"]))]
        data = data.reset_index()

        # data = data.set_index(["base_x", "operation"])

        output_path = plot_entry["output"]
        create_directory(output_path)

        # Transform data if necessary
        plot_entry["data"] = data

    # Launch parallel plots
    num_processes = mp.cpu_count()
    print(f"Plotting {len(plot_args)} operations with {num_processes} processes")

    with mp.Pool(num_processes) as pool:
        results = pool.map(try_plot, plot_args)

        cnt = 1
        for result in results:
            print(f"#{cnt}/{len(results)}", end="\r")
            cnt += 1


def plot_common_accuracy(
    fig,
    ax,
    data,
    dest,
    opname,
    op_dtype,
    plot_override_fun=None,
    plot_mean=True,
    color_palette=sns.color_palette("deep"),
):
    # plt.axhline(y=0, color="k", linewidth=2)

    sns.lineplot(data=data, x="base_x", y="metric_max", hue="operation_mean", ax=ax, palette=color_palette)

    if plot_mean:
        data["operation"] = data["operation"] + " (mean)"
        sns.lineplot(
            data=data,
            x="base_x",
            y="metric_mean",
            hue="operation",
            ax=ax,
            linestyle="--",
            palette=color_palette,
        )

    ax.set_xscale("symlog", base=10)
    ax.set_xlabel("X")

    xmin, xmax = ax.get_xlim()
    ax.set_ylim(0, None)

    if xmax < 1:
        plt.gca().xaxis.set_major_formatter(ScalarFormatter(useOffset=False, useMathText=False))

    xticker = ticker.SymmetricalLogLocator(linthresh=1, base=10)
    xticker.set_params(numticks=9)
    ax.xaxis.set_major_locator(xticker)

    def xticks_formatter(x, pos):
        x = float(x)  # Ensure x is a floating point number
        if abs(x) >= 100:
            exponent = int(math.log10(abs(x)))
            sign = "-" if x < 0 else ""
            return f"${sign}10^{{{exponent}}}$"
        else:
            return f"{x:.1f}" if not x.is_integer() else f"{int(x)}"

    ax.xaxis.set_major_formatter(ticker.FuncFormatter(xticks_formatter))

    return plt, ax


def plot_abs_accuracy(data, dest, opname, op_dtype, plot_override_fun=None, plot_mean=True):
    data["metric_max"] = data["max_abs_error"]
    data["metric_mean"] = data["mean_abs_error"]

    data = preprocess_operation(data)

    fig, ax = plt.subplots(figsize=(25, 15))

    ncolors = len(data["operation_max"].unique())
    color_palette = sns.color_palette("deep", ncolors + 3)[3:]

    (fig, ax) = plot_common_accuracy(fig, ax, data, dest, opname, op_dtype, plot_override_fun, plot_mean, color_palette)

    ax.set_yscale("log", base=10)
    yticker = ticker.LogLocator(base=10)
    yticker.set_params(numticks=7)
    ax.yaxis.set_major_locator(yticker)

    ax.set_title(f"Absolute error of ttnn.{opname} against torch implementation\n[{op_dtype}]")
    ax.set_ylabel("Absolute error vs. torch\n(lower is better)")

    if plot_override_fun is not None:
        plot_override_fun(ax)

    plt.legend(ncol=2)
    plt.savefig(f"{dest}.png", bbox_inches="tight", pad_inches=0.0)
    plt.close()


def plot_relative_accuracy(data, dest, opname, op_dtype, plot_override_fun=None, plot_mean=True):
    data["metric_max"] = data["max_rel_error"]
    data["metric_mean"] = data["mean_rel_error"]

    data = preprocess_operation(data)

    fig, ax = plt.subplots(figsize=(25, 15))

    ncolors = len(data["operation_max"].unique())
    color_palette = sns.color_palette("deep", ncolors)

    (fig, ax) = plot_common_accuracy(fig, ax, data, dest, opname, op_dtype, plot_override_fun, plot_mean, color_palette)

    ax.set_yscale("asinh", linear_width=0.01)

    ax.set_title(f"Relative error of ttnn.{opname} against torch implementation\n[{op_dtype}]")

    yticker = ticker.SymmetricalLogLocator(linthresh=1, base=10)
    yticker.set_params(numticks=7)
    ax.yaxis.set_major_locator(yticker)

    ymin, ymax = ax.get_ylim()
    custom_ticks = [0, 0.005, 0.01, 0.05, 0.1, 0.5, 1]
    if ymax > 1:
        custom_ticks += [int(10**i) for i in range(3, int(min(10, math.log10(ymax))) + 1)]

    plt.gca().set_yticks(custom_ticks)
    plt.gca().set_yticklabels([f"{100*x}%" for x in custom_ticks])

    ax.set_ylabel("Error vs. torch [%]\n(lower is better)")

    if plot_override_fun is not None:
        plot_override_fun(ax)

    plt.legend(ncol=2)
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


class PlotBasic:
    def __init__(self, zoomed_in_xrange=None, zoomed_in_yrange=None):
        self.zoomed_in_xrange = zoomed_in_xrange
        self.zoomed_in_yrange = zoomed_in_yrange

    def override_accuracy(self, ax):
        pass

    def override_accuracy_zoom(self, ax):
        if self.zoomed_in_xrange is not None:
            ax.set_xlim(self.zoomed_in_xrange[0], self.zoomed_in_xrange[1])
        if self.zoomed_in_yrange is not None:
            ax.set_ylim(self.zoomed_in_yrange[0], self.zoomed_in_yrange[1])

        ax.set_xscale("symlog", base=2)

    def override_absolute(self, ax):
        ax.set_ylim(1e-8, 1e8)
        ax.set_xlim(None, 1e10)
        pass

    def override_values(self, ax):
        pass


class PlotExp:
    def override_accuracy(self, ax):
        ax.set_xscale("symlog", base=2)
        ax.set_xlim(-100, 100)
        pass

    def override_accuracy_zoom(self, ax):
        # ax.axvline(x=math.e, color="k", linestyle="--")
        ax.set_xscale("symlog", base=2)
        # custom_ticks = [-10, -2, -1,1, 2, 10]
        # ax.set_xticks(custom_ticks)
        # ax.set_xticklabels([f"{x:g}" for x in custom_ticks])

        pass

    def override_absolute(self, ax):
        pass

    def override_values(self, ax):
        ax.set_xlim(-4.0, 8.0)
        ax.set_ylim(0, math.exp(8.05))
        pass


class PlotTanh:
    def override_accuracy(self, ax):
        pass

    def override_accuracy_zoom(self, ax):
        ax.set_xlim(-10, 10.0)
        ax.set_ylim(0, 1)

        custom_ticks = [-10, -2, -1, 0, 1, 2, 10]
        ax.set_xticks(custom_ticks)
        ax.set_xticklabels([f"{x:g}" for x in custom_ticks])

    def override_absolute(self, ax):
        pass

    def override_values(self, ax):
        pass


class PlotLog:
    def __init__(self, base):
        self.base = base

    def override_common(self, ax):
        ax.set_xscale("log", base=10)

    def override_accuracy(self, ax):
        self.override_common(ax)

        custom_ticks = [10e-30, 10e-17, 10e-4, 1, 10e9, 10e22, 10e35]

        base_str = f"{self.base}" if self.base != math.e else "e"

        xmin, xmax = ax.get_xlim()

        # Found experimentally
        for k in [-64, -32, -16, 16, 32, 64]:
            xval = math.pow(self.base, k)

            if xval < xmin or xval > xmax:
                continue

            ax.axvline(x=xval, color="k", linestyle="--")
            plt.text(xval * 1.05, 0.3, f"$x={base_str}^{{{k}}}$", fontsize=30)

        plt.gca().set_xticks(custom_ticks)

    def override_accuracy_zoom(self, ax):
        self.override_common(ax)

        ax.set_xlim(1e-1, 100)

        custom_xticks = [1e-1, 0.5, 1.0, 2.0, 5, 10, 20, 100]
        plt.gca().set_xticks(custom_xticks)
        plt.gca().set_xticklabels([f"{x:g}" for x in custom_xticks])  # For decimal form

    def override_absolute(self, ax):
        pass

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
        ax.set_ylim(0, 1)

    def override_absolute(self, ax):
        ax.set_yscale("log", base=10)
        ax.set_ylim(0, 1)

        yticker = ticker.LogLocator(base=10)
        yticker.set_params(numticks=7)
        ax.yaxis.set_major_locator(yticker)

        pass

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

    def override_absolute(self, ax):
        ax.set_xscale("symlog", base=10, linthresh=1e-6)
        ax.set_yscale("log", base=10)
        ax.set_ylim(1e-9, 1)
        ax.set_xlim(-1e10, 1e10)

        custom_ticks = [1e-9, 1e-6, 1e-3, 1e-2, 1e-1, 1, 1e3]
        ax.set_yticks(custom_ticks)
        ax.set_yticklabels([f"{x:g}" for x in custom_ticks])

        pass

    def override_values(self, ax):
        ax.set_xlim(-8.0, 4.0),
        ax.set_ylim(-1.0, 4.0)
        ax.axvline(x=-5.0, color="k", linestyle="--", label="x=-5", linewidth=3)
        plt.text(-5 + 0.2, 3, "x=-5", fontsize=40)
        pass


class PlotLogit:
    def override_common(self, ax):
        ax.set_xlim(-0.1, 1)
        ax.set_xscale("linear")

        ymin, ymax = ax.get_ylim()
        ax.set_ylim(ymin, min(ymax, 5))

        custom_ticks = [0, 0.1, 0.5, 1]
        ax.set_xticks(custom_ticks)
        ax.set_xticklabels([f"{x:g}" for x in custom_ticks])

    def override_accuracy(self, ax):
        self.override_common(ax)

    def override_accuracy_zoom(self, ax):
        self.override_common(ax)

        pass

    def override_absolute(self, ax):
        ax.set_xlim(-0.1, 1.1)
        ax.set_ylim(0, 0.05)
        ax.set_xscale("log")
        ax.set_yscale("linear")

        ax.set_xticks([0, 0.1, 0.25, 0.5, 0.75, 0.9, 1])
        ax.set_xticklabels([f"{x:g}" for x in ax.get_xticks()])

        ytickslabels = ax.get_yticks()
        ax.set_yticks(ytickslabels)
        ax.set_yticklabels([f"{x:g}" for x in ytickslabels])
        pass

    def override_values(self, ax):
        pass


class PlotMish:
    def override_accuracy(self, ax):
        ax.set_yscale("log", base=10)

        custom_ticks = [1e-12, 1e-9, 1e-6, 1e-3, 1e-2, 1e-1, 1, 10, 1e3, 1e6, 1e9, 1e12]
        ax.set_yticks(custom_ticks)
        ax.set_yticklabels([f"{x:g}" for x in custom_ticks])

        ax.set_ylim(1e-18, 1e18)
        ax.set_xlim(-1e20, 1e20)

        pass

    def override_accuracy_zoom(self, ax):
        ax.set_xscale("symlog", base=10)

        custom_ticks = [-8, -4, -2, -1, 0, 1, 2, 4, 8]
        ax.set_xticks(custom_ticks)
        ax.set_xticklabels([f"{x:g}" for x in custom_ticks])

    def override_absolute(self, ax):
        ax.set_xscale("symlog", base=10)
        ax.set_yscale("symlog", base=10, linthresh=1e-6)
        ax.set_ylim(1e-6, 1e2)
        ax.set_xlim(-1e6, 1e6)
        pass

    def override_values(self, ax):
        pass


class PlotCos:
    def __init__(self, is_sin=False):
        self.is_sin = is_sin

    def override_accuracy(self, ax):
        ax.set_ylim(0, 1e3)
        pass

    def override_accuracy_zoom(self, ax):
        if self.is_sin:
            phasej = 1

        custom_ticks = [i for i in range(-6, 7)]
        if not self.is_sin:
            ax.set_xticks([j * math.pi / 2 for j in custom_ticks])
            ax.set_xticklabels([f"$\\frac{{{j}\\pi}}{{2}}$" for j in custom_ticks])
        else:
            ax.set_xticks([j * math.pi / 4 for j in custom_ticks])
            ax.set_xticklabels([f"$\\frac{{{3*j}\\pi}}{{4}}$" for j in custom_ticks])

        ax.set_ylim(0, 1)

    def override_absolute(self, ax):
        ax.set_ylim(1e-5, 1e2)
        pass

    def override_values(self, ax):
        pass


class PlotTan:
    def override_accuracy(self, ax):
        pass

    def override_accuracy_zoom(self, ax):
        xmax = math.pi
        ax.set_xscale("linear")
        kpi = math.pi / 2

        k = 0
        while kpi < xmax:
            ax.axvline(x=kpi, color="k", linewidth=3, linestyle="--")
            ax.axvline(x=-kpi, color="k", linewidth=3, linestyle="--")
            plt.text(kpi + 0.1, 0.5, f"$\\frac{{{2*k+1}\\pi}}{{2}}$", fontsize=50)
            plt.text(-kpi - 0.1, 0.5, f"$-\\frac{{{2*k+1}\\pi}}{{2}}$", fontsize=50)

            k += 1
            kpi = k * math.pi + math.pi / 2.0

    def override_absolute(self, ax):
        ax.set_ylim(1e-12, 1e6)
        pass

    def override_values(self, ax):
        ax.set_xlim(255.0, 257.0)
        ax.set_ylim(-0, 30)
        pass


class PlotSqrt:
    def override_accuracy(self, ax):
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 1)

    def override_accuracy_zoom(self, ax):
        self.override_accuracy(ax)
        ax.set_xscale("log", base=2)

        custom_ticks = [0, 1, 2, 4, 8, 16, 32, 64]
        ax.set_xticks(custom_ticks)
        ax.set_xticklabels([f"{x:g}" for x in custom_ticks])

    def override_absolute(self, ax):
        ax.set_xscale("log", base=2)

        pass

    def override_values(self, ax):
        pass


class PlotDiagamma:
    def override_accuracy(self, ax):
        ax.set_xscale("symlog", base=10, linthresh=1e-6)
        ax.set_ylim(1e-8, 2e1)
        ax.set_xlim(1e-12, None)

    def override_accuracy_zoom(self, ax):
        ax.set_xscale("symlog", base=2, linthresh=1e-6)
        ax.set_ylim(1e-8, 1e3)
        ax.set_xlim(2**-2, 10)
        pass

    def override_absolute(self, ax):
        ax.set_xscale("symlog", base=10, linthresh=1e-6)
        ax.set_xlim(0, 1e39)
        ax.set_ylim(1e-4, 1e6)

    def override_values(self, ax):
        pass


class PlotLgamma:
    def override_accuracy(self, ax):
        ax.set_xscale("symlog", base=10, linthresh=1e-6)

    def override_accuracy_zoom(self, ax):
        pass

    def override_absolute(self, ax):
        ax.set_xlim(None, 1e10)
        ax.set_ylim(1e-6, 1e10)
        pass

    def override_values(self, ax):
        pass


class PlotTanhshrink:
    def override_accuracy(self, ax):
        ax.set_xlim(-1e4, 10e4)
        ax.set_ylim(1e-3, 50)
        pass

    def override_accuracy_zoom(self, ax):
        pass

    def override_absolute(self, ax):
        ax.set_xlim(-1e4, 10e4)
        ax.set_ylim(1e-12, 1e2)
        pass

    def override_values(self, ax):
        pass


class PlotReciprocal:
    def override_accuracy(self, ax):
        ax.set_xlim(-1e7, 1e7)
        pass

    def override_accuracy_zoom(self, ax):
        ax.set_xlim(-0.8, 0.8)
        pass

    def override_absolute(self, ax):
        pass

    def override_values(self, ax):
        pass


Y_OFFSET = 0  # Offset y=0 curve, to make f(x)=0 values visible

all_override_plot_funs = {
    # Exponential functions
    "exp": (PlotExp(), (-100, 100)),
    "pow": (PlotExp(), (-100, 100)),
    "tanh": (PlotTanh(), (-10, 10)),  # reuse exp settings
    "cosh": (PlotMish(), (-10, 10)),  # reuse exp settings
    "sinh": (PlotMish(), (-10, 10)),  # reuse exp settings
    # Logarithmic functions
    "log": (PlotLog(math.e), None),
    "log10": (PlotLog(10), None),
    "log2": (PlotLog(2), None),
    "log1p": (PlotLog1p(), (-1, None)),
    # Activation functions
    "silu": (PlotSiLU(), (-4, 4)),  # reuse silu settings
    "logit": (PlotLogit(), (-0.1, 1.1)),  # reuse silu settings
    "gelu": (PlotSiLU(), (-4, 4)),  # reuse silu settings
    "gelu_approx": (PlotSiLU(), (-4, 4)),  # reuse silu settings
    "swish": (PlotSiLU(), (-4, 4)),  # reuse silu settings
    "mish": (PlotMish(), (-4, 4)),  # reuse silu settings
    "elu": (PlotMish(), (-10, 10)),  # reuse silu settings
    "selu": (PlotSiLU(), (-4, 4)),  # reuse silu settings
    # "softplus": (PlotBasic(zoomed_in_xrange=[-9, 9], zoomed_in_yrange=[Y_OFFSET, 1]), (-9, 9)),  # reuse silu settings
    # "softsign": (PlotBasic(zoomed_in_xrange=[-3, 3], zoomed_in_yrange=[Y_OFFSET, 1]), (-3, 3)),  # reuse silu settings
    "softplus": (PlotSiLU(), (-4, 4)),  # reuse silu settings
    "softsign": (PlotSiLU(), (-4, 4)),  # reuse silu settings
    # Trigonometric functions
    "tan": (PlotTan(), (-10, 10)),  # reuse tan settings
    "atan": (PlotBasic(zoomed_in_xrange=[-3, 3], zoomed_in_yrange=[Y_OFFSET, 1]), (-3, 3)),
    "sin": (PlotCos(is_sin=True), (-3 * math.pi, 3 * math.pi)),
    "cos": (PlotCos(), (-3 * math.pi, 3 * math.pi)),
    # Miscellaneous functions
    "sqrt": (PlotSqrt(), (0, 100)),
    "rsqrt": (PlotBasic(zoomed_in_xrange=[0, 3], zoomed_in_yrange=[0, 1]), (0, 3)),
    "reciprocal": (PlotReciprocal(), (0, 3)),
    "rsqrt_approx": None,
    "digamma": (PlotDiagamma(), (0, 10)),
    "lgamma": (PlotLgamma(), (0, 10)),
    "tanhshrink": (PlotTanhshrink(), (-3, 3)),
}

powers_list = [2, 3, 5, 7, 10]
for power in powers_list:
    all_override_plot_funs[f"pow_{power}"] = (PlotExp(), (-100, 100))


def plot_hist_accuracy(data, dest, threshold=0.1, xrange=None):
    thresholds = [0.005, 0.01, 0.05, 0.1]
    # Assign xmin and xmax from xrange
    xmin, xmax = xrange if xrange is not None else (None, None)

    # Filter to only bfloat16 and group_size=1 entries
    filtered_data = [df for (op, dtype, group_size), df in data.items() if dtype == "bfloat16" and group_size == 1]

    # Concatenate all DataFrames
    combined_df = pd.concat(filtered_data)

    # If base_y and base_yref are the same then set max_rel_error to 0
    combined_df["max_rel_error"] = np.where(
        combined_df["base_y"] == combined_df["base_yref"], 0, combined_df["max_rel_error"]
    )

    # Filter NaN and infinite values
    combined_df = combined_df[combined_df["max_rel_error"].notna()]
    combined_df = combined_df[combined_df["max_rel_error"] != float("inf")]

    # Apply x-range filter if provided
    if xmin is not None:
        combined_df = combined_df[combined_df["base_x"] >= xmin]
    if xmax is not None:
        combined_df = combined_df[combined_df["base_x"] <= xmax]

    # Calculate proportions for each threshold
    all_proportions = []
    for threshold in thresholds:
        exceeding_threshold = combined_df[combined_df["max_rel_error"] > threshold]
        counts = exceeding_threshold.groupby("operation").size()
        total_counts = combined_df.groupby("operation").size()
        proportions = (counts / total_counts).reset_index()
        proportions.columns = ["operation", "proportion"]
        proportions["error_margin"] = f"ε > {100*threshold}%"  # Format as percent
        all_proportions.append(proportions)

    # Combine all proportions into a single dataframe
    plot_df = pd.concat(all_proportions, ignore_index=True)

    # Create figure
    fig, ax = plt.subplots(figsize=(50, 25))

    # Plot using seaborn's barplot
    sns.barplot(x="operation", y="proportion", hue="error_margin", data=plot_df, ax=ax)

    # Customize plot
    ax.set_xlabel("Operation")
    ax.set_ylabel("Proportion of bad values")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    ax.legend(title="Error Margin")
    ax.set_ylim(0, 1)

    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: "{:.0%}".format(y)))
    xmin_str = f"{xmin}" if xmin is not None else "-inf"
    xmax_str = f"{xmax}" if xmax is not None else "+inf"
    xrange_str = f"[{xmin_str}, {xmax_str}]"

    threshold_percent_str = ", ".join([f"{100*t}%" for t in thresholds])
    ax.set_title(f"Proportion of bad values on {xrange_str}\n(threshold $\\in$ [{threshold_percent_str}])")

    # Add grid
    ax.grid(True, axis="y", linestyle="--", alpha=0.7)

    # Adjust layout to prevent label cutoff
    plt.tight_layout()

    # Save plot
    plt.savefig(dest, bbox_inches="tight", pad_inches=0.0)
    plt.close()


def plot_hist_accuracy_bis(data, dest, threshold, xrange=None):
    # Assign xmin and xmax from xrange
    xmin, xmax = xrange if xrange is not None else (None, None)

    # Filter to only bfloat16 and group_size=1 entries
    filtered_data = [df for (op, dtype, group_size), df in data.items() if dtype == "bfloat16" and group_size == 1]

    print(f"FILTERED DATA = {filtered_data}")
    filtered_data = [df for df in filtered_data]

    # Concatenate all DataFrames
    combined_df = pd.concat(filtered_data)

    print(f"COMBINED DF = {combined_df}")

    # If base_y and base_yref are the same then set max_rel_error to 0
    combined_df["max_rel_error"] = np.where(
        combined_df["base_y"] == combined_df["base_yref"], 0, combined_df["max_rel_error"]
    )

    # Filter NaN and infinite values
    combined_df = combined_df[combined_df["max_rel_error"].notna()]
    combined_df = combined_df[combined_df["max_rel_error"] != float("inf")]

    # Apply x-range filter if provided
    if xmin is not None:
        combined_df = combined_df[combined_df["base_x"] >= xmin]
    if xmax is not None:
        combined_df = combined_df[combined_df["base_x"] <= xmax]

    # Filter values > threshold and group by operation
    exceeding_threshold = combined_df[combined_df["max_rel_error"] > threshold]
    counts = exceeding_threshold.groupby("operation").size()

    # Calculate proportions
    total_counts = combined_df.groupby("operation").size()
    proportions = counts / total_counts

    # Create bar plot
    fig, ax = plt.subplots(figsize=(35, 25))

    bars = ax.bar(proportions.index, proportions.values)

    ax.set_ylim(0, 1)

    # Customize plot
    ax.set_xlabel("Operation")
    ax.set_ylabel(f"Proportion of relative errors > {threshold}")

    xmin_str = f"{xmin}" if xmin is not None else "-inf"
    xmax_str = f"{xmax}" if xmax is not None else "+inf"
    xrange_str = f"[{xmin_str}, {xmax_str}]"

    ax.set_title(f"Proportion of bad values on {xrange_str}\n(threshold = {threshold})")

    # Rotate x-axis labels for better readability and format y-axis as percentage
    plt.xticks(rotation=45, ha="right")
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: "{:.0%}".format(y)))

    # Add percentage labels on top of bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2.0, height, f"{height*100:.1f}%", ha="center", va="bottom")

    # Adjust layout to prevent label cutoff
    plt.tight_layout()

    # Save plot
    plt.savefig(f"{dest}.png", bbox_inches="tight", pad_inches=0.0)
    # plt.savefig(f"{dest}.svg", bbox_inches="tight", pad_inches=0.0)
    plt.close()


def plot_single_op(args):
    op, dtype, group_size, data_op, dest_dir, highres, plot_absolute = args

    start_time = time.time()

    # Each operation may have specific properties
    # To highlight these, custom functions have been defined
    override_plot_fun = None
    dest_file = f"{dest_dir}/{op}-{dtype}"

    plot_mean = True
    if highres:
        dest_file += "-zoom"
        plot_mean = False

    if all_override_plot_funs[op] is not None:
        (override_instance, xrange) = all_override_plot_funs[op]

        xmin = None
        xmax = None
        if highres and xrange is not None:
            (xmin, xmax) = xrange

        if highres and xmin is not None and xmax is not None:
            # Remove excess data (make plot faster + adjust y-axis values more easily)
            data_op = data_op[data_op["base_x"] >= xmin]
            data_op = data_op[data_op["base_x"] <= xmax]

        if override_instance is not None:
            override_plot_fun = override_instance.override_accuracy
            if highres:
                override_plot_fun = override_instance.override_accuracy_zoom
            if plot_absolute:
                override_plot_fun = override_instance.override_absolute

    if plot_absolute:
        plot_abs_accuracy(data_op, dest_file, op, dtype, override_plot_fun, plot_mean=plot_mean)
    else:
        plot_relative_accuracy(data_op, dest_file, op, dtype, override_plot_fun, plot_mean=plot_mean)

    return f"Plotting {op} {dtype} {group_size} - done in {time.time() - start_time:.2f}s"


def plot_all_ops(accuracy_dir, ops_list, dest_dir, highres=False, plot_absolute=False):
    # This is not great because we have to define each parameter manually,
    # but this works for now.

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

    # if highres:
    #     # Plot recap (histogram of relative error)
    #     plot_hist_accuracy(all_op_data, f"{dest_dir}/hist-rel-error", 0.01)
    #     plot_hist_accuracy(all_op_data, f"{dest_dir}/hist-rel-error[-10,10]", 0.01, xrange=[-10, 10])
    #     plot_hist_accuracy(all_op_data, f"{dest_dir}/hist-rel-error[-1,1]", 0.01, xrange=[-1, 1])

    # return

    # Transform data:
    # 1) Concatenate all data into a single dataframe
    if len(all_op_data) == 0:
        return

    all_data = pd.concat(all_op_data.values())

    # Remove data with NaN or infinity (might speedup plotting)
    all_data = all_data[all_data["max_rel_error"].notna()]
    all_data = all_data[all_data["max_rel_error"] != float("inf")]

    all_parent_ops = all_data["parent_op"].unique()
    groupby_parent = all_data.groupby(("parent_op"), sort=True)

    # Aggregate process arguments
    plot_args = []
    for parent_op in all_parent_ops:
        data_op = groupby_parent.get_group(parent_op).reset_index()
        plot_args.append((parent_op, dtype, group_size, data_op, dest_dir, highres, plot_absolute))

    # Launch parallel plots
    num_processes = mp.cpu_count()
    print(f"Plotting {len(plot_args)} operations with {num_processes} processes")

    with mp.Pool(num_processes) as pool:
        results = pool.map(plot_single_op, plot_args)

        for result in results:
            print(result)


def main():
    sns.set(
        style="ticks",
        rc={
            "axes.grid": True,
            # "axes.edgecolor": None,
            "axes.titlesize": 60,
            "axes.labelsize": 60,
            "xtick.labelsize": 60,
            "ytick.labelsize": 60,
            "font.size": 60,
            "legend.title_fontsize": 50,
            "legend.fontsize": 40,
            "lines.linewidth": 4,
            "axes.linewidth": 1,
            "font.serif": ["Latin Modern Math"],
        },
    )

    all_operations = [
        ("exp", "bfloat16", 32),
        ("exp_approx", "bfloat16", 32),
        ("pow", "bfloat16", 32),
        ("pow_2", "bfloat16", 32),
        ("pow_5", "bfloat16", 32),
        ("pow_10", "bfloat16", 32),
        ("log", "bfloat16", 32),
        ("tanh", "bfloat16", 32),
        ("cosh", "bfloat16", 32),
        ("sinh", "bfloat16", 32),
        ("log10", "bfloat16", 32),
        ("log2", "bfloat16", 32),
        ("log1p", "bfloat16", 32),
        ("silu", "bfloat16", 32),
        ("gelu", "bfloat16", 32),
        ("gelu_approx", "bfloat16", 32),
        ("logit", "bfloat16", 32),
        ("swish", "bfloat16", 32),
        ("mish", "bfloat16", 32),
        ("elu", "bfloat16", 32),
        ("selu", "bfloat16", 32),
        ("softplus", "bfloat16", 32),
        ("softsign", "bfloat16", 32),
        ("tan", "bfloat16", 32),
        ("atan", "bfloat16", 32),
        ("sin", "bfloat16", 32),
        ("cos", "bfloat16", 32),
        ("sqrt", "bfloat16", 32),
        ("rsqrt", "bfloat16", 32),
        ("rsqrt_approx", "bfloat16", 32),
        ("reciprocal", "bfloat16", 32),
        ("digamma", "bfloat16", 32),
        ("lgamma", "bfloat16", 32),
        ("tanhshrink", "bfloat16", 32),
    ]

    highres_operations = [
        ("exp", "bfloat16", 1),
        ("exp_approx", "bfloat16", 1),
        ("pow", "bfloat16", 1),
        ("pow_2", "bfloat16", 1),
        ("pow_5", "bfloat16", 1),
        ("pow_10", "bfloat16", 1),
        ("tanh", "bfloat16", 1),
        ("cosh", "bfloat16", 1),
        ("sinh", "bfloat16", 1),
        ("log", "bfloat16", 1),
        ("log10", "bfloat16", 1),
        ("log2", "bfloat16", 1),
        ("log1p", "bfloat16", 1),
        ("tan", "bfloat16", 1),
        ("atan", "bfloat16", 1),
        ("cos", "bfloat16", 1),
        ("sin", "bfloat16", 1),
        ("silu", "bfloat16", 1),
        ("selu", "bfloat16", 1),
        ("softplus", "bfloat16", 1),
        ("softsign", "bfloat16", 1),
        ("gelu", "bfloat16", 1),
        ("gelu_approx", "bfloat16", 1),
        ("logit", "bfloat16", 1),
        ("swish", "bfloat16", 1),
        ("mish", "bfloat16", 1),
        ("elu", "bfloat16", 1),
        ("sqrt", "bfloat16", 1),
        ("rsqrt_approx", "bfloat16", 1),
        ("rsqrt", "bfloat16", 1),
        ("reciprocal", "bfloat16", 1),
        ("digamma", "bfloat16", 1),
        ("lgamma", "bfloat16", 1),
        ("tanhshrink", "bfloat16", 1),
    ]
    powers = [(f"pow_{i}", "bfloat16", 1) for i in [2, 5, 10]]
    # all_operations += powers
    # highres_operations += powers

    accuracy_dir = "accuracy_results/results/"
    dest_dir = "accuracy_results/plots"

    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    # plot_all_ops(accuracy_dir, all_operations, dest_dir)
    # plot_all_ops(accuracy_dir, highres_operations, dest_dir, highres=True)

    if not os.path.exists(f"{dest_dir}/abs/"):
        os.makedirs(f"{dest_dir}/abs/")

    # plot_all_ops(f"{accuracy_dir}", all_operations, f"{dest_dir}/abs/", highres=False, plot_absolute=True)

    plot_config = parse_plot_config("eltwise-accuracy/plot-params.json")

    last_hashes = load_plot_config_hashes(f"accuracy_results/plot-hashes.csv")

    # Get time stamp of last modification of this script
    script_mtime = os.path.getmtime(__file__)
    force_replot = False
    if script_mtime > last_hashes["last_modified"]:
        force_replot = True

    current_hashes = generate_plot_config_hashes(plot_config)

    plot_all(plot_config, dest_dir, last_hashes, current_hashes, force_replot)

    save_plot_config_hashes(current_hashes, f"accuracy_results/plot-hashes.csv")


main()
