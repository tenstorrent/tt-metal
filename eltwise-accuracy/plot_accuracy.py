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
    short_name = plot_entry["name"] if "name" in plot_entry else id

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

    plot_type = plot_entry["type"] if "type" in plot_entry else "lineplot"
    print(f"Plot type: {id} - {plot_type}")

    # Remove data that exceeds xmin,xmax,ymin,ymax
    if xmin is not None:
        data = data[data[xname] >= xmin]
    if xmax is not None:
        data = data[data[xname] <= xmax]

    fig, ax = plt.subplots(figsize=(25, 15))

    # color_palette = sns.color_palette("deep", len(ynames))
    ncolors = len(data[hseries].unique())
    color_palette = sns.color_palette("deep", ncolors + palette_offset)[palette_offset:]

    for y in ynames:
        d2 = data.copy()

        [yname, ysuffix, linestyle] = y
        d2["operation"] += " " + ysuffix

        # Remove data that exceeds ymin,ymax
        if ymin is not None:
            d2 = d2[d2[yname] >= ymin]
        if ymax is not None:
            d2 = d2[d2[yname] <= ymax]

        if plot_type == "lineplot":
            if hseries is not None:
                ax = sns.lineplot(
                    data=d2, x=xname, y=yname, ax=ax, hue=hseries, linestyle=linestyle, palette=color_palette
                )
            else:
                ax = sns.lineplot(
                    data=d2, x=xname, y=yname, ax=ax, label=yname, linestyle=linestyle, palette=color_palette
                )
        elif plot_type == "scatterplot":
            ax = sns.scatterplot(data=d2, x=xname, y=yname, ax=ax, hue=hseries, palette=color_palette, edgecolor="none")

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
            label_y = ax.get_ylim()[1] / 2
            ax.text(vertical_line[0], label_y, vertical_line[1])

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
            "lines.markersize": 8,
            "lines.markeredgecolor": "none",
        },
    )

    accuracy_dir = "accuracy_results/results/"
    dest_dir = "accuracy_results/plots"

    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

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
