import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import colormaps as cmap


def plot_data(data_dict, graph_title, x_label, y_label, x_ticks, relative_plot=False):
    rel_factor = 2 if relative_plot == False else 1

    num_of_groups = len(list(data_dict.values())[0])
    x = np.arange(num_of_groups)

    num_of_bins = len(list(data_dict.keys()))
    num_of_spaces = (num_of_bins / rel_factor) - 1
    space_width = 0.2
    group_width = (num_of_bins + (num_of_spaces * space_width)) * 1.3
    width = 1 / group_width
    x_tick_offset = ((num_of_bins + (num_of_spaces * space_width)) / 2 - 0.5) * width
    multiplier = 0

    fig, ax = plt.subplots(layout="constrained")

    for attribute, measurement in data_dict.items():
        offset = width * multiplier
        offset += (multiplier // rel_factor) * space_width * width

        hatch = "x" if multiplier % 2 == 1 and relative_plot == False else None
        color = cmap["Dark2"].colors[multiplier // rel_factor]

        rects = ax.bar(
            x + offset, measurement, width, label=attribute, hatch=hatch, color=color, edgecolor="black", zorder=3
        )
        multiplier += 1
    ax.set_ylabel(y_label)
    ax.set_xlabel(x_label)
    ax.set_title(graph_title)
    ax.grid(axis="y", linestyle="--", alpha=0.7, zorder=0)
    ax.set_xticks(x + x_tick_offset, x_ticks)
    # plt.xticks(rotation=45, ha='right')
    ax.legend(loc="best", ncols=num_of_bins // rel_factor)
    plt.show()


def generate_bin_name_matmul_dram(data_frame):
    bin_names = []
    for m, n, k in data_frame.values:
        bin_names.append(f"{m}x{n}x{k}")
    return bin_names


def generate_bin_name_matmul_sram(data_frame):
    bin_names = []
    for counter, [m, n, k] in enumerate(data_frame.values):
        test_type = "global" if counter < 2 else "local"
        bin_names.append(f"{m}x{n}x{k} - {test_type}")
    return bin_names


def generate_bin_name_matmul_sharded(data_frame):
    bin_names = []
    for (
        m,
        n,
        k,
        dtype,
        fidel,
        matmul_block,
        num_blocks,
        packer_l1_acc,
        fp32_dest_acc,
        interm_cb_dtype,
        subblock_index,
    ) in data_frame.values:
        bin_names.append(
            f"{m}x{n}x{k}\n({dtype},{fidel},{matmul_block},{num_blocks},{packer_l1_acc},{fp32_dest_acc},{interm_cb_dtype},{subblock_index})"
        )
    return bin_names


test_type = "single_core_sharded"
bh_dir = "/Users/skrsmanovic/perf/yyzo-bh-04/run3/generated/profiler/.logs/"
wh_dir = "/Users/skrsmanovic/perf/yyz-jb-13/run4_800/generated/profiler/.logs/"

graph_title = f"Matmul {test_type} Throughput: WH[aiclk forced to 800MHz] vs BH[aiclk forced to 800MHz]"
y_label = "Throughput[TFLOPs]"
x_label = {
    "dram": "Input configuration in format:\n({m}x{n}x{k})",
    "sram": "Input configuration in format:\n({m}x{n}x{k}) - {test_type}",
    "single_core_sharded": "Input configuration in format:\n{m}x{n}x{k}\n({dtype},{fidel},{matmul_block},{num_blocks},{packer_l1_acc},{fp32_dest_acc},{interm_cb_dtype},{subblock_index})",
}
y_label_relative = "Relative Throughput[%]: BH/WH * 100"

analysis_data = {
    "dram": ["TFLOPs"],
    "sram": ["TFLOPs"],
    "single_core_sharded": ["TFLOPs"],
}

bin_columns = {
    "dram": ["M", "N", "K"],
    "sram": ["M", "N", "K"],
    "single_core_sharded": [
        "m",
        "n",
        "k",
        "dtype",
        "fidel",
        "matmul_block",
        "num_blocks",
        "packer_l1_acc",
        "fp32_dest_acc",
        "interm_cb_dtype",
        "subblock_index",
    ],
}

csv_filenames = {
    "dram": "moreh_old_Matmul_DRAM.csv",
    "sram": "moreh_old_Matmul_SRAM.csv",
    "single_core_sharded": "moreh_single_core_Matmul_Sharded.csv",
}

generate_bin_name = {
    "dram": generate_bin_name_matmul_dram,
    "sram": generate_bin_name_matmul_sram,
    "single_core_sharded": generate_bin_name_matmul_sharded,
}

data_dict = {}
relative_dict = {}
wh_df = pd.read_csv(wh_dir + csv_filenames[test_type])
bh_df = pd.read_csv(bh_dir + csv_filenames[test_type])

wh_data = wh_df[analysis_data[test_type][0]].values
wh_bin_name = generate_bin_name[test_type](wh_df.loc[:, bin_columns[test_type][0] : bin_columns[test_type][-1]])

bh_data = bh_df[analysis_data[test_type][0]].values
bh_bin_name = generate_bin_name[test_type](bh_df.loc[:, bin_columns[test_type][0] : bin_columns[test_type][-1]])

assert bh_bin_name == wh_bin_name, "Columns in .csv file for BH and WH are not same"

data_dict[f"WH"] = wh_data
data_dict[f"BH"] = bh_data
relative_dict[f"BH vs WH"] = (bh_data / wh_data) * 100

plot_data(data_dict, graph_title, x_label[test_type], y_label, bh_bin_name)
plot_data(relative_dict, graph_title, x_label[test_type], y_label_relative, bh_bin_name, True)
