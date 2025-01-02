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


def generate_bin_name_remote_cb_sync(data_frame):
    df_dict = {
        0: "bfp8_b",
        1: "float16_b",
    }

    bin_names = []
    for (
        test,
        m,
        n,
        k,
        nblock,
        cb_nblock,
        cb_padding,
        data_format,
        num_receivers,
        num_mixed_df_layers,
    ) in data_frame.values:
        if test == "Matmul":
            # Matmul
            bin_names.append(
                f"Matmul:{m}x{k}x{n}\n({nblock},{cb_nblock},{cb_padding},{df_dict[data_format]},{num_receivers},{num_mixed_df_layers})"
            )
        else:
            # Dram cb read
            bin_names.append(
                f"DRAM:{k}x{n}\n({nblock},{cb_nblock},{cb_padding},{df_dict[data_format]},{num_receivers},{num_mixed_df_layers})"
            )
    return bin_names


def generate_bin_name_all_core(data_frame):
    df_dict = {
        0: "bfp8_b",
        1: "float16_b",
    }

    bin_names = []
    for k, n, nblock, data_format, num_banks in data_frame.values:
        if num_banks == 8:
            arch = "BH"
        else:
            arch = "WH"
        bin_names.append(f"{arch}:{k}x{n}\n({nblock},{df_dict[data_format]},{num_banks})")

    return bin_names


test_type = "remote_cb_sync"
bh_dir = "/Users/skrsmanovic/perf/yyzo-bh-04/run3/generated/profiler/.logs/"
wh_dir = "/Users/skrsmanovic/perf/yyz-jb-13/run4_800/generated/profiler/.logs/"

graph_title = f"DRAM read_{test_type} Bandwidth: WH[aiclk forced to 800MHz] vs BH[aiclk forced to 800MHz]"
y_label = "Bandwidth[GB/s]"
y_label_relative = "Relative BW[%]: BH/WH * 100"
x_label = {
    "all_core": "Input configuration in format\n{k}x{n}\n({nblock},{data_format},{num_banks})",
    "remote_cb_sync": "Input configuration in format:\n{test_type}:{m}x{k}x{n}\n({nblock},{cb_nblock},{cb_padding},{data_format},{num_receivers},{num_mixed_df_layers})",
    "l1_write_core": None,
}

analysis_data = {
    "all_core": ["Read throughput[GB/s]"],
    "remote_cb_sync": ["Read throughput[GB/s]"],
    "l1_write_core": ["Read throughput[GB/s]"],
}

bin_columns = {
    "all_core": ["k", "n", "nblock", "data_format", "num_banks"],
    "remote_cb_sync": [
        "test",
        "m",
        "n",
        "k",
        "nblock",
        "cb_nblock",
        "cb_padding",
        "data_format",
        "num_receivers",
        "num_mixed_df_layers",
    ],
    "l1_write_core": [],
}

group_columns = {
    "all_core": [],
    "remote_cb_sync": ["use_sub_devices"],
    "l1_write_core": [],
}

group_configurations = {
    "all_core": [None],
    "remote_cb_sync": [True, False],
    "l1_write_core": [],
}

csv_filenames = {
    "all_core": "moreh_dram_read_all_core.csv",
    "remote_cb_sync": "moreh_dram_read_remote_cb_sync.csv",
    "l1_write_core": "moreh_dram_read_l1_write_core.csv",
}

generate_bin_name = {
    "all_core": generate_bin_name_all_core,
    "remote_cb_sync": generate_bin_name_remote_cb_sync,
    "l1_write_core": None,
}

data_dict = {}
relative_dict = {}
wh_df = pd.read_csv(wh_dir + csv_filenames[test_type])
bh_df = pd.read_csv(bh_dir + csv_filenames[test_type])

# TODO provide more general solution: when there is no grouping of data or when there is more than 1 group
for group in group_configurations[test_type]:
    if group == None:
        wh_filt = wh_df
    else:
        wh_filt = wh_df[wh_df[group_columns[test_type][0]] == group]
    wh_data = wh_filt[analysis_data[test_type][0]].values
    wh_bin_name = generate_bin_name[test_type](wh_filt.loc[:, bin_columns[test_type][0] : bin_columns[test_type][-1]])

    if group == None:
        bh_filt = bh_df
    else:
        bh_filt = bh_df[bh_df[group_columns[test_type][0]] == group]
    bh_data = bh_filt[analysis_data[test_type][0]].values
    bh_bin_name = generate_bin_name[test_type](bh_filt.loc[:, bin_columns[test_type][0] : bin_columns[test_type][-1]])

    if test_type != "all_core":
        assert bh_bin_name == wh_bin_name, "Columns in .csv file for BH and WH are not same"
    else:
        combined_bin_name = [f"{x}\n{y}" for x, y in zip(wh_bin_name, bh_bin_name)]
        bh_bin_name = combined_bin_name

    if group == None:
        wh_key = "WH"
        bh_key = "BH"
        relative_key = "BH vs WH"
    else:
        wh_key = f"WH({group_columns[test_type][0]}={group})"
        bh_key = f"BH({group_columns[test_type][0]}={group})"
        relative_key = f"{group_columns[test_type][0]}={group}"

    data_dict[wh_key] = wh_data
    data_dict[bh_key] = bh_data
    relative_dict[relative_key] = (bh_data / wh_data) * 100

plot_data(data_dict, graph_title, x_label[test_type], y_label, bh_bin_name)
plot_data(relative_dict, graph_title, x_label[test_type], y_label_relative, bh_bin_name, True)
