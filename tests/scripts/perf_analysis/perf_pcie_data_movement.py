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

    fig, ax = plt.subplots(layout="tight")

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
    plt.xticks(rotation=45, ha="right")
    ax.legend(loc="upper left", ncols=num_of_bins // rel_factor)
    fig.set_size_inches(17, 10)
    plt.savefig(f"{graph_title}.png", dpi=150)


test_type = "H2D DRAM"
# test_type = "D2H DRAM"
# test_type = "H2D L1"
# test_type = "D2H L1"
bh_dir = "/proj_sw/user_dev/rdjogo/work/blackhole/tt-metal/"
wh_dir = "/proj_sw/user_dev/rdjogo/work/wormhole_b0/tt-metal/"

graph_title = f"{test_type} Bandwidth: WH[aiclk forced to 800MHz] vs BH[aiclk forced to 800MHz]"
y_label = "Bandwidth[GB/s]"
x_label = "Transfer size in bytes"
y_label_relative = "Relative BW[%]: BH/WH * 100"

input_sizes = {
    "H2D DRAM": [8192, 32768, 131072, 524288, 2097152, 8388608, 33554432, 134217728, 536870912],
    "D2H DRAM": [8192, 32768, 131072, 524288, 2097152, 8388608, 33554432, 134217728, 536870912],
    "H2D L1": [4096, 16384, 65536, 262144, 1048576, 4194304, 16777216],
    "D2H L1": [4096, 16384, 65536],
}

data_groups = {
    "H2D DRAM": ["WriteToDeviceDRAMChannel", "WriteToBuffer", "EnqueueWriteBuffer"],
    "D2H DRAM": ["ReadFromDeviceDRAMChannel", "ReadFromBuffer", "EnqueueReadBuffer"],
    "H2D L1": ["WriteToDeviceL1", "WriteToBuffer", "EnqueueWriteBuffer"],
    "D2H L1": ["ReadFromDeviceL1", "ReadFromBuffer", "EnqueueReadBuffer"],
}

csv_filenames = {
    "H2D DRAM": "moreh_old_H2D_DRAM_Bandwidth.csv",
    "D2H DRAM": "moreh_old_D2H_DRAM_Bandwidth.csv",
    "H2D L1": "moreh_old_H2D_L1_Bandwidth.csv",
    "D2H L1": "moreh_old_D2H_L1_Bandwidth.csv",
}

data_dict = {}
relative_dict = {}
wh_df = pd.read_csv(wh_dir + csv_filenames[test_type])
bh_df = pd.read_csv(bh_dir + csv_filenames[test_type])

# TODO put BH, WH inside for loop to not repeat same code
for data in data_groups[test_type]:
    wh_data = []
    bh_data = []
    for size in input_sizes[test_type]:
        # WH data
        wh_filt = wh_df[(wh_df["Transfer Size"] == size)]
        assert wh_filt.shape[0] < 2, f"Mutliple results of same data size found in WH .csv file: size = {size}"
        assert data in wh_df.columns, f"Column with name: {data} not available in WH .csv file"
        if wh_filt.empty:
            # No data for this size
            wh_data.append(np.nan)
        else:
            wh_data.append(wh_filt[data].values[0])

        # BH data
        bh_filt = bh_df[(bh_df["Transfer Size"] == size)]
        assert bh_filt.shape[0] < 2, f"Mutliple results of same data size found in BH .csv file: size = {size}"
        assert data in bh_df.columns, f"Column with name: {data} not available in BH .csv file"
        if bh_filt.empty:
            # No data for this size
            bh_data.append(np.nan)
        else:
            bh_data.append(bh_filt[data].values[0])

    data_dict[f"WH.{data}"] = wh_data
    data_dict[f"BH.{data}"] = bh_data
    relative_dict[f"{data}"] = (np.array(bh_data) / np.array(wh_data)) * 100

plot_data(data_dict, graph_title + " Abs", x_label, y_label, input_sizes[test_type])
plot_data(relative_dict, graph_title + " Rel", x_label, y_label_relative, input_sizes[test_type], True)
