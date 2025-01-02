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
    plt.xticks(rotation=45, ha="right")
    ax.legend(loc="upper left", ncols=num_of_bins // rel_factor)
    plt.show()


test_type = "out of box"
wh_dir = "/Users/skrsmanovic/perf/yyz-jb-13/run1_800/generated/"
bh_dir = "/Users/skrsmanovic/perf/yyzo-bh-05/run1/"

y_label = "TFLOPs"
y_label_relative = "Relative TFLOPs[%]: BH/WH * 100"
x_label = "MxKxN input matrix dimension"
graph_title = f"Matmul sweep {test_type}: WH[aiclk forced to 800MHz] vs BH[aiclk forced to 800MHz]"

# Basic sweep config
input_sizes = {
    "regular": [
        (512, 512, 512),
        (512, 1024, 1024),
        (512, 1024, 2048),
        (1024, 1024, 1024),
        (1024, 1024, 2048),
        (1024, 2048, 2048),
        (2048, 2048, 2048),
        (2048, 2048, 3072),
        (2048, 3072, 3072),
        (3072, 3072, 3072),
        (3072, 3072, 4096),
        (3072, 4096, 4096),
        (4096, 4096, 4096),
        (8192, 8192, 8192),
        (16384, 16384, 16384),
    ],
    "out of box": [
        (512, 512, 512),
        (512, 1024, 1024),
        (512, 1024, 2048),
        (1024, 1024, 1024),
        (1024, 1024, 2048),
        (1024, 2048, 2048),
        (2048, 2048, 2048),
        (2048, 2048, 3072),
        (2048, 3072, 3072),
        (3072, 3072, 3072),
        (3072, 3072, 4096),
        (3072, 4096, 4096),
        (4096, 4096, 4096),
    ],
}

data_groups = {
    "regular": [
        ("BFLOAT16", "HiFi2"),
        ("BFLOAT16", "HiFi4"),
        ("BFLOAT8_B", "HiFi2"),
        ("BFLOAT8_B", "LoFi"),
        ("BFLOAT4_B", "LoFi"),
    ],
    "out of box": [
        ("BFLOAT16", "HiFi2"),
        ("BFLOAT8_B", "LoFi"),
        ("BFLOAT4_B", "LoFi"),
    ],
}

csv_filenames = {
    "regular": "matmul_2d_host_perf_report.csv",
    "out of box": "matmul_2d_host_perf_out_of_box_report.csv",
}

data_dict = {}
relative_dict = {}
wh_df = pd.read_csv(wh_dir + csv_filenames[test_type])
bh_df = pd.read_csv(bh_dir + csv_filenames[test_type])
plot_column = "TFLOPs (avg)"

# TODO put BH, WH inside for loop to not repeat same code
for data in data_groups[test_type]:
    wh_exec_time = []
    bh_exec_time = []
    for size in input_sizes[test_type]:
        wh_filt = wh_df[
            (wh_df["m"] == size[0])
            & (wh_df["k"] == size[1])
            & (wh_df["n"] == size[2])
            & (wh_df["dtype"] == ("DataType." + data[0]))
            & (wh_df["math_fidelity"] == ("MathFidelity." + data[1]))
            & (wh_df["use_trace"] == True)
        ]
        assert (
            wh_filt.shape[0] < 2
        ), f"Mutliple results of same matmul configuration found in WH .csv file: size({size[0]}, {size[1]}, {size[2]}) data({data[0]}, {data[1]})"
        if wh_filt.empty:
            wh_exec_time.append(np.nan)
        else:
            wh_exec_time.append(wh_filt[plot_column].values[0])

        bh_filt = bh_df[
            (bh_df["m"] == size[0])
            & (bh_df["k"] == size[1])
            & (bh_df["n"] == size[2])
            & (bh_df["dtype"] == ("DataType." + data[0]))
            & (bh_df["math_fidelity"] == ("MathFidelity." + data[1]))
            & (bh_df["use_trace"] == True)
        ]
        assert (
            bh_filt.shape[0] < 2
        ), f"Mutliple results of same matmul configuration found in BH .csv file: size({size[0]}, {size[1]}, {size[2]}) data({data[0]}, {data[1]})"
        if bh_filt.empty:
            bh_exec_time.append(np.nan)
        else:
            bh_exec_time.append(bh_filt[plot_column].values[0])
    data_dict[f"WH.{data[0]}.{data[1]}"] = wh_exec_time
    data_dict[f"BH.{data[0]}.{data[1]}"] = bh_exec_time
    relative_dict[f"{data[0]}.{data[1]}"] = (np.array(bh_exec_time) / np.array(wh_exec_time)) * 100

plot_data(data_dict, graph_title, x_label, y_label, [f"{x[0]}x{x[1]}x{x[2]}" for x in input_sizes[test_type]])
plot_data(
    relative_dict,
    graph_title,
    x_label,
    y_label_relative,
    [f"{x[0]}x{x[1]}x{x[2]}" for x in input_sizes[test_type]],
    True,
)
