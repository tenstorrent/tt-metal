import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import colormaps as cmap


# TODO this shoul be part of perf measurement script - workaround for already collected results
def get_ideal_cycles_per_tile(fidelity="LoFi"):
    if fidelity == "HiFi2":
        return 2 * 16
    elif fidelity == "HiFi3":
        return 3 * 16
    elif fidelity == "HiFi4":
        return 4 * 16
    else:
        return 16


# TODO this shoul be part of perf measurement script - workaround for already collected results
matmul_block_cycles = {
    (512, 512, 512, "BFLOAT16", "HiFi2"): 256,
    (512, 1024, 1024, "BFLOAT16", "HiFi2"): 1024,
    (512, 1024, 2048, "BFLOAT16", "HiFi2"): 2048,
    (1024, 1024, 1024, "BFLOAT16", "HiFi2"): 2048,
    (1024, 1024, 2048, "BFLOAT16", "HiFi2"): 4096,
    (1024, 2048, 2048, "BFLOAT16", "HiFi2"): 8192,
    (2048, 2048, 2048, "BFLOAT16", "HiFi2"): 16384,
    (2048, 2048, 3072, "BFLOAT16", "HiFi2"): 24576,
    (2048, 3072, 3072, "BFLOAT16", "HiFi2"): 18432,
    (3072, 3072, 3072, "BFLOAT16", "HiFi2"): 13824,
    (3072, 3072, 4096, "BFLOAT16", "HiFi2"): 36864,
    (3072, 4096, 4096, "BFLOAT16", "HiFi2"): 49152,
    (4096, 4096, 4096, "BFLOAT16", "HiFi2"): 32768,
    (8192, 8192, 8192, "BFLOAT16", "HiFi2"): 32768,
    (16384, 16384, 16384, "BFLOAT16", "HiFi2"): 32768,
    (512, 512, 512, "BFLOAT16", "HiFi4"): 512,
    (512, 1024, 1024, "BFLOAT16", "HiFi4"): 2048,
    (512, 1024, 2048, "BFLOAT16", "HiFi4"): 4096,
    (1024, 1024, 1024, "BFLOAT16", "HiFi4"): 4096,
    (1024, 1024, 2048, "BFLOAT16", "HiFi4"): 8192,
    (1024, 2048, 2048, "BFLOAT16", "HiFi4"): 16384,
    (2048, 2048, 2048, "BFLOAT16", "HiFi4"): 32768,
    (2048, 2048, 3072, "BFLOAT16", "HiFi4"): 49152,
    (2048, 3072, 3072, "BFLOAT16", "HiFi4"): 36864,
    (3072, 3072, 3072, "BFLOAT16", "HiFi4"): 27648,
    (3072, 3072, 4096, "BFLOAT16", "HiFi4"): 73728,
    (3072, 4096, 4096, "BFLOAT16", "HiFi4"): 98304,
    (4096, 4096, 4096, "BFLOAT16", "HiFi4"): 65536,
    (8192, 8192, 8192, "BFLOAT16", "HiFi4"): 65536,
    (16384, 16384, 16384, "BFLOAT16", "HiFi4"): 65536,
    (512, 512, 512, "BFLOAT8_B", "HiFi2"): 256,
    (512, 1024, 1024, "BFLOAT8_B", "HiFi2"): 1024,
    (512, 1024, 2048, "BFLOAT8_B", "HiFi2"): 2048,
    (1024, 1024, 1024, "BFLOAT8_B", "HiFi2"): 2048,
    (1024, 1024, 2048, "BFLOAT8_B", "HiFi2"): 4096,
    (1024, 2048, 2048, "BFLOAT8_B", "HiFi2"): 8192,
    (2048, 2048, 2048, "BFLOAT8_B", "HiFi2"): 16384,
    (2048, 2048, 3072, "BFLOAT8_B", "HiFi2"): 24576,
    (2048, 3072, 3072, "BFLOAT8_B", "HiFi2"): 36864,
    (3072, 3072, 3072, "BFLOAT8_B", "HiFi2"): 27648,
    (3072, 3072, 4096, "BFLOAT8_B", "HiFi2"): 36864,
    (3072, 4096, 4096, "BFLOAT8_B", "HiFi2"): 24576,
    (4096, 4096, 4096, "BFLOAT8_B", "HiFi2"): 32768,
    (8192, 8192, 8192, "BFLOAT8_B", "HiFi2"): 32768,
    (16384, 16384, 16384, "BFLOAT8_B", "HiFi2"): 32768,
    (512, 512, 512, "BFLOAT8_B", "LoFi"): 128,
    (512, 1024, 1024, "BFLOAT8_B", "LoFi"): 512,
    (512, 1024, 2048, "BFLOAT8_B", "LoFi"): 1024,
    (1024, 1024, 1024, "BFLOAT8_B", "LoFi"): 1024,
    (1024, 1024, 2048, "BFLOAT8_B", "LoFi"): 2048,
    (1024, 2048, 2048, "BFLOAT8_B", "LoFi"): 4096,
    (2048, 2048, 2048, "BFLOAT8_B", "LoFi"): 8192,
    (2048, 2048, 3072, "BFLOAT8_B", "LoFi"): 12288,
    (2048, 3072, 3072, "BFLOAT8_B", "LoFi"): 18432,
    (3072, 3072, 3072, "BFLOAT8_B", "LoFi"): 13824,
    (3072, 3072, 4096, "BFLOAT8_B", "LoFi"): 18432,
    (3072, 4096, 4096, "BFLOAT8_B", "LoFi"): 12288,
    (4096, 4096, 4096, "BFLOAT8_B", "LoFi"): 16384,
    (8192, 8192, 8192, "BFLOAT8_B", "LoFi"): 16384,
    (16384, 16384, 16384, "BFLOAT8_B", "LoFi"): 16384,
    (512, 512, 512, "BFLOAT4_B", "LoFi"): 128,
    (512, 1024, 1024, "BFLOAT4_B", "LoFi"): 512,
    (512, 1024, 2048, "BFLOAT4_B", "LoFi"): 1024,
    (1024, 1024, 1024, "BFLOAT4_B", "LoFi"): 1024,
    (1024, 1024, 2048, "BFLOAT4_B", "LoFi"): 2048,
    (1024, 2048, 2048, "BFLOAT4_B", "LoFi"): 4096,
    (2048, 2048, 2048, "BFLOAT4_B", "LoFi"): 8192,
    (2048, 2048, 3072, "BFLOAT4_B", "LoFi"): 12288,
    (2048, 3072, 3072, "BFLOAT4_B", "LoFi"): 18432,
    (3072, 3072, 3072, "BFLOAT4_B", "LoFi"): 27648,
    (3072, 3072, 4096, "BFLOAT4_B", "LoFi"): 36864,
    (3072, 4096, 4096, "BFLOAT4_B", "LoFi"): 24576,
    (4096, 4096, 4096, "BFLOAT4_B", "LoFi"): 32768,
    (8192, 8192, 8192, "BFLOAT4_B", "LoFi"): 65536,
    (16384, 16384, 16384, "BFLOAT4_B", "LoFi"): 65536,
}


def get_matmul_block_cycles(m, k, n, data_type, fidelity):
    condition = tuple([m, k, n, data_type, fidelity])
    return matmul_block_cycles.get(condition)


def get_data(df, data_groups, input_sizes, data_column, grid_size=[8, 8], use_trace=True):
    for data in data_groups:
        data_values = []
        for size in input_sizes:
            df_filt = df[
                (df["m"] == (size[0] // 8 * grid_size[1]))
                & (df["k"] == (size[1] // 8 * grid_size[0]))
                & (df["n"] == (size[2] // 8 * grid_size[0]))
                & (df["dtype"] == ("DataType." + data[0]))
                & (df["math_fidelity"] == ("MathFidelity." + data[1]))
                & (df["use_trace"] == use_trace)
            ]
            assert (
                df_filt.shape[0] < 2
            ), f"Mutliple results of same matmul configuration found in .csv file: size({size[0]}, {size[1]}, {size[2]}) data({data[0]}, {data[1]})"
            if df_filt.empty:
                data_values.append(np.nan)
            else:
                data_value = df_filt[data_column].values[0]
                if "utilization" in data_column.lower():
                    # Extract float number from percentage string
                    data_value = float(data_value.strip("%"))
                elif "trisc1_kernel_cycles" in data_column.lower():
                    # Calculate ideal number of cycles to get utilization
                    # TODO this shoul be part of perf measurement script, here only percentage value should be read
                    ideal_cycles = (
                        df_filt["m"].values[0]
                        * df_filt["n"].values[0]
                        * df_filt["k"].values[0]
                        / 32
                        / 32
                        / 32
                        * get_ideal_cycles_per_tile(data[1])
                        / (grid_size[0] * grid_size[1])
                    )
                    data_value = ideal_cycles / data_value * 100
                elif "trisc0_math_block_cycles" in data_column.lower():
                    # Calculate ideal number of cycles to get utilization
                    # TODO this shoul be part of perf measurement script, here only percentage value should be read
                    # Here size[] is used as there is no difference in grid size (matrix size is scled so parameters of matmul are same)
                    ideal_cycles = get_matmul_block_cycles(size[0], size[1], size[2], data[0], data[1])
                    data_value = ideal_cycles / data_value * 100
                data_values.append(data_value)
        data_dict[f"{data[0]}.{data[1]}"] = data_values

    return data_dict


def plot_data(data_dict, graph_title, x_label, y_label, x_ticks, output_dir, plot_name, relative_plot=False, yticks=[]):
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
    if len(yticks) != 0:
        ax.set_yticks(yticks)
    plt.xticks(rotation=45, ha="right")
    ax.legend(loc="upper left", ncols=num_of_bins // rel_factor)
    fig.set_size_inches(17, 10)
    plt.savefig(f"{output_dir}/{plot_name}.png", dpi=150)


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


# User data
test_type = "regular"
results_root = "/localdev/skrsmanovic/gitrepos/tt-metal-profiler/perf_results/bh_testing/8x8/1350M/"
y_label = "TFLOPs"
x_label = "MxKxN input matrix dimension"
graph_title = f"Matmul sweep {test_type}"
plot_column = "TFLOPs (avg)"
yticks_tflops = np.arange(0, 501, 20)
yticks_percent = np.arange(0, 101, 5)
grid_size = [8, 8]

##############################################
# Analyze data collected without profiler logs
data_dict = {}
relative_dict = {}
data_dir = results_root + "no_profiler/"
data_df = pd.read_csv(data_dir + csv_filenames[test_type])
# Remove skipped test conditions from csv file
data_df.iloc[:, 0] = data_df.iloc[:, 0].astype(str)
data_df = data_df[~data_df.iloc[:, 0].str.contains("SKIPPED", na=False)]
data_df["m"] = pd.to_numeric(data_df["m"])

# Get and plot TFLOPs data
plot_column = "TFLOPs (avg)"
y_label = "TFLOPs"
yticks = yticks_tflops
data = get_data(data_df, data_groups[test_type], input_sizes[test_type], plot_column, grid_size)
plot_data(
    data,
    graph_title + ": TFLOPs measured by host - using trace",
    x_label,
    y_label,
    [
        f"{x[0] // 8 * grid_size[1]}x{x[1] // 8 * grid_size[0]}x{x[2] // 8 * grid_size[0]}"
        for x in input_sizes[test_type]
    ],
    data_dir,
    "tflops",
    True,
    yticks,
)

# Get and plot host utilization data with trace (replay buffer)
plot_column = f"Utilization (vs {grid_size[0]}x{grid_size[1]} user grid)"
y_label = "Utilization in %"
yticks = yticks_percent
data = get_data(data_df, data_groups[test_type], input_sizes[test_type], plot_column, grid_size, True)
plot_data(
    data,
    graph_title + ": Utilization based on time measured by host - using trace",
    x_label,
    y_label,
    [
        f"{x[0] // 8 * grid_size[1]}x{x[1] // 8 * grid_size[0]}x{x[2] // 8 * grid_size[0]}"
        for x in input_sizes[test_type]
    ],
    data_dir,
    "utilization_host_trace",
    True,
    yticks,
)

# Get and plot host utilization data without trace (replay buffer)
plot_column = f"Utilization (vs {grid_size[0]}x{grid_size[1]} user grid)"
y_label = "Utilization in %"
yticks = yticks_percent
data = get_data(data_df, data_groups[test_type], input_sizes[test_type], plot_column, grid_size, False)
plot_data(
    data,
    graph_title + ": Utilization based time measured by host - not using trace",
    x_label,
    y_label,
    [
        f"{x[0] // 8 * grid_size[1]}x{x[1] // 8 * grid_size[0]}x{x[2] // 8 * grid_size[0]}"
        for x in input_sizes[test_type]
    ],
    data_dir,
    "utilization_host_no_trace",
    True,
    yticks,
)


##############################################
# Analyze data collected with profiler logs - no extra profiler zones
data_dict = {}
relative_dict = {}
data_dir = results_root + "profiler_all_iterations/"
data_df = pd.read_csv(data_dir + csv_filenames[test_type])
# Remove skipped test conditions from csv file
data_df.iloc[:, 0] = data_df.iloc[:, 0].astype(str)
data_df = data_df[~data_df.iloc[:, 0].str.contains("SKIPPED", na=False)]
data_df["m"] = pd.to_numeric(data_df["m"])

# Get and plot TFLOPs data
plot_column = "TFLOPs (avg)"
y_label = "TFLOPs"
yticks = yticks_tflops
data = get_data(data_df, data_groups[test_type], input_sizes[test_type], plot_column, grid_size)
plot_data(
    data,
    graph_title + ": TFLOPs measured by host - using trace",
    x_label,
    y_label,
    [
        f"{x[0] // 8 * grid_size[1]}x{x[1] // 8 * grid_size[0]}x{x[2] // 8 * grid_size[0]}"
        for x in input_sizes[test_type]
    ],
    data_dir,
    "tflops",
    True,
    yticks,
)

# Get and plot host utilization data with trace (replay buffer)
plot_column = f"Utilization (vs {grid_size[0]}x{grid_size[1]} user grid)"
y_label = "Utilization in %"
yticks = yticks_percent
data = get_data(data_df, data_groups[test_type], input_sizes[test_type], plot_column, grid_size, True)
plot_data(
    data,
    graph_title + ": Utilization based on time measured by host - using trace",
    x_label,
    y_label,
    [
        f"{x[0] // 8 * grid_size[1]}x{x[1] // 8 * grid_size[0]}x{x[2] // 8 * grid_size[0]}"
        for x in input_sizes[test_type]
    ],
    data_dir,
    "utilization_host_trace",
    True,
    yticks,
)

# Get and plot host utilization data without trace (replay buffer)
plot_column = f"Utilization (vs {grid_size[0]}x{grid_size[1]} user grid)"
y_label = "Utilization in %"
yticks = yticks_percent
data = get_data(data_df, data_groups[test_type], input_sizes[test_type], plot_column, grid_size, False)
plot_data(
    data,
    graph_title + ": Utilization based on time measured by host - not using trace",
    x_label,
    y_label,
    [
        f"{x[0] // 8 * grid_size[1]}x{x[1] // 8 * grid_size[0]}x{x[2] // 8 * grid_size[0]}"
        for x in input_sizes[test_type]
    ],
    data_dir,
    "utilization_host_no_trace",
    True,
    yticks,
)

# Get and plot math kernel utilization data with trace (replay buffer)
plot_column = "trisc1_kernel_cycles"
y_label = "Utilization in %"
yticks = yticks_percent
data = get_data(data_df, data_groups[test_type], input_sizes[test_type], plot_column, grid_size, True)
plot_data(
    data,
    graph_title + ": Utilization based on math kernel cycles measured by profiler - using trace",
    x_label,
    y_label,
    [
        f"{x[0] // 8 * grid_size[1]}x{x[1] // 8 * grid_size[0]}x{x[2] // 8 * grid_size[0]}"
        for x in input_sizes[test_type]
    ],
    data_dir,
    "utilization_kernel_trace",
    True,
    yticks,
)

# Get and plot math kernel utilization data without trace (replay buffer)
plot_column = "trisc1_kernel_cycles"
y_label = "Utilization in %"
yticks = yticks_percent
data = get_data(data_df, data_groups[test_type], input_sizes[test_type], plot_column, grid_size, False)
plot_data(
    data,
    graph_title + ": Utilization based on math kernel cycles measured by profiler - not using trace",
    x_label,
    y_label,
    [
        f"{x[0] // 8 * grid_size[1]}x{x[1] // 8 * grid_size[0]}x{x[2] // 8 * grid_size[0]}"
        for x in input_sizes[test_type]
    ],
    data_dir,
    "utilization_kernel_no_trace",
    True,
    yticks,
)


##############################################
# Analyze data collected with profiler logs - no extra profiler zones
data_dict = {}
relative_dict = {}
data_dir = results_root + "profiler_single_iteration_math/"
data_df = pd.read_csv(data_dir + csv_filenames[test_type])
# Remove skipped test conditions from csv file
data_df.iloc[:, 0] = data_df.iloc[:, 0].astype(str)
data_df = data_df[~data_df.iloc[:, 0].str.contains("SKIPPED", na=False)]
data_df["m"] = pd.to_numeric(data_df["m"])

# Get and plot TFLOPs data
plot_column = "TFLOPs (avg)"
y_label = "TFLOPs"
yticks = yticks_tflops
data = get_data(data_df, data_groups[test_type], input_sizes[test_type], plot_column, grid_size, False)
plot_data(
    data,
    graph_title + ": TFLOPs measured by host - not using trace",
    x_label,
    y_label,
    [
        f"{x[0] // 8 * grid_size[1]}x{x[1] // 8 * grid_size[0]}x{x[2] // 8 * grid_size[0]}"
        for x in input_sizes[test_type]
    ],
    data_dir,
    "tflops",
    True,
    yticks,
)

# Get and plot host utilization data without trace (replay buffer)
plot_column = f"Utilization (vs {grid_size[0]}x{grid_size[1]} user grid)"
y_label = "Utilization in %"
yticks = yticks_percent
data = get_data(data_df, data_groups[test_type], input_sizes[test_type], plot_column, grid_size, False)
plot_data(
    data,
    graph_title + ": Utilization based on time measured by host - not using trace",
    x_label,
    y_label,
    [
        f"{x[0] // 8 * grid_size[1]}x{x[1] // 8 * grid_size[0]}x{x[2] // 8 * grid_size[0]}"
        for x in input_sizes[test_type]
    ],
    data_dir,
    "utilization_host_no_trace",
    True,
    yticks,
)

# Get and plot math kernel utilization data without trace (replay buffer)
plot_column = "trisc1_kernel_cycles"
y_label = "Utilization in %"
yticks = yticks_percent
data = get_data(data_df, data_groups[test_type], input_sizes[test_type], plot_column, grid_size, False)
plot_data(
    data,
    graph_title + ": Utilization based on math kernel cycles measured by profiler - not using trace",
    x_label,
    y_label,
    [
        f"{x[0] // 8 * grid_size[1]}x{x[1] // 8 * grid_size[0]}x{x[2] // 8 * grid_size[0]}"
        for x in input_sizes[test_type]
    ],
    data_dir,
    "utilization_kernel_no_trace",
    True,
    yticks,
)

# Get and plot matmul block utilization data without trace (replay buffer)
plot_column = "trisc0_math_block_cycles"
y_label = "Utilization in %"
yticks = yticks_percent
data = get_data(data_df, data_groups[test_type], input_sizes[test_type], plot_column, grid_size, False)
plot_data(
    data,
    graph_title + ": Utilization based on matmul block cycles measured by profiler - not using trace",
    x_label,
    y_label,
    [
        f"{x[0] // 8 * grid_size[1]}x{x[1] // 8 * grid_size[0]}x{x[2] // 8 * grid_size[0]}"
        for x in input_sizes[test_type]
    ],
    data_dir,
    "utilization_matmul_block_no_trace",
    True,
    yticks,
)
