# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import os
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import itertools
import matplotlib.ticker as mticker
from loguru import logger  # type: ignore

from tests.tt_metal.tt_metal.data_movement._python_script.constants import *


class Plotter:
    def __init__(self, dm_stats, aggregate_stats, output_dir, arch, test_id_to_name, test_id_to_comment):
        self.dm_stats = dm_stats
        self.aggregate_stats = aggregate_stats
        self.output_dir = output_dir
        self.arch = arch
        self.test_id_to_name = test_id_to_name
        self.test_id_to_comment = test_id_to_comment

    def get_plot_config(self):
        """Returns plot configuration parameters as a dictionary"""
        return {
            "plot_width": DEFAULT_PLOT_WIDTH,  # Width of an individual plot
            "plot_height": DEFAULT_PLOT_HEIGHT,  # Height of an individual plot
            "nrows_per_figure": 1,  # Number of rows of plots per figure
            "ncols_per_figure": 2,  # Number of columns of plots per figure
            "comment_section_height_ratio": DEFAULT_COMMENT_HEIGHT_RATIO,  # Height ratio for the comment section
        }

    def add_comment_section(self, fig, gridspec, test_id):
        """Add comments section to the figure below the plots"""
        comment_ax = fig.add_subplot(gridspec[-1, :])
        comment_ax.axis("off")  # Hide axes for the comments section

        comment_text = (
            self.test_id_to_comment.get(test_id, "No comment available, test has not been analyzed")
            if self.test_id_to_comment
            else "No comment available"
        )

        comment_ax.text(
            0.5,
            0.5,
            f"Comments: {comment_text}",
            ha="center",
            va="center",
            fontsize=10,
            style="italic",
        )

    def plot_dm_stats(self):
        # Set noc_width based on architecture
        noc_width = NOC_WIDTHS.get(self.arch, 64)  # Default to 64 if architecture not found
        multicast_schemes_test_ids = MULTICAST_SCHEMES_TEST_IDS

        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)

        # Get plot configuration
        config = self.get_plot_config()

        # Extract data for plotting
        riscv_1_series = self.aggregate_stats["riscv_1"]
        riscv_0_series = self.aggregate_stats["riscv_0"]

        # Group data by Test id
        test_ids = set()
        for kernel in self.aggregate_stats.keys():
            for stats in self.aggregate_stats[kernel].values():
                test_ids.add(stats["attributes"]["Test id"])

        test_ids = sorted(test_ids)  # Sort for consistent ordering

        # Extract test IDs
        # test_ids = sorted(
        #    {attributes["Test id"] for riscv in RISCV_PROCESSORS for attributes in self.dm_stats[riscv]["attributes"].values()}
        # )

        # Extract data for plotting
        series = {riscv: self.dm_stats[riscv]["analysis"]["series"] for riscv in RISCV_PROCESSORS}

        # Iterate over test IDs and create figures
        for test_id in test_ids:
            test_name = (
                self.test_id_to_name.get(test_id, f"Test ID {test_id}")
                if self.test_id_to_name
                else f"Test ID {test_id}"
            )

            # Prepare figure for the current test ID
            figure_height = (
                config["plot_height"] * config["nrows_per_figure"]
                + config["comment_section_height_ratio"] * config["plot_height"]
            )

            fig = plt.figure(figsize=(config["plot_width"] * config["ncols_per_figure"], figure_height))

            # Create a GridSpec layout
            gridspec = GridSpec(
                config["nrows_per_figure"] + 1,
                config["ncols_per_figure"],
                height_ratios=[config["plot_height"]] * config["nrows_per_figure"]
                + [config["comment_section_height_ratio"] * config["plot_height"]],
            )

            # Create subplots within the figure
            axes = [fig.add_subplot(gridspec[0, col]) for col in range(config["ncols_per_figure"])]

            # Extract data for riscv_1 and riscv_0
            data = {
                riscv: self.extract_data(series[riscv], self.dm_stats[riscv]["attributes"], test_id)
                for riscv in RISCV_PROCESSORS
            }

            # Generate plots based on test type
            if test_id in multicast_schemes_test_ids:
                self.plot_bandwidth_multicast(
                    axes[0],
                    data,
                    x_axis="grid_dimensions",
                    lines="multicast_scheme_number",
                    riscv="riscv_0",
                    noc_index=0,
                )
                self.plot_bandwidth_multicast(
                    axes[1],
                    data,
                    x_axis="grid_dimensions",
                    lines="multicast_scheme_number",
                    riscv="riscv_0",
                    noc_index=1,
                )
            else:  # Packet Sizes
                self.plot_durations(axes[0], data)
                self.plot_data_size_vs_bandwidth(axes[1], data, noc_width)

            # Add comments section to the figure below the plots
            self.add_comment_section(fig, gridspec, test_id)

            # Save the plot for this test id
            test_name = (
                self.test_id_to_name.get(test_id, f"Test ID {test_id}")
                if self.test_id_to_name
                else f"Test ID {test_id}"
            )
            output_file = os.path.join(self.output_dir, f"{test_name}.png")
            plt.savefig(output_file)
            plt.close(fig)
            logger.info(f"dm_stats plot for test id {test_id} saved at {output_file}")

    # Helper: Extract data for a specific test_id
    def extract_data(self, series, attributes, test_id):
        filtered = [
            entry for entry in series if attributes[entry["duration_type"][0]["run_host_id"]]["Test id"] == test_id
        ]

        data = {
            "durations": [],
            "data_sizes": [],
            "bandwidths": [],
            "transactions": [],
        }
        if test_id in MULTICAST_SCHEMES_TEST_IDS:
            data["noc_index"] = []
            data["multicast_scheme_number"] = []
            data["grid_dimensions"] = []

        for entry in filtered:
            runtime_host_id = entry["duration_type"][0]["run_host_id"]
            attr = attributes[runtime_host_id]

            duration = entry["duration_cycles"]
            transaction_size = attr["Transaction size in bytes"]
            num_transactions = attr["Number of transactions"]
            bandwidth = num_transactions * transaction_size / duration

            data["durations"].append(duration)
            data["data_sizes"].append(transaction_size)
            data["bandwidths"].append(bandwidth)
            data["transactions"].append(num_transactions)

            if test_id in MULTICAST_SCHEMES_TEST_IDS:
                noc_index = attr["NoC Index"]
                multicast_scheme_type = attr["Multicast Scheme Type"]
                grid_dimensions = f"{attr['Subordinate Grid Size X']} x {attr['Subordinate Grid Size Y']}"

                data["noc_index"].append(noc_index)
                data["multicast_scheme_number"].append(multicast_scheme_type)
                data["grid_dimensions"].append(grid_dimensions)

        return data

    ## Plotting functions

    # Helper: Plot Type 1 - Test Index vs Duration
    def plot_durations(self, ax, data):
        risc_to_kernel_map = RISC_TO_KERNEL_MAP

        unique_transactions = sorted(set(itertools.chain.from_iterable(data[riscv]["transactions"] for riscv in data)))
        for num_transactions in unique_transactions:
            for riscv in RISCV_PROCESSORS:
                # Group and plot RISCV data
                grouped = [
                    (size, duration)
                    for size, duration, transactions in zip(
                        data[riscv]["data_sizes"], data[riscv]["durations"], data[riscv]["transactions"]
                    )
                    if transactions == num_transactions
                ]
                if grouped:
                    sizes, durations = zip(*grouped)
                    ax.plot(
                        sizes,
                        durations,
                        label=f"{risc_to_kernel_map[riscv]} (Number of Transactions={num_transactions})",
                        marker="o",
                    )

        # Adjust the plot area to leave space for the legend
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])  # Shrink plot width to 80% of allocated space

        # Place the legend outside the plot but within the allocated subfigure space
        ax.legend(
            loc="center left",
            bbox_to_anchor=(1.0, 0.5),  # Position the legend outside the plot area
            borderaxespad=0,
            fontsize=8,
        )

        ax.set_xlabel("Transaction Size (bytes)")
        ax.set_ylabel("Duration (cycles)")
        ax.set_title("Transaction Size vs Duration")
        ax.set_xscale("log", base=2)
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x)}"))
        ax.set_yscale("log", base=10)
        ax.grid()

    # Helper: Plot Type 2 - Transaction Size vs Bandwidth
    def plot_data_size_vs_bandwidth(self, ax, data, noc_width):
        risc_to_kernel_map = RISC_TO_KERNEL_MAP

        unique_transactions = sorted(set(itertools.chain.from_iterable(data[riscv]["transactions"] for riscv in data)))
        # unique_transactions = sorted(set(data["riscv_1"]["transactions"] + data["riscv_0"]["transactions"]))
        for num_transactions in unique_transactions:
            grouped = {}
            for riscv in RISCV_PROCESSORS:
                grouped[riscv] = [
                    (size, bw)
                    for size, bw, transactions in zip(
                        data[riscv]["data_sizes"], data[riscv]["bandwidths"], data[riscv]["transactions"]
                    )
                    if transactions == num_transactions
                ]
                if grouped[riscv]:
                    # Sort by data sizes (x-axis) before plotting
                    grouped[riscv].sort(key=lambda x: x[0])
                    sizes, bws = zip(*grouped[riscv])
                    ax.plot(
                        sizes, bws, label=f"{risc_to_kernel_map[riscv]} (Transactions={num_transactions})", marker="o"
                    )

        # Add theoretical max bandwidth curve
        transaction_sizes = sorted(set(data["riscv_1"]["data_sizes"] + data["riscv_0"]["data_sizes"]))
        max_bandwidths = [noc_width * ((size / noc_width) / ((size / noc_width) + 1)) for size in transaction_sizes]
        ax.plot(transaction_sizes, max_bandwidths, label="Theoretical Max BW", linestyle="--", color="black")

        # Adjust the plot area to leave space for the legend
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])  # Shrink plot width to 80% of allocated space

        # Place the legend outside the plot but within the allocated subfigure space
        ax.legend(
            loc="center left",
            bbox_to_anchor=(1.0, 0.5),  # Position the legend outside the plot area
            borderaxespad=0,
            fontsize=8,
        )

        # Set labels and title
        ax.set_xlabel("Transaction Size (bytes)")
        ax.set_ylabel("Bandwidth (bytes/cycle)")
        ax.set_title("Transaction Size vs Bandwidth")
        ax.set_xscale("log", base=2)
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x)}"))
        ax.grid()

    # Helper: Plot Bandwidth for Multicast Schemes
    def plot_bandwidth_multicast(self, ax, data, x_axis, lines, riscv, noc_index):
        # Filter data where "noc_index" matches the input noc_index
        filtered_data = [
            (data[riscv][x_axis][i], data[riscv]["bandwidths"][i], data[riscv][lines][i])
            for i in range(len(data[riscv][x_axis]))
            if data[riscv].get("noc_index", [None])[i] == noc_index
        ]

        if not filtered_data:
            return  # No data to plot for the given noc_index

        # Sort the filtered data by the x_axis values
        filtered_data.sort(key=lambda x: x[0])

        # Extract sorted x_axis, bandwidths, and lines
        sorted_x_axis, sorted_bandwidths, sorted_lines = zip(*filtered_data)

        # Get unique line categories
        lines_list = sorted(set(sorted_lines))

        for line in lines_list:
            # Filter data for the current line category
            line_data = [(x, bw) for x, bw, l in zip(sorted_x_axis, sorted_bandwidths, sorted_lines) if l == line]

            if line_data:
                sizes, bws = zip(*line_data)
                ax.plot(
                    sizes,
                    bws,
                    label=f"{riscv.upper()}, NoC {noc_index}, {lines.replace('_', ' ').title()}: {line}",
                    marker="o",
                )

        # Adjust the plot area to leave space for the legend
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])  # Shrink plot width to 80% of allocated space

        # Place the legend outside the plot but within the allocated subfigure space
        ax.legend(
            loc="center left",
            bbox_to_anchor=(1.0, 0.5),  # Position the legend outside the plot area
            borderaxespad=0,
            fontsize=8,
        )

        ax.set_xlabel(f"{x_axis.replace('_', ' ').title()}")
        ax.set_ylabel("Bandwidth (bytes/cycle)")
        ax.set_title(f"{x_axis.replace('_', ' ').title()} vs Bandwidth")
        ax.grid()
