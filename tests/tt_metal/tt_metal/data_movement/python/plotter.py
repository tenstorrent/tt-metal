# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import os
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import itertools
import matplotlib.ticker as mticker
from loguru import logger  # type: ignore

from tests.tt_metal.tt_metal.data_movement.python.constants import *


class Plotter:
    def __init__(self, dm_stats, aggregate_stats, output_dir, arch, test_id_to_name, test_id_to_comment):
        self.dm_stats = dm_stats
        self.aggregate_stats = aggregate_stats
        self.output_dir = output_dir
        self.noc_width = NOC_WIDTHS.get(arch, 64)  # Default to 64 if architecture not found
        self.arch = arch
        self.test_id_to_name = test_id_to_name
        self.test_id_to_comment = test_id_to_comment
        self.plot_config = self.get_plot_config()

    def get_plot_config(self):
        """Returns plot configuration parameters as a dictionary"""
        return {
            "plot_width": DEFAULT_PLOT_WIDTH,  # Width of an individual plot
            "plot_height": DEFAULT_PLOT_HEIGHT,  # Height of an individual plot
            "nrows_per_figure": 1,  # Number of rows of plots per figure
            "ncols_per_figure": 2,  # Number of columns of plots per figure
            "comment_section_height_ratio": DEFAULT_COMMENT_HEIGHT_RATIO,  # Height ratio for the comment section
            "wspace": 0.3,  # Horizontal space between plots
        }

    def plot_dm_stats(self):
        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)

        # Group data by Test id
        test_ids = set()
        for kernel in self.aggregate_stats.keys():
            for stats in self.aggregate_stats[kernel].values():
                test_ids.add(stats["attributes"]["Test id"])

        test_ids = sorted(test_ids)  # Sort for consistent ordering

        # Prepare data for plotting
        plot_data = {}
        for test_id in test_ids:
            plot_data[test_id] = {riscv: [] for riscv in RISCV_PROCESSORS}
            for riscv in RISCV_PROCESSORS:
                for run_stats in self.aggregate_stats[riscv].values():
                    if run_stats["attributes"]["Test id"] == test_id:
                        plot_data[test_id][riscv].append(run_stats)

        # Iterate over test IDs and create figures
        for test_id in test_ids:
            test_name = (
                self.test_id_to_name.get(test_id, f"Test ID {test_id}")
                if self.test_id_to_name
                else f"Test ID {test_id}"
            )

            # Prepare figure for the current test ID
            figure_height = (
                self.plot_config["plot_height"] * self.plot_config["nrows_per_figure"]
                + self.plot_config["comment_section_height_ratio"] * self.plot_config["plot_height"]
            )

            fig = plt.figure(
                figsize=(self.plot_config["plot_width"] * self.plot_config["ncols_per_figure"], figure_height)
            )

            # Create a GridSpec layout
            gridspec = GridSpec(
                self.plot_config["nrows_per_figure"] + 1,
                self.plot_config["ncols_per_figure"],
                height_ratios=[self.plot_config["plot_height"]] * self.plot_config["nrows_per_figure"]
                + [self.plot_config["comment_section_height_ratio"] * self.plot_config["plot_height"]],
                wspace=self.plot_config["wspace"],
            )

            # Create subplots within the figure
            axes = [fig.add_subplot(gridspec[0, col]) for col in range(self.plot_config["ncols_per_figure"])]

            # Generate plots based on test type
            if test_id in TEST_TYPE_ATTRIBUTES["multicast_schemes"]["test_ids"]:
                self.plot_bandwidth_multicast(
                    axes[0],
                    plot_data[test_id],
                    x_axis="grid_dimensions",
                    lines="multicast_scheme_number",
                    riscv="riscv_0",
                    noc_index=0,
                )
                self.plot_bandwidth_multicast(
                    axes[1],
                    plot_data[test_id],
                    x_axis="grid_dimensions",
                    lines="multicast_scheme_number",
                    riscv="riscv_0",
                    noc_index=1,
                )
            else:  # Packet Sizes
                self.plot_durations(axes[0], plot_data[test_id])
                self.plot_data_size_vs_bandwidth(axes[1], plot_data[test_id])

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

    ## PLOTTING FUNCTIONS ##

    def _plot_series(
        self,
        ax,
        data,
        x_key,
        y_key,
        series_keys,
        label_format,
        title,
        xlabel,
        ylabel,
        xscale="linear",
        yscale="linear",
        xbase=2,
        ybase=10,
        add_theoretical_max_bw=False,
    ):
        # Flatten data and add riscv to each run
        all_runs = []
        for riscv, runs in data.items():
            for run in runs:
                new_run = run.copy()
                new_run["riscv"] = riscv
                all_runs.append(new_run)

        if not all_runs:
            return

        # Get unique values for each series key
        unique_series_values = {
            key: sorted(list(set(run[key] for run in all_runs if key in run))) for key in series_keys
        }

        # Create all combinations of series values
        series_combinations = list(itertools.product(*[unique_series_values[key] for key in series_keys]))

        for combo in series_combinations:
            series_filter = {key: combo[i] for i, key in enumerate(series_keys)}

            current_series_runs = [run for run in all_runs if all(run.get(k) == v for k, v in series_filter.items())]

            if not current_series_runs:
                continue

            # Sort by x-axis value
            current_series_runs.sort(key=lambda r: r[x_key])

            x_vals = [run[x_key] for run in current_series_runs]
            y_vals = [run[y_key] for run in current_series_runs]

            label = label_format(combo, series_keys)

            ax.plot(x_vals, y_vals, label=label, marker="o")

        if add_theoretical_max_bw:
            all_x_vals = sorted(set(run[x_key] for run in all_runs))
            max_bandwidths = [
                self.noc_width * ((size / self.noc_width) / ((size / self.noc_width) + 1)) for size in all_x_vals
            ]
            ax.plot(all_x_vals, max_bandwidths, label="Theoretical Max BW", linestyle="--", color="black")

        # Adjust the plot area to leave space for the legend
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

        # Place the legend outside the plot
        ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), borderaxespad=0, fontsize=8)

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)

        if xscale == "log":
            ax.set_xscale("log", base=xbase)
            if xbase == 2:
                ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x)}"))

        if yscale == "log":
            ax.set_yscale("log", base=ybase)

        ax.grid()

    # Packet Sizes: Transaction Size vs Duration
    def plot_durations(self, ax, data):
        risc_to_kernel_map = RISC_TO_KERNEL_MAP
        self._plot_series(
            ax=ax,
            data=data,
            x_key="transaction_size",
            y_key="duration_cycles",
            series_keys=["riscv", "num_transactions"],
            label_format=lambda combo, keys: f"{risc_to_kernel_map[combo[0]]} (Number of Transactions={combo[1]})",
            title="Transaction Size vs Duration",
            xlabel="Transaction Size (bytes)",
            ylabel="Duration (cycles)",
            xscale="log",
            xbase=2,
            yscale="log",
            ybase=10,
        )

    # Packet Sizes: Transaction Size vs Bandwidth
    def plot_data_size_vs_bandwidth(self, ax, data):
        risc_to_kernel_map = RISC_TO_KERNEL_MAP
        self._plot_series(
            ax=ax,
            data=data,
            x_key="transaction_size",
            y_key="bandwidth",
            series_keys=["riscv", "num_transactions"],
            label_format=lambda combo, keys: f"{risc_to_kernel_map[combo[0]]} (Transactions={combo[1]})",
            title="Transaction Size vs Bandwidth",
            xlabel="Transaction Size (bytes)",
            ylabel="Bandwidth (bytes/cycle)",
            xscale="log",
            xbase=2,
            add_theoretical_max_bw=True,
        )

    # Multicast Schemes: Grid Dimensions vs Bandwidth
    def plot_bandwidth_multicast(self, ax, data, x_axis, lines, riscv, noc_index):
        filtered_data = {
            r: [run for run in runs if run.get("noc_index") == noc_index] for r, runs in data.items() if r == riscv
        }

        if not filtered_data.get(riscv):
            return  # No data to plot

        self._plot_series(
            ax=ax,
            data=filtered_data,
            x_key=x_axis,
            y_key="bandwidth",
            series_keys=[lines],
            label_format=lambda combo, keys: f"{riscv.upper()}, NoC {noc_index}, {keys[0].replace('_', ' ').title()}: {combo[0]}",
            title=f"{x_axis.replace('_', ' ').title()} vs Bandwidth",
            xlabel=f"{x_axis.replace('_', ' ').title()}",
            ylabel="Bandwidth (bytes/cycle)",
        )

    # Add comments section to the figure below the plots
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
