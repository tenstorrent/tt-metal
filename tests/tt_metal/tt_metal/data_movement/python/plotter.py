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
        # Create architecture-specific subdirectory
        self.output_dir = os.path.join(output_dir, arch)
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
            "nrows_per_figure": 1,  # Number of rows of plots per figure (will be updated dynamically)
            "ncols_per_figure": 2,  # Number of columns of plots per figure (will be updated dynamically)
            "comment_section_height_ratio": DEFAULT_COMMENT_HEIGHT_RATIO,  # Height ratio for the comment section
            "wspace": 0.3,  # Horizontal space between plots
            "hspace": 0.3,  # Vertical space between plots
        }

    def get_dynamic_plot_config(self, test_name):
        """Returns dynamic plot configuration based on test type"""
        config = self.plot_config.copy()

        if "Virtual Channels" in test_name:
            # For virtual channels tests, use two plots (1x2 grid) for NOC 0 and NOC 1
            config["nrows_per_figure"] = 1
            config["ncols_per_figure"] = 2
        if "Direct Write" in test_name:
            # For direct write tests, use one plot (1x1 grid)
            config["nrows_per_figure"] = 1
            config["ncols_per_figure"] = 1
        return config

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

            # Get dynamic plot configuration for this test type
            plot_config = self.get_dynamic_plot_config(test_name)

            # Prepare figure for the current test ID
            figure_height = (
                plot_config["plot_height"] * plot_config["nrows_per_figure"]
                + plot_config["comment_section_height_ratio"] * plot_config["plot_height"]
            )

            fig = plt.figure(figsize=(plot_config["plot_width"] * plot_config["ncols_per_figure"], figure_height))

            # Create a GridSpec layout
            gridspec = GridSpec(
                plot_config["nrows_per_figure"] + 1,
                plot_config["ncols_per_figure"],
                height_ratios=[plot_config["plot_height"]] * plot_config["nrows_per_figure"]
                + [plot_config["comment_section_height_ratio"] * plot_config["plot_height"]],
                wspace=plot_config["wspace"],
                hspace=plot_config.get("hspace", 0.3),
            )

            # Create subplots within the figure
            axes = []
            for row in range(plot_config["nrows_per_figure"]):
                for col in range(plot_config["ncols_per_figure"]):
                    axes.append(fig.add_subplot(gridspec[row, col]))

            # Generate plots based on test type
            if "Multicast Schemes" in test_name:
                self.plot_bandwidth_multicast(
                    axes[0],
                    plot_data[test_id],
                    riscv="riscv_0",
                    noc_index=0,
                )
                self.plot_bandwidth_multicast(
                    axes[1],
                    plot_data[test_id],
                    riscv="riscv_0",
                    noc_index=1,
                )
            elif "Virtual Channels" in test_name:
                self.plot_bandwidth_virtual_channels(axes[0], plot_data[test_id], noc_index=0)
                self.plot_bandwidth_virtual_channels(axes[1], plot_data[test_id], noc_index=1)
            elif "Transaction ID" in test_name:
                # For Transaction ID tests, plot both data size vs bandwidth and transaction ID count vs bandwidth
                self.plot_data_size_vs_bandwidth(axes[0], plot_data[test_id])
                self.plot_transaction_id_count_vs_bandwidth(axes[1], plot_data[test_id])
            elif "Direct Write Performance Comparison" in test_name:
                self.plot_bandwidth_direct_write_performance(axes[0], plot_data[test_id])
            elif "Direct Write Address Pattern" in test_name:
                self.plot_bandwidth_direct_write_address_pattern(axes[0], plot_data[test_id])
            else:  # Packet Sizes
                self.plot_durations(axes[0], plot_data[test_id])
                self.plot_data_size_vs_bandwidth(axes[1], plot_data[test_id])

            # Add figure title
            fig.suptitle(f"{test_name} ({self.arch.upper()})", fontsize=16, fontweight="bold", y=0.98)

            # Add comments section to the figure below the plots
            self.add_comment_section(fig, gridspec, test_id)

            # Save the plot for this test id
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
        x_key = "transaction_size"
        y_key = "duration_cycles"
        series_keys = ["riscv", "num_transactions"]

        title = "Transaction Size vs Duration"
        xlabel = "Transaction Size (bytes)"
        ylabel = "Duration (cycles)"

        risc_to_kernel_map = RISC_TO_KERNEL_MAP
        self._plot_series(
            ax=ax,
            data=data,
            x_key=x_key,
            y_key=y_key,
            series_keys=series_keys,
            label_format=lambda combo, keys: f"{risc_to_kernel_map[combo[0]]} (Number of Transactions={combo[1]})",
            title=title,
            xlabel=xlabel,
            ylabel=ylabel,
            xscale="log",
            xbase=2,
            yscale="log",
            ybase=10,
        )

    # Packet Sizes: Transaction Size vs Bandwidth
    def plot_data_size_vs_bandwidth(self, ax, data):
        x_key = "transaction_size"
        y_key = "bandwidth"
        series_keys = ["riscv", "num_transactions"]

        title = "Transaction Size vs Bandwidth"
        xlabel = "Transaction Size (bytes)"
        ylabel = "Bandwidth (bytes/cycle)"

        risc_to_kernel_map = RISC_TO_KERNEL_MAP
        self._plot_series(
            ax=ax,
            data=data,
            x_key=x_key,
            y_key=y_key,
            series_keys=series_keys,
            label_format=lambda combo, keys: f"{risc_to_kernel_map[combo[0]]} (Transactions={combo[1]})",
            title=title,
            xlabel=xlabel,
            ylabel=ylabel,
            xscale="log",
            xbase=2,
            add_theoretical_max_bw=True,
        )

    # Transaction ID: Transaction ID Count vs Bandwidth (grouped by transaction size)
    def plot_transaction_id_count_vs_bandwidth(self, ax, data):
        # Flatten data and add riscv to each run, also compute transaction_id_count
        all_runs = []
        for riscv, runs in data.items():
            for run in runs:
                new_run = run.copy()
                new_run["riscv"] = riscv
                # Calculate transaction_id_count as num_of_transactions / 2
                new_run["transaction_id_count"] = run["num_transactions"] / 2
                all_runs.append(new_run)

        if not all_runs:
            return

        # Group by transaction_size for plotting
        transaction_sizes = sorted(list(set(run["transaction_size"] for run in all_runs)))

        risc_to_kernel_map = RISC_TO_KERNEL_MAP

        for transaction_size in transaction_sizes:
            # Filter runs for this transaction size
            size_runs = [run for run in all_runs if run["transaction_size"] == transaction_size]

            # Group by riscv
            for riscv in set(run["riscv"] for run in size_runs):
                riscv_runs = [run for run in size_runs if run["riscv"] == riscv]

                # Sort by transaction_id_count for proper line plotting
                riscv_runs.sort(key=lambda r: r["transaction_id_count"])

                x_vals = [run["transaction_id_count"] for run in riscv_runs]
                y_vals = [run["bandwidth"] for run in riscv_runs]

                label = f"{risc_to_kernel_map[riscv]} (Transaction Size={transaction_size}B)"

                ax.plot(x_vals, y_vals, label=label, marker="o")

        # Adjust the plot area to leave space for the legend
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

        # Place the legend outside the plot
        ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), borderaxespad=0, fontsize=8)

        ax.set_xlabel("Transaction ID Count")
        ax.set_ylabel("Bandwidth (bytes/cycle)")
        ax.set_title("Transaction ID Count vs Bandwidth")
        ax.grid()
        
    # Direct Write: Performance test
    def plot_bandwidth_direct_write_performance(self, ax, data):
        x_key = "num_transactions"
        y_key = "bandwidth"
        series_keys = ["stateful", "posted"]

        title = "Number of Transactions vs Bandwidth"
        xlabel = "Number of Transactions"
        ylabel = "Bandwidth (bytes/cycle)"

        self._plot_series(
            ax=ax,
            data=data,
            x_key=x_key,
            y_key=y_key,
            series_keys=series_keys,
            label_format=lambda combo, keys: f"{'Stateful' if combo[0] else 'Non-stateful'}, {'Posted' if combo[1] else 'Non-posted'}",
            title=title,
            xlabel=xlabel,
            ylabel=ylabel,
            xscale="log",
            xbase=2,
            add_theoretical_max_bw=False,
        )

    # Direct Write: Address pattern
    def plot_bandwidth_direct_write_address_pattern(self, ax, data):
        x_key = "num_transactions"
        y_key = "bandwidth"
        series_keys = ["stateful", "same_dest", "same_value"]

        title = "Number of Transactions vs Bandwidth"
        xlabel = "Number of Transactions"
        ylabel = "Bandwidth (bytes/cycle)"

        self._plot_series(
            ax=ax,
            data=data,
            x_key=x_key,
            y_key=y_key,
            series_keys=series_keys,
            label_format=lambda combo, keys: f"{'Stateful' if combo[0] else 'Non-stateful'}, {'Same destinations' if combo[1] else 'Different destinations'}, {'Same value' if combo[2] else 'Different values'}",
            title=title,
            xlabel=xlabel,
            ylabel=ylabel,
            xscale="log",
            xbase=2,
            add_theoretical_max_bw=False,
        )

    # Multicast Schemes: Grid Dimensions vs Bandwidth
    def plot_bandwidth_multicast(self, ax, data, riscv, noc_index):
        x_key = "grid_dimensions"
        y_key = "bandwidth"
        series_keys = ["multicast_scheme_number"]

        title = "Grid Dimensions vs Bandwidth"
        xlabel = "Grid Dimensions"
        ylabel = "Bandwidth (bytes/cycle)"

        filtered_data = {
            r: [run for run in runs if run.get("noc_index") == noc_index] for r, runs in data.items() if r == riscv
        }

        if not filtered_data.get(riscv):
            return  # No data to plot

        self._plot_series(
            ax=ax,
            data=filtered_data,
            x_key=x_key,
            y_key=y_key,
            series_keys=series_keys,
            label_format=lambda combo, keys: f"{riscv.upper()}, NoC {noc_index}, {keys[0].replace('_', ' ').title()}: {combo[0]}",
            title=title,
            xlabel=xlabel,
            ylabel=ylabel,
        )

    def plot_bandwidth_virtual_channels(self, ax, data, noc_index):
        x_key = "transaction_size"
        y_key = "bandwidth"
        series_keys = ["num_virtual_channels"]

        title = f"Transaction Size vs Bandwidth (NOC {noc_index})"
        xlabel = "Transaction Size (bytes)"
        ylabel = "Bandwidth (bytes/cycle)"

        # Filter data for the specific NOC index
        filtered_data = {}
        for riscv, runs in data.items():
            filtered_runs = [run for run in runs if run.get("noc_index") == noc_index]
            if filtered_runs:
                filtered_data[riscv] = filtered_runs

        if not filtered_data:
            # If no data for this NOC, show empty plot with appropriate message
            ax.text(
                0.5,
                0.5,
                f"No data available for NOC {noc_index}",
                horizontalalignment="center",
                verticalalignment="center",
                transform=ax.transAxes,
                fontsize=12,
            )
            ax.set_title(title)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            return

        self._plot_series(
            ax=ax,
            data=filtered_data,
            x_key=x_key,
            y_key=y_key,
            series_keys=series_keys,
            label_format=lambda combo, keys: f"{combo[0]} Virtual Channels",
            title=title,
            xlabel=xlabel,
            ylabel=ylabel,
            xscale="log",
            xbase=2,
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
