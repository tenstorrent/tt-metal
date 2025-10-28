# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from loguru import logger  # type: ignore
from collections import defaultdict
import numpy as np

from tracy.process_device_log import import_log_run_stats
import tracy.device_post_proc_config as device_post_proc_config

from tests.tt_metal.tt_metal.data_movement.python.constants import *


class StatsCollector:
    def __init__(self, file_path, test_id_to_name, test_type_attributes, verbose=False):
        self.file_path = file_path
        self.verbose = verbose
        self.test_id_to_name = test_id_to_name
        self.test_type_attributes = test_type_attributes
        # Map each RISC-V processor to its corresponding analysis/event key
        self.riscv_to_analysis_event = {
            "riscv_1": {"analysis": "riscv_1_analysis", "events": "riscv_1_events"},
            "riscv_0": {"analysis": "riscv_0_analysis", "events": "riscv_0_events"},
        }

    def gather_analysis_stats(self, stats=None):
        # Gather stats from csv and set up analysis
        if stats is None:
            stats = self.gather_stats_from_csv()
        cores = [
            key
            for key in stats["devices"][0]["cores"].keys()
            if (
                "BRISC" in stats["devices"][0]["cores"][key]["riscs"]
                or "NCRISC" in stats["devices"][0]["cores"][key]["riscs"]
            )
            and key != "DEVICE"
        ]

        dm_stats = {}
        for risc in RISCV_PROCESSORS:
            dm_stats[risc] = {
                "analysis": {"stats": dict(), "series": []},
                "attributes": dict(),
            }

        # Gather analysis stats
        # Statistics are recorded per core, but timeseries data is aggregated for all cores
        for core in cores:
            core_analysis = stats["devices"][0]["cores"][core]["riscs"]["TENSIX"]["analysis"]

            for risc in RISCV_PROCESSORS:
                analysis_key = self.riscv_to_analysis_event[risc]["analysis"]

                if analysis_key in core_analysis:
                    for run in core_analysis[analysis_key]["series"]:
                        run["core"] = core
                    dm_stats[risc]["analysis"]["stats"][core] = core_analysis[analysis_key]["stats"]
                    dm_stats[risc]["analysis"]["series"].extend(core_analysis[analysis_key]["series"])

        # Gather test attributes

        for risc in dm_stats.keys():
            attributes = dm_stats[risc]["attributes"]
            events_key = self.riscv_to_analysis_event[risc]["events"]

            for event in stats["devices"][0]["cores"]["DEVICE"]["riscs"]["TENSIX"]["events"][events_key]:
                run_host_id = event[0]["run_host_id"]

                if run_host_id in attributes.keys():
                    attributes[run_host_id][event[0]["zone_name"]] = event[2]
                else:
                    attributes[run_host_id] = {event[0]["zone_name"]: event[2]}

        aggregate_stats = self.aggregate_performance(dm_stats)

        return dm_stats, aggregate_stats

    def gather_stats_from_csv(self):
        # Configure post proc script
        setup = device_post_proc_config.default_setup()
        setup.deviceInputLog = self.file_path

        # Build timer analysis configuration using the mapping dictionary
        timer_analysis = {}
        for risc in RISCV_PROCESSORS:
            analysis_key = self.riscv_to_analysis_event[risc]["analysis"]
            events_key = self.riscv_to_analysis_event[risc]["events"]

            # Add analysis configuration
            timer_analysis[analysis_key] = {
                "across": "core",
                "type": "adjacent",
                "start": {"risc": "ANY", "zone_phase": "ZONE_START", "zone_name": risc.upper().replace("_", "")},
                "end": {"risc": "ANY", "zone_phase": "ZONE_END", "zone_name": risc.upper().replace("_", "")},
            }

            # Add events configuration
            marker_risc = "NCRISC" if risc == "riscv_1" else "BRISC"
            timer_analysis[events_key] = {
                "across": "device",
                "type": "event",
                "marker": {"risc": marker_risc},
            }

        setup.timerAnalysis = timer_analysis

        # Gather stats from csv
        if not self.verbose:
            logger.disable("tracy.process_device_log")

        return import_log_run_stats(setup)

    def aggregate_performance(self, dm_stats, method="median"):
        """
        Aggregates duration and bandwidth per run_host_id for each kernel,
        and includes the attributes for each run_host_id.

        This function now also dynamically adds test-specific attributes
        based on the test_type_attributes mapping in test_config.yaml.

        Args:
            dm_stats: nested dict as produced by gather_analysis_stats
            method: 'median' or 'average'

        Returns:
            Dict: {kernel: {run_host_id: {
                "duration_cycles": aggregated_value,
                "bandwidth": aggregated_value,
                "attributes": attributes_dict,
                "all_durations": [...],
                "all_bandwidths": [...],
            }}}
        """
        result = {}

        for riscv in RISCV_PROCESSORS:
            grouped_durations = defaultdict(list)
            grouped_bandwidths = defaultdict(list)
            grouped_cores = defaultdict(list)

            for entry in dm_stats[riscv]["analysis"]["series"]:
                run_host_id = entry["duration_type"][0]["run_host_id"]
                attributes = dm_stats[riscv]["attributes"][run_host_id]
                num_transactions = attributes.get("Number of transactions", 1)
                transaction_size = attributes.get("Transaction size in bytes", 0)

                duration = entry["duration_cycles"]
                bandwidth = num_transactions * transaction_size / duration if duration else 0

                grouped_durations[run_host_id].append(duration)
                grouped_bandwidths[run_host_id].append(bandwidth)
                grouped_cores[run_host_id].append(entry["core"])

            agg = {}
            for run_host_id, durations in grouped_durations.items():
                bandwidths = grouped_bandwidths[run_host_id]
                attributes = dm_stats[riscv]["attributes"][run_host_id]
                test_id = attributes.get("Test id")
                cores = grouped_cores[run_host_id]

                if method == "median":
                    agg_duration = float(np.median(durations))
                    agg_bandwidth = float(np.median(bandwidths))
                elif method == "average":
                    agg_duration = float(np.mean(durations))
                    agg_bandwidth = float(np.mean(bandwidths))
                else:
                    raise ValueError(f"Unknown method: {method}")

                agg_data = {
                    "duration_cycles": agg_duration,
                    "bandwidth": agg_bandwidth,
                    "attributes": attributes,
                    "all_durations": durations,
                    "all_bandwidths": bandwidths,
                    "all_cores": cores,
                    "transaction_size": attributes.get("Transaction size in bytes"),
                    "num_transactions": attributes.get("Number of transactions"),
                }

                # Dynamically add attributes based on test type
                test_name = self.test_id_to_name.get(test_id, "")
                for test_type, config in self.test_type_attributes.items():
                    if test_type.replace("_", " ").title() in test_name:
                        for key, value in config["attributes"].items():
                            agg_data[value] = attributes.get(key)
                        # For multicast, create a grid dimension string
                        if test_type == "multicast_schemes":
                            grid_x = attributes.get("Subordinate Grid Size X", "N/A")
                            grid_y = attributes.get("Subordinate Grid Size Y", "N/A")
                            agg_data["grid_dimensions"] = f"{grid_x} x {grid_y}"

                agg[run_host_id] = agg_data

            result[riscv] = agg
        return result

    def gather_bw_per_core(self, num_transactions=None, transaction_size=None, risc=None, test_id=None):
        _, aggregate_stats = self.gather_analysis_stats()
        result = defaultdict(int)

        if risc is None:
            riscs_with_stats = [r for r, v in aggregate_stats.items() if len(v) > 0]
            if len(riscs_with_stats) == 1:
                risc = riscs_with_stats[0]
            elif len(riscs_with_stats) == 0:
                raise ValueError("Data not found for any RISC-V processors.")
            else:
                raise ValueError("Data found for multiple RISC-V processors. Please specify one.")

        # Find unique parameter sets and their associated run_host_ids
        parameter_sets = defaultdict(list)
        valid_run_host_ids = []
        for run_host_id, data in aggregate_stats[risc].items():
            attributes = data["attributes"]
            param_set = (
                attributes.get("Test id"),
                attributes.get("Number of transactions"),
                attributes.get("Transaction size in bytes"),
            )
            parameter_sets[param_set].append(run_host_id)

        # If filters are provided, find matching parameter set
        if test_id is not None or num_transactions is not None or transaction_size is not None:
            matching_sets = [
                param_set
                for param_set in parameter_sets.keys()
                if (test_id is None or param_set[0] == test_id)
                and (num_transactions is None or param_set[1] == num_transactions)
                and (transaction_size is None or param_set[2] == transaction_size)
            ]

            if len(matching_sets) == 0:
                raise ValueError("No parameter sets found matching the specified criteria.")
            elif len(matching_sets) > 1:
                raise ValueError(f"Multiple parameter sets found: {matching_sets}. Please be more specific.")
            else:
                selected_param_set = matching_sets[0]
                valid_run_host_ids.extend(parameter_sets[selected_param_set])
        else:
            # If no filters provided, check if there's only one unique parameter set
            if len(parameter_sets) == 1:
                selected_param_set = list(parameter_sets.keys())[0]
                valid_run_host_ids.extend(parameter_sets[selected_param_set])
            else:
                raise ValueError(
                    f"Multiple parameter sets found: {list(parameter_sets.keys())}. Please specify test_id, num_transactions, or transaction_size."
                )

        # Use the first run_host_id from the selected parameter set
        if len(valid_run_host_ids) > 1:
            logger.warning(
                f"Multiple run_host_ids found for the selected parameter set: {valid_run_host_ids}. Using the first one."
            )
        run_host_id = valid_run_host_ids[0]

        test_case = aggregate_stats[risc][run_host_id]
        for core, bandwidth in zip(test_case["all_cores"], test_case["all_bandwidths"]):
            result[core] = bandwidth
        attrs = test_case["attributes"]
        attrs["Risc"] = 0 if risc == "riscv_0" else 1
        return result, attrs


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python stats_collector.py <path_to_log_csv>")
        sys.exit(1)

    file_path = sys.argv[1]
    collector = StatsCollector(file_path, test_id_to_name={}, test_type_attributes={}, verbose=True)
    dm_stats, aggregate_stats = collector.gather_analysis_stats()
    for riscv, stats in aggregate_stats.items():
        print(f"RISC-V Processor: {riscv}")
        for run_host_id, data in stats.items():
            print(f"  Run Host ID: {run_host_id}")
            print(f"    Duration (cycles): {data['duration_cycles']}")
            print(f"    Bandwidth (bytes/cycle): {data['bandwidth']}")
            print(f"    Attributes: {data['attributes']}")
            print(f"    All Durations: {data['all_durations']}")
            print(f"    All Bandwidths: {data['all_bandwidths']}")
