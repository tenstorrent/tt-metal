# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from loguru import logger  # type: ignore
import itertools
import os
import csv

from tests.tt_metal.tt_metal.data_movement.python.constants import *


class StatsReporter:
    def __init__(self, dm_stats, aggregate_stats, test_id_to_name, test_type_attributes, output_dir, arch):
        self.dm_stats = dm_stats
        self.aggregate_stats = aggregate_stats
        self.test_id_to_name = test_id_to_name
        self.test_type_attributes = test_type_attributes
        # Create architecture-specific subdirectory
        self.output_dir = os.path.join(output_dir, arch)

    def print_stats(self):
        # Print stats per runtime host id
        for riscv1_run, riscv0_run in itertools.zip_longest(
            self.dm_stats["riscv_1"]["analysis"]["series"],
            self.dm_stats["riscv_0"]["analysis"]["series"],
            fillvalue=None,
        ):
            logger.info(f"")
            logger.info(f"")

            riscv1_run_host_id = riscv1_run["duration_type"][0]["run_host_id"] if riscv1_run else None
            riscv0_run_host_id = riscv0_run["duration_type"][0]["run_host_id"] if riscv0_run else None

            logger.info(f"Run host id (riscv_1): {riscv1_run_host_id}")
            if riscv0_run_host_id is not None and riscv0_run_host_id != riscv1_run_host_id:
                logger.info(f"Run host id (riscv_0): {riscv0_run_host_id}")

            for riscv, run in [("RISCV 1", riscv1_run), ("RISCV 0", riscv0_run)]:
                if run:
                    logger.info(f"")
                    logger.info(f'{riscv} duration: {run["duration_cycles"]}')

                    logger.info("Attributes:")
                    riscv_key = riscv.lower().replace(" ", "_")
                    run_host_id = run["duration_type"][0]["run_host_id"]
                    for attr, val in self.dm_stats[riscv_key]["attributes"][run_host_id].items():
                        if attr == "Test id":
                            test_name = self.test_id_to_name.get(val, "Unknown Test")
                            logger.info(f"  {attr}: {val} ({test_name})")
                        else:
                            logger.info(f"  {attr}: {val}")

        logger.info(f"")
        logger.info(f"")

    def export_dm_stats_to_csv(self):
        os.makedirs(self.output_dir, exist_ok=True)

        # Group by test id
        test_ids = set()
        for riscv in self.aggregate_stats.keys():
            for test_run in self.aggregate_stats[riscv].values():
                test_ids.add(test_run["attributes"]["Test id"])
        test_ids = sorted(test_ids)

        for test_id in test_ids:
            test_name = (
                self.test_id_to_name.get(test_id, f"Test ID {test_id}")
                if self.test_id_to_name
                else f"Test ID {test_id}"
            )

            csv_file = os.path.join(self.output_dir, f"{test_name}.csv")
            with open(csv_file, mode="w", newline="") as csvfile:
                writer = csv.writer(csvfile)

                # Base header
                header = [
                    "RISC-V Processor",
                    "Kernel",
                    "Run Host ID",
                    "Transaction Size (bytes)",
                    "Number of Transactions",
                    "Latency (cycles)",
                    "Bandwidth (bytes/cycle)",
                ]
                # Add test-specific headers
                for test_type, test_type_attributes in self.test_type_attributes.items():
                    if test_type.replace("_", " ").title() in test_name:
                        header.extend(test_type_attributes["attributes"].keys())
                        if test_type == "multicast_schemes":
                            header.append("Grid Dimensions")
                        break
                writer.writerow(header)

                for riscv in self.aggregate_stats.keys():
                    for run_host_id, run_stats in self.aggregate_stats[riscv].items():
                        attributes = run_stats["attributes"]
                        if attributes.get("Test id") != test_id:
                            continue

                        # Base row data
                        row = [
                            riscv,
                            "Receiver" if riscv == "riscv_1" else "Sender",
                            run_host_id,
                            run_stats.get("transaction_size"),
                            run_stats.get("num_transactions"),
                            run_stats["duration_cycles"],
                            run_stats["bandwidth"],
                        ]

                        # Add test-specific data
                        for test_type, test_type_attributes in self.test_type_attributes.items():
                            if test_type.replace("_", " ").title() in test_name:
                                row.extend([run_stats.get(val) for val in test_type_attributes["attributes"].values()])
                                if test_type == "multicast_schemes":
                                    row.append(run_stats.get("grid_dimensions"))
                                break
                        writer.writerow(row)

            logger.info(f"CSV report for test id {test_id} saved at {csv_file}")
