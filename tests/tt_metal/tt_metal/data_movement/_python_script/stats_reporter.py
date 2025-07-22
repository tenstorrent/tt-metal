# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from loguru import logger  # type: ignore
import itertools
import os
import csv


class StatsReporter:
    def __init__(self, dm_stats, aggregate_stats, test_id_to_name, output_dir):
        self.dm_stats = dm_stats
        self.aggregate_stats = aggregate_stats
        self.test_id_to_name = test_id_to_name
        self.output_dir = output_dir

    def print_stats(self):
        # Print stats per runtime host id
        for riscv1_run, riscv0_run in itertools.zip_longest(
            self.dm_stats["riscv_1"]["analysis"]["series"],
            self.dm_stats["riscv_0"]["analysis"]["series"],
            fillvalue=None,
        ):
            logger.info(f"")
            logger.info(f"")

            run_host_id = (riscv1_run if riscv1_run else riscv0_run)["duration_type"][0]["run_host_id"]
            logger.info(f"Run host id: {run_host_id}")

            for riscv, run in [("RISCV 1", riscv1_run), ("RISCV 0", riscv0_run)]:
                if run:
                    logger.info(f"")
                    logger.info(f'{riscv} duration: {run["duration_cycles"]}')

                    logger.info("Attributes:")
                    for attr, val in self.dm_stats[riscv.lower().replace(" ", "_")]["attributes"][run_host_id].items():
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
                writer.writerow(
                    [
                        "Kernel",  # Or "RISC-V Processor"?
                        "Run Host ID",
                        "Transaction Size (bytes)",
                        "Number of Transactions",
                        "Latency (cycles)",
                        "Bandwidth (bytes/cycle)",
                    ]
                )

                for riscv in self.aggregate_stats.keys():
                    for run_host_id, run_stats in self.aggregate_stats[riscv].items():
                        # run_host_id = entry["duration_type"][0]["run_host_id"]
                        attributes = run_stats["attributes"]
                        if attributes.get("Test id") != test_id:
                            continue
                        transaction_size = attributes.get("Transaction size in bytes", 0)
                        num_transactions = attributes.get("Number of transactions", 0)
                        duration_cycles = run_stats["duration_cycles"]
                        bandwidth = run_stats["bandwidth"]

                        writer.writerow(
                            [
                                "Receiver" if riscv == "riscv_1" else "Sender",
                                run_host_id,
                                transaction_size,
                                num_transactions,
                                duration_cycles,
                                bandwidth,
                            ]
                        )

            logger.info(f"CSV report for test id {test_id} saved at {csv_file}")
