# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from loguru import logger  # type: ignore
import itertools
import os
import csv

from tests.tt_metal.tt_metal.data_movement.python.constants import *


class StatsReporter:
    def __init__(
        self, dm_stats, aggregate_stats, test_id_to_name, test_type_attributes, output_dir, arch, metadata_loader=None
    ):
        self.dm_stats = dm_stats
        self.aggregate_stats = aggregate_stats
        self.test_id_to_name = test_id_to_name
        self.test_type_attributes = test_type_attributes
        self.metadata_loader = metadata_loader
        self.arch = arch
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

                # Get test metadata if metadata loader is available
                test_metadata = None
                if self.metadata_loader is not None:
                    try:
                        test_metadata = self.metadata_loader.get_test_metadata(test_id, self.arch)
                    except Exception as e:
                        # If metadata not found, log warning and continue without it
                        logger.warning(f"Could not load metadata for test ID {test_id}: {e}")

                header = [
                    "Transaction Size (bytes)",
                    "Number of Transactions",
                ]

                # Add metadata columns for NOC estimator consumption
                # All metadata fields are automatically included (except 'name' and 'comment')
                if test_metadata is not None:
                    # Standard metadata fields in specific order for csv_reader.cpp compatibility
                    standard_fields = ["architecture", "mechanism", "memory_type", "pattern"]
                    for field in standard_fields:
                        if field in test_metadata:
                            column_name = self.metadata_loader.metadata_field_to_column_name(field)
                            header.append(column_name)

                    # Add any additional optional fields in alphabetical order
                    optional_fields = sorted([k for k in test_metadata.keys() if k not in standard_fields])
                    for field in optional_fields:
                        column_name = self.metadata_loader.metadata_field_to_column_name(field)
                        header.append(column_name)

                header.extend(
                    [
                        "RISC-V Processor",
                        "Run Host ID",
                    ]
                )

                # Add test-specific headers
                for test_type, test_type_attributes in self.test_type_attributes.items():
                    if test_type.replace("_", " ").title() in test_name:
                        header.extend(test_type_attributes["attributes"].keys())
                        if test_type == "multicast_schemes":
                            header.append("Grid Dimensions")
                        break

                # Add performance metrics at the end
                header.extend(
                    [
                        "Latency (cycles)",
                        "Bandwidth (bytes/cycle)",
                    ]
                )

                writer.writerow(header)

                for riscv in self.aggregate_stats.keys():
                    for run_host_id, run_stats in self.aggregate_stats[riscv].items():
                        attributes = run_stats["attributes"]
                        if attributes.get("Test id") != test_id:
                            continue

                        # Base row data - configuration columns first
                        row = [
                            run_stats.get("transaction_size"),
                            run_stats.get("num_transactions"),
                        ]

                        # Add metadata columns
                        if test_metadata is not None:
                            # Standard metadata fields in specific order
                            standard_fields = ["architecture", "mechanism", "memory_type", "pattern"]
                            for field in standard_fields:
                                if field in test_metadata:
                                    row.append(test_metadata[field])

                            # Add any additional optional fields in alphabetical order
                            optional_fields = sorted([k for k in test_metadata.keys() if k not in standard_fields])
                            for field in optional_fields:
                                value = test_metadata[field]
                                # If we have profiled data and it's not set in the test_information
                                if field == "num_subordinates" and value == "":
                                    value = run_stats.get("num_subordinates", 0)
                                if field == "same_axis" and value == "":
                                    value = bool(run_stats.get("same_axis", 0))
                                row.append(value)
                        row.extend(
                            [
                                riscv,
                                run_host_id,
                            ]
                        )

                        # Add test-specific data
                        for test_type, test_type_attributes in self.test_type_attributes.items():
                            if test_type.replace("_", " ").title() in test_name:
                                row.extend([run_stats.get(val) for val in test_type_attributes["attributes"].values()])
                                if test_type == "multicast_schemes":
                                    row.append(run_stats.get("grid_dimensions"))
                                break

                        # Add performance metrics at the end
                        row.extend(
                            [
                                run_stats["duration_cycles"],
                                run_stats["bandwidth"],
                            ]
                        )

                        writer.writerow(row)

            logger.info(f"CSV report for test id {test_id} saved at {csv_file}")
