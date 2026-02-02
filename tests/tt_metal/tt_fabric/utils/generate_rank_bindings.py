#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import os
import subprocess
import sys
import time
from pathlib import Path

import yaml
from loguru import logger


def generate_tray_to_pcie_device_mapping(mapping_file, host_vector):
    # Use absolute path to the test executable
    test_executable = Path("build/test/tt_metal/tt_fabric/test_physical_discovery")

    if not test_executable.exists():
        logger.error(f"Test executable not found at {test_executable}")
        logger.info("Please build with: ./build_metal.sh --build-tests")
        sys.exit(1)
    host_vector_str = ",".join(host_vector)
    MAPPING_GENERATION_CMD = [
        "mpirun",
        "--host",
        host_vector_str,
        "--mca",
        "btl",
        "self,tcp",
        "--mca",
        "btl_tcp_if_include",
        "ens5f0np0",
        "--bind-to",
        "none",
        "--tag-output",
        str(test_executable),
        "--gtest_filter=*GenerateTrayToPCIeDeviceMapping*",
    ]

    logger.info(f"Running: {' '.join(MAPPING_GENERATION_CMD)}")

    try:
        result = subprocess.run(MAPPING_GENERATION_CMD)
        if result.returncode != 0:
            logger.error(f"{MAPPING_GENERATION_CMD} Failed to generate tray to pcie device mapping")
            sys.exit(result.returncode)
    except KeyboardInterrupt:
        logger.error(f"{MAPPING_GENERATION_CMD} Interrupted")
        sys.exit(1)

    logger.info("Waiting 30 seconds for mapping files to be ready...")
    time.sleep(30)

    # Collect all host-specific mapping files
    consolidated_mapping = {"device_mapping": {}, "arch": None}

    for hostname in host_vector:
        host_mapping_file = f"tray_to_pcie_device_mapping_{hostname}.yaml"
        if not os.path.exists(host_mapping_file):
            logger.warning(f"Mapping file {host_mapping_file} not found for host {hostname}")
            continue

        logger.info(f"Reading mapping file from host {hostname}: {host_mapping_file}")
        with open(host_mapping_file, "r") as f:
            host_mapping = yaml.safe_load(f)

        # Extract arch from first file (should be same for all)
        if consolidated_mapping["arch"] is None and "arch" in host_mapping:
            consolidated_mapping["arch"] = host_mapping["arch"]

        # Consolidate device mappings by hostname
        if "device_mapping" in host_mapping:
            consolidated_mapping["device_mapping"][hostname] = host_mapping["device_mapping"]

    # Write consolidated mapping to output file
    with open(mapping_file, "w") as f:
        yaml.dump(consolidated_mapping, f, default_flow_style=False, sort_keys=False)

    logger.info(f"Consolidated mapping written to {mapping_file}")

    if not os.path.exists(mapping_file):
        logger.error(f"{mapping_file} not found after consolidation")
        sys.exit(1)


def generate_rank_binding_yaml(
    tray_to_pcie_device_mapping, rank_bindings, rank_to_tray_mapping, mesh_graph_desc_path, output_file
):
    # Populate TT_VISIBLE_DEVICES for each rank based on tray mapping
    for binding in rank_bindings:
        rank = binding["rank"]
        hostnames_and_tray_ids = rank_to_tray_mapping[rank]
        for hostname, tray_id in hostnames_and_tray_ids.items():
            if hostname in tray_to_pcie_device_mapping["device_mapping"]:
                if tray_id in tray_to_pcie_device_mapping["device_mapping"][hostname]:
                    # Get PCIe devices for this tray and convert to comma-separated string
                    pcie_devices = tray_to_pcie_device_mapping["device_mapping"][hostname][tray_id]
                    # Sort the devices for consistent ordering
                    pcie_devices_sorted = sorted(pcie_devices)

                    # Create env_overrides if it doesn't exist
                    if "env_overrides" not in binding:
                        binding["env_overrides"] = {}
                    if "TT_VISIBLE_DEVICES" not in binding["env_overrides"]:
                        binding["env_overrides"]["TT_VISIBLE_DEVICES"] = ""
                    # Add comma separator if there's already content
                    if binding["env_overrides"]["TT_VISIBLE_DEVICES"]:
                        binding["env_overrides"]["TT_VISIBLE_DEVICES"] += ","

                    binding["env_overrides"]["TT_VISIBLE_DEVICES"] += ",".join(map(str, pcie_devices_sorted))
                else:
                    logger.warning(f"Tray {tray_id} not found in mapping for rank {rank}")
            else:
                logger.warning(f"Hostname {hostname} not found in mapping for rank {rank}")

    # Create the full rank configuration
    rank_config = {
        "rank_bindings": rank_bindings,
        "mesh_graph_desc_path": mesh_graph_desc_path,
    }

    # Save to YAML file
    with open(output_file, "w") as f:
        yaml.dump(rank_config, f, default_flow_style=False, sort_keys=False)

    logger.info(f"Generated rank configuration file: {output_file}")
    return rank_config


def validate_device_mapping(tray_to_pcie_device_mapping):
    if "arch" not in tray_to_pcie_device_mapping:
        logger.error("Tray to PCIe device mapping does not contain arch")
        sys.exit(1)
    for hostname in tray_to_pcie_device_mapping["device_mapping"]:
        num_devices = 0
        for tray in tray_to_pcie_device_mapping["device_mapping"][hostname]:
            num_devices += len(tray_to_pcie_device_mapping["device_mapping"][hostname][tray])
        if num_devices != 32:
            logger.error(
                "Customized splitting of PCIe devices across processes is currently supported only for WORMHOLE Galaxies"
            )
            sys.exit(1)


def generate_supported_rank_bindings(host_vector):
    # Process Rank ID To Tray ID Mapping when spawning 2 processes on a WH Galaxy
    BH_POD_RANK_TO_TRAY_MAPPING = {
        0: {"bh-glx-c02u08": 3},
        1: {"bh-glx-c02u02": 1},
        2: {"bh-glx-c02u02": 3},
        3: {"bh-glx-c02u02": 4},
        4: {"bh-glx-c02u02": 2},
        5: {"bh-glx-c02u08": 4},
        6: {"bh-glx-c01u08": 3},
        7: {"bh-glx-c01u02": 1},
        8: {"bh-glx-c01u02": 3},
        9: {"bh-glx-c01u02": 4},
        10: {"bh-glx-c01u02": 2},
        11: {"bh-glx-c01u08": 4},
        12: {"bh-glx-c01u08": 2},
        13: {"bh-glx-c01u08": 1},
        14: {"bh-glx-c02u08": 2},
        15: {"bh-glx-c02u08": 1},
        16: {"bh-glx-c05u08": 2},
        17: {"bh-glx-c05u08": 4},
        18: {"bh-glx-c05u02": 2},
        19: {"bh-glx-c05u02": 4},
        20: {"bh-glx-c05u02": 3},
        21: {"bh-glx-c05u02": 1},
        22: {"bh-glx-c05u08": 3},
        23: {"bh-glx-c05u08": 1},
        24: {"bh-glx-c06u08": 2},
        25: {"bh-glx-c06u08": 4},
        26: {"bh-glx-c06u02": 2},
        27: {"bh-glx-c06u02": 4},
        28: {"bh-glx-c06u02": 3},
        29: {"bh-glx-c06u02": 1},
        30: {"bh-glx-c06u08": 3},
        31: {"bh-glx-c06u08": 1},
        32: {"bh-glx-c03u08": 2},
        33: {"bh-glx-c03u08": 1},
        34: {"bh-glx-c04u08": 2},
        35: {"bh-glx-c04u08": 1},
        36: {"bh-glx-c04u08": 3},
        37: {"bh-glx-c04u02": 1},
        38: {"bh-glx-c04u02": 3},
        39: {"bh-glx-c04u02": 4},
        40: {"bh-glx-c04u02": 2},
        41: {"bh-glx-c04u08": 4},
        42: {"bh-glx-c03u08": 3},
        43: {"bh-glx-c03u02": 1},
        44: {"bh-glx-c03u02": 3},
        45: {"bh-glx-c03u02": 4},
        46: {"bh-glx-c03u02": 2},
        47: {"bh-glx-c03u08": 4},
    }

    # Rank bindings for Dual Mesh Setup (1 process per mesh)
    BH_DUAL_POD_PIPELINE_RANK_BINDINGS = [
        {
            "rank": 0,
            "mesh_id": 0,
            "mesh_host_rank": 0,
        },
        {
            "rank": 1,
            "mesh_id": 1,
            "mesh_host_rank": 0,
        },
        {
            "rank": 2,
            "mesh_id": 2,
            "mesh_host_rank": 0,
        },
        {
            "rank": 3,
            "mesh_id": 3,
            "mesh_host_rank": 0,
        },
        {
            "rank": 4,
            "mesh_id": 4,
            "mesh_host_rank": 0,
        },
        {
            "rank": 5,
            "mesh_id": 5,
            "mesh_host_rank": 0,
        },
        {
            "rank": 6,
            "mesh_id": 6,
            "mesh_host_rank": 0,
        },
        {
            "rank": 7,
            "mesh_id": 7,
            "mesh_host_rank": 0,
        },
        {
            "rank": 8,
            "mesh_id": 8,
            "mesh_host_rank": 0,
        },
        {
            "rank": 9,
            "mesh_id": 9,
            "mesh_host_rank": 0,
        },
        {
            "rank": 10,
            "mesh_id": 10,
            "mesh_host_rank": 0,
        },
        {
            "rank": 11,
            "mesh_id": 11,
            "mesh_host_rank": 0,
        },
        {
            "rank": 12,
            "mesh_id": 12,
            "mesh_host_rank": 0,
        },
        {
            "rank": 13,
            "mesh_id": 13,
            "mesh_host_rank": 0,
        },
        {
            "rank": 14,
            "mesh_id": 14,
            "mesh_host_rank": 0,
        },
        {
            "rank": 15,
            "mesh_id": 15,
            "mesh_host_rank": 0,
        },
        {
            "rank": 16,
            "mesh_id": 16,
            "mesh_host_rank": 0,
        },
        {
            "rank": 17,
            "mesh_id": 17,
            "mesh_host_rank": 0,
        },
        {
            "rank": 18,
            "mesh_id": 18,
            "mesh_host_rank": 0,
        },
        {
            "rank": 19,
            "mesh_id": 19,
            "mesh_host_rank": 0,
        },
        {
            "rank": 20,
            "mesh_id": 20,
            "mesh_host_rank": 0,
        },
        {
            "rank": 21,
            "mesh_id": 21,
            "mesh_host_rank": 0,
        },
        {
            "rank": 22,
            "mesh_id": 22,
            "mesh_host_rank": 0,
        },
        {
            "rank": 23,
            "mesh_id": 23,
            "mesh_host_rank": 0,
        },
        {
            "rank": 24,
            "mesh_id": 24,
            "mesh_host_rank": 0,
        },
        {
            "rank": 25,
            "mesh_id": 25,
            "mesh_host_rank": 0,
        },
        {
            "rank": 26,
            "mesh_id": 26,
            "mesh_host_rank": 0,
        },
        {
            "rank": 27,
            "mesh_id": 27,
            "mesh_host_rank": 0,
        },
        {
            "rank": 28,
            "mesh_id": 28,
            "mesh_host_rank": 0,
        },
        {
            "rank": 29,
            "mesh_id": 29,
            "mesh_host_rank": 0,
        },
        {
            "rank": 30,
            "mesh_id": 30,
            "mesh_host_rank": 0,
        },
        {
            "rank": 31,
            "mesh_id": 31,
            "mesh_host_rank": 0,
        },
        {
            "rank": 32,
            "mesh_id": 32,
            "mesh_host_rank": 0,
        },
        {
            "rank": 33,
            "mesh_id": 33,
            "mesh_host_rank": 0,
        },
        {
            "rank": 34,
            "mesh_id": 34,
            "mesh_host_rank": 0,
        },
        {
            "rank": 35,
            "mesh_id": 35,
            "mesh_host_rank": 0,
        },
        {
            "rank": 36,
            "mesh_id": 36,
            "mesh_host_rank": 0,
        },
        {
            "rank": 37,
            "mesh_id": 37,
            "mesh_host_rank": 0,
        },
        {
            "rank": 38,
            "mesh_id": 38,
            "mesh_host_rank": 0,
        },
        {
            "rank": 39,
            "mesh_id": 39,
            "mesh_host_rank": 0,
        },
        {
            "rank": 40,
            "mesh_id": 40,
            "mesh_host_rank": 0,
        },
        {
            "rank": 41,
            "mesh_id": 41,
            "mesh_host_rank": 0,
        },
        {
            "rank": 42,
            "mesh_id": 42,
            "mesh_host_rank": 0,
        },
        {
            "rank": 43,
            "mesh_id": 43,
            "mesh_host_rank": 0,
        },
        {
            "rank": 44,
            "mesh_id": 44,
            "mesh_host_rank": 0,
        },
        {
            "rank": 45,
            "mesh_id": 45,
            "mesh_host_rank": 0,
        },
        {
            "rank": 46,
            "mesh_id": 46,
            "mesh_host_rank": 0,
        },
        {
            "rank": 47,
            "mesh_id": 47,
            "mesh_host_rank": 0,
        },
    ]

    mapping_file = "tray_to_pcie_device_mapping.yaml"
    generate_tray_to_pcie_device_mapping(mapping_file, host_vector)
    with open(mapping_file, "r") as f:
        tray_to_pcie_device_mapping = yaml.safe_load(f)
    validate_device_mapping(tray_to_pcie_device_mapping)

    generate_rank_binding_yaml(
        tray_to_pcie_device_mapping,
        BH_DUAL_POD_PIPELINE_RANK_BINDINGS,
        BH_POD_RANK_TO_TRAY_MAPPING,
        "tt_metal/fabric/mesh_graph_descriptors/bh_glx_split_4x2.textproto",
        "bh_blitz_pipeline_rank_bindings.yaml",
    )


if __name__ == "__main__":
    host_vector = [
        "bh-glx-c02u08",
        "bh-glx-c02u02",
        "bh-glx-c01u08",
        "bh-glx-c01u02",
        "bh-glx-c05u08",
        "bh-glx-c05u02",
        "bh-glx-c06u08",
        "bh-glx-c06u02",
        "bh-glx-c03u08",
        "bh-glx-c04u08",
        "bh-glx-c04u02",
        "bh-glx-c03u02",
    ]
    generate_supported_rank_bindings(host_vector)
