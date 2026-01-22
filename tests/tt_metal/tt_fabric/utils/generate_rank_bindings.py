#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import os
import subprocess
import sys
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

    if not os.path.exists(mapping_file):
        logger.error(f"{mapping_file} not found")
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
        0: {"bh-glx-b08u02": 1},
        1: {"bh-glx-b08u02": 3},
        2: {"bh-glx-b08u02": 4},
        3: {"bh-glx-b08u02": 2},
        4: {"bh-glx-b08u08": 1},
        5: {"bh-glx-b08u08": 3},
        6: {"bh-glx-b08u08": 4},
        7: {"bh-glx-b08u08": 2},
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
        "tests/tt_metal/tt_fabric/custom_mesh_descriptors/wh_galaxy_split_4x4_multi_mesh.textproto",
        "bh_blitz_pipeline_rank_bindings.yaml",
    )


if __name__ == "__main__":
    host_vector = ["bh-glx-b08u02", "bh-glx-b08u08", "bh-glx-b09u02", "bh-glx-b09u08"]
    generate_supported_rank_bindings(host_vector)
