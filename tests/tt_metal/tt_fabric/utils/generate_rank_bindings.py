#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import os
import subprocess
import sys
from pathlib import Path

import yaml
from loguru import logger


def generate_tray_to_pcie_device_mapping(mapping_file):
    # Use absolute path to the test executable
    test_executable = Path("build/test/tt_metal/tt_fabric/test_physical_discovery")

    if not test_executable.exists():
        logger.error(f"Test executable not found at {test_executable}")
        logger.info("Please build with: ./build_metal.sh --build-tests")
        sys.exit(1)

    MAPPING_GENERATION_CMD = [str(test_executable), "--gtest_filter=*GenerateTrayToPCIeDeviceMapping*"]

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
        tray_ids = rank_to_tray_mapping[rank]
        for tray_id in tray_ids:
            if tray_id in tray_to_pcie_device_mapping["device_mapping"]:
                # Get PCIe devices for this tray and convert to comma-separated string
                pcie_devices = tray_to_pcie_device_mapping["device_mapping"][tray_id]
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
    num_devices = 0
    for tray in tray_to_pcie_device_mapping["device_mapping"]:
        num_devices += len(tray_to_pcie_device_mapping["device_mapping"][tray])
    if tray_to_pcie_device_mapping["arch"] != "WORMHOLE_B0" or num_devices != 32:
        logger.error(
            "Customized splitting of PCIe devices across processes is currently supported only for WORMHOLE Galaxies"
        )
        sys.exit(1)


def generate_supported_rank_bindings():
    # Process Rank ID To Tray ID Mapping when spawning 2 processes on a WH Galaxy
    WH_GLX_DUAL_RANK_TO_TRAY_MAPPING = {
        0: [1, 2],
        1: [3, 4],
    }
    # Process Rank ID To Tray ID Mapping when spawning 4 processes on a WH Galaxy
    WH_GLX_QUAD_RANK_TO_TRAY_MAPPING = {
        0: [1],
        1: [2],
        2: [3],
        3: [4],
    }
    # Process Rank ID To Tray ID Mapping for 2x4 cyclic mesh configuration
    WH_GLX_2X4_CYCLIC_RANK_TO_TRAY_MAPPING = {
        0: [1],
        1: [3],
        2: [4],
        3: [2],
    }

    # Rank bindings for Dual Mesh Setup (1 process per mesh)
    DUAL_MESH_RANK_BINDINGS = [
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
    ]

    # Rank bindings for Dual Big Mesh Setup (2 processes per mesh)
    DUAL_BIG_MESH_RANK_BINDINGS = [
        {
            "rank": 0,
            "mesh_id": 0,
            "mesh_host_rank": 0,
        },
        {
            "rank": 1,
            "mesh_id": 0,
            "mesh_host_rank": 1,
        },
        {
            "rank": 2,
            "mesh_id": 1,
            "mesh_host_rank": 0,
        },
        {
            "rank": 3,
            "mesh_id": 1,
            "mesh_host_rank": 1,
        },
    ]
    # Rank bindings for Quad Mesh Setup (1 process per mesh)
    QUAD_MESH_RANK_BINDINGS = [
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
    ]

    CYCLIC_MESH_RANK_BINDINGS = [
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
    ]

    mapping_file = "tray_to_pcie_device_mapping.yaml"
    generate_tray_to_pcie_device_mapping(mapping_file)
    with open(mapping_file, "r") as f:
        tray_to_pcie_device_mapping = yaml.safe_load(f)
    validate_device_mapping(tray_to_pcie_device_mapping)

    generate_rank_binding_yaml(
        tray_to_pcie_device_mapping,
        DUAL_MESH_RANK_BINDINGS,
        WH_GLX_DUAL_RANK_TO_TRAY_MAPPING,
        "tests/tt_metal/tt_fabric/custom_mesh_descriptors/wh_galaxy_split_4x4_multi_mesh.textproto",
        "4x4_multi_mesh_rank_binding.yaml",
    )
    generate_rank_binding_yaml(
        tray_to_pcie_device_mapping,
        DUAL_BIG_MESH_RANK_BINDINGS,
        WH_GLX_QUAD_RANK_TO_TRAY_MAPPING,
        "tests/tt_metal/tt_fabric/custom_mesh_descriptors/wh_galaxy_split_4x4_multi_big_mesh.textproto",
        "4x4_multi_big_mesh_rank_binding.yaml",
    )
    generate_rank_binding_yaml(
        tray_to_pcie_device_mapping,
        QUAD_MESH_RANK_BINDINGS,
        WH_GLX_QUAD_RANK_TO_TRAY_MAPPING,
        "tests/tt_metal/tt_fabric/custom_mesh_descriptors/wh_galaxy_split_4x2_multi_mesh.textproto",
        "4x2_multi_mesh_rank_binding.yaml",
    )
    generate_rank_binding_yaml(
        tray_to_pcie_device_mapping,
        CYCLIC_MESH_RANK_BINDINGS,
        WH_GLX_2X4_CYCLIC_RANK_TO_TRAY_MAPPING,
        "tests/tt_metal/tt_fabric/custom_mesh_descriptors/wh_galaxy_2x4_mesh_graph_descriptor.textproto",
        "2x4_multi_mesh_cyclic_rank_binding.yaml",
    )


if __name__ == "__main__":
    generate_supported_rank_bindings()
