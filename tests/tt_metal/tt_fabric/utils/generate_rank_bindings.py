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


def generate_slice_to_pcie_device_mapping(mapping_file, host_vector):
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
        "--gtest_filter=*Generate2x4SliceToPCIeDeviceMapping*",
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
    slice_to_pcie_device_mapping, rank_bindings, rank_to_tray_mapping, mesh_graph_desc_path, output_file
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


def generate_rank_binding_yaml(
    slice_to_pcie_device_mapping, rank_bindings, rank_to_slice_mapping, mesh_graph_desc_path, output_file
):
    # Populate TT_VISIBLE_DEVICES for each rank based on tray mapping
    for binding in rank_bindings:
        rank = binding["rank"]
        hostname_and_slice_ids = rank_to_slice_mapping[rank]

        for hostname, slice_id in hostname_and_slice_ids.items():
            print(f"Hostname: {hostname}, Slice ID for rank {rank}: {slice_id}")
            if hostname in slice_to_pcie_device_mapping["device_mapping"]:
                if slice_id in slice_to_pcie_device_mapping["device_mapping"][hostname]:
                    # Get PCIe devices for this tray and convert to comma-separated string
                    pcie_devices = slice_to_pcie_device_mapping["device_mapping"][hostname][slice_id]
                    # Sort the devices for consistent ordering
                    pcie_devices_sorted = sorted(pcie_devices)

                    if "env_overrides" not in binding:
                        binding["env_overrides"] = {}
                    if "TT_VISIBLE_DEVICES" not in binding["env_overrides"]:
                        binding["env_overrides"]["TT_VISIBLE_DEVICES"] = ""
                    # Add comma separator if there's already content
                    if binding["env_overrides"]["TT_VISIBLE_DEVICES"]:
                        binding["env_overrides"]["TT_VISIBLE_DEVICES"] += ","

                    binding["env_overrides"]["TT_VISIBLE_DEVICES"] += ",".join(map(str, pcie_devices_sorted))
                else:
                    logger.warning(f"Slice {slice_id} not found in mapping for rank {rank}")
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
    num_devices = 0
    for tray in tray_to_pcie_device_mapping["device_mapping"]:
        num_devices += len(tray_to_pcie_device_mapping["device_mapping"][tray])
    if num_devices != 32:
        logger.error(
            "Customized splitting of PCIe devices across processes is currently supported only for WORMHOLE Galaxies"
        )
        sys.exit(1)


def generate_supported_rank_bindings(host_vector):
    # Process Rank ID To Tray ID Mapping when spawning 2 processes on a WH Galaxy
    WH_GLX_DUAL_RANK_TO_TRAY_MAPPING = {
        0: [1, 3],
        1: [2, 4],
    }
    # Process Rank ID To Tray ID Mapping when spawning 4 processes on a WH Galaxy
    WH_GLX_QUAD_RANK_TO_TRAY_MAPPING = {
        0: [1],
        1: [3],
        2: [4],
        3: [2],
    }
    # Process Rank ID To Tray ID Mapping for 2x4 cyclic mesh configuration
    WH_GLX_2X4_CYCLIC_RANK_TO_TRAY_MAPPING = {
        0: [1],
        1: [3],
        2: [4],
        3: [2],
    }
    # Process Rank ID To Tray ID Mapping for 4x4 + 2x4 + 2x4 (3 mesh) configuration
    WH_GLX_4X4_2X4_3_MESH_RANK_TO_TRAY_MAPPING = {
        0: [1, 2],  # 4x4 mesh needs 2 trays (16 devices)
        1: [3],  # 2x4 mesh needs 1 tray (8 devices)
        2: [4],  # 2x4 mesh needs 1 tray (8 devices)
    }
    # Process Rank ID To Tray ID Mapping for 2x4 + 2x4 + 2x8 (3 mesh) configuration
    WH_GLX_2X8_2X4_3_MESH_RANK_TO_TRAY_MAPPING = {
        0: [1],  # 2x4 mesh needs 1 tray (8 devices)
        1: [3],  # 2x4 mesh needs 1 tray (8 devices)
        2: [2, 4],  # 2x8 mesh needs 2 trays (16 devices)
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

    # Rank bindings for Tri Mesh Setup: 4x4 + 2x4 + 2x4 (1 process per mesh, 3 meshes)
    TRI_MESH_4X4_2X4_RANK_BINDINGS = [
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
    ]

    # Rank bindings for Tri Mesh Setup: 2x4 + 2x4 + 2x8 (1 process per mesh, 3 meshes)
    TRI_MESH_2X8_2X4_RANK_BINDINGS = [
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
    ]

    BH_2x4_RANK_TO_SLICE_MAPPING = {
        0: {"bh-glx-d04u02": 0},
        1: {"bh-glx-d04u02": 1},
        2: {"bh-glx-d04u02": 2},
        3: {"bh-glx-d04u02": 3},
        4: {"bh-glx-d04u08": 0},
        5: {"bh-glx-d04u08": 1},
        6: {"bh-glx-d04u08": 2},
        7: {"bh-glx-d04u08": 3},
        8: {"bh-glx-d03u08": 0},
        9: {"bh-glx-d03u08": 1},
        10: {"bh-glx-d03u08": 2},
        11: {"bh-glx-d03u08": 3},
        12: {"bh-glx-d03u02": 0},
        13: {"bh-glx-d03u02": 1},
        14: {"bh-glx-d03u02": 2},
        15: {"bh-glx-d03u02": 3},
    }

    BH_2x4_SLICE_RANK_BINDINGS = [
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
    ]

    mapping_file = "slice_to_pcie_device_mapping.yaml"
    generate_slice_to_pcie_device_mapping(mapping_file, host_vector)
    with open(mapping_file, "r") as f:
        slice_to_pcie_device_mapping = yaml.safe_load(f)
    # validate_device_mapping(slice_to_pcie_device_mapping)

    generate_rank_binding_yaml(
        slice_to_pcie_device_mapping,
        BH_2x4_SLICE_RANK_BINDINGS,
        BH_2x4_RANK_TO_SLICE_MAPPING,
        "tt_metal/fabric/mesh_graph_descriptors/bh_glx_split_4x2.textproto",
        "bh_blitz_pipeline_rank_bindings.yaml",
    )


if __name__ == "__main__":
    host_vector = [
        "bh-glx-d04u02",
        "bh-glx-d04u08",
        "bh-glx-d03u08",
        "bh-glx-d03u02",
        # "bh-glx-d04u08",
        # "bh-glx-c02u02",
        # "bh-glx-c01u08",
        # "bh-glx-c01u02",
        # "bh-glx-c05u08",
        # "bh-glx-c05u02",
        # "bh-glx-c06u08",
        # "bh-glx-c06u02",
        # "bh-glx-c03u08",
        # "bh-glx-c04u08",
        # "bh-glx-c04u02",
        # "bh-glx-c03u02",
    ]
    generate_supported_rank_bindings(host_vector)
