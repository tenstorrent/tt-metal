#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import argparse
import copy
import os
import subprocess
import sys
from pathlib import Path

import yaml
from loguru import logger

# Machine-readable stdout prefix for run_fabric_tests.sh (--print-devices).
FABRIC_VISIBLE_DEVICES_PREFIX = "FABRIC_VISIBLE_DEVICES:"


def generate_tray_to_pcie_device_mapping(mapping_file="tray_to_pcie_device_mapping.yaml", work_dir=None):
    cwd = Path(work_dir) if work_dir else Path.cwd()
    test_executable = cwd / "build/test/tt_metal/tt_fabric/test_physical_discovery"

    if not test_executable.exists():
        logger.error(f"Test executable not found at {test_executable}")
        logger.info("Please build with: ./build_metal.sh --build-tests")
        sys.exit(1)

    mapping_path = Path(mapping_file)
    if not mapping_path.is_absolute():
        mapping_path = cwd / mapping_path

    MAPPING_GENERATION_CMD = [str(test_executable), "--gtest_filter=*GenerateTrayToPCIeDeviceMapping*"]

    logger.info(f"Running in {cwd}: {' '.join(MAPPING_GENERATION_CMD)}")

    original_cwd = os.getcwd()
    try:
        os.chdir(cwd)
        result = subprocess.run(
            MAPPING_GENERATION_CMD,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
        )
        if result.returncode != 0:
            if result.stderr:
                logger.error(result.stderr.rstrip())
            logger.error(f"{MAPPING_GENERATION_CMD} Failed to generate tray to pcie device mapping")
            sys.exit(result.returncode)
    except KeyboardInterrupt:
        logger.error(f"{MAPPING_GENERATION_CMD} Interrupted")
        sys.exit(1)
    finally:
        os.chdir(original_cwd)

    if not mapping_path.exists():
        logger.error(f"{mapping_path} not found")
        sys.exit(1)


def generate_rank_binding_yaml(
    tray_to_pcie_device_mapping, rank_bindings, rank_to_tray_mapping, mesh_graph_desc_path, output_file
):
    # Create a deep copy to avoid mutating the original bindings
    rank_bindings = copy.deepcopy(rank_bindings)
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


def visible_devices_for_rank(tray_to_pcie_device_mapping, rank_to_tray_mapping, rank):
    """Return comma-separated TT_VISIBLE_DEVICES for one rank from tray discovery."""
    tray_ids = rank_to_tray_mapping[rank]
    device_ids = []
    device_mapping = tray_to_pcie_device_mapping["device_mapping"]
    for tray_id in tray_ids:
        if tray_id not in device_mapping:
            logger.error(f"Tray {tray_id} not found in mapping for rank {rank}")
            sys.exit(1)
        device_ids.extend(sorted(device_mapping[tray_id]))
    return ",".join(map(str, device_ids))


# Rev C Blackhole Galaxy single-host layouts used by run_fabric_tests.sh (4x8z / 2x4x4z).
# Quad configs (16x4x4z) reuse these per-host layouts via SSH discovery on
# each galaxy host; mesh_id is assigned globally in rankfile order (host0 meshes first).
# Tray ids come from test_physical_discovery (tray_to_pcie_device_mapping.yaml).
# 4x8z must follow BH_GLX_QUAD rank->tray order. Physical 2x2 tray grid per host:
#   m0(tr1)  m1(tr3)
#   m3(tr2)  m2(tr4)
# Intra-host Z: 0-1, 1-2, 2-3, 0-3 (bh_galaxy_split_4x2_multi_mesh).
# Quad inter-host: same local mesh index across chassis stack (see quad_bh_galaxy_16x4x2_z MGD; 4x4x8z TBD).
FABRIC_TEST_TRAY_MAPPINGS = {
    # 2 Z-connected 4x4 meshes: trays 1+2 (top) and 3+4 (bottom) on Rev C.
    "2x4x4z": {
        0: [1, 2],
        1: [3, 4],
    },
    # 4 Z-connected 4x2 meshes: one tray per mesh (same as BH_GLX_QUAD_RANK_TO_TRAY_MAPPING).
    "4x8z": {
        0: [1],
        1: [3],
        2: [4],
        3: [2],
    },
}


def resolve_fabric_test_visible_devices(
    fabric_config,
    mapping_file="tray_to_pcie_device_mapping.yaml",
    run_discovery=True,
    work_dir=None,
):
    """Discover trays and return TT_VISIBLE_DEVICES strings (one per MPI rank)."""
    if fabric_config not in FABRIC_TEST_TRAY_MAPPINGS:
        logger.error(f"Unknown fabric config {fabric_config!r}. Supported: {sorted(FABRIC_TEST_TRAY_MAPPINGS)}")
        sys.exit(1)

    rank_to_tray = FABRIC_TEST_TRAY_MAPPINGS[fabric_config]
    cwd = Path(work_dir) if work_dir else Path.cwd()
    mapping_path = cwd / mapping_file

    if run_discovery:
        generate_tray_to_pcie_device_mapping(mapping_file, work_dir=cwd)
    elif not mapping_path.exists():
        logger.error(f"{mapping_path} not found (use discovery or pass --no-discovery after generating it)")
        sys.exit(1)

    with open(mapping_path, "r") as f:
        tray_to_pcie_device_mapping = yaml.safe_load(f)
    validate_device_mapping(tray_to_pcie_device_mapping)

    return [visible_devices_for_rank(tray_to_pcie_device_mapping, rank_to_tray, rank) for rank in sorted(rank_to_tray)]


def parse_args():
    parser = argparse.ArgumentParser(description="Generate rank bindings from tray discovery")
    parser.add_argument(
        "--fabric-config",
        choices=sorted(FABRIC_TEST_TRAY_MAPPINGS.keys()),
        help="Single-host fabric test layout (used by run_fabric_tests.sh 4x8z/2x4x4z and per-host on 16x4x4z)",
    )
    parser.add_argument(
        "--print-devices",
        action="store_true",
        help="With --fabric-config: print one TT_VISIBLE_DEVICES line per rank (stdout)",
    )
    parser.add_argument(
        "--no-discovery",
        action="store_true",
        help="With --fabric-config: use existing tray_to_pcie_device_mapping.yaml in cwd",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=None,
        help="Directory for discovery output and test binary (default: cwd)",
    )
    return parser.parse_args()


def validate_device_mapping(tray_to_pcie_device_mapping):
    if "arch" not in tray_to_pcie_device_mapping:
        logger.error("Tray to PCIe device mapping does not contain arch")
        sys.exit(1)
    num_devices = 0
    for tray in tray_to_pcie_device_mapping["device_mapping"]:
        num_devices += len(tray_to_pcie_device_mapping["device_mapping"][tray])
    if tray_to_pcie_device_mapping["arch"] not in ["WORMHOLE_B0", "BLACKHOLE"] or num_devices != 32:
        arch = tray_to_pcie_device_mapping["arch"]
        logger.error(
            f"Customized splitting of PCIe devices across processes is currently supported only for Wormhole and Blackhole Galaxies. Found arch: {arch}, num_devices: {num_devices}"
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
    # Process Rank ID To Tray ID Mapping when spawning 2 processes on a BH Galaxy
    # Trays 1+2 form the top 4x4 half, trays 3+4 form the bottom 4x4 half
    BH_GLX_DUAL_RANK_TO_TRAY_MAPPING = {
        0: [1, 3],
        1: [2, 4],
    }
    # Process Rank ID To Tray ID Mapping when spawning 4 processes on a BH Galaxy
    BH_GLX_QUAD_RANK_TO_TRAY_MAPPING = {
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

    # Rank bindings for Dual Mesh Setup (1 process per mesh) for BLACKHOLE
    DUAL_MESH_BLACKHOLE_RANK_BINDINGS = [
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

    # Rank bindings for Quad Mesh Setup (1 process per mesh) for BLACKHOLE
    QUAD_MESH_BLACKHOLE_RANK_BINDINGS = [
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

    mapping_file = "tray_to_pcie_device_mapping.yaml"
    generate_tray_to_pcie_device_mapping(mapping_file)
    with open(mapping_file, "r") as f:
        tray_to_pcie_device_mapping = yaml.safe_load(f)
    validate_device_mapping(tray_to_pcie_device_mapping)

    arch = tray_to_pcie_device_mapping["arch"]
    if arch == "WORMHOLE_B0":
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
        generate_rank_binding_yaml(
            tray_to_pcie_device_mapping,
            TRI_MESH_4X4_2X4_RANK_BINDINGS,
            WH_GLX_4X4_2X4_3_MESH_RANK_TO_TRAY_MAPPING,
            "tests/tt_metal/tt_fabric/custom_mesh_descriptors/wh_galaxy_split_4x4_2x4_3_mesh.textproto",
            "4x4_2x4_3_mesh_rank_binding.yaml",
        )
        generate_rank_binding_yaml(
            tray_to_pcie_device_mapping,
            TRI_MESH_2X8_2X4_RANK_BINDINGS,
            WH_GLX_2X8_2X4_3_MESH_RANK_TO_TRAY_MAPPING,
            "tests/tt_metal/tt_fabric/custom_mesh_descriptors/wh_galaxy_split_2x8_2x4_3_mesh.textproto",
            "2x8_2x4_3_mesh_rank_binding.yaml",
        )
    elif arch == "BLACKHOLE":
        generate_rank_binding_yaml(
            tray_to_pcie_device_mapping,
            DUAL_MESH_BLACKHOLE_RANK_BINDINGS,
            BH_GLX_DUAL_RANK_TO_TRAY_MAPPING,
            "tests/tt_metal/tt_fabric/custom_mesh_descriptors/bh_galaxy_4x4_mesh_graph_descriptor.textproto",
            "bh_4x4_multi_mesh_rank_binding.yaml",
        )
        generate_rank_binding_yaml(
            tray_to_pcie_device_mapping,
            DUAL_MESH_BLACKHOLE_RANK_BINDINGS,
            BH_GLX_DUAL_RANK_TO_TRAY_MAPPING,
            "tests/tt_metal/tt_fabric/custom_mesh_descriptors/bh_galaxy_4x4_z_mesh_graph_descriptor.textproto",
            "bh_4x4_z_multi_mesh_rank_binding.yaml",
        )
        generate_rank_binding_yaml(
            tray_to_pcie_device_mapping,
            QUAD_MESH_BLACKHOLE_RANK_BINDINGS,
            BH_GLX_QUAD_RANK_TO_TRAY_MAPPING,
            "tests/tt_metal/tt_fabric/custom_mesh_descriptors/bh_galaxy_4x2_mesh_graph_descriptor.textproto",
            "bh_4x2_multi_mesh_rank_binding.yaml",
        )
        generate_rank_binding_yaml(
            tray_to_pcie_device_mapping,
            QUAD_MESH_BLACKHOLE_RANK_BINDINGS,
            BH_GLX_QUAD_RANK_TO_TRAY_MAPPING,
            "tests/tt_metal/tt_fabric/custom_mesh_descriptors/bh_galaxy_split_4x2_multi_mesh.textproto",
            "bh_galaxy_split_4x2_multi_mesh_rank_binding.yaml",
        )


if __name__ == "__main__":
    args = parse_args()
    if args.fabric_config:
        devices = resolve_fabric_test_visible_devices(
            args.fabric_config,
            run_discovery=not args.no_discovery,
            work_dir=args.work_dir,
        )
        if args.print_devices:
            for line in devices:
                print(f"{FABRIC_VISIBLE_DEVICES_PREFIX}{line}")
        else:
            for rank, visible in enumerate(devices):
                print(f"rank {rank}: TT_VISIBLE_DEVICES={visible}")
    else:
        generate_supported_rank_bindings()
