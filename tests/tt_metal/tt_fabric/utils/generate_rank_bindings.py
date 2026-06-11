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
# Machine-readable stdout prefix for run_fabric_tests.sh (--print-rank-table).
# Format: FABRIC_RANK:<mesh_id>;<host_rank>;<host>;<devices_csv>
FABRIC_RANK_PREFIX = "FABRIC_RANK:"


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


# Rev C Blackhole Galaxy single-host tray layout used by run_fabric_tests.sh (4x8z).
# Tray ids come from test_physical_discovery (tray_to_pcie_device_mapping.yaml).
# 4x8z must follow BH_GLX_QUAD rank->tray order. Physical 2x2 tray grid per host:
#   m0(tr1)  m1(tr3)
#   m3(tr2)  m2(tr4)
# Intra-host Z: 0-1, 1-2, 2-3, 0-3 (bh_galaxy_split_4x2_multi_mesh).
# NOTE: 2x4x4z / 8x4x4z now use the 2x4 slice discovery instead (see
# FABRIC_TEST_SLICE_MAPPINGS / resolve_fabric_test_visible_devices_slices below).
FABRIC_TEST_TRAY_MAPPINGS = {
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


# --- Slice-based device resolution (2x4x4z / 8x4x4z) -----------------------
# Uses the 2x4 "slice" discovery (Generate2x4SliceToPCIeDeviceMapping) instead
# of per-tray discovery. A slice is a host-local 2x4 (8-chip) block spanning two
# trays; the gtest already accounts for the Rev C tray swap, so slice ids 0..3
# are logical. Each 4x4 mesh is composed from two slices.
#
# Slice -> 4x4 composition follows the canonical slice layout used by the blitz
# decode pipeline (blitz_pipeline_config_quad_galaxy_4x4.yaml): the two 4x4
# meshes are *interleaved* across all four trays, not split into tray-pairs.
#   inner pair  -> slices {1, 2}  (single-host 4x4 in the blitz config)
#   outer pair  -> slices {0, 3}  (the wrap/split 4x4 in the blitz config)
# Both lists are ordered top-tray-pair slice first, bottom-tray-pair slice
# second (1 before 2, 0 before 3) so the two meshes share a consistent
# device ordering. Discovery is a single-host (--np 1) mpirun; the quad layout
# (8x4x4z) replicates that one host's slice map across the identical ring hosts
# (see resolve_quad_split_rank_table) rather than discovering each host.
SLICE_MAPPING_GTEST_FILTER = "*Generate2x4SliceToPCIeDeviceMapping*"
SLICE_MAPPING_FILE = "slice_to_pcie_device_mapping.yaml"
DEVICES_PER_SLICE = 8

# Single-host (2x4x4z) local mesh -> slice ids. With no adjacent host both 4x4
# meshes are self-contained: mesh 0 = inner pair {1,2}, mesh 1 = outer pair {0,3}.
# The quad layout (8x4x4z) instead splits the outer slices across adjacent ring
# hosts -- see build_quad_split_rank_table / resolve_quad_split_rank_table.
FABRIC_TEST_SLICE_MAPPINGS = {
    "2x4x4z": {0: [1, 2], 1: [0, 3]},
}
# CLI choices for --slice-config: single-host plus the quad split layout.

SLICE_CONFIG_CHOICES = ["2x4x4z", "8x4x4z"]


def generate_slice_to_pcie_device_mapping(hosts, mpi_if=None, work_dir=None, mapping_file=SLICE_MAPPING_FILE):
    """Run the 2x4 slice discovery; writes slice_to_pcie_device_mapping.yaml.

    With one or more hosts, runs one rank per host (cross-host mpirun). With no
    hosts (None/empty), runs a single LOCAL rank (mpirun --np 1, no --host) so
    discovery works inside a single container with no cross-host ssh.
    """
    cwd = Path(work_dir) if work_dir else Path.cwd()
    test_executable = cwd / "build/test/tt_metal/tt_fabric/test_physical_discovery"
    if not test_executable.exists():
        logger.error(f"Test executable not found at {test_executable}")
        logger.info("Please build with: ./build_metal.sh --build-tests")
        sys.exit(1)

    if hosts:
        cmd = [
            "mpirun",
            "--np",
            str(len(hosts)),
            "--host",
            ",".join(hosts),
            "--mca",
            "btl",
            "self,tcp",
        ]
        if mpi_if:
            cmd.extend(["--mca", "btl_tcp_if_include", mpi_if])
    else:
        # Purely local single-rank discovery: no --host => mpirun never ssh's out.
        cmd = ["mpirun", "--np", "1", "--mca", "btl", "self,tcp"]
        if mpi_if:
            cmd.extend(["--mca", "btl_tcp_if_include", mpi_if])
    cmd.extend(["--bind-to", "none", "--tag-output", "--wdir", str(cwd)])
    cmd.extend([str(test_executable), f"--gtest_filter={SLICE_MAPPING_GTEST_FILTER}"])

    logger.info(f"Running: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
        if result.returncode != 0:
            if result.stderr:
                logger.error(result.stderr.rstrip())
            logger.error(f"{cmd} Failed to generate slice to pcie device mapping")
            sys.exit(result.returncode)
    except KeyboardInterrupt:
        logger.error(f"{cmd} Interrupted")
        sys.exit(1)

    mapping_path = cwd / mapping_file
    if not mapping_path.exists():
        logger.error(f"{mapping_path} not found")
        sys.exit(1)
    return mapping_path


def _match_slice_host_key(device_mapping, host):
    """Map a requested host to its key in the slice mapping (exact or short-name match)."""
    if host in device_mapping:
        return host
    host_short = host.split(".")[0]
    for key in device_mapping:
        if key == host_short or key.split(".")[0] == host_short:
            return key
    logger.error(f"Host {host!r} not found in slice mapping (keys: {sorted(device_mapping)})")
    sys.exit(1)


def resolve_fabric_test_visible_devices_slices(
    fabric_config,
    hosts,
    mpi_if=None,
    run_discovery=True,
    work_dir=None,
):
    """Discover 2x4 slices across hosts; return one TT_VISIBLE_DEVICES per rank.

    Ranks are ordered host-major: host0 local mesh 0, host0 local mesh 1,
    host1 local mesh 0, ... matching the rankfile slot order in run_fabric_tests.sh.
    """
    if fabric_config not in FABRIC_TEST_SLICE_MAPPINGS:
        logger.error(f"Unknown slice config {fabric_config!r}. Supported: {sorted(FABRIC_TEST_SLICE_MAPPINGS)}")
        sys.exit(1)
    if not hosts:
        logger.error("Slice-based resolution requires at least one host (--hosts)")
        sys.exit(1)

    local_mesh_to_slices = FABRIC_TEST_SLICE_MAPPINGS[fabric_config]
    cwd = Path(work_dir) if work_dir else Path.cwd()
    mapping_path = cwd / SLICE_MAPPING_FILE

    if run_discovery:
        generate_slice_to_pcie_device_mapping(hosts, mpi_if=mpi_if, work_dir=cwd)
    elif not mapping_path.exists():
        logger.error(f"{mapping_path} not found (run discovery or generate it first)")
        sys.exit(1)

    with open(mapping_path, "r") as f:
        slice_mapping = yaml.safe_load(f)
    device_mapping = slice_mapping.get("device_mapping", {})

    visible = []
    for host in hosts:
        host_key = _match_slice_host_key(device_mapping, host)
        host_slices = device_mapping[host_key]
        for local_mesh in sorted(local_mesh_to_slices):
            devices = []
            for slice_id in local_mesh_to_slices[local_mesh]:
                if slice_id not in host_slices:
                    logger.error(f"Slice {slice_id} not found for host {host_key} in {mapping_path}")
                    sys.exit(1)
                slice_devices = sorted(host_slices[slice_id])
                if len(slice_devices) != DEVICES_PER_SLICE:
                    logger.error(
                        f"Slice {slice_id} on {host_key} resolved to {len(slice_devices)} devices, "
                        f"expected {DEVICES_PER_SLICE}"
                    )
                    sys.exit(1)
                devices.extend(slice_devices)
            visible.append(",".join(map(str, devices)))
    return visible


def build_quad_split_rank_table(hosts):
    """Canonical 12-rank table for the quad split 4x4 layout (8x4x4z).

    8 meshes across 4 ring-ordered hosts:
      even mesh 2i  -> single-host 4x4 on hosts[i], slices {1,2}, host_rank 0
      odd  mesh 2i+1 -> split 4x4: host_rank 0 = hosts[i]   slice {3}
                                    host_rank 1 = hosts[i+1] slice {0}  (ring wrap)

    Entries are ordered mesh-major then host-rank-minor, matching control_plane's
    (mesh_id, host_rank) -> mpi_rank assignment, so the launcher must emit MPMD
    segments / rankfile slots in exactly this order. Each entry is a tuple
    (mesh_id, host_rank, host, slice_ids).
    """
    if len(hosts) != 4:
        logger.error(f"8x4x4z requires exactly 4 hosts in ring order, got {len(hosts)}: {hosts}")
        sys.exit(1)

    table = []
    for mesh_id in range(8):
        host_idx = mesh_id // 2
        if mesh_id % 2 == 0:
            table.append((mesh_id, 0, hosts[host_idx], [1, 2]))
        else:
            table.append((mesh_id, 0, hosts[host_idx], [3]))
            table.append((mesh_id, 1, hosts[(host_idx + 1) % 4], [0]))
    return table


def resolve_quad_split_rank_table(hosts, mpi_if=None, run_discovery=True, work_dir=None):
    """Resolve the 8x4x4z 12-rank table from a single local 2x4 slice discovery.

    Discovery runs --np 1 on the local host and the resulting slice -> device map
    is replicated across all four ring hosts (identical galaxies), so no
    cross-host discovery is needed and the whole thing runs inside one docker
    container.

    Returns a list (in canonical rank order) of dicts:
      {"mesh_id": int, "host_rank": int, "host": str, "devices": "d0,d1,..."}.
    """
    table = build_quad_split_rank_table(hosts)
    cwd = Path(work_dir) if work_dir else Path.cwd()
    mapping_path = cwd / SLICE_MAPPING_FILE

    # Discover the 2x4 slice -> device numbering with a single LOCAL rank
    # (mpirun --np 1, no --host => no ssh) rather than across all four ring hosts.
    # Every Blackhole Galaxy in the pod is wired identically, so the local host's
    # slice -> device map applies to every host. This keeps discovery entirely
    # inside one docker container, so the script works when the local env only has
    # the pulled image and no native build. The launch host need not be hosts[0].
    if run_discovery:
        generate_slice_to_pcie_device_mapping(None, mpi_if=mpi_if, work_dir=cwd)
    elif not mapping_path.exists():
        logger.error(f"{mapping_path} not found (run discovery or generate it first)")
        sys.exit(1)

    with open(mapping_path, "r") as f:
        slice_mapping = yaml.safe_load(f)
    device_mapping = slice_mapping.get("device_mapping", {})
    if not device_mapping:
        logger.error(f"No device_mapping found in {mapping_path}")
        sys.exit(1)

    # Single-host discovery yields exactly one host entry; reuse its slice -> device
    # map for every ring host (homogeneous galaxies).
    reference_slices = next(iter(device_mapping.values()))

    rank_table = []
    for mesh_id, host_rank, host, slice_ids in table:
        devices = []
        for slice_id in slice_ids:
            if slice_id not in reference_slices:
                logger.error(f"Slice {slice_id} not found in {mapping_path}")
                sys.exit(1)
            slice_devices = sorted(reference_slices[slice_id])
            if len(slice_devices) != DEVICES_PER_SLICE:
                logger.error(
                    f"Slice {slice_id} resolved to {len(slice_devices)} devices, " f"expected {DEVICES_PER_SLICE}"
                )
                sys.exit(1)
            devices.extend(slice_devices)
        rank_table.append(
            {
                "mesh_id": mesh_id,
                "host_rank": host_rank,
                "host": host,
                "devices": ",".join(map(str, devices)),
            }
        )
    return rank_table


def parse_args():
    parser = argparse.ArgumentParser(description="Generate rank bindings from tray discovery")
    parser.add_argument(
        "--fabric-config",
        choices=sorted(FABRIC_TEST_TRAY_MAPPINGS.keys()),
        help="Single-host fabric test layout (used by run_fabric_tests.sh 4x8z/2x4x4z and per-host on 8x4x4z)",
    )
    parser.add_argument(
        "--slice-config",
        choices=SLICE_CONFIG_CHOICES,
        help="Slice-based fabric test layout (used by run_fabric_tests.sh 2x4x4z/8x4x4z)",
    )
    parser.add_argument(
        "--hosts",
        default="",
        help="With --slice-config: comma-separated hosts (in rank/mesh order; ring order for 8x4x4z)",
    )
    parser.add_argument(
        "--mpi-if",
        default=None,
        help="With --slice-config: network interface for the discovery mpirun (btl_tcp_if_include)",
    )
    parser.add_argument(
        "--print-devices",
        action="store_true",
        help="With --fabric-config/--slice-config 2x4x4z: print one TT_VISIBLE_DEVICES line per rank (stdout)",
    )
    parser.add_argument(
        "--print-rank-table",
        action="store_true",
        help="With --slice-config 8x4x4z: print one FABRIC_RANK:<mesh_id>;<host_rank>;<host>;<devices> line per rank",
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
    if args.slice_config == "8x4x4z":
        hosts = [h for h in args.hosts.split(",") if h]
        rank_table = resolve_quad_split_rank_table(
            hosts,
            mpi_if=args.mpi_if,
            run_discovery=not args.no_discovery,
            work_dir=args.work_dir,
        )
        if args.print_rank_table:
            for entry in rank_table:
                print(
                    f"{FABRIC_RANK_PREFIX}{entry['mesh_id']};{entry['host_rank']};"
                    f"{entry['host']};{entry['devices']}"
                )
        else:
            for rank, entry in enumerate(rank_table):
                print(
                    f"rank {rank}: mesh_id={entry['mesh_id']} host_rank={entry['host_rank']} "
                    f"host={entry['host']} TT_VISIBLE_DEVICES={entry['devices']}"
                )
    elif args.slice_config:
        hosts = [h for h in args.hosts.split(",") if h]
        devices = resolve_fabric_test_visible_devices_slices(
            args.slice_config,
            hosts,
            mpi_if=args.mpi_if,
            run_discovery=not args.no_discovery,
            work_dir=args.work_dir,
        )
        if args.print_devices:
            for line in devices:
                print(f"{FABRIC_VISIBLE_DEVICES_PREFIX}{line}")
        else:
            for rank, visible in enumerate(devices):
                print(f"rank {rank}: TT_VISIBLE_DEVICES={visible}")
    elif args.fabric_config:
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
