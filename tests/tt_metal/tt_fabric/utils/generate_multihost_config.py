#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import argparse
import copy
import os
import subprocess
import sys
from pathlib import Path

import yaml
from loguru import logger


def run_physical_discovery_multihost(hostfile, rankfile, num_hosts):
    """
    Run the physical discovery test across all hosts to generate:
    1. Mesh graph descriptor (textproto) - describes the full multi-host topology
    2. Physical system descriptor (YAML) - contains host-to-host connectivity
    """
    test_executable = Path("build/test/tt_metal/tt_fabric/test_physical_discovery")

    if not test_executable.exists():
        logger.error(f"Test executable not found at {test_executable}")
        logger.info("Please build with: ./build_metal.sh --build-tests")
        sys.exit(1)

    # Run the TestPhysicalSystemDescriptor test across all hosts
    # This will generate:
    # - physical_system_descriptor.textproto (mesh graph descriptor)
    # - physical_system_descriptor.yaml (system topology)

    # Use mpirun directly - no rank binding needed for discovery test
    mpi_args = [
        "--hostfile",
        hostfile,
        "--mca",
        "btl_tcp_if_exclude",
        "docker0,lo",
        "--mca",
        "btl",
        "self,tcp",
        "--tag-output",
        "--oversubscribe",
    ]

    if rankfile:
        mpi_args.extend(["--map-by", f"rankfile:file={rankfile}"])

    DISCOVERY_CMD = ["mpirun"] + mpi_args + [str(test_executable), "--gtest_filter=*TestPhysicalSystemDescriptor*"]

    logger.info(f"Running: {' '.join(DISCOVERY_CMD)}")

    try:
        result = subprocess.run(DISCOVERY_CMD, timeout=300)  # 5 minute timeout
        if result.returncode != 0:
            logger.error("Physical discovery test failed")
            sys.exit(result.returncode)
    except subprocess.TimeoutExpired:
        logger.error("Discovery test timed out after 5 minutes - possible hang or communication issue")
        sys.exit(1)
    except KeyboardInterrupt:
        logger.error("Discovery interrupted")
        sys.exit(1)

    # Check for generated files
    textproto_file = "physical_system_descriptor.textproto"
    yaml_file = "physical_system_descriptor.yaml"

    if not os.path.exists(textproto_file):
        logger.error(f"{textproto_file} not found - discovery may have failed")
        sys.exit(1)

    logger.info(f"Generated mesh graph descriptor: {textproto_file}")

    if os.path.exists(yaml_file):
        logger.info(f"Generated system topology: {yaml_file}")
        return textproto_file, yaml_file

    return textproto_file, None


def generate_rank_binding_for_multihost(
    num_hosts, num_devices_per_host, mesh_graph_desc_path, output_file, mesh_shape=None
):
    """
    Generate rank bindings for multi-host setup.

    For a 4-host setup with 32 devices per host (Galaxy):
    - Each host becomes a separate rank (4 ranks total)
    - Each rank sees all 32 devices on its host
    - All ranks participate in the same mesh
    """
    rank_bindings = []

    for rank in range(num_hosts):
        binding = {
            "rank": rank,
            "mesh_id": 0,  # All ranks in same mesh
            "mesh_host_rank": rank,  # Each rank has unique mesh_host_rank
        }
        rank_bindings.append(binding)

    rank_config = {
        "rank_bindings": rank_bindings,
        "mesh_graph_desc_path": mesh_graph_desc_path,
    }

    # Save to YAML file
    with open(output_file, "w") as f:
        yaml.dump(rank_config, f, default_flow_style=False, sort_keys=False)

    logger.info(f"Generated rank configuration file: {output_file}")
    logger.info(f"  - {num_hosts} ranks (1 per host)")
    logger.info(f"  - {num_devices_per_host} devices per host")
    logger.info(f"  - Mesh graph descriptor: {mesh_graph_desc_path}")

    return rank_config


def parse_system_yaml(yaml_file):
    """Parse the physical system descriptor YAML to extract system information."""
    if not yaml_file or not os.path.exists(yaml_file):
        return None

    with open(yaml_file, "r") as f:
        system_desc = yaml.safe_load(f)

    # Extract useful information
    info = {
        "num_hosts": len(system_desc.get("hosts", {})),
        "hosts": list(system_desc.get("hosts", {}).keys()),
        "exit_nodes": system_desc.get("exit_nodes", {}),
    }

    return info


def generate_multihost_configs(hostfile, rankfile=None, output_dir="."):
    """
    Main function to generate multi-host configuration files.

    This will:
    1. Run physical discovery across all hosts
    2. Generate mesh graph descriptor (textproto)
    3. Generate rank bindings for common multi-host topologies
    """
    # Determine number of hosts from hostfile
    num_hosts = 0
    if os.path.exists(hostfile):
        with open(hostfile, "r") as f:
            num_hosts = len([line for line in f if line.strip() and not line.startswith("#")])
    else:
        logger.error(f"Hostfile {hostfile} not found")
        sys.exit(1)

    logger.info(f"Detected {num_hosts} hosts in {hostfile}")

    # Run physical discovery to generate mesh graph descriptor
    textproto_file, yaml_file = run_physical_discovery_multihost(hostfile, rankfile, num_hosts)

    # Parse system information if available
    system_info = parse_system_yaml(yaml_file)
    if system_info:
        logger.info(f"System info: {system_info['num_hosts']} hosts detected")
        logger.info(f"Hosts: {', '.join(system_info['hosts'])}")

    # Generate rank bindings
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # For Galaxy systems (32 devices per host)
    if num_hosts == 2:
        logger.info("Generating 2-host (dual Galaxy) rank bindings")
        generate_rank_binding_for_multihost(
            num_hosts=2,
            num_devices_per_host=32,
            mesh_graph_desc_path=textproto_file,
            output_file=output_path / "dual_galaxy_rank_binding.yaml",
        )
    elif num_hosts == 4:
        logger.info("Generating 4-host (quad Galaxy) rank bindings")
        generate_rank_binding_for_multihost(
            num_hosts=4,
            num_devices_per_host=32,
            mesh_graph_desc_path=textproto_file,
            output_file=output_path / "quad_galaxy_rank_binding.yaml",
        )
    else:
        logger.info(f"Generating {num_hosts}-host rank bindings")
        generate_rank_binding_for_multihost(
            num_hosts=num_hosts,
            num_devices_per_host=32,  # Assume Galaxy
            mesh_graph_desc_path=textproto_file,
            output_file=output_path / f"{num_hosts}host_rank_binding.yaml",
        )

    logger.info(f"Multi-host configuration generation complete!")
    logger.info(f"Files generated:")
    logger.info(f"  - {textproto_file} (mesh graph descriptor)")
    if yaml_file:
        logger.info(f"  - {yaml_file} (system topology)")
    logger.info(f"  - {output_path}/*_rank_binding.yaml (rank bindings)")


def main():
    parser = argparse.ArgumentParser(
        description="Generate multi-host mesh graph descriptors and rank bindings by running physical discovery"
    )
    parser.add_argument(
        "--hostfile",
        type=str,
        required=True,
        help="Path to MPI hostfile listing all hosts in the cluster",
    )
    parser.add_argument(
        "--rankfile",
        type=str,
        help="Optional path to MPI rankfile for rank binding",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=".",
        help="Output directory for generated files (default: current directory)",
    )

    args = parser.parse_args()

    generate_multihost_configs(
        hostfile=args.hostfile,
        rankfile=args.rankfile,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
