#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import argparse
import os
import subprocess
import sys
from pathlib import Path

import yaml
from loguru import logger


def generate_slice_to_pcie_device_mapping(
    mapping_file, host_vector, test_executable_path=None, mpi_user=None, skip_mpi_net_filter=False
):
    # Use optional input, then env (e.g. CI where build lives on workers), then default
    test_executable = Path(
        test_executable_path
        or os.environ.get("TT_PHYSICAL_DISCOVERY_TEST_PATH")
        or "build/test/tt_metal/tt_fabric/test_physical_discovery"
    )

    explicit_path = test_executable_path or os.environ.get("TT_PHYSICAL_DISCOVERY_TEST_PATH")
    if not explicit_path and not test_executable.exists():
        logger.error(f"Test executable not found at {test_executable}")
        logger.info("Please build with: ./build_metal.sh --build-tests")
        sys.exit(1)
    if mpi_user:
        host_vector_str = ",".join(f"{mpi_user}@{h}" for h in host_vector)
    else:
        host_vector_str = ",".join(host_vector)
    MAPPING_GENERATION_CMD = [
        "mpirun",
        "--np",
        str(len(host_vector)),
        "--host",
        host_vector_str,
        "--mca",
        "btl",
        "self,tcp",
    ]
    if not skip_mpi_net_filter:
        MAPPING_GENERATION_CMD.extend(["--mca", "btl_tcp_if_include", "ens5f0np0"])
    MAPPING_GENERATION_CMD.extend(["--bind-to", "none", "--tag-output"])
    # When using a remote/CI path, pass lib and runtime root so workers can load libtt_metal.so
    if explicit_path and test_executable.is_absolute():
        build_dir = test_executable.parent.parent.parent.parent  # .../build/test/tt_metal/tt_fabric
        runtime_root = build_dir.parent
        ld_library_path = build_dir / "lib"
        MAPPING_GENERATION_CMD.extend(
            ["-x", f"LD_LIBRARY_PATH={ld_library_path}", "-x", f"TT_METAL_RUNTIME_ROOT={runtime_root}"]
        )
    MAPPING_GENERATION_CMD.extend([str(test_executable), "--gtest_filter=*Generate2x4SliceToPCIeDeviceMapping*"])

    logger.info(f"Running: {' '.join(MAPPING_GENERATION_CMD)}")

    try:
        result = subprocess.run(MAPPING_GENERATION_CMD)
        if result.returncode != 0:
            logger.error(f"{MAPPING_GENERATION_CMD} Failed to generate slice to PCIe device mapping")
            sys.exit(result.returncode)
    except KeyboardInterrupt:
        logger.error(f"{MAPPING_GENERATION_CMD} Interrupted")
        sys.exit(1)

    if not os.path.exists(mapping_file):
        logger.error(f"{mapping_file} not found")
        sys.exit(1)


def generate_rank_bindings(pipeline_config, physical_mapping_file):
    with open(physical_mapping_file, "r") as f:
        slice_to_pcie_device_mapping = yaml.safe_load(f)

    num_pipeline_stages = len(pipeline_config["stage_to_slice_mapping"])
    rank_bindings = []

    for stage in range(num_pipeline_stages):
        if stage not in pipeline_config["stage_to_slice_mapping"]:
            logger.error(f"Stage {stage} not found in stage to slice mapping. Please check the pipeline config file.")
            sys.exit(1)
        stage_host = pipeline_config["stage_to_slice_mapping"][stage]["host"]
        stage_slice = pipeline_config["stage_to_slice_mapping"][stage]["slice"]
        devices_for_stage = sorted(slice_to_pcie_device_mapping["device_mapping"][stage_host][stage_slice])
        rank_bindings.append(
            {
                "rank": stage,
                "mesh_id": stage,
                "mesh_host_rank": 0,
                "env_overrides": {"TT_VISIBLE_DEVICES": ",".join(map(str, devices_for_stage))},
            }
        )

    rank_binding_configs = {
        "rank_bindings": rank_bindings,
        "mesh_graph_desc_path": pipeline_config["mesh_graph_desc_path"],
    }
    with open(pipeline_config["rank_binding_file"], "w") as f:
        yaml.dump(rank_binding_configs, f, default_flow_style=False, sort_keys=False)


def generate_rank_file(pipeline_config):
    num_pipeline_stages = len(pipeline_config["stage_to_slice_mapping"])
    with open(pipeline_config["rank_file"], "w") as f:
        for stage in range(num_pipeline_stages):
            host = pipeline_config["stage_to_slice_mapping"][stage]["host"]
            f.write(f"rank {stage}={host} slot=0-31\n")


def generate_pipeline_config_files(
    pipeline_config_file, test_executable_path=None, mpi_user=None, skip_mpi_net_filter=False
):
    with open(pipeline_config_file, "r") as f:
        config = yaml.safe_load(f)

    host_vector = []
    seen_hosts = set()
    for entry in config["stage_to_slice_mapping"].values():
        host = entry["host"]
        if host not in seen_hosts:
            host_vector.append(host)
            seen_hosts.add(host)
    # Generate the list of PCIe devices for each physical slice in the pipeline
    # Metal generates the list of PCIe devices per physical slice in this file
    # when generate_slice_to_pcie_device_mapping is called
    physical_mapping_file = "slice_to_pcie_device_mapping.yaml"
    generate_slice_to_pcie_device_mapping(
        physical_mapping_file, host_vector, test_executable_path, mpi_user, skip_mpi_net_filter
    )
    # Using the generated list of PCIe devices per slice and the stage to physical
    # slice mapping, generate rank bindings for the pipeline
    generate_rank_bindings(config, physical_mapping_file)

    # Using the stage to physical slice mapping, generate the rank file for the pipeline
    generate_rank_file(config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate pipeline config files for blitz decode")
    parser.add_argument("pipeline_config_file", type=str, help="Path to the pipeline config YAML file")
    parser.add_argument(
        "--physical-discovery-test-path",
        type=str,
        default=None,
        help="Path to test_physical_discovery executable (e.g. when build lives on remote workers in CI)",
    )
    parser.add_argument(
        "--mpi-user",
        type=str,
        default=None,
        help="SSH user for mpirun (e.g. 'user' to connect as user@host instead of current user)",
    )
    parser.add_argument(
        "--skip-mpi-net-filter",
        action="store_true",
        help="Skip adding --mca btl_tcp_if_include ens5f0np0 (use in CI where btl_tcp_if_exclude is already set)",
    )
    args = parser.parse_args()
    generate_pipeline_config_files(
        args.pipeline_config_file, args.physical_discovery_test_path, args.mpi_user, args.skip_mpi_net_filter
    )
