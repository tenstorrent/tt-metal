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

    # When running remotely, the file is written on the workers not the runner.
    # Copy it back from the first host.
    if not os.path.exists(mapping_file) and mpi_user:
        remote_host = f"{mpi_user}@{host_vector[0]}"
        scp_cmd = [
            "scp",
            "-o",
            "StrictHostKeyChecking=no",
            "-o",
            "UserKnownHostsFile=/dev/null",
            f"{remote_host}:{mapping_file}",
            mapping_file,
        ]
        logger.info(f"Copying mapping file from worker: {' '.join(scp_cmd)}")
        scp_result = subprocess.run(scp_cmd)
        if scp_result.returncode != 0:
            logger.error(f"Failed to copy {mapping_file} from {remote_host}")
            sys.exit(1)

    if not os.path.exists(mapping_file):
        logger.error(f"{mapping_file} not found")
        sys.exit(1)


def generate_rank_bindings(pipeline_config, physical_mapping_file, worker_tt_metal_home=None):
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

    # When workers have tt-metal at a different path than the runner, override
    # TT_MESH_GRAPH_DESC_PATH via global_env so workers resolve the correct absolute path.
    # tt-run validates mesh_graph_desc_path against the local (runner) filesystem,
    # but global_env overrides the env var that actually reaches the workers.
    if worker_tt_metal_home:
        mgd_path = pipeline_config["mesh_graph_desc_path"]
        rank_binding_configs["global_env"] = {
            "TT_MESH_GRAPH_DESC_PATH": str(Path(worker_tt_metal_home) / mgd_path),
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
    pipeline_config_file,
    test_executable_path=None,
    mpi_user=None,
    skip_mpi_net_filter=False,
    hostfile=None,
    worker_tt_metal_home=None,
):
    with open(pipeline_config_file, "r") as f:
        config = yaml.safe_load(f)

    # Extract unique hosts in order of first appearance
    config_hosts = []
    seen_hosts = set()
    for entry in config["stage_to_slice_mapping"].values():
        host = entry["host"]
        if host not in seen_hosts:
            config_hosts.append(host)
            seen_hosts.add(host)

    # If a hostfile is provided, replace config hosts with allocated hosts
    if hostfile:
        with open(hostfile, "r") as f:
            allocated_hosts = [line.strip() for line in f if line.strip()]
        if len(allocated_hosts) != len(config_hosts):
            logger.error(
                f"Hostfile has {len(allocated_hosts)} hosts but pipeline config has {len(config_hosts)} unique hosts"
            )
            sys.exit(1)
        host_map = dict(zip(config_hosts, allocated_hosts))
        logger.info(f"Remapping hosts: {host_map}")
        for entry in config["stage_to_slice_mapping"].values():
            entry["host"] = host_map[entry["host"]]
        config_hosts = allocated_hosts

    host_vector = config_hosts
    # Generate the list of PCIe devices for each physical slice in the pipeline
    # Metal generates the list of PCIe devices per physical slice in this file
    # when generate_slice_to_pcie_device_mapping is called
    physical_mapping_file = "slice_to_pcie_device_mapping.yaml"
    generate_slice_to_pcie_device_mapping(
        physical_mapping_file, host_vector, test_executable_path, mpi_user, skip_mpi_net_filter
    )
    # Using the generated list of PCIe devices per slice and the stage to physical
    # slice mapping, generate rank bindings for the pipeline
    generate_rank_bindings(config, physical_mapping_file, worker_tt_metal_home)

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
    parser.add_argument(
        "--hostfile",
        type=str,
        default=None,
        help="File with one hostname per line. Overrides hosts in pipeline config (matched by order of appearance).",
    )
    parser.add_argument(
        "--worker-tt-metal-home",
        type=str,
        default=None,
        help="Absolute path to tt-metal on workers (e.g. /home/user/tt-metal). "
        "Used to override TT_MESH_GRAPH_DESC_PATH when workers have a different filesystem layout than the runner.",
    )
    args = parser.parse_args()
    generate_pipeline_config_files(
        args.pipeline_config_file,
        args.physical_discovery_test_path,
        args.mpi_user,
        args.skip_mpi_net_filter,
        args.hostfile,
        args.worker_tt_metal_home,
    )
