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
    mapping_file, host_vector, mpi_user=None, worker_tt_metal_home=None, output_dir=None
):
    if worker_tt_metal_home:
        wh = Path(worker_tt_metal_home)
        test_executable = wh / "build/test/tt_metal/tt_fabric/test_physical_discovery"
    else:
        test_executable = Path("build/test/tt_metal/tt_fabric/test_physical_discovery")
        if not test_executable.exists():
            logger.error(f"Test executable not found at {test_executable}")
            logger.info("Please build with: ./build_metal.sh --build-tests")
            sys.exit(1)

    if mpi_user:
        host_vector_str = ",".join(f"{mpi_user}@{h}" for h in host_vector)
    else:
        host_vector_str = ",".join(host_vector)

    cmd = [
        "mpirun",
        "--np",
        str(len(host_vector)),
        "--host",
        host_vector_str,
        "--mca",
        "btl",
        "self,tcp",
    ]

    # When running locally, use btl_tcp_if_include. When running remotely
    # (worker_tt_metal_home is set), skip it to avoid conflicts with the
    # runner's OMPI_MCA_btl_tcp_if_exclude env var.
    if not worker_tt_metal_home:
        cmd.extend(["--mca", "btl_tcp_if_include", "ens5f0np0"])

    cmd.extend(["--bind-to", "none", "--tag-output"])

    if output_dir:
        cmd.extend(["--wdir", output_dir])

    if worker_tt_metal_home:
        wh = Path(worker_tt_metal_home)
        cmd.extend(["-x", f"LD_LIBRARY_PATH={wh / 'build/lib'}", "-x", f"TT_METAL_RUNTIME_ROOT={wh}"])

    cmd.extend([str(test_executable), "--gtest_filter=*Generate2x4SliceToPCIeDeviceMapping*"])

    logger.info(f"Running: {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd)
        if result.returncode != 0:
            logger.error(f"{cmd} Failed to generate slice to PCIe device mapping")
            sys.exit(result.returncode)
    except KeyboardInterrupt:
        logger.error(f"{cmd} Interrupted")
        sys.exit(1)

    actual_mapping_file = os.path.join(output_dir, mapping_file) if output_dir else mapping_file

    if not os.path.exists(actual_mapping_file):
        logger.error(f"{actual_mapping_file} not found")
        sys.exit(1)

    return actual_mapping_file


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
    mpi_user=None,
    hostfile=None,
    worker_tt_metal_home=None,
    output_dir=None,
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
    physical_mapping_file = "slice_to_pcie_device_mapping.yaml"
    actual_mapping_file = generate_slice_to_pcie_device_mapping(
        physical_mapping_file, host_vector, mpi_user, worker_tt_metal_home, output_dir
    )
    generate_rank_bindings(config, actual_mapping_file, worker_tt_metal_home)
    generate_rank_file(config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate pipeline config files for blitz decode")
    parser.add_argument("pipeline_config_file", type=str, help="Path to the pipeline config YAML file")
    parser.add_argument(
        "--mpi-user",
        type=str,
        default=None,
        help="SSH user for mpirun (e.g. 'user' to connect as user@host instead of current user)",
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
        "Implies remote execution: derives test executable path, library paths, "
        "skips local checks, skips btl_tcp_if_include, and overrides TT_MESH_GRAPH_DESC_PATH for workers.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Shared NFS directory for mpirun output (e.g. /ci/pipeline-123). "
        "When set, mpirun uses --wdir to write generated files here, avoiding scp.",
    )
    args = parser.parse_args()
    generate_pipeline_config_files(
        args.pipeline_config_file,
        args.mpi_user,
        args.hostfile,
        args.worker_tt_metal_home,
        args.output_dir,
    )
