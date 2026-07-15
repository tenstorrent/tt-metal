#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

import yaml
from loguru import logger

SLICE_MAPPING_GTEST_FILTER = "*Generate2x4SliceToPCIeDeviceMapping*"
SLICE_MAPPING_FILE = "slice_to_pcie_device_mapping.yaml"
DEVICES_PER_SLICE = 8

SLICES_PER_STAGE = {
    "4x2": 1,
    "4x4": 2,
    "8x4": 4,
}

ALLOWED_STAGE_SLICE_SPLITS = {
    "4x2": {(1,)},
    "4x4": {(2,), (1, 1)},
    "8x4": {(4,)},
}


def _parse_host(hostname):
    """Parse hostname into (host_num, u_num) or None if unrecognised."""
    match = re.search(r"(\d+)u(\d{2})$", hostname)
    if not match:
        logger.warning(
            f"Hostname '{hostname}' does not match pattern '<digits>u<2 digits>'; placing last in canonical order"
        )
        return None
    return int(match.group(1)), int(match.group(2))


def sort_hosts_canonical(hosts):
    """
    Sort hostnames into canonical pipeline order (snake/zigzag):
    low_host u08, low_host u02, high_host u02, high_host u08.
    Example: c09u08, c09u02, c10u02, c10u08.

    Within even-indexed host-number groups (0th, 2nd, …) u is descending (08 before 02).
    Within odd-indexed host-number groups (1st, 3rd, …) u is ascending (02 before 08).
    """
    parsed = [(h, _parse_host(h)) for h in hosts]
    unrecognised = [h for h, p in parsed if p is None]
    recognised = [(h, p) for h, p in parsed if p is not None]

    from collections import defaultdict

    groups = defaultdict(list)
    for h, (host_num, u_num) in recognised:
        groups[host_num].append((u_num, h))

    result = []
    for group_idx, host_num in enumerate(sorted(groups)):
        entries = groups[host_num]
        reverse = group_idx % 2 == 0
        entries.sort(key=lambda e: e[0], reverse=reverse)
        result.extend(h for _, h in entries)

    result.extend(unrecognised)
    return result


def _ordered_dense_int_key_values(raw_mapping, mapping_name):
    """Normalize a YAML mapping keyed by 0..N-1 into an ordered value list.

    The pipeline config uses mappings for stages and mesh_host_ranks, but YAML may
    deserialize those keys as strings. We canonicalize them to integers, reject
    sparse/non-monotonic layouts such as {0, 2}, and then return values in
    deterministic order so the rest of the code can safely use enumerate(...).
    """
    if not isinstance(raw_mapping, dict) or not raw_mapping:
        logger.error(f"{mapping_name} must be a non-empty mapping.")
        sys.exit(1)

    normalized_values = {}
    for raw_key, value in raw_mapping.items():
        try:
            int_key = int(raw_key)
        except (TypeError, ValueError):
            logger.error(f"{mapping_name} keys must be integers. Found key: {raw_key!r}")
            sys.exit(1)
        if int_key in normalized_values:
            logger.error(f"{mapping_name} contains duplicate key {int_key!r} after integer normalization.")
            sys.exit(1)
        normalized_values[int_key] = value

    expected_keys = list(range(len(normalized_values)))
    if sorted(normalized_values) != expected_keys:
        logger.error(
            f"{mapping_name} keys must be dense and monotonic starting at 0. "
            f"Expected {expected_keys}, found {sorted(normalized_values)}."
        )
        sys.exit(1)

    return [normalized_values[index] for index in expected_keys]


def _normalize_slice_id(raw_slice_id, context):
    """Convert a single slice ID to int before using it for lookups/validation."""
    try:
        return int(raw_slice_id)
    except (TypeError, ValueError):
        logger.error(f"{context} must be an integer. Found {raw_slice_id!r}.")
        sys.exit(1)


def _normalize_slice_ids(raw_slice_ids, context):
    """Normalize one contribution's slice list and reject duplicate slice IDs."""
    if not isinstance(raw_slice_ids, list) or not raw_slice_ids:
        logger.error(f"{context} must contain a non-empty list of slice IDs.")
        sys.exit(1)

    slice_ids = [_normalize_slice_id(raw_slice_id, f"{context} entry") for raw_slice_id in raw_slice_ids]
    if len(set(slice_ids)) != len(slice_ids):
        logger.error(f"{context} must not contain duplicate slice IDs. Found {slice_ids}.")
        sys.exit(1)
    return slice_ids


def _format_allowed_stage_slice_splits(stage_size):
    return ", ".join(str(list(split)) for split in sorted(ALLOWED_STAGE_SLICE_SPLITS[stage_size]))


def _validate_stage_contributions(stage_contributions, stage_size):
    """Enforce the allowed slice layout for the requested stage size.

    This runs after schema-specific parsing so both config formats feed into the
    same validation step:
      - 4x2 stages must resolve to [1]
      - 4x4 stages must resolve to [2] or [1, 1]
      - 8x4 stages must resolve to [4]
    """
    expected_slices_per_stage = SLICES_PER_STAGE[stage_size]
    allowed_slice_splits = ALLOWED_STAGE_SLICE_SPLITS[stage_size]

    for stage, contributions in enumerate(stage_contributions):
        stage_slice_split = tuple(len(contribution["slice_ids"]) for contribution in contributions)
        total_slices = sum(stage_slice_split)
        if total_slices != expected_slices_per_stage:
            logger.error(
                f"Stage {stage} must resolve to exactly {expected_slices_per_stage} local 2x4 slices "
                f"for stage size {stage_size}. Found {total_slices}."
            )
            sys.exit(1)
        if stage_slice_split not in allowed_slice_splits:
            logger.error(
                f"Stage {stage} has invalid slice split {list(stage_slice_split)} for stage size {stage_size}. "
                f"Allowed splits: {_format_allowed_stage_slice_splits(stage_size)}."
            )
            sys.exit(1)


def build_stage_contributions(pipeline_config, stage_size):
    """Normalize either config schema into per-stage host contributions.

    We keep the existing external schemas:
      - stage_to_slice_mapping for simple 4x2 configs
      - stage_to_mesh_host_mapping for multi-slice stages

    Both paths are converted into the same internal shape so downstream device
    resolution only needs to reason about hosts plus slice_ids.
    """
    if stage_size == "4x2":
        stage_to_slice_mapping = _ordered_dense_int_key_values(
            pipeline_config.get("stage_to_slice_mapping"), "stage_to_slice_mapping"
        )
        stage_contributions = []
        for stage, entry in enumerate(stage_to_slice_mapping):
            if "host" not in entry or "slice" not in entry:
                logger.error(f"Stage {stage} must contain both 'host' and 'slice'.")
                sys.exit(1)
            stage_contributions.append(
                [
                    {
                        "mesh_host_rank": 0,
                        "host": entry["host"],
                        "slice_ids": [_normalize_slice_id(entry["slice"], f"Stage {stage} 'slice'")],
                    }
                ]
            )
        _validate_stage_contributions(stage_contributions, stage_size)
        return stage_contributions

    stage_to_mesh_host_mapping = _ordered_dense_int_key_values(
        pipeline_config.get("stage_to_mesh_host_mapping"), "stage_to_mesh_host_mapping"
    )
    stage_contributions = []
    for stage, mesh_host_mapping in enumerate(stage_to_mesh_host_mapping):
        mesh_host_contributions = _ordered_dense_int_key_values(
            mesh_host_mapping, f"stage_to_mesh_host_mapping[{stage}]"
        )
        contributions = []
        for mesh_host_rank, contribution in enumerate(mesh_host_contributions):
            if "host" not in contribution or "slices" not in contribution:
                logger.error(f"Stage {stage}, mesh_host_rank {mesh_host_rank} must contain 'host' and 'slices'.")
                sys.exit(1)

            contributions.append(
                {
                    "mesh_host_rank": mesh_host_rank,
                    "host": contribution["host"],
                    "slice_ids": _normalize_slice_ids(
                        contribution["slices"], f"Stage {stage}, mesh_host_rank {mesh_host_rank} 'slices'"
                    ),
                }
            )

        stage_contributions.append(contributions)

    _validate_stage_contributions(stage_contributions, stage_size)
    return stage_contributions


def collect_unique_hosts(stage_contributions):
    unique_hosts = []
    seen_hosts = set()
    for contributions in stage_contributions:
        for contribution in contributions:
            host = contribution["host"]
            if host not in seen_hosts:
                unique_hosts.append(host)
                seen_hosts.add(host)
    return unique_hosts


def remap_stage_contribution_hosts(stage_contributions, host_map):
    remapped_contributions = []
    for contributions in stage_contributions:
        remapped_stage_contributions = []
        for contribution in contributions:
            remapped_stage_contributions.append(
                {
                    **contribution,
                    "host": host_map[contribution["host"]],
                }
            )
        remapped_contributions.append(remapped_stage_contributions)
    return remapped_contributions


def generate_slice_to_pcie_device_mapping(host_vector, mpi_user=None, worker_tt_metal_home=None, output_dir=None):
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

    cmd.extend([str(test_executable), f"--gtest_filter={SLICE_MAPPING_GTEST_FILTER}"])

    logger.info(f"Running: {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd)
        if result.returncode != 0:
            logger.error(f"{cmd} Failed to generate device mapping")
            sys.exit(result.returncode)
    except KeyboardInterrupt:
        logger.error(f"{cmd} Interrupted")
        sys.exit(1)

    actual_mapping_file = os.path.join(output_dir, SLICE_MAPPING_FILE) if output_dir else SLICE_MAPPING_FILE

    if not os.path.exists(actual_mapping_file):
        logger.error(f"{actual_mapping_file} not found")
        sys.exit(1)

    return actual_mapping_file


def resolve_devices_for_contribution(host, slice_ids, device_mapping, physical_mapping_file):
    if host not in device_mapping:
        logger.error(f"Host {host} not found in {physical_mapping_file}.")
        sys.exit(1)

    host_device_mapping = device_mapping[host]
    resolved_devices = []
    for slice_id in slice_ids:
        if slice_id not in host_device_mapping:
            logger.error(f"Slice {slice_id} for host {host} not found in {physical_mapping_file}.")
            sys.exit(1)
        devices_for_slice = sorted(host_device_mapping[slice_id])
        if len(devices_for_slice) != DEVICES_PER_SLICE:
            logger.error(
                f"Slice {slice_id} on host {host} resolved to {len(devices_for_slice)} devices, "
                f"expected {DEVICES_PER_SLICE}."
            )
            sys.exit(1)
        resolved_devices.extend(devices_for_slice)

    if len(set(resolved_devices)) != len(resolved_devices):
        logger.error(f"Host {host} contribution {slice_ids} resolves to duplicate devices in {physical_mapping_file}.")
        sys.exit(1)

    return sorted(resolved_devices)


def generate_rank_bindings(
    pipeline_config, stage_contributions, physical_mapping_file, worker_tt_metal_home=None, stage_size="4x2"
):
    with open(physical_mapping_file, "r") as f:
        physical_device_mapping = yaml.safe_load(f)

    rank_bindings = []
    rank_to_host = []
    device_mapping = physical_device_mapping.get("device_mapping", {})
    expected_devices_per_stage = SLICES_PER_STAGE[stage_size] * DEVICES_PER_SLICE

    for stage, contributions in enumerate(stage_contributions):
        stage_device_count = 0
        for contribution in contributions:
            devices_for_rank = resolve_devices_for_contribution(
                contribution["host"],
                contribution["slice_ids"],
                device_mapping,
                physical_mapping_file,
            )
            expected_devices_for_rank = DEVICES_PER_SLICE * len(contribution["slice_ids"])
            if len(devices_for_rank) != expected_devices_for_rank:
                logger.error(
                    f"Stage {stage}, mesh_host_rank {contribution['mesh_host_rank']} resolved to "
                    f"{len(devices_for_rank)} devices, expected {expected_devices_for_rank}."
                )
                sys.exit(1)

            rank_bindings.append(
                {
                    "rank": len(rank_bindings),
                    "mesh_id": stage,
                    "mesh_host_rank": contribution["mesh_host_rank"],
                    "env_overrides": {"TT_VISIBLE_DEVICES": ",".join(map(str, devices_for_rank))},
                }
            )
            rank_to_host.append(contribution["host"])
            stage_device_count += len(devices_for_rank)

        if stage_device_count != expected_devices_per_stage:
            logger.error(
                f"Stage {stage} resolved to {stage_device_count} devices across all host contributions, "
                f"expected {expected_devices_per_stage} for stage size {stage_size}."
            )
            sys.exit(1)

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

    return rank_to_host


def generate_rank_file(rank_to_host, rank_file):
    with open(rank_file, "w") as f:
        for rank, host in enumerate(rank_to_host):
            f.write(f"rank {rank}={host} slot=0-31\n")


def generate_pipeline_config_files(
    pipeline_config_file,
    mpi_user=None,
    hostfile=None,
    worker_tt_metal_home=None,
    output_dir=None,
    stage_size="4x2",
):
    with open(pipeline_config_file, "r") as f:
        config = yaml.safe_load(f)
    stage_contributions = build_stage_contributions(config, stage_size)

    # Extract unique hosts in order of first appearance
    config_hosts = collect_unique_hosts(stage_contributions)

    # If a hostfile is provided, replace config hosts with allocated hosts
    if hostfile:
        with open(hostfile, "r") as f:
            allocated_hosts = [line.strip() for line in f if line.strip()]
        if len(allocated_hosts) != len(config_hosts):
            logger.error(
                f"Hostfile has {len(allocated_hosts)} hosts but pipeline config has {len(config_hosts)} unique hosts"
            )
            sys.exit(1)
        # Sort allocated hosts into canonical order (low_u08, low_u02, high_u02, high_u08)
        # so that they match config_hosts order regardless of hostfile ordering.
        allocated_hosts = sort_hosts_canonical(allocated_hosts)
        host_map = dict(zip(config_hosts, allocated_hosts))
        logger.info(f"Remapping hosts: {host_map}")
        stage_contributions = remap_stage_contribution_hosts(stage_contributions, host_map)
        config_hosts = allocated_hosts

    logger.info("Host index -> hostname mapping (for debugging):")
    for idx, hostname in enumerate(config_hosts):
        logger.info(f"  Host {idx}: {hostname}")

    host_vector = config_hosts
    actual_mapping_file = generate_slice_to_pcie_device_mapping(
        host_vector,
        mpi_user,
        worker_tt_metal_home,
        output_dir,
    )
    rank_to_host = generate_rank_bindings(
        config, stage_contributions, actual_mapping_file, worker_tt_metal_home, stage_size=stage_size
    )
    generate_rank_file(rank_to_host, config["rank_file"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate pipeline config files for blitz decode")
    parser.add_argument("pipeline_config_file", type=str, help="Path to the pipeline config YAML file")
    parser.add_argument(
        "--stage-size",
        type=str,
        choices=sorted(SLICES_PER_STAGE.keys()),
        default="4x2",
        help="Pipeline stage size used to validate slice composition and generate TT_VISIBLE_DEVICES mappings.",
    )
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
        help="File with one hostname per line. Overrides hosts in pipeline config. "
        "Hosts are sorted into canonical order (low_u08, low_u02, high_u02, high_u08) before matching.",
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
        stage_size=args.stage_size,
    )
