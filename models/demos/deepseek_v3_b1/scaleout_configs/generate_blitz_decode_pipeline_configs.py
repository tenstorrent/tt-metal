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


def _canonical_output_dir(output_dir: str) -> Path:
    """Resolve --output-dir to an absolute path; must exist and be a directory."""
    base = Path(output_dir).expanduser().resolve(strict=False)
    if not base.is_dir():
        logger.error(f"--output-dir must be an existing directory: {base}")
        sys.exit(1)
    return base


def _safe_path_under_output_dir(output_dir: str, configured_name: str) -> Path:
    """Resolve a writable path under output_dir using only the basename of configured_name.

    Prevents path traversal from untrusted --output-dir or YAML fields (SAST / Cycode).
    """
    base = _canonical_output_dir(output_dir)
    name = Path(configured_name).name
    if not name or name in (".", ".."):
        logger.error(f"Invalid output file name in pipeline config: {configured_name!r}")
        sys.exit(1)
    if os.sep in name or (os.altsep and os.altsep in name):
        logger.error(f"Output file name must be a single path component: {configured_name!r}")
        sys.exit(1)
    candidate = (base / name).resolve(strict=False)
    try:
        candidate.relative_to(base)
    except ValueError:
        logger.error(f"Refusing path outside --output-dir: {candidate} is not under {base}")
        sys.exit(1)
    return candidate


def _resolve_coordinator_tt_metal_home(explicit: str | None) -> Path | None:
    """tt-metal root on the runner (for resolving mesh_graph_desc_path)."""
    for candidate in (
        explicit,
        os.environ.get("TT_METAL_HOME"),
        os.environ.get("TT_METAL_COORDINATOR_HOME"),
    ):
        if not candidate:
            continue
        root = Path(candidate).expanduser().resolve(strict=False)
        if root.is_dir():
            return root
    return None


def _mesh_graph_desc_path_for_rank_binding(
    mesh_graph_desc_path_cfg: str,
    output_dir: str | None,
    coordinator_tt_metal_home: str | None,
) -> str:
    """Rank binding path tt-run validates on the runner.

    With --output-dir, CI does ``cd`` into the pipeline dir first; tt-run's
    ORIGINAL_CWD is then that directory, so repo-relative paths would wrongly
    resolve under the pipeline dir unless we emit an absolute path here.
    """
    if not output_dir:
        return mesh_graph_desc_path_cfg
    raw = Path(mesh_graph_desc_path_cfg)
    if raw.is_absolute():
        resolved = raw.resolve(strict=False)
        if not resolved.is_file():
            logger.error(f"mesh_graph_desc_path is not a file: {resolved}")
            sys.exit(1)
        return str(resolved)
    root = _resolve_coordinator_tt_metal_home(coordinator_tt_metal_home)
    if root is None:
        logger.error(
            "With --output-dir, mesh_graph_desc_path must resolve on the runner. "
            "Set TT_METAL_HOME or TT_METAL_COORDINATOR_HOME to the coordinator tt-metal checkout, "
            "or pass --coordinator-tt-metal-home."
        )
        sys.exit(1)
    root_resolved = root.resolve(strict=False)
    resolved = (root_resolved / raw).resolve(strict=False)
    try:
        resolved.relative_to(root_resolved)
    except ValueError:
        logger.error(f"mesh_graph_desc_path must stay under coordinator tt-metal home ({root_resolved})")
        sys.exit(1)
    if not resolved.is_file():
        logger.error(f"Mesh graph descriptor not found at {resolved}")
        sys.exit(1)
    return str(resolved)


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
        wdir = str(_canonical_output_dir(output_dir))
        cmd.extend(["--wdir", wdir])

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

    actual_mapping_file = str(_safe_path_under_output_dir(output_dir, mapping_file)) if output_dir else mapping_file

    if not os.path.exists(actual_mapping_file):
        logger.error(f"{actual_mapping_file} not found")
        sys.exit(1)

    return actual_mapping_file


def generate_rank_bindings(
    pipeline_config,
    physical_mapping_file,
    worker_tt_metal_home=None,
    output_dir=None,
    coordinator_tt_metal_home=None,
):
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

    mesh_graph_desc_path = _mesh_graph_desc_path_for_rank_binding(
        pipeline_config["mesh_graph_desc_path"],
        output_dir,
        coordinator_tt_metal_home,
    )
    rank_binding_configs = {
        "rank_bindings": rank_bindings,
        "mesh_graph_desc_path": mesh_graph_desc_path,
    }

    # When workers have tt-metal at a different path than the runner, override
    # TT_MESH_GRAPH_DESC_PATH via global_env so workers resolve the correct absolute path.
    # tt-run validates mesh_graph_desc_path against the local (runner) filesystem,
    # but global_env overrides the env var that actually reaches the workers.
    if worker_tt_metal_home:
        mgd_path = pipeline_config["mesh_graph_desc_path"]
        mgd_rel = Path(mgd_path)
        worker_mgd = (
            str((Path(worker_tt_metal_home) / mgd_rel).resolve(strict=False))
            if not mgd_rel.is_absolute()
            else str(mgd_rel.resolve(strict=False))
        )
        rank_binding_configs["global_env"] = {
            "TT_MESH_GRAPH_DESC_PATH": worker_mgd,
        }

    # With --output-dir, mpirun uses --wdir there for the slice map; write rank binding
    # there too so CI (cwd often /ci/tt-metal) matches tt-run after cd $PIPELINE_DIR.
    rank_binding_path = (
        str(_safe_path_under_output_dir(output_dir, pipeline_config["rank_binding_file"]))
        if output_dir
        else pipeline_config["rank_binding_file"]
    )
    with open(rank_binding_path, "w") as f:
        yaml.dump(rank_binding_configs, f, default_flow_style=False, sort_keys=False)


def generate_rank_file(pipeline_config, output_dir=None):
    num_pipeline_stages = len(pipeline_config["stage_to_slice_mapping"])
    rank_file_path = (
        str(_safe_path_under_output_dir(output_dir, pipeline_config["rank_file"]))
        if output_dir
        else pipeline_config["rank_file"]
    )
    with open(rank_file_path, "w") as f:
        for stage in range(num_pipeline_stages):
            host = pipeline_config["stage_to_slice_mapping"][stage]["host"]
            f.write(f"rank {stage}={host} slot=0-31\n")


def generate_pipeline_config_files(
    pipeline_config_file,
    mpi_user=None,
    hostfile=None,
    worker_tt_metal_home=None,
    output_dir=None,
    coordinator_tt_metal_home=None,
):
    if output_dir:
        output_dir = str(_canonical_output_dir(output_dir))

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
        # Sort allocated hosts into canonical order (low_u08, low_u02, high_u02, high_u08)
        # so that they match config_hosts order regardless of hostfile ordering.
        allocated_hosts = sort_hosts_canonical(allocated_hosts)
        host_map = dict(zip(config_hosts, allocated_hosts))
        logger.info(f"Remapping hosts: {host_map}")
        for entry in config["stage_to_slice_mapping"].values():
            entry["host"] = host_map[entry["host"]]
        config_hosts = allocated_hosts

    logger.info("Host index -> hostname mapping (for debugging):")
    for idx, hostname in enumerate(config_hosts):
        logger.info(f"  Host {idx}: {hostname}")

    host_vector = config_hosts
    physical_mapping_file = "slice_to_pcie_device_mapping.yaml"
    actual_mapping_file = generate_slice_to_pcie_device_mapping(
        physical_mapping_file, host_vector, mpi_user, worker_tt_metal_home, output_dir
    )
    generate_rank_bindings(config, actual_mapping_file, worker_tt_metal_home, output_dir, coordinator_tt_metal_home)
    generate_rank_file(config, output_dir)


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
    parser.add_argument(
        "--coordinator-tt-metal-home",
        type=str,
        default=None,
        help="Runner-side tt-metal root used to absolutize mesh_graph_desc_path when --output-dir is set. "
        "Defaults to TT_METAL_HOME, then TT_METAL_COORDINATOR_HOME.",
    )
    args = parser.parse_args()
    generate_pipeline_config_files(
        args.pipeline_config_file,
        args.mpi_user,
        args.hostfile,
        args.worker_tt_metal_home,
        args.output_dir,
        args.coordinator_tt_metal_home,
    )
