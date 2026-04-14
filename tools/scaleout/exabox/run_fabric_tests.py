#!/usr/bin/env python3

# SPDX-FileCopyrightText: (c) 2025 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Run fabric tests on 4x8, 4x32, or 8x16 cluster configuration.

Two-phase execution:
  Phase 1: generate_rank_bindings runs inside Docker (via mpi-docker) to
           discover physical topology and produce rank_bindings.yaml
  Phase 2: mpi-docker launches the test binary inside Docker using
           the per-rank env vars from the generated rank bindings
"""

import argparse
import os
import re
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
MPI_DOCKER = str(SCRIPT_DIR / "mpi-docker")

MGD_PATHS = {
    "4x8": "tt_metal/fabric/mesh_graph_descriptors/single_bh_galaxy_torus_xy_graph_descriptor.textproto",
    "4x32": "tt_metal/fabric/mesh_graph_descriptors/32x4_quad_bh_galaxy_torus_xy_graph_descriptor.textproto",
    "8x16": "tt_metal/fabric/mesh_graph_descriptors/16x8_quad_bh_galaxy_torus_xy_graph_descriptor.textproto",
}

DEFAULT_TEST_BINARY = "./build/test/tt_metal/perf_microbenchmark/routing/test_tt_fabric"
DEFAULT_TEST_CONFIG = "tests/tt_metal/tt_metal/perf_microbenchmark/routing/test_bh_glx_2d_torus_stability.yaml"
GENERATE_RANK_BINDINGS = "build/tools/scaleout/generate_rank_bindings"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run fabric tests on 4x8, 4x32, or 8x16 cluster configuration.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Example:\n"
            "  %(prog)s --hosts bh-glx-c01u02,bh-glx-c01u08 \\\n"
            "     --image ghcr.io/tenstorrent/tt-metal/upstream-tests-wh-6u:latest"
        ),
    )
    parser.add_argument("--hosts", required=True, help="Comma-separated list of hosts (single host for 4x8)")
    parser.add_argument("--image", required=True, help="Docker image to use")
    parser.add_argument(
        "--config", choices=["4x8", "4x32", "8x16"], default="4x32", help="Mesh configuration (default: 4x32)"
    )
    parser.add_argument(
        "--output", default="fabric_test_logs", help="Output directory for log files (default: fabric_test_logs)"
    )
    parser.add_argument(
        "--mesh-graph-desc-path", default=None, help="Path to mesh graph descriptor file (overrides --config)"
    )
    parser.add_argument(
        "--test-binary", default=DEFAULT_TEST_BINARY, help=f"Path to test binary (default: {DEFAULT_TEST_BINARY})"
    )
    parser.add_argument(
        "--test-config",
        default=DEFAULT_TEST_CONFIG,
        help=f"Path to test configuration file (default: {DEFAULT_TEST_CONFIG})",
    )
    parser.add_argument("--filter", default=None, help="Filter pattern passed to test_tt_fabric --filter")
    return parser.parse_args()


def parse_rank_bindings_yaml(yaml_path: str) -> list[dict]:
    """Parse rank_bindings.yaml without PyYAML dependency.

    Extracts rank, mesh_id, and mesh_host_rank from each binding entry.
    """
    content = Path(yaml_path).read_text()
    bindings = []
    current = {}
    for line in content.splitlines():
        stripped = line.strip()
        if stripped.startswith("- rank:"):
            if current:
                bindings.append(current)
            current = {"rank": int(stripped.split(":")[1].strip())}
        elif stripped.startswith("mesh_id:") and current:
            current["mesh_id"] = int(stripped.split(":")[1].strip())
        elif stripped.startswith("mesh_host_rank:") and current:
            current["mesh_host_rank"] = int(stripped.split(":")[1].strip())
    if current:
        bindings.append(current)

    bindings.sort(key=lambda b: b["rank"])
    return bindings


def run_phase1(image: str, hosts_csv: str, mgd_path: str) -> list[dict]:
    """Run generate_rank_bindings inside Docker via mpi-docker.

    Returns parsed rank bindings (list of dicts with rank, mesh_id, mesh_host_rank).
    """
    output_dir = tempfile.mkdtemp(prefix="ttrun_phase1_")
    num_hosts = len(hosts_csv.split(","))

    cmd = [
        MPI_DOCKER,
        "--image",
        image,
        "--empty-entrypoint",
        "--host",
        hosts_csv,
        "-np",
        str(num_hosts),
        "-x",
        "LD_LIBRARY_PATH=build/lib",
        GENERATE_RANK_BINDINGS,
        "--mesh-graph-descriptor",
        mgd_path,
        "--output-dir",
        output_dir,
    ]

    print("Phase 1: Generating rank bindings via mpi-docker...")
    print(f"  Output dir: {output_dir}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"Error: Phase 1 (generate_rank_bindings) failed with exit code {result.returncode}", file=sys.stderr)
        sys.exit(result.returncode)

    rb_path = os.path.join(output_dir, "rank_bindings.yaml")
    if not os.path.exists(rb_path):
        print(f"Error: rank_bindings.yaml not found at {rb_path}", file=sys.stderr)
        sys.exit(1)

    bindings = parse_rank_bindings_yaml(rb_path)
    print(f"  Loaded {len(bindings)} rank bindings from {rb_path}")
    return bindings


def build_phase2_cmd(args: argparse.Namespace, mgd_path: str, rank_bindings: list[dict]) -> list:
    """Build mpi-docker command for Phase 2 using rank bindings from Phase 1."""
    extra_args = []
    if args.test_binary.endswith("test_tt_fabric"):
        extra_args += ["--show-progress", "--show-workers"]
    if args.filter:
        extra_args += ["--filter", args.filter]

    hosts = args.hosts
    if args.config == "4x8":
        hosts = args.hosts.split(",")[0]

    cmd = [
        MPI_DOCKER,
        "--image",
        args.image,
        "--empty-entrypoint",
        "--bind-to",
        "none",
        "--host",
        hosts,
    ]

    for i, binding in enumerate(rank_bindings):
        if i > 0:
            cmd.append(":")
        cmd += [
            "-np",
            "1",
            "-x",
            f"TT_MESH_ID={binding['mesh_id']}",
            "-x",
            f"TT_MESH_GRAPH_DESC_PATH={mgd_path}",
            "-x",
            f"TT_MESH_HOST_RANK={binding['mesh_host_rank']}",
            args.test_binary,
            "--test_config",
            args.test_config,
        ] + extra_args

    return cmd


def run_with_tee(cmd: list, log_path: str) -> int:
    """Run cmd, streaming combined stdout+stderr to both the terminal and log_path."""
    with open(log_path, "w") as log_file:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        try:
            for line in proc.stdout:
                sys.stdout.buffer.write(line)
                sys.stdout.buffer.flush()
                log_file.write(line.decode("utf-8", errors="replace"))
                log_file.flush()
        except KeyboardInterrupt:
            proc.terminate()
            proc.wait()
            raise
        return proc.wait()


def main() -> None:
    args = parse_args()

    mgd_path = args.mesh_graph_desc_path if args.mesh_graph_desc_path else MGD_PATHS[args.config]

    os.makedirs(args.output, exist_ok=True)
    log_file = os.path.join(args.output, f"fabric_tests_{datetime.now():%Y%m%d_%H%M%S}.log")

    hosts_csv = args.hosts
    if args.config == "4x8":
        hosts_csv = args.hosts.split(",")[0]

    print("==========================================")
    print("Running fabric tests...")
    print(f"Using hosts: {args.hosts}")
    print(f"Using docker image: {args.image}")
    print(f"Configuration: {args.config}")
    print(f"Output directory: {args.output}")
    print(f"Mesh graph descriptor: {mgd_path}")
    print(f"Test binary: {args.test_binary}")
    print(f"Test config: {args.test_config}")
    if args.filter:
        print(f"Filter: {args.filter}")
    print(f"Log file: {log_file}")
    print("==========================================")
    print()

    if args.config == "4x8":
        print(f"Running single-host 4x8 on: {hosts_csv}")
        print()

    # Phase 1: generate rank bindings inside Docker
    rank_bindings = run_phase1(args.image, hosts_csv, mgd_path)
    print()

    # Phase 2: launch test binary inside Docker with per-rank env vars
    print("Phase 2: Launching tests via mpi-docker...")
    cmd = build_phase2_cmd(args, mgd_path, rank_bindings)
    rc = run_with_tee(cmd, log_file)

    print()
    print("==========================================")
    print(f"Tests completed at {datetime.now():%a %b %d %H:%M:%S %Z %Y}")
    print(f"Results logged to: {log_file}")
    print("==========================================")

    sys.exit(rc)


if __name__ == "__main__":
    main()
