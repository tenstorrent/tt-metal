#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import argparse
import os
import re
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path

# --------------------------
# Defaults (customize here)
# --------------------------
DEFAULT_METAL_HOME = os.environ.get("TT_METAL_HOME", "~/git/tt-metal")
DEFAULT_CONFIG = "training_shakespeare_nanogpt_3tier_mpi.yaml"
SSH_USER = "ttuser"
BINARIES = ("nano_gpt", "nano_gpt_aggregator", "nano_gpt_optimizer")
SCP_OPTS = ["-p"]  # preserve times & modes

HOSTS = [
    "metal-wh-01",
    "metal-wh-05",
    "metal-wh-03",
    "metal-wh-04",
    "metal-wh-06",
]

# Default MESH_IDS per global rank; falls back to rank id if list is shorter than TOTAL_RANKS
DEFAULT_MESH_IDS = [0, 0, 0, 0, 0]
# If config contains "socket_type: fabric", override MESH_IDS with this:
FABRIC_MESH_IDS = [4, 1, 3, 2, 0]
MESH_GRAPH_DESC_REL = "tests/tt_metal/tt_fabric/custom_mesh_descriptors/new_nano_exabox_1x8_mesh_graph_descriptor.yaml"


# --------------------------
# Helpers
# --------------------------
def die(msg: str, code: int = 1):
    print(f"ERROR: {msg}", file=sys.stderr)
    sys.exit(code)


def run(cmd, dry=False, check=True, capture=False):
    printable = " ".join(shlex.quote(c) for c in cmd)
    if dry:
        print(f"DRY RUN: {printable}")
        return ""
    if capture:
        return subprocess.run(cmd, check=check, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True).stdout
    else:
        return subprocess.run(cmd, check=check)


def verify_local_files(metal_home: Path, config_name: str, mesh_desc_rel_path: str = None):
    bin_dir = metal_home / "tt-train" / "build" / "sources" / "examples" / "nano_gpt"
    cfg_dir = metal_home / "tt-train" / "configs"

    print("Verifying local files...")
    missing = 0

    for b in BINARIES:
        p = bin_dir / b
        if not p.exists():
            print(f"ERROR: Binary not found: {p}", file=sys.stderr)
            missing += 1
        elif not os.access(p, os.X_OK):
            print(f"ERROR: Binary not executable: {p}", file=sys.stderr)
            missing += 1
        else:
            print(f"✓ Found: {p}")

    cfg = cfg_dir / config_name
    if not cfg.exists():
        print(f"ERROR: Config file not found: {cfg}", file=sys.stderr)
        missing += 1
    else:
        print(f"✓ Found: {cfg}")

    # Check mesh descriptor if provided
    if mesh_desc_rel_path:
        mesh_desc = metal_home / mesh_desc_rel_path
        if not mesh_desc.exists():
            print(f"ERROR: Mesh descriptor not found: {mesh_desc}", file=sys.stderr)
            missing += 1
        else:
            print(f"✓ Found: {mesh_desc}")

    if missing:
        print(f"ERROR: {missing} required files are missing or invalid", file=sys.stderr)
        print(f"Build directory: {bin_dir}", file=sys.stderr)
        print(f"Config directory: {cfg_dir}", file=sys.stderr)
        print("\nTo build the missing binaries, try:", file=sys.stderr)
        print(f"  cd {metal_home}", file=sys.stderr)
        print("  make -C tt-train build", file=sys.stderr)
        print("  # or", file=sys.stderr)
        print("  cd tt-train && mkdir -p build && cd build", file=sys.stderr)
        print("  cmake .. && make -j$(nproc)", file=sys.stderr)
        sys.exit(1)

    return bin_dir, cfg_dir


def detect_fabric(cfg_path: Path) -> bool:
    try:
        text = cfg_path.read_text()
    except Exception:
        return False
    return re.search(r"\bsocket_type:\s*fabric\b", text) is not None


def copy_to_remote_hosts(
    hosts, ssh_user, bin_dir: Path, cfg_dir: Path, config_name: str, mesh_desc_path: Path, use_fabric: bool, dry: bool
):
    # de-duplicate
    unique_hosts = list(dict.fromkeys(hosts))
    if len(unique_hosts) == 1:
        print(f"All hosts are the same ({unique_hosts[0]}) - skipping remote copy")
        return

    print("Copying binaries, config, and mesh descriptor to remote hosts...")
    for host in unique_hosts[1:]:  # skip index 0 (often local)
        print(f" -> {host}")

        # SSH connectivity test
        try:
            run(
                [
                    "ssh",
                    "-o",
                    "ConnectTimeout=10",
                    "-o",
                    "BatchMode=yes",
                    f"{ssh_user}@{host}",
                    "echo",
                    "SSH test successful",
                ],
                dry=dry,
                check=True,
            )
        except subprocess.CalledProcessError as e:
            print(f"   ERROR: SSH connection failed to {host}", file=sys.stderr)
            print("   Check: 1) Host reachable, 2) SSH keys set up, 3) User exists", file=sys.stderr)
            continue

        # mkdirs
        mesh_desc_dir = mesh_desc_path.parent
        run(
            ["ssh", f"{ssh_user}@{host}", "mkdir", "-p", str(bin_dir), str(cfg_dir), str(mesh_desc_dir)],
            dry=dry,
            check=True,
        )

        # copy binaries
        for b in BINARIES:
            src = bin_dir / b
            dest = f"{ssh_user}@{host}:{bin_dir}/"
            try:
                run(["scp", *SCP_OPTS, str(src), dest], dry=dry, check=True)
                print(f"   ✓ {b} copied successfully")
            except subprocess.CalledProcessError as e:
                print(f"   ERROR: Failed to copy {b} to {host}", file=sys.stderr)

        # copy config
        run(["scp", *SCP_OPTS, str(cfg_dir / config_name), f"{ssh_user}@{host}:{cfg_dir}/"], dry=dry, check=True)

        # copy mesh descriptor if using fabric
        if use_fabric:
            try:
                run(["scp", *SCP_OPTS, str(mesh_desc_path), f"{ssh_user}@{host}:{mesh_desc_path}"], dry=dry, check=True)
                print(f"   ✓ mesh descriptor copied successfully")
            except subprocess.CalledProcessError as e:
                print(f"   ERROR: Failed to copy mesh descriptor to {host}", file=sys.stderr)

    print("✔ Remote copy step done.")


def build_mapping(hosts, mesh_ids, worker_count, agg_count, opt_count):
    total = worker_count + agg_count + opt_count
    mapping = []
    for rank in range(total):
        host = hosts[rank % len(hosts)]
        # Assign which binary this rank runs
        if rank < worker_count:
            binary_idx = 0  # nano_gpt
        elif rank < worker_count + agg_count:
            binary_idx = 1  # nano_gpt_aggregator
        else:
            binary_idx = 2  # nano_gpt_optimizer

        tt_mesh_id = mesh_ids[rank] if rank < len(mesh_ids) else rank
        mapping.append(
            {
                "rank": rank,
                "host": host,
                "binary_idx": binary_idx,
                "tt_mesh_id": tt_mesh_id,
            }
        )
    return mapping


def write_hostfile(hosts, fpath: Path):
    with fpath.open("w") as f:
        for h in hosts:
            f.write(f"{h} slots=1\n")


def write_appfile(
    mapping,
    metal_home: Path,
    cfg_dir: Path,
    config_name: str,
    bin_dir: Path,
    use_fabric: bool,
    mesh_graph_desc_path: Path,
    extra_args: list[str],
    fpath: Path,
):
    """
    Writes an Open MPI appfile with one app context per rank.
    We pass per-rank env using '-x KEY=VALUE' (works across OMPI versions).
    """
    common_env = {
        "TT_METAL_HOME": str(metal_home),
        "TT_LOGGER_LEVEL": "DEBUG",
        "TT_HOST_RANK": "0",
    }
    if use_fabric:
        common_env["TT_MESH_GRAPH_DESC_PATH"] = str(mesh_graph_desc_path)

    lines = []
    for m in mapping:
        exe = str(bin_dir / BINARIES[m["binary_idx"]])
        args = [exe, "-c", str(cfg_dir / config_name)]
        if extra_args:
            args += extra_args

        # Build -x KEY=VALUE flags (common + per-rank)
        xflags = []
        for k, v in common_env.items():
            xflags += ["-x", f"{k}={v}"]
        xflags += ["-x", f"TT_MESH_ID={m['tt_mesh_id']}"]

        # One app context per rank
        line = " ".join(["-np", "1", "-host", m["host"], *xflags] + [shlex.quote(a) for a in args])
        lines.append(line)

    fpath.write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Copy nano_gpt 3-tier binaries and config to remote hosts and launch via MPI, "
        "precomputing per-rank TT_MESH_ID into an Open MPI appfile."
    )
    parser.add_argument("-m", "--metal-home", default=DEFAULT_METAL_HOME, help="TT_METAL_HOME")
    parser.add_argument("-c", "--config", default=DEFAULT_CONFIG, help="Config filename")
    parser.add_argument("-n", "--dry-run", action="store_true", help="Dry run (print commands only)")
    parser.add_argument("--hosts", nargs="*", default=HOSTS, help="Host list (MPI rank order)")
    parser.add_argument("--mesh-ids", nargs="*", type=int, default=DEFAULT_MESH_IDS, help="MESH_IDs per global rank")
    parser.add_argument("--ssh-user", default=SSH_USER, help="SSH username for remote copy")
    parser.add_argument("--skip-copy", action="store_true", help="Skip remote copy step")
    parser.add_argument("--workers", type=int, default=None, help="Number of worker ranks (default: len(hosts)-2)")
    parser.add_argument("--aggregators", type=int, default=1, help="Number of aggregator ranks")
    parser.add_argument("--optimizers", type=int, default=1, help="Number of optimizer ranks")
    parser.add_argument(
        "remainder", nargs=argparse.REMAINDER, help="Use '-- <args>' to forward extra args to all ranks"
    )

    args = parser.parse_args()

    # Extract extra args after "--"
    extra_args = []
    if args.remainder:
        # argparse keeps the "--" as first token if present
        extra_args = args.remainder[1:] if args.remainder and args.remainder[0] == "--" else args.remainder
        if extra_args:
            print("Additional arguments to forward to workers:", " ".join(extra_args))
        else:
            print("No additional arguments to forward to workers")
    else:
        print("No additional arguments to forward to workers")

    metal_home = Path(args.metal_home).resolve()

    # Need to detect fabric first to know if mesh descriptor is required
    cfg_path = metal_home / "tt-train" / "configs" / args.config
    use_fabric = detect_fabric(cfg_path)

    # Verify files including mesh descriptor if using fabric
    mesh_desc_rel_path = MESH_GRAPH_DESC_REL if use_fabric else None
    bin_dir, cfg_dir = verify_local_files(metal_home, args.config, mesh_desc_rel_path)
    cfg_path = cfg_dir / args.config

    num_hosts = len(args.hosts)
    if num_hosts < 3:
        die("need at least 3 hosts (2 workers + 1 aggregator + 1 optimizer)")

    agg_count = args.aggregators
    opt_count = args.optimizers
    if args.workers is None:
        worker_count = num_hosts - agg_count - opt_count
    else:
        worker_count = args.workers

    total_ranks = worker_count + agg_count + opt_count
    if total_ranks <= 0:
        die("total ranks computed to 0; check workers/aggregators/optimizers/hosts")

    # Set mesh IDs based on fabric configuration
    if use_fabric:
        print("Fabric configuration detected - using mesh graph descriptor")
        mesh_ids = FABRIC_MESH_IDS
    else:
        print("Standard configuration detected - fabric disabled")
        mesh_ids = args.mesh_ids

    mesh_graph_desc_path = metal_home / MESH_GRAPH_DESC_REL

    # Informative mapping print
    print("Planned mapping:")
    for i in range(total_ranks):
        mid = mesh_ids[i] if i < len(mesh_ids) else i
        print(f"  rank {i} -> TT_MESH_ID={mid}")

    # Remote copy step
    if not args.skip_copy:
        copy_to_remote_hosts(
            args.hosts, args.ssh_user, bin_dir, cfg_dir, args.config, mesh_graph_desc_path, use_fabric, args.dry_run
        )
    else:
        print("Skipping remote copy step (--skip-copy)")

    # Validate mpirun presence
    try:
        run(["mpirun", "--version"], dry=False, check=True, capture=True)
    except Exception:
        die("mpirun command not found in PATH")

    # Build per-rank mapping and write temp hostfile + appfile
    mapping = build_mapping(args.hosts, mesh_ids, worker_count, agg_count, opt_count)

    with tempfile.TemporaryDirectory() as td:
        hostfile = Path(td) / "mpi_hosts"
        appfile = Path(td) / "mpi_appfile"

        write_hostfile(args.hosts, hostfile)
        write_appfile(
            mapping, metal_home, cfg_dir, args.config, bin_dir, use_fabric, mesh_graph_desc_path, extra_args, appfile
        )

        # print content of app file
        with open(appfile, "r") as f:
            print("Content of app file:")
            print(f.read())

        mpi_cmd = ["mpirun", "--hostfile", str(hostfile), "--app", str(appfile)]

        print(
            "Launching MPI 3-tier demo with "
            f"{worker_count} workers, {agg_count} aggregators, {opt_count} optimizers..."
        )
        print(f"Environment: USE_FABRIC={use_fabric}")
        print("=== MPI Command Being Executed ===")
        print(" ".join(shlex.quote(c) for c in mpi_cmd))
        print("==================================")

        if args.dry_run:
            print("DRY RUN: not executing mpirun")
            return

        try:
            run(mpi_cmd, dry=False, check=True)
        except subprocess.CalledProcessError:
            die("MPI job failed!")

    print("✔ MPI job finished successfully.")


if __name__ == "__main__":
    main()
