#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Remove one or more hosts from a cabling + deployment descriptor pair.

The host's node, any internal connections touching it, and any subgraph left empty
are dropped. Remaining host_ids are renumbered to stay contiguous 0..N-1. Unknown
hostnames produce a warning and no mutation.

Usage:
    ./remove_hosts.py \
        --cabling <path-or-dir> \
        --deployment <path> \
        --remove <hostname> [--remove <hostname> ...] \
        --output-dir <path> \
        [--build-dir <name>]

Generates in output-dir:
    - updated_fsd.textproto
    - updated_cabling_descriptor.textproto
    - updated_deployment_descriptor.textproto
"""

import argparse
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path


def validate_paths(paths):
    for name, path in paths.items():
        if not os.path.exists(path):
            print(f"Error: {name} not found: {path}", file=sys.stderr)
            sys.exit(1)


def find_cabling_generator(repo_root, build_dir=None):
    """Find run_cabling_generator binary, checking multiple build directories."""
    binary_subpath = Path("tools") / "scaleout" / "run_cabling_generator"

    if build_dir:
        candidate = repo_root / build_dir / binary_subpath
        if candidate.exists():
            return candidate
        print(f"Error: run_cabling_generator not found at {candidate}", file=sys.stderr)
        sys.exit(1)

    for bd in ("build_Release", "build_Debug", "build"):
        candidate = repo_root / bd / binary_subpath
        if candidate.exists():
            return candidate

    print("Error: run_cabling_generator not found in any build directory", file=sys.stderr)
    print("Searched: build_Release, build_Debug, build", file=sys.stderr)
    print("Build with: ./build_metal.sh --build-tests", file=sys.stderr)
    sys.exit(1)


def count_hosts(deployment_path):
    """Count host entries in a deployment descriptor (regex-based for robustness)."""
    with open(deployment_path, "r") as f:
        content = f.read()
    return len(re.findall(r"^\s*hosts\s*\{", content, re.MULTILINE))


def main():
    parser = argparse.ArgumentParser(
        description="Remove hosts from a cabling + deployment descriptor pair.",
    )
    parser.add_argument(
        "--cabling",
        required=True,
        help="Cabling descriptor .textproto file, or a directory of them to be merged first",
    )
    parser.add_argument("--deployment", required=True, help="Deployment descriptor .textproto file")
    parser.add_argument(
        "--remove",
        action="append",
        required=True,
        metavar="HOSTNAME",
        help="Hostname to remove; pass multiple times to remove several hosts",
    )
    parser.add_argument("--output-dir", "-o", required=True, help="Output directory")
    parser.add_argument(
        "--build-dir",
        default=None,
        help="Build directory name (e.g. build_Release). Auto-detected if not specified.",
    )

    args = parser.parse_args()

    validate_paths({"cabling": args.cabling, "deployment": args.deployment})

    # Drop accidental empty --remove values; preserve user-given order otherwise.
    remove_list = [h.strip() for h in args.remove if h and h.strip()]
    if not remove_list:
        print("Error: --remove must specify at least one non-empty hostname", file=sys.stderr)
        sys.exit(1)

    script_dir = Path(__file__).resolve().parent
    # tools/scaleout/cabling_generator/remove_hosts.py -> repo root is 3 levels up
    repo_root = script_dir.parent.parent.parent
    cabling_gen = find_cabling_generator(repo_root, args.build_dir)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # The C++ tool runs with cwd=repo_root, so convert input paths to absolute to ensure
    # they resolve regardless of the user's working directory.
    cabling_path = str(Path(args.cabling).resolve())
    deployment_path = str(Path(args.deployment).resolve())

    cmd = [
        str(cabling_gen),
        "--cabling", cabling_path,
        "--deployment", deployment_path,
        "--remove-hosts", ",".join(remove_list),
        "--output", "updated",
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(repo_root))
    if result.stdout:
        sys.stdout.write(result.stdout)
    # Surface warnings (e.g. unknown-hostname no-ops) regardless of exit code.
    if result.stderr:
        sys.stderr.write(result.stderr)
    if result.returncode != 0:
        print("run_cabling_generator failed", file=sys.stderr)
        sys.exit(1)

    generated_fsd = repo_root / "out" / "scaleout" / "factory_system_descriptor_updated.textproto"
    generated_cabling = repo_root / "out" / "scaleout" / "cabling_descriptor_updated.textproto"
    generated_deployment = repo_root / "out" / "scaleout" / "deployment_descriptor_updated.textproto"

    if not generated_fsd.exists():
        print(f"Error: Output file not found: {generated_fsd}", file=sys.stderr)
        sys.exit(1)

    shutil.copy(generated_fsd, output_dir / "updated_fsd.textproto")
    if generated_cabling.exists():
        shutil.copy(generated_cabling, output_dir / "updated_cabling_descriptor.textproto")
    if generated_deployment.exists():
        shutil.copy(generated_deployment, output_dir / "updated_deployment_descriptor.textproto")

    final_deployment = output_dir / "updated_deployment_descriptor.textproto"
    if final_deployment.exists():
        print(f"Requested removal of {len(remove_list)} host(s); {count_hosts(str(final_deployment))} remain")
    print(f"Output: {output_dir}")


if __name__ == "__main__":
    main()
