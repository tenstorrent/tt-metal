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


# RFC 1123 host/label characters only. Anchored, no path separators, no leading
# dash. This is an allowlist: any input that does not match is rejected outright,
# which prevents both argument injection (e.g. a value starting with "-") and any
# attempt to smuggle shell/path metacharacters into the downstream binary call.
HOSTNAME_RE = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9._-]{0,253}[A-Za-z0-9])?$")


def validate_hostname(hostname):
    """Reject any hostname that is not a plain RFC 1123-style name."""
    if not HOSTNAME_RE.match(hostname):
        print(f"Error: invalid hostname: {hostname!r}", file=sys.stderr)
        sys.exit(1)
    return hostname


def validate_build_dir(build_dir):
    """Constrain --build-dir to a single, simple directory name.

    Rejecting path separators and traversal keeps the binary-discovery path
    rooted under repo_root and prevents user input from escaping it.
    """
    if (
        build_dir in (".", "..")
        or "/" in build_dir
        or "\\" in build_dir
        or os.sep in build_dir
        or (os.altsep and os.altsep in build_dir)
    ):
        print(f"Error: invalid --build-dir (must be a simple directory name): {build_dir!r}", file=sys.stderr)
        sys.exit(1)
    return build_dir


def resolve_input_file(name, raw_path):
    """Resolve a user-supplied input path to an absolute, existing regular file.

    Resolving normalizes any '..' traversal to a concrete absolute path, and the
    absolute form guarantees the value cannot be misread as a CLI flag by the
    downstream binary (it always begins with the filesystem root).
    """
    resolved = Path(raw_path).resolve()
    if not resolved.is_file():
        print(f"Error: {name} not found or not a regular file: {raw_path}", file=sys.stderr)
        sys.exit(1)
    return str(resolved)


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

    # Drop accidental empty --remove values; preserve user-given order otherwise.
    # Every retained hostname is validated against a strict allowlist before it is
    # ever placed on the downstream command line.
    remove_list = [validate_hostname(h.strip()) for h in args.remove if h and h.strip()]
    if not remove_list:
        print("Error: --remove must specify at least one non-empty hostname", file=sys.stderr)
        sys.exit(1)

    build_dir = validate_build_dir(args.build_dir) if args.build_dir else None

    script_dir = Path(__file__).resolve().parent
    # tools/scaleout/cabling_generator/remove_hosts.py -> repo root is 3 levels up
    repo_root = script_dir.parent.parent.parent
    cabling_gen = find_cabling_generator(repo_root, build_dir)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # The C++ tool runs with cwd=repo_root, so convert input paths to absolute to ensure
    # they resolve regardless of the user's working directory. resolve_input_file also
    # normalizes traversal and verifies the target is an existing regular file.
    cabling_path = resolve_input_file("cabling", args.cabling)
    deployment_path = resolve_input_file("deployment", args.deployment)

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
