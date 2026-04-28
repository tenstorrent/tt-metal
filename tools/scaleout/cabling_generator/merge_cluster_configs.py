#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Merge two cluster configurations (cabling + deployment descriptors).

Usage:
    ./merge_cluster_configs.py \
        --cabling1 <path> \
        --cabling2 <path> \
        --deployment1 <path> \
        [--deployment2 <path>] \
        --output-dir <path> \
        [--temp-dir <path>] \
        [--build-dir <name>]

  deployment1 is required (first cluster: cabling + deployment).
  deployment2 is optional; if omitted, the merged deployment is deployment1 only
  (use when merging with a cabling-only second cluster).
  build-dir optionally specifies which build directory to use (e.g., build_Release,
  build_Debug). If not specified, searches common build directories automatically.

Generates in output-dir:
    - merged_fsd.textproto
    - merged_cabling_descriptor.textproto
    - merged_deployment_descriptor.textproto
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def validate_files(files):
    for name, path in files.items():
        if not os.path.exists(path):
            print(f"Error: {name} not found: {path}", file=sys.stderr)
            sys.exit(1)


def merge_deployment_descriptors(dep1_path, dep2_path, output_path):
    with open(output_path, "w") as outfile:
        with open(dep1_path, "r") as f1:
            outfile.write(f1.read())
        with open(dep2_path, "r") as f2:
            outfile.write(f2.read())


def find_cabling_generator(repo_root, build_dir=None):
    """Find run_cabling_generator binary, checking multiple build directories."""
    binary_subpath = Path("tools") / "scaleout" / "run_cabling_generator"

    if build_dir:
        # User-specified build directory
        candidate = repo_root / build_dir / binary_subpath
        if candidate.exists():
            return candidate
        print(f"Error: run_cabling_generator not found at {candidate}", file=sys.stderr)
        sys.exit(1)

    # Check common build directories in order of preference
    build_dirs = ["build_Release", "build_Debug", "build"]
    for bd in build_dirs:
        candidate = repo_root / bd / binary_subpath
        if candidate.exists():
            return candidate

    print("Error: run_cabling_generator not found in any build directory", file=sys.stderr)
    print(f"Searched: {', '.join(build_dirs)}", file=sys.stderr)
    print("Build with: ./build_metal.sh --build-tests", file=sys.stderr)
    sys.exit(1)


def run_cabling_generator(cabling_dir, deployment_path, output_fsd, build_dir=None):
    script_dir = Path(__file__).resolve().parent
    # Script lives at tools/scaleout/cabling_generator/merge_cluster_configs.py -> repo root is 3 levels up
    repo_root = script_dir.parent.parent.parent

    cabling_gen = find_cabling_generator(repo_root, build_dir)

    cmd = [str(cabling_gen), "--cabling", cabling_dir, "--deployment", deployment_path, "--output", "merged"]

    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(repo_root))

    if result.returncode != 0:
        print(f"CablingGenerator failed", file=sys.stderr)
        if result.stderr:
            print(result.stderr, file=sys.stderr)
        sys.exit(1)

    generated_fsd = repo_root / "out" / "scaleout" / "factory_system_descriptor_merged.textproto"
    generated_cabling = repo_root / "out" / "scaleout" / "cabling_descriptor_merged.textproto"
    generated_deployment = repo_root / "out" / "scaleout" / "deployment_descriptor_merged.textproto"

    if not generated_fsd.exists():
        print(f"Error: Output file not found: {generated_fsd}", file=sys.stderr)
        sys.exit(1)

    shutil.copy(generated_fsd, output_fsd)

    if generated_cabling.exists():
        cabling_output = Path(str(output_fsd).replace("merged_fsd", "merged_cabling_descriptor"))
        shutil.copy(generated_cabling, cabling_output)

    # Return path to C++-generated deployment descriptor (one host per node, no duplicates)
    # or None if not produced (fallback to concatenated temp file)
    return generated_deployment if generated_deployment.exists() else None


def count_nodes(deployment_path):
    """Count host entries in deployment descriptor using regex for robustness."""
    with open(deployment_path, "r") as f:
        content = f.read()
    # Match 'hosts {' at line start (with optional leading whitespace) to avoid matching in comments/strings
    pattern = r"^\s*hosts\s*\{"
    return len(re.findall(pattern, content, re.MULTILINE))


def validate_cluster_node_types(cabling_path, deployment_path):
    """
    Validate that:
    1. All hosts within each cluster have the same node_type (deployment descriptor)
    2. Each host's node_descriptor (cabling) matches its node_type (deployment)
    Uses cabling_descriptor_analysis.py to identify clusters.
    """
    script_dir = Path(__file__).resolve().parent
    analysis_script = script_dir / "cabling_descriptor_analysis.py"

    if not analysis_script.exists():
        print(f"Warning: Cluster analysis script not found at {analysis_script}, skipping validation", file=sys.stderr)
        return

    # Run cluster analysis to get cluster information
    try:
        result = subprocess.run(
            ["python3", str(analysis_script), str(cabling_path), "--json"], capture_output=True, text=True, check=True
        )
        cluster_data = json.loads(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"Warning: Failed to run cluster analysis: {e}", file=sys.stderr)
        return
    except json.JSONDecodeError as e:
        print(f"Warning: Failed to parse cluster analysis JSON: {e}", file=sys.stderr)
        return

    # Parse deployment descriptor to build hostname -> node_type mapping
    hostname_to_node_type = {}
    with open(deployment_path, "r") as f:
        content = f.read()

    # Extract host blocks using regex
    host_pattern = r"hosts\s*\{([^}]+)\}"
    for host_match in re.finditer(host_pattern, content):
        host_block = host_match.group(1)

        # Extract hostname and node_type from block
        hostname_match = re.search(r'host:\s*"([^"]+)"', host_block)
        node_type_match = re.search(r'node_type:\s*"([^"]+)"', host_block)

        if hostname_match and node_type_match:
            hostname_to_node_type[hostname_match.group(1)] = node_type_match.group(1)

    # Parse cabling descriptor to build hostname -> node_descriptor mapping
    hostname_to_node_descriptor = {}
    with open(cabling_path, "r") as f:
        content = f.read()

    # Extract children blocks with name and node_descriptor
    children_pattern = r"children\s*\{([^}]+)\}"
    for child_match in re.finditer(children_pattern, content):
        child_block = child_match.group(1)

        # Extract name and node_descriptor from block
        name_match = re.search(r'name:\s*"([^"]+)"', child_block)
        descriptor_match = re.search(r'node_descriptor:\s*"([^"]+)"', child_block)

        if name_match and descriptor_match:
            hostname_to_node_descriptor[name_match.group(1)] = descriptor_match.group(1)

    # Validate: 1) cabling vs deployment consistency, 2) cluster homogeneity
    print(f"\nValidating node type consistency...")
    errors = []

    # Check 1: Cabling descriptor matches deployment descriptor for each host
    print("  Checking cabling descriptor vs deployment descriptor...")
    cabling_deployment_mismatches = []
    for hostname, node_descriptor in hostname_to_node_descriptor.items():
        if hostname in hostname_to_node_type:
            node_type = hostname_to_node_type[hostname]
            if node_descriptor != node_type:
                cabling_deployment_mismatches.append(
                    f"    {hostname}: cabling has '{node_descriptor}' but deployment has '{node_type}'"
                )

    if cabling_deployment_mismatches:
        errors.append("\n  Cabling/Deployment mismatch errors:")
        errors.extend(cabling_deployment_mismatches)
    else:
        print("    All hosts match between cabling and deployment ✓")

    # Check 2: All hosts within each cluster have the same node type
    print(f"  Checking consistency across {cluster_data['total_clusters']} clusters...")
    for cluster in cluster_data["clusters"]:
        cluster_id = cluster["cluster_id"]
        hostnames = cluster["hostnames"]

        # Get node types for all hosts in this cluster
        node_types_in_cluster = {}
        for hostname in hostnames:
            if hostname in hostname_to_node_type:
                node_type = hostname_to_node_type[hostname]
                if node_type not in node_types_in_cluster:
                    node_types_in_cluster[node_type] = []
                node_types_in_cluster[node_type].append(hostname)

        # Check if all hosts have the same node type
        if len(node_types_in_cluster) > 1:
            error_msg = f"\n  Cluster {cluster_id} has mixed node types:"
            for node_type, hosts in sorted(node_types_in_cluster.items()):
                error_msg += f"\n    {node_type}: {len(hosts)} hosts ({', '.join(hosts[:3])}"
                if len(hosts) > 3:
                    error_msg += f", ... and {len(hosts) - 3} more"
                error_msg += ")"
            errors.append(error_msg)
        else:
            node_type = list(node_types_in_cluster.keys())[0] if node_types_in_cluster else "unknown"
            print(f"    Cluster {cluster_id}: {cluster['num_hosts']} hosts, all {node_type} ✓")

    if errors:
        print("\n❌ Validation failed with errors:", file=sys.stderr)
        for error in errors:
            print(error, file=sys.stderr)
        sys.exit(1)

    print("\n✅ All validation checks passed\n")


def main():
    parser = argparse.ArgumentParser(description="Merge two cluster configurations.")
    parser.add_argument("--cabling1", required=True, help="First cabling_descriptor.textproto")
    parser.add_argument("--cabling2", required=True, help="Second cabling_descriptor.textproto")
    parser.add_argument("--deployment1", required=True, help="First deployment_descriptor.textproto (required)")
    parser.add_argument(
        "--deployment2",
        default=None,
        help="Second deployment_descriptor.textproto (optional; if omitted, merged deployment is deployment1 only)",
    )
    parser.add_argument("--output-dir", "-o", required=True, help="Output directory")
    parser.add_argument("--temp-dir", default=None, help="Temporary directory")
    parser.add_argument(
        "--build-dir",
        default=None,
        help="Build directory name (e.g., build_Release, build_Debug). Auto-detected if not specified.",
    )

    args = parser.parse_args()

    files_to_validate = {"cabling1": args.cabling1, "cabling2": args.cabling2, "deployment1": args.deployment1}
    if args.deployment2 is not None:
        files_to_validate["deployment2"] = args.deployment2
    validate_files(files_to_validate)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.temp_dir:
        temp_dir = Path(args.temp_dir)
        temp_dir.mkdir(parents=True, exist_ok=True)
        temp_created = False
    else:
        temp_dir = Path(tempfile.mkdtemp(prefix="cluster_merge_"))
        temp_created = True

    try:
        cabling_dir = temp_dir / "cabling"
        cabling_dir.mkdir(exist_ok=True)

        shutil.copy(args.cabling1, cabling_dir / "cluster1_cabling.textproto")
        shutil.copy(args.cabling2, cabling_dir / "cluster2_cabling.textproto")

        merged_deployment = temp_dir / "merged_deployment_descriptor.textproto"
        if args.deployment2 is not None:
            merge_deployment_descriptors(args.deployment1, args.deployment2, merged_deployment)
        else:
            shutil.copy(args.deployment1, merged_deployment)

        # Write merged files to temp directory first
        merged_fsd_temp = temp_dir / "merged_fsd.textproto"
        generated_deployment_path = run_cabling_generator(
            str(cabling_dir), str(merged_deployment), str(merged_fsd_temp), args.build_dir
        )

        final_deployment_temp = temp_dir / "final_deployment_descriptor.textproto"
        if generated_deployment_path is not None:
            # Use C++-generated deployment (one host per node in host_id order, no duplicates)
            shutil.copy(generated_deployment_path, final_deployment_temp)
        else:
            shutil.copy(merged_deployment, final_deployment_temp)

        # Validate node type consistency within each cluster before writing to output
        merged_cabling_temp = temp_dir / "merged_cabling_descriptor.textproto"
        if merged_cabling_temp.exists():
            validate_cluster_node_types(merged_cabling_temp, final_deployment_temp)

        # Validation passed - now copy files to output directory
        shutil.copy(merged_fsd_temp, output_dir / "merged_fsd.textproto")
        shutil.copy(final_deployment_temp, output_dir / "merged_deployment_descriptor.textproto")
        if merged_cabling_temp.exists():
            shutil.copy(merged_cabling_temp, output_dir / "merged_cabling_descriptor.textproto")

        total_nodes = count_nodes(str(final_deployment_temp))
        print(f"\n✅ Merge successful!")
        print(f"Merged {total_nodes} nodes")
        print(f"Output: {output_dir}")

    finally:
        if temp_created:
            shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
