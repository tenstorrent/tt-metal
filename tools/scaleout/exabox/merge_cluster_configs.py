#!/usr/bin/env python3
"""
Merge two cluster configurations (cabling + deployment descriptors).

Usage:
    ./merge_cluster_configs.py \
        --cabling1 <path> \
        --cabling2 <path> \
        --deployment1 <path> \
        --deployment2 <path> \
        --output-dir <path> \
        [--temp-dir <path>]

Generates in output-dir:
    - merged_fsd.textproto
    - merged_cabling_descriptor.textproto
    - merged_deployment_descriptor.textproto
"""

import argparse
import os
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


def run_cabling_generator(cabling_dir, deployment_path, output_fsd, work_dir):
    script_dir = Path(__file__).parent
    repo_root = script_dir.parent.parent

    cabling_gen = repo_root / "build_Release" / "tools" / "scaleout" / "run_cabling_generator"
    if not cabling_gen.exists():
        print(f"Error: run_cabling_generator not found at {cabling_gen}", file=sys.stderr)
        print("Build with: ./build_metal.sh --build-tests", file=sys.stderr)
        sys.exit(1)

    cmd = [str(cabling_gen), "--cabling", cabling_dir, "--deployment", deployment_path, "--output", "merged"]

    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(repo_root))

    if result.returncode != 0:
        print(f"CablingGenerator failed", file=sys.stderr)
        if result.stderr:
            print(result.stderr, file=sys.stderr)
        sys.exit(1)

    generated_fsd = repo_root / "out" / "scaleout" / "factory_system_descriptor_merged.textproto"
    generated_cabling = repo_root / "out" / "scaleout" / "cabling_descriptor_merged.textproto"

    if not generated_fsd.exists():
        print(f"Error: Output file not found: {generated_fsd}", file=sys.stderr)
        sys.exit(1)

    shutil.copy(generated_fsd, output_fsd)

    if generated_cabling.exists():
        cabling_output = Path(str(output_fsd).replace("merged_fsd", "merged_cabling_descriptor"))
        shutil.copy(generated_cabling, cabling_output)
        return True
    return False


def count_nodes(deployment_path):
    with open(deployment_path, "r") as f:
        return f.read().count("hosts {")


def main():
    parser = argparse.ArgumentParser(description="Merge two cluster configurations.")
    parser.add_argument("--cabling1", required=True, help="First cabling_descriptor.textproto")
    parser.add_argument("--cabling2", required=True, help="Second cabling_descriptor.textproto")
    parser.add_argument("--deployment1", required=True, help="First deployment_descriptor.textproto")
    parser.add_argument("--deployment2", required=True, help="Second deployment_descriptor.textproto")
    parser.add_argument("--output-dir", "-o", required=True, help="Output directory")
    parser.add_argument("--temp-dir", default=None, help="Temporary directory")

    args = parser.parse_args()

    validate_files(
        {
            "cabling1": args.cabling1,
            "cabling2": args.cabling2,
            "deployment1": args.deployment1,
            "deployment2": args.deployment2,
        }
    )

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
        merge_deployment_descriptors(args.deployment1, args.deployment2, merged_deployment)

        merged_fsd = output_dir / "merged_fsd.textproto"
        has_cabling = run_cabling_generator(str(cabling_dir), str(merged_deployment), str(merged_fsd), temp_dir)

        final_deployment = output_dir / "merged_deployment_descriptor.textproto"
        shutil.copy(merged_deployment, final_deployment)

        total_nodes = count_nodes(str(merged_deployment))
        print(f"Merged {total_nodes} nodes")
        print(f"Output: {output_dir}")

    finally:
        if temp_created:
            shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
