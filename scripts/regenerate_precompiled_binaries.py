#!/usr/bin/env python3
"""
Regenerate pre-compiled kernel binaries for AOT compilation tests.

Copies kernel binaries from JIT cache to test directories for current architecture.

Tests using pre-compiled binaries:
- tests/tt_metal/tt_metal/api/test_kernel_creation.cpp::TestCreateKernelFromBinary
- tests/ttnn/unit_tests/gtests/test_generic_op.cpp::TestGenericOpMatmulFromBinary

Usage:
    python3 scripts/regenerate_precompiled_binaries.py
"""

import os
import sys
import shutil
from pathlib import Path


def get_repo_root():
    """Get the repository root directory."""
    return Path(__file__).resolve().parent.parent


def get_cache_dir():
    """Get the JIT cache directory."""
    cache_env = os.environ.get("TT_METAL_CACHE")
    if cache_env:
        return Path(cache_env)

    home = os.environ.get("HOME")
    if home:
        return Path(home) / ".cache" / "tt-metal-cache"

    return Path("/tmp") / "tt-metal-cache"


def find_latest_cache_build():
    """Find the latest cache build directory."""
    cache_root = get_cache_dir()
    if not cache_root.exists():
        print(f"Error: Cache directory {cache_root} does not exist", file=sys.stderr)
        return None

    # Find git hash directory (most recently modified)
    git_dirs = [d for d in cache_root.iterdir() if d.is_dir() and len(d.name) == 10]
    if not git_dirs:
        print(f"Error: No git hash directories found in {cache_root}", file=sys.stderr)
        return None

    git_dir = max(git_dirs, key=lambda d: d.stat().st_mtime)

    # Find build key directory (first one with kernels/)
    for build_dir in git_dir.iterdir():
        if build_dir.is_dir() and (build_dir / "kernels").exists():
            return build_dir / "kernels"

    print(f"Error: No build with kernels/ found in {git_dir}", file=sys.stderr)
    return None


def copy_kernel(cache_kernels_dir, test_kernels_dir, kernel_name):
    """Copy all hash variants of a kernel from cache to test directory."""
    src_kernel_dir = cache_kernels_dir / kernel_name
    if not src_kernel_dir.exists():
        print(f"  Warning: {kernel_name} not found in cache", file=sys.stderr)
        return False

    dst_kernel_dir = test_kernels_dir / kernel_name

    # Get all hash directories
    hash_dirs = [d for d in src_kernel_dir.iterdir() if d.is_dir()]
    if not hash_dirs:
        print(f"  Warning: No hashes found for {kernel_name}", file=sys.stderr)
        return False

    # Remove old kernel directory and copy fresh
    if dst_kernel_dir.exists():
        shutil.rmtree(dst_kernel_dir)

    shutil.copytree(src_kernel_dir, dst_kernel_dir)

    if len(hash_dirs) == 1:
        print(f"  ✓ {kernel_name}: {hash_dirs[0].name}")
    else:
        print(f"  ✓ {kernel_name}: {len(hash_dirs)} variants")

    return True


def main():
    repo_root = get_repo_root()

    print("Finding cache...")
    cache_kernels_dir = find_latest_cache_build()
    if not cache_kernels_dir:
        return 1

    print(f"Cache: {cache_kernels_dir.parent}")
    print()

    # Test configurations: (test_dir, [kernel_names])
    tests = [
        ("tests/tt_metal/tt_metal/api/simple_add_binaries", ["simple_add"]),
        ("tests/ttnn/unit_tests/matmul_binaries", [
            "reader_bmm_8bank_output_tiles_partitioned",
            "writer_unary_interleaved_start_id",
            "bmm",
        ]),
    ]

    success = True
    for test_base, kernels in tests:
        # Detect architecture from existing directory structure
        test_base_path = repo_root / test_base
        arch_dirs = [d.name for d in test_base_path.iterdir() if d.is_dir() and d.name in ["wormhole", "blackhole"]]

        if not arch_dirs:
            print(f"Warning: No architecture directories in {test_base}", file=sys.stderr)
            continue

        # Use the first architecture found (since we can only generate for current arch anyway)
        arch = arch_dirs[0]
        test_kernels_dir = test_base_path / arch / "kernels"

        print(f"{test_base.split('/')[-1]} ({arch}):")
        for kernel in kernels:
            if not copy_kernel(cache_kernels_dir, test_kernels_dir, kernel):
                success = False
        print()

    if success:
        print("✓ Done!")
        return 0
    else:
        print("⚠ Some kernels not found. Run tests with JIT first to populate cache.", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
