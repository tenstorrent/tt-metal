#!/usr/bin/env python3
"""
For each include directory, list device-side source files that depend on it.

A source file "depends on" an include directory if at least one of its
#include directives is resolved by that directory.

Outputs JSON to stdout.

Usage:
    python3 include_scripts/find_device_side_includes.py > data.json
    python3 include_scripts/map_sources_to_include_dirs.py data.json > sources_to_dirs.json
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path


# System headers that don't need resolution
SYSTEM_HEADERS = {
    "algorithm", "array", "cmath", "cstddef", "cstdint", "cstdlib",
    "cstring", "functional", "limits", "stdint.h", "type_traits",
    "utility", "gtest/gtest.h", "sys/types.h",
    # Additional C++ standard library headers
    "filesystem", "fstream", "iostream", "limits.h", "numeric",
    "ostream", "regex", "stdexcept", "string", "thread", "tuple", "vector"
}


def get_include_dirs(root: str) -> list[tuple[str, str]]:
    """
    Return the include directories from build.cpp:264-275 plus HAL dynamic includes.
    Returns list of (display_name, full_path) tuples.
    """
    dirs = [
        # Static includes from build.cpp:264-275
        ("(root)", root),
        ("ttnn", os.path.join(root, "ttnn")),
        ("ttnn/cpp", os.path.join(root, "ttnn/cpp")),
        ("tt_metal", os.path.join(root, "tt_metal")),
        ("tt_metal/include", os.path.join(root, "tt_metal/include")),
        ("tt_metal/hw/inc", os.path.join(root, "tt_metal/hw/inc")),
        ("tt_metal/hostdevcommon/api", os.path.join(root, "tt_metal/hostdevcommon/api")),
        ("tt_metal/api/", os.path.join(root, "tt_metal/api")),

        # HAL dynamic includes (from HalJitBuildQueryInterface::includes)
        # Architecture-specific: blackhole
        ("tt_metal/hw/ckernels/blackhole/metal/common", os.path.join(root, "tt_metal/hw/ckernels/blackhole/metal/common")),
        ("tt_metal/hw/ckernels/blackhole/metal/llk_io", os.path.join(root, "tt_metal/hw/ckernels/blackhole/metal/llk_io")),
        ("tt_metal/hw/ckernels/blackhole/metal/llk_api", os.path.join(root, "tt_metal/hw/ckernels/blackhole/metal/llk_api")),
        ("tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu", os.path.join(root, "tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu")),
        ("tt_metal/hw/inc/internal/tt-1xx", os.path.join(root, "tt_metal/hw/inc/internal/tt-1xx")),
        ("tt_metal/hw/inc/internal/tt-1xx/blackhole", os.path.join(root, "tt_metal/hw/inc/internal/tt-1xx/blackhole")),
        ("tt_metal/hw/inc/internal/tt-1xx/blackhole/blackhole_defines", os.path.join(root, "tt_metal/hw/inc/internal/tt-1xx/blackhole/blackhole_defines")),
        ("tt_metal/hw/inc/internal/tt-1xx/blackhole/noc", os.path.join(root, "tt_metal/hw/inc/internal/tt-1xx/blackhole/noc")),
        ("tt_metal/third_party/tt_llk/tt_llk_blackhole/common/inc", os.path.join(root, "tt_metal/third_party/tt_llk/tt_llk_blackhole/common/inc")),
        ("tt_metal/third_party/tt_llk/tt_llk_blackhole/llk_lib", os.path.join(root, "tt_metal/third_party/tt_llk/tt_llk_blackhole/llk_lib")),

        # Architecture-specific: wormhole_b0
        ("tt_metal/hw/ckernels/wormhole_b0/metal/common", os.path.join(root, "tt_metal/hw/ckernels/wormhole_b0/metal/common")),
        ("tt_metal/hw/ckernels/wormhole_b0/metal/llk_io", os.path.join(root, "tt_metal/hw/ckernels/wormhole_b0/metal/llk_io")),
        ("tt_metal/hw/ckernels/wormhole_b0/metal/llk_api", os.path.join(root, "tt_metal/hw/ckernels/wormhole_b0/metal/llk_api")),
        ("tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu", os.path.join(root, "tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu")),
        ("tt_metal/hw/inc/internal/tt-1xx/wormhole", os.path.join(root, "tt_metal/hw/inc/internal/tt-1xx/wormhole")),
        ("tt_metal/hw/inc/internal/tt-1xx/wormhole/wormhole_b0_defines", os.path.join(root, "tt_metal/hw/inc/internal/tt-1xx/wormhole/wormhole_b0_defines")),
        ("tt_metal/hw/inc/internal/tt-1xx/wormhole/noc", os.path.join(root, "tt_metal/hw/inc/internal/tt-1xx/wormhole/noc")),
        ("tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc", os.path.join(root, "tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc")),
        ("tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/llk_lib", os.path.join(root, "tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/llk_lib")),

        # Architecture-specific: quasar
        ("tt_metal/hw/inc/internal/tt-2xx", os.path.join(root, "tt_metal/hw/inc/internal/tt-2xx")),
        ("tt_metal/hw/inc/internal/tt-2xx/quasar", os.path.join(root, "tt_metal/hw/inc/internal/tt-2xx/quasar")),
        ("tt_metal/hw/inc/internal/tt-2xx/quasar/quasar_defines", os.path.join(root, "tt_metal/hw/inc/internal/tt-2xx/quasar/quasar_defines")),
        ("tt_metal/hw/inc/internal/tt-2xx/quasar/noc", os.path.join(root, "tt_metal/hw/inc/internal/tt-2xx/quasar/noc")),

        # Common HAL includes
        ("tt_metal/hw/firmware/src/tt-1xx", os.path.join(root, "tt_metal/hw/firmware/src/tt-1xx")),
        ("tt_metal/hw/inc/ethernet", os.path.join(root, "tt_metal/hw/inc/ethernet")),

        # SFPI compiler includes (compute kernels only)
        ("runtime/sfpi/include", os.path.join(root, "runtime/sfpi/include")),
    ]
    return dirs


def resolve_header(header: str, source_path: str, root: str, include_dirs: list[tuple[str, str]]) -> str | None:
    """
    Resolve a header to the include directory that provides it.
    Returns the directory name, or None if unresolved.
    Checks: static dirs first, then . (source dir), then .. (parent dir).
    """
    if header in SYSTEM_HEADERS:
        return None  # System header, skip

    if header.startswith(".."):
        return None  # Relative path, skip

    # Try static directories
    for dir_name, dir_path in include_dirs:
        full_path = os.path.join(dir_path, header)
        if os.path.isfile(full_path):
            return dir_name

    # Try . (same directory as source)
    source_dir = os.path.dirname(os.path.join(root, source_path))
    dot_path = os.path.join(source_dir, header)
    if os.path.isfile(dot_path):
        return "."

    # Try .. (parent directory)
    parent_dir = os.path.dirname(source_dir)
    dotdot_path = os.path.join(parent_dir, header)
    if os.path.isfile(dotdot_path):
        return ".."

    return "(unresolved)"


def map_sources_to_dirs(data: list[dict], root: str, include_dirs: list[tuple[str, str]]) -> dict:
    """
    For each source file, find which include directories it depends on.
    Returns dict mapping dir_name -> list of (source_path, [headers resolved by this dir])
    """
    # dir_name -> {source_path -> [headers]}
    dir_to_sources = defaultdict(lambda: defaultdict(list))

    for entry in data:
        source_path = entry["path"]

        for header in entry["includes"]:
            resolved_by = resolve_header(header, source_path, root, include_dirs)
            if resolved_by:
                dir_to_sources[resolved_by][source_path].append(header)

    return dir_to_sources


def output_json(dir_to_sources: dict, include_dirs: list[tuple[str, str]], total_sources: int):
    """Output analysis as JSON."""
    # Convert nested defaultdict to regular dict for JSON serialization
    result = {
        "total_sources": total_sources,
        "include_directories": [name for name, _ in include_dirs],
        "directory_dependencies": {}
    }

    # Add special directories
    all_dirs = [d[0] for d in include_dirs] + [".", "..", "(unresolved)"]

    for dir_name in all_dirs:
        sources = dir_to_sources.get(dir_name, {})
        result["directory_dependencies"][dir_name] = {
            "source_count": len(sources),
            "sources": {
                source_path: headers
                for source_path, headers in sources.items()
            }
        }

    print(json.dumps(result, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "json_path",
        help="Path to data.json file from find_device_side_includes.py"
    )
    parser.add_argument(
        "repo_root",
        nargs="?",
        help="Repository root directory (default: parent of script directory)"
    )
    args = parser.parse_args()

    # Determine repo root
    if args.repo_root:
        root = args.repo_root
    else:
        # Default: assume script is in include_scripts/ directory
        root = str(Path(__file__).parent.parent)

    root = os.path.abspath(root)
    if not root.endswith("/"):
        root += "/"

    # Load data
    with open(args.json_path) as f:
        data = json.load(f)

    # Analyze and output JSON
    include_dirs = get_include_dirs(root)
    dir_to_sources = map_sources_to_dirs(data, root, include_dirs)
    output_json(dir_to_sources, include_dirs, len(data))


if __name__ == "__main__":
    main()
