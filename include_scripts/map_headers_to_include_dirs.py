#!/usr/bin/env python3
"""
Map headers to include directories: for each unique #include, show which
include directory resolves it.

Outputs JSON to stdout.

Usage:
    python3 include_scripts/find_device_side_includes.py > data.json
    python3 include_scripts/map_headers_to_include_dirs.py data.json > headers_to_dirs.json
"""

import argparse
import json
import os
import re
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


def extract_unique_includes(data: list[dict]) -> set[str]:
    """Extract unique include paths, excluding relative paths."""
    includes = set()
    for entry in data:
        for inc in entry["includes"]:
            if not inc.startswith(".."):
                includes.add(inc)
    return includes


def resolve_headers_static(includes: set[str], include_dirs: list[tuple[str, str]]) -> dict:
    """
    For each header, find which include directory resolves it (static dirs only).
    Returns dict with 'resolved', 'unresolved', 'system', and 'by_directory'.
    """
    resolved = {}  # header -> resolving directory name
    by_directory = defaultdict(list)  # directory name -> list of headers

    for header in includes:
        if header in SYSTEM_HEADERS:
            continue

        for dir_name, dir_path in include_dirs:
            full_path = os.path.join(dir_path, header)
            if os.path.isfile(full_path):
                resolved[header] = dir_name
                by_directory[dir_name].append(header)
                break

    system = includes & SYSTEM_HEADERS
    unresolved = includes - set(resolved.keys()) - system

    return {
        "resolved": resolved,
        "unresolved": unresolved,
        "system": system,
        "by_directory": dict(by_directory),
    }


def resolve_headers_relative(data: list[dict], root: str, include_dirs: list[tuple[str, str]]) -> dict:
    """
    Check if headers resolve via . or .. relative to the source file.
    Returns dict with 'resolved_by_dot', 'resolved_by_dotdot', 'still_unresolved'.
    """
    def resolves_statically(header):
        if header in SYSTEM_HEADERS:
            return True
        for _, dir_path in include_dirs:
            if os.path.isfile(os.path.join(dir_path, header)):
                return True
        return False

    resolved_by_dot = defaultdict(list)      # header -> [(source, resolved_path), ...]
    resolved_by_dotdot = defaultdict(list)   # header -> [(source, resolved_path), ...]
    still_unresolved = defaultdict(list)     # header -> [source, ...]

    for entry in data:
        source_path = entry["path"]
        source_dir = os.path.dirname(os.path.join(root, source_path))
        parent_dir = os.path.dirname(source_dir)

        for header in entry["includes"]:
            if resolves_statically(header):
                continue

            # Try . (source_dir)
            dot_path = os.path.join(source_dir, header)
            if os.path.isfile(dot_path):
                resolved_by_dot[header].append((source_path, dot_path.replace(root, "")))
                continue

            # Try .. (parent_dir)
            dotdot_path = os.path.join(parent_dir, header)
            if os.path.isfile(dotdot_path):
                resolved_by_dotdot[header].append((source_path, dotdot_path.replace(root, "")))
                continue

            still_unresolved[header].append(source_path)

    return {
        "resolved_by_dot": dict(resolved_by_dot),
        "resolved_by_dotdot": dict(resolved_by_dotdot),
        "still_unresolved": dict(still_unresolved),
    }


def find_host_side_includes(root: str) -> dict[str, list[str]]:
    """
    Scan host-side C++ files to find which headers they include.
    Returns dict: {header: [host_file_paths]}
    """
    exclude_dirs = ["build_", ".git", "__pycache__", "docs"]
    host_includes = defaultdict(list)

    print("Scanning for host-side files...", file=sys.stderr)

    for dirpath, dirnames, filenames in os.walk(root):
        # Skip excluded directories
        dirnames[:] = [d for d in dirnames if not any(excl in d for excl in exclude_dirs)]

        for filename in filenames:
            if not (filename.endswith(('.cpp', '.cc', '.h', '.hpp'))):
                continue

            filepath = Path(dirpath) / filename
            try:
                rel_path = str(filepath.relative_to(root))
            except ValueError:
                continue

            # Skip jit_build (host code but not relevant for header sharing)
            if rel_path.startswith('tt_metal/jit_build/'):
                continue

            try:
                content = filepath.read_text(encoding='utf-8', errors='replace')
            except Exception:
                continue

            # Skip device kernels (they're in device_data already)
            if re.search(r'\bvoid\s+(kernel_main|MAIN)\b', content):
                continue

            # Extract includes
            for match in re.finditer(r'^\s*#include\s*[<"]([^>"]+)[>"]', content, re.MULTILINE):
                header = match.group(1)
                host_includes[header].append(rel_path)

    print(f"Scanned host-side files", file=sys.stderr)
    return dict(host_includes)


def categorize_headers_by_usage(includes: set[str], static_result: dict, relative_result: dict, host_includes: dict) -> dict:
    """
    Categorize headers by whether they're device-only or shared with host.
    """
    all_device_headers = (
        set(static_result['resolved'].keys()) |
        set(relative_result['resolved_by_dot'].keys()) |
        set(relative_result['resolved_by_dotdot'].keys())
    )

    shared_headers = {}
    device_only_headers = []

    for header in all_device_headers:
        if header in host_includes:
            shared_headers[header] = {
                "host_file_count": len(host_includes[header]),
                "host_files_sample": host_includes[header][:5]  # Keep first 5 as sample
            }
        else:
            device_only_headers.append(header)

    return {
        "shared_headers": shared_headers,
        "device_only_headers": device_only_headers
    }


def output_json(includes: set[str], static_result: dict, relative_result: dict, include_dirs: list[tuple[str, str]], root: str):
    """Output analysis as JSON."""
    # Check host usage
    host_includes = find_host_side_includes(root)
    usage_categorization = categorize_headers_by_usage(includes, static_result, relative_result, host_includes)

    result = {
        "summary": {
            "total_unique_includes": len(includes),
            "system_headers": len(static_result['system']),
            "resolved_by_static_dirs": len(static_result['resolved']),
            "resolved_by_dot": len(relative_result['resolved_by_dot']),
            "resolved_by_dotdot": len(relative_result['resolved_by_dotdot']),
            "still_unresolved": len(relative_result['still_unresolved']),
            "shared_with_host": len(usage_categorization['shared_headers']),
            "device_only": len(usage_categorization['device_only_headers'])
        },
        "headers_by_directory": {
            name: static_result["by_directory"].get(name, [])
            for name, _ in include_dirs
        },
        "host_usage": usage_categorization,
        "resolved_by_dot": {
            header: [{"source": src, "resolved_path": path} for src, path in sources]
            for header, sources in relative_result['resolved_by_dot'].items()
        },
        "resolved_by_dotdot": {
            header: [{"source": src, "resolved_path": path} for src, path in sources]
            for header, sources in relative_result['resolved_by_dotdot'].items()
        },
        "still_unresolved": {
            header: sources
            for header, sources in relative_result['still_unresolved'].items()
        },
        "system_headers": sorted(list(static_result['system']))
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
    includes = extract_unique_includes(data)
    static_result = resolve_headers_static(includes, include_dirs)
    relative_result = resolve_headers_relative(data, root, include_dirs)
    output_json(includes, static_result, relative_result, include_dirs, root)


if __name__ == "__main__":
    main()
