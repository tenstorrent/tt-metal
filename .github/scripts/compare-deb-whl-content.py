#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Compare JIT kernel sources and supporting headers between .deb packages and .whl.

The tt-metalium and tt-nn projects ship runtime files (NN kernel sources,
supporting headers, device descriptors, etc.) that the JIT compiler needs at
runtime.  These files are packaged two ways:

  1. CMake / CPack  → .deb packages
     - tt-metalium_*.deb  (metalium-runtime component)
     - tt-nn_*.deb        (ttnn-runtime component)
     Both install JIT files under /usr/libexec/tt-metalium/.

  2. Python wheel  → .whl
     The same files are bundled inside the wheel under:
       ttnn/tt_metal/...      (from tt_metal source tree)
       ttnn/ttnn/cpp/...      (from ttnn/cpp source tree)
       ttnn/api/...           (ttnn API headers needed for JIT)

This script extracts the file manifests from both formats, normalises the
paths to a common root, and reports any discrepancies.  If the sets differ
the script exits with code 1 so CI can gate on it.

Usage:
    python compare-deb-whl-content.py \\
        --metalium-deb  ./pkgs/tt-metalium_*.deb \\
        --nn-deb        ./pkgs/tt-nn_*.deb \\
        --whl           ./wheelhouse/ttnn-*.whl
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import zipfile
from pathlib import Path


# ── helpers ──────────────────────────────────────────────────────────────────

DEB_JIT_PREFIX = "./usr/libexec/tt-metalium/"


def files_in_deb(deb_path: Path) -> set[str]:
    """Return the set of regular-file paths inside a .deb."""
    result = subprocess.run(
        ["dpkg-deb", "--contents", str(deb_path)],
        capture_output=True,
        text=True,
        check=True,
    )
    files: set[str] = set()
    for line in result.stdout.splitlines():
        if not line:
            continue
        # typical line:
        #   -rw-r--r-- root/root  1234 2024-01-01 00:00 ./usr/libexec/tt-metalium/...
        parts = line.split()
        if len(parts) < 6:
            continue
        path = parts[-1]
        if not path.endswith("/"):  # skip directories
            files.add(path)
    return files


def files_in_whl(whl_path: Path) -> set[str]:
    """Return the set of file paths inside a .whl (which is a zip)."""
    with zipfile.ZipFile(whl_path) as zf:
        return {n for n in zf.namelist() if not n.endswith("/")}


# ── normalisation ────────────────────────────────────────────────────────────


def jit_files_from_debs(
    metalium_deb: Path,
    nn_deb: Path,
) -> set[str]:
    """
    Collect JIT files from the two runtime .deb packages.

    Only files installed under /usr/libexec/tt-metalium/ are considered;
    shared libraries, cmake configs, etc. are ignored.
    """
    all_deb_files = files_in_deb(metalium_deb) | files_in_deb(nn_deb)
    normalised: set[str] = set()
    for f in all_deb_files:
        if f.startswith(DEB_JIT_PREFIX):
            rel = f[len(DEB_JIT_PREFIX) :]
            if rel:  # skip the directory entry itself
                normalised.add(rel)
    return normalised


def jit_files_from_whl(whl_path: Path) -> set[str]:
    """
    Collect JIT files from the Python wheel.

    The wheel bundles JIT content under:
      ttnn/tt_metal/…       → normalised to  tt_metal/…
      ttnn/ttnn/cpp/…       → normalised to  ttnn/cpp/…
      ttnn/api/…            → normalised to  api/…

    Everything else (*.py, *.so, .dist-info, __pycache__, runtime/hw, etc.)
    is ignored.
    """
    whl_files = files_in_whl(whl_path)
    normalised: set[str] = set()

    # Prefixes inside the wheel that correspond to JIT content, with
    # the prefix to strip.  After stripping "ttnn/" the relative paths
    # should line up with the deb layout under /usr/libexec/tt-metalium/.
    jit_prefixes = (
        "ttnn/tt_metal/",
        "ttnn/ttnn/",  # ttnn/ttnn/cpp/… → ttnn/cpp/…
        "ttnn/api/",  # ttnn/api/ttnn/… → api/ttnn/…
    )

    skip_suffixes = (
        ".py",
        ".pyc",
        ".pyi",
        ".so",
        ".pth",
    )
    skip_infixes = (
        ".dist-info/",
        "__pycache__/",
    )

    for f in whl_files:
        if any(f.endswith(s) for s in skip_suffixes):
            continue
        if any(s in f for s in skip_infixes):
            continue

        for prefix in jit_prefixes:
            if f.startswith(prefix):
                rel = f[len("ttnn/") :]  # strip the top-level "ttnn/"
                if rel:
                    normalised.add(rel)
                break

    return normalised


# ── main ─────────────────────────────────────────────────────────────────────


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare JIT content between .deb packages and .whl",
    )
    parser.add_argument(
        "--metalium-deb",
        required=True,
        type=Path,
        help="Path to the tt-metalium_*.deb runtime package",
    )
    parser.add_argument(
        "--nn-deb",
        required=True,
        type=Path,
        help="Path to the tt-nn_*.deb runtime package",
    )
    parser.add_argument(
        "--whl",
        required=True,
        type=Path,
        help="Path to the ttnn-*.whl",
    )
    args = parser.parse_args()

    # Sanity-check inputs
    for p, label in [
        (args.metalium_deb, "--metalium-deb"),
        (args.nn_deb, "--nn-deb"),
        (args.whl, "--whl"),
    ]:
        if not p.is_file():
            print(f"ERROR: {label} path does not exist: {p}", file=sys.stderr)
            return 2

    print("Extracting file lists …")
    deb_files = jit_files_from_debs(args.metalium_deb, args.nn_deb)
    whl_files = jit_files_from_whl(args.whl)

    print(f"  .deb JIT files : {len(deb_files)}")
    print(f"  .whl JIT files : {len(whl_files)}")
    print()

    only_in_deb = sorted(deb_files - whl_files)
    only_in_whl = sorted(whl_files - deb_files)

    if not only_in_deb and not only_in_whl:
        print("✅ All JIT kernel sources and supporting headers match between .deb and .whl!")
        return 0

    # ── report discrepancies ─────────────────────────────────────────────
    print("❌ Discrepancies found between .deb and .whl packaging!\n")

    if only_in_deb:
        print(f"### Files in .deb but MISSING from .whl ({len(only_in_deb)}):")
        print("These files are installed by CMake (CPack) but are not bundled")
        print("into the Python wheel.  Add matching patterns to setup.py.\n")
        for f in only_in_deb:
            print(f"  {f}")
        print()

    if only_in_whl:
        print(f"### Files in .whl but MISSING from .deb ({len(only_in_whl)}):")
        print("These files are bundled into the Python wheel but are not installed")
        print("by CMake.  Add them to the appropriate CMake FILE_SET or install().\n")
        for f in only_in_whl:
            print(f"  {f}")
        print()

    print(
        "To fix: ensure that every JIT file (kernel source, supporting header,\n"
        "device descriptor, etc.) is declared in BOTH packaging systems.\n"
        "  • CMake side  : FILE_SET jit_api / kernels in the relevant CMakeLists.txt\n"
        "  • Wheel side  : tt_metal_patterns / ttnn_cpp_patterns in setup.py"
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
