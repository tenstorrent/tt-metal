#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Thin wrapper around generate_rank_bindings.generate_supported_rank_bindings()
that writes all output files (rank binding YAMLs and tray-to-PCIe mapping) into
a subdirectory named after the local hostname, safe for use on a shared filesystem.

Must be run from the repo root (same requirement as the underlying script).
"""

import os
import socket
import subprocess
import sys
from pathlib import Path

# Ensure the sibling module is importable regardless of working directory.
sys.path.insert(0, str(Path(__file__).parent))
import generate_rank_bindings as grb  # noqa: E402


def main():
    output_dir = Path(socket.gethostname()).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Resolve the test executable to an absolute path now, before os.chdir moves us
    # into output_dir (where the relative path "build/..." would no longer resolve).
    abs_test_exec = Path("build/test/tt_metal/tt_fabric/test_physical_discovery").resolve()

    def _mapping_with_abs_exec(mapping_file):
        """Drop-in replacement for grb.generate_tray_to_pcie_device_mapping that uses
        an absolute executable path, required after os.chdir(output_dir) below."""
        from loguru import logger

        if not abs_test_exec.exists():
            logger.error(f"Test executable not found at {abs_test_exec}")
            sys.exit(1)
        cmd = [str(abs_test_exec), "--gtest_filter=*GenerateTrayToPCIeDeviceMapping*"]
        logger.info(f"Running: {' '.join(cmd)}")
        try:
            result = subprocess.run(cmd)
            if result.returncode != 0:
                logger.error(f"{cmd} Failed to generate tray to pcie device mapping")
                sys.exit(result.returncode)
        except KeyboardInterrupt:
            logger.error(f"{cmd} Interrupted")
            sys.exit(1)
        if not os.path.exists(mapping_file):
            logger.error(f"{mapping_file} not found")
            sys.exit(1)

    grb.generate_tray_to_pcie_device_mapping = _mapping_with_abs_exec

    # chdir so that every relative-path write in generate_supported_rank_bindings
    # (the mapping YAML and all binding YAMLs) lands in output_dir.
    os.chdir(output_dir)
    grb.generate_supported_rank_bindings()


if __name__ == "__main__":
    main()
