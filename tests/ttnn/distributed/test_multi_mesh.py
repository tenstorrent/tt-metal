# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

"""
Unified multi-mesh test that auto-detects cluster type and splits mesh into 2 sub-meshes.

Supports Galaxy and P300 systems (all chips must have PCI access).
Demonstrates pipeline parallelism with inter-mesh communication via sockets.

Run:
    tt-run --rank-binding <binding>.yaml --mpi-args "--tag-output" python3 tests/ttnn/distributed/test_multi_mesh.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from multi_mesh_utils import run_multiprocess_pipeline


def test_galaxy_multi_mesh():
    """
    Galaxy multi-mesh test. Auto-detects mesh shape.

    Run on a Tenstorrent Galaxy system:
        tt-run --rank-binding 4x4_multi_mesh_rank_binding.yaml --mpi-args "--tag-output" python3 tests/ttnn/distributed/test_multi_mesh.py

    The rank binding file initializes a single physical 4x4 mesh per process.
    The mesh graph descriptor used: tests/tt_metal/tt_fabric/custom_mesh_descriptors/wh_galaxy_split_4x4_multi_mesh.textproto
    """
    run_multiprocess_pipeline()


def test_p300_multi_mesh():
    """
    P300 multi-mesh test. Auto-detects mesh shape.

    Run on a P300 system:
        tt-run --rank-binding tests/tt_metal/distributed/config/p300_1x1_multi_mesh_rank_binding.yaml --mpi-args "--tag-output" python3 tests/ttnn/distributed/test_multi_mesh.py

    The rank binding file initializes a single physical 1x1 mesh per process (1 chip per process).
    The mesh graph descriptor used: tests/tt_metal/tt_fabric/custom_mesh_descriptors/p300_split_1x1_multi_mesh.textproto
    """
    run_multiprocess_pipeline()


if __name__ == "__main__":
    run_multiprocess_pipeline()
