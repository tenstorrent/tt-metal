# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Device opening for the standalone entry points (demo, perf test, the harness selftests).

The pipeline itself (tt/) never opens a device: build_pipeline runs on the device it is handed. The
e2e test gets its mesh from the conftest `mesh_device` fixture with the same parameters as here.
"""
from __future__ import annotations

import ttnn

MESH_SHAPE = (2, 4)

# trace_region_size is sized from the LARGEST stage trace, measured: denoise 760.3 MB (precise transformer:
# 60 blocks x cond + uncond, 2-limb projections, exact-lane QK^T), vision_encode 548.2 MB (exact-lane
# precise vision tower), vae_decode 266.5 MB, all at B=32. 896 MB is ~1.18x the largest.
DEVICE_PARAMS = {"l1_small_size": 24576, "trace_region_size": 896 * 1024 * 1024, "num_command_queues": 1}


def open_mesh(device_params=None, mesh_shape=MESH_SHAPE):
    """Open the T3K as a 2x4 mesh with 1D fabric (what the e2e test's mesh_device fixture opens)."""
    params = dict(DEVICE_PARAMS)
    params.update(device_params or {})
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    return ttnn.open_mesh_device(ttnn.MeshShape(*mesh_shape), **params)


def close_mesh(mesh):
    ttnn.close_mesh_device(mesh)
    ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
