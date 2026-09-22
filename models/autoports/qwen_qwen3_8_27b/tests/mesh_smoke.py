# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Bounded by the caller; open and close the functional stage's 1x1 mesh."""

import ttnn

if __name__ == "__main__":
    print("OPEN_BEGIN", flush=True)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=0)
    try:
        print("GRID", mesh.compute_with_storage_grid_size(), flush=True)
    finally:
        ttnn.close_mesh_device(mesh)
    print("MESH_SMOKE_OK", flush=True)
