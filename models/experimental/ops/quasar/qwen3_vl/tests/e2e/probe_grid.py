# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0
"""Print arch and compute grid of the device the current env targets."""
import ttnn


def main():
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1))
    try:
        g = mesh.compute_with_storage_grid_size()
        print(f"PROBE arch={ttnn.get_arch_name()} grid={g.x}x{g.y} dram_grid={mesh.dram_grid_size()}")
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
