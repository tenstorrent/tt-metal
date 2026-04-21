# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

"""
Standalone workload for the real-time profiler sync-accuracy test.

Opens a mesh device (triggers the init-sync handshake), runs a handful of
matmuls with intermittent `ttnn.synchronize_device` calls (each one triggers
the mid-run sync path), then closes the device. The C++ mesh device logs
"init sync check diff_ns=..." and "midrun sync check diff_ns=..." for every
sync cycle — the outer pytest scrapes those lines and validates them.

Meant to be run under Tracy capture by the outer test
(test_realtime_profiler_sync_accuracy.py).
"""

import torch

import ttnn


def main():
    # RT profiler requires a tensix dispatch core (it is a BRISC kernel that
    # cannot run on an ethernet core). Force WORKER dispatch so the
    # dispatch_core_manager reserves a tensix slot at construction time.
    mesh_device = ttnn.open_mesh_device(
        ttnn.MeshShape(1, 1),
        l1_small_size=24576,
        dispatch_core_config=ttnn.DispatchCoreConfig(ttnn.DispatchCoreType.WORKER),
    )

    try:
        torch.manual_seed(0)
        m = 1024
        k = 1024
        n = 1024
        torch_a = torch.randn((m, k), dtype=torch.bfloat16)
        torch_b = torch.randn((k, n), dtype=torch.bfloat16)
        a = ttnn.from_torch(torch_a, layout=ttnn.TILE_LAYOUT, device=mesh_device)
        b = ttnn.from_torch(torch_b, layout=ttnn.TILE_LAYOUT, device=mesh_device)

        # Issue a few synchronize_device calls. Each call internally performs
        # a finish, which (after enough host time has elapsed) triggers the
        # mid-run sync check. We run enough iterations/matmuls to produce
        # several mid-run syncs.
        for _ in range(8):
            _ = a @ b
            _ = a @ b
            _ = a @ b
            ttnn.synchronize_device(mesh_device)
    finally:
        ttnn.close_mesh_device(mesh_device)


if __name__ == "__main__":
    main()
