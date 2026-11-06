# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""One tt-run rank of test_streaming_profiler_torus_routers_fit: set the fabric config named by the first argument with
the max packet payload given as the fourth, open this rank's mesh of the given rows and columns, then close it. Opening
the mesh builds the fabric routers and lays out their kernel config buffers (on a mock cluster too), so a router that
overflows its buffer fails here."""

import sys

import ttnn


def main(fabric_config: str, rows: int, cols: int, max_packet_payload_size_bytes: int) -> None:
    router_config = ttnn.FabricRouterConfig()
    router_config.max_packet_payload_size_bytes = max_packet_payload_size_bytes
    ttnn.set_fabric_config(
        getattr(ttnn.FabricConfig, fabric_config),
        ttnn.FabricReliabilityMode.STRICT_INIT,
        None,
        ttnn.FabricTensixConfig.DISABLED,
        ttnn.FabricUDMMode.DISABLED,
        ttnn.FabricManagerMode.DEFAULT,
        router_config,
    )
    mesh_device = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(rows, cols))
    print(f"TORUS_OPEN_OK devices={mesh_device.get_num_devices()}", flush=True)
    # A rank that closes early takes down its end of the links between meshes before the peer's routers finish their
    # handshake, so every rank waits here first.
    ttnn.distributed_context_barrier()
    ttnn.close_mesh_device(mesh_device)
    ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]))
