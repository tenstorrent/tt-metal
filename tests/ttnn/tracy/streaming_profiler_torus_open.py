# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""One tt-run rank of test_streaming_profiler_galaxy_torus_router_fits: open this rank's 4x2 mesh on 2D torus fabric
with the max packet payload given as the first argument, then close it. Opening the mesh builds and loads the fabric
routers, so a router that overflows its kernel config buffer fails here."""

import sys

import ttnn


def main(max_packet_payload_size_bytes: int) -> None:
    router_config = ttnn.FabricRouterConfig()
    router_config.max_packet_payload_size_bytes = max_packet_payload_size_bytes
    ttnn.set_fabric_config(
        ttnn.FabricConfig.FABRIC_2D_TORUS_Y,
        ttnn.FabricReliabilityMode.STRICT_INIT,
        None,
        ttnn.FabricTensixConfig.DISABLED,
        ttnn.FabricUDMMode.DISABLED,
        ttnn.FabricManagerMode.DEFAULT,
        router_config,
    )
    mesh_device = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(4, 2))
    print(f"TORUS_OPEN_OK devices={mesh_device.get_num_devices()}", flush=True)
    # A rank that closes early takes down its end of the links between meshes before the peer's routers finish their
    # handshake, so every rank waits here first.
    ttnn.distributed_context_barrier()
    ttnn.close_mesh_device(mesh_device)
    ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


if __name__ == "__main__":
    main(int(sys.argv[1]))
