#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Minimal 2-Galaxy exabox test - verifies multi-host connectivity.

Run with:
tt-run --rank-binding tests/ttnn/distributed/exabox_2_galaxy_rank_binding.yaml \
    --mpi-args "--host wh-glx-a03u02,wh-glx-a04u08 --mca btl self,tcp --tag-output" \
    python tests/ttnn/distributed/test_exabox_2_galaxy_minimal.py
"""

import socket
import ttnn
from loguru import logger


def main():
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_2D)

    mesh_shape = ttnn.MeshShape(8, 4)
    device = ttnn.open_mesh_device(mesh_shape=mesh_shape)

    if not ttnn.distributed_context_is_initialized():
        raise RuntimeError("Distributed context not initialized")

    rank = int(ttnn.distributed_context_get_rank())
    world_size = int(ttnn.distributed_context_get_size())
    hostname = socket.gethostname()

    logger.info(f"Rank {rank}/{world_size} on {hostname}: Device opened successfully")

    num_devices = device.get_num_devices()
    logger.info(f"Rank {rank}: Mesh has {num_devices} devices")

    ttnn.distributed_context_barrier()
    logger.info(f"Rank {rank}: Barrier passed - all ranks synchronized")

    ttnn.close_device(device)
    logger.info(f"Rank {rank}: Test PASSED")


if __name__ == "__main__":
    main()
