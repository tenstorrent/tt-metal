# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from contextlib import contextmanager
from itertools import count, takewhile

import ttnn

from loguru import logger
from tests.scripts.common import get_updated_device_params


def mesh_shape_iterator(num_devices, limit=None):
    assert num_devices % 2 == 0
    for r in takewhile(lambda x: x <= num_devices, (2**i for i in (range(limit) if limit else count()))):
        yield (num_devices // r, r)


@contextmanager
def device_context(mesh_shape, fabric_config, device_params=None):
    device_params = device_params or {}
    mesh_device = None
    try:
        logger.info("Setting up device")
        ttnn.set_fabric_config(fabric_config)
        mesh_device = ttnn.open_mesh_device(
            mesh_shape=ttnn.MeshShape(mesh_shape), **get_updated_device_params(device_params)
        )
        yield mesh_device, None
    except AssertionError as e:
        logger.error(f"Device error: {e}")
        yield None, f"Device error {e}"
    finally:
        logger.info("Tearing down device")
        if mesh_device:
            ttnn.close_mesh_device(mesh_device)
            ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
            del mesh_device


def get_mem_config(buffer_type, shard_shape, shard_strategy, device):
    if shard_shape is None or shard_strategy is None:
        return ttnn.MemoryConfig(buffer_type=buffer_type)
    else:
        core_grid_size = device.compute_with_storage_grid_size()
        core_grid = ttnn.CoreRangeSet(
            {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(core_grid_size.x - 1, core_grid_size.y - 1))}
        )

        shard_spec = ttnn.ShardSpec(core_grid, shard_shape, ttnn.ShardOrientation.ROW_MAJOR)
        return ttnn.MemoryConfig(shard_strategy, buffer_type, shard_spec)
