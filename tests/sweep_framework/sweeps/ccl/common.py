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
def device_context(mesh_shape, fabric_config):
    mesh_device = None
    try:
        logger.info("Setting up device")
        ttnn.set_fabric_config(fabric_config)
        mesh_device = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(mesh_shape), **get_updated_device_params({}))
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