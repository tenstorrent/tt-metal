# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import ttnn

from enum import Enum


class DeviceGetter:
    _instance = None
    l1_small_size = 1 << 15

    def __init__(self):
        raise RuntimeError("This is Singleton, invoke get_device() instead.")

    @classmethod
    def get_device(cls):
        if cls._instance == None:
            ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)  # Confirm if/why this is needed, hangs without it
            cls._instance = ttnn.open_mesh_device(
                mesh_shape=ttnn.MeshShape(1, 8),
                l1_small_size=cls.l1_small_size
            )
            print(f"Device: {cls._instance}")
        return cls._instance
