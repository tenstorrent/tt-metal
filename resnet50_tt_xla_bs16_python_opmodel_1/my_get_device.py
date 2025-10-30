# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import ttnn


class DeviceGetter:
    _instance = None
    l1_small_size = 1 << 15

    def __init__(self):
        raise RuntimeError("This is Singleton, invoke get_device() instead.")

    @classmethod
    def get_device(cls):
        if cls._instance == None:
            cls._instance = ttnn.open_device(device_id=0, l1_small_size=cls.l1_small_size)
        return cls._instance
