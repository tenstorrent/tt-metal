# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
import torch


class TensorCache:
    def __init__(
        self, num_users: int, entries_per_user: int, entry_size: int, device: ttnn.Device, on_host: bool = True
    ):
        self.device = device
        self.num_users = num_users
        self.entries_per_user = entries_per_user
        self.entry_shape = [1, 1, 1, entry_size]
        self.on_host = on_host

        if self.on_host:
            self.cache_device = None
            self.cache_memory_config = None
        else:
            self.cache_device = device
            self.cache_memory_config = ttnn.DRAM_MEMORY_CONFIG
        self.cache = [
            [
                ttnn.from_torch(
                    torch.zeros(self.entry_shape),
                    device=self.cache_device,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                    memory_config=self.cache_memory_config,
                    dtype=ttnn.bfloat16,
                )
                for _ in range(self.num_users)
            ]
            for _ in range(self.entries_per_user)
        ]

    def set(self, user_idx: int, entry_idx: int, value: ttnn.Tensor):
        assert entry_idx < len(self.cache), f"Expected key {entry_idx} to exist in cache"
        assert value.layout == ttnn.ROW_MAJOR_LAYOUT, f"Expected value tensor to be row-major layout"
        assert list(value.shape) == self.entry_shape, f"Expected value tensor to be correct shape ({self.entry_shape})"
        if self.on_host:
            self.cache[entry_idx][user_idx] = value.cpu()
        else:
            ttnn.copy(value, self.cache[entry_idx][user_idx])

    def get(self, user_idx: int, entry_idx: int) -> ttnn.Tensor:
        assert entry_idx < len(self.cache), f"Expected key {entry_idx} to exist in cache"
        return ttnn.to_device(
            self.cache[entry_idx][user_idx], device=self.device, memory_config=self.cache_memory_config
        )

    def concat_users(self, entry_idx: int, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG):
        assert entry_idx < len(self.cache), f"Expected key {entry_idx} to exist in cache"
        values = self.cache[entry_idx]
        if self.on_host:
            values = [
                ttnn.to_device(values[i], device=self.device, memory_config=memory_config)
                for i in range(self.num_users)
            ]
        return ttnn.to_layout(ttnn.concat(values, dim=2), layout)

    def reset(self):
        for entry_idx in range(len(self.cache)):
            for user_idx in range(self.num_users):
                self.cache[entry_idx][user_idx] = ttnn.from_torch(
                    torch.zeros(self.entry_shape),
                    device=self.cache_device,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                    memory_config=self.cache_memory_config,
                    dtype=ttnn.bfloat16,
                )
