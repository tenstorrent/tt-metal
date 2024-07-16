import ttnn
import torch
import tt_lib as ttl


class TensorCache:
    def __init__(self, num_users: int, entries_per_user: int, entry_size: int, device: ttnn.Device):
        self.num_users = num_users
        self.entries_per_user = entries_per_user
        self.entry_shape = [1, 1, 1, entry_size]

        self.cache = [
            [
                ttnn.from_torch(
                    torch.zeros(self.entry_shape),
                    device=device,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
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
        ttl.tensor.copy(value, self.cache[entry_idx][user_idx])

    def get(self, user_idx: int, entry_idx: int) -> ttnn.Tensor:
        assert entry_idx < len(self.cache), f"Expected key {entry_idx} to exist in cache"
        return self.cache[entry_idx][user_idx]

    def concat_users(self, entry_idx: int, layout=ttnn.TILE_LAYOUT):
        assert entry_idx < len(self.cache), f"Expected key {entry_idx} to exist in cache"
        return ttnn.to_layout(ttnn.concat(self.cache[entry_idx], dim=-2), layout)
