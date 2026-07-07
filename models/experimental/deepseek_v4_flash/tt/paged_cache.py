"""Multi-user paged sliding KV cache helpers (vLLM / GPT-OSS style).

Physical layout: ``[num_blocks, 1, block_size, head_dim]`` DRAM tiles.
Each user owns a contiguous slice of blocks via a ``page_table`` row that maps
logical block index -> physical block id. Updates use
``ttnn.experimental.paged_update_cache``; sliding-only attention reads use
``ttnn.transformer.paged_scaled_dot_product_attention_decode``.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import ttnn


@dataclass(frozen=True)
class PagedCacheConfig:
    num_users: int
    max_seq: int
    block_size: int = 64
    sliding_window: int = 128

    @property
    def blocks_per_user(self) -> int:
        return (self.max_seq + self.block_size - 1) // self.block_size

    @property
    def total_blocks(self) -> int:
        return self.num_users * self.blocks_per_user


def build_page_table(cfg: PagedCacheConfig) -> torch.Tensor:
    """``[num_users, blocks_per_user]`` INT32 — user ``u`` owns physical blocks ``u*B .. u*B+B-1``."""
    bpu = cfg.blocks_per_user
    pt = torch.zeros(cfg.num_users, bpu, dtype=torch.int32)
    for u in range(cfg.num_users):
        base = u * bpu
        pt[u] = torch.arange(base, base + bpu, dtype=torch.int32)
    return pt


def page_table_row_to_tt(page_table: torch.Tensor, user_id: int, device: ttnn.MeshDevice) -> ttnn.Tensor:
    """Single-user page table slice ``[1, blocks_per_user]`` on ``device``."""
    row = page_table[user_id : user_id + 1]
    return ttnn.from_torch(row, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)


def build_paged_sliding_pool(device: ttnn.MeshDevice, cfg: PagedCacheConfig, head_dim: int) -> ttnn.Tensor:
    """Shared physical pool ``[total_blocks, 1, block_size, head_dim]`` (all-zero)."""
    return ttnn.from_torch(
        torch.zeros(cfg.total_blocks, 1, cfg.block_size, head_dim),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
