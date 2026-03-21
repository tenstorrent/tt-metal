# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Compressed KVPE paged attention cache for GLM-4.7-Flash hybrid.

Uses a single [num_blocks, 1, block_size, kvpe_dim] tensor (where kvpe_dim =
kv_lora_rank + qk_rope_head_dim = 576) instead of separate K/V caches. This
halves memory usage vs. the standard separate-cache approach.

Supports BF8 dtype by default for further memory savings.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import torch

import ttnn
from models.demos.glm4_moe_lite_hybrid.core.config import Glm4MoeLiteHParams


def _env_kv_cache_dtype() -> ttnn.DataType:
    override = os.environ.get("GLM4_MOE_LITE_KV_CACHE_TT_DTYPE", "").strip().lower()
    if not override:
        return ttnn.bfloat8_b
    table = {
        "bf8": ttnn.bfloat8_b,
        "bfloat8_b": ttnn.bfloat8_b,
        "bf16": ttnn.bfloat16,
        "bfloat16": ttnn.bfloat16,
        "f32": ttnn.float32,
        "float32": ttnn.float32,
    }
    if override not in table:
        raise ValueError(f"Invalid GLM4_MOE_LITE_KV_CACHE_TT_DTYPE={override!r}")
    return table[override]


@dataclass
class CompressedKVPECacheConfig:
    """Configuration for compressed KVPE paged cache."""

    block_size: int = 64
    max_num_blocks: int = 2048
    num_layers: int = 47
    dtype: ttnn.DataType | None = None

    def __post_init__(self):
        if self.dtype is None:
            self.dtype = _env_kv_cache_dtype()


class CompressedKVPECache:
    """Compressed paged KV cache storing [kv_nope || k_rope] per token.

    Each layer gets a single cache tensor of shape:
        [max_num_blocks, 1, block_size, kvpe_dim]

    where kvpe_dim = kv_lora_rank + qk_rope_head_dim.

    This exploits the MLA architecture: after the kv_a_layernorm and RoPE,
    the KV state is already in compressed latent form.
    """

    def __init__(
        self,
        hparams: Glm4MoeLiteHParams,
        config: CompressedKVPECacheConfig,
    ):
        self.hparams = hparams
        self.config = config
        self.kvpe_dim = int(hparams.kv_lora_rank) + int(hparams.qk_rope_head_dim)
        self.num_layers = config.num_layers
        self._caches: list[ttnn.Tensor | None] = [None] * self.num_layers
        self._page_table: ttnn.Tensor | None = None
        self._device = None

    def to_device(self, device: Any, batch_size: int = 1) -> "CompressedKVPECache":
        self._device = device
        is_mesh = device.__class__.__name__ == "MeshDevice"
        mapper = ttnn.ReplicateTensorToMesh(device) if is_mesh else None

        cache_shape = (
            self.config.max_num_blocks,
            1,
            self.config.block_size,
            self.kvpe_dim,
        )
        for layer_idx in range(self.num_layers):
            cache_host = torch.zeros(cache_shape, dtype=torch.bfloat16)
            self._caches[layer_idx] = ttnn.from_torch(
                cache_host,
                device=device,
                dtype=self.config.dtype,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=mapper,
            )

        max_blocks_per_req = self.config.max_num_blocks // max(1, batch_size)
        page_table_host = torch.zeros((batch_size, max_blocks_per_req), dtype=torch.int32)
        for b in range(batch_size):
            for i in range(max_blocks_per_req):
                page_table_host[b, i] = b * max_blocks_per_req + i
        self._page_table = ttnn.from_torch(
            page_table_host,
            device=device,
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=mapper,
        )
        return self

    def get_cache(self, layer_idx: int) -> ttnn.Tensor:
        assert (
            self._caches[layer_idx] is not None
        ), f"Cache for layer {layer_idx} not initialized. Call to_device() first."
        return self._caches[layer_idx]

    @property
    def page_table(self) -> ttnn.Tensor:
        assert self._page_table is not None, "Page table not initialized."
        return self._page_table

    def fill_cache(
        self,
        layer_idx: int,
        kvpe: ttnn.Tensor,
        batch_idx: int,
    ) -> None:
        ttnn.experimental.paged_fill_cache(
            self._caches[layer_idx],
            kvpe,
            page_table=self._page_table,
            batch_idx=batch_idx,
        )

    def update_cache(
        self,
        layer_idx: int,
        kvpe_new_sharded: ttnn.Tensor,
        positions: ttnn.Tensor,
        positions_main: ttnn.Tensor | None = None,
        positions_draft: ttnn.Tensor | None = None,
    ) -> None:
        cache = self._caches[layer_idx]
        kwargs = {"page_table": self._page_table}

        if self._device.__class__.__name__ == "MeshDevice":
            try:
                mesh_rows = int(self._device.shape[0])
                mesh_cols = int(self._device.shape[1])
                kwargs["mesh_coords"] = {ttnn.MeshCoordinate(r, c) for r in range(mesh_rows) for c in range(mesh_cols)}
            except Exception:
                pass

        if positions_main is not None and positions_draft is not None:
            ttnn.experimental.paged_update_cache(cache, kvpe_new_sharded, update_idxs_tensor=positions_main, **kwargs)
            ttnn.experimental.paged_update_cache(cache, kvpe_new_sharded, update_idxs_tensor=positions_draft, **kwargs)
        else:
            ttnn.experimental.paged_update_cache(cache, kvpe_new_sharded, update_idxs_tensor=positions, **kwargs)
