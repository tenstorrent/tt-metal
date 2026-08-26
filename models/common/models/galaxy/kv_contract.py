# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Model-owned per-layer paged KV metadata for Galaxy 2D transformers.

``Attention2D`` takes its KV cache as an externally owned binding, so the
reconstructed models own the metadata that describes it. The common
``PagedKVCacheManager`` derives cache shapes from per-layer attention metadata;
:class:`GalaxyPagedKVContract` presents exactly that view without a runtime
change and without teaching the runtime anything about Galaxy.

On Galaxy the KV heads are sharded over the eight mesh rows and the users over
the four mesh columns, so a layer's device-local cache holds
``n_kv_heads / 8`` heads.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import ttnn
from models.common.models.galaxy.recipes import GALAXY_ROWS
from models.common.modules.attention.attention_2d import PagedKVMetadata


@dataclass(frozen=True)
class GalaxyPagedAttentionConfig:
    """Paged KV block geometry shared by every layer of one model."""

    block_size: int
    max_num_blocks: int

    def __post_init__(self) -> None:
        if self.block_size <= 0 or self.max_num_blocks <= 0:
            raise ValueError("paged block_size and max_num_blocks must be positive")


@dataclass(frozen=True)
class GalaxyAttentionKVSpec:
    """Per-layer KV metadata consumed by the common paged KV manager."""

    n_local_kv_heads: int
    head_dim: int
    kv_cache_dtype: Any
    paged_attention_config: GalaxyPagedAttentionConfig | None = None
    page_table_dtype: Any = ttnn.int32

    def __post_init__(self) -> None:
        if self.n_local_kv_heads <= 0 or self.head_dim <= 0:
            raise ValueError("local KV heads and head_dim must be positive")
        if self.kv_cache_dtype is None or self.page_table_dtype is None:
            raise ValueError("KV cache and page-table dtypes must be resolved")

    @classmethod
    def from_geometry(
        cls,
        *,
        n_kv_heads: int,
        head_dim: int,
        kv_cache_dtype: Any,
        paged_attention_config: GalaxyPagedAttentionConfig | None = None,
        page_table_dtype: Any = ttnn.int32,
    ) -> "GalaxyAttentionKVSpec":
        if n_kv_heads % GALAXY_ROWS:
            raise ValueError(f"n_kv_heads {n_kv_heads} must shard over {GALAXY_ROWS} mesh rows")
        return cls(
            n_local_kv_heads=n_kv_heads // GALAXY_ROWS,
            head_dim=head_dim,
            kv_cache_dtype=kv_cache_dtype,
            paged_attention_config=paged_attention_config,
            page_table_dtype=page_table_dtype,
        )

    def paged_kv_metadata(self) -> PagedKVMetadata | None:
        """Return the Attention2D binding metadata, or None for a contiguous cache."""

        if self.paged_attention_config is None:
            return None
        return PagedKVMetadata(
            block_size=self.paged_attention_config.block_size,
            max_num_blocks=self.paged_attention_config.max_num_blocks,
            cache_dtype=self.kv_cache_dtype,
            page_table_dtype=self.page_table_dtype,
        )

    def local_cache_shape(self) -> tuple[int, int, int, int]:
        """Return the device-local paged cache shape for one layer."""

        if self.paged_attention_config is None:
            raise ValueError("a contiguous KV cache has no paged shape")
        return (
            self.paged_attention_config.max_num_blocks,
            self.n_local_kv_heads,
            self.paged_attention_config.block_size,
            self.head_dim,
        )

    def with_paged_config(self, config: GalaxyPagedAttentionConfig) -> "GalaxyAttentionKVSpec":
        return GalaxyAttentionKVSpec(
            n_local_kv_heads=self.n_local_kv_heads,
            head_dim=self.head_dim,
            kv_cache_dtype=self.kv_cache_dtype,
            paged_attention_config=config,
            page_table_dtype=self.page_table_dtype,
        )


@dataclass(frozen=True)
class _GalaxyKVLayerView:
    attention_config: GalaxyAttentionKVSpec


@dataclass(frozen=True)
class _GalaxyKVModelConfigView:
    mesh_device: Any
    num_devices: int
    n_layers: int
    block_configs: tuple[_GalaxyKVLayerView, ...]


class GalaxyPagedKVContract:
    """Present model-owned KV metadata and binding to the common KV manager.

    The common ``PagedKVCacheManager`` reads per-layer attention metadata from
    ``config.block_configs[i].attention_config`` and binds through
    ``set_kv_cache``. Galaxy modules keep their own placement contract, so the
    model exposes this narrow view instead of reshaping its module configs to
    match a runtime expectation.
    """

    def __init__(self, model: Any, specs: tuple[GalaxyAttentionKVSpec, ...]):
        if not specs:
            raise ValueError("a paged KV contract requires at least one layer spec")
        mesh_device = getattr(getattr(model, "config", None), "mesh_device", None) or getattr(
            model, "mesh_device", None
        )
        if mesh_device is None:
            raise ValueError("the model must expose a resolved mesh_device")
        self._model = model
        self._specs = tuple(specs)
        self.config = _GalaxyKVModelConfigView(
            mesh_device=mesh_device,
            num_devices=GALAXY_ROWS,
            n_layers=len(self._specs),
            block_configs=tuple(_GalaxyKVLayerView(attention_config=spec) for spec in self._specs),
        )

    @property
    def specs(self) -> tuple[GalaxyAttentionKVSpec, ...]:
        return self._specs

    @property
    def model_args(self) -> Any:
        return getattr(self._model, "model_args", None)

    def set_kv_cache(self, cache: Any) -> None:
        self._model.set_kv_cache(cache)
