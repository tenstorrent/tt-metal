# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Token embedding on device: ``ttnn.embedding`` over the BF16 [248320, 5120] ROW_MAJOR table."""

from __future__ import annotations

from dataclasses import dataclass, replace

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.common.modules.lazy_weight import LazyWeight
from models.demos.pplx_decider_v1_27b.tt.common import resolve


@dataclass
class EmbeddingConfig:
    weight: LazyWeight
    mesh_device: object | None = None
    output_memcfg: ttnn.MemoryConfig | None = None


class PplxEmbedding(LightweightModule):
    def __init__(self, weight: LazyWeight, mesh_device=None):
        super().__init__()
        self.config = _resolve(EmbeddingConfig(weight=weight, mesh_device=mesh_device))
        self.weight = None

    @classmethod
    def from_config(cls, config: EmbeddingConfig) -> "PplxEmbedding":
        instance = object.__new__(cls)
        LightweightModule.__init__(instance)
        instance.config = _resolve(config)
        instance.weight = None
        return instance

    def load_device_weights(self) -> None:
        if self.weight is None:
            self.weight = self.config.weight.get_device_weight()

    def forward(self, token_ids: ttnn.Tensor) -> ttnn.Tensor:
        """token_ids: [1, S] UINT32 ROW_MAJOR on device -> [1, S, 5120] BF16 TILE."""
        self.load_device_weights()
        out = ttnn.embedding(token_ids, self.weight, layout=ttnn.TILE_LAYOUT, memory_config=self.config.output_memcfg)
        return ttnn.reshape(out, [token_ids.shape[0], token_ids.shape[-1], self.weight.shape[-1]])


def _resolve(config: EmbeddingConfig) -> EmbeddingConfig:
    device = config.mesh_device or config.weight.device or ttnn.GetDefaultDevice()
    return replace(
        config,
        mesh_device=device,
        weight=resolve(config.weight, device, layout=ttnn.ROW_MAJOR_LAYOUT),
        output_memcfg=config.output_memcfg or ttnn.DRAM_MEMORY_CONFIG,
    )
