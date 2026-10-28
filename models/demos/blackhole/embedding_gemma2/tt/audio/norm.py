# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Audio conformer RMSNorm with the precision policy's norm options (E6b).

`AudioRMSNorm` is the shared `RMSNorm` (compute config from `audio.norms`) plus the block-sharded
round trip of E5c c21 when `audio.norms.sharded` is true:

The interleaved `ttnn.rms_norm` parallelises over 32-row tile rows only: [S, 1024] runs on S / 32
cores, so a 256-token forward normalises on 8 cores and takes as long as a 1024-token one (18 us,
E5c `norm_micro.py`). Up to SHARDED_NORM_MAX_TILE_ROWS tile rows the sharded path moves the tensor
to a block-sharded layout (gx x gy cores, the hidden dim split gx ways), runs the sharded multi-core
RMSNorm (same compute config, same weight) and moves it back: 11.4 vs 18.1 us at 256 rows,
15.8 vs 18.4 us at 768 (round trip included). Larger inputs use the interleaved op. Results differ
from the interleaved op by the reduction order (E5c: soft PCC s30 0.99900 -> 0.99823).

Contract
    input   [B, 1, S, D] TILE (bf16 or fp32), DRAM or L1 interleaved
    output  same shape / dtype, `memory_config` interleaved
"""

from __future__ import annotations

from pathlib import Path

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.demos.blackhole.embedding_gemma2.tt.config import RMSNormConfig
from models.demos.blackhole.embedding_gemma2.tt.precision import AudioNormPolicy
from models.demos.blackhole.embedding_gemma2.tt.rms_norm import RMSNorm

SHARDED_NORM_MAX_TILE_ROWS = 24


def _largest_divisor(n: int, limit: int) -> int:
    return max(d for d in range(1, limit + 1) if n % d == 0)


def sharded_rms_norm(norm: RMSNorm, x: ttnn.Tensor, memory_config) -> ttnn.Tensor | None:
    """norm(x) through the block-sharded op, or None when x is too large / not shardable."""
    shape = x.shape
    M = shape[0] * shape[1] * shape[2]
    D = shape[3]
    Mt, Dt = M // ttnn.TILE_SIZE, D // ttnn.TILE_SIZE
    if Mt > SHARDED_NORM_MAX_TILE_ROWS or x.is_sharded():
        return None
    gx = 8 if Mt > 8 else 4
    if Dt % gx != 0:
        return None
    gy = _largest_divisor(Mt, 8)
    shard = ttnn.create_sharded_memory_config(
        (M // gy, D // gx),
        core_grid=ttnn.CoreGrid(y=gy, x=gx),
        strategy=ttnn.ShardStrategy.BLOCK,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )
    block_w = Dt // gx
    pc = ttnn.LayerNormShardedMultiCoreProgramConfig(
        compute_with_storage_grid_size=(gx, gy),
        subblock_w=_largest_divisor(block_w, 4),
        block_h=Mt // gy,
        block_w=block_w,
        inplace=False,
    )
    x2 = ttnn.reshape(x, (1, 1, M, D)) if M != shape[2] else x
    xs = ttnn.to_memory_config(x2, shard)
    ys = ttnn.rms_norm(
        xs,
        epsilon=norm.config.eps,
        weight=norm.weight,
        program_config=pc,
        memory_config=shard,
        compute_kernel_config=norm.compute_kernel_config,
    )
    ttnn.deallocate(xs)
    y = ttnn.to_memory_config(ys, memory_config or ttnn.DRAM_MEMORY_CONFIG)
    ttnn.deallocate(ys)
    if M != shape[2]:
        y = ttnn.reshape(y, tuple(shape))
    return y


class AudioRMSNorm(LightweightModule):
    def __init__(self, norm: RMSNorm, sharded: bool):
        super().__init__()
        self.norm = norm
        self.sharded = sharded
        self.config = norm.config
        self.weight = norm.weight
        self.compute_kernel_config = norm.compute_kernel_config

    @classmethod
    def from_state_dict(
        cls,
        state_dict: dict[str, torch.Tensor] | None,
        config: RMSNormConfig,
        device,
        policy: AudioNormPolicy,
        *,
        cache_dir: Path | None = None,
        name: str = "audio.norm",
    ) -> "AudioRMSNorm":
        norm = RMSNorm.from_state_dict(
            state_dict,
            config,
            device,
            cache_dir=cache_dir,
            name=name,
            compute_kernel_config=policy.compute_kernel_config,
        )
        return cls(norm, policy.sharded)

    def forward(self, x: ttnn.Tensor, memory_config=None) -> ttnn.Tensor:
        if self.sharded:
            y = sharded_rms_norm(self.norm, x, memory_config)
            if y is not None:
                return y
        return self.norm(x, memory_config=memory_config)
