# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Zero-centred RMSNorm: ``x * rsqrt(mean(x^2) + eps) * (1 + w)``. The ``+1`` is folded at setup."""

from __future__ import annotations

from dataclasses import dataclass, replace

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.common.modules.lazy_weight import LazyWeight
from models.demos.pplx_decider_v1_27b.tt.common import resolve
from models.demos.pplx_decider_v1_27b.tt.optimizations import NormOptimizations


@dataclass
class RMSNormConfig:
    weight: LazyWeight  # (1 + w), [1, 1, D]
    eps: float
    mesh_device: object | None = None
    compute_kernel_cfg: object | None = None
    output_memcfg: ttnn.MemoryConfig | None = None


class PplxRMSNorm(LightweightModule):
    def __init__(self, weight: LazyWeight, eps: float, optimizations: NormOptimizations | None = None):
        super().__init__()
        config = RMSNormConfig(weight=weight, eps=eps)
        if optimizations is not None:
            config = replace(
                config,
                compute_kernel_cfg=optimizations.compute_kernel_cfg,
                output_memcfg=optimizations.output_memcfg,
            )
        self.config = _resolve(config)
        self.weight = None
        self.norm_opts = optimizations
        self._sharded_cache = {}

    def _sharded_configs(self, x: ttnn.Tensor):
        """(block-sharded L1 memory config, sharded program config) for short inputs, else None."""
        opts = getattr(self, "norm_opts", None)
        if opts is None or not opts.sharded_max_rows or len(x.shape) != 3 or x.shape[0] != 1:
            return None
        rows, width = x.shape[-2], x.shape[-1]
        m_tiles, w_tiles = x.padded_shape[-2] // 32, x.padded_shape[-1] // 32
        if rows > opts.sharded_max_rows or width != x.padded_shape[-1] or w_tiles % opts.sharded_grid_x:
            return None
        key = (m_tiles, w_tiles, opts.sharded_grid_x, opts.sharded_grid_y)
        if key not in self._sharded_cache:
            gx = opts.sharded_grid_x
            gy = max(d for d in range(1, opts.sharded_grid_y + 1) if m_tiles % d == 0)
            block_h, block_w = m_tiles // gy, w_tiles // gx
            memcfg = ttnn.create_sharded_memory_config(
                shape=(block_h * 32, block_w * 32),
                core_grid=ttnn.CoreGrid(x=gx, y=gy),
                strategy=ttnn.ShardStrategy.BLOCK,
                orientation=ttnn.ShardOrientation.ROW_MAJOR,
                use_height_and_width_as_shard_shape=True,
            )
            program_config = ttnn.LayerNormShardedMultiCoreProgramConfig(
                compute_with_storage_grid_size=(gx, gy),
                subblock_w=max(d for d in range(1, 5) if block_w % d == 0),
                block_h=block_h,
                block_w=block_w,
                inplace=False,
            )
            self._sharded_cache[key] = (memcfg, program_config)
        return self._sharded_cache[key]

    @classmethod
    def from_config(cls, config: RMSNormConfig) -> "PplxRMSNorm":
        instance = object.__new__(cls)
        LightweightModule.__init__(instance)
        instance.config = _resolve(config)
        instance.weight = None
        instance.norm_opts = None
        instance._sharded_cache = {}
        return instance

    def load_device_weights(self) -> None:
        if self.weight is None:
            self.weight = self.config.weight.get_device_weight()

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        self.load_device_weights()
        sharded = self._sharded_configs(x)
        if sharded is not None:
            memcfg, program_config = sharded
            x_l1 = ttnn.to_memory_config(x, memcfg)
            y = ttnn.rms_norm(
                x_l1,
                weight=self.weight,
                epsilon=self.config.eps,
                program_config=program_config,
                compute_kernel_config=self.config.compute_kernel_cfg,
                memory_config=memcfg,
            )
            ttnn.deallocate(x_l1)
            out = ttnn.to_memory_config(y, self.config.output_memcfg)
            ttnn.deallocate(y)
            return out
        return ttnn.rms_norm(
            x,
            weight=self.weight,
            epsilon=self.config.eps,
            compute_kernel_config=self.config.compute_kernel_cfg,
            memory_config=self.config.output_memcfg,
        )


def _resolve(config: RMSNormConfig) -> RMSNormConfig:
    device = config.mesh_device or config.weight.device or ttnn.GetDefaultDevice()
    if device is None:
        raise ValueError("PplxRMSNorm needs a device")
    return replace(
        config,
        mesh_device=device,
        weight=resolve(config.weight, device),
        compute_kernel_cfg=config.compute_kernel_cfg
        or ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=True
        ),
        output_memcfg=config.output_memcfg or ttnn.DRAM_MEMORY_CONFIG,
    )
