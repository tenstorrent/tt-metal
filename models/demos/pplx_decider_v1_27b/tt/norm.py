# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Zero-centred RMSNorm: ``x * rsqrt(mean(x^2) + eps) * (1 + w)``. The ``+1`` is folded at setup."""

from __future__ import annotations

from dataclasses import dataclass, replace

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.common.modules.lazy_weight import LazyWeight
from models.demos.pplx_decider_v1_27b.tt.common import concat_owned, resolve
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

    def _sharded_configs(self, x: ttnn.Tensor, row_blocks: int = 1):
        """(block-sharded L1 memory config, sharded program config) for short inputs, else None.

        ``row_blocks`` > 1: x holds that many requests of equal length, one per core row (grid
        ``sharded_grid_x`` x ``row_blocks``), so every core gets the block of a batch-1 request."""
        opts = getattr(self, "norm_opts", None)
        if opts is None or not opts.sharded_max_rows or len(x.shape) != 3 or x.shape[0] != 1:
            return None
        rows, width = x.shape[-2] // row_blocks, x.shape[-1]
        m_tiles, w_tiles = x.padded_shape[-2] // 32, x.padded_shape[-1] // 32
        if rows > opts.sharded_max_rows or width != x.padded_shape[-1] or w_tiles % opts.sharded_grid_x:
            return None
        key = (m_tiles, w_tiles, opts.sharded_grid_x, opts.sharded_grid_y, row_blocks)
        if key not in self._sharded_cache:
            gx = opts.sharded_grid_x
            gy = row_blocks if row_blocks > 1 else max(d for d in range(1, opts.sharded_grid_y + 1) if m_tiles % d == 0)
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
        if len(x.shape) == 3 and x.shape[0] > 1:
            return self._forward_batch(x)
        return self._forward_one(x, self._sharded_configs(x))

    def _forward_batch(self, x: ttnn.Tensor) -> ttnn.Tensor:
        """A batch [B, S, D]. The norm is row-wise. Long rows: one interleaved norm on the [1, B*S, D] view
        (as batch 1). Short rows (batch 1 runs the block-sharded norm): groups of up to the grid height of
        requests, one request per core row, so each core runs batch 1's block."""
        b, s, d = x.shape
        flat = ttnn.reshape(x, [1, b * s, d])
        opts = getattr(self, "norm_opts", None)
        short = opts is not None and opts.sharded_max_rows and s <= opts.sharded_max_rows
        if not short:
            out = self._forward_one(flat, None)
            return ttnn.reshape(out, [b, s, out.shape[-1]])
        group = max(1, min(b, self.config.mesh_device.compute_with_storage_grid_size().y))
        outs = []
        for start in range(0, b, group):
            g = min(group, b - start)
            part = flat if g == b else ttnn.slice(flat, [0, start * s, 0], [1, (start + g) * s, d])
            out = self._forward_one(
                part, self._sharded_configs(part, row_blocks=g) if g > 1 else self._sharded_configs(part)
            )
            if part is not flat:
                ttnn.deallocate(part)
            outs.append(ttnn.reshape(out, [g, s, out.shape[-1]]))
        return concat_owned(outs, dim=0)

    def _forward_one(self, x: ttnn.Tensor, sharded) -> ttnn.Tensor:
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
