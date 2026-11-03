# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""DBCache (cache-dit) on the FLUX 3 Klein single-stream blocks: the first Fn blocks always run, and when their
residual barely moved since the last computed step the remaining blocks are replaced by their cached residual."""

from __future__ import annotations

import functools

import torch

import ttnn

from ...utils import tensor
from ...utils.dbcache import DBCacheConfig, DBCacheContext


def flux3_klein_dbcache_config(num_references: int = 0) -> DBCacheConfig:
    return DBCacheConfig(
        Fn_compute_blocks=4,
        Bn_compute_blocks=0,
        residual_diff_threshold=0.30,
        max_warmup_steps=8 if num_references else 4,
        max_cached_steps=-1,
        max_continuous_cached_steps=-1,
    )


class Flux3KleinDBCache:
    def __init__(self, config: DBCacheConfig, device: ttnn.MeshDevice, *, num_branches: int) -> None:
        assert config.taylorseer_order == 0 and config.steps_computation_mask is None
        self.context = DBCacheContext(config, name="flux3_klein", num_branches=num_branches)
        self._device = device
        self._fn_residual = [None] * num_branches
        self._residual = [None] * num_branches
        self._masks: dict[tuple, ttnn.Tensor] = {}

    def runner(self, branch: int):
        return functools.partial(self.run, branch=branch)

    def _content_mask(self, layout, sp_axis: int) -> ttnn.Tensor:
        # the diff is measured on the image and reference rows only, as cache-dit's hook on the image stream does
        content = layout.lengths["image"] + layout.lengths["image_cond"]
        key = (layout.total_padded, content, sp_axis)
        if key not in self._masks:
            mask = torch.zeros(1, layout.total_padded, 1)
            mask[:, :content] = 1.0
            self._masks[key] = tensor.from_torch(
                mask, device=self._device, dtype=ttnn.bfloat16, mesh_axes=[None, sp_axis, None]
            )
        return self._masks[key]

    @staticmethod
    def _abs_sum(t: ttnn.Tensor) -> float:
        a = ttnn.abs(t)
        a32 = ttnn.typecast(a, ttnn.float32)
        ttnn.deallocate(a)
        s = ttnn.sum(a32)
        ttnn.deallocate(a32)
        total = sum(float(ttnn.to_torch(d).double().sum().item()) for d in ttnn.get_device_tensors(s))
        ttnn.deallocate(s)
        return total

    def _relative_diff(self, fn_residual: ttnn.Tensor, previous: ttnn.Tensor) -> float:
        d = ttnn.subtract(fn_residual, previous)
        num = self._abs_sum(d)
        ttnn.deallocate(d)
        den = self._abs_sum(previous)
        return num / den if den > 0 else float("inf")

    def run(self, blocks, h: ttnn.Tensor, run_block, *, layout, sp_axis: int, branch: int) -> ttnn.Tensor:
        config = self.context.config
        n = len(blocks)
        fn = min(config.Fn_compute_blocks, n)
        bn = min(config.Bn_compute_blocks, n - fn)

        h_in = h
        for block in blocks[:fn]:
            h = run_block(block, h)
        fn_residual = ttnn.subtract(h, h_in)
        if layout.lengths["image"] + layout.lengths["image_cond"] < layout.total_padded:
            masked = ttnn.multiply(fn_residual, self._content_mask(layout, sp_axis))
            ttnn.deallocate(fn_residual)
            fn_residual = masked

        gate = self.context.gate(branch)
        if gate == "dynamic":
            use_cache = self.context.decide(branch, self._relative_diff(fn_residual, self._fn_residual[branch]))
        else:
            use_cache = gate == "cache"

        if use_cache:
            ttnn.deallocate(fn_residual)
            self.context.add_cached_step(branch)
            h = ttnn.add(h, self._residual[branch])
        else:
            if self._fn_residual[branch] is not None:
                ttnn.deallocate(self._fn_residual[branch])
            self._fn_residual[branch] = fn_residual
            h_fn = ttnn.clone(h)
            for block in blocks[fn : n - bn]:
                h = run_block(block, h)
            residual = ttnn.subtract(h, h_fn)
            ttnn.deallocate(h_fn)
            if self._residual[branch] is not None:
                ttnn.deallocate(self._residual[branch])
            self._residual[branch] = residual
            self.context.add_computed_step(branch)

        for block in blocks[n - bn :]:
            h = run_block(block, h)
        return h

    def close(self) -> None:
        for t in (*self._fn_residual, *self._residual, *self._masks.values()):
            if t is not None:
                ttnn.deallocate(t)
