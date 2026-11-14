# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""DBCache (cache-dit) on the FLUX 3 Klein single-stream blocks, run by ``utils.dbcache_runner.DBCache``: the first Fn
blocks always run, and when their residual barely moved since the last computed step the remaining blocks are replaced
by their cached residual."""

from __future__ import annotations

import torch

import ttnn

from ...utils import tensor
from ...utils.dbcache import DBCacheConfig


def flux3_klein_dbcache_config(num_references: int = 0) -> DBCacheConfig:
    return DBCacheConfig(
        Fn_compute_blocks=4,
        Bn_compute_blocks=0,
        residual_diff_threshold=0.30,
        max_warmup_steps=8 if num_references else 4,
        max_cached_steps=-1,
        max_continuous_cached_steps=-1,
    )


class Flux3KleinCachedSample:
    """One sample of a request as a ``CachedTransformer``; the step's inputs are the prologue's arguments."""

    def __init__(self, transformer, **request_inputs) -> None:
        self._transformer = transformer
        self._request_inputs = request_inputs
        self.num_blocks = transformer.num_blocks
        self.run_blocks = transformer.run_blocks
        self.epilogue = transformer.epilogue

    def prologue(self, **step_inputs):
        return self._transformer.prologue(**self._request_inputs, **step_inputs)


def content_mask(layout, device: ttnn.MeshDevice, sp_axis: int) -> ttnn.Tensor:
    """1 on the image and reference rows, which come first in the single stage; the difference skips text and padding."""
    mask = torch.zeros(1, layout.total_padded, 1)
    mask[:, : layout.attended("image") + layout.attended("image_cond")] = 1.0
    return tensor.from_torch(mask, device=device, dtype=ttnn.bfloat16, mesh_axes=[None, sp_axis, None])
