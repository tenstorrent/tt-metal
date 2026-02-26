# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Low-level helpers for distributed (TP/DP) tensor creation.

All helpers use bfloat16 to avoid OOM (float32 doubles memory).
"""

import numpy as np
import torch
import ttnn
import ttml

from utils.context_managers import is_empty_init
from utils.tensor_utils import (
    get_device,
    get_tp_size,
    tile_pad as _tile_pad,
    make_empty_on_device as _make_empty_on_device,
    make_weight as _make_weight,
    make_ones as _make_ones,
    make_zeros as _make_zeros,
)


def _make_sharded_weight(shape, shard_dim_tensor, shard_dim_mesh=None, std=0.02):
    """Create a sharded bfloat16 parameter tensor."""
    if is_empty_init():
        per_device = list(shape)
        per_device[shard_dim_tensor] //= get_tp_size(shard_dim_mesh)
        return _make_empty_on_device(per_device)
    device = get_device()
    data = (torch.randn(shape) * std).float().numpy()
    mapper = ttml.core.distributed.shard_tensor_to_mesh_mapper(
        device, shard_dim_tensor, shard_dim_mesh
    )
    return ttml.autograd.Tensor.from_numpy(
        data, ttnn.Layout.TILE, ttnn.bfloat16, mapper
    )


def _make_sharded_zeros(shape, shard_dim_tensor, shard_dim_mesh=None):
    """Create a sharded zero bfloat16 tensor."""
    if is_empty_init():
        per_device = list(shape)
        per_device[shard_dim_tensor] //= get_tp_size(shard_dim_mesh)
        return _make_empty_on_device(per_device)
    device = get_device()
    data = np.zeros(shape, dtype=np.float32)
    mapper = ttml.core.distributed.shard_tensor_to_mesh_mapper(
        device, shard_dim_tensor, shard_dim_mesh
    )
    return ttml.autograd.Tensor.from_numpy(
        data, ttnn.Layout.TILE, ttnn.bfloat16, mapper
    )


def _make_replicated(data_np):
    """Create a replicated bfloat16 ttml tensor from numpy."""
    return ttml.autograd.Tensor.from_numpy(data_np, ttnn.Layout.TILE, ttnn.bfloat16)


def _make_replicated_ones(shape):
    if is_empty_init():
        return _make_empty_on_device(shape)
    return _make_replicated(np.ones(shape, dtype=np.float32))


def _make_replicated_zeros(shape):
    if is_empty_init():
        return _make_empty_on_device(shape)
    return _make_replicated(np.zeros(shape, dtype=np.float32))


def _make_replicated_weight(shape, std=0.02):
    if is_empty_init():
        return _make_empty_on_device(shape)
    data = (torch.randn(shape) * std).float().numpy()
    return _make_replicated(data)
