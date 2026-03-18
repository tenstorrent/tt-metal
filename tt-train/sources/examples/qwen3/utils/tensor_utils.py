# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Tensor creation and extraction helpers (always real — bypass empty_init)."""

import numpy as np
import torch
import ttnn
import ttml

from utils.context_managers import is_empty_init


def get_device():
    return ttml.autograd.AutoContext.get_instance().get_device()


# ------------------------------------------------------------------
# Loss / logits extraction
# ------------------------------------------------------------------


def get_loss_value(loss, distributed=False):
    """Extract a scalar loss value from a ttml Tensor."""
    if distributed:
        device = get_device()
        composer = ttml.core.distributed.concat_mesh_to_tensor_composer(device, 0)
        loss_np = loss.to_numpy(composer=composer)
        return float(loss_np.mean())
    return float(loss.to_numpy().mean())


def extract_logits(logits, distributed=False):
    """Bring logits back to CPU as float32 numpy.

    For distributed tensors the first device-replica is returned.
    """
    if distributed:
        device = get_device()
        composer = ttml.core.distributed.concat_mesh_to_tensor_composer(device, 0)
        out = logits.to_numpy(composer=composer).astype(np.float32)
        return out[:1]
    return logits.to_numpy().astype(np.float32)


def get_tp_size(shard_dim=None):
    ctx = ttml.autograd.AutoContext.get_instance()
    pctx = ctx.get_parallelism_context()
    return pctx.get_tp_size()


def tile_pad(dim: int) -> int:
    """Round *dim* up to the nearest multiple of 32 (tile alignment)."""
    return ((dim + 31) // 32) * 32


def torch_to_ttml(t: torch.Tensor):
    """Convert torch.Tensor → ttml.autograd.Tensor (bfloat16 on device)."""
    device = get_device()
    ttnn_host = ttnn.from_torch(t, dtype=ttnn.bfloat16)
    ttnn_dev = ttnn.to_device(ttnn_host, device)
    ttnn_tiled = ttnn.tilize_with_zero_padding(ttnn_dev)
    return ttml.autograd.create_tensor(ttnn_tiled)


def make_weight(shape, std=0.02):
    if is_empty_init():
        return make_empty_on_device(shape)
    return torch_to_ttml(torch.randn(shape) * std)


def make_ones(shape):
    if is_empty_init():
        return make_empty_on_device(shape)
    return torch_to_ttml(torch.ones(shape))


def make_zeros(shape):
    if is_empty_init():
        return make_empty_on_device(shape)
    return torch_to_ttml(torch.zeros(shape))


def make_empty_on_device(shape):
    """Allocate an empty bfloat16 TILE tensor directly on device.

    Skips CPU tensor creation, host-to-device copy, and tilization entirely.
    The last two dims are padded to tile alignment (multiples of 32).
    """
    device = get_device()
    padded = list(shape)
    padded[-2] = tile_pad(padded[-2])
    padded[-1] = tile_pad(padded[-1])
    ttnn_tensor = ttnn.empty(padded, ttnn.bfloat16, ttnn.TILE_LAYOUT, device)
    return ttml.autograd.create_tensor(ttnn_tensor)


def make_dist_replicated(data_np):
    return ttml.autograd.Tensor.from_numpy(data_np, ttnn.Layout.TILE, ttnn.bfloat16)


def make_dist_replicated_weight(shape, std):
    data = (torch.randn(shape) * std).float().numpy()
    return make_dist_replicated(data)


def make_dist_replicated_zeros(shape):
    return make_dist_replicated(np.zeros(shape, dtype=np.float32))


def make_dist_sharded(data_np, shard_dim_tensor, shard_dim_mesh):
    device = get_device()
    mapper = ttml.core.distributed.shard_tensor_to_mesh_mapper(device, shard_dim_tensor, shard_dim_mesh)
    return ttml.autograd.Tensor.from_numpy(data_np, ttnn.Layout.TILE, ttnn.bfloat16, mapper)


def make_dist_sharded_weight(shape, shard_dim_tensor, shard_dim_mesh, std):
    data = (torch.randn(shape) * std).float().numpy()
    return make_dist_sharded(data, shard_dim_tensor, shard_dim_mesh)


def make_dist_sharded_zeros(shape, shard_dim_tensor, shard_dim_mesh):
    return make_dist_sharded(np.zeros(shape, dtype=np.float32), shard_dim_tensor, shard_dim_mesh)


def make_sharded_weight(shape, shard_dim_tensor, shard_dim_mesh=None, std=0.02):
    """Create a sharded bfloat16 parameter tensor."""
    if is_empty_init():
        per_device = list(shape)
        per_device[shard_dim_tensor] //= get_tp_size(shard_dim_mesh)
        return make_empty_on_device(per_device)
    device = get_device()
    data = (torch.randn(shape) * std).float().numpy()
    mapper = ttml.core.distributed.shard_tensor_to_mesh_mapper(device, shard_dim_tensor, shard_dim_mesh)
    return ttml.autograd.Tensor.from_numpy(data, ttnn.Layout.TILE, ttnn.bfloat16, mapper)


def make_sharded_zeros(shape, shard_dim_tensor, shard_dim_mesh=None):
    """Create a sharded zero bfloat16 tensor."""
    if is_empty_init():
        per_device = list(shape)
        per_device[shard_dim_tensor] //= get_tp_size(shard_dim_mesh)
        return make_empty_on_device(per_device)
    device = get_device()
    data = np.zeros(shape, dtype=np.float32)
    mapper = ttml.core.distributed.shard_tensor_to_mesh_mapper(device, shard_dim_tensor, shard_dim_mesh)
    return ttml.autograd.Tensor.from_numpy(data, ttnn.Layout.TILE, ttnn.bfloat16, mapper)


def make_replicated_ones(shape):
    if is_empty_init():
        return make_empty_on_device(shape)
    return make_dist_replicated(np.ones(shape, dtype=np.float32))


def make_replicated_zeros(shape):
    if is_empty_init():
        return make_empty_on_device(shape)
    return make_dist_replicated(np.zeros(shape, dtype=np.float32))


def make_replicated_weight(shape, std=0.02):
    if is_empty_init():
        return make_empty_on_device(shape)
    data = (torch.randn(shape) * std).float().numpy()
    return make_dist_replicated(data)


def gather_mesh_to_cpu(ttnn_val, device, distributed, dp_size, tp_size, per_device_batch):
    """Bring a multi-device tensor to CPU, selecting one copy per sample.

    After concat along dim 0, layout is blocks of per_device_batch rows per
    device in mesh order: (dp0,tp0), (dp0,tp1), ..., (dp1,tp0), ...
    TP replicas within a DP group are identical, so we keep only tp-rank 0.
    """
    if not distributed:
        return ttnn.to_torch(ttnn_val)
    composer = ttml.core.distributed.concat_mesh_to_tensor_composer(device, 0)
    cpu = ttnn.to_torch(ttnn_val, mesh_composer=composer)
    if dp_size > 1:
        rest = cpu.shape[1:]
        cpu = cpu.reshape(dp_size, tp_size, per_device_batch, *rest)
        return cpu[:, 0].reshape(dp_size * per_device_batch, *rest)
    return cpu[:per_device_batch]


def create_input_tensor_from_torch(token_ids, device):
    """UINT32 ROW_MAJOR input tensor on device (replicated across mesh)."""
    host = ttnn.from_torch(token_ids.to(torch.int32), dtype=ttnn.uint32)
    dev = ttnn.to_device(host, device)
    return ttml.autograd.create_tensor(dev, requires_grad=False)


def create_input_tensor_dp(token_ids_np, device):
    """DP-sharded UINT32 input (batch dim sharded across mesh dim 0)."""
    mapper = ttml.core.distributed.shard_tensor_to_mesh_mapper(device, 0, 0)
    return ttml.autograd.Tensor.from_numpy(
        token_ids_np.astype(np.uint32),
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.UINT32,
        mapper,
    )


def create_causal_mask(seq_len):
    """Create causal attention mask as a ttml Tensor (numpy path)."""
    mask = np.tril(np.ones((1, 1, seq_len, seq_len), dtype=np.float32))
    return ttml.autograd.Tensor.from_numpy(mask, ttnn.Layout.TILE, ttnn.DataType.BFLOAT16)


def create_input_tensor(x_np, dp_mapper=None):
    """Create UINT32 ROW_MAJOR input tensor from numpy [B,1,1,T].

    When dp_mapper is provided (DP mode), shards batch dim across DP groups.
    Otherwise replicates across all devices.
    """
    return ttml.autograd.Tensor.from_numpy(
        x_np,
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.UINT32,
        dp_mapper,
    )


def create_target_tensor(y_np, dp_mapper=None):
    """Create UINT32 ROW_MAJOR target tensor from numpy [B,T].

    When dp_mapper is provided (DP mode), shards batch dim across DP groups.
    Otherwise replicates across all devices.
    """
    return ttml.autograd.Tensor.from_numpy(
        y_np,
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.UINT32,
        dp_mapper,
    )
