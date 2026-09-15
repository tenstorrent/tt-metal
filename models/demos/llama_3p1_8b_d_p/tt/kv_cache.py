# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Packed, block-cyclic Llama-3.1 K/V cache for Galaxy prefill."""

from dataclasses import dataclass

import torch

import ttnn
from models.demos.common.prefill.adapter import KvCaches
from models.demos.common.prefill.runners.migration import get_num_dram_banks
from models.demos.llama_3p1_8b_d_p.reference.llama_3p1_8b_config import Llama31_8BConfig

_MESH_SHAPE = (4, 8)
_SP = 4
_TP = 8
_SP_AXIS = 0
_TP_AXIS = 1
_NUM_USERS = 2
_NUM_LAYERS = Llama31_8BConfig.NUM_LAYERS
_MAX_SEQ_LEN = 2048
_GLOBAL_CHUNK = 1024
_LOCAL_CHUNK = _GLOBAL_CHUNK // _SP
_LOCAL_CACHE_SEQUENCE = _MAX_SEQ_LEN // _SP
_HEAD_DIM = Llama31_8BConfig.HEAD_DIM
_CACHE_SHAPE = (_NUM_USERS * _NUM_LAYERS, 1, _LOCAL_CACHE_SEQUENCE, _HEAD_DIM)
_DRAM_PAGE_TOKENS = 32
_SUPPORTED_DTYPES = (ttnn.bfloat16, ttnn.bfloat8_b)


@dataclass
class LlamaKVCache(KvCaches):
    """Externally owned, user-major Llama K/V caches."""

    k: ttnn.Tensor
    v: ttnn.Tensor
    num_users: int
    num_layers: int
    max_seq_len: int
    sp: int


def _validate_target(mesh_device, mesh_config, *, num_users, num_layers, max_seq_len, cache_dtype):
    required = ("mesh_shape", "tp", "tp_axis", "sp_axis", "sp")
    missing = [name for name in required if not hasattr(mesh_config, name)]
    if missing:
        raise ValueError(f"Llama KV cache mesh_config is missing: {', '.join(missing)}")
    if tuple(mesh_config.mesh_shape) != _MESH_SHAPE:
        raise ValueError(f"Llama KV cache requires mesh_shape={_MESH_SHAPE}, got {tuple(mesh_config.mesh_shape)}")
    if (mesh_config.sp, mesh_config.tp, mesh_config.sp_axis, mesh_config.tp_axis) != (
        _SP,
        _TP,
        _SP_AXIS,
        _TP_AXIS,
    ):
        raise ValueError(
            "Llama KV cache requires SP=4 on mesh axis 0 and TP=8 on mesh axis 1; "
            f"got SP={mesh_config.sp}, TP={mesh_config.tp}, "
            f"sp_axis={mesh_config.sp_axis}, tp_axis={mesh_config.tp_axis}"
        )
    if tuple(mesh_device.shape) != _MESH_SHAPE or mesh_device.get_num_devices() != _SP * _TP:
        raise ValueError(
            f"Llama KV cache device requires {_MESH_SHAPE} with {_SP * _TP} chips; "
            f"got shape={tuple(mesh_device.shape)}, devices={mesh_device.get_num_devices()}"
        )
    if type(num_users) is not int or num_users != _NUM_USERS:
        raise ValueError(f"Llama KV cache requires num_users=2, got {num_users!r}")
    if type(num_layers) is not int or num_layers != _NUM_LAYERS:
        raise ValueError(f"Llama KV cache requires num_layers=32, got {num_layers!r}")
    if type(max_seq_len) is not int or max_seq_len != _MAX_SEQ_LEN:
        raise ValueError(f"Llama KV cache requires max_seq_len=2048, got {max_seq_len!r}")
    if cache_dtype not in _SUPPORTED_DTYPES:
        raise ValueError(f"Llama KV cache dtype must be bfloat16 or bfloat8_b, got {cache_dtype}")


def _cache_memory_config(mesh_device):
    bank_grid = ttnn.CoreRangeSet(
        [
            ttnn.CoreRange(ttnn.CoreCoord(bank_id, 0), ttnn.CoreCoord(bank_id, 0))
            for bank_id in range(get_num_dram_banks(mesh_device))
        ]
    )
    return ttnn.MemoryConfig(
        buffer_type=ttnn.BufferType.DRAM,
        nd_shard_spec=ttnn.NdShardSpec(
            shard_shape=[1, 1, _DRAM_PAGE_TOKENS, _HEAD_DIM],
            grid=bank_grid,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            shard_distribution_strategy=ttnn.ShardDistributionStrategy.ROUND_ROBIN_1D,
        ),
    )


def allocate_kv_cache(
    mesh_device,
    mesh_config,
    *,
    num_users=2,
    num_layers=32,
    max_seq_len=2048,
    cache_dtype=ttnn.bfloat8_b,
):
    """Allocate zeroed K/V caches in the fixed Llama migration layout."""
    _validate_target(
        mesh_device,
        mesh_config,
        num_users=num_users,
        num_layers=num_layers,
        max_seq_len=max_seq_len,
        cache_dtype=cache_dtype,
    )
    memory_config = _cache_memory_config(mesh_device)

    def allocate_one():
        return ttnn.from_torch(
            torch.zeros(_CACHE_SHAPE),
            device=mesh_device,
            dtype=cache_dtype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=memory_config,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )

    return LlamaKVCache(
        k=allocate_one(),
        v=allocate_one(),
        num_users=num_users,
        num_layers=num_layers,
        max_seq_len=max_seq_len,
        sp=_SP,
    )


def _validate_scalar(name, value):
    if type(value) is not int:
        raise TypeError(f"{name} must be an eager Python int, got {type(value).__name__}")


def _validate_cache_tensor(name, tensor, mesh_device):
    if not isinstance(tensor, ttnn.Tensor) or not ttnn.is_tensor_storage_on_device(tensor):
        raise ValueError(f"Llama KV cache {name} must be a device ttnn.Tensor")
    if tensor.device() != mesh_device:
        raise ValueError(f"Llama KV cache {name} must reside on the input mesh")
    if tuple(tensor.shape) != _CACHE_SHAPE:
        raise ValueError(f"Llama KV cache {name} must have local shape {_CACHE_SHAPE}, got {tuple(tensor.shape)}")
    if tensor.dtype not in _SUPPORTED_DTYPES:
        raise ValueError(f"Llama KV cache {name} has unsupported dtype {tensor.dtype}")
    if tensor.layout != ttnn.TILE_LAYOUT:
        raise ValueError(f"Llama KV cache {name} must use TILE_LAYOUT, got {tensor.layout}")
    memory_config = tensor.memory_config()
    shard_spec = memory_config.nd_shard_spec
    if memory_config.buffer_type != ttnn.BufferType.DRAM or shard_spec is None:
        raise ValueError(f"Llama KV cache {name} must use NdShard DRAM")
    if tuple(shard_spec.shard_shape) != (1, 1, _DRAM_PAGE_TOKENS, _HEAD_DIM):
        raise ValueError(f"Llama KV cache {name} must use shard [1,1,32,128], got {tuple(shard_spec.shard_shape)}")
    if shard_spec.orientation != ttnn.ShardOrientation.ROW_MAJOR:
        raise ValueError(f"Llama KV cache {name} must use ROW_MAJOR shard orientation")
    if shard_spec.shard_distribution_strategy != ttnn.ShardDistributionStrategy.ROUND_ROBIN_1D:
        raise ValueError(f"Llama KV cache {name} must use ROUND_ROBIN_1D shard distribution")
    if len(ttnn.get_device_tensors(tensor)) != _SP * _TP:
        raise ValueError(f"Llama KV cache {name} must cover {_SP * _TP} mesh devices")


def _validate_input(name, tensor, mesh_device):
    if not isinstance(tensor, ttnn.Tensor) or not ttnn.is_tensor_storage_on_device(tensor):
        raise ValueError(f"{name} input must be a device ttnn.Tensor")
    if tensor.device() != mesh_device:
        raise ValueError(f"{name} input must reside on the cache mesh")
    expected_shape = (1, 1, _LOCAL_CHUNK, _HEAD_DIM)
    if tuple(tensor.shape) != expected_shape:
        raise ValueError(f"{name} input must have local shape {expected_shape}, got {tuple(tensor.shape)}")
    if tensor.dtype != ttnn.bfloat16:
        raise ValueError(f"{name} input must be bfloat16, got {tensor.dtype}")
    if tensor.layout != ttnn.TILE_LAYOUT:
        raise ValueError(f"{name} input must use TILE_LAYOUT, got {tensor.layout}")
    if tensor.memory_config() != ttnn.DRAM_MEMORY_CONFIG:
        raise ValueError(f"{name} input must use interleaved DRAM, got {tensor.memory_config()}")
    if len(ttnn.get_device_tensors(tensor)) != _SP * _TP:
        raise ValueError(f"{name} input must cover {_SP * _TP} mesh devices")


def _validate_write(kv_cache, k, v, *, slot_idx, layer_idx, actual_start, actual_end):
    if not isinstance(kv_cache, LlamaKVCache):
        raise ValueError(f"kv_cache must be LlamaKVCache, got {type(kv_cache).__name__}")
    for name, value in (
        ("slot_idx", slot_idx),
        ("layer_idx", layer_idx),
        ("actual_start", actual_start),
        ("actual_end", actual_end),
    ):
        _validate_scalar(name, value)
    if (kv_cache.num_users, kv_cache.num_layers, kv_cache.max_seq_len, kv_cache.sp) != (
        _NUM_USERS,
        _NUM_LAYERS,
        _MAX_SEQ_LEN,
        _SP,
    ):
        raise ValueError(
            "Llama KV cache metadata must be num_users=2, num_layers=32, max_seq_len=2048, sp=4; "
            f"got {(kv_cache.num_users, kv_cache.num_layers, kv_cache.max_seq_len, kv_cache.sp)}"
        )
    if not 0 <= slot_idx < _NUM_USERS:
        raise ValueError(f"slot_idx {slot_idx} out of range [0, {_NUM_USERS})")
    if not 0 <= layer_idx < _NUM_LAYERS:
        raise ValueError(f"layer_idx {layer_idx} out of range [0, {_NUM_LAYERS})")
    if actual_start < 0 or actual_start % ttnn.TILE_SIZE:
        raise ValueError(f"actual_start must be nonnegative and tile-aligned, got {actual_start}")
    if not actual_start <= actual_end <= _MAX_SEQ_LEN:
        raise ValueError(
            f"actual range must satisfy actual_start <= actual_end <= {_MAX_SEQ_LEN}, "
            f"got [{actual_start}, {actual_end})"
        )
    if actual_end > actual_start + _GLOBAL_CHUNK:
        raise ValueError(
            f"actual_end must be within the {_GLOBAL_CHUNK}-token input chunk; "
            f"got start={actual_start}, end={actual_end}"
        )
    mesh_device = kv_cache.k.device()
    _validate_cache_tensor("k", kv_cache.k, mesh_device)
    _validate_cache_tensor("v", kv_cache.v, mesh_device)
    if kv_cache.k.dtype != kv_cache.v.dtype:
        raise ValueError(f"Llama K/V cache dtypes must match, got {kv_cache.k.dtype} and {kv_cache.v.dtype}")
    _validate_input("K", k, mesh_device)
    _validate_input("V", v, mesh_device)


def _write_one(cache, tensor, *, slot_idx, layer_idx, actual_start, actual_end):
    staging = tensor if tensor.dtype == cache.dtype else ttnn.typecast(tensor, cache.dtype)
    ttnn.experimental.deepseek_prefill.update_padded_kv_cache(
        cache,
        staging,
        slot_idx=slot_idx,
        layer_idx=layer_idx,
        num_layers=_NUM_LAYERS,
        kv_actual_global=actual_start,
        cluster_axis=_SP_AXIS,
        valid_global=actual_end,
    )
    ttnn.experimental.deepseek_prefill.zero_padded_kv_cache(
        cache,
        slot_idx,
        layer_idx,
        _NUM_LAYERS,
        actual_end,
        _GLOBAL_CHUNK,
        _SP_AXIS,
        _DRAM_PAGE_TOKENS,
    )
    if staging is not tensor:
        staging.deallocate(True)


def write_kv_chunk(kv_cache, k, v, *, slot_idx, layer_idx, actual_start, actual_end):
    """Write post-RoPE K and raw V into one packed user/layer plane."""
    _validate_write(
        kv_cache,
        k,
        v,
        slot_idx=slot_idx,
        layer_idx=layer_idx,
        actual_start=actual_start,
        actual_end=actual_end,
    )
    if actual_start == actual_end:
        return
    _write_one(
        kv_cache.k,
        k,
        slot_idx=slot_idx,
        layer_idx=layer_idx,
        actual_start=actual_start,
        actual_end=actual_end,
    )
    _write_one(
        kv_cache.v,
        v,
        slot_idx=slot_idx,
        layer_idx=layer_idx,
        actual_start=actual_start,
        actual_end=actual_end,
    )
