# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Packed, block-cyclic Llama-3.1 K/V cache for Galaxy prefill."""

from dataclasses import dataclass, field

import ttnn
from models.demos.llama_3p1_8b_d_p.reference.llama_3p1_8b_config import Llama31_8BConfig as Model
from models.demos.llama_3p1_8b_d_p.tt.prefill_geometry import DEFAULT_MAX_SEQ_LEN, DEFAULT_NUM_USERS
from models.demos.llama_3p1_8b_d_p.tt.prefill_geometry import PREFILL_LAYOUT as layout
from models.demos.llama_3p1_8b_d_p.tt.prefill_geometry import PrefillGeometry, validate_mesh

SUPPORTED_CACHE_DTYPES = (ttnn.bfloat16, ttnn.bfloat8_b)
# bfloat8_b is block-float: a 32x32 tile carries 1024 mantissa bytes plus one shared exponent per
# 16 datums, so 1088 bytes per tile rather than 1024.
_BYTES_PER_ELEMENT = {ttnn.bfloat16: 2.0, ttnn.bfloat8_b: 1.0625}
# DRAM held back from the cache so a chunk can still run in it: activations, the vocabulary logits,
# and slack for the allocator to place two multi-GiB buffers in banks the weights have fragmented.
# A chunk is a fixed 1024 tokens whatever the capacity, so this is a constant rather than a function
# of max_seq_len. Measured on a 4x8 Blackhole galaxy by squeezing free DRAM with ballast until a
# chunk stopped completing: chunks still ran with 0.45 GiB/chip free, the smallest figure probed.
# The margin over that is for placement, which is what actually fails first -- see
# docs/kv-slot-capacity.md.
DEFAULT_RUN_RESERVE_BYTES = 1024**3


@dataclass
class LlamaKVCache:
    """Externally owned, user-major Llama K/V caches."""

    k: ttnn.Tensor
    v: ttnn.Tensor
    num_users: int
    num_layers: int
    max_seq_len: int
    sp: int
    _populated_ends: dict[tuple[int, int], int] = field(default_factory=dict, init=False, repr=False)

    def populated_end(self, slot_idx, num_layers):
        """Return the contiguous prefix written in every requested layer of this slot.

        This tracks successful K/V enqueue operations, not device completion. Callers must
        preserve the same execution order as cache writes. Only writes through this object
        count; external DMA is not detected, and imported tensors start with no advertised prefix.
        """
        _validate_scalar("slot_idx", slot_idx)
        _validate_scalar("num_layers", num_layers)
        if not 0 <= slot_idx < self.num_users:
            raise ValueError(f"slot_idx {slot_idx} out of range [0, {self.num_users})")
        if not 1 <= num_layers <= self.num_layers:
            raise ValueError(f"num_layers must be in [1, {self.num_layers}], got {num_layers}")
        return min(self._populated_ends.get((slot_idx, layer_idx), 0) for layer_idx in range(num_layers))

    def truncate_prefix(self, slot_idx, actual_start):
        """Invalidate a slot suffix in every layer before a model call can partially fail.

        All layers are truncated, even when the caller executes a reduced layer count, so
        later full-model calls cannot inherit downstream cache state from an older prompt.
        """
        _validate_scalar("slot_idx", slot_idx)
        _validate_scalar("actual_start", actual_start)
        if not 0 <= slot_idx < self.num_users:
            raise ValueError(f"slot_idx {slot_idx} out of range [0, {self.num_users})")
        if not 0 <= actual_start <= self.max_seq_len:
            raise ValueError(f"actual_start must be in [0, {self.max_seq_len}], got {actual_start}")
        for layer_idx in range(self.num_layers):
            key = (slot_idx, layer_idx)
            self._populated_ends[key] = min(self._populated_ends.get(key, 0), actual_start)


def _validate_target(mesh_device, mesh_config, *, num_users, num_layers, max_seq_len, cache_dtype):
    validate_mesh(mesh_device, mesh_config, "Llama KV cache")
    if type(num_layers) is not int or num_layers != Model.NUM_LAYERS:
        raise ValueError(f"Llama KV cache requires num_layers={Model.NUM_LAYERS}, got {num_layers!r}")
    # num_users (the concurrent slot count) is bounded by DRAM, not by the layout: PrefillGeometry
    # rejects anything but a positive int, and allocate_kv_cache reports what does not fit.
    PrefillGeometry(max_seq_len, num_users)
    if cache_dtype not in SUPPORTED_CACHE_DTYPES:
        raise ValueError(f"Llama KV cache dtype must be bfloat16 or bfloat8_b, got {cache_dtype}")


def _cache_memory_config(mesh_device):
    bank_grid = ttnn.CoreRangeSet(
        [
            ttnn.CoreRange(ttnn.CoreCoord(bank_id, 0), ttnn.CoreCoord(bank_id, 0))
            for bank_id in range(mesh_device.dram_grid_size().x)
        ]
    )
    return ttnn.MemoryConfig(
        buffer_type=ttnn.BufferType.DRAM,
        nd_shard_spec=ttnn.NdShardSpec(
            shard_shape=[1, 1, layout.cache_page_size, Model.HEAD_DIM],
            grid=bank_grid,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            shard_distribution_strategy=ttnn.ShardDistributionStrategy.ROUND_ROBIN_1D,
        ),
    )


def slot_bytes_per_chip(max_seq_len=DEFAULT_MAX_SEQ_LEN, cache_dtype=ttnn.bfloat8_b):
    """DRAM one slot costs on every chip, counting both caches.

    The caches are replicated, not sharded, across the mesh, so every chip pays the full shape and
    this is the per-chip cost rather than a mesh total. It reduces to 2176 bytes per token of
    capacity at bfloat8_b: 32 layers x max_seq_len/4 local rows x 128 head_dim x 2 caches.
    """
    if cache_dtype not in SUPPORTED_CACHE_DTYPES:
        raise ValueError(f"Llama KV cache dtype must be bfloat16 or bfloat8_b, got {cache_dtype}")
    elements = Model.NUM_LAYERS * PrefillGeometry(max_seq_len).local_cache_sequence * Model.HEAD_DIM
    return int(2 * elements * _BYTES_PER_ELEMENT[cache_dtype])


def max_user_slots(
    mesh_device,
    *,
    max_seq_len=DEFAULT_MAX_SEQ_LEN,
    cache_dtype=ttnn.bfloat8_b,
    reserve_bytes=DEFAULT_RUN_RESERVE_BYTES,
):
    """How many slots fit in the DRAM that is free *right now*, with room left to run.

    Call this after the weights are resident: the answer is a measurement of the current allocator
    state, not a property of the hardware, and weights are the largest thing competing for it.

    Two corrections are applied to the naive division. Free space is capped by the largest
    contiguous block per bank, because each cache is a single buffer that has to land in one run per
    bank and weights leave the banks slightly fragmented; and ``reserve_bytes`` is withheld for the
    activations a chunk allocates after the cache exists, since a cache that fits but leaves no room
    to run is not useful.
    """
    view = ttnn.get_memory_view(mesh_device, ttnn.BufferType.DRAM)
    banks = mesh_device.dram_grid_size().x
    usable_per_bank = min(view.total_bytes_free_per_bank, view.largest_contiguous_bytes_free_per_bank)
    budget = usable_per_bank * banks - reserve_bytes
    if budget <= 0:
        return 0
    return int(budget // slot_bytes_per_chip(max_seq_len, cache_dtype))


def allocate_kv_cache(
    mesh_device,
    mesh_config,
    *,
    num_users=DEFAULT_NUM_USERS,
    num_layers=Model.NUM_LAYERS,
    max_seq_len=DEFAULT_MAX_SEQ_LEN,
    cache_dtype=ttnn.bfloat8_b,
    reserve_bytes=DEFAULT_RUN_RESERVE_BYTES,
):
    """Allocate zeroed K/V caches in the packed, block-cyclic SP4/TP8 layout.

    ``num_users`` is how many sequences the caches can hold at once. Each slot is an independent
    ``num_layers``-plane K/V region addressed by ``slot_idx``, so the footprint is linear in the slot
    count: one slot of a 128K-token context is ~272 MiB per chip across both caches at bfloat8_b.

    Pass ``num_users="max"`` to take everything DRAM allows at this capacity, which is what a serving
    front end wants when it would rather admit more sequences than choose a number by hand. The count
    is then derived from free DRAM at call time via :func:`max_user_slots`, so it must be called with
    the weights already loaded, and ``reserve_bytes`` is what stays free for the chunk to run in.
    """
    if num_users == "max":
        num_users = max_user_slots(
            mesh_device, max_seq_len=max_seq_len, cache_dtype=cache_dtype, reserve_bytes=reserve_bytes
        )
        if num_users < 1:
            free_per_bank = ttnn.get_memory_view(mesh_device, ttnn.BufferType.DRAM).total_bytes_free_per_bank
            free = free_per_bank * mesh_device.dram_grid_size().x
            raise RuntimeError(
                f"no KV slot fits: a slot at max_seq_len={max_seq_len} costs "
                f"{slot_bytes_per_chip(max_seq_len, cache_dtype) / 2**20:.1f} MiB per chip, and only "
                f"{free / 2**30:.2f} GiB per chip is free before the {reserve_bytes / 2**30:.2f} GiB run "
                f"reserve. Lower max_seq_len or free device memory."
            )
    _validate_target(
        mesh_device,
        mesh_config,
        num_users=num_users,
        num_layers=num_layers,
        max_seq_len=max_seq_len,
        cache_dtype=cache_dtype,
    )
    memory_config = _cache_memory_config(mesh_device)
    geometry = PrefillGeometry(max_seq_len, num_users)

    def allocate_one():
        # Zeroed on the device rather than uploaded. A host torch.zeros of this shape is fp32, so it
        # would cost 4 B/element against 1.0625 on device -- 44 GiB of host RAM for a slot count the
        # device holds in 21 GiB -- and the host would cap the slot count long before DRAM did.
        return ttnn.zeros(
            geometry.cache_shape,
            dtype=cache_dtype,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=memory_config,
        )

    allocated = []
    try:
        for _ in range(2):
            allocated.append(allocate_one())
    except RuntimeError as error:
        for tensor in allocated:
            tensor.deallocate(True)
        # Each cache is replicated per chip, so an out-of-memory here is a per-chip DRAM limit and the
        # two knobs are the slot count and the context length. Say so: the allocator's own message
        # reports a byte shortfall with no hint that num_users is what multiplies it.
        raise RuntimeError(
            f"Llama KV cache allocation failed for num_users={num_users}, max_seq_len={max_seq_len}, "
            f"{cache_dtype}: both caches need {geometry.cache_shape} per chip, and the footprint is "
            f"linear in num_users. Lower num_users or max_seq_len."
        ) from error

    return LlamaKVCache(
        k=allocated[0],
        v=allocated[1],
        num_users=num_users,
        num_layers=num_layers,
        max_seq_len=max_seq_len,
        sp=layout.sp,
    )


def _validate_scalar(name, value):
    if type(value) is not int:
        raise TypeError(f"{name} must be an eager Python int, got {type(value).__name__}")


def _validate_cache_tensor(name, tensor, mesh_device, *, max_seq_len=DEFAULT_MAX_SEQ_LEN, num_users=DEFAULT_NUM_USERS):
    cache_shape = PrefillGeometry(max_seq_len, num_users).cache_shape
    if not isinstance(tensor, ttnn.Tensor) or not ttnn.is_tensor_storage_on_device(tensor):
        raise ValueError(f"Llama KV cache {name} must be a device ttnn.Tensor")
    if tensor.device() != mesh_device:
        raise ValueError(f"Llama KV cache {name} must reside on the input mesh")
    if tuple(tensor.shape) != cache_shape:
        raise ValueError(f"Llama KV cache {name} must have local shape {cache_shape}, got {tuple(tensor.shape)}")
    if tensor.dtype not in SUPPORTED_CACHE_DTYPES:
        raise ValueError(f"Llama KV cache {name} has unsupported dtype {tensor.dtype}")
    if tensor.layout != ttnn.TILE_LAYOUT:
        raise ValueError(f"Llama KV cache {name} must use TILE_LAYOUT, got {tensor.layout}")
    memory_config = tensor.memory_config()
    shard_spec = memory_config.nd_shard_spec
    if memory_config.buffer_type != ttnn.BufferType.DRAM or shard_spec is None:
        raise ValueError(f"Llama KV cache {name} must use NdShard DRAM")
    if tuple(shard_spec.shard_shape) != (1, 1, layout.cache_page_size, Model.HEAD_DIM):
        raise ValueError(f"Llama KV cache {name} must use shard [1,1,32,128], got {tuple(shard_spec.shard_shape)}")
    if shard_spec.orientation != ttnn.ShardOrientation.ROW_MAJOR:
        raise ValueError(f"Llama KV cache {name} must use ROW_MAJOR shard orientation")
    if shard_spec.shard_distribution_strategy != ttnn.ShardDistributionStrategy.ROUND_ROBIN_1D:
        raise ValueError(f"Llama KV cache {name} must use ROUND_ROBIN_1D shard distribution")
    if len(ttnn.get_device_tensors(tensor)) != layout.num_devices:
        raise ValueError(f"Llama KV cache {name} must cover {layout.num_devices} mesh devices")


def _validate_input(name, tensor, mesh_device):
    if not isinstance(tensor, ttnn.Tensor) or not ttnn.is_tensor_storage_on_device(tensor):
        raise ValueError(f"{name} input must be a device ttnn.Tensor")
    if tensor.device() != mesh_device:
        raise ValueError(f"{name} input must reside on the cache mesh")
    expected_shape = (1, 1, layout.local_sequence, Model.HEAD_DIM)
    if tuple(tensor.shape) != expected_shape:
        raise ValueError(f"{name} input must have local shape {expected_shape}, got {tuple(tensor.shape)}")
    if tensor.dtype != ttnn.bfloat16:
        raise ValueError(f"{name} input must be bfloat16, got {tensor.dtype}")
    if tensor.layout != ttnn.TILE_LAYOUT:
        raise ValueError(f"{name} input must use TILE_LAYOUT, got {tensor.layout}")
    if tensor.memory_config() != ttnn.DRAM_MEMORY_CONFIG:
        raise ValueError(f"{name} input must use interleaved DRAM, got {tensor.memory_config()}")
    if len(ttnn.get_device_tensors(tensor)) != layout.num_devices:
        raise ValueError(f"{name} input must cover {layout.num_devices} mesh devices")


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
    geometry = PrefillGeometry(kv_cache.max_seq_len, kv_cache.num_users)
    geometry.validate_cache_metadata(kv_cache)
    if not 0 <= slot_idx < kv_cache.num_users:
        raise ValueError(f"slot_idx {slot_idx} out of range [0, {kv_cache.num_users})")
    if not 0 <= layer_idx < Model.NUM_LAYERS:
        raise ValueError(f"layer_idx {layer_idx} out of range [0, {Model.NUM_LAYERS})")
    geometry.validate_chunk_range(actual_start, actual_end, allow_empty=True)
    mesh_device = kv_cache.k.device()
    for name, tensor in (("k", kv_cache.k), ("v", kv_cache.v)):
        _validate_cache_tensor(
            name,
            tensor,
            mesh_device,
            max_seq_len=geometry.max_seq_len,
            num_users=geometry.num_users,
        )
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
        num_layers=Model.NUM_LAYERS,
        kv_actual_global=actual_start,
        cluster_axis=layout.sp_axis,
        valid_global=actual_end,
    )
    ttnn.experimental.deepseek_prefill.zero_padded_kv_cache(
        cache,
        slot_idx=slot_idx,
        layer_idx=layer_idx,
        num_layers=Model.NUM_LAYERS,
        valid_global=actual_end,
        chunk_size_global=layout.chunk_size,
        cluster_axis=layout.sp_axis,
        pad_align=layout.cache_page_size,
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
    key = (slot_idx, layer_idx)
    populated_end = kv_cache._populated_ends.get(key, 0)
    # Invalidate the rewritten suffix before either enqueue can fail. In particular, a
    # new prompt at position zero must not inherit a previously populated suffix.
    kv_cache._populated_ends[key] = min(populated_end, actual_start)
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
    # Arbitrary low-level writes remain supported, but a write beyond the contiguous
    # prefix cannot make a missing range safe for model attention to read.
    if actual_start <= populated_end:
        kv_cache._populated_ends[key] = actual_end
