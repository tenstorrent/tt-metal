# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""1x8 T3K multichip TTNN decoder for meta-llama/Llama-3.2-1B-Instruct.

The multichip decoder uses ``OptimizedDecoder`` as the single-chip baseline for
precision, attention, paged KV cache, and program-config policy.  The target
runtime is the local 8-chip Wormhole/T3K mesh with ``FABRIC_1D_RING``.

Runtime contract
----------------
Inputs and outputs are full hidden-size replicated mesh tensors.  Inside the
layer, attention and MLP weights are tensor-parallel across the 1x8 mesh:

* WQKV, W1, W3: column/output sharded.
* WO: fused all-gather plus column-sharded output projection on T3K Ring.
* W2: row/input sharded with reduce-scatter in ``_MultichipLlamaMLP``.
* KV cache: local one-KV-head paged cache per chip, page table replicated.
* Inter-chip attention/MLP residual payloads: BFP8 by default, with BF16
  reproducible through ``MD_MULTICHIP_*_DTYPE`` overrides.
* Standalone residual all-gather/reduce-scatter operations reuse persistent
  ping-pong output buffers by default; set
  ``MD_MULTICHIP_USE_PERSISTENT_CCL_BUFFERS=0`` to reproduce the preallocated
  buffer trial baseline.

The residual stream is gathered back to replicated hidden-size tensors after
attention and after MLP.  This keeps RMSNorm local and exact while still
applying 8 devices to the bandwidth-heavy matmuls.
"""

from __future__ import annotations

import math
import os
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from types import MethodType
from typing import Any

import torch
import ttnn

from models.autoports.meta_llama_llama_3_2_1b_instruct.tt.functional_decoder import (
    MODEL_ID,
    _layer_prefix,
    _reverse_permute,
    _state_tensor,
)
from models.autoports.meta_llama_llama_3_2_1b_instruct.tt.optimized_decoder import (
    OptimizedDecoder,
    OptimizedDecoderPrecisionPolicy,
    _compute_kernel_config_hifi2_fp16,
    _core_grid_for_k_n,
    _create_dram_sharded_mem_config,
    _dram_matmul_config,
    _find_prefill_grid,
    _matmul_2d_config,
    _mesh_mapper_config,
    dtype_from_config_name,
)
from models.common.lightweightmodule import LightweightModule
from models.common.modules.lazy_weight import LazyWeight
from models.common.modules.rmsnorm.rmsnorm_1d import RMSNorm1D, RMSNorm1DConfig
from models.common.modules.tt_ccl import (
    CCL_CHUNKS_PER_SYNC,
    CCL_NUM_BUFFERS_PER_CHANNEL,
    CCL_NUM_WORKERS_PER_LINK,
    TT_CCL,
    get_tt_ccl,
)
from models.common.tensor_utils import TILE_SIZE
from models.common.utility_functions import is_blackhole


TARGET_MESH_SHAPE = (1, 8)
_CCL_DTYPE_POLICY: dict[str, ttnn.DataType | None] = {}


def set_multichip_ccl_dtype_policy(
    *,
    all_gather_dtype: str | ttnn.DataType | None = ttnn.bfloat8_b,
    reduce_scatter_dtype: str | ttnn.DataType | None = ttnn.bfloat8_b,
) -> None:
    _CCL_DTYPE_POLICY["all_gather_dtype"] = dtype_from_config_name(all_gather_dtype)
    _CCL_DTYPE_POLICY["reduce_scatter_dtype"] = dtype_from_config_name(reduce_scatter_dtype)


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return int(value)


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return value.lower() in {"1", "true", "yes", "on"}


def _env_dtype(name: str, default: ttnn.DataType | None = None) -> ttnn.DataType | None:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    if value == "bfloat8_b":
        return ttnn.bfloat8_b
    if value == "bfloat16":
        return ttnn.bfloat16
    raise ValueError(f"{name}={value!r} is not supported; use bfloat8_b or bfloat16")


def _configured_ccl_dtype(
    policy_key: str,
    env_name: str,
    default: ttnn.DataType | None,
) -> ttnn.DataType | None:
    env_value = _env_dtype(env_name, None)
    if env_value is not None:
        return env_value
    return _CCL_DTYPE_POLICY.get(policy_key, default)


def _dense_core_range_set(grid_size: Any) -> ttnn.CoreRangeSet:
    if isinstance(grid_size, tuple):
        x, y = int(grid_size[0]), int(grid_size[1])
    else:
        x, y = int(grid_size.x), int(grid_size.y)
    return ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(x - 1, y - 1))})


def _populate_allowed_worker_cores(program_config: Any) -> None:
    if program_config is None or not hasattr(program_config, "allowed_worker_cores"):
        return
    if program_config.allowed_worker_cores is not None:
        return
    program_config.allowed_worker_cores = _dense_core_range_set(program_config.compute_with_storage_grid_size)


@dataclass(frozen=True)
class MultichipDecoderMeshPlan:
    """Serializable 1x8 tensor-parallel decoder plan."""

    model_id: str
    mesh_shape: tuple[int, int]
    topology: str
    tp: int
    hidden_size: int
    intermediate_size: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    page_block_size: int
    max_seq_len: int
    max_batch_size: int

    @property
    def local_query_heads(self) -> int:
        return self.num_attention_heads // self.tp

    @property
    def local_kv_heads(self) -> int:
        return self.num_key_value_heads // self.tp

    @property
    def local_qkv_width(self) -> int:
        return self.head_dim * (self.local_query_heads + 2 * self.local_kv_heads)

    @property
    def local_hidden(self) -> int:
        return self.hidden_size // self.tp

    @property
    def local_intermediate(self) -> int:
        return self.intermediate_size // self.tp

    def to_dict(self) -> dict[str, Any]:
        plan = asdict(self)
        plan.update(
            {
                "activation_strategy": {
                    "decoder_input": "replicated full hidden stream",
                    "decoder_output": "replicated full hidden stream",
                    "internal_attention_output": "hidden-sharded, then all-gathered before residual add",
                    "internal_mlp_output": "reduce-scattered hidden shard, then all-gathered before residual add",
                    "rmsnorm": "local RMSNorm on replicated hidden stream",
                },
                "tensor_parallel_shapes": [
                    {
                        "tensor": "WQKV",
                        "global_shape": [self.hidden_size, self.head_dim * (self.num_attention_heads + 2 * self.num_key_value_heads)],
                        "mesh_sharding": "column/output dim over TP",
                        "per_device_shape": [self.hidden_size, self.local_qkv_width],
                        "padding": "none",
                    },
                    {
                        "tensor": "Q heads",
                        "global_shape": [self.num_attention_heads, self.head_dim],
                        "mesh_sharding": "head axis over TP",
                        "per_device_shape": [self.local_query_heads, self.head_dim],
                        "padding": "none",
                    },
                    {
                        "tensor": "K/V heads and paged KV cache",
                        "global_shape": [self.num_key_value_heads, self.head_dim],
                        "mesh_sharding": "KV head axis over TP",
                        "per_device_shape": [self.local_kv_heads, self.head_dim],
                        "padding": "none",
                    },
                    {
                        "tensor": "WO",
                        "global_shape": [self.hidden_size, self.hidden_size],
                        "mesh_sharding": "column/output dim over TP after fused all-gather matmul",
                        "per_device_shape": [self.hidden_size, self.local_hidden],
                        "padding": "none",
                    },
                    {
                        "tensor": "W1/W3",
                        "global_shape": [self.hidden_size, self.intermediate_size],
                        "mesh_sharding": "column/output dim over TP",
                        "per_device_shape": [self.hidden_size, self.local_intermediate],
                        "padding": "none",
                    },
                    {
                        "tensor": "W2",
                        "global_shape": [self.intermediate_size, self.hidden_size],
                        "mesh_sharding": "row/input dim over TP",
                        "per_device_shape": [self.local_intermediate, self.hidden_size],
                        "padding": "none",
                    },
                    {
                        "tensor": "RMSNorm weights",
                        "global_shape": [self.hidden_size],
                        "mesh_sharding": "replicated",
                        "per_device_shape": [self.hidden_size],
                        "padding": "none",
                    },
                ],
                "collectives": [
                    "Attention fused all-gather matmul gathers local heads before WO on Ring.",
                    "Attention output all-gather on hidden dim uses BFP8 payloads and restores replicated residual stream.",
                    "MLP W2 reduce-scatter in _MultichipLlamaMLP uses BFP8 payloads and reduces row-parallel partials.",
                    "MLP output all-gather on hidden dim uses BFP8 payloads and restores replicated residual stream.",
                ],
                "kv_cache_strategy": {
                    "page_table": "replicated int32 page table",
                    "current_pos": "replicated int32 current positions",
                    "cache_dtype": "bfloat8_b",
                    "local_kv_heads_per_device": self.local_kv_heads,
                    "paged_cache_shape_per_device": [
                        math.ceil(self.max_seq_len / self.page_block_size) * self.max_batch_size,
                        self.local_kv_heads,
                        self.page_block_size,
                        self.head_dim,
                    ],
                },
                "moe_strategy": "not applicable; Llama-3.2-1B-Instruct decoder is dense",
                "rejected_alternatives": [
                    "1x1 single-chip: baseline only, leaves seven T3K devices idle.",
                    "1x4 TP: valid divisibility but halves the available weight bandwidth versus 1x8.",
                    "2D/Galaxy plan: rejected because the local hardware is T3K 1x8, not Galaxy 4x8.",
                    "Hidden-sharded residual stream: rejected for this state because QKV/W1/W3 still require gathered K inputs and the common 1D norm path does not provide traced decode distributed RMSNorm.",
                ],
            }
        )
        return plan


def _mesh_shape_tuple(mesh_device: ttnn.MeshDevice) -> tuple[int, int]:
    shape = tuple(mesh_device.shape)
    if len(shape) != 2:
        raise ValueError(f"expected a 2D mesh shape, got {shape}")
    return int(shape[0]), int(shape[1])


def _validate_target_mesh(mesh_device: ttnn.MeshDevice) -> None:
    shape = _mesh_shape_tuple(mesh_device)
    if shape != TARGET_MESH_SHAPE or mesh_device.get_num_devices() != 8:
        raise ValueError(
            f"{MODEL_ID} multichip decoder is specialized for 1x8 T3K TP, got shape={shape} "
            f"num_devices={mesh_device.get_num_devices()}"
        )


def _reordered_qkv_for_1x8_tp(
    state_dict: dict[str, torch.Tensor],
    *,
    prefix: str,
    hidden_size: int,
    n_heads: int,
    n_kv_heads: int,
    head_dim: int,
    tp: int,
) -> torch.Tensor:
    q_raw = _state_tensor(state_dict, prefix, "self_attn.q_proj.weight")
    k_raw = _state_tensor(state_dict, prefix, "self_attn.k_proj.weight")
    v_raw = _state_tensor(state_dict, prefix, "self_attn.v_proj.weight")

    q_meta = _reverse_permute(q_raw, n_heads, n_heads * head_dim, hidden_size).transpose(0, 1).contiguous()
    k_meta = _reverse_permute(k_raw, n_kv_heads, n_kv_heads * head_dim, hidden_size).transpose(0, 1).contiguous()
    v_meta = v_raw.transpose(0, 1).contiguous()

    q_chunks = torch.chunk(q_meta, tp, dim=-1)
    k_chunks = torch.chunk(k_meta, tp, dim=-1)
    v_chunks = torch.chunk(v_meta, tp, dim=-1)
    qkv_chunks = [torch.cat([q_chunks[i], k_chunks[i], v_chunks[i]], dim=-1) for i in range(tp)]
    return torch.cat(qkv_chunks, dim=-1).unsqueeze(0).unsqueeze(0)


def _cycle_persistent_buffer(tt_ccl: TT_CCL, key: tuple[Any, ...], create_fn: Any) -> Any:
    cache = getattr(tt_ccl, "_multichip_persistent_ccl_buffers", None)
    if cache is None:
        cache = {}
        setattr(tt_ccl, "_multichip_persistent_ccl_buffers", cache)
    if key not in cache:
        ttnn.synchronize_device(tt_ccl.mesh_device)
        cache[key] = {"buffers": [create_fn(), create_fn()], "index": 0}
        ttnn.synchronize_device(tt_ccl.mesh_device)
    entry = cache[key]
    index = entry["index"]
    entry["index"] = 1 - index
    return entry["buffers"][index]


def _persistent_all_gather_buffer(
    tensor: ttnn.Tensor,
    *,
    tt_ccl: TT_CCL,
    dim: int,
    dtype: ttnn.DataType,
) -> ttnn.Tensor:
    output_shape = list(tensor.shape)
    output_shape[dim] *= tt_ccl.mesh_device.shape[1]
    key = ("ag", tuple(output_shape), dim, dtype)

    def create() -> ttnn.Tensor:
        return ttnn.from_torch(
            torch.empty(output_shape),
            layout=ttnn.TILE_LAYOUT,
            dtype=dtype,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            device=tt_ccl.mesh_device,
        )

    return _cycle_persistent_buffer(tt_ccl, key, create)


def _persistent_reduce_scatter_buffers(
    tensor: ttnn.Tensor,
    *,
    tt_ccl: TT_CCL,
    dim: int,
    dtype: ttnn.DataType,
    output_memory_config: ttnn.MemoryConfig,
) -> list[ttnn.Tensor]:
    output_shape = list(tensor.shape)
    output_shape[dim] //= tt_ccl.mesh_device.shape[1]
    intermediate_shape = [2] + list(tensor.shape)
    key = ("rs", tuple(tensor.shape), dim, dtype, repr(output_memory_config))

    def create() -> list[ttnn.Tensor]:
        intermediate = ttnn.from_torch(
            torch.empty(intermediate_shape),
            layout=ttnn.TILE_LAYOUT,
            dtype=dtype,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            device=tt_ccl.mesh_device,
        )
        output = ttnn.from_torch(
            torch.empty(output_shape),
            layout=ttnn.TILE_LAYOUT,
            dtype=dtype,
            memory_config=output_memory_config,
            device=tt_ccl.mesh_device,
        )
        return [intermediate, output]

    return _cycle_persistent_buffer(tt_ccl, key, create)


def _all_gather_hidden(
    tensor: ttnn.Tensor,
    *,
    tt_ccl: TT_CCL,
    output_memory_config: ttnn.MemoryConfig,
) -> ttnn.Tensor:
    """Gather a TP hidden shard on dim 3 and return a replicated hidden tensor."""

    gather_dtype = _configured_ccl_dtype("all_gather_dtype", "MD_MULTICHIP_ALL_GATHER_DTYPE", ttnn.bfloat8_b)
    if gather_dtype is not None and tensor.dtype != gather_dtype:
        tensor = ttnn.typecast(tensor, dtype=gather_dtype)
    tensor = ttnn.to_memory_config(tensor, ttnn.DRAM_MEMORY_CONFIG)
    persistent_output_buffer = (
        _persistent_all_gather_buffer(tensor, tt_ccl=tt_ccl, dim=3, dtype=tensor.dtype)
        if _env_bool("MD_MULTICHIP_USE_PERSISTENT_CCL_BUFFERS", True)
        else None
    )
    gathered = ttnn.experimental.all_gather_async(
        tensor,
        persistent_output_buffer=persistent_output_buffer,
        dim=3,
        multi_device_global_semaphore=tt_ccl.get_and_cycle_ag_semaphore_handles(),
        num_links=tt_ccl.get_num_links(),
        topology=ttnn.Topology.Ring,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        barrier_semaphore=None
        if persistent_output_buffer is not None
        else tt_ccl.get_and_cycle_barrier_semaphore_handle(),
        chunks_per_sync=CCL_CHUNKS_PER_SYNC,
        num_workers_per_link=CCL_NUM_WORKERS_PER_LINK,
        num_buffers_per_channel=CCL_NUM_BUFFERS_PER_CHANNEL,
    )
    if output_memory_config != ttnn.DRAM_MEMORY_CONFIG:
        gathered = ttnn.to_memory_config(gathered, output_memory_config)
    return gathered


def _reduce_scatter_hidden(
    tensor: ttnn.Tensor,
    *,
    tt_ccl: TT_CCL,
    output_memory_config: ttnn.MemoryConfig,
) -> ttnn.Tensor:
    reduce_dtype = _configured_ccl_dtype("reduce_scatter_dtype", "MD_MULTICHIP_REDUCE_SCATTER_DTYPE", ttnn.bfloat8_b)
    if reduce_dtype is not None and tensor.dtype != reduce_dtype:
        tensor = ttnn.typecast(tensor, dtype=reduce_dtype)
    persistent_output_buffers = (
        _persistent_reduce_scatter_buffers(
            tensor,
            tt_ccl=tt_ccl,
            dim=3,
            dtype=tensor.dtype,
            output_memory_config=output_memory_config,
        )
        if _env_bool("MD_MULTICHIP_USE_PERSISTENT_CCL_BUFFERS", True)
        else None
    )
    return ttnn.experimental.reduce_scatter_minimal_async(
        tensor,
        persistent_output_buffers=persistent_output_buffers,
        dim=3,
        multi_device_global_semaphore=tt_ccl.get_and_cycle_rs_semaphore_handles(),
        barrier_semaphore=None
        if persistent_output_buffers is not None
        else tt_ccl.get_and_cycle_barrier_semaphore_handle(),
        num_links=tt_ccl.get_num_links(),
        memory_config=output_memory_config,
        intermediate_memory_config=ttnn.DRAM_MEMORY_CONFIG,
        topology=ttnn.Topology.Ring,
        chunks_per_sync=CCL_CHUNKS_PER_SYNC,
        num_workers_per_link=CCL_NUM_WORKERS_PER_LINK,
        num_buffers_per_channel=CCL_NUM_BUFFERS_PER_CHANNEL,
    )


def _bind_multichip_paged_attention_overrides(attention: Any, *, local_kv_heads: int, page_block_size: int) -> None:
    """Pass explicit local KV-cache layout to paged decode ops.

    The common Attention1D path can infer the KV-head view for single-chip
    decode, but on this 1x8 GQA path the paged SDPA kernel needs the local
    one-KV-head layout explicitly for long-position decode.
    """

    attention._multichip_local_kv_heads = local_kv_heads
    attention._multichip_page_block_size = page_block_size

    def _kv_fill_prefill_paged(self, keys, values, k_fill, v_fill, user_id, page_table, chunk_page_table) -> None:
        fill_page_table = chunk_page_table if chunk_page_table is not None else page_table
        page_len = fill_page_table.shape[1] * self._multichip_page_block_size

        k_fill_sliced = k_fill[:, :, :page_len, :] if page_len < k_fill.shape[2] else k_fill
        v_fill_sliced = v_fill[:, :, :page_len, :] if page_len < v_fill.shape[2] else v_fill

        if k_fill_sliced.dtype != keys.dtype or v_fill_sliced.dtype != values.dtype:
            raise ValueError(
                "paged KV cache prefill requires K/V fill tensors to already match the cache dtype: "
                f"k_fill={k_fill_sliced.dtype}, keys={keys.dtype}, "
                f"v_fill={v_fill_sliced.dtype}, values={values.dtype}"
            )

        ttnn.experimental.paged_fill_cache(
            keys,
            k_fill_sliced,
            fill_page_table,
            batch_idx=user_id,
            block_size=self._multichip_page_block_size,
        )
        ttnn.experimental.paged_fill_cache(
            values,
            v_fill_sliced,
            fill_page_table,
            batch_idx=user_id,
            block_size=self._multichip_page_block_size,
        )

    def _kv_update_decode_nonfused(self, keys, values, k_heads, v_heads, current_pos, page_table) -> None:
        ttnn.experimental.paged_update_cache(
            keys,
            k_heads,
            update_idxs_tensor=current_pos,
            page_table=page_table,
            block_size=self._multichip_page_block_size,
            num_kv_heads=self._multichip_local_kv_heads,
        )
        ttnn.experimental.paged_update_cache(
            values,
            v_heads,
            update_idxs_tensor=current_pos,
            page_table=page_table,
            block_size=self._multichip_page_block_size,
            num_kv_heads=self._multichip_local_kv_heads,
        )

    def _sdpa_decode_paged(self, q_heads, keys, values, current_pos, page_table) -> ttnn.Tensor:
        cfg = self.config
        return ttnn.transformer.paged_scaled_dot_product_attention_decode(
            q_heads,
            keys,
            values,
            page_table_tensor=page_table,
            cur_pos_tensor=current_pos,
            scale=cfg.scale,
            sliding_window_size=cfg.sliding_window,
            program_config=cfg.decode_sdpa_prg_config,
            compute_kernel_config=cfg.sdpa_decode_compute_kernel_cfg,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            block_size=self._multichip_page_block_size,
            num_kv_heads=self._multichip_local_kv_heads,
        )

    attention._kv_fill_prefill = MethodType(_kv_fill_prefill_paged, attention)
    attention._kv_update_decode = MethodType(_kv_update_decode_nonfused, attention)
    attention._sdpa_decode = MethodType(_sdpa_decode_paged, attention)


@dataclass(frozen=True)
class _MultichipMLPConfig:
    dim: int
    hidden_dim: int
    max_batch_size: int
    mesh_device: ttnn.MeshDevice
    decode_input_memcfg: ttnn.MemoryConfig
    decode_w1_w3_output_memcfg: ttnn.MemoryConfig
    decode_w2_input_memcfg: ttnn.MemoryConfig
    decode_w2_partial_output_memcfg: ttnn.MemoryConfig
    decode_residual_memcfg: ttnn.MemoryConfig
    decode_w1_w3_prg_config: ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig
    decode_w2_prg_config: ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig
    prefill_w1_w3_prg_config: Any
    prefill_w2_prg_config: Any
    ff1_3_compute_kernel_cfg: ttnn.WormholeComputeKernelConfig
    ff2_compute_kernel_cfg: ttnn.WormholeComputeKernelConfig
    linear_dtype: ttnn.DataType
    mul_dtype: ttnn.DataType
    prefill_len_cutoff: int


class _MultichipLlamaMLP(LightweightModule):
    """Optimized local MLP plus W2 reduce-scatter and final all-gather."""

    def __init__(
        self,
        *,
        gate_weight: LazyWeight,
        up_weight: LazyWeight,
        down_weight: LazyWeight,
        config: _MultichipMLPConfig,
        tt_ccl: TT_CCL,
        decode_output_memcfg: ttnn.MemoryConfig,
    ) -> None:
        super().__init__()
        self.gate_weight_lazy = gate_weight
        self.up_weight_lazy = up_weight
        self.down_weight_lazy = down_weight
        self.config = config
        self.tt_ccl = tt_ccl
        self.decode_output_memcfg = decode_output_memcfg
        self._device_weights_loaded = False

    @classmethod
    def from_state_dict(
        cls,
        state_dict: dict[str, torch.Tensor],
        *,
        prefix: str,
        mesh_device: ttnn.MeshDevice,
        dim: int,
        hidden_dim: int,
        max_batch_size: int,
        decode_residual_memcfg: ttnn.MemoryConfig,
        decode_output_memcfg: ttnn.MemoryConfig,
        policy: OptimizedDecoderPrecisionPolicy,
        tt_ccl: TT_CCL,
        cache_dir: Path | None = None,
        cache_prefix: str = "",
    ) -> "_MultichipLlamaMLP":
        gate = _state_tensor(state_dict, prefix, "mlp.gate_proj.weight").transpose(0, 1).contiguous()
        up = _state_tensor(state_dict, prefix, "mlp.up_proj.weight").transpose(0, 1).contiguous()
        down = _state_tensor(state_dict, prefix, "mlp.down_proj.weight").transpose(0, 1).contiguous()

        if hidden_dim % mesh_device.get_num_devices() != 0:
            raise ValueError(f"hidden_dim {hidden_dim} must divide TP={mesh_device.get_num_devices()}")

        num_devices = mesh_device.get_num_devices()
        per_device_hidden_dim = hidden_dim // num_devices
        tile_padded_batch_rows = TILE_SIZE * math.ceil(max_batch_size / TILE_SIZE)
        decode_w1_w3_target_cores = _env_int("MD_MULTICHIP_MLP_W1_W3_TARGET_CORES", 32)
        decode_w2_target_cores = _env_int("MD_MULTICHIP_MLP_W2_TARGET_CORES", 16)
        decode_grid = _core_grid_for_k_n(dim, per_device_hidden_dim, target_cores=decode_w1_w3_target_cores)
        decode_mlp2_grid = _core_grid_for_k_n(per_device_hidden_dim, dim, target_cores=decode_w2_target_cores)

        decode_input_memcfg = ttnn.create_sharded_memory_config(
            (tile_padded_batch_rows, dim // decode_grid.num_cores),
            decode_grid,
            ttnn.ShardStrategy.WIDTH,
            ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
        decode_w1_w3_output_memcfg = ttnn.create_sharded_memory_config(
            (tile_padded_batch_rows, per_device_hidden_dim // decode_grid.num_cores),
            decode_grid,
            ttnn.ShardStrategy.WIDTH,
            ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
        decode_w2_input_memcfg = ttnn.create_sharded_memory_config(
            (tile_padded_batch_rows, per_device_hidden_dim // decode_mlp2_grid.num_cores),
            decode_mlp2_grid,
            ttnn.ShardStrategy.WIDTH,
            ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
        decode_w2_partial_output_memcfg = ttnn.create_sharded_memory_config(
            (tile_padded_batch_rows, dim // decode_mlp2_grid.num_cores),
            decode_mlp2_grid,
            ttnn.ShardStrategy.WIDTH,
            ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )

        decode_w1_w3_prg_config = _dram_matmul_config(
            m=tile_padded_batch_rows,
            k=dim,
            n=per_device_hidden_dim,
            num_cores=decode_grid.num_cores,
        )
        decode_w2_prg_config = _dram_matmul_config(
            m=tile_padded_batch_rows,
            k=per_device_hidden_dim,
            n=dim,
            num_cores=decode_mlp2_grid.num_cores,
        )

        prefill_len_cutoff = 512 if is_blackhole() else 1024
        dram_shard_grid_width = 8 if not is_blackhole() else mesh_device.dram_grid_size().x
        prefill_rows = 8
        w1_w3_grid_size = _find_prefill_grid(prefill_rows, dim // TILE_SIZE)
        w2_grid_size = _find_prefill_grid(prefill_rows, per_device_hidden_dim // TILE_SIZE)

        @lru_cache
        def prefill_w1_w3_prg_config(seq_len: int):
            return _matmul_2d_config(
                m=min(seq_len, prefill_len_cutoff),
                k=dim,
                n=per_device_hidden_dim,
                grid_size=w1_w3_grid_size,
                per_core_n=math.ceil(per_device_hidden_dim / (TILE_SIZE * dram_shard_grid_width)),
            )

        @lru_cache
        def prefill_w2_prg_config(seq_len: int):
            return _matmul_2d_config(
                m=min(seq_len, prefill_len_cutoff),
                k=per_device_hidden_dim,
                n=dim,
                grid_size=w2_grid_size,
                per_core_n=math.ceil(dim / (TILE_SIZE * dram_shard_grid_width)),
            )

        dram_grid_size = mesh_device.dram_grid_size()
        dram_grid = ttnn.CoreRangeSet(
            {
                ttnn.CoreRange(
                    ttnn.CoreCoord(0, 0),
                    ttnn.CoreCoord(dram_grid_size.x - 1, dram_grid_size.y - 1),
                )
            }
        )
        w1_w3_memcfg = _create_dram_sharded_mem_config(
            k=dim,
            n=per_device_hidden_dim,
            dram_grid=dram_grid,
            dram_cores=dram_grid_size.x,
        )
        w2_memcfg = _create_dram_sharded_mem_config(
            k=per_device_hidden_dim,
            n=dim,
            dram_grid=dram_grid,
            dram_cores=dram_grid_size.x,
        )

        def cache_name(name: str) -> tuple[Path, str] | None:
            if cache_dir is None:
                return None
            return cache_dir, f"{cache_prefix}_{name}" if cache_prefix else name

        gate_lazy = LazyWeight(
            source=gate,
            dtype=policy.mlp_ff1_ff3_weight_dtype,
            device=mesh_device,
            mesh_mapper_config=_mesh_mapper_config(mesh_device, -1),
            layout=ttnn.TILE_LAYOUT,
            memory_config=w1_w3_memcfg,
            cache_dir_weight_name=cache_name("multichip_mlp_gate"),
        )
        up_lazy = LazyWeight(
            source=up,
            dtype=policy.mlp_ff1_ff3_weight_dtype,
            device=mesh_device,
            mesh_mapper_config=_mesh_mapper_config(mesh_device, -1),
            layout=ttnn.TILE_LAYOUT,
            memory_config=w1_w3_memcfg,
            cache_dir_weight_name=cache_name("multichip_mlp_up"),
        )
        down_lazy = LazyWeight(
            source=down,
            dtype=policy.mlp_ff2_weight_dtype,
            device=mesh_device,
            mesh_mapper_config=_mesh_mapper_config(mesh_device, -2),
            layout=ttnn.TILE_LAYOUT,
            memory_config=w2_memcfg,
            cache_dir_weight_name=cache_name("multichip_mlp_down"),
        )
        config = _MultichipMLPConfig(
            dim=dim,
            hidden_dim=hidden_dim,
            max_batch_size=max_batch_size,
            mesh_device=mesh_device,
            decode_input_memcfg=decode_input_memcfg,
            decode_w1_w3_output_memcfg=decode_w1_w3_output_memcfg,
            decode_w2_input_memcfg=decode_w2_input_memcfg,
            decode_w2_partial_output_memcfg=decode_w2_partial_output_memcfg,
            decode_residual_memcfg=decode_residual_memcfg,
            decode_w1_w3_prg_config=decode_w1_w3_prg_config,
            decode_w2_prg_config=decode_w2_prg_config,
            prefill_w1_w3_prg_config=prefill_w1_w3_prg_config,
            prefill_w2_prg_config=prefill_w2_prg_config,
            ff1_3_compute_kernel_cfg=_compute_kernel_config_hifi2_fp16(),
            ff2_compute_kernel_cfg=_compute_kernel_config_hifi2_fp16(),
            linear_dtype=policy.activation_dtype,
            mul_dtype=policy.mul_dtype,
            prefill_len_cutoff=prefill_len_cutoff,
        )
        return cls(
            gate_weight=gate_lazy,
            up_weight=up_lazy,
            down_weight=down_lazy,
            config=config,
            tt_ccl=tt_ccl,
            decode_output_memcfg=decode_output_memcfg,
        )

    def load_device_weights(self) -> None:
        if self._device_weights_loaded:
            return
        self.gate_weight = self.gate_weight_lazy.get_device_weight()
        self.up_weight = self.up_weight_lazy.get_device_weight()
        self.down_weight = self.down_weight_lazy.get_device_weight()
        self._device_weights_loaded = True

    def decode_forward(self, hidden_states: ttnn.Tensor) -> ttnn.Tensor:
        self.load_device_weights()
        cfg = self.config
        if hidden_states.memory_config() != cfg.decode_input_memcfg:
            hidden_states = ttnn.to_memory_config(hidden_states, cfg.decode_input_memcfg)

        gate = ttnn.linear(
            hidden_states,
            self.gate_weight,
            dtype=cfg.linear_dtype,
            compute_kernel_config=cfg.ff1_3_compute_kernel_cfg,
            program_config=cfg.decode_w1_w3_prg_config,
            memory_config=cfg.decode_w1_w3_output_memcfg,
        )
        up = ttnn.linear(
            hidden_states,
            self.up_weight,
            dtype=cfg.linear_dtype,
            compute_kernel_config=cfg.ff1_3_compute_kernel_cfg,
            program_config=cfg.decode_w1_w3_prg_config,
            memory_config=cfg.decode_w1_w3_output_memcfg,
        )
        ttnn.deallocate(hidden_states)

        fused = ttnn.mul(
            gate,
            up,
            input_tensor_a_activations=[ttnn.UnaryOpType.SILU],
            dtype=cfg.mul_dtype,
            memory_config=cfg.decode_w1_w3_output_memcfg,
        )
        ttnn.deallocate(gate)
        ttnn.deallocate(up)
        if fused.memory_config() != cfg.decode_w2_input_memcfg:
            fused = ttnn.to_memory_config(fused, cfg.decode_w2_input_memcfg)

        partial = ttnn.linear(
            fused,
            self.down_weight,
            dtype=cfg.linear_dtype,
            compute_kernel_config=cfg.ff2_compute_kernel_cfg,
            program_config=cfg.decode_w2_prg_config,
            memory_config=cfg.decode_w2_partial_output_memcfg,
        )
        ttnn.deallocate(fused)
        sharded = _reduce_scatter_hidden(partial, tt_ccl=self.tt_ccl, output_memory_config=cfg.decode_residual_memcfg)
        ttnn.deallocate(partial)
        return _all_gather_hidden(sharded, tt_ccl=self.tt_ccl, output_memory_config=self.decode_output_memcfg)

    def prefill_forward(self, hidden_states: ttnn.Tensor) -> ttnn.Tensor:
        self.load_device_weights()
        cfg = self.config
        hidden_states = ttnn.to_memory_config(hidden_states, ttnn.DRAM_MEMORY_CONFIG)
        seq_len = hidden_states.shape[-2]

        if seq_len >= cfg.prefill_len_cutoff:
            if seq_len % cfg.prefill_len_cutoff != 0:
                raise ValueError(f"seq_len {seq_len} must be divisible by {cfg.prefill_len_cutoff}")
            hidden_states = ttnn.reshape(hidden_states, [1, seq_len // cfg.prefill_len_cutoff, cfg.prefill_len_cutoff, -1])

        gate = ttnn.linear(
            hidden_states,
            self.gate_weight,
            dtype=cfg.linear_dtype,
            compute_kernel_config=cfg.ff1_3_compute_kernel_cfg,
            program_config=cfg.prefill_w1_w3_prg_config(seq_len),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        up = ttnn.linear(
            hidden_states,
            self.up_weight,
            dtype=cfg.linear_dtype,
            compute_kernel_config=cfg.ff1_3_compute_kernel_cfg,
            program_config=cfg.prefill_w1_w3_prg_config(seq_len),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(hidden_states)

        fused = ttnn.mul(
            gate,
            up,
            input_tensor_a_activations=[ttnn.UnaryOpType.SILU],
            dtype=cfg.mul_dtype,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(gate)
        ttnn.deallocate(up)

        partial = ttnn.linear(
            fused,
            self.down_weight,
            dtype=cfg.linear_dtype,
            compute_kernel_config=cfg.ff2_compute_kernel_cfg,
            program_config=cfg.prefill_w2_prg_config(seq_len),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(fused)
        original_shape = partial.shape
        partial = ttnn.reshape(
            partial,
            (1, 1, original_shape[-4] * original_shape[-3] * original_shape[-2], original_shape[-1]),
        )
        sharded = _reduce_scatter_hidden(partial, tt_ccl=self.tt_ccl, output_memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(partial)
        return _all_gather_hidden(sharded, tt_ccl=self.tt_ccl, output_memory_config=ttnn.DRAM_MEMORY_CONFIG)


class MultichipDecoder(OptimizedDecoder):
    """Specialized 1x8 Ring tensor-parallel decoder layer."""

    @classmethod
    def from_state_dict(
        cls,
        state_dict: dict[str, torch.Tensor],
        *,
        hf_config: Any,
        layer_idx: int,
        mesh_device: ttnn.MeshDevice,
        page_block_size: int = 64,
        max_seq_len: int | None = None,
        max_batch_size: int = 1,
        precision_policy: OptimizedDecoderPrecisionPolicy | None = None,
        cache_path: str | Path | None = None,
        materialize: bool = True,
        **kwargs: Any,
    ) -> "MultichipDecoder":
        _validate_target_mesh(mesh_device)

        hidden_size = int(hf_config.hidden_size)
        intermediate_size = int(hf_config.intermediate_size)
        n_heads = int(hf_config.num_attention_heads)
        n_kv_heads = int(getattr(hf_config, "num_key_value_heads", n_heads))
        head_dim = int(getattr(hf_config, "head_dim", hidden_size // n_heads) or (hidden_size // n_heads))
        if hidden_size != 2048 or intermediate_size != 8192 or n_heads != 32 or n_kv_heads != 8 or head_dim != 64:
            raise ValueError(
                f"{MODEL_ID} multichip decoder expected hidden=2048 intermediate=8192 heads=32 "
                f"kv_heads=8 head_dim=64, got hidden={hidden_size} intermediate={intermediate_size} "
                f"heads={n_heads} kv_heads={n_kv_heads} head_dim={head_dim}"
            )

        policy = precision_policy or OptimizedDecoderPrecisionPolicy(
            use_qk_fused_decode=_env_bool("MD_MULTICHIP_USE_QK_FUSED_DECODE", False)
        )
        max_seq_len = int(max_seq_len or hf_config.max_position_embeddings)
        prefix = _layer_prefix(state_dict, layer_idx)
        cache_dir = Path(cache_path) if cache_path is not None else None
        cache_prefix = f"layer{layer_idx}"
        tt_ccl = get_tt_ccl(mesh_device)

        optimized = OptimizedDecoder.from_state_dict(
            state_dict,
            hf_config=hf_config,
            layer_idx=layer_idx,
            mesh_device=mesh_device,
            page_block_size=page_block_size,
            max_seq_len=max_seq_len,
            max_batch_size=max_batch_size,
            precision_policy=policy,
            cache_path=cache_path,
            materialize=False,
            **kwargs,
        )
        optimized.attention.config.topology = ttnn.Topology.Ring
        optimized.attention.config.tt_ccl = tt_ccl
        _populate_allowed_worker_cores(optimized.attention.config.decode_all_gather_matmul_prg_config)
        _bind_multichip_paged_attention_overrides(
            optimized.attention,
            local_kv_heads=n_kv_heads // mesh_device.get_num_devices(),
            page_block_size=page_block_size,
        )
        old_wqkv = optimized.attention.config.wqkv
        optimized.attention.config.wqkv = LazyWeight(
            source=_reordered_qkv_for_1x8_tp(
                state_dict,
                prefix=prefix,
                hidden_size=hidden_size,
                n_heads=n_heads,
                n_kv_heads=n_kv_heads,
                head_dim=head_dim,
                tp=mesh_device.get_num_devices(),
            ),
            dtype=old_wqkv.dtype,
            device=mesh_device,
            mesh_mapper_config=old_wqkv.mesh_mapper_config,
            layout=old_wqkv.layout,
            memory_config=old_wqkv.memory_config,
            cache_dir_weight_name=(
                (cache_dir, f"{cache_prefix}_multichip_wqkv_reordered") if cache_dir is not None else None
            ),
        )
        full_hidden_decode_memcfg = optimized.attention.config.decode_input_memcfg
        row_parallel_output_memcfg = optimized.decode_residual_memcfg

        mlp = _MultichipLlamaMLP.from_state_dict(
            state_dict,
            prefix=prefix,
            mesh_device=mesh_device,
            dim=hidden_size,
            hidden_dim=intermediate_size,
            max_batch_size=max_batch_size,
            decode_residual_memcfg=row_parallel_output_memcfg,
            decode_output_memcfg=full_hidden_decode_memcfg,
            policy=policy,
            tt_ccl=tt_ccl,
            cache_dir=cache_dir,
            cache_prefix=cache_prefix,
        )

        norm_eps = float(hf_config.rms_norm_eps)

        def norm_weight(name: str, tensor_name: str) -> LazyWeight:
            return LazyWeight(
                source=_state_tensor(state_dict, prefix, tensor_name),
                dtype=policy.norm_weight_dtype,
                cache_dir_weight_name=(cache_dir, f"{cache_prefix}_{name}") if cache_dir is not None else None,
            )

        attention_norm = RMSNorm1D.from_config(
            RMSNorm1DConfig(
                weight=norm_weight("multichip_input_layernorm", "input_layernorm.weight"),
                mesh_device=mesh_device,
                eps=norm_eps,
                max_batch_size=max_batch_size,
                prefill_distributed=False,
                decode_memory_config=full_hidden_decode_memcfg,
                compute_kernel_config=_compute_kernel_config_hifi2_fp16(),
            )
        )
        post_attention_norm = RMSNorm1D.from_config(
            RMSNorm1DConfig(
                weight=norm_weight("multichip_post_attention_layernorm", "post_attention_layernorm.weight"),
                mesh_device=mesh_device,
                eps=norm_eps,
                max_batch_size=max_batch_size,
                prefill_distributed=False,
                decode_memory_config=mlp.config.decode_input_memcfg,
                compute_kernel_config=_compute_kernel_config_hifi2_fp16(),
            )
        )

        decoder = cls(
            attention_norm=attention_norm,
            attention=optimized.attention,
            post_attention_norm=post_attention_norm,
            mlp=mlp,
            decode_residual_memcfg=full_hidden_decode_memcfg,
            mesh_device=mesh_device,
            hf_config=hf_config,
            layer_idx=layer_idx,
            page_block_size=page_block_size,
            max_seq_len=max_seq_len,
            max_batch_size=max_batch_size,
            precision_policy=policy,
        )
        decoder.tt_ccl = tt_ccl
        decoder.mesh_plan = MultichipDecoderMeshPlan(
            model_id=MODEL_ID,
            mesh_shape=TARGET_MESH_SHAPE,
            topology="Ring",
            tp=mesh_device.get_num_devices(),
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_attention_heads=n_heads,
            num_key_value_heads=n_kv_heads,
            head_dim=head_dim,
            page_block_size=page_block_size,
            max_seq_len=max_seq_len,
            max_batch_size=max_batch_size,
        )
        if materialize:
            decoder.load_device_weights()
        return decoder

    @property
    def mesh_strategy(self) -> dict[str, Any]:
        return self.mesh_plan.to_dict()

    def prefill_forward(
        self,
        hidden_states: ttnn.Tensor,
        *,
        rot_mats: tuple[ttnn.Tensor, ttnn.Tensor],
        page_table: ttnn.Tensor,
        user_id: int = 0,
        chunk_page_table: ttnn.Tensor | None = None,
        chunk_start_idx: int | None = None,
    ) -> ttnn.Tensor:
        residual = hidden_states
        normed = self.attention_norm.prefill_forward(hidden_states)
        attn_sharded = self.attention.prefill_forward(
            normed,
            rot_mats,
            user_id=user_id,
            page_table=page_table,
            chunk_page_table=chunk_page_table,
            chunk_start_idx=chunk_start_idx,
        )
        attn_out = _all_gather_hidden(attn_sharded, tt_ccl=self.tt_ccl, output_memory_config=ttnn.DRAM_MEMORY_CONFIG)
        hidden_states = ttnn.add(
            residual,
            attn_out,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=self.precision_policy.residual_dtype,
        )

        residual = hidden_states
        normed = self.post_attention_norm.prefill_forward(hidden_states)
        mlp_out = self.mlp.prefill_forward(normed)
        return ttnn.add(
            residual,
            mlp_out,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=self.precision_policy.residual_dtype,
        )

    def decode_forward(
        self,
        hidden_states: ttnn.Tensor,
        *,
        current_pos: ttnn.Tensor,
        rot_mats: tuple[ttnn.Tensor, ttnn.Tensor],
        page_table: ttnn.Tensor,
    ) -> ttnn.Tensor:
        hidden_states = ttnn.to_memory_config(hidden_states, self.decode_residual_memcfg)
        residual = hidden_states
        normed = self.attention_norm.decode_forward(hidden_states)
        attn_sharded = self.attention.decode_forward(normed, current_pos, rot_mats, page_table=page_table)
        attn_out = _all_gather_hidden(
            attn_sharded,
            tt_ccl=self.tt_ccl,
            output_memory_config=self.decode_residual_memcfg,
        )
        hidden_states = ttnn.add(
            residual,
            attn_out,
            memory_config=self.decode_residual_memcfg,
            dtype=self.precision_policy.residual_dtype,
        )

        residual = hidden_states
        normed = self.post_attention_norm.decode_forward(hidden_states)
        mlp_out = self.mlp.decode_forward(normed)
        return ttnn.add(
            residual,
            mlp_out,
            memory_config=self.decode_residual_memcfg,
            dtype=self.precision_policy.residual_dtype,
        )
