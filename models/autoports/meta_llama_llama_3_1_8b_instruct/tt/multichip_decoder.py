# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Multichip TTNN decoder layer for meta-llama/Llama-3.1-8B-Instruct.

This stage starts from the optimized single-chip decoder policy and maps the
dense Llama decoder layer to the local T3K 1x8 mesh with 1D tensor parallelism.

The decoder boundary is a replicated full-hidden residual stream. Internally,
attention and MLP weights are tensor-parallel across the mesh; their row/output
parallel stages return hidden shards, which are gathered back to the replicated
boundary before residual adds and the next stacked layer.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable

import torch
import ttnn

from models.autoports.meta_llama_llama_3_1_8b_instruct.tt.functional_decoder import (
    PagedAttentionConfig,
    _get_layer_tensor,
    _require_llama31_8b_config,
    _reverse_permute,
)
from models.autoports.meta_llama_llama_3_1_8b_instruct.tt.optimized_decoder import (
    OptimizedDecoder,
    OptimizedDecoderPolicy,
    _compute_kernel_config_hifi2_fp16,
    _compute_kernel_config_hifi4,
    _compute_kernel_config_lofi,
    _core_grid_for_tiles,
    _dram_matmul_config,
    _dram_sharded_weight_memcfg,
    _find_prefill_grid,
    _get_out_subblock_w,
    _matmul_2d_config,
    _norm_weight,
    _width_sharded_l1_memcfg,
)
from models.common.lightweightmodule import LightweightModule
from models.common.modules.attention.attention_1d import Attention1D, Attention1DConfig
from models.common.modules.lazy_weight import LazyWeight
from models.common.modules.rmsnorm.rmsnorm_1d import RMSNorm1D
from models.common.modules.tt_ccl import (
    CCL_CHUNKS_PER_SYNC,
    CCL_NUM_BUFFERS_PER_CHANNEL,
    CCL_NUM_WORKERS_PER_LINK,
    TT_CCL,
    get_tt_ccl,
)
from models.common.tensor_utils import TILE_SIZE, get_padded_hidden_dim, zeros_like_paged_cache
from models.common.utility_functions import is_blackhole


TARGET_MESH_SHAPE = (1, 8)
TARGET_TOPOLOGY = ttnn.Topology.Ring
MULTICHIP_CCL_CHUNKS_PER_SYNC = int(os.environ.get("MULTICHIP_DECODER_CCL_CHUNKS_PER_SYNC", CCL_CHUNKS_PER_SYNC))
MULTICHIP_CCL_NUM_WORKERS_PER_LINK = int(
    os.environ.get("MULTICHIP_DECODER_CCL_NUM_WORKERS_PER_LINK", CCL_NUM_WORKERS_PER_LINK)
)
MULTICHIP_CCL_NUM_BUFFERS_PER_CHANNEL = int(
    os.environ.get("MULTICHIP_DECODER_CCL_NUM_BUFFERS_PER_CHANNEL", CCL_NUM_BUFFERS_PER_CHANNEL)
)


@dataclass(frozen=True)
class MultiChipDecoderPolicy(OptimizedDecoderPolicy):
    """Optimized single-chip dtype policy, bound to the 1x8 multichip path."""

    name: str = "llama31_8b_t3k_1x8_tp8_bfp4_attn_bfp4_mlp_bfp8_act_decode_v2"
    activation_dtype: ttnn.DataType = ttnn.bfloat8_b
    attention_weight_dtype: ttnn.DataType = ttnn.bfloat4_b


def _mesh_mapper_1d(num_devices: int, placement_dim: int) -> ttnn.MeshMapperConfig:
    return ttnn.MeshMapperConfig(
        placements=[ttnn.PlacementShard(placement_dim)],
        mesh_shape_override=ttnn.MeshShape([num_devices]),
    )


def _require_target_mesh(mesh_device: ttnn.MeshDevice) -> None:
    shape = tuple(mesh_device.shape)
    if shape != TARGET_MESH_SHAPE or mesh_device.get_num_devices() != 8:
        raise ValueError(
            "MultiChipDecoder is specialized for the local 1x8 T3K mesh; "
            f"got shape={shape}, devices={mesh_device.get_num_devices()}"
        )


def _worker_cores_from_grid(grid_size: Any) -> ttnn.CoreRangeSet:
    grid_x = int(grid_size.x)
    grid_y = int(grid_size.y)
    return ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid_x - 1, grid_y - 1))})


def _with_allowed_worker_cores(program_config: Any) -> Any:
    if hasattr(program_config, "allowed_worker_cores") and program_config.allowed_worker_cores is None:
        program_config.allowed_worker_cores = _worker_cores_from_grid(program_config.compute_with_storage_grid_size)
    return program_config


class _TensorParallelMLP(LightweightModule):
    """1D TP Llama SwiGLU MLP with optimized dtype policy.

    Gate/up are column-parallel. Down is row-parallel and reduce-scattered to a
    hidden shard. The owning decoder gathers that shard back to its replicated
    boundary.
    """

    def __init__(
        self,
        *,
        gate: LazyWeight,
        up: LazyWeight,
        down: LazyWeight,
        dim: int,
        hidden_dim: int,
        max_batch_size: int,
        mesh_device: ttnn.MeshDevice,
        tt_ccl: TT_CCL,
        topology: ttnn.Topology,
        activation_dtype: ttnn.DataType,
        mul_dtype: ttnn.DataType,
        compute_kernel_config: ttnn.WormholeComputeKernelConfig,
        num_reduce_scatter_links: int,
        prefill_len_cutoff: int = 1024,
    ) -> None:
        super().__init__()
        self.gate_lazy = gate
        self.up_lazy = up
        self.down_lazy = down
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.max_batch_size = max_batch_size
        self.mesh_device = mesh_device
        self.tt_ccl = tt_ccl
        self.topology = topology
        self.activation_dtype = activation_dtype
        self.mul_dtype = mul_dtype
        self.num_reduce_scatter_links = num_reduce_scatter_links
        self.prefill_len_cutoff = prefill_len_cutoff
        self.num_devices = mesh_device.get_num_devices()
        self.hidden_dim_per_device = hidden_dim // self.num_devices
        self.decode_rows = TILE_SIZE * math.ceil(max_batch_size / TILE_SIZE)

        gate_up_grid = _core_grid_for_tiles(
            math.gcd(dim // TILE_SIZE, self.hidden_dim_per_device // TILE_SIZE),
            target=8,
        )
        down_grid = _core_grid_for_tiles(
            math.gcd(self.hidden_dim_per_device // TILE_SIZE, dim // TILE_SIZE),
            target=8,
        )
        residual_grid = _core_grid_for_tiles((dim // self.num_devices) // TILE_SIZE, target=16)

        self.decode_input_memcfg = _width_sharded_l1_memcfg(dim, gate_up_grid, rows=self.decode_rows)
        self.decode_hidden_memcfg = _width_sharded_l1_memcfg(
            self.hidden_dim_per_device, down_grid, rows=self.decode_rows
        )
        self.decode_sharded_output_memcfg = _width_sharded_l1_memcfg(
            dim // self.num_devices, residual_grid, rows=self.decode_rows
        )
        self.decode_gate_up_prg_config = _dram_matmul_config(
            m=self.decode_rows,
            k=dim,
            n=self.hidden_dim_per_device,
            num_cores=gate_up_grid.num_cores,
        )
        self.decode_down_prg_config = _dram_matmul_config(
            m=self.decode_rows,
            k=self.hidden_dim_per_device,
            n=dim,
            num_cores=down_grid.num_cores,
        )
        self.ff1_3_compute_kernel_cfg = compute_kernel_config
        self.ff2_compute_kernel_cfg = compute_kernel_config
        self._loaded = False

        dram_grid_width = 8 if not is_blackhole() else mesh_device.dram_grid_size().x
        prefill_rows = 8
        gate_up_prefill_grid = _find_prefill_grid(prefill_rows, dim // TILE_SIZE)
        down_prefill_grid = _find_prefill_grid(prefill_rows, hidden_dim // TILE_SIZE)

        @lru_cache
        def gate_up_prefill_prg_config(seq_len: int) -> ttnn.MatmulMultiCoreReuseMultiCastProgramConfig:
            long_prefill = seq_len >= self.prefill_len_cutoff
            per_core_n = math.ceil(self.hidden_dim_per_device / (TILE_SIZE * dram_grid_width))
            return _with_allowed_worker_cores(
                _matmul_2d_config(
                    m=min(seq_len, self.prefill_len_cutoff),
                    k=dim,
                    n=self.hidden_dim_per_device,
                    grid_size=gate_up_prefill_grid,
                    in0_block_w=4 if long_prefill else None,
                    out_subblock_w=_get_out_subblock_w(per_core_n) if long_prefill else None,
                    per_core_n=per_core_n,
                )
            )

        @lru_cache
        def down_prefill_prg_config(seq_len: int) -> ttnn.MatmulMultiCoreReuseMultiCastProgramConfig:
            long_prefill = seq_len >= self.prefill_len_cutoff
            per_core_n = math.ceil(dim / (TILE_SIZE * dram_grid_width))
            return _with_allowed_worker_cores(
                _matmul_2d_config(
                    m=min(seq_len, self.prefill_len_cutoff),
                    k=hidden_dim,
                    n=dim,
                    grid_size=down_prefill_grid,
                    in0_block_w=4 if long_prefill else None,
                    out_subblock_w=_get_out_subblock_w(per_core_n) if long_prefill else None,
                    per_core_n=per_core_n,
                )
            )

        self.prefill_gate_up_prg_config: Callable[[int], ttnn.MatmulMultiCoreReuseMultiCastProgramConfig] = (
            gate_up_prefill_prg_config
        )
        self.prefill_down_prg_config: Callable[[int], ttnn.MatmulMultiCoreReuseMultiCastProgramConfig] = (
            down_prefill_prg_config
        )
        self._reduce_scatter_intermediate_buffers: dict[str, ttnn.Tensor] = {}

    def load_device_weights(self) -> None:
        if self._loaded:
            return
        self.gate = self.gate_lazy.get_device_weight()
        self.up = self.up_lazy.get_device_weight()
        self.down = self.down_lazy.get_device_weight()
        self._loaded = True

    def _reduce_scatter(self, partial: ttnn.Tensor, *, mode: str) -> ttnn.Tensor:
        memory_config = self.decode_sharded_output_memcfg if mode == "decode" else ttnn.DRAM_MEMORY_CONFIG
        if partial.is_sharded():
            partial_interleaved = ttnn.sharded_to_interleaved(partial, ttnn.L1_MEMORY_CONFIG)
            partial.deallocate(True)
        else:
            partial_interleaved = partial

        intermediate_buffer = self._reduce_scatter_intermediate_buffers.get(mode)
        if intermediate_buffer is None:
            intermediate_buffer = ttnn.allocate_tensor_on_device(
                list(partial_interleaved.shape),
                partial_interleaved.dtype,
                partial_interleaved.layout,
                self.mesh_device,
                ttnn.DRAM_MEMORY_CONFIG,
            )
            self._reduce_scatter_intermediate_buffers[mode] = intermediate_buffer

        reduced = ttnn.experimental.reduce_scatter_minimal_async(
            partial_interleaved,
            persistent_output_buffers=[intermediate_buffer],
            dim=3,
            multi_device_global_semaphore=self.tt_ccl.get_and_cycle_rs_semaphore_handles(),
            barrier_semaphore=self.tt_ccl.get_and_cycle_barrier_semaphore_handle(),
            num_links=self.num_reduce_scatter_links,
            memory_config=memory_config,
            intermediate_memory_config=ttnn.DRAM_MEMORY_CONFIG,
            topology=self.topology,
            chunks_per_sync=MULTICHIP_CCL_CHUNKS_PER_SYNC,
            num_workers_per_link=MULTICHIP_CCL_NUM_WORKERS_PER_LINK,
            num_buffers_per_channel=MULTICHIP_CCL_NUM_BUFFERS_PER_CHANNEL,
        )
        partial_interleaved.deallocate(True)
        return reduced

    def release_decode_persistent_buffers(self) -> None:
        tensor = self._reduce_scatter_intermediate_buffers.pop("decode", None)
        if tensor is not None and tensor.is_allocated():
            tensor.deallocate(True)

    def prefill_forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        self.load_device_weights()
        x = ttnn.to_memory_config(x, ttnn.DRAM_MEMORY_CONFIG)
        seq_len = x.shape[-2]
        reshaped = False
        if seq_len >= self.prefill_len_cutoff:
            if seq_len % self.prefill_len_cutoff != 0:
                raise ValueError(
                    f"seq_len ({seq_len}) must be divisible by prefill_len_cutoff ({self.prefill_len_cutoff})"
                )
            x = ttnn.reshape(x, [1, seq_len // self.prefill_len_cutoff, self.prefill_len_cutoff, -1])
            reshaped = True

        gate = ttnn.linear(
            x,
            self.gate,
            dtype=self.activation_dtype,
            compute_kernel_config=self.ff1_3_compute_kernel_cfg,
            program_config=self.prefill_gate_up_prg_config(seq_len),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        up = ttnn.linear(
            x,
            self.up,
            dtype=self.activation_dtype,
            compute_kernel_config=self.ff1_3_compute_kernel_cfg,
            program_config=self.prefill_gate_up_prg_config(seq_len),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(x)

        hidden = ttnn.mul(
            gate,
            up,
            input_tensor_a_activations=[ttnn.UnaryOpType.SILU],
            dtype=self.mul_dtype,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(gate)
        ttnn.deallocate(up)

        partial = ttnn.linear(
            hidden,
            self.down,
            dtype=self.activation_dtype,
            compute_kernel_config=self.ff2_compute_kernel_cfg,
            program_config=self.prefill_down_prg_config(seq_len),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(hidden)
        out = self._reduce_scatter(partial, mode="prefill")
        if reshaped:
            out = ttnn.reshape(out, [1, 1, seq_len, -1])
        return out

    def decode_forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        self.load_device_weights()
        x = ttnn.to_memory_config(x, self.decode_input_memcfg)

        gate = ttnn.linear(
            x,
            self.gate,
            dtype=self.activation_dtype,
            compute_kernel_config=self.ff1_3_compute_kernel_cfg,
            program_config=self.decode_gate_up_prg_config,
            memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
        )
        up = ttnn.linear(
            x,
            self.up,
            dtype=self.activation_dtype,
            compute_kernel_config=self.ff1_3_compute_kernel_cfg,
            program_config=self.decode_gate_up_prg_config,
            memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
        )
        ttnn.deallocate(x)

        hidden = ttnn.mul(
            gate,
            up,
            input_tensor_a_activations=[ttnn.UnaryOpType.SILU],
            dtype=self.mul_dtype,
            memory_config=self.decode_hidden_memcfg,
        )
        ttnn.deallocate(gate)
        ttnn.deallocate(up)

        partial = ttnn.linear(
            hidden,
            self.down,
            dtype=self.activation_dtype,
            compute_kernel_config=self.ff2_compute_kernel_cfg,
            program_config=self.decode_down_prg_config,
            memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
        )
        ttnn.deallocate(hidden)
        return self._reduce_scatter(partial, mode="decode")

    def forward(self, x: ttnn.Tensor, *, mode: str) -> ttnn.Tensor:
        if mode == "prefill":
            return self.prefill_forward(x)
        if mode == "decode":
            return self.decode_forward(x)
        raise ValueError(f"Unknown MLP mode {mode!r}; expected 'prefill' or 'decode'.")


class MultiChipDecoder(LightweightModule):
    """Single Llama decoder layer parallelized over the local 1x8 T3K mesh."""

    single_chip_baseline_cls = OptimizedDecoder

    def __init__(
        self,
        *,
        input_layernorm: RMSNorm1D,
        self_attn: Attention1D,
        post_attention_layernorm: RMSNorm1D,
        mlp: _TensorParallelMLP,
        policy: MultiChipDecoderPolicy,
        mesh_device: ttnn.MeshDevice,
        tt_ccl: TT_CCL,
        topology: ttnn.Topology,
        num_all_gather_links: int,
    ) -> None:
        super().__init__()
        self.input_layernorm = input_layernorm
        self.self_attn = self_attn
        self.post_attention_layernorm = post_attention_layernorm
        self.mlp = mlp
        self.policy = policy
        self.mesh_device = mesh_device
        self.tt_ccl = tt_ccl
        self.topology = topology
        self.num_all_gather_links = num_all_gather_links
        self.decode_residual_memcfg = input_layernorm.config.decode_memory_config
        self.prefill_residual_memcfg = ttnn.DRAM_MEMORY_CONFIG
        self._all_gather_output_buffers: dict[tuple[str, str], ttnn.Tensor] = {}

    @classmethod
    def from_state_dict(
        cls,
        state_dict: dict[str, torch.Tensor],
        *,
        hf_config: Any,
        layer_idx: int,
        mesh_device: ttnn.MeshDevice,
        max_batch_size: int = 1,
        max_seq_len: int | None = None,
        page_block_size: int = 64,
        max_num_blocks: int | None = None,
        policy: MultiChipDecoderPolicy | OptimizedDecoderPolicy | None = None,
        weight_dtype: ttnn.DataType | None = None,
        activation_dtype: ttnn.DataType | None = None,
        kv_cache_dtype: ttnn.DataType | None = None,
        mlp_gate_up_dtype: ttnn.DataType | None = None,
        mlp_down_dtype: ttnn.DataType | None = None,
        cache_dir: str | Path | None = None,
        **kwargs,
    ) -> "MultiChipDecoder":
        if kwargs:
            raise TypeError(f"Unexpected MultiChipDecoder.from_state_dict kwargs: {sorted(kwargs)}")
        _require_llama31_8b_config(hf_config)
        _require_target_mesh(mesh_device)

        base_policy = policy or MultiChipDecoderPolicy()
        policy = MultiChipDecoderPolicy(
            name=base_policy.name,
            activation_dtype=activation_dtype or base_policy.activation_dtype,
            attention_weight_dtype=weight_dtype or base_policy.attention_weight_dtype,
            mlp_gate_up_dtype=mlp_gate_up_dtype or base_policy.mlp_gate_up_dtype,
            mlp_down_dtype=mlp_down_dtype or base_policy.mlp_down_dtype,
            kv_cache_dtype=kv_cache_dtype or base_policy.kv_cache_dtype,
            mlp_mul_dtype=base_policy.mlp_mul_dtype,
            mlp_math_fidelity=base_policy.mlp_math_fidelity,
        )

        max_seq_len = int(max_seq_len or hf_config.max_position_embeddings)
        if max_num_blocks is None:
            max_num_blocks = max(1, (max_batch_size * max_seq_len + page_block_size - 1) // page_block_size)
        paged_attention_config = PagedAttentionConfig(block_size=page_block_size, max_num_blocks=max_num_blocks)
        cache_path = Path(cache_dir) if cache_dir is not None else None
        num_devices = mesh_device.get_num_devices()
        tt_ccl = get_tt_ccl(mesh_device)
        num_links = tt_ccl.get_num_links()

        dim = hf_config.hidden_size
        head_dim = hf_config.head_dim
        n_heads = hf_config.num_attention_heads
        n_kv_heads = hf_config.num_key_value_heads
        q_size = n_heads * head_dim
        kv_size = n_kv_heads * head_dim
        qkv_size = q_size + 2 * kv_size
        if n_heads % num_devices != 0 or n_kv_heads % num_devices != 0:
            raise ValueError("Llama 3.1 8B head counts must divide the TP mesh")

        wq_raw = _get_layer_tensor(state_dict, layer_idx, "self_attn.q_proj.weight")
        wk_raw = _get_layer_tensor(state_dict, layer_idx, "self_attn.k_proj.weight")
        wv_raw = _get_layer_tensor(state_dict, layer_idx, "self_attn.v_proj.weight")
        wo_raw = _get_layer_tensor(state_dict, layer_idx, "self_attn.o_proj.weight")

        wq = _reverse_permute(wq_raw, n_heads, q_size, dim).transpose(-2, -1)
        wk = _reverse_permute(wk_raw, n_kv_heads, kv_size, dim).transpose(-2, -1)
        wv = wv_raw.transpose(-2, -1)
        wq_chunks = torch.chunk(wq, num_devices, dim=-1)
        wk_chunks = torch.chunk(wk, num_devices, dim=-1)
        wv_chunks = torch.chunk(wv, num_devices, dim=-1)
        wqkv = torch.cat(
            [torch.cat([wq_chunks[idx], wk_chunks[idx], wv_chunks[idx]], dim=-1) for idx in range(num_devices)],
            dim=-1,
        ).unsqueeze(0).unsqueeze(0)
        wo = wo_raw.transpose(-2, -1).unsqueeze(0).unsqueeze(0)

        dram_grid_width = 8 if not is_blackhole() else mesh_device.dram_grid_size().x
        attention_prefill_grid = _find_prefill_grid(8, dim // TILE_SIZE)
        qkv_size_per_device = qkv_size // num_devices

        @lru_cache
        def attention_xqkv_prefill_prg_config(seq_len: int) -> ttnn.MatmulMultiCoreReuseMultiCastProgramConfig:
            return _with_allowed_worker_cores(
                _matmul_2d_config(
                    m=seq_len,
                    k=dim,
                    n=qkv_size_per_device,
                    grid_size=attention_prefill_grid,
                    in0_block_w=4,
                    per_core_m=max(1, 8 if seq_len >= 2048 else math.ceil(seq_len / TILE_SIZE / 8)),
                    per_core_n=math.ceil(qkv_size_per_device / (TILE_SIZE * dram_grid_width)),
                    fuse_batch=seq_len <= 2048,
                )
            )

        @lru_cache
        def attention_wo_prefill_prg_config(seq_len: int) -> ttnn.MatmulMultiCoreReuseMultiCastProgramConfig:
            return _with_allowed_worker_cores(
                _matmul_2d_config(
                    m=min(seq_len, 1024),
                    k=dim // num_devices,
                    n=dim,
                    grid_size=attention_prefill_grid,
                    in0_block_w=4,
                    out_subblock_w=4,
                    per_core_n=math.ceil(dim / (TILE_SIZE * dram_grid_width)),
                    fuse_batch=seq_len <= 1024,
                )
            )

        decode_all_gather_matmul_grid = ttnn.CoreCoord(8, 1)
        decode_all_gather_matmul_per_core_n = dim // num_devices // TILE_SIZE // (
            decode_all_gather_matmul_grid.x * decode_all_gather_matmul_grid.y
        )
        decode_all_gather_matmul_prg_config = _with_allowed_worker_cores(
            ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                compute_with_storage_grid_size=decode_all_gather_matmul_grid,
                in0_block_w=dim // TILE_SIZE // (decode_all_gather_matmul_grid.x * decode_all_gather_matmul_grid.y),
                out_subblock_h=1,
                out_subblock_w=_get_out_subblock_w(decode_all_gather_matmul_per_core_n, out_subblock_h=1),
                per_core_M=(TILE_SIZE * math.ceil(max_batch_size / TILE_SIZE)) // TILE_SIZE,
                per_core_N=decode_all_gather_matmul_per_core_n,
                fuse_batch=True,
                fused_activation=None,
                mcast_in0=True,
            )
        )

        input_layernorm = _norm_weight(
            state_dict,
            hf_config=hf_config,
            layer_idx=layer_idx,
            name="input_layernorm",
            mesh_device=mesh_device,
            max_batch_size=max_batch_size,
            cache_dir=cache_path,
        )
        post_attention_layernorm = _norm_weight(
            state_dict,
            hf_config=hf_config,
            layer_idx=layer_idx,
            name="post_attention_layernorm",
            mesh_device=mesh_device,
            max_batch_size=max_batch_size,
            cache_dir=cache_path,
        )

        self_attn = Attention1D.from_config(
            Attention1DConfig(
                wqkv=LazyWeight(
                    source=wqkv,
                    dtype=policy.attention_weight_dtype,
                    cache_dir_weight_name=(cache_path, "self_attn_wqkv_multichip") if cache_path else None,
                ),
                wo=LazyWeight(
                    source=wo,
                    dtype=policy.attention_weight_dtype,
                    cache_dir_weight_name=(cache_path, "self_attn_wo_multichip") if cache_path else None,
                ),
                mesh_device=mesh_device,
                tt_ccl=tt_ccl,
                topology=TARGET_TOPOLOGY,
                num_reduce_scatter_links=num_links,
                num_all_gather_links=num_links,
                dim=dim,
                n_heads=n_heads,
                n_kv_heads=n_kv_heads,
                head_dim=head_dim,
                qkv_size=qkv_size,
                max_batch_size=max_batch_size,
                max_seq_len=max_seq_len,
                paged_attention_config=paged_attention_config,
                kv_cache=(
                    LazyWeight(
                        source=zeros_like_paged_cache(paged_attention_config, n_kv_heads // num_devices, head_dim),
                        dtype=policy.kv_cache_dtype,
                    ),
                    LazyWeight(
                        source=zeros_like_paged_cache(paged_attention_config, n_kv_heads // num_devices, head_dim),
                        dtype=policy.kv_cache_dtype,
                    ),
                ),
                kv_cache_dtype=policy.kv_cache_dtype,
                wqkv_dtype=policy.attention_weight_dtype,
                wo_dtype=policy.attention_weight_dtype,
                activation_dtype=policy.activation_dtype,
                scale=head_dim**-0.5,
                prefill_input_memcfg=ttnn.DRAM_MEMORY_CONFIG,
                prefill_xqkv_prg_config=attention_xqkv_prefill_prg_config,
                prefill_wo_prg_config=attention_wo_prefill_prg_config,
                decode_all_gather_matmul_prg_config=decode_all_gather_matmul_prg_config,
                use_fused_all_gather_matmul=True,
            )
        )

        hidden_dim = hf_config.intermediate_size
        padded_hidden_dim = get_padded_hidden_dim(hidden_dim, num_devices, TILE_SIZE)
        if padded_hidden_dim != hidden_dim:
            raise ValueError(f"Unexpected padded hidden dim {padded_hidden_dim}; target path assumes no MLP padding")
        gate = _get_layer_tensor(state_dict, layer_idx, "mlp.gate_proj.weight").transpose(-2, -1).unsqueeze(0).unsqueeze(0)
        up = _get_layer_tensor(state_dict, layer_idx, "mlp.up_proj.weight").transpose(-2, -1).unsqueeze(0).unsqueeze(0)
        down = _get_layer_tensor(state_dict, layer_idx, "mlp.down_proj.weight").transpose(-2, -1).unsqueeze(0).unsqueeze(0)

        hidden_dim_per_device = hidden_dim // num_devices
        gate_up_memcfg = _dram_sharded_weight_memcfg(dim, hidden_dim_per_device, mesh_device)
        down_memcfg = _dram_sharded_weight_memcfg(hidden_dim_per_device, dim, mesh_device)
        if policy.mlp_math_fidelity == ttnn.MathFidelity.LoFi:
            mlp_compute_kernel_cfg = _compute_kernel_config_lofi()
        elif policy.mlp_math_fidelity == ttnn.MathFidelity.HiFi2:
            mlp_compute_kernel_cfg = _compute_kernel_config_hifi2_fp16()
        elif policy.mlp_math_fidelity == ttnn.MathFidelity.HiFi4:
            mlp_compute_kernel_cfg = _compute_kernel_config_hifi4()
        else:
            raise ValueError(f"Unsupported MLP math fidelity: {policy.mlp_math_fidelity}")

        mlp = _TensorParallelMLP(
            gate=LazyWeight(
                source=gate,
                dtype=policy.mlp_gate_up_dtype,
                device=mesh_device,
                mesh_mapper_config=_mesh_mapper_1d(num_devices, -1),
                layout=ttnn.TILE_LAYOUT,
                memory_config=gate_up_memcfg,
                cache_dir_weight_name=(cache_path, "mlp_gate_multichip") if cache_path else None,
            ),
            up=LazyWeight(
                source=up,
                dtype=policy.mlp_gate_up_dtype,
                device=mesh_device,
                mesh_mapper_config=_mesh_mapper_1d(num_devices, -1),
                layout=ttnn.TILE_LAYOUT,
                memory_config=gate_up_memcfg,
                cache_dir_weight_name=(cache_path, "mlp_up_multichip") if cache_path else None,
            ),
            down=LazyWeight(
                source=down,
                dtype=policy.mlp_down_dtype,
                device=mesh_device,
                mesh_mapper_config=_mesh_mapper_1d(num_devices, -2),
                layout=ttnn.TILE_LAYOUT,
                memory_config=down_memcfg,
                cache_dir_weight_name=(cache_path, "mlp_down_multichip") if cache_path else None,
            ),
            dim=dim,
            hidden_dim=hidden_dim,
            max_batch_size=max_batch_size,
            mesh_device=mesh_device,
            tt_ccl=tt_ccl,
            topology=TARGET_TOPOLOGY,
            activation_dtype=policy.activation_dtype,
            mul_dtype=policy.mlp_mul_dtype,
            compute_kernel_config=mlp_compute_kernel_cfg,
            num_reduce_scatter_links=num_links,
        )

        return cls(
            input_layernorm=input_layernorm,
            self_attn=self_attn,
            post_attention_layernorm=post_attention_layernorm,
            mlp=mlp,
            policy=policy,
            mesh_device=mesh_device,
            tt_ccl=tt_ccl,
            topology=TARGET_TOPOLOGY,
            num_all_gather_links=num_links,
        )

    def _all_gather_hidden(self, hidden_shard: ttnn.Tensor, *, mode: str, stage: str) -> ttnn.Tensor:
        memory_config = self.decode_residual_memcfg if mode == "decode" else self.prefill_residual_memcfg
        buffer_key = (mode, stage)
        persistent_output_buffer = self._all_gather_output_buffers.get(buffer_key) if mode == "decode" else None
        gathered = ttnn.experimental.all_gather_async(
            hidden_shard,
            persistent_output_buffer=persistent_output_buffer,
            dim=3,
            multi_device_global_semaphore=self.tt_ccl.get_and_cycle_ag_semaphore_handles(),
            num_links=self.num_all_gather_links,
            topology=self.topology,
            memory_config=memory_config,
            barrier_semaphore=self.tt_ccl.get_and_cycle_barrier_semaphore_handle(),
            chunks_per_sync=MULTICHIP_CCL_CHUNKS_PER_SYNC,
            num_workers_per_link=MULTICHIP_CCL_NUM_WORKERS_PER_LINK,
            num_buffers_per_channel=MULTICHIP_CCL_NUM_BUFFERS_PER_CHANNEL,
        )
        if mode == "decode" and persistent_output_buffer is None:
            self._all_gather_output_buffers[buffer_key] = ttnn.allocate_tensor_on_device(
                gathered.spec, self.mesh_device
            )
        hidden_shard.deallocate(True)
        return gathered

    def release_decode_persistent_buffers(self) -> None:
        for key, tensor in list(self._all_gather_output_buffers.items()):
            if key[0] != "decode":
                continue
            if tensor.is_allocated():
                tensor.deallocate(True)
            del self._all_gather_output_buffers[key]
        self.mlp.release_decode_persistent_buffers()

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
        residual = ttnn.to_memory_config(hidden_states, ttnn.DRAM_MEMORY_CONFIG)
        hidden_states = self.input_layernorm.prefill_forward(residual)
        hidden_states = self.self_attn.prefill_forward(
            hidden_states,
            rot_mats=rot_mats,
            user_id=user_id,
            page_table=page_table,
            chunk_page_table=chunk_page_table,
            chunk_start_idx=chunk_start_idx,
        )
        hidden_states = self._all_gather_hidden(hidden_states, mode="prefill", stage="attention")
        hidden_states = ttnn.add(residual, hidden_states, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        residual = hidden_states
        hidden_states = self.post_attention_layernorm.prefill_forward(hidden_states)
        hidden_states = self.mlp.prefill_forward(hidden_states)
        hidden_states = self._all_gather_hidden(hidden_states, mode="prefill", stage="mlp")
        return ttnn.add(residual, hidden_states, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    def decode_forward(
        self,
        hidden_states: ttnn.Tensor,
        *,
        current_pos: ttnn.Tensor,
        rot_mats: tuple[ttnn.Tensor, ttnn.Tensor],
        page_table: ttnn.Tensor,
    ) -> ttnn.Tensor:
        residual = ttnn.to_memory_config(hidden_states, self.decode_residual_memcfg)
        hidden_states = self.input_layernorm.decode_forward(residual)
        hidden_states = ttnn.to_memory_config(hidden_states, self.self_attn.config.decode_input_memcfg)
        hidden_states = self.self_attn.decode_forward(
            hidden_states,
            current_pos=current_pos,
            rot_mats=rot_mats,
            page_table=page_table,
        )
        hidden_states = self._all_gather_hidden(hidden_states, mode="decode", stage="attention")
        hidden_states = ttnn.add(residual, hidden_states, memory_config=self.decode_residual_memcfg)

        residual = hidden_states
        hidden_states = self.post_attention_layernorm.decode_forward(hidden_states)
        hidden_states = self.mlp.decode_forward(hidden_states)
        hidden_states = self._all_gather_hidden(hidden_states, mode="decode", stage="mlp")
        return ttnn.add(residual, hidden_states, memory_config=self.decode_residual_memcfg)

    def forward(self, hidden_states: ttnn.Tensor, *, mode: str, **kwargs) -> ttnn.Tensor:
        if mode == "prefill":
            return self.prefill_forward(hidden_states, **kwargs)
        if mode == "decode":
            return self.decode_forward(hidden_states, **kwargs)
        raise ValueError(f"Unknown decoder mode {mode!r}; expected 'prefill' or 'decode'.")
