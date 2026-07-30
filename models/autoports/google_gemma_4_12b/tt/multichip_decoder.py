# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""T3K tensor-parallel TTNN decoder layer for ``google/gemma-4-12B``.

This is the multichip-decoder stage for the repo-local autoport pipeline.  It
uses the optimized single-chip decoder as the TP=1 baseline and implements the
target T3K path as 1x8 tensor parallelism with replicated residual activations,
local KV-cache heads, and ring all-reduce after row-parallel projections.
"""

from __future__ import annotations

import importlib.util
import math
from functools import lru_cache
from pathlib import Path

import torch
import ttnn
from models.common.lightweightmodule import LightweightModule
from models.common.tensor_utils import TILE_SIZE
from models.common.utility_functions import is_blackhole
from models.demos.gemma4.config import MeshConfig, ModeConfig
from models.demos.gemma4.tt.attention import Gemma4AttentionConfig
from models.demos.gemma4.tt.attention.kv_cache import init_kv_cache
from models.demos.gemma4.tt.attention.operations import (
    apply_per_head_norm,
    apply_rope,
    effective_block_size,
    split_qkv_heads_decode,
    split_qkv_heads_prefill,
)
from models.demos.gemma4.tt.model_config import Gemma4ModelArgs
from models.demos.gemma4.utils.general_utils import get_cache_file_name
from models.demos.gemma4.utils.substate import substate
from models.tt_transformers.tt.common import PagedAttentionConfig


SUPPORTED_HF_MODEL_ID = "google/gemma-4-12B"
TARGET_MESH_SHAPE = (1, 8)
TARGET_TP = 8
TARGET_TOPOLOGY = ttnn.Topology.Ring
TARGET_FABRIC = "FABRIC_1D_RING"
_SUPPORTED_LAYER_TYPES = ("sliding_attention", "full_attention")
_MAX_MM_SEQ_LEN = 1024
_MAX_QKV_MM_SEQ_LEN = 1024


def _load_optimized_module():
    path = Path(__file__).with_name("optimized_decoder.py")
    spec = importlib.util.spec_from_file_location("gemma4_12b_optimized_decoder_for_multichip", path)
    module = importlib.util.module_from_spec(spec)
    if spec.loader is None:
        raise ImportError(f"cannot load optimized decoder from {path}")
    spec.loader.exec_module(module)
    return module


_optimized = _load_optimized_module()
OptimizedDecoder = _optimized.OptimizedDecoder
OptimizedRMSNorm = _optimized.OptimizedRMSNorm
_as_text_config = _optimized._as_text_config
_compute_kernel_config = _optimized._compute_kernel_config
_create_dram_sharded_mem_config = _optimized._create_dram_sharded_mem_config
_dram_matmul_config = _optimized._dram_matmul_config
_dram_shard_core_grid = _optimized._dram_shard_core_grid
_dram_shard_core_grid_k_n = _optimized._dram_shard_core_grid_k_n
_dtype_name = _optimized._dtype_name
_find_prefill_grid = _optimized._find_prefill_grid
_matmul_config = _optimized._matmul_config
_normalize_layer_state_dict = _optimized._normalize_layer_state_dict
_require_target_config = _optimized._require_target_config
_width_sharded_mem_config = _optimized._width_sharded_mem_config


class RingCCLManager:
    def __init__(self, mesh_device: ttnn.MeshDevice, num_links: int = 1, topology: ttnn.Topology = TARGET_TOPOLOGY):
        self.mesh_device = mesh_device
        self.num_links = num_links
        self.topology = topology


def _mesh_shape_tuple(mesh_device) -> tuple[int, int]:
    shape = getattr(mesh_device, "shape", None)
    if shape is None:
        return (1, 1)
    return tuple(int(x) for x in shape)


def _validate_target_mesh(mesh_device, mesh_config: MeshConfig):
    mesh_shape = _mesh_shape_tuple(mesh_device)
    if mesh_shape != TARGET_MESH_SHAPE or mesh_config.tp != TARGET_TP or mesh_config.tp_axis != 1:
        raise NotImplementedError(
            f"{SUPPORTED_HF_MODEL_ID} multichip decoder targets T3K {TARGET_MESH_SHAPE} TP={TARGET_TP}; "
            f"got mesh_shape={mesh_shape}, tp={mesh_config.tp}, tp_axis={mesh_config.tp_axis}"
        )
    if mesh_device.get_num_devices() != TARGET_TP:
        raise ValueError(f"target mesh must expose {TARGET_TP} devices, got {mesh_device.get_num_devices()}")


def _all_reduce_tp(tensor, mesh_config: MeshConfig, ccl_manager: RingCCLManager, memory_config=None):
    if mesh_config.tp <= 1:
        return tensor
    reduced = ttnn.all_reduce(
        tensor,
        num_links=ccl_manager.num_links,
        topology=ccl_manager.topology,
        cluster_axis=mesh_config.tp_axis,
        memory_config=memory_config if memory_config is not None else tensor.memory_config(),
    )
    tensor.deallocate(True)
    return reduced


def _local_kv_heads(config: Gemma4AttentionConfig, tp: int) -> tuple[int, bool]:
    if config.num_attention_heads % tp != 0:
        raise ValueError(f"num_attention_heads={config.num_attention_heads} must divide TP={tp}")
    if config.num_key_value_heads < tp:
        return 1, True
    if config.num_key_value_heads % tp != 0:
        raise ValueError(f"num_key_value_heads={config.num_key_value_heads} must divide TP={tp}")
    return config.num_key_value_heads // tp, False


def _fused_qkv_weight_for_tp(config: Gemma4AttentionConfig, state_dict: dict, tp: int):
    is_global = config.use_kv_tying
    q_w = state_dict["q_proj.weight"]
    k_w = state_dict["k_proj.weight"]
    v_w = k_w if is_global else state_dict["v_proj.weight"]

    num_q_heads = config.num_attention_heads
    num_kv_heads = config.num_key_value_heads
    head_dim = config.head_dim
    q_per_device = num_q_heads // tp
    local_kv_heads, kv_replicated = _local_kv_heads(config, tp)

    chunks = []
    for mesh_idx in range(tp):
        q_chunk = q_w.chunk(tp, dim=0)[mesh_idx].transpose(-2, -1)
        if kv_replicated:
            kv_idx = (mesh_idx * q_per_device) * num_kv_heads // num_q_heads
            kv_start = kv_idx * head_dim
            kv_end = kv_start + local_kv_heads * head_dim
            k_chunk = k_w[kv_start:kv_end].transpose(-2, -1)
            v_chunk = v_w[kv_start:kv_end].transpose(-2, -1)
        else:
            k_chunk = k_w.chunk(tp, dim=0)[mesh_idx].transpose(-2, -1)
            v_chunk = v_w.chunk(tp, dim=0)[mesh_idx].transpose(-2, -1)
        chunks.append(torch.cat([q_chunk, k_chunk, v_chunk], dim=-1))

    return torch.cat(chunks, dim=-1).unsqueeze(0).unsqueeze(0), chunks[0].shape[-1], kv_replicated


class MultichipGemma4Attention:
    """Tensor-parallel Gemma4 attention using optimized local matmul configs."""

    def __init__(
        self,
        *,
        mesh_device,
        config: Gemma4AttentionConfig,
        state_dict,
        mesh_config: MeshConfig,
        ccl_manager: RingCCLManager,
        layer_idx: int,
        tensor_cache_path=None,
        weight_dtype=ttnn.bfloat16,
        qkv_weight_dtype=None,
        o_weight_dtype=None,
        prefill_weight_dtype=ttnn.bfloat16,
        activation_dtype=ttnn.bfloat16,
        bounded_sliding_kv_cache: bool = False,
    ):
        self.mesh_device = mesh_device
        self.config = config
        self.mesh_config = mesh_config
        self.ccl_manager = ccl_manager
        self.layer_idx = layer_idx
        self.tp = mesh_config.tp
        self.activation_dtype = activation_dtype
        self.weight_dtype = weight_dtype
        self.qkv_weight_dtype = qkv_weight_dtype if qkv_weight_dtype is not None else weight_dtype
        self.o_weight_dtype = o_weight_dtype if o_weight_dtype is not None else weight_dtype
        self.prefill_weight_dtype = prefill_weight_dtype
        self.prefill_uses_default_programs = prefill_weight_dtype is not None
        self.bounded_sliding_kv_cache = (
            bounded_sliding_kv_cache and config.is_sliding and config.sliding_window is not None
        )
        if self.bounded_sliding_kv_cache:
            config.cache_position_modulo = config.sliding_window

        self.hidden_size = config.hidden_size
        self.q_size = config.num_attention_heads * config.head_dim
        self.local_q_size = self.q_size // self.tp
        self.local_kv_heads, self.kv_replicated = _local_kv_heads(config, self.tp)
        self.local_kv_size = self.local_kv_heads * config.head_dim
        self.local_qkv_size = self.local_q_size + 2 * self.local_kv_size
        self.qkv_size = self.local_qkv_size * self.tp

        self.qkv_prefill_compute_config = _compute_kernel_config(ttnn.MathFidelity.HiFi2, fp32_dest_acc_en=False)
        if config.is_sliding:
            self.qkv_decode_compute_config = _compute_kernel_config(ttnn.MathFidelity.HiFi2, fp32_dest_acc_en=False)
            self.qkv_decode_long_compute_config = _compute_kernel_config(
                ttnn.MathFidelity.HiFi3, fp32_dest_acc_en=True
            )
            self.o_compute_config = _compute_kernel_config(ttnn.MathFidelity.HiFi3, fp32_dest_acc_en=True)
        else:
            self.qkv_decode_compute_config = _compute_kernel_config(ttnn.MathFidelity.HiFi3, fp32_dest_acc_en=True)
            self.qkv_decode_long_compute_config = self.qkv_decode_compute_config
            self.o_compute_config = _compute_kernel_config(ttnn.MathFidelity.HiFi2, fp32_dest_acc_en=False)
        self.sdpa_prefill_compute_config = _compute_kernel_config(ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True)
        self.sdpa_decode_compute_config = None

        self._init_configs()
        self._load_weights(state_dict, tensor_cache_path)

    def _init_configs(self):
        tile_rows = TILE_SIZE
        qkv_grid = _dram_shard_core_grid_k_n(self.hidden_size, self.local_qkv_size)
        out_grid = _dram_shard_core_grid_k_n(self.local_q_size, self.hidden_size)
        residual_grid = _dram_shard_core_grid(self.hidden_size)

        self.decode_input_memcfg = _width_sharded_mem_config(rows=tile_rows, width=self.hidden_size, grid=qkv_grid)
        self.decode_o_input_memcfg = _width_sharded_mem_config(rows=tile_rows, width=self.local_q_size, grid=out_grid)
        self.decode_residual_memcfg = _width_sharded_mem_config(
            rows=tile_rows, width=self.hidden_size, grid=residual_grid
        )
        self.decode_qkv_program_config = _dram_matmul_config(
            m=tile_rows,
            k=self.hidden_size,
            n=self.local_qkv_size,
            num_cores=qkv_grid.num_cores,
        )
        self.decode_o_program_config = _dram_matmul_config(
            m=tile_rows,
            k=self.local_q_size,
            n=self.hidden_size,
            num_cores=out_grid.num_cores,
        )

        device_grid = self.mesh_device.compute_with_storage_grid_size()
        if self.config.head_dim >= 512:
            sdpa_grid = ttnn.CoreCoord(min(8, device_grid.x), min(4, device_grid.y))
        else:
            sdpa_grid = ttnn.CoreCoord(device_grid.x, device_grid.y)
        self.decode_sdpa_program_config = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=sdpa_grid,
            q_chunk_size=32,
            k_chunk_size=64,
            exp_approx_mode=False,
        )

        dram_shard_grid_width = 8 if not is_blackhole() else self.mesh_device.dram_grid_size().x
        prefill_rows = 8
        qkv_prefill_grid = _find_prefill_grid(prefill_rows, self.hidden_size // TILE_SIZE)
        out_prefill_grid = _find_prefill_grid(prefill_rows, self.local_q_size // TILE_SIZE)

        @lru_cache
        def qkv_prefill_program_config(seq_len: int):
            return _matmul_config(
                m=min(seq_len, _MAX_QKV_MM_SEQ_LEN),
                k=self.hidden_size,
                n=self.local_qkv_size,
                grid_size=qkv_prefill_grid,
                in0_block_w=1,
                per_core_m=max(1, 8 if seq_len >= _MAX_QKV_MM_SEQ_LEN else math.ceil(seq_len / TILE_SIZE / 8)),
                per_core_n=math.ceil(self.local_qkv_size / (TILE_SIZE * dram_shard_grid_width)),
                fuse_batch=seq_len <= _MAX_QKV_MM_SEQ_LEN,
            )

        @lru_cache
        def out_prefill_program_config(seq_len: int):
            return _matmul_config(
                m=min(seq_len, _MAX_MM_SEQ_LEN),
                k=self.local_q_size,
                n=self.hidden_size,
                grid_size=out_prefill_grid,
                in0_block_w=1,
                per_core_n=math.ceil(self.hidden_size / (TILE_SIZE * dram_shard_grid_width)),
                fuse_batch=seq_len <= _MAX_MM_SEQ_LEN,
            )

        @lru_cache
        def sdpa_prefill_program_config(seq_len: int):
            q_chunk = 256 if seq_len >= 2048 else 64
            k_chunk = 256 if seq_len >= 2048 else 64
            return ttnn.SDPAProgramConfig(
                compute_with_storage_grid_size=(8, 8),
                exp_approx_mode=False,
                q_chunk_size=q_chunk,
                k_chunk_size=k_chunk,
            )

        self.prefill_qkv_program_config = qkv_prefill_program_config
        self.prefill_o_program_config = out_prefill_program_config
        self.prefill_sdpa_program_config = sdpa_prefill_program_config

    def _load_weights(self, state_dict, tensor_cache_path):
        if state_dict:
            qkv, local_qkv_size, kv_replicated = _fused_qkv_weight_for_tp(self.config, state_dict, self.tp)
            if local_qkv_size != self.local_qkv_size or kv_replicated != self.kv_replicated:
                raise AssertionError("attention local QKV packing mismatch")
            o_w = state_dict["o_proj.weight"].transpose(-2, -1).unsqueeze(0).unsqueeze(0)
            q_norm_w = state_dict["q_norm.weight"].reshape(1, 1, -1, TILE_SIZE)
            k_norm_w = state_dict["k_norm.weight"].reshape(1, 1, -1, TILE_SIZE)
        else:
            qkv = None
            o_w = None
            q_norm_w = None
            k_norm_w = None

        cache_root = str(tensor_cache_path) if tensor_cache_path is not None else None
        qkv_dtype_suffix = _dtype_name(self.qkv_weight_dtype)
        o_dtype_suffix = _dtype_name(self.o_weight_dtype)
        tp_suffix = f"_tp{self.tp}_ring"
        col_mapper = self.mesh_config.column_parallel(self.mesh_device)
        row_mapper = self.mesh_config.row_parallel(self.mesh_device)
        replicate_mapper = ttnn.ReplicateTensorToMesh(self.mesh_device)

        self.wqkv = ttnn.as_tensor(
            qkv,
            device=self.mesh_device,
            dtype=self.qkv_weight_dtype,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=col_mapper,
            cache_file_name=get_cache_file_name(cache_root, f"wqkv_dram_sharded{tp_suffix}_{qkv_dtype_suffix}"),
            memory_config=_create_dram_sharded_mem_config(
                mesh_device=self.mesh_device, k=self.hidden_size, n=self.local_qkv_size
            ),
        )
        self.o_proj = ttnn.as_tensor(
            o_w,
            device=self.mesh_device,
            dtype=self.o_weight_dtype,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=row_mapper,
            cache_file_name=get_cache_file_name(cache_root, f"o_proj_dram_sharded{tp_suffix}_{o_dtype_suffix}"),
            memory_config=_create_dram_sharded_mem_config(
                mesh_device=self.mesh_device, k=self.local_q_size, n=self.hidden_size
            ),
        )
        if self.prefill_weight_dtype is None:
            self.prefill_wqkv = self.wqkv
            self.prefill_o_proj = self.o_proj
        else:
            prefill_dtype_suffix = _dtype_name(self.prefill_weight_dtype)
            self.prefill_wqkv = ttnn.as_tensor(
                qkv,
                device=self.mesh_device,
                dtype=self.prefill_weight_dtype,
                layout=ttnn.TILE_LAYOUT,
                mesh_mapper=col_mapper,
                cache_file_name=get_cache_file_name(cache_root, f"wqkv_prefill{tp_suffix}_{prefill_dtype_suffix}"),
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            self.prefill_o_proj = ttnn.as_tensor(
                o_w,
                device=self.mesh_device,
                dtype=self.prefill_weight_dtype,
                layout=ttnn.TILE_LAYOUT,
                mesh_mapper=row_mapper,
                cache_file_name=get_cache_file_name(cache_root, f"o_proj_prefill{tp_suffix}_{prefill_dtype_suffix}"),
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        self.q_norm_weight = ttnn.as_tensor(
            q_norm_w,
            device=self.mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=replicate_mapper,
            cache_file_name=get_cache_file_name(cache_root, f"q_norm.weight{tp_suffix}"),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        self.k_norm_weight = ttnn.as_tensor(
            k_norm_w,
            device=self.mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=replicate_mapper,
            cache_file_name=get_cache_file_name(cache_root, f"k_norm.weight{tp_suffix}"),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def __call__(
        self,
        hidden_states,
        *,
        rope_mats,
        position_idx=None,
        page_table=None,
        kv_cache=None,
        is_decode=True,
        token_index=None,
        position_idx_cache=None,
        decode_position=None,
    ):
        if is_decode:
            return self.decode_forward(
                hidden_states,
                cos_cache=rope_mats[0],
                sin_cache=rope_mats[1],
                kv_cache=kv_cache,
                position_idx=position_idx,
                token_index=token_index,
                page_table=page_table,
                position_idx_cache=position_idx_cache,
                decode_position=decode_position,
            )
        return self.prefill_forward(
            hidden_states,
            cos_cache=rope_mats[0],
            sin_cache=rope_mats[1],
            kv_cache=kv_cache,
            page_table=page_table,
        )

    def prefill_forward(self, hidden_states, *, cos_cache, sin_cache, kv_cache, page_table=None, user_id=0):
        seq_len = hidden_states.shape[-2]
        original_shape = hidden_states.shape
        if seq_len > _MAX_QKV_MM_SEQ_LEN:
            if seq_len % _MAX_QKV_MM_SEQ_LEN != 0:
                raise ValueError(f"seq_len {seq_len} must be divisible by {_MAX_QKV_MM_SEQ_LEN}")
            hidden_states = ttnn.reshape(hidden_states, [1, seq_len // _MAX_QKV_MM_SEQ_LEN, _MAX_QKV_MM_SEQ_LEN, -1])

        if self.prefill_uses_default_programs:
            xqkv = ttnn.linear(
                hidden_states,
                self.prefill_wqkv,
                dtype=self.activation_dtype,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        else:
            xqkv = ttnn.linear(
                hidden_states,
                self.prefill_wqkv,
                dtype=self.activation_dtype,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                compute_kernel_config=self.qkv_prefill_compute_config,
                program_config=self.prefill_qkv_program_config(seq_len),
            )
        hidden_states.deallocate(True)
        if seq_len > _MAX_QKV_MM_SEQ_LEN:
            xqkv = ttnn.reshape(xqkv, [1, 1, seq_len, -1])

        tt_q, tt_k, tt_v = split_qkv_heads_prefill(
            xqkv, self.config, self.config.use_kv_tying, tp=self.tp, kv_replicated=self.kv_replicated
        )
        xqkv.deallocate(True)

        tt_q = apply_per_head_norm(tt_q, self.q_norm_weight, self.config.rms_norm_eps, with_scale=True)
        tt_k = apply_per_head_norm(tt_k, self.k_norm_weight, self.config.rms_norm_eps, with_scale=True)
        tt_v = apply_per_head_norm(tt_v, None, self.config.rms_norm_eps, with_scale=False)

        tt_q = apply_rope(tt_q, cos_cache, sin_cache)
        tt_k = apply_rope(tt_k, cos_cache, sin_cache)

        if kv_cache is not None:
            k_cache, v_cache = kv_cache
            if page_table is not None:
                eff_bs = effective_block_size(k_cache, self.config.head_dim, self.local_kv_heads)
                paged_modulo_kwargs = (
                    {"cache_position_modulo": self.config.cache_position_modulo}
                    if self.config.cache_position_modulo is not None
                    else {}
                )
                ttnn.experimental.paged_fill_cache(
                    k_cache, tt_k, page_table, batch_idx=user_id, block_size=eff_bs, **paged_modulo_kwargs
                )
                ttnn.experimental.paged_fill_cache(
                    v_cache, tt_v, page_table, batch_idx=user_id, block_size=eff_bs, **paged_modulo_kwargs
                )
            else:
                ttnn.fill_cache(k_cache, tt_k, batch_idx=user_id)
                ttnn.fill_cache(v_cache, tt_v, batch_idx=user_id)

        sliding_window = self.config.sliding_window if self.config.is_sliding else None
        if self.prefill_uses_default_programs:
            tt_sdpa = ttnn.transformer.scaled_dot_product_attention(
                tt_q,
                tt_k,
                tt_v,
                is_causal=True,
                scale=1.0,
                sliding_window_size=sliding_window,
            )
        else:
            tt_sdpa = ttnn.transformer.scaled_dot_product_attention(
                tt_q,
                tt_k,
                tt_v,
                is_causal=True,
                scale=1.0,
                sliding_window_size=sliding_window,
                compute_kernel_config=self.sdpa_prefill_compute_config,
                program_config=self.prefill_sdpa_program_config(seq_len),
            )
        tt_q.deallocate(True)
        tt_k.deallocate(True)
        tt_v.deallocate(True)

        tt_out = ttnn.experimental.nlp_concat_heads(tt_sdpa, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        tt_sdpa.deallocate(True)
        if seq_len > _MAX_MM_SEQ_LEN:
            tt_out = ttnn.reshape(tt_out, [1, seq_len // _MAX_MM_SEQ_LEN, _MAX_MM_SEQ_LEN, -1])
        if self.prefill_uses_default_programs:
            tt_out = ttnn.linear(
                tt_out,
                self.prefill_o_proj,
                dtype=self.activation_dtype,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        else:
            tt_out = ttnn.linear(
                tt_out,
                self.prefill_o_proj,
                dtype=self.activation_dtype,
                compute_kernel_config=self.o_compute_config,
                program_config=self.prefill_o_program_config(seq_len),
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        if seq_len > _MAX_MM_SEQ_LEN:
            tt_out = ttnn.reshape(tt_out, original_shape)
        return _all_reduce_tp(tt_out, self.mesh_config, self.ccl_manager, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    def decode_forward(
        self,
        hidden_states,
        *,
        cos_cache,
        sin_cache,
        kv_cache,
        position_idx,
        token_index,
        page_table=None,
        position_idx_cache=None,
        decode_position=None,
    ):
        hidden_states = ttnn.to_memory_config(hidden_states, self.decode_input_memcfg)
        qkv_compute_config = self.qkv_decode_compute_config
        if (
            self.config.is_sliding
            and decode_position is not None
            and self.config.sliding_window is not None
            and decode_position >= self.config.sliding_window
        ):
            qkv_compute_config = self.qkv_decode_long_compute_config
        xqkv = ttnn.linear(
            hidden_states,
            self.wqkv,
            dtype=self.activation_dtype,
            memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
            compute_kernel_config=qkv_compute_config,
            program_config=self.decode_qkv_program_config,
        )
        hidden_states.deallocate(True)
        if xqkv.is_sharded():
            xqkv_interleaved = ttnn.sharded_to_interleaved(xqkv, ttnn.L1_MEMORY_CONFIG, ttnn.bfloat16)
            xqkv.deallocate(True)
            xqkv = xqkv_interleaved

        tt_q, tt_k, tt_v = split_qkv_heads_decode(
            xqkv, self.config, self.config.use_kv_tying, tp=self.tp, kv_replicated=self.kv_replicated
        )
        xqkv.deallocate(True)

        q_sharded_mem = tt_q.memory_config()
        tt_q = ttnn.to_memory_config(tt_q, ttnn.DRAM_MEMORY_CONFIG)
        tt_k = ttnn.to_memory_config(tt_k, ttnn.DRAM_MEMORY_CONFIG)
        tt_v = ttnn.to_memory_config(tt_v, ttnn.DRAM_MEMORY_CONFIG)
        tt_q = apply_per_head_norm(tt_q, self.q_norm_weight, self.config.rms_norm_eps, with_scale=True)
        tt_k = apply_per_head_norm(tt_k, self.k_norm_weight, self.config.rms_norm_eps, with_scale=True)
        tt_v = apply_per_head_norm(tt_v, None, self.config.rms_norm_eps, with_scale=False)

        use_embedding_rope = len(cos_cache.shape) == 2
        if use_embedding_rope:
            cos_pos = ttnn.embedding(position_idx, cos_cache, layout=ttnn.TILE_LAYOUT)
            sin_pos = ttnn.embedding(position_idx, sin_cache, layout=ttnn.TILE_LAYOUT)
            cos_pos = ttnn.unsqueeze_to_4D(cos_pos)
            sin_pos = ttnn.unsqueeze_to_4D(sin_pos)
            tt_q = apply_rope(tt_q, cos_pos, sin_pos, token_index=0)
            tt_k = apply_rope(tt_k, cos_pos, sin_pos, token_index=0)
        else:
            tt_q = apply_rope(tt_q, cos_cache, sin_cache, token_index=token_index)
            tt_k = apply_rope(tt_k, cos_cache, sin_cache, token_index=token_index)

        cache_pos = position_idx_cache if position_idx_cache is not None else position_idx
        paged_modulo_kwargs = (
            {"cache_position_modulo": self.config.cache_position_modulo}
            if self.config.cache_position_modulo is not None
            else {}
        )
        if kv_cache is not None:
            k_cache, v_cache = kv_cache
            tt_k = ttnn.to_memory_config(tt_k, q_sharded_mem)
            tt_v = ttnn.to_memory_config(tt_v, q_sharded_mem)
            if page_table is not None:
                eff_bs = effective_block_size(k_cache, self.config.head_dim, self.local_kv_heads)
                ttnn.experimental.paged_update_cache(
                    k_cache,
                    tt_k,
                    update_idxs_tensor=cache_pos,
                    page_table=page_table,
                    block_size=eff_bs,
                    num_kv_heads=self.local_kv_heads,
                    **paged_modulo_kwargs,
                )
                ttnn.experimental.paged_update_cache(
                    v_cache,
                    tt_v,
                    update_idxs_tensor=cache_pos,
                    page_table=page_table,
                    block_size=eff_bs,
                    num_kv_heads=self.local_kv_heads,
                    **paged_modulo_kwargs,
                )
            else:
                ttnn.experimental.paged_update_cache(k_cache, tt_k, update_idxs_tensor=cache_pos)
                ttnn.experimental.paged_update_cache(v_cache, tt_v, update_idxs_tensor=cache_pos)
        else:
            k_cache = tt_k
            v_cache = tt_v

        sliding_window = self.config.sliding_window if self.config.is_sliding else None
        if page_table is not None:
            tt_sdpa = ttnn.transformer.paged_scaled_dot_product_attention_decode(
                tt_q,
                k_cache,
                v_cache,
                cur_pos_tensor=cache_pos,
                page_table_tensor=page_table,
                scale=1.0,
                sliding_window_size=sliding_window,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                program_config=self.decode_sdpa_program_config,
                compute_kernel_config=self.sdpa_decode_compute_config,
                block_size=effective_block_size(k_cache, self.config.head_dim, self.local_kv_heads),
                num_kv_heads=self.local_kv_heads,
                **paged_modulo_kwargs,
            )
        else:
            tt_sdpa = ttnn.transformer.scaled_dot_product_attention_decode(
                tt_q,
                k_cache,
                v_cache,
                cur_pos_tensor=cache_pos,
                scale=1.0,
                sliding_window_size=sliding_window,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                program_config=self.decode_sdpa_program_config,
                compute_kernel_config=self.sdpa_decode_compute_config,
            )
        tt_q.deallocate(True)

        tt_out = ttnn.transpose(tt_sdpa, 1, 2)
        tt_sdpa.deallocate(True)
        tt_out = ttnn.experimental.nlp_concat_heads(tt_out, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        tt_out = ttnn.to_memory_config(tt_out, self.decode_o_input_memcfg)
        tt_out = ttnn.linear(
            tt_out,
            self.o_proj,
            dtype=self.activation_dtype,
            compute_kernel_config=self.o_compute_config,
            program_config=self.decode_o_program_config,
            memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
        )
        tt_out = _all_reduce_tp(tt_out, self.mesh_config, self.ccl_manager)
        return ttnn.to_memory_config(tt_out, self.decode_residual_memcfg)


class MultichipSharedMLP:
    """Gemma GeGLU MLP with TP8 gate/up column sharding and down row sharding."""

    def __init__(
        self,
        *,
        mesh_device,
        hf_config,
        state_dict,
        mesh_config: MeshConfig,
        ccl_manager: RingCCLManager,
        tensor_cache_path=None,
        dtype=ttnn.bfloat8_b,
        down_dtype=None,
        decode_dtype=None,
        decode_down_dtype=None,
        prefill_dtype=ttnn.bfloat16,
        activation_dtype=ttnn.bfloat16,
        fuse_gelu_mul: bool = True,
    ):
        self.mesh_device = mesh_device
        self.mesh_config = mesh_config
        self.ccl_manager = ccl_manager
        self.tp = mesh_config.tp
        self.hidden_size = hf_config.hidden_size
        self.intermediate_size = hf_config.intermediate_size
        if self.intermediate_size % self.tp != 0:
            raise ValueError(f"intermediate_size={self.intermediate_size} must divide TP={self.tp}")
        self.local_intermediate_size = self.intermediate_size // self.tp
        self.dtype = dtype
        self.down_dtype = down_dtype if down_dtype is not None else dtype
        self.decode_dtype = decode_dtype if decode_dtype is not None else dtype
        self.decode_down_dtype = (
            decode_down_dtype
            if decode_down_dtype is not None
            else (self.decode_dtype if decode_dtype is not None else self.down_dtype)
        )
        self.prefill_dtype = prefill_dtype
        self.activation_dtype = activation_dtype
        self.fuse_gelu_mul = fuse_gelu_mul
        self.linear_compute_config = _compute_kernel_config(ttnn.MathFidelity.HiFi2, fp32_dest_acc_en=False)

        gate_grid = _dram_shard_core_grid_k_n(self.hidden_size, self.local_intermediate_size)
        down_grid = _dram_shard_core_grid_k_n(self.local_intermediate_size, self.hidden_size)
        residual_grid = _dram_shard_core_grid(self.hidden_size)
        tile_rows = TILE_SIZE
        self.decode_input_memcfg = _width_sharded_mem_config(rows=tile_rows, width=self.hidden_size, grid=gate_grid)
        self.decode_mlp2_input_memcfg = _width_sharded_mem_config(
            rows=tile_rows, width=self.local_intermediate_size, grid=down_grid
        )
        self.decode_residual_memcfg = _width_sharded_mem_config(
            rows=tile_rows, width=self.hidden_size, grid=residual_grid
        )

        self.decode_gate_up_program_config = _dram_matmul_config(
            m=tile_rows,
            k=self.hidden_size,
            n=self.local_intermediate_size,
            num_cores=gate_grid.num_cores,
        )
        self.decode_down_program_config = _dram_matmul_config(
            m=tile_rows,
            k=self.local_intermediate_size,
            n=self.hidden_size,
            num_cores=down_grid.num_cores,
        )

        dram_shard_grid_width = 8 if not is_blackhole() else mesh_device.dram_grid_size().x
        prefill_rows = 8
        gate_up_prefill_grid = _find_prefill_grid(prefill_rows, self.hidden_size // TILE_SIZE)
        down_prefill_grid = _find_prefill_grid(prefill_rows, self.local_intermediate_size // TILE_SIZE)

        @lru_cache
        def gate_up_prefill_program_config(seq_len: int):
            return _matmul_config(
                m=min(seq_len, _MAX_MM_SEQ_LEN),
                k=self.hidden_size,
                n=self.local_intermediate_size,
                grid_size=gate_up_prefill_grid,
                per_core_n=math.ceil(self.local_intermediate_size / (TILE_SIZE * dram_shard_grid_width)),
                fuse_batch=seq_len <= _MAX_MM_SEQ_LEN,
            )

        @lru_cache
        def down_prefill_program_config(seq_len: int):
            return _matmul_config(
                m=min(seq_len, _MAX_MM_SEQ_LEN),
                k=self.local_intermediate_size,
                n=self.hidden_size,
                grid_size=down_prefill_grid,
                per_core_n=math.ceil(self.hidden_size / (TILE_SIZE * dram_shard_grid_width)),
                fuse_batch=seq_len <= _MAX_MM_SEQ_LEN,
            )

        self.prefill_gate_up_program_config = gate_up_prefill_program_config
        self.prefill_down_program_config = down_prefill_program_config
        self._load_weights(state_dict, tensor_cache_path)

    def _load_weights(self, state_dict, tensor_cache_path):
        if state_dict:
            gate_proj_weight = state_dict["gate_proj.weight"].transpose(-2, -1).unsqueeze(0).unsqueeze(0)
            up_proj_weight = state_dict["up_proj.weight"].transpose(-2, -1).unsqueeze(0).unsqueeze(0)
            down_proj_weight = state_dict["down_proj.weight"].transpose(-2, -1).unsqueeze(0).unsqueeze(0)
        else:
            gate_proj_weight = None
            up_proj_weight = None
            down_proj_weight = None

        col_mapper = self.mesh_config.column_parallel(self.mesh_device)
        row_mapper = self.mesh_config.row_parallel(self.mesh_device)
        gate_up_memcfg = _create_dram_sharded_mem_config(
            mesh_device=self.mesh_device, k=self.hidden_size, n=self.local_intermediate_size
        )
        down_memcfg = _create_dram_sharded_mem_config(
            mesh_device=self.mesh_device, k=self.local_intermediate_size, n=self.hidden_size
        )
        cache_root = str(tensor_cache_path) if tensor_cache_path is not None else None
        gate_suffix = _dtype_name(self.dtype)
        down_suffix = _dtype_name(self.down_dtype)
        prefill_suffix = _dtype_name(self.prefill_dtype)
        tp_suffix = f"_tp{self.tp}_ring"

        self.prefill_gate_proj = ttnn.as_tensor(
            gate_proj_weight,
            device=self.mesh_device,
            dtype=self.prefill_dtype,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=col_mapper,
            cache_file_name=get_cache_file_name(cache_root, f"gate_proj.weight_prefill{tp_suffix}_{prefill_suffix}"),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        self.prefill_up_proj = ttnn.as_tensor(
            up_proj_weight,
            device=self.mesh_device,
            dtype=self.prefill_dtype,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=col_mapper,
            cache_file_name=get_cache_file_name(cache_root, f"up_proj.weight_prefill{tp_suffix}_{prefill_suffix}"),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        self.prefill_down_proj = ttnn.as_tensor(
            down_proj_weight,
            device=self.mesh_device,
            dtype=self.prefill_dtype,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=row_mapper,
            cache_file_name=get_cache_file_name(cache_root, f"down_proj.weight_prefill{tp_suffix}_{prefill_suffix}"),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        self.gate_proj = ttnn.as_tensor(
            gate_proj_weight,
            device=self.mesh_device,
            dtype=self.dtype,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=col_mapper,
            cache_file_name=get_cache_file_name(cache_root, f"gate_proj.weight_dram_sharded{tp_suffix}_{gate_suffix}"),
            memory_config=gate_up_memcfg,
        )
        self.up_proj = ttnn.as_tensor(
            up_proj_weight,
            device=self.mesh_device,
            dtype=self.dtype,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=col_mapper,
            cache_file_name=get_cache_file_name(cache_root, f"up_proj.weight_dram_sharded{tp_suffix}_{gate_suffix}"),
            memory_config=gate_up_memcfg,
        )
        self.down_proj = ttnn.as_tensor(
            down_proj_weight,
            device=self.mesh_device,
            dtype=self.down_dtype,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=row_mapper,
            cache_file_name=get_cache_file_name(cache_root, f"down_proj.weight_dram_sharded{tp_suffix}_{down_suffix}"),
            memory_config=down_memcfg,
        )
        if self.decode_dtype == self.dtype:
            self.decode_gate_proj = self.gate_proj
            self.decode_up_proj = self.up_proj
        else:
            decode_gate_suffix = _dtype_name(self.decode_dtype)
            self.decode_gate_proj = ttnn.as_tensor(
                gate_proj_weight,
                device=self.mesh_device,
                dtype=self.decode_dtype,
                layout=ttnn.TILE_LAYOUT,
                mesh_mapper=col_mapper,
                cache_file_name=get_cache_file_name(
                    cache_root, f"gate_proj.weight_decode_dram_sharded{tp_suffix}_{decode_gate_suffix}"
                ),
                memory_config=gate_up_memcfg,
            )
            self.decode_up_proj = ttnn.as_tensor(
                up_proj_weight,
                device=self.mesh_device,
                dtype=self.decode_dtype,
                layout=ttnn.TILE_LAYOUT,
                mesh_mapper=col_mapper,
                cache_file_name=get_cache_file_name(
                    cache_root, f"up_proj.weight_decode_dram_sharded{tp_suffix}_{decode_gate_suffix}"
                ),
                memory_config=gate_up_memcfg,
            )
        if self.decode_down_dtype == self.down_dtype:
            self.decode_down_proj = self.down_proj
        else:
            decode_down_suffix = _dtype_name(self.decode_down_dtype)
            self.decode_down_proj = ttnn.as_tensor(
                down_proj_weight,
                device=self.mesh_device,
                dtype=self.decode_down_dtype,
                layout=ttnn.TILE_LAYOUT,
                mesh_mapper=row_mapper,
                cache_file_name=get_cache_file_name(
                    cache_root, f"down_proj.weight_decode_dram_sharded{tp_suffix}_{decode_down_suffix}"
                ),
                memory_config=down_memcfg,
            )

    def __call__(self, hidden_states, *, is_decode: bool):
        if is_decode:
            return self.decode_forward(hidden_states)
        return self.prefill_forward(hidden_states)

    def prefill_forward(self, hidden_states):
        seq_len = hidden_states.shape[-2]
        original_shape = hidden_states.shape
        if seq_len > _MAX_MM_SEQ_LEN:
            if seq_len % _MAX_MM_SEQ_LEN != 0:
                raise ValueError(f"seq_len {seq_len} must be divisible by {_MAX_MM_SEQ_LEN}")
            hidden_states = ttnn.reshape(hidden_states, [1, seq_len // _MAX_MM_SEQ_LEN, _MAX_MM_SEQ_LEN, -1])

        gate = ttnn.linear(
            hidden_states,
            self.prefill_gate_proj,
            dtype=self.activation_dtype,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        up = ttnn.linear(
            hidden_states,
            self.prefill_up_proj,
            dtype=self.activation_dtype,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        hidden_states.deallocate(True)

        if self.fuse_gelu_mul:
            hidden = ttnn.mul(
                gate,
                up,
                input_tensor_a_activations=[ttnn.UnaryOpType.GELU],
                dtype=self.activation_dtype,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        else:
            gate = ttnn.gelu(gate, fast_and_approximate_mode=True)
            hidden = ttnn.mul(gate, up, dtype=self.activation_dtype, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        gate.deallocate(True)
        up.deallocate(True)

        output = ttnn.linear(
            hidden,
            self.prefill_down_proj,
            dtype=self.activation_dtype,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        hidden.deallocate(True)
        output = _all_reduce_tp(output, self.mesh_config, self.ccl_manager, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        if seq_len > _MAX_MM_SEQ_LEN:
            output = ttnn.reshape(output, original_shape)
        return output

    def decode_forward(self, hidden_states):
        hidden_states = ttnn.to_memory_config(hidden_states, self.decode_input_memcfg)
        gate = ttnn.linear(
            hidden_states,
            self.decode_gate_proj,
            dtype=self.activation_dtype,
            compute_kernel_config=self.linear_compute_config,
            program_config=self.decode_gate_up_program_config,
            memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
        )
        up = ttnn.linear(
            hidden_states,
            self.decode_up_proj,
            dtype=self.activation_dtype,
            compute_kernel_config=self.linear_compute_config,
            program_config=self.decode_gate_up_program_config,
            memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
        )
        hidden_states.deallocate(True)

        if self.fuse_gelu_mul:
            hidden = ttnn.mul(
                gate,
                up,
                input_tensor_a_activations=[ttnn.UnaryOpType.GELU],
                dtype=self.activation_dtype,
                memory_config=gate.memory_config(),
            )
        else:
            gate = ttnn.gelu(gate, fast_and_approximate_mode=True)
            hidden = ttnn.mul(gate, up, dtype=self.activation_dtype, memory_config=gate.memory_config())
        gate.deallocate(True)
        up.deallocate(True)

        hidden = ttnn.to_memory_config(hidden, self.decode_mlp2_input_memcfg)
        output = ttnn.linear(
            hidden,
            self.decode_down_proj,
            dtype=self.activation_dtype,
            compute_kernel_config=self.linear_compute_config,
            program_config=self.decode_down_program_config,
            memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
        )
        hidden.deallocate(True)
        output = _all_reduce_tp(output, self.mesh_config, self.ccl_manager)
        return ttnn.to_memory_config(output, self.decode_residual_memcfg)


class MultichipDecoder(LightweightModule):
    """Dense Gemma 4 TP8 decoder layer for T3K."""

    def __init__(
        self,
        *,
        mesh_device,
        hf_config: Gemma4ModelArgs,
        layer_idx: int,
        layer_state: dict,
        mesh_config: MeshConfig,
        ccl_manager: RingCCLManager,
        dtype=ttnn.bfloat16,
        attention_dtype=None,
        attention_qkv_dtype=None,
        attention_o_dtype=None,
        shared_mlp_dtype=ttnn.bfloat8_b,
        shared_mlp_down_dtype=None,
        shared_mlp_decode_dtype=None,
        shared_mlp_decode_down_dtype=None,
        kv_cache_dtype=ttnn.bfloat16,
        tensor_cache_path: str | Path | None = None,
        bounded_sliding_kv_cache: bool = False,
        fuse_mlp_gelu: bool = True,
        decode_norm_sharded: bool = True,
    ):
        _validate_target_mesh(mesh_device, mesh_config)
        self.mesh_device = mesh_device
        self.hf_config = hf_config
        self.layer_idx = layer_idx
        self.mesh_config = mesh_config
        self.ccl_manager = ccl_manager
        self.hidden_size = hf_config.hidden_size
        self.layer_type = hf_config.layer_types[layer_idx]
        self.attention_config = Gemma4AttentionConfig(hf_config, layer_idx)
        self.kv_cache_dtype = kv_cache_dtype
        self.dtype = dtype
        if attention_dtype is None:
            attention_dtype = ttnn.bfloat8_b if self.layer_type == "sliding_attention" else ttnn.bfloat16
        attention_qkv_dtype = attention_qkv_dtype if attention_qkv_dtype is not None else attention_dtype
        if attention_o_dtype is None:
            attention_o_dtype = ttnn.bfloat16 if self.layer_type == "sliding_attention" else attention_dtype
        self.attention_dtype = attention_dtype
        self.attention_qkv_dtype = attention_qkv_dtype
        self.attention_o_dtype = attention_o_dtype
        self.shared_mlp_dtype = shared_mlp_dtype
        self.shared_mlp_down_dtype = shared_mlp_down_dtype if shared_mlp_down_dtype is not None else shared_mlp_dtype
        self.shared_mlp_decode_dtype = (
            shared_mlp_decode_dtype if shared_mlp_decode_dtype is not None else shared_mlp_dtype
        )
        self.shared_mlp_decode_down_dtype = (
            shared_mlp_decode_down_dtype
            if shared_mlp_decode_down_dtype is not None
            else (self.shared_mlp_decode_dtype if shared_mlp_decode_dtype is not None else self.shared_mlp_down_dtype)
        )
        self.fuse_mlp_gelu = fuse_mlp_gelu
        self.decode_residual_memcfg = _width_sharded_mem_config(
            rows=TILE_SIZE, width=self.hidden_size, grid=_dram_shard_core_grid(self.hidden_size)
        )

        cache_root = str(tensor_cache_path) if tensor_cache_path is not None else None

        def norm(name):
            return OptimizedRMSNorm(
                mesh_device=mesh_device,
                hf_config=hf_config,
                state_dict=substate(layer_state, name) if layer_state else {},
                tensor_cache_path=f"{cache_root}/layer_{layer_idx}/{name}" if cache_root else None,
                decode_sharded=decode_norm_sharded,
            )

        self.input_layernorm = norm("input_layernorm")
        self.post_attention_layernorm = norm("post_attention_layernorm")
        self.pre_feedforward_layernorm = norm("pre_feedforward_layernorm")
        self.post_feedforward_layernorm = norm("post_feedforward_layernorm")

        if layer_state and "layer_scalar" in layer_state:
            self.layer_scalar = layer_state["layer_scalar"].item()
        else:
            self.layer_scalar = 1.0

        attention_prefill_weight_dtype = ttnn.bfloat16 if self.layer_type == "sliding_attention" else None
        attention_prefill_layout = "tp8_dram_interleaved" if attention_prefill_weight_dtype is not None else "tp8_dram_sharded"
        self.self_attn = MultichipGemma4Attention(
            mesh_device=mesh_device,
            config=self.attention_config,
            state_dict=substate(layer_state, "self_attn") if layer_state else {},
            mesh_config=mesh_config,
            ccl_manager=ccl_manager,
            layer_idx=layer_idx,
            tensor_cache_path=f"{cache_root}/layer_{layer_idx}/self_attn" if cache_root else None,
            weight_dtype=attention_dtype,
            qkv_weight_dtype=attention_qkv_dtype,
            o_weight_dtype=attention_o_dtype,
            prefill_weight_dtype=attention_prefill_weight_dtype,
            activation_dtype=dtype,
            bounded_sliding_kv_cache=bounded_sliding_kv_cache,
        )
        self.shared_mlp = MultichipSharedMLP(
            mesh_device=mesh_device,
            hf_config=hf_config,
            state_dict=substate(layer_state, "mlp") if layer_state else {},
            mesh_config=mesh_config,
            ccl_manager=ccl_manager,
            dtype=shared_mlp_dtype,
            down_dtype=self.shared_mlp_down_dtype,
            decode_dtype=self.shared_mlp_decode_dtype,
            decode_down_dtype=self.shared_mlp_decode_down_dtype,
            activation_dtype=dtype,
            tensor_cache_path=f"{cache_root}/layer_{layer_idx}/mlp" if cache_root else None,
            fuse_gelu_mul=fuse_mlp_gelu,
        )

        self.multichip_summary = {
            "single_chip_baseline": "OptimizedDecoder",
            "target_mesh": f"{TARGET_MESH_SHAPE[0]}x{TARGET_MESH_SHAPE[1]} T3K",
            "fabric": TARGET_FABRIC,
            "topology": "Ring",
            "tp": self.mesh_config.tp,
            "tp_axis": self.mesh_config.tp_axis,
            "activation_strategy": "replicated residual stream; local width-sharded L1 decode tensors per device",
            "attention_strategy": "WQKV column-parallel, local SDPA heads, WO row-parallel, ring all-reduce",
            "mlp_strategy": "gate/up column-parallel, GeGLU local, down row-parallel, ring all-reduce",
            "kv_cache_strategy": "paged per-device local KV heads; replicated page table and current positions",
            "moe": "not_applicable_dense_model",
            "hidden_size": self.hidden_size,
            "hidden_per_device_if_sharded": self.hidden_size // self.mesh_config.tp,
            "mlp_intermediate_per_device": self.shared_mlp.local_intermediate_size,
            "local_q_size": self.self_attn.local_q_size,
            "local_qkv_size": self.self_attn.local_qkv_size,
            "local_kv_heads": self.self_attn.local_kv_heads,
            "kv_replicated": self.self_attn.kv_replicated,
            "padding": "none required for TP8: hidden=3840, intermediate=15360, local attention widths are tile aligned",
            "attention_dtype": _dtype_name(attention_dtype),
            "attention_qkv_dtype": _dtype_name(attention_qkv_dtype),
            "attention_o_dtype": _dtype_name(attention_o_dtype),
            "attention_decode_qkv_compute": "HiFi2"
            if self.layer_type == "sliding_attention"
            else "HiFi3_fp32_accumulation",
            "attention_prefill_weight_dtype": _dtype_name(
                attention_prefill_weight_dtype if attention_prefill_weight_dtype is not None else attention_qkv_dtype
            ),
            "attention_prefill_weight_layout": attention_prefill_layout,
            "shared_mlp_gate_up_dtype": _dtype_name(shared_mlp_dtype),
            "shared_mlp_down_dtype": _dtype_name(self.shared_mlp_down_dtype),
            "shared_mlp_prefill_dtype": _dtype_name(ttnn.bfloat16),
            "shared_mlp_decode_gate_up_dtype": _dtype_name(self.shared_mlp_decode_dtype),
            "shared_mlp_decode_down_dtype": _dtype_name(self.shared_mlp_decode_down_dtype),
            "kv_cache_dtype": _dtype_name(kv_cache_dtype),
            "activation_dtype": _dtype_name(dtype),
            "decode_norm_sharded": bool(decode_norm_sharded),
        }
        self.optimization_summary = self.multichip_summary

    @classmethod
    def from_state_dict(
        cls,
        state_dict,
        *,
        hf_config,
        layer_idx: int,
        mesh_device,
        mesh_config: MeshConfig | None = None,
        ccl_manager=None,
        dtype=ttnn.bfloat16,
        attention_dtype=None,
        attention_qkv_dtype=None,
        attention_o_dtype=None,
        shared_mlp_dtype=ttnn.bfloat8_b,
        shared_mlp_down_dtype=None,
        shared_mlp_decode_dtype=None,
        shared_mlp_decode_down_dtype=None,
        kv_cache_dtype=ttnn.bfloat16,
        tensor_cache_path: str | Path | None = None,
        bounded_sliding_kv_cache: bool = False,
        fuse_mlp_gelu: bool = True,
        decode_norm_sharded: bool = True,
        **kwargs,
    ):
        if kwargs:
            raise TypeError(f"unsupported MultichipDecoder.from_state_dict kwargs: {sorted(kwargs)}")

        model_args = _require_target_config(hf_config, layer_idx)
        if mesh_config is None:
            mesh_shape = _mesh_shape_tuple(mesh_device)
            tp = mesh_shape[1] if mesh_shape != (1, 1) else 1
            mesh_config = MeshConfig(mesh_shape, decode=ModeConfig(tp=tp), prefill=ModeConfig(tp=tp))
        if mesh_config.tp == 1:
            return OptimizedDecoder.from_state_dict(
                state_dict,
                hf_config=hf_config,
                layer_idx=layer_idx,
                mesh_device=mesh_device,
                mesh_config=mesh_config,
                ccl_manager=ccl_manager,
                dtype=dtype,
                attention_dtype=attention_dtype,
                attention_qkv_dtype=attention_qkv_dtype,
                attention_o_dtype=attention_o_dtype,
                shared_mlp_dtype=shared_mlp_dtype,
                shared_mlp_down_dtype=shared_mlp_down_dtype,
                shared_mlp_decode_dtype=shared_mlp_decode_dtype,
                shared_mlp_decode_down_dtype=shared_mlp_decode_down_dtype,
                kv_cache_dtype=kv_cache_dtype,
                tensor_cache_path=tensor_cache_path,
                bounded_sliding_kv_cache=bounded_sliding_kv_cache,
                fuse_mlp_gelu=fuse_mlp_gelu,
                decode_norm_sharded=decode_norm_sharded,
            )
        if ccl_manager is None:
            ccl_manager = RingCCLManager(mesh_device=mesh_device)

        normalized_state = _normalize_layer_state_dict(state_dict, layer_idx)
        layer_state = {}
        for prefix in (f"model.layers.{layer_idx}", f"model.language_model.layers.{layer_idx}"):
            layer_state = substate(normalized_state, prefix)
            if layer_state:
                break

        return cls(
            mesh_device=mesh_device,
            hf_config=model_args,
            layer_idx=layer_idx,
            layer_state=layer_state,
            mesh_config=mesh_config,
            ccl_manager=ccl_manager,
            dtype=dtype,
            attention_dtype=attention_dtype,
            attention_qkv_dtype=attention_qkv_dtype,
            attention_o_dtype=attention_o_dtype,
            shared_mlp_dtype=shared_mlp_dtype,
            shared_mlp_down_dtype=shared_mlp_down_dtype,
            shared_mlp_decode_dtype=shared_mlp_decode_dtype,
            shared_mlp_decode_down_dtype=shared_mlp_decode_down_dtype,
            kv_cache_dtype=kv_cache_dtype,
            tensor_cache_path=tensor_cache_path,
            bounded_sliding_kv_cache=bounded_sliding_kv_cache,
            fuse_mlp_gelu=fuse_mlp_gelu,
            decode_norm_sharded=decode_norm_sharded,
        )

    def create_paged_kv_cache(
        self,
        *,
        block_size: int,
        max_num_blocks: int,
        cache_dtype=None,
        tensor_cache_path: str | Path | None = None,
    ):
        return init_kv_cache(
            mesh_device=self.mesh_device,
            config=self.attention_config,
            paged_attention_config=PagedAttentionConfig(block_size=block_size, max_num_blocks=max_num_blocks),
            cache_dtype=cache_dtype if cache_dtype is not None else self.kv_cache_dtype,
            tensor_cache_path=str(tensor_cache_path) if tensor_cache_path is not None else None,
        )

    def forward(self, *, mode: str, **kwargs):
        if mode == "prefill":
            return self.prefill_forward(**kwargs)
        if mode == "decode":
            return self.decode_forward(**kwargs)
        raise ValueError(f"mode must be 'prefill' or 'decode', got {mode!r}")

    def _feed_forward(self, hidden_states, *, is_decode: bool):
        residual = hidden_states
        normed = self.pre_feedforward_layernorm.forward(hidden_states, is_decode=is_decode)
        mlp_output = self.shared_mlp(normed, is_decode=is_decode)

        hidden_states = self.post_feedforward_layernorm.forward(mlp_output, is_decode=is_decode)
        mlp_output.deallocate(True)
        combined = ttnn.add(residual, hidden_states)
        residual.deallocate(True)
        hidden_states.deallocate(True)

        if self.layer_scalar != 1.0:
            combined = ttnn.mul(combined, self.layer_scalar)
        return combined

    def _attention_block(
        self,
        hidden_states,
        *,
        rope_mats,
        page_table,
        kv_cache,
        is_decode: bool,
        position_idx=None,
        token_index=None,
        position_idx_cache=None,
        decode_position=None,
    ):
        residual = hidden_states
        normed = self.input_layernorm.forward(hidden_states, is_decode=is_decode)
        attn_output = self.self_attn(
            normed,
            rope_mats=rope_mats,
            position_idx=position_idx,
            page_table=page_table,
            kv_cache=kv_cache,
            is_decode=is_decode,
            token_index=token_index,
            position_idx_cache=position_idx_cache,
            decode_position=decode_position,
        )

        attn_output = self.post_attention_layernorm.forward(attn_output, is_decode=is_decode)
        hidden_states = ttnn.add(residual, attn_output)
        attn_output.deallocate(True)
        return hidden_states

    def prefill_forward(
        self,
        hidden_states,
        *,
        rope_mats,
        page_table,
        kv_cache,
    ):
        self._last_prefill_seq_len = hidden_states.shape[-2]
        hidden_states = self._attention_block(
            hidden_states,
            rope_mats=rope_mats,
            page_table=page_table,
            kv_cache=kv_cache,
            is_decode=False,
        )
        return self._feed_forward(hidden_states, is_decode=False)

    def decode_forward(
        self,
        hidden_states,
        *,
        rope_mats,
        page_table,
        kv_cache,
        position_idx,
        token_index: int | None = None,
        position_idx_cache=None,
    ):
        cos_cache, _ = rope_mats
        if len(cos_cache.shape) != 2 and token_index is None:
            raise ValueError("token_index is required when decode rope_mats are 4D tables")

        decode_position = token_index if token_index is not None else getattr(self, "_last_prefill_seq_len", None)
        hidden_states = ttnn.to_memory_config(hidden_states, self.decode_residual_memcfg)
        hidden_states = self._attention_block(
            hidden_states,
            rope_mats=rope_mats,
            page_table=page_table,
            kv_cache=kv_cache,
            is_decode=True,
            position_idx=position_idx,
            token_index=token_index,
            position_idx_cache=position_idx_cache,
            decode_position=decode_position,
        )
        return self._feed_forward(hidden_states, is_decode=True)


__all__ = ["MultichipDecoder", "RingCCLManager", "SUPPORTED_HF_MODEL_ID", "TARGET_MESH_SHAPE"]
