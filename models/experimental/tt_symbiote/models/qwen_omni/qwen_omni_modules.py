# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Qwen-Omni specific TTNN modules."""

import os
from typing import Optional

import torch
import ttnn
from models.experimental.tt_symbiote.core.module import TTNNModule
from models.experimental.tt_symbiote.core.tensor import TorchTTNNTensor
from models.experimental.tt_symbiote.core.utils import torch_dtype_to_ttnn_dtype, tree_map
from models.experimental.tt_symbiote.modules.attention import (
    TTNNBailingMoEAttention,
    TTNNPagedAttentionKVCache,
)


def _mesh_host_stitch_device_shards(tt_tensor: ttnn.Tensor, mesh_device) -> torch.Tensor | None:
    """Concat host tensors when each mesh device holds a shard on one logical dimension."""
    if mesh_device is None or not hasattr(mesh_device, "get_num_devices"):
        return None
    nd = int(mesh_device.get_num_devices())
    if nd <= 1:
        return None
    shards = ttnn.get_device_tensors(tt_tensor)
    if len(shards) != nd:
        return None
    local = tuple(int(x) for x in shards[0].shape)
    for t in (tt_tensor.shape, getattr(tt_tensor, "padded_shape", None)):
        if t is None:
            continue
        logical = tuple(int(x) for x in t)
        if len(logical) != len(local):
            continue
        for d in range(len(logical)):
            if local[d] != logical[d] and local[d] * nd == logical[d]:
                parts = [ttnn.to_torch(s).contiguous() for s in shards]
                return torch.cat(parts, dim=d)
    return None


def _ttnn_mesh_to_torch_one_replica(tt_tensor: ttnn.Tensor, mesh_device) -> torch.Tensor:
    """Host tensor matching one logical replica (avoid ambiguous mesh compose behavior on replicated tensors)."""
    if mesh_device is None or mesh_device.get_num_devices() <= 1:
        return ttnn.to_torch(tt_tensor).contiguous()
    stitched = _mesh_host_stitch_device_shards(tt_tensor, mesh_device)
    if stitched is not None:
        return stitched.contiguous()
    shards = ttnn.get_device_tensors(tt_tensor)
    if shards:
        return ttnn.to_torch(shards[0]).contiguous()
    composer = ttnn.ConcatMeshToTensor(mesh_device, dim=0)
    result = ttnn.to_torch(tt_tensor, mesh_composer=composer)
    lead = int(tt_tensor.shape[0])
    if result.dim() >= 1 and int(result.shape[0]) > lead:
        result = result[:lead].contiguous()
    return result.contiguous()


def _upload_bct_replicated(x_t: torch.Tensor, mesh_device):
    """Upload host tensor with ReplicateTensorToMesh on multi-device meshes."""
    mesh_mapper = None
    if mesh_device is not None and hasattr(mesh_device, "get_num_devices") and mesh_device.get_num_devices() > 1:
        mesh_mapper = ttnn.ReplicateTensorToMesh(mesh_device)
    return ttnn.from_torch(
        x_t.contiguous(),
        dtype=torch_dtype_to_ttnn_dtype(x_t.dtype),
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        mesh_mapper=mesh_mapper,
    )


class TTNNSnakeBeta(TTNNModule):
    """TTNN SnakeBeta (HF ``SnakeBeta`` in Qwen3-Omni code2wav decoder): x + 1/b * sin^2(x*a)."""

    def __init__(self, in_features: int):
        super().__init__()
        self.in_features = in_features
        self.no_div_by_zero = 0.000000001
        self.alpha = None
        self.beta = None

    @classmethod
    def from_torch(cls, torch_layer, *args, **kwargs):
        in_features = int(getattr(torch_layer, "in_features", torch_layer.alpha.shape[0]))
        new_layer = cls(in_features)
        new_layer._fallback_torch_layer = torch_layer
        return new_layer

    def move_weights_to_device_impl(self):
        super().move_weights_to_device_impl()
        tl = self.torch_layer
        if tl is None:
            return
        w_alpha = tl.alpha.detach().float().contiguous()
        w_beta = tl.beta.detach().float().contiguous()
        mesh_mapper = None
        if self.device is not None and hasattr(self.device, "get_num_devices") and self.device.get_num_devices() > 1:
            mesh_mapper = ttnn.ReplicateTensorToMesh(self.device)
        self.alpha = ttnn.from_torch(
            w_alpha,
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
            mesh_mapper=mesh_mapper,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        self.beta = ttnn.from_torch(
            w_beta,
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
            mesh_mapper=mesh_mapper,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def set_output_tensors_config_impl(self, output_tensors):
        if self.device_state is None or self.device is None or self.device.get_num_devices() <= 1:
            return super().set_output_tensors_config_impl(output_tensors)

        from models.experimental.tt_symbiote.core.tensor import TorchTTNNTensor

        def _materialize_one_replica(t):
            if not isinstance(t, TorchTTNNTensor) or t.ttnn_tensor is None:
                return t
            tt = t.ttnn_tensor
            pt = _ttnn_mesh_to_torch_one_replica(tt, self.device)
            t.elem = pt.contiguous()
            t.ttnn_tensor = None
            if getattr(t, "_distributed_tensor_config", None) is not None:
                t._distributed_tensor_config = None
            return t

        return tree_map(_materialize_one_replica, output_tensors)

    @staticmethod
    def _snake_beta_chunk_t() -> int:
        raw = os.environ.get("TT_SYMBIOTE_SNAKEBETA_CHUNK_T", "4096")
        try:
            v = int(raw)
        except ValueError:
            v = 4096
        return max(512, v)

    def _forward_fp32_core(
        self,
        input_fp32: ttnn.Tensor,
        alpha_exp: ttnn.Tensor,
        reciprocal_beta: ttnn.Tensor,
    ) -> ttnn.Tensor:
        x_times_alpha = ttnn.multiply(input_fp32, alpha_exp)
        sin_result = ttnn.sin(x_times_alpha)
        sin_squared = ttnn.pow(sin_result, 2.0)
        scaled_sin = ttnn.multiply(reciprocal_beta, sin_squared)
        result = ttnn.add(input_fp32, scaled_sin, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        for t in (x_times_alpha, sin_result, sin_squared, scaled_sin):
            try:
                ttnn.deallocate(t)
            except Exception:
                pass
        return result

    def forward(self, input_tensor: ttnn.Tensor) -> ttnn.Tensor:
        if input_tensor.layout != ttnn.TILE_LAYOUT:
            input_tensor = ttnn.to_layout(input_tensor, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        out_dtype = input_tensor.dtype
        shape = tuple(int(s) for s in input_tensor.shape)
        if len(shape) != 3:
            raise ValueError(f"TTNNSnakeBeta expects rank-3 [B,C,T], got shape {shape}")
        b, c, t_len = shape

        chunk_t = self._snake_beta_chunk_t()
        alpha_expanded = ttnn.unsqueeze(self.alpha, 0)
        alpha_expanded = ttnn.unsqueeze(alpha_expanded, -1)
        beta_expanded = ttnn.unsqueeze(self.beta, 0)
        beta_expanded = ttnn.unsqueeze(beta_expanded, -1)
        alpha_exp = ttnn.exp(alpha_expanded)
        try:
            ttnn.deallocate(alpha_expanded)
        except Exception:
            pass
        beta_exp = ttnn.exp(beta_expanded)
        beta_plus_eps = ttnn.add(beta_exp, self.no_div_by_zero)
        reciprocal_beta = ttnn.reciprocal(beta_plus_eps)
        try:
            ttnn.deallocate(beta_expanded)
            ttnn.deallocate(beta_exp)
            ttnn.deallocate(beta_plus_eps)
        except Exception:
            pass

        if t_len <= chunk_t:
            if out_dtype != ttnn.float32:
                input_fp32 = ttnn.typecast(input_tensor, ttnn.float32)
            else:
                input_fp32 = input_tensor
            result = self._forward_fp32_core(input_fp32, alpha_exp, reciprocal_beta)
            if out_dtype != ttnn.float32:
                if input_fp32 is not input_tensor:
                    try:
                        ttnn.deallocate(input_fp32)
                    except Exception:
                        pass
                result = ttnn.typecast(result, out_dtype)
            try:
                ttnn.deallocate(alpha_exp)
                ttnn.deallocate(reciprocal_beta)
            except Exception:
                pass
            return result

        out_chunks = []
        for t0 in range(0, t_len, chunk_t):
            t1 = min(t0 + chunk_t, t_len)
            sl = ttnn.slice(input_tensor, (0, 0, t0), (b, c, t1))
            if out_dtype != ttnn.float32:
                sl_fp32 = ttnn.typecast(sl, ttnn.float32)
            else:
                sl_fp32 = sl
            res_fp32 = self._forward_fp32_core(sl_fp32, alpha_exp, reciprocal_beta)
            if out_dtype != ttnn.float32:
                out_chunks.append(ttnn.typecast(res_fp32, out_dtype))
            else:
                out_chunks.append(res_fp32)

        torch_parts = []
        mesh_dev = self.device
        for ch in out_chunks:
            torch_parts.append(_ttnn_mesh_to_torch_one_replica(ch, mesh_dev))
            try:
                ttnn.deallocate(ch)
            except Exception:
                pass
        merged_torch = torch.cat(torch_parts, dim=2)
        try:
            ttnn.deallocate(alpha_exp)
            ttnn.deallocate(reciprocal_beta)
        except Exception:
            pass
        return _upload_bct_replicated(merged_torch, mesh_dev)


class TTNNBailingMoEAttentionHostCachePositionRoPE(TTNNBailingMoEAttention):
    """BailingMoE decode variant — RoPE cos/sin from host ``cache_position_tensor``.

    Base :class:`~models.experimental.tt_symbiote.modules.attention.TTNNBailingMoEAttention`
    uses :meth:`~models.experimental.tt_symbiote.modules.attention.TTNNBailingMoEAttention._get_cur_pos_device_tensor`
    so decode is trace-friendly. This subclass keeps the older path where
    :meth:`BailingRotarySetup.get_cos_sin_for_decode` is driven by CPU indices
    while paged KV still receives a device ``cur_pos_tt`` built via
    :func:`ttnn.from_torch`.
    """

    def _forward_decode_paged(
        self,
        hidden_states: ttnn.Tensor,
        position_embeddings: tuple,
        attention_mask: Optional[ttnn.Tensor],
        past_key_values: "TTNNPagedAttentionKVCache",
        cache_position: Optional[torch.LongTensor],
    ) -> tuple:
        batch_size, seq_length = hidden_states.shape[0], hidden_states.shape[1]

        if hidden_states.layout != ttnn.TILE_LAYOUT:
            hidden_states = ttnn.to_layout(hidden_states, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        qkv_states = self.qkv_proj(hidden_states)
        q_size = self.num_heads * self.head_dim
        kv_size = self.num_kv_heads * self.head_dim
        query_states = ttnn.slice(qkv_states, [0, 0, 0], [batch_size, 1, q_size])
        key_states = ttnn.slice(qkv_states, [0, 0, q_size], [batch_size, 1, q_size + kv_size])
        value_states = ttnn.slice(qkv_states, [0, 0, q_size + kv_size], [batch_size, 1, q_size + 2 * kv_size])
        ttnn.deallocate(qkv_states)

        query_states = ttnn.reshape(query_states, (1, batch_size, self.num_heads, self.head_dim))
        key_states = ttnn.reshape(key_states, (1, batch_size, self.num_kv_heads, self.head_dim))
        value_states = ttnn.reshape(value_states, (1, batch_size, self.num_kv_heads, self.head_dim))

        if query_states.dtype != ttnn.bfloat16:
            query_states = ttnn.typecast(query_states, ttnn.bfloat16)
        if key_states.dtype != ttnn.bfloat16:
            key_states = ttnn.typecast(key_states, ttnn.bfloat16)
        if value_states.dtype != ttnn.bfloat16:
            value_states = ttnn.typecast(value_states, ttnn.bfloat16)

        query_states = ttnn.to_memory_config(query_states, ttnn.L1_MEMORY_CONFIG)
        key_states = ttnn.to_memory_config(key_states, ttnn.L1_MEMORY_CONFIG)

        query_states, key_states = self._apply_qk_norm(query_states, key_states)

        layer_idx = self._fallback_torch_layer.layer_idx

        if cache_position is None:
            cur_pos = past_key_values.get_seq_length(layer_idx)
            cache_position_tensor = torch.tensor([cur_pos], dtype=torch.int32)
        else:
            cp = cache_position
            if isinstance(cp, TorchTTNNTensor):
                cp = cp.to_torch
            if isinstance(cp, ttnn.Tensor):
                mesh_composer = None
                if hasattr(cp, "device") and cp.device() is not None and cp.device().get_num_devices() > 1:
                    mesh_composer = ttnn.ConcatMeshToTensor(cp.device(), dim=0)
                cp = ttnn.to_torch(cp, mesh_composer=mesh_composer)
            cache_position_tensor = cp.flatten()[:batch_size].to(torch.int32)

        mesh_mapper = ttnn.ReplicateTensorToMesh(self.device)
        cur_pos_tt = ttnn.from_torch(
            cache_position_tensor,
            device=self.device,
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=mesh_mapper,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        if isinstance(query_states, ttnn.Tensor) and query_states.dtype != ttnn.bfloat16:
            query_states = ttnn.typecast(query_states, ttnn.bfloat16)
        if isinstance(key_states, ttnn.Tensor) and key_states.dtype != ttnn.bfloat16:
            key_states = ttnn.typecast(key_states, ttnn.bfloat16)

        cos_ttnn, sin_ttnn = self._rotary_setup.get_cos_sin_for_decode(cache_position_tensor)

        batch_grid = ttnn.num_cores_to_corerangeset(batch_size, self.device.compute_with_storage_grid_size(), True)

        rope_shard_mem = ttnn.create_sharded_memory_config(
            shape=(ttnn.TILE_SIZE, self.head_dim),
            core_grid=batch_grid,
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
        cos_ttnn = ttnn.to_memory_config(cos_ttnn, rope_shard_mem)
        sin_ttnn = ttnn.to_memory_config(sin_ttnn, rope_shard_mem)

        q_shard_mem = ttnn.create_sharded_memory_config(
            shape=(ttnn.TILE_SIZE, self.head_dim),
            core_grid=batch_grid,
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
        query_states = ttnn.to_memory_config(query_states, q_shard_mem)

        k_shard_mem = ttnn.create_sharded_memory_config(
            shape=(ttnn.TILE_SIZE, key_states.shape[-1]),
            core_grid=batch_grid,
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
        key_states = ttnn.to_memory_config(key_states, k_shard_mem)

        trans_mat = self._rotary_setup.get_trans_mat_decode_sharded(batch_size)

        query_states = ttnn.experimental.rotary_embedding_llama(
            query_states, cos_ttnn, sin_ttnn, trans_mat, is_decode_mode=True
        )
        key_states = ttnn.experimental.rotary_embedding_llama(
            key_states, cos_ttnn, sin_ttnn, trans_mat, is_decode_mode=True
        )

        num_cores = batch_size
        compute_grid = self.device.compute_with_storage_grid_size()
        shard_grid = ttnn.num_cores_to_corerangeset(num_cores, compute_grid, True)
        kv_vol = key_states.volume() // key_states.padded_shape[-1] // num_cores
        kv_shard = ttnn.ShardSpec(
            shard_grid,
            [kv_vol, key_states.padded_shape[-1]],
            ttnn.ShardOrientation.ROW_MAJOR,
        )
        kv_mem = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            kv_shard,
        )
        key_states = ttnn.to_memory_config(key_states, kv_mem)
        value_states = ttnn.to_memory_config(value_states, kv_mem)

        past_key_values.paged_update_on_device(
            key_states,
            value_states,
            layer_idx=layer_idx,
            current_pos=cur_pos_tt,
        )
        ttnn.deallocate(key_states)
        ttnn.deallocate(value_states)

        attn_output = past_key_values.paged_sdpa_decode(
            query_states,
            layer_idx,
            current_pos=cur_pos_tt,
            scale=self.scaling,
            program_config=self.sdpa.decode_program_config,
            compute_kernel_config=self.sdpa.compute_kernel_config,
        )

        sdpa_output_memcfg = ttnn.create_sharded_memory_config(
            shape=(32, self.head_dim),
            core_grid=ttnn.CoreGrid(y=1, x=batch_size),
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
        attn_output = ttnn.to_memory_config(attn_output, sdpa_output_memcfg)

        attn_output = ttnn.experimental.nlp_concat_heads_decode(
            attn_output,
            num_heads=self.num_heads,
        )
        attn_output = self.dense(attn_output)

        if batch_size < 32:
            attn_output = ttnn.slice(attn_output, [0, 0, 0, 0], [1, 1, batch_size, attn_output.shape[-1]])

        attn_output = ttnn.reshape(attn_output, (batch_size, seq_length, -1))

        return attn_output, None, past_key_values
