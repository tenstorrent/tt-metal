# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Qwen-Omni specific TTNN modules."""

from __future__ import annotations

import os
from typing import Optional

import torch
from torch import nn

from ttnn.model_preprocessing import preprocess_linear_bias, preprocess_linear_weight

import ttnn
from models.tt_cnn.tt.builder import Conv2dConfiguration
from models.experimental.tt_symbiote.core.module import TTNNModule
from models.experimental.tt_symbiote.core.run_config import trace_enabled
from models.experimental.tt_symbiote.core.tensor import TorchTTNNTensor
from models.experimental.tt_symbiote.core.utils import torch_dtype_to_ttnn_dtype, tree_map
from models.experimental.tt_symbiote.models.qwen_omni.distributed_config import (
    qwen_omni_replicated_concat_dim0_tensor_config,
)
from models.experimental.tt_symbiote.modules.attention import (
    TTNNBailingMoEAttention,
    TTNNPagedAttentionKVCache,
)
from models.experimental.tt_symbiote.modules.conv import NHWCConvPytorch, TTNNConv2dNHWC
from models.experimental.tt_symbiote.modules.linear import TTNNLinear, TTNNLinearIColShardedWAllReduced
from models.experimental.tt_symbiote.modules.tensor import TTNNReshape


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


def _ttnn_linear_mesh_all_gather_input_if_needed(input_tensor, mesh_dev, in_features: int):
    """When activations are width-sharded (``last * num_devices == in_features``), gather before ``ttnn.linear``."""
    if mesh_dev is None or not hasattr(mesh_dev, "get_num_devices") or mesh_dev.get_num_devices() <= 1:
        return input_tensor
    last_dim = int(input_tensor.shape[-1])
    n = int(mesh_dev.get_num_devices())
    if last_dim != in_features and last_dim * n == in_features:
        return ttnn.all_gather(
            input_tensor,
            dim=-1,
            cluster_axis=1,
            num_links=1,
            topology=ttnn.Topology.Linear,
        )
    return input_tensor


@trace_enabled
class TTNNQwenOmniLinear(TTNNLinear):
    """Qwen3-Omni ``nn.Linear`` replacement: mesh width all-gather, TorchTTNNTensor upload, replicated readback."""

    def set_output_tensors_config_impl(self, output_tensors):
        """On mesh, default readback can concat the last dim (``out_features * N``); materialize logical width."""
        if self.device_state is None or self.device is None or self.device.get_num_devices() <= 1:
            return super().set_output_tensors_config_impl(output_tensors)

        from models.experimental.tt_symbiote.core.tensor import TorchTTNNTensor

        def _materialize_one_replica(e):
            if not isinstance(e, TorchTTNNTensor) or e.ttnn_tensor is None:
                return e
            t = e.ttnn_tensor
            n = int(t.shape[0])
            h = int(self.out_features)
            pt = ttnn.to_torch(t, mesh_composer=ttnn.ConcatMeshToTensor(self.device, dim=0))
            if pt.shape[0] > n:
                pt = pt[:n]
            if pt.shape[-1] > h:
                pt = pt[..., :h]
            e.elem = pt.contiguous()
            e.ttnn_tensor = None
            if getattr(e, "_distributed_tensor_config", None) is not None:
                e._distributed_tensor_config = None
            return e

        return tree_map(_materialize_one_replica, output_tensors)

    def forward(self, input_tensor: ttnn.Tensor) -> ttnn.Tensor:
        """Forward pass through linear layer."""
        from models.experimental.tt_symbiote.core.tensor import TorchTTNNTensor

        if isinstance(input_tensor, TorchTTNNTensor):
            if input_tensor.ttnn_tensor is not None:
                input_tensor = input_tensor.ttnn_tensor
            elif input_tensor.elem is not None:
                dev = self.device
                mesh_mapper = (
                    ttnn.ReplicateTensorToMesh(dev)
                    if dev is not None and hasattr(dev, "get_num_devices") and dev.get_num_devices() > 1
                    else None
                )
                input_tensor = ttnn.from_torch(
                    input_tensor.elem.contiguous().to(torch.bfloat16),
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    device=dev,
                    mesh_mapper=mesh_mapper,
                )
            else:
                raise TypeError("TTNNQwenOmniLinear.forward: TorchTTNNTensor has neither ttnn_tensor nor elem")

        if input_tensor.layout != ttnn.TILE_LAYOUT:
            input_tensor = ttnn.to_layout(input_tensor, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        input_tensor = _ttnn_linear_mesh_all_gather_input_if_needed(input_tensor, self.device, self.in_features)
        input_tensor_shape = list(input_tensor.shape)
        input_shape = list(input_tensor_shape)
        while len(input_shape) < 4:
            input_shape.insert(1, 1)  # Add batch dimensions if needed
        input_tensor = ttnn.reshape(input_tensor, input_shape)
        tt_output = ttnn.linear(input_tensor, self.tt_weight, bias=self.tt_bias, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        tt_output = ttnn.reshape(tt_output, input_tensor_shape[:-1] + [self.out_features])
        return tt_output


class TTNNQwen3OmniMoeAudioEncoderConvOutLinear(TTNNQwenOmniLinear):
    """``audio_tower.conv_out``: same mesh rules as :class:`TTNNQwenOmniLinear`; distinct type for upgrades / tests."""

    @classmethod
    def from_torch(cls, linear: nn.Linear):
        new_linear = cls(
            in_features=linear.in_features,
            out_features=linear.out_features,
        )
        new_linear._fallback_torch_layer = linear
        new_linear.weight = linear.weight
        new_linear.bias = linear.bias
        return new_linear


class TTNNQwenOmniIColShardedWAllReduced(TTNNLinearIColShardedWAllReduced):
    """Omni vision / talker MLP fc2: after RS+AG the activation is full width per device; bias must be replicated.

    Base :class:`~models.experimental.tt_symbiote.modules.linear.TTNNLinearIColShardedWAllReduced` shards bias
    like fc1; adding it to the all-reduced output caused 1152 vs 144 style broadcast errors on mesh.
    """

    def move_weights_to_device_impl(self):
        if isinstance(self.tt_weight_host, torch.Tensor):
            self.tt_weight_host = preprocess_linear_weight(
                self.tt_weight_host,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                weights_mesh_mapper=ttnn.shard_tensor_to_mesh_mapper(self.device, dim=self.weight_dim),
            )
        if isinstance(self.tt_bias_host, torch.Tensor):
            self.tt_bias_host = preprocess_linear_bias(
                self.tt_bias_host,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                weights_mesh_mapper=ttnn.ReplicateTensorToMesh(self.device),
            )
        self.tt_weight = ttnn.to_device(self.tt_weight_host, self.device)
        self.tt_bias = ttnn.to_device(self.tt_bias_host, self.device) if self.tt_bias_host is not None else None


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


class NHWCConvTransposePytorch(nn.Module):
    """A wrapper around nn.ConvTranspose2d to handle NHWC input/output."""

    def __init__(self, conv: nn.ConvTranspose2d) -> None:
        super().__init__()
        self.conv = conv

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 3, 1, 2)
        x = self.conv(x)
        x = x.permute(0, 2, 3, 1)
        return x


class NCDHWConv3dPytorch(nn.Module):
    """PyTorch fallback for :class:`TTNNConv3d` (NCDHW in/out)."""

    def __init__(self, conv: nn.Conv3d) -> None:
        super().__init__()
        self.conv = conv

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


def _conv1d_to_height1_conv2d(conv1d: nn.Conv1d) -> nn.Conv2d:
    """Map ``nn.Conv1d`` to ``nn.Conv2d`` with ``kernel_size=(1, k)`` for TTNN 2D conv."""

    def _one(x):
        if isinstance(x, int):
            return x
        return int(x[0])

    k = _one(conv1d.kernel_size)
    s = _one(conv1d.stride)
    p = _one(conv1d.padding)
    d = _one(conv1d.dilation)

    conv2d = nn.Conv2d(
        in_channels=conv1d.in_channels,
        out_channels=conv1d.out_channels,
        kernel_size=(1, k),
        stride=(1, s),
        padding=(0, p),
        dilation=(1, d),
        groups=conv1d.groups,
        bias=conv1d.bias is not None,
        device=conv1d.weight.device,
        dtype=conv1d.weight.dtype,
    )
    with torch.no_grad():
        conv2d.weight.copy_(conv1d.weight.unsqueeze(2))
        if conv1d.bias is not None:
            conv2d.bias.copy_(conv1d.bias)
    return conv2d


class TTNNConv1d(TTNNModule):
    """1D convolution via TTNN ``conv2d`` with height 1 (activations ``[B, C, T]`` → NHWC ``[B, 1, T, C]``).

    Pairs with :class:`TTNNQwen3OmniMoeCausalConvNet` in ``qwen_omni_activation`` for HF ``Qwen3OmniMoeCausalConvNet``.
    """

    def __init__(self):
        super().__init__()
        self.conv2d = None

    @classmethod
    def from_torch(cls, conv1d: nn.Conv1d, slice_config=None):
        new = cls()
        new._fallback_torch_layer = conv1d
        equiv = _conv1d_to_height1_conv2d(conv1d)
        # Qwen Omni NHWC permute + mesh width-shard gather live on TTNNQwenOmniConv2dNHWC (defined below).
        new.conv2d = TTNNQwenOmniConv2dNHWC.from_torch(equiv, slice_config=slice_config)
        return new

    def set_output_tensors_config_impl(self, output_tensors):
        cfg = _qwen_omni_conv2d_mesh_output_config(self.device)
        if cfg is None:
            return super().set_output_tensors_config_impl(output_tensors)

        def apply(e):
            if isinstance(e, TorchTTNNTensor):
                e.set_distributed_tensor_config(cfg)
            return e

        return tree_map(apply, output_tensors)

    def forward(self, input_tensor, reshape_output=True):
        """``input_tensor``: ``ttnn.Tensor`` or ``torch.Tensor`` ``[B, C, T]``. Returns ``[B, C_out, T_out]``.

        TTNN children of TTNN parents use ``_bypass_tensor_wrapping`` (see ``device_management``), so
        :class:`TTNNQwen3OmniMoeCausalConvNet` (``qwen_omni_activation``) passes host torch after ``F.pad``; convert here.
        """
        if isinstance(input_tensor, TorchTTNNTensor):
            input_tensor = input_tensor.ttnn_tensor if input_tensor.ttnn_tensor is not None else input_tensor.elem
        if isinstance(input_tensor, torch.Tensor):
            mesh_mapper = None
            dev = self.device
            if dev is not None and hasattr(dev, "get_num_devices") and dev.get_num_devices() > 1:
                mesh_mapper = ttnn.ReplicateTensorToMesh(dev)
            input_tensor = ttnn.from_torch(
                input_tensor,
                dtype=torch_dtype_to_ttnn_dtype(input_tensor.dtype),
                layout=ttnn.TILE_LAYOUT,
                device=dev,
                mesh_mapper=mesh_mapper,
            )
        x = ttnn.permute(input_tensor, (0, 2, 1))
        x = ttnn.unsqueeze(x, 1)
        out = self.conv2d(x, reshape_output=reshape_output)
        out = ttnn.squeeze(out, 1)
        return ttnn.permute(out, (0, 2, 1))


def _int_pair_2d(x) -> tuple:
    if isinstance(x, int):
        return (x, x)
    t = tuple(int(v) for v in x)
    if len(t) == 1:
        return (t[0], t[0])
    return (t[0], t[1])


def _transpose_conv_output_w(
    input_width: int,
    kernel_w: int,
    stride_w: int,
    pad_w: int,
    dilation_w: int,
    out_pad_w: int,
) -> int:
    """Output width for conv transpose (symmetric padding), matches PyTorch / TTNN sliding window."""
    return (input_width - 1) * stride_w - 2 * pad_w + dilation_w * (kernel_w - 1) + out_pad_w + 1


def _conv_transpose2d_dram_width_slice_config(w_out: int):
    """Explicit DRAM width slicing for large transpose convs.

    Without this, TTNN auto-slice can pick a marginal slice count (e.g. 12 for ~5.4k-wide
    output) that still exhausts L1 vs circular buffers on Wormhole (TT_THROW CB clash).
    """
    max_slices = 172
    if w_out <= 512:
        return None
    # Smaller slices than auto (~output/450) to keep per-slice L1 well under free DRAM/L1 budget.
    num_slices = max(48, min(max_slices, (w_out + 79) // 80))
    return ttnn.Op2DSliceConfig(num_slices=num_slices, slice_type=ttnn.Op2DDRAMSliceWidth)


def _conv_transpose1d_to_height1_conv_transpose2d(c1: nn.ConvTranspose1d) -> nn.ConvTranspose2d:
    """Map ``nn.ConvTranspose1d`` to ``nn.ConvTranspose2d`` with ``kernel_size=(1, k)`` for TTNN 2D transpose conv."""

    k = _int_pair_2d(c1.kernel_size)[0]
    s = _int_pair_2d(c1.stride)[0]
    p = _int_pair_2d(c1.padding)[0]
    op = _int_pair_2d(c1.output_padding)[0]
    d = _int_pair_2d(c1.dilation)[0]

    c2 = nn.ConvTranspose2d(
        in_channels=c1.in_channels,
        out_channels=c1.out_channels,
        kernel_size=(1, k),
        stride=(1, s),
        padding=(0, p),
        output_padding=(0, op),
        dilation=(1, d),
        groups=c1.groups,
        bias=c1.bias is not None,
        device=c1.weight.device,
        dtype=c1.weight.dtype,
    )
    with torch.no_grad():
        c2.weight.copy_(c1.weight.unsqueeze(2))
        if c1.bias is not None:
            c2.bias.copy_(c1.bias)
    return c2


@trace_enabled
class TTNNConvTranspose2dNHWC(TTNNModule):
    """TTNN ``conv_transpose2d`` on NHWC activations (``[B, H, W, C]``)."""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        output_padding,
        dilation,
        groups: int = 1,
        slice_config=None,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.dilation = dilation
        self.groups = groups
        self.slice_config = slice_config
        self.reshape = TTNNReshape()

    @classmethod
    def from_torch(cls, conv: nn.ConvTranspose2d, slice_config=None) -> "TTNNConvTranspose2dNHWC":
        new_conv = cls(
            in_channels=conv.in_channels,
            out_channels=conv.out_channels,
            kernel_size=_int_pair_2d(conv.kernel_size),
            stride=_int_pair_2d(conv.stride),
            padding=_int_pair_2d(conv.padding),
            output_padding=_int_pair_2d(conv.output_padding),
            dilation=_int_pair_2d(conv.dilation),
            groups=conv.groups,
            slice_config=slice_config,
        )
        new_conv._fallback_torch_layer = NHWCConvTransposePytorch(conv)
        return new_conv

    def preprocess_weights_impl(self):
        if self.torch_layer is None:
            self._fallback_torch_layer = NHWCConvTransposePytorch(
                nn.ConvTranspose2d(
                    in_channels=self.in_channels,
                    out_channels=self.out_channels,
                    kernel_size=self.kernel_size,
                    stride=self.stride,
                    padding=self.padding,
                    output_padding=self.output_padding,
                    dilation=self.dilation,
                    groups=self.groups,
                )
            )
        inner = self.torch_layer.conv
        self.tt_weight, self.tt_bias = Conv2dConfiguration.convert_torch_weight_and_bias_to_ttnn(
            inner.weight, inner.bias
        )
        super().preprocess_weights_impl()

    def deallocate_weights_impl(self):
        ttnn.deallocate(self.tt_weight)
        if self.tt_bias is not None:
            ttnn.deallocate(self.tt_bias)
        super().deallocate_weights_impl()

    def forward(self, input_tensor: ttnn.Tensor, reshape_output=True) -> ttnn.Tensor:
        batch_size, input_height, input_width, _ = input_tensor.shape
        batch_size, input_height, input_width = int(batch_size), int(input_height), int(input_width)

        conv_config = ttnn.Conv2dConfig(
            weights_dtype=ttnn.bfloat16,
            output_layout=ttnn.TILE_LAYOUT,
            deallocate_activation=True,
            config_tensors_in_dram=True,
        )
        if self.slice_config is not None:
            conv_config.shard_layout = self.slice_config

        dev = input_tensor.device()
        w_out_est = _transpose_conv_output_w(
            input_width,
            int(self.kernel_size[1]),
            int(self.stride[1]),
            int(self.padding[1]),
            int(self.dilation[1]),
            int(self.output_padding[1]),
        )
        dram_slice_config = _conv_transpose2d_dram_width_slice_config(w_out_est)
        # LoFi reduces circular-buffer pressure for very wide transpose paths (still DRAM-sliced).
        math_fidelity = ttnn.MathFidelity.LoFi if w_out_est > 4096 else ttnn.MathFidelity.HiFi4
        compute_config = ttnn.init_device_compute_kernel_config(
            dev.arch(),
            math_fidelity=math_fidelity,
            fp32_dest_acc_en=False,
            packer_l1_acc=False,
        )

        if input_tensor.memory_config().is_sharded():
            input_tensor = ttnn.sharded_to_interleaved(input_tensor)

        kwargs = dict(
            input_tensor=input_tensor,
            weight_tensor=self.tt_weight,
            device=dev,
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            batch_size=batch_size,
            input_height=input_height,
            input_width=input_width,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            output_padding=self.output_padding,
            dilation=self.dilation,
            groups=self.groups,
            bias_tensor=self.tt_bias,
            conv_config=conv_config,
            compute_config=compute_config,
            mirror_kernel=True,
            dtype=ttnn.bfloat16,
        )
        if dram_slice_config is not None:
            kwargs["dram_slice_config"] = dram_slice_config
        if reshape_output:
            out, (oh, ow) = ttnn.conv_transpose2d(**kwargs, return_output_dim=True)
            return self.reshape(out, [batch_size, oh, ow, -1])
        return ttnn.conv_transpose2d(**kwargs, return_output_dim=False)


class TTNNConvTranspose1d(TTNNModule):
    """1D transposed convolution via TTNN ``conv_transpose2d`` with height 1 (``[B,C,T]`` ↔ NHWC ``[B,1,T,C]``)."""

    def __init__(self):
        super().__init__()
        self.conv2d_t = None
        self.in_channels = None
        self.out_channels = None
        self.kernel_size = None
        self.stride = None
        self.padding = None
        self.output_padding = None
        self.dilation = None
        self.groups = None

    @classmethod
    def from_torch(cls, conv1d: nn.ConvTranspose1d, slice_config=None):
        new = cls()
        new._fallback_torch_layer = conv1d
        new.in_channels = conv1d.in_channels
        new.out_channels = conv1d.out_channels
        new.kernel_size = conv1d.kernel_size
        new.stride = conv1d.stride
        new.padding = conv1d.padding
        new.output_padding = conv1d.output_padding
        new.dilation = conv1d.dilation
        new.groups = conv1d.groups
        equiv = _conv_transpose1d_to_height1_conv_transpose2d(conv1d)
        new.conv2d_t = TTNNConvTranspose2dNHWC.from_torch(equiv, slice_config=slice_config)
        return new

    def set_output_tensors_config_impl(self, output_tensors):
        cfg = _qwen_omni_conv2d_mesh_output_config(self.device)
        if cfg is None:
            return super().set_output_tensors_config_impl(output_tensors)

        def apply(e):
            if isinstance(e, TorchTTNNTensor):
                e.set_distributed_tensor_config(cfg)
            return e

        return tree_map(apply, output_tensors)

    def forward(self, input_tensor, reshape_output=True):
        if isinstance(input_tensor, TorchTTNNTensor):
            input_tensor = input_tensor.ttnn_tensor if input_tensor.ttnn_tensor is not None else input_tensor.elem
        if isinstance(input_tensor, torch.Tensor):
            mesh_mapper = None
            dev = self.device
            if dev is not None and hasattr(dev, "get_num_devices") and dev.get_num_devices() > 1:
                mesh_mapper = ttnn.ReplicateTensorToMesh(dev)
            input_tensor = ttnn.from_torch(
                input_tensor,
                dtype=torch_dtype_to_ttnn_dtype(input_tensor.dtype),
                layout=ttnn.TILE_LAYOUT,
                device=dev,
                mesh_mapper=mesh_mapper,
            )
        x = ttnn.permute(input_tensor, (0, 2, 1))
        x = ttnn.unsqueeze(x, 1)
        out = self.conv2d_t(x, reshape_output=reshape_output)
        out = ttnn.squeeze(out, 1)
        return ttnn.permute(out, (0, 2, 1))


# TTNN ``experimental.conv3d`` layout (see ``tests/ttnn/unit_tests/operations/conv/test_conv3d.py``).
_CONV3D_CHANNEL_ALIGNMENT = 32


def _conv3d_chunk_n_from_env() -> int:
    """Chunk size along conv3d / patch-embed batch dim (env ``TT_SYMBIOTE_CONV3D_CHUNK_N``, default 1024)."""
    import os

    return max(1, int(os.environ.get("TT_SYMBIOTE_CONV3D_CHUNK_N", "1024")))


def _conv3d_mesh_max_n_per_chunk(dev) -> int:
    """Upper bound on leading batch N for one conv3d compute slice on a mesh (patch embed).

    Without chunking, ``from_torch`` + ``experimental.conv3d`` + output readback can request 100+ GiB DRAM
    for large ``N``.  Set ``TT_SYMBIOTE_CONV3D_CHUNK_N`` to tune (default 1024).
    """
    if dev is None or not hasattr(dev, "get_num_devices") or dev.get_num_devices() <= 1:
        return 0
    return _conv3d_chunk_n_from_env()


def _int_triplet_3d(x) -> tuple:
    if isinstance(x, int):
        return (x, x, x)
    t = tuple(int(v) for v in x)
    if len(t) == 1:
        return (t[0], t[0], t[0])
    if len(t) == 3:
        return (t[0], t[1], t[2])
    raise ValueError(f"Expected int or length-1/3 tuple for 3D conv hyperparameter, got {x!r}")


def _conv3d_out_dim(size_in: int, pad: int, stride: int, kernel: int, dilation: int = 1) -> int:
    eff_k = dilation * (kernel - 1) + 1
    return (size_in + 2 * pad - eff_k) // stride + 1


def _conv3d_padding_triple_for_ttnn(padding) -> tuple:
    """Map ``nn.Conv3d.padding`` to a ``(pd, ph, pw)`` triple for TTNN (symmetric triple only)."""
    if isinstance(padding, int):
        return (padding, padding, padding)
    p = tuple(int(x) for x in padding)
    if len(p) == 3:
        return (p[0], p[1], p[2])
    if len(p) == 6:
        raise ValueError("TTNNConv3d does not support 6-tuple asymmetric padding; use PyTorch fallback.")
    raise ValueError(f"Unsupported Conv3d padding: {padding!r}")


def _conv3d_torch_weights_to_ttnn_layout(
    conv: nn.Conv3d,
    *,
    alignment: int = _CONV3D_CHANNEL_ALIGNMENT,
    c_in_block: int = 0,
) -> tuple:
    """Layout ``nn.Conv3d`` weights/bias like ``prepare_weights`` in ``test_conv3d`` (host torch)."""
    w = conv.weight.data  # [out, in, kD, kH, kW]
    c_in = int(conv.in_channels)
    out_ch = int(conv.out_channels)
    w = w.permute(2, 3, 4, 1, 0)  # kD, kH, kW, C, out
    align_pad = (alignment - c_in % alignment) % alignment
    if align_pad:
        w = torch.nn.functional.pad(w, (0, 0, 0, align_pad))
    kD, kH, kW, c_pad, _ = w.shape
    c_in_block_eff = c_pad if c_in_block == 0 else int(c_in_block)
    num_c_in_blocks = c_pad // c_in_block_eff
    assert num_c_in_blocks * c_in_block_eff == c_pad
    w = w.reshape(kD, kH, kW, num_c_in_blocks, c_in_block_eff, out_ch)
    w = w.permute(3, 0, 1, 2, 4, 5)
    w = w.reshape(-1, out_ch).contiguous()
    b = conv.bias.data.reshape(1, -1).contiguous() if conv.bias is not None else None
    return w, b


def _conv3d_torch_input_to_ndhwc_row_major(
    x: torch.Tensor, *, alignment: int = _CONV3D_CHANNEL_ALIGNMENT
) -> torch.Tensor:
    """NCDHW → NDHWC and pad channel to ``alignment`` (TTNN conv3d input)."""
    if x.dim() != 5:
        raise RuntimeError(f"TTNNConv3d expects 5D NCDHW input, got shape {tuple(x.shape)}")
    c_in = int(x.shape[1])
    tt_input = x.permute(0, 2, 3, 4, 1).contiguous()
    align_pad = (alignment - c_in % alignment) % alignment
    if align_pad:
        tt_input = torch.nn.functional.pad(tt_input, (0, align_pad))
    return tt_input


def _ttnn_conv3d_output_to_torch_ncdhw(
    tt_out: ttnn.Tensor,
    *,
    n: int,
    d_out: int,
    h_out: int,
    w_out: int,
    out_channels: int,
    mesh_device,
) -> torch.Tensor:
    """Match ``reshape_output`` in ``test_conv3d``; host ``[N,C,D,H,W]``."""
    if mesh_device is None or mesh_device.get_num_devices() <= 1:
        t = ttnn.to_torch(tt_out)
    else:
        shards = ttnn.get_device_tensors(tt_out)
        if shards:
            t = ttnn.to_torch(shards[0])
        else:
            composer = ttnn.ConcatMeshToTensor(mesh_device, dim=0)
            t = ttnn.to_torch(tt_out, mesh_composer=composer)
            lead = int(tt_out.shape[0])
            if t.dim() >= 1 and int(t.shape[0]) > lead:
                t = t[:lead].contiguous()
    if t.dtype != torch.bfloat16:
        t = t.to(torch.bfloat16)
    t = t.reshape(n, d_out, h_out, w_out, out_channels)
    return t.permute(0, 4, 1, 2, 3).contiguous()


def _ttnn_tensor_to_torch_batch_leading(tt: ttnn.Tensor, mesh_device) -> torch.Tensor:
    """Device/mesh tensor → host torch; for multi-device concat on dim 0, slice to logical batch."""
    n0 = int(tt.shape[0])
    if mesh_device is None or not hasattr(mesh_device, "get_num_devices") or mesh_device.get_num_devices() <= 1:
        return ttnn.to_torch(tt)
    shards = ttnn.get_device_tensors(tt)
    if shards:
        return ttnn.to_torch(shards[0])
    composer = ttnn.ConcatMeshToTensor(mesh_device, dim=0)
    t = ttnn.to_torch(tt, mesh_composer=composer)
    if t.dim() >= 1 and int(t.shape[0]) > n0:
        t = t[:n0].contiguous()
    return t


@trace_enabled
class TTNNConv3d(TTNNModule):
    """3D convolution via ``ttnn.experimental.conv3d`` (NDHWC activations, Qwen3-Omni vision patch embed).

    Matches the weight/input layout used in ``tests/ttnn/unit_tests/operations/conv/test_conv3d.py``.
    For large patch kernels (e.g. ``kernel_size == stride`` with spatial product ≥ 196) and
    ``out_channels % 32 == 0``, uses Qwen-VL-style L1 blocking (``C_in_block=16``, ``C_out_block=32``).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple,
        stride: tuple,
        padding: tuple,
        dilation: tuple,
        groups: int,
        padding_mode: str,
        conv3d_blocking=None,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.padding_mode = padding_mode
        self.conv3d_blocking = conv3d_blocking
        self.tt_weight = None
        self.tt_bias = None

    @property
    def weight(self):
        """HF ``Qwen3OmniMoeVisionPatchEmbed`` reads ``proj.weight.dtype``; mirror ``nn.Conv3d``."""
        if self.torch_layer is None:
            raise AttributeError("weight")
        return self.torch_layer.conv.weight

    @property
    def bias(self):
        if self.torch_layer is None:
            raise AttributeError("bias")
        return self.torch_layer.conv.bias

    @classmethod
    def from_torch(cls, conv: nn.Conv3d, conv3d_blocking=None) -> "TTNNConv3d":
        ks = _int_triplet_3d(conv.kernel_size)
        st = _int_triplet_3d(conv.stride)
        try:
            pad_triple = _conv3d_padding_triple_for_ttnn(conv.padding)
        except ValueError:
            pad_triple = (0, 0, 0)
        blocking = conv3d_blocking
        if blocking is None:
            if ks == st and ks[0] * ks[1] * ks[2] >= 196 and conv.out_channels % 32 == 0 and conv.groups == 1:
                blocking = (16, 32, 1, 1, 1)
        new_mod = cls(
            in_channels=conv.in_channels,
            out_channels=conv.out_channels,
            kernel_size=ks,
            stride=st,
            padding=pad_triple,
            dilation=_int_triplet_3d(conv.dilation),
            groups=conv.groups,
            padding_mode=conv.padding_mode,
            conv3d_blocking=blocking,
        )
        new_mod._fallback_torch_layer = NCDHWConv3dPytorch(conv)
        return new_mod

    def preprocess_weights_impl(self):
        if self.torch_layer is None:
            self._fallback_torch_layer = NCDHWConv3dPytorch(
                nn.Conv3d(
                    self.in_channels,
                    self.out_channels,
                    self.kernel_size,
                    stride=self.stride,
                    padding=self.padding,
                    dilation=self.dilation,
                    groups=self.groups,
                    bias=True,
                )
            )
        conv = self.torch_layer.conv
        c_in_block = 0
        if self.conv3d_blocking is not None:
            c_in_block = int(self.conv3d_blocking[0])
        w_torch, b_torch = _conv3d_torch_weights_to_ttnn_layout(conv, c_in_block=c_in_block)
        self.tt_weight = ttnn.from_torch(w_torch, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, pad_value=0)
        if b_torch is not None:
            self.tt_bias = ttnn.from_torch(b_torch, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, pad_value=0)
        else:
            self.tt_bias = None
        super().preprocess_weights_impl()

    def move_weights_to_device_impl(self):
        if self.tt_weight is not None:
            self.tt_weight = ttnn.to_device(self.tt_weight, self.device)
        if self.tt_bias is not None:
            self.tt_bias = ttnn.to_device(self.tt_bias, self.device)
        super().move_weights_to_device_impl()

    def deallocate_weights_impl(self):
        if self.tt_weight is not None:
            ttnn.deallocate(self.tt_weight)
        if self.tt_bias is not None:
            ttnn.deallocate(self.tt_bias)
        super().deallocate_weights_impl()

    def set_output_tensors_config_impl(self, output_tensors):
        cfg = _qwen_omni_conv2d_mesh_output_config(self.device)
        if cfg is None:
            return super().set_output_tensors_config_impl(output_tensors)

        def apply(e):
            if isinstance(e, TorchTTNNTensor):
                e.set_distributed_tensor_config(cfg)
            return e

        return tree_map(apply, output_tensors)

    def __call__(self, *args, **kwds):
        """Same as ``NormalRun.module_run`` but skip ``to_ttnn_wrap`` on inputs.

        Default mesh ``to_ttnn`` for 5D NCDHW activations can size buffers incorrectly; we keep
        ``TorchTTNNTensor.elem`` as torch until :meth:`forward` uploads NDHWC via ``from_torch``.
        """
        bypass = getattr(self, "_bypass_tensor_wrapping", False)
        if bypass:
            from models.experimental.tt_symbiote.core.run_config import NormalRun

            return NormalRun.module_run(self, *args, **kwds)

        import time

        from tracy import signpost

        from models.experimental.tt_symbiote.core.run_config import (
            DispatchManager,
            NormalRun,
            compose_transforms,
            post_process_ttnn_module_output,
            set_device_wrap,
            wrap_to_torch_ttnn_tensor,
        )

        print(f"{self.__class__.__name__}: {self.module_name} on device {self.device}")
        assert self.device is not None, "Device must be set for TTNN module execution."
        begin_full = time.time()
        transform = compose_transforms(wrap_to_torch_ttnn_tensor, set_device_wrap(self.device))
        func_args = tree_map(transform, args)
        other_kwargs = {k: v for k, v in kwds.items() if "past_key_value" not in k}
        func_kwargs = tree_map(transform, other_kwargs)
        func_kwargs.update({k: v for k, v in kwds.items() if "past_key_value" in k})
        begin = time.time()
        self.preprocess_weights()
        end = time.time()
        DispatchManager.set_current_module_name(self.module_name)
        DispatchManager.record_timing(
            "TTNN", self.module_name, self.__class__.__name__ + "_preprocess_weights", {}, end - begin
        )
        begin = time.time()
        self.move_weights_to_device()
        end = time.time()
        DispatchManager.record_timing(
            "TTNN", self.module_name, self.__class__.__name__ + "_move_weights_to_device", {}, end - begin
        )
        if NormalRun.signpost_mode is not None:
            signpost(f"{self.module_name}", f"{self.__class__.__name__}")
        begin = time.time()
        result = post_process_ttnn_module_output(self, self.forward(*func_args, **func_kwargs))
        end = time.time()
        DispatchManager.record_timing("TTNN", self.module_name, self.__class__.__name__ + "_forward", {}, end - begin)
        DispatchManager.set_current_module_name(None)
        end_full = time.time()
        DispatchManager.record_timing(
            "TorchModules", self.module_name, self.__class__.__name__, {}, end_full - begin_full
        )
        return result

    def _experimental_conv3d_ncdhw_torch(self, x: torch.Tensor) -> torch.Tensor:
        """Run ``ttnn.experimental.conv3d`` for one CPU ``[N,C,D,H,W]`` batch; return host ``[N,C_out,D_out,H_out,W_out]``."""
        n, _, d_in, h_in, w_in = x.shape
        pd, ph, pw = self.padding
        sd, sh, sw = self.stride
        dd, dh, dw = self.dilation
        kd, kh, kw = self.kernel_size
        d_out = _conv3d_out_dim(d_in, pd, sd, kd, dd)
        h_out = _conv3d_out_dim(h_in, ph, sh, kh, dh)
        w_out = _conv3d_out_dim(w_in, pw, sw, kw, dw)

        dev = self.device
        mesh_mapper = None
        if dev is not None and hasattr(dev, "get_num_devices") and dev.get_num_devices() > 1:
            mesh_mapper = ttnn.ReplicateTensorToMesh(dev)
        ndhwc = _conv3d_torch_input_to_ndhwc_row_major(x)
        tt_in = ttnn.from_torch(
            ndhwc,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=dev,
            mesh_mapper=mesh_mapper,
        )
        compute_kernel_config = ttnn.init_device_compute_kernel_config(
            dev.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )
        grid = dev.compute_with_storage_grid_size()
        grid_xy = (int(grid.x), int(grid.y))

        kwargs = dict(
            input_tensor=tt_in,
            weight_tensor=self.tt_weight,
            bias_tensor=self.tt_bias,
            dtype=ttnn.bfloat16,
            output_channels=self.out_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            padding_mode=self.padding_mode,
            compute_kernel_config=compute_kernel_config,
        )
        if self.conv3d_blocking is not None:
            cin_b, cout_b, tb, hb, wb = self.conv3d_blocking
            kwargs["config"] = ttnn.Conv3dConfig(
                weights_dtype=ttnn.bfloat16,
                output_layout=ttnn.ROW_MAJOR_LAYOUT,
                T_out_block=int(tb),
                W_out_block=int(wb),
                H_out_block=int(hb),
                C_out_block=int(cout_b),
                C_in_block=int(cin_b),
                dilation=self.dilation,
                compute_with_storage_grid_size=grid_xy,
            )
        tt_out = ttnn.experimental.conv3d(**kwargs)
        out_torch = _ttnn_conv3d_output_to_torch_ncdhw(
            tt_out,
            n=n,
            d_out=d_out,
            h_out=h_out,
            w_out=w_out,
            out_channels=self.out_channels,
            mesh_device=dev,
        )
        ttnn.deallocate(tt_in)
        ttnn.deallocate(tt_out)
        return out_torch

    def forward(self, x):
        if isinstance(x, TorchTTNNTensor):
            x = x.ttnn_tensor if x.ttnn_tensor is not None else x.elem
        dev = self.device
        conv_pad = self.torch_layer.conv.padding
        if isinstance(conv_pad, tuple) and len(conv_pad) == 6:
            if isinstance(x, ttnn.Tensor):
                x = _ttnn_tensor_to_torch_batch_leading(x, dev)
            if not isinstance(x, torch.Tensor):
                raise TypeError(f"TTNNConv3d expected torch or ttnn tensor, got {type(x)}")
            return self.torch_layer.conv(x)

        if self.groups != 1 or self.padding_mode != "zeros":
            if isinstance(x, ttnn.Tensor):
                x = _ttnn_tensor_to_torch_batch_leading(x, dev)
            if not isinstance(x, torch.Tensor):
                raise TypeError(f"TTNNConv3d expected torch or ttnn tensor, got {type(x)}")
            x = x.to(dtype=self.torch_layer.conv.weight.dtype)
            return self.torch_layer.conv(x)

        if isinstance(x, ttnn.Tensor):
            x = _ttnn_tensor_to_torch_batch_leading(x, dev)
            x = x.to(dtype=torch.bfloat16).contiguous()

        if not isinstance(x, torch.Tensor):
            raise TypeError(f"TTNNConv3d expected torch or ttnn tensor, got {type(x)}")
        x = x.to(dtype=torch.bfloat16).contiguous()
        n = int(x.shape[0])
        max_chunk = _conv3d_mesh_max_n_per_chunk(dev)
        if max_chunk > 0 and n > max_chunk:
            parts = []
            for s in range(0, n, max_chunk):
                e = min(s + max_chunk, n)
                parts.append(self._experimental_conv3d_ncdhw_torch(x[s:e].contiguous()))
            out_torch = torch.cat(parts, dim=0)
        else:
            out_torch = self._experimental_conv3d_ncdhw_torch(x)

        self.deallocate_weights()

        c_out = int(out_torch.shape[1])
        out_2d = out_torch.reshape(-1, c_out).to(dtype=torch.bfloat16).contiguous()

        mesh_mapper = None
        if dev is not None and hasattr(dev, "get_num_devices") and dev.get_num_devices() > 1:
            mesh_mapper = ttnn.ReplicateTensorToMesh(dev)
        return ttnn.from_torch(
            out_2d,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=dev,
            mesh_mapper=mesh_mapper,
        )


def _qwen_omni_conv2d_mesh_output_config(mesh_device):
    """Replicated conv readback: Replicate + ConcatMesh(dim=0) slice (avoid ConcatMesh2d inflation on audio). TTNNQwenOmniConv2dNHWC AGs width when sharded."""
    return qwen_omni_replicated_concat_dim0_tensor_config(mesh_device)


def _pair_int2_conv(x) -> tuple:
    if isinstance(x, int):
        return (x, x)
    t = tuple(int(v) for v in x)
    if len(t) == 1:
        return (t[0], t[0])
    return (t[0], t[1])


def _expected_nhwc_conv2d_out_width(w_in: int, conv_mod: TTNNConv2dNHWC) -> int:
    _, kw = _pair_int2_conv(conv_mod.kernel_size)
    _, sw = _pair_int2_conv(conv_mod.stride)
    _, pw = _pair_int2_conv(conv_mod.padding)
    _, dw = _pair_int2_conv(conv_mod.dilation)
    return (w_in + 2 * pw - dw * (kw - 1) - 1) // sw + 1


def _nhwc_spatial_w_per_device_shard(out: ttnn.Tensor, mesh_num_devices: int) -> int | None:
    """If ``out`` is one width shard per mesh device, return local W; else None.

    ``out.shape[2]`` can still reflect the **logical** full width while each device's tensor
    only holds ``W_full / nd`` (e.g. 48 vs 384). Using shard geometry avoids skipping
    ``all_gather`` and fixes residual adds that saw ``T=48`` vs ``T=384``.
    """
    if mesh_num_devices <= 1:
        return None
    shards = ttnn.get_device_tensors(out)
    if len(shards) != mesh_num_devices:
        return None
    return int(shards[0].shape[2])


def _maybe_all_gather_nhwc_width_across_mesh(
    conv_mod: TTNNConv2dNHWC, out: ttnn.Tensor, input_nhwc: ttnn.Tensor
) -> ttnn.Tensor:
    """Stitch per-device width shards (``W_local * num_devices == W_full``) before NCHW permute."""
    dev = conv_mod.device
    if dev is None or not hasattr(dev, "get_num_devices") or dev.get_num_devices() <= 1:
        return out
    nd = int(dev.get_num_devices())
    w_in = int(input_nhwc.shape[2])
    w_exp = _expected_nhwc_conv2d_out_width(w_in, conv_mod)
    w_meta = int(out.shape[2])
    w_shard = _nhwc_spatial_w_per_device_shard(out, nd)
    # Prefer per-device width when we see a full mesh of spatial shards (avoids false w_meta == w_exp).
    w_cur = w_shard if w_shard is not None else w_meta
    if w_cur == w_exp:
        return out
    if w_cur * nd == w_exp:
        ds = getattr(conv_mod, "device_state", None)
        if ds is not None and getattr(ds, "ccl_manager", None) is not None:
            out = ttnn.experimental.all_gather_async(
                out,
                dim=2,
                multi_device_global_semaphore=ds.ccl_manager.get_and_cycle_ag_semaphore_handles(1),
                barrier_semaphore=ds.ccl_manager.get_and_cycle_barrier_semaphore_handle(1),
                num_links=1,
                topology=ttnn.Topology.Linear,
            )
        else:
            out = ttnn.all_gather(out, dim=2, num_links=1, topology=ttnn.Topology.Linear)
        ttnn.synchronize_device(dev)
    return out


@trace_enabled
class TTNNQwenOmniConv2dNHWC(TTNNConv2dNHWC):
    """Qwen3-Omni Conv2d: permute audio NCHW [B,C,F,T] vs vision NHWC for TTNN conv2d; restore layout for HF. from_torch always returns this subclass."""

    def _logical_in_channels(self) -> int:
        tl = self.torch_layer
        if tl is None:
            return int(self.in_channels)
        inner = getattr(tl, "conv", None)
        if inner is not None:
            return int(inner.in_channels)
        return int(self.in_channels)

    @classmethod
    def from_torch(cls, conv: nn.Conv2d, slice_config=None) -> "TTNNQwenOmniConv2dNHWC":
        # Mirror :class:`TTNNConv2dNHWCInputMultipleOf16` but always construct ``cls`` so
        # :meth:`forward` below is used (``InputMultipleOf16.from_torch`` returns ``TTNNConv2dNHWC``).
        if conv.in_channels > 16 or conv.in_channels % 16 == 0:
            new_mod = cls(
                in_channels=conv.in_channels,
                out_channels=conv.out_channels,
                kernel_size=conv.kernel_size,
                stride=conv.stride,
                padding=conv.padding,
                dilation=conv.dilation,
                groups=conv.groups,
                slice_config=slice_config,
            )
            new_mod._fallback_torch_layer = NHWCConvPytorch(conv)
            return new_mod
        new_mod = cls(
            in_channels=16,
            out_channels=conv.out_channels,
            kernel_size=conv.kernel_size,
            stride=conv.stride,
            padding=conv.padding,
            dilation=conv.dilation,
            groups=conv.groups,
            slice_config=slice_config,
        )
        conv.weight = nn.Parameter(
            torch.nn.functional.pad(conv.weight, (0, 0, 0, 0, 0, (16 - conv.in_channels % 16) % 16))
        )
        new_mod._fallback_torch_layer = NHWCConvPytorch(conv)
        return new_mod

    def set_output_tensors_config_impl(self, output_tensors):
        cfg = _qwen_omni_conv2d_mesh_output_config(self.device)
        if cfg is None:
            return super().set_output_tensors_config_impl(output_tensors)

        def apply(e):
            if isinstance(e, TorchTTNNTensor):
                e.set_distributed_tensor_config(cfg)
            return e

        return tree_map(apply, output_tensors)

    def forward(self, input_tensor: ttnn.Tensor, reshape_output=True) -> ttnn.Tensor:
        logical_in = self._logical_in_channels()
        shape = input_tensor.shape
        nchw_in = False
        if len(shape) != 4:
            raise RuntimeError(f"TTNNQwenOmniConv2dNHWC: expected 4D input, got shape={list(shape)}")
        if int(shape[1]) == logical_in and int(shape[-1]) != logical_in:
            nchw_in = True
            input_tensor = ttnn.permute(input_tensor, (0, 2, 3, 1))
        elif int(shape[-1]) != logical_in and int(shape[1]) != logical_in:
            raise RuntimeError(
                f"TTNNQwenOmniConv2dNHWC: cannot match in_channels={logical_in} to layout " f"shape={list(shape)}"
            )

        if int(self.in_channels) > logical_in and int(input_tensor.shape[-1]) == logical_in:
            pad_c = int(self.in_channels) - logical_in
            layout_in = input_tensor.layout
            if layout_in == ttnn.TILE_LAYOUT:
                input_tensor = ttnn.to_layout(input_tensor, ttnn.ROW_MAJOR_LAYOUT)
            input_tensor = ttnn.pad(
                input_tensor,
                ((0, 0), (0, 0), (0, 0), (0, pad_c)),
                value=0.0,
            )
            if layout_in == ttnn.TILE_LAYOUT:
                input_tensor = ttnn.to_layout(input_tensor, ttnn.TILE_LAYOUT)

        out = TTNNConv2dNHWC.forward(self, input_tensor, reshape_output=reshape_output)
        out = _maybe_all_gather_nhwc_width_across_mesh(self, out, input_tensor)
        if nchw_in:
            out = ttnn.permute(out, (0, 3, 1, 2))
        return out


__all__ = [
    "TTNNBailingMoEAttentionHostCachePositionRoPE",
    "TTNNConv1d",
    "TTNNConv3d",
    "TTNNConvTranspose1d",
    "TTNNConvTranspose2dNHWC",
    "TTNNQwen3OmniMoeAudioEncoderConvOutLinear",
    "TTNNQwenOmniConv2dNHWC",
    "TTNNQwenOmniIColShardedWAllReduced",
    "TTNNQwenOmniLinear",
    "TTNNSnakeBeta",
]
