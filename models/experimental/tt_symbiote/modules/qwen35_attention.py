# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Qwen3.5-27B Full Attention implementation for TTNN.

Implements Grouped Query Attention (GQA) with:
- 32 attention heads, 4 KV heads (8:1 ratio)
- head_dim = 256
- Partial RoPE (rotary_dim=64, head_dim=256, factor=0.25)
- Q gating: q_proj outputs 2x dimension, split into Q and gate, sigmoid gate on output
- Q/K normalization: Qwen3_5RMSNorm (1+weight adjustment) on head_dim

Uses paged attention via TTNNPagedAttentionKVCache / TTNNQwenPagedAttentionKVCache
for on-device KV cache. No CPU round-trips in the forward path.
"""

from typing import Optional

import torch

import ttnn
from models.experimental.tt_symbiote.core.module import TTNNModule
from models.experimental.tt_symbiote.core.tensor import TorchTTNNTensor
from models.experimental.tt_symbiote.modules.linear import TTNNLinear, TTNNLinearIReplicatedWColSharded
from models.experimental.tt_symbiote.modules.rope import TTNNRotaryPositionEmbedding
from models.experimental.tt_symbiote.modules.attention import (
    TTNNSDPAAttention,
    TTNNPagedAttentionKVCache,
)


class TTNNQwen35FullAttention(TTNNModule):
    """TTNN-accelerated Full Attention for Qwen3.5-27B.

    Implements Grouped Query Attention (GQA) with:
    - 32 attention heads
    - 4 KV heads (8:1 ratio, each KV head serves 8 Q heads)
    - head_dim = 256
    - RoPE position embeddings (partial rotary: 64 of 256 dims)
    - Q gating: q_proj outputs 2x dimension, split into Q and gate
    - Q/K normalization: RMSNorm on head_dim with (1+weight) adjustment

    Uses paged attention for on-device KV cache:
    - Prefill: paged_fill_on_device + standard SDPA
    - Decode: paged_update_on_device + paged_sdpa_decode
    """

    def __init__(self):
        super().__init__()
        self.num_attention_heads = None
        self.num_key_value_heads = None
        self.num_key_value_groups = None
        self.head_dim = None
        self.hidden_size = None
        self.scaling = None
        self.is_causal = True
        self.layer_idx = None

        self.q_proj = None
        self.k_proj = None
        self.v_proj = None
        self.o_proj = None
        self.rope = None
        self.sdpa = None
        self.core_grid = None

        # Q gating support
        self.has_q_gate = False

        # Q/K normalization (RMSNorm on head_dim)
        self.q_norm_weight = None
        self.k_norm_weight = None
        self.tt_q_norm_weight = None
        self.tt_k_norm_weight = None
        self.rms_norm_eps = 1e-6

    @classmethod
    def from_torch(cls, torch_attn, distributed=True):
        """Create TTNNQwen35FullAttention from PyTorch Qwen3_5Attention.

        Args:
            torch_attn: PyTorch Qwen3_5Attention layer.
            distributed: Use col-sharded weights for multi-device (default True for T3K).

        Returns:
            TTNNQwen35FullAttention instance.
        """
        new_attn = cls()
        new_attn._fallback_torch_layer = torch_attn

        # Extract configuration from torch layer
        config = torch_attn.config
        new_attn.num_attention_heads = config.num_attention_heads  # 32
        new_attn.num_key_value_heads = config.num_key_value_heads  # 4
        new_attn.num_key_value_groups = new_attn.num_attention_heads // new_attn.num_key_value_heads  # 8
        new_attn.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)  # 256
        new_attn.hidden_size = config.hidden_size
        new_attn.scaling = new_attn.head_dim**-0.5
        new_attn.layer_idx = torch_attn.layer_idx
        new_attn.distributed = distributed

        # Check for Q gating: q_proj outputs 2x dimension for Q + gate
        q_proj_out_features = torch_attn.q_proj.out_features
        expected_q_dim = new_attn.num_attention_heads * new_attn.head_dim
        new_attn.has_q_gate = q_proj_out_features == expected_q_dim * 2

        # Extract Q/K normalization weights
        # Qwen3_5RMSNorm: output = rms_norm(x.float()) * (1.0 + weight.float())
        if hasattr(torch_attn, "q_norm") and torch_attn.q_norm is not None:
            new_attn.q_norm_weight = torch_attn.q_norm.weight.detach().clone()
            new_attn.rms_norm_eps = getattr(torch_attn.q_norm, "eps", 1e-6)
        if hasattr(torch_attn, "k_norm") and torch_attn.k_norm is not None:
            new_attn.k_norm_weight = torch_attn.k_norm.weight.detach().clone()

        # Choose linear class: col-sharded for distributed, replicated for single device
        LinearCls = TTNNLinearIReplicatedWColSharded if distributed else TTNNLinear

        # Create TTNN linear projections
        new_attn.q_proj = LinearCls.from_torch(torch_attn.q_proj)
        new_attn.k_proj = LinearCls.from_torch(torch_attn.k_proj)
        new_attn.v_proj = LinearCls.from_torch(torch_attn.v_proj)
        new_attn.o_proj = LinearCls.from_torch(torch_attn.o_proj)

        # RoPE for position embeddings (partial rotary: 64 of 256 dims)
        new_attn.rope = TTNNRotaryPositionEmbedding()

        # SDPA for attention computation
        new_attn.sdpa = TTNNSDPAAttention()

        # Default core grid
        new_attn.core_grid = ttnn.CoreGrid(y=8, x=8)

        return new_attn

    def preprocess_weights_impl(self):
        """Preprocess Q/K normalization weights for TTNN."""
        super().preprocess_weights_impl()

        # Prepare Q/K norm weights with (1.0 + weight) adjustment
        if self.q_norm_weight is not None:
            q_norm_adjusted = (1.0 + self.q_norm_weight.float()).to(self.q_norm_weight.dtype)
            self.tt_q_norm_weight_host = ttnn.from_torch(
                q_norm_adjusted.unsqueeze(0),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
            )
        if self.k_norm_weight is not None:
            k_norm_adjusted = (1.0 + self.k_norm_weight.float()).to(self.k_norm_weight.dtype)
            self.tt_k_norm_weight_host = ttnn.from_torch(
                k_norm_adjusted.unsqueeze(0),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
            )

    def move_weights_to_device_impl(self):
        """Move norm weights to device and configure SDPA."""
        super().move_weights_to_device_impl()

        mesh_mapper = ttnn.ReplicateTensorToMesh(self.device) if self.device.get_num_devices() > 1 else None

        # Query grid from device
        grid = self.device.compute_with_storage_grid_size()
        self.core_grid = ttnn.CoreGrid(y=grid.y, x=grid.x)

        # Move Q/K norm weights to device
        if hasattr(self, "tt_q_norm_weight_host") and self.tt_q_norm_weight_host is not None:
            q_norm_torch = ttnn.to_torch(self.tt_q_norm_weight_host)
            self.tt_q_norm_weight = ttnn.from_torch(
                q_norm_torch,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=self.device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=mesh_mapper,
            )
        if hasattr(self, "tt_k_norm_weight_host") and self.tt_k_norm_weight_host is not None:
            k_norm_torch = ttnn.to_torch(self.tt_k_norm_weight_host)
            self.tt_k_norm_weight = ttnn.from_torch(
                k_norm_torch,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=self.device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=mesh_mapper,
            )

        # Configure SDPA for head_dim=256
        if self.sdpa.program_config is None:
            self.sdpa.program_config = ttnn.SDPAProgramConfig(
                compute_with_storage_grid_size=(self.core_grid.x, self.core_grid.y),
                q_chunk_size=128,
                k_chunk_size=128,
                exp_approx_mode=False,
            )
            self.sdpa.compute_kernel_config = ttnn.init_device_compute_kernel_config(
                self.device.arch(),
                math_fidelity=ttnn.MathFidelity.HiFi4,
                math_approx_mode=False,
                fp32_dest_acc_en=False,
                packer_l1_acc=False,
            )

    @property
    def _is_distributed(self):
        """Check if running in distributed mode with CCL manager."""
        return (
            getattr(self, "distributed", True)
            and self.device_state is not None
            and hasattr(self.device_state, "ccl_manager")
            and self.device_state.ccl_manager is not None
        )

    def _maybe_all_gather(self, tensor):
        """All-gather tensor across mesh devices if in distributed mode."""
        if not self._is_distributed:
            return tensor
        gathered = ttnn.all_gather(
            tensor,
            dim=-1,
            num_links=1,
            cluster_axis=1,
            topology=ttnn.Topology.Linear,
        )
        ttnn.synchronize_device(self.device)
        return gathered

    def _repeat_kv(self, hidden_states: ttnn.Tensor, n_rep: int) -> ttnn.Tensor:
        """Repeat KV heads to match Q heads for GQA.

        For Qwen3.5-27B: 4 KV heads -> 32 Q heads, so n_rep=8.
        """
        if n_rep == 1:
            return hidden_states
        return ttnn.repeat_interleave(hidden_states, n_rep, dim=1)

    def _project_qkv(self, hidden_states, batch_size, seq_length, position_embeddings):
        """Project hidden states to Q, K, V, apply Q/K norm and RoPE.

        Returns:
            Tuple of (query_states, key_states, value_states, gate, cos, sin).
            gate is None if has_q_gate is False.
        """
        if hidden_states.layout != ttnn.TILE_LAYOUT:
            hidden_states = ttnn.to_layout(hidden_states, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        # Project to Q, K, V
        q_proj_output = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # All-gather projections for distributed mode (col-sharded -> replicated)
        q_proj_output = self._maybe_all_gather(q_proj_output)
        key_states = self._maybe_all_gather(key_states)
        value_states = self._maybe_all_gather(value_states)

        # Handle Q gating: split q_proj output into query_states and gate
        gate = None
        if self.has_q_gate:
            q_proj_output = ttnn.reshape(
                q_proj_output, (batch_size, seq_length, self.num_attention_heads, self.head_dim * 2)
            )
            query_states = q_proj_output[:, :, :, : self.head_dim]
            gate = q_proj_output[:, :, :, self.head_dim :]
            gate = ttnn.reshape(gate, (batch_size, seq_length, self.num_attention_heads * self.head_dim))
        else:
            query_states = ttnn.reshape(
                q_proj_output, (batch_size, seq_length, self.num_attention_heads, self.head_dim)
            )

        # Apply Q/K normalization (RMSNorm on head_dim)
        if self.tt_q_norm_weight is not None:
            orig_q_shape = query_states.shape
            query_states = ttnn.reshape(
                query_states, (batch_size * seq_length * self.num_attention_heads, self.head_dim)
            )
            query_states = ttnn.rms_norm(query_states, weight=self.tt_q_norm_weight, epsilon=self.rms_norm_eps)
            query_states = ttnn.reshape(query_states, orig_q_shape)

        key_states = ttnn.reshape(key_states, (batch_size, seq_length, self.num_key_value_heads, self.head_dim))

        if self.tt_k_norm_weight is not None:
            orig_k_shape = key_states.shape
            key_states = ttnn.reshape(key_states, (batch_size * seq_length * self.num_key_value_heads, self.head_dim))
            key_states = ttnn.rms_norm(key_states, weight=self.tt_k_norm_weight, epsilon=self.rms_norm_eps)
            key_states = ttnn.reshape(key_states, orig_k_shape)

        # Permute to [batch, num_heads, seq_len, head_dim]
        query_states = ttnn.permute(query_states, (0, 2, 1, 3))
        key_states = ttnn.permute(key_states, (0, 2, 1, 3))

        value_states = ttnn.reshape(value_states, (batch_size, seq_length, self.num_key_value_heads, self.head_dim))
        value_states = ttnn.permute(value_states, (0, 2, 1, 3))

        # Apply RoPE (partial rotary: 64 of 256 dims)
        cos, sin = position_embeddings
        if len(cos.shape) == 3:
            cos = ttnn.unsqueeze(cos, 1)
        if len(sin.shape) == 3:
            sin = ttnn.unsqueeze(sin, 1)

        query_states, key_states = self.rope(query_states, key_states, cos, sin)

        # Expand KV to match Q heads (GQA)
        key_states = self._repeat_kv(key_states, self.num_key_value_groups)
        value_states = self._repeat_kv(value_states, self.num_key_value_groups)

        return query_states, key_states, value_states, gate, cos, sin

    def forward(
        self,
        hidden_states: ttnn.Tensor,
        position_embeddings: tuple,
        attention_mask: Optional[ttnn.Tensor] = None,
        past_key_values=None,
        cache_position=None,
        **kwargs,
    ) -> tuple:
        """Forward pass through Qwen3.5-27B full attention.

        Uses paged attention for on-device KV cache management.

        Args:
            hidden_states: Input tensor [batch, seq_len, hidden_size].
            position_embeddings: Tuple of (cos, sin) for RoPE.
            attention_mask: Optional attention mask.
            past_key_values: TTNNPagedAttentionKVCache or TTNNQwenPagedAttentionKVCache.
            cache_position: Cache position tensor for decode.

        Returns:
            Tuple of (attention_output, None).
        """
        seq_length = hidden_states.shape[1]

        use_paged = isinstance(past_key_values, TTNNPagedAttentionKVCache)

        if use_paged and seq_length == 1:
            return self._forward_decode_paged(
                hidden_states,
                position_embeddings,
                attention_mask,
                past_key_values,
                cache_position,
            )

        return self._forward_prefill_paged(
            hidden_states,
            position_embeddings,
            attention_mask,
            past_key_values,
        )

    def _forward_prefill_paged(
        self,
        hidden_states,
        position_embeddings,
        attention_mask,
        past_key_values,
    ):
        """Prefill path: project QKV, SDPA, fill paged KV cache.

        For prefill, we compute attention on the full Q/K/V sequence using SDPA,
        then fill the paged KV cache for subsequent decode steps.
        """
        batch_size, seq_length = hidden_states.shape[0], hidden_states.shape[1]

        query_states, key_states, value_states, gate, cos, sin = self._project_qkv(
            hidden_states, batch_size, seq_length, position_embeddings
        )

        # Fill paged KV cache if available
        if past_key_values is not None and isinstance(past_key_values, TTNNPagedAttentionKVCache):
            # Get unexpanded KV for cache storage
            kv_key = key_states[:, :: self.num_key_value_groups, :, :]
            kv_value = value_states[:, :: self.num_key_value_groups, :, :]

            past_key_values.paged_fill_on_device(
                kv_key,
                kv_value,
                layer_idx=self.layer_idx,
                batch_idx=0,
            )

        # Compute attention via SDPA on full Q/K/V
        attn_output = self.sdpa(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0,
            scaling=self.scaling,
            is_causal=self.is_causal,
            transpose_output=True,
        )

        # Reshape: [batch, seq_len, num_heads, head_dim] -> [batch, seq_len, hidden]
        attn_shape = list(attn_output.shape)
        attn_output = ttnn.reshape(
            attn_output, (attn_shape[0], attn_shape[1], self.num_attention_heads * self.head_dim)
        )

        # Apply Q gating
        if gate is not None:
            gate_sigmoid = ttnn.sigmoid(gate)
            attn_output = ttnn.mul(attn_output, gate_sigmoid)
            ttnn.deallocate(gate_sigmoid)

        # Output projection (col-sharded: each device has hidden_size/num_devices)
        # Do NOT all-gather here — decoder layer handles col-sharded residual add
        attn_output = self.o_proj(attn_output)

        return attn_output, None

    def _forward_decode_paged(
        self,
        hidden_states,
        position_embeddings,
        attention_mask,
        past_key_values,
        cache_position,
    ):
        """Decode path using paged attention with on-device KV cache.

        TTNN paged kernels require tensors in [1, batch, heads, head_dim]
        layout (S B H D) whereas _project_qkv returns the standard
        [batch, heads, seq, head_dim] (B H S D). This method handles
        the permute, L1 sharding required by paged_update_cache, and
        the GQA-aware SDPA decode call.
        """
        batch_size, seq_length = hidden_states.shape[0], hidden_states.shape[1]

        query_states, key_states, value_states, gate, cos, sin = self._project_qkv(
            hidden_states, batch_size, seq_length, position_embeddings
        )

        layer_idx = self.layer_idx

        # Get unexpanded KV for cache update
        kv_key = key_states[:, :: self.num_key_value_groups, :, :]
        kv_value = value_states[:, :: self.num_key_value_groups, :, :]

        # Resolve cache position to a 1-D int32 device tensor [batch].
        # Trace-compatible: if cache_position is already on-device, use directly
        # (no host round-trip). TracedRun pre-allocates device buffers for kwargs.
        cur_pos_tt = self._get_cur_pos_device_tensor(cache_position, past_key_values, layer_idx, batch_size)

        # Permute B H S D -> S B H D (the layout paged kernels expect)
        query_states_paged = ttnn.permute(query_states, (2, 0, 1, 3))
        kv_key = ttnn.permute(kv_key, (2, 0, 1, 3))
        kv_value = ttnn.permute(kv_value, (2, 0, 1, 3))

        # Multi-device: convert all-gathered topology -> replicated
        if self.device.get_num_devices() > 1:
            query_states_paged = self._to_replicated(query_states_paged)
            kv_key = self._to_replicated(kv_key)
            kv_value = self._to_replicated(kv_value)

        # HEIGHT_SHARDED L1 for paged_update_cache
        tile_size = 32
        shard_h = ((self.num_key_value_heads + tile_size - 1) // tile_size) * tile_size

        core_grid = ttnn.CoreGrid(y=1, x=batch_size)
        shard_cfg = ttnn.create_sharded_memory_config(
            shape=(shard_h, self.head_dim),
            core_grid=core_grid,
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
        )
        kv_key = ttnn.to_memory_config(kv_key, shard_cfg)
        kv_value = ttnn.to_memory_config(kv_value, shard_cfg)

        # Update the on-device paged KV cache
        past_key_values.paged_update_on_device(
            kv_key,
            kv_value,
            layer_idx=layer_idx,
            current_pos=cur_pos_tt,
        )
        ttnn.deallocate(kv_key)
        ttnn.deallocate(kv_value)

        # Paged SDPA decode
        attn_output = past_key_values.paged_sdpa_decode(
            query_states_paged,
            layer_idx,
            current_pos=cur_pos_tt,
            scale=self.scaling,
            program_config=self.sdpa.program_config,
            compute_kernel_config=self.sdpa.compute_kernel_config,
        )
        # attn_output: [1, B, H, head_dim]

        # Convert back to [B, S, H*D] for the output projection
        attn_output = ttnn.permute(attn_output, (1, 0, 2, 3))  # [B, 1, H, head_dim]
        attn_shape = list(attn_output.shape)
        attn_output = ttnn.reshape(
            attn_output, (attn_shape[0], attn_shape[1], self.num_attention_heads * self.head_dim)
        )

        # Apply Q gating
        if gate is not None:
            gate_sigmoid = ttnn.sigmoid(gate)
            attn_output = ttnn.mul(attn_output, gate_sigmoid)
            ttnn.deallocate(gate_sigmoid)

        # Output projection (col-sharded: each device has hidden_size/num_devices)
        # Do NOT all-gather here — decoder layer handles col-sharded residual add
        attn_output = self.o_proj(attn_output)

        return attn_output, None

    def _to_replicated(self, tensor):
        """Convert all-gathered tensor topology to replicated for paged ops."""
        if hasattr(tensor, "device") and tensor.device() is not None and tensor.device().get_num_devices() > 1:
            return ttnn.to_memory_config(tensor, ttnn.DRAM_MEMORY_CONFIG)
        return tensor

    def _get_cur_pos_device_tensor(self, cache_position, past_key_values, layer_idx, batch_size):
        """Get cur_pos as a device tensor, trace-compatible.

        During trace capture, no new device buffers can be allocated.
        If cache_position is already on-device (pre-allocated by TracedRun),
        use it directly. Otherwise, fall back to the host path.
        """
        cp = cache_position
        if isinstance(cp, TorchTTNNTensor) and cp.ttnn_tensor is not None:
            cp = cp.ttnn_tensor

        # Trace-safe: cache_position is already a device tensor
        if isinstance(cp, ttnn.Tensor) and hasattr(cp, "storage_type") and cp.storage_type() != ttnn.StorageType.HOST:
            if len(cp.shape) > 1:
                total_elems = 1
                for d in cp.shape:
                    total_elems *= d
                cp = ttnn.reshape(cp, (total_elems,))
            if cp.shape[0] > batch_size:
                cp = ttnn.slice(cp, [0], [batch_size])
            if cp.dtype != ttnn.int32:
                orig_size = cp.shape[0]
                if orig_size % 32 != 0:
                    pad_amount = 32 - (orig_size % 32)
                    if not hasattr(self, "_pos_typecast_pad_buf") or self._pos_typecast_pad_buf is None:
                        pad_torch = torch.zeros(pad_amount, dtype=torch.int64)
                        mesh_mapper = ttnn.ReplicateTensorToMesh(self.device)
                        self._pos_typecast_pad_buf = ttnn.from_torch(
                            pad_torch,
                            device=self.device,
                            dtype=cp.dtype,
                            layout=ttnn.ROW_MAJOR_LAYOUT,
                            mesh_mapper=mesh_mapper,
                            memory_config=ttnn.DRAM_MEMORY_CONFIG,
                        )
                    cp = ttnn.concat([cp, self._pos_typecast_pad_buf], dim=-1)
                cp = ttnn.typecast(cp, ttnn.int32)
                if orig_size % 32 != 0:
                    cp = ttnn.slice(cp, [0], [orig_size])
            return cp

        # Non-traced / host-tensor path
        if cache_position is None:
            cur_pos = past_key_values.get_seq_length(layer_idx)
            cache_position_tensor = torch.tensor([cur_pos], dtype=torch.int32)
        else:
            if isinstance(cp, TorchTTNNTensor):
                cp = cp.to_torch
            if isinstance(cp, ttnn.Tensor):
                mesh_composer = None
                if hasattr(cp, "device") and cp.device() is not None and cp.device().get_num_devices() > 1:
                    mesh_composer = ttnn.ConcatMeshToTensor(cp.device(), dim=0)
                cp = ttnn.to_torch(cp, mesh_composer=mesh_composer)
            cache_position_tensor = cp.flatten()[:batch_size].to(torch.int32)

        mesh_mapper = ttnn.ReplicateTensorToMesh(self.device) if self.device.get_num_devices() > 1 else None
        return ttnn.from_torch(
            cache_position_tensor,
            device=self.device,
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=mesh_mapper,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
