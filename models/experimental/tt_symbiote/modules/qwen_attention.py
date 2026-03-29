# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Qwen3.5-35B-A3B Attention implementations for TTNN.

This module provides TTNN-accelerated attention mechanisms specific to Qwen3.5-35B-A3B:
- TTNNQwenPagedAttentionKVCache: Paged KV cache with layer_indices mapping for hybrid attention
- TTNNQwen3FullAttention: Full GQA attention with Q gating and Q/K normalization
- TTNNQwen3LinearAttention: Linear attention (DeltaNet) with TTNN-accelerated projections
"""

import os
from typing import Optional

import torch
import torch.nn.functional as F

import ttnn
from models.experimental.tt_symbiote.core.module import TTNNModule, tree_map
from models.experimental.tt_symbiote.core.tensor import TorchTTNNTensor
from models.experimental.tt_symbiote.core.run_config import DistributedTensorConfig
from models.experimental.tt_symbiote.modules.attention import (
    TTNNPagedAttentionKVCache,
    PagedAttentionConfig,
    TTNNSDPAAttention,
)
from models.experimental.tt_symbiote.modules.linear import (
    TTNNLinear,
    TTNNLinearIReplicatedWColSharded,
)
from models.experimental.tt_symbiote.modules.rope import (
    TTNNRotaryPositionEmbedding,
)


class TTNNQwenPagedAttentionKVCache(TTNNPagedAttentionKVCache):
    """Paged attention KV cache with layer indices mapping for Qwen3.5 hybrid attention.

    Qwen3.5 uses hybrid attention with pattern [linear, linear, linear, full] x 10.
    Only full attention layers (10 total) use KV cache, so we need to map
    absolute layer_idx (3, 7, 11, ..., 39) to cache indices (0, 1, 2, ..., 9).
    """

    def __init__(
        self,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        config: PagedAttentionConfig,
        device=None,
        dtype: torch.dtype = torch.bfloat16,
        layer_indices: Optional[list[int]] = None,
    ):
        """Initialize Qwen paged attention KV cache.

        Args:
            num_layers: Number of cache slots (10 for Qwen3.5 full attention layers)
            num_kv_heads: Number of KV heads (2 for Qwen3.5)
            head_dim: Head dimension (256 for Qwen3.5)
            config: Paged attention configuration
            device: TTNN device or mesh device
            dtype: Data type for cache tensors
            layer_indices: Optional list mapping cache indices to absolute layer indices.
                           If None, uses identity mapping.
                           For Qwen3.5: [3, 7, 11, 15, 19, 23, 27, 31, 35, 39]
        """
        super().__init__(
            num_layers=num_layers,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            config=config,
            device=device,
            dtype=dtype,
        )
        # Map absolute layer_idx to cache index
        # For Qwen3.5: layer_indices = [3, 7, 11, 15, 19, 23, 27, 31, 35, 39]
        # This means layer_idx=3 maps to cache_idx=0, layer_idx=7 maps to cache_idx=1, etc.
        if layer_indices is not None:
            self._layer_to_cache_idx = {layer_idx: cache_idx for cache_idx, layer_idx in enumerate(layer_indices)}
        else:
            self._layer_to_cache_idx = None

        # Linear attention compatibility properties
        # Native Qwen linear attention layers (Qwen3_5MoeGatedDeltaNet) check these attributes
        # These are needed when TT_QWEN_CPU_LINEAR_ATTN=1 falls back to native PyTorch linear attention
        self._has_previous_state = False  # Track if cache has been used
        self.conv_states = {}  # DeltaNet conv states per layer
        self.recurrent_states = {}  # DeltaNet recurrent states per layer

    @property
    def has_previous_state(self) -> bool:
        """Check if cache has been used previously.

        This property is required for compatibility with native Qwen linear attention (DeltaNet).
        Returns True if any linear attention layer's conv_state has been populated.
        This mimics the PyTorch Qwen cache behavior which checks if conv_states exist.
        """
        # Check if any conv_state has been populated during prefill
        if self.conv_states:
            # Return True if any layer has a conv_state (meaning prefill has occurred)
            return any(v is not None for v in self.conv_states.values())
        return False

    @has_previous_state.setter
    def has_previous_state(self, value: bool):
        """Set previous state flag."""
        self._has_previous_state = value

    def _get_cache_idx(self, layer_idx: int) -> int:
        """Convert absolute layer_idx to cache index.

        Args:
            layer_idx: Absolute layer index in the model (e.g., 3, 7, 11, ...)

        Returns:
            Cache index (0, 1, 2, ...)
        """
        if self._layer_to_cache_idx is not None:
            return self._layer_to_cache_idx.get(layer_idx, layer_idx)
        return layer_idx

    def paged_fill_on_device(
        self,
        key_states: ttnn.Tensor,
        value_states: ttnn.Tensor,
        layer_idx: int,
        batch_idx: int = 0,
    ):
        """Fill KV cache for a layer, mapping layer_idx to cache_idx."""
        cache_idx = self._get_cache_idx(layer_idx)
        super().paged_fill_on_device(key_states, value_states, cache_idx, batch_idx)

    def paged_update_on_device(
        self,
        key_states: ttnn.Tensor,
        value_states: ttnn.Tensor,
        layer_idx: int,
        current_pos: ttnn.Tensor,
    ):
        """Update KV cache for a layer, mapping layer_idx to cache_idx."""
        cache_idx = self._get_cache_idx(layer_idx)
        super().paged_update_on_device(key_states, value_states, cache_idx, current_pos)

    def paged_sdpa_decode(
        self,
        query: ttnn.Tensor,
        layer_idx: int,
        current_pos: ttnn.Tensor,
        scale: float = 1.0,
        program_config=None,
        compute_kernel_config=None,
    ) -> ttnn.Tensor:
        """Decode using paged KV cache, mapping layer_idx to cache_idx."""
        cache_idx = self._get_cache_idx(layer_idx)
        return super().paged_sdpa_decode(query, cache_idx, current_pos, scale, program_config, compute_kernel_config)

    def get_seq_length(self, layer_idx: int = 0) -> int:
        """Get sequence length for a layer, mapping layer_idx to cache_idx."""
        cache_idx = self._get_cache_idx(layer_idx) if layer_idx != 0 else 0
        return super().get_seq_length(cache_idx)


class TTNNQwen3FullAttention(TTNNModule):
    """TTNN-accelerated Full Attention for Qwen3.5-35B-A3B.

    Implements Grouped Query Attention (GQA) with:
    - 16 attention heads
    - 2 KV heads (8:1 ratio, each KV head serves 8 Q heads)
    - head_dim = 256
    - RoPE position embeddings
    - Q gating: q_proj outputs 2x dimension, split into Q and gate
    - Q/K normalization: RMSNorm on head_dim

    Supports both standard DynamicCache and TTNNQwenPagedAttentionKVCache
    for paged attention with on-device KV storage.
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

        # Q gating support - q_proj outputs 2x dimension, split into Q and gate
        # gate is applied after attention: output *= sigmoid(gate)
        self.has_q_gate = False

        # Q/K normalization (RMSNorm on head_dim)
        self.q_norm_weight = None  # Host tensor
        self.k_norm_weight = None  # Host tensor
        self.tt_q_norm_weight = None  # Device tensor
        self.tt_k_norm_weight = None  # Device tensor
        self.rms_norm_eps = 1e-6

    @classmethod
    def from_torch(cls, torch_attn, distributed: bool = True):
        """Create TTNNQwen3FullAttention from PyTorch Qwen3_5MoeAttention.

        Args:
            torch_attn: PyTorch Qwen3_5MoeAttention layer
            distributed: Whether to use distributed linear layers (default True for T3K)

        Returns:
            TTNNQwen3FullAttention instance
        """
        new_attn = cls()
        new_attn._fallback_torch_layer = torch_attn

        # Extract configuration from torch layer
        config = torch_attn.config
        new_attn.num_attention_heads = config.num_attention_heads  # 16
        new_attn.num_key_value_heads = config.num_key_value_heads  # 2
        new_attn.num_key_value_groups = new_attn.num_attention_heads // new_attn.num_key_value_heads  # 8
        new_attn.head_dim = config.head_dim  # 256
        new_attn.hidden_size = config.hidden_size  # 2048
        new_attn.scaling = new_attn.head_dim**-0.5
        new_attn.layer_idx = torch_attn.layer_idx

        # Check for Q gating: q_proj outputs 2x dimension for Q + gate
        # PyTorch: query_states, gate = torch.chunk(self.q_proj(hidden_states).view(..., head_dim * 2), 2, dim=-1)
        q_proj_out_features = torch_attn.q_proj.out_features
        expected_q_dim = new_attn.num_attention_heads * new_attn.head_dim
        new_attn.has_q_gate = q_proj_out_features == expected_q_dim * 2

        # Extract Q/K normalization weights if present
        # Qwen3 uses RMSNorm on head_dim with weight initialized to zeros
        # Forward: output = rms_norm(x) * (1.0 + weight)
        if hasattr(torch_attn, "q_norm") and torch_attn.q_norm is not None:
            new_attn.q_norm_weight = torch_attn.q_norm.weight.detach().clone()
            new_attn.rms_norm_eps = getattr(torch_attn.q_norm, "eps", 1e-6)
        if hasattr(torch_attn, "k_norm") and torch_attn.k_norm is not None:
            new_attn.k_norm_weight = torch_attn.k_norm.weight.detach().clone()

        # Choose linear layer classes based on distributed mode
        # Input projections take replicated input and produce col-sharded output (like linear attention)
        LinearClsIn = TTNNLinearIReplicatedWColSharded if distributed else TTNNLinear
        # Output projection also takes replicated input (after attention gather) and produces col-sharded output
        LinearClsOut = TTNNLinearIReplicatedWColSharded if distributed else TTNNLinear

        # Create TTNN linear projections
        new_attn.q_proj = LinearClsIn.from_torch(torch_attn.q_proj)
        new_attn.k_proj = LinearClsIn.from_torch(torch_attn.k_proj)
        new_attn.v_proj = LinearClsIn.from_torch(torch_attn.v_proj)
        new_attn.o_proj = LinearClsOut.from_torch(torch_attn.o_proj)

        # RoPE for position embeddings
        # Qwen3.5 uses partial rotary (rotary_dim=64, head_dim=256, factor=0.25),
        # so we always use non-distributed RoPE which handles partial rotary correctly
        new_attn.rope = TTNNRotaryPositionEmbedding()

        # SDPA for attention computation
        new_attn.sdpa = TTNNSDPAAttention()

        # Default core grid (will be updated in move_weights_to_device_impl)
        new_attn.core_grid = ttnn.CoreGrid(y=8, x=8)

        return new_attn

    def preprocess_weights_impl(self):
        """Preprocess Q/K normalization weights for TTNN."""
        super().preprocess_weights_impl()

        # Prepare Q/K norm weights for TTNN
        # Qwen3 RMSNorm applies: output = rms_norm(x) * (1.0 + weight)
        # TTNN rms_norm applies: output = rms_norm(x) * weight
        # So we need to add 1.0 to the weight before converting
        if self.q_norm_weight is not None:
            # (1.0 + weight) for Qwen3-style RMSNorm
            q_norm_adjusted = (1.0 + self.q_norm_weight.float()).to(self.q_norm_weight.dtype)
            # Expand to [1, head_dim] for broadcasting in rms_norm
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
        """Initialize SDPA config and move norm weights to device."""
        super().move_weights_to_device_impl()

        # Query grid dynamically from device (REQUIRED per CLAUDE.md)
        grid = self.device.compute_with_storage_grid_size()
        self.core_grid = ttnn.CoreGrid(y=grid.y, x=grid.x)

        # Move Q/K norm weights to device with proper mesh replication
        # These weights must be replicated to ALL mesh devices, not just device 0
        mesh_mapper = ttnn.ReplicateTensorToMesh(self.device) if self.device.get_num_devices() > 1 else None

        if hasattr(self, "tt_q_norm_weight_host") and self.tt_q_norm_weight_host is not None:
            q_norm_torch = ttnn.to_torch(self.tt_q_norm_weight_host)
            self.tt_q_norm_weight = ttnn.from_torch(
                q_norm_torch,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=self.device,
                mesh_mapper=mesh_mapper,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        if hasattr(self, "tt_k_norm_weight_host") and self.tt_k_norm_weight_host is not None:
            k_norm_torch = ttnn.to_torch(self.tt_k_norm_weight_host)
            self.tt_k_norm_weight = ttnn.from_torch(
                k_norm_torch,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=self.device,
                mesh_mapper=mesh_mapper,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

        if self.sdpa.program_config is None:
            self.sdpa.program_config = ttnn.SDPAProgramConfig(
                compute_with_storage_grid_size=(self.core_grid.x, self.core_grid.y),
                q_chunk_size=128,  # Reduced for head_dim=256 (matches DeepSeek V3)
                k_chunk_size=128,  # Reduced for head_dim=256 (matches DeepSeek V3)
                exp_approx_mode=False,
            )
            # Match DeepSeek V3 settings for head_dim=256 compatibility:
            # fp32_dest_acc_en=False increases dst_size from 4 to 8
            # packer_l1_acc=False reduces L1 pressure
            self.sdpa.compute_kernel_config = ttnn.init_device_compute_kernel_config(
                self.device.arch(),
                math_fidelity=ttnn.MathFidelity.HiFi4,
                math_approx_mode=False,
                fp32_dest_acc_en=False,
                packer_l1_acc=False,
            )

    @property
    def _is_distributed(self):
        return (
            self.device_state is not None
            and hasattr(self.device_state, "ccl_manager")
            and self.device_state.ccl_manager is not None
        )

    def set_output_tensors_config_impl(self, output_tensors):
        """Set output tensor config for col-sharded output.

        The o_proj output is col-sharded (each device has [batch, seq, hidden_size/8]).
        We need to use ConcatMeshToTensor on dim=-1 to concatenate the shards.
        """

        def set_col_sharded_config(e):
            if isinstance(e, TorchTTNNTensor) and e.ttnn_tensor is not None:
                if self._is_distributed and self.device is not None:
                    # Use ConcatMeshToTensor on dim=-1 only (not batch dim)
                    # This concatenates the col-sharded output from all devices
                    mesh_composer = ttnn.ConcatMeshToTensor(self.device, dim=-1)
                    mesh_mapper = ttnn.ShardTensorToMesh(self.device, dim=-1)

                    def logical_shape_for_col_sharded(shape):
                        """Compute logical shape by multiplying last dim by num_devices."""
                        shape_list = list(shape)
                        num_devices = self.device.get_num_devices()
                        shape_list[-1] = shape_list[-1] * num_devices
                        return tuple(shape_list)

                    config = DistributedTensorConfig(
                        mesh_mapper=mesh_mapper,
                        mesh_composer=mesh_composer,
                        logical_shape_fn=logical_shape_for_col_sharded,
                    )
                    e.set_distributed_tensor_config(config)
            return e

        # Use the default config from parent if not distributed
        if not self._is_distributed:
            return super().set_output_tensors_config_impl(output_tensors)

        return tree_map(set_col_sharded_config, output_tensors)

    def _maybe_all_gather(self, tensor):
        """All-gather tensor across mesh devices if in distributed mode."""
        if not self._is_distributed:
            return tensor
        t = tensor
        gathered = ttnn.all_gather(
            t,
            dim=-1,
            num_links=1,
            topology=ttnn.Topology.Linear,
        )
        # Synchronize to ensure all-gather completes before returning
        ttnn.synchronize_device(self.device)
        return gathered

    def _is_tensor_replicated(self, tensor) -> bool:
        """Check if tensor is replicated across devices (vs sharded).

        Returns True if tensor uses ReplicateTensorToMesh or has full hidden_size,
        False if tensor is sharded (each device has hidden_size/num_devices).
        """
        if tensor is None:
            return True

        # Check for distributed config first
        if hasattr(tensor, "ttnn_distributed_tensor_config"):
            config = tensor.ttnn_distributed_tensor_config
            if config is not None:
                mapper = config.mesh_mapper
                if mapper is not None:
                    mapper_type = type(mapper).__name__
                    if "Replicate" in mapper_type:
                        return True
                    if "Shard" in mapper_type:
                        return False
                return False

        # Check physical shape - sharded tensors have hidden_size/num_devices on last dim
        physical_shape = None
        if hasattr(tensor, "ttnn_tensor") and tensor.ttnn_tensor is not None:
            physical_shape = tuple(int(i) for i in tensor.ttnn_tensor.shape)
        elif isinstance(tensor, ttnn.Tensor):
            physical_shape = tuple(int(i) for i in tensor.shape)
        elif hasattr(tensor, "shape") and tensor.shape is not None:
            physical_shape = tuple(tensor.shape)

        if physical_shape is not None and len(physical_shape) >= 1 and self.device is not None:
            num_devices = self.device.get_num_devices() if hasattr(self.device, "get_num_devices") else 1
            if num_devices > 1:
                last_dim = physical_shape[-1]
                if last_dim == self.hidden_size:
                    return True  # Full hidden_size = replicated
                elif last_dim == self.hidden_size / num_devices:
                    return False  # Partial hidden_size = sharded

        return False  # Default to sharded (safer)

    def _repeat_kv(self, hidden_states: ttnn.Tensor, n_rep: int) -> ttnn.Tensor:
        """Repeat KV heads to match Q heads for GQA.

        For Qwen3.5: 2 KV heads -> 16 Q heads, so n_rep=8
        [batch, num_kv_heads, seq_len, head_dim] -> [batch, num_attention_heads, seq_len, head_dim]

        Uses repeat_interleave for correct GQA head ordering:
        - Correct: [K0,K0,...,K0, K1,K1,...,K1] - Q heads 0-7 attend to K0, Q heads 8-15 attend to K1
        - Wrong (ttnn.repeat tiles): [K0,K1,K0,K1,...] - Q head 1 would wrongly attend to K1
        """
        if n_rep == 1:
            return hidden_states
        # Use repeat_interleave for correct GQA head ordering
        return ttnn.repeat_interleave(hidden_states, n_rep, dim=1)

    def _to_replicated(self, tensor: ttnn.Tensor) -> ttnn.Tensor:
        """Convert a multi-device tensor to an explicitly replicated tensor.

        After all-gather the data is identical on every device but the mesh
        topology metadata differs from ReplicateTensorToMesh. Paged-attention
        kernels require the replicated topology, so we round-trip through the
        host for decode tokens (tiny tensors, negligible overhead).
        """
        if self.device.get_num_devices() <= 1:
            return tensor
        t = tensor
        if isinstance(t, TorchTTNNTensor):
            t = t.to_ttnn if hasattr(t, "to_ttnn") else t
        orig_shape = list(t.shape)
        mesh_composer = ttnn.ConcatMeshToTensor(self.device, dim=0)
        t_torch = ttnn.to_torch(t, mesh_composer=mesh_composer)
        t_torch = t_torch[: orig_shape[0]]
        return ttnn.from_torch(
            t_torch,
            device=self.device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.device),
            dtype=t.dtype,
            layout=t.layout,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def _project_qkv(self, hidden_states, batch_size, seq_length, position_embeddings):
        """Project hidden states to Q, K, V, apply Q/K norm, and apply RoPE.

        Args:
            hidden_states: Input tensor [batch, seq_len, hidden_size]
            batch_size: Batch dimension
            seq_length: Sequence length
            position_embeddings: Tuple of (cos, sin) for RoPE

        Returns:
            Tuple of (query_states, key_states, value_states, gate, cos, sin)
            gate is None if has_q_gate is False
        """
        # ALL-GATHER INPUT IF SHARDED: TTNNLinearIReplicatedWColSharded expects
        # replicated input [batch, seq, hidden_size]. If input is col-sharded
        # (from previous MoE layer), all-gather it first.
        if self._is_distributed and not self._is_tensor_replicated(hidden_states):
            hidden_states = self._maybe_all_gather(hidden_states)

        if hidden_states.layout != ttnn.TILE_LAYOUT:
            hidden_states = ttnn.to_layout(hidden_states, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        # Project to Q, K, V
        # For Qwen3: q_proj outputs [batch, seq, num_heads * head_dim * 2] when has_q_gate=True
        q_proj_output = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)  # [batch, seq, num_kv_heads * head_dim]
        value_states = self.v_proj(hidden_states)  # [batch, seq, num_kv_heads * head_dim]

        # All-gather for distributed mode
        q_proj_output = self._maybe_all_gather(q_proj_output)
        key_states = self._maybe_all_gather(key_states)
        value_states = self._maybe_all_gather(value_states)

        # Handle Q gating: split q_proj output into query_states and gate
        # PyTorch: query_states, gate = torch.chunk(q_proj(x).view(..., head_dim * 2), 2, dim=-1)
        gate = None
        if self.has_q_gate:
            # q_proj_output: [batch, seq, num_heads * head_dim * 2]
            # Reshape to [batch, seq, num_heads, head_dim * 2] then split
            q_proj_output = ttnn.reshape(
                q_proj_output, (batch_size, seq_length, self.num_attention_heads, self.head_dim * 2)
            )
            # Split along last dim: first half is Q, second half is gate
            # TTNN doesn't have a direct chunk, use slicing
            query_states = q_proj_output[:, :, :, : self.head_dim]  # [batch, seq, num_heads, head_dim]
            gate = q_proj_output[:, :, :, self.head_dim :]  # [batch, seq, num_heads, head_dim]
            # Reshape gate to [batch, seq, num_heads * head_dim] for later sigmoid multiply
            gate = ttnn.reshape(gate, (batch_size, seq_length, self.num_attention_heads * self.head_dim))
        else:
            # No gating - standard Q projection
            query_states = ttnn.reshape(
                q_proj_output, (batch_size, seq_length, self.num_attention_heads, self.head_dim)
            )

        # Apply Q/K normalization (RMSNorm on head_dim) before RoPE
        # PyTorch: query_states = self.q_norm(query_states.view(hidden_shape))
        # query_states shape here: [batch, seq, num_heads, head_dim]
        if self.tt_q_norm_weight is not None:
            # Flatten for rms_norm: [batch * seq * num_heads, head_dim]
            orig_q_shape = query_states.shape
            query_states = ttnn.reshape(
                query_states, (batch_size * seq_length * self.num_attention_heads, self.head_dim)
            )
            query_states = ttnn.rms_norm(query_states, weight=self.tt_q_norm_weight, epsilon=self.rms_norm_eps)
            query_states = ttnn.reshape(query_states, orig_q_shape)

        # Reshape K to [batch, seq, num_kv_heads, head_dim] for normalization
        key_states = ttnn.reshape(key_states, (batch_size, seq_length, self.num_key_value_heads, self.head_dim))

        if self.tt_k_norm_weight is not None:
            orig_k_shape = key_states.shape
            key_states = ttnn.reshape(key_states, (batch_size * seq_length * self.num_key_value_heads, self.head_dim))
            key_states = ttnn.rms_norm(key_states, weight=self.tt_k_norm_weight, epsilon=self.rms_norm_eps)
            key_states = ttnn.reshape(key_states, orig_k_shape)

        # Permute to [batch, num_heads, seq_len, head_dim] for attention
        query_states = ttnn.permute(query_states, (0, 2, 1, 3))
        key_states = ttnn.permute(key_states, (0, 2, 1, 3))

        value_states = ttnn.reshape(value_states, (batch_size, seq_length, self.num_key_value_heads, self.head_dim))
        value_states = ttnn.permute(value_states, (0, 2, 1, 3))

        # Apply RoPE to Q and K
        cos, sin = position_embeddings
        if len(cos.shape) == 3:
            cos = ttnn.unsqueeze(cos, 1)
        if len(sin.shape) == 3:
            sin = ttnn.unsqueeze(sin, 1)
        if self._is_distributed:
            cos = self._maybe_all_gather(cos)
            sin = self._maybe_all_gather(sin)

        query_states, key_states = self.rope(query_states, key_states, cos, sin)

        # Expand KV to match Q heads (GQA)
        key_states = self._repeat_kv(key_states, self.num_key_value_groups)
        value_states = self._repeat_kv(value_states, self.num_key_value_groups)

        return query_states, key_states, value_states, gate, cos, sin

    def _forward_prefill(
        self,
        hidden_states: ttnn.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[ttnn.Tensor],
        past_key_values,
        cache_position: Optional[torch.LongTensor],
    ) -> tuple[ttnn.Tensor, Optional[torch.Tensor]]:
        """Prefill path for attention computation."""
        batch_size, seq_length = hidden_states.shape[0], hidden_states.shape[1]

        query_states, key_states, value_states, gate, cos, sin = self._project_qkv(
            hidden_states, batch_size, seq_length, position_embeddings
        )

        use_paged = isinstance(past_key_values, TTNNQwenPagedAttentionKVCache) or isinstance(
            past_key_values, TTNNPagedAttentionKVCache
        )

        if past_key_values is not None:
            # For paged cache with layer_indices mapping, pass the actual layer_idx.
            # The KV cache will internally map it to the correct cache slot.
            # For non-paged cache, also use the actual layer_idx.
            cache_layer_idx = self.layer_idx

            if use_paged:
                # For paged cache, key_states and value_states need to be in
                # [batch, num_kv_heads, seq, head_dim] format before fill
                # But they're already expanded for GQA, so we need the original
                # Recompute unexpanded KV for cache storage
                kv_key = key_states[:, :: self.num_key_value_groups, :, :]  # Sample every n-th head
                kv_value = value_states[:, :: self.num_key_value_groups, :, :]

                past_key_values.paged_fill_on_device(
                    kv_key,
                    kv_value,
                    layer_idx=cache_layer_idx,
                    batch_idx=0,
                )
            else:
                # Standard cache path (non-paged) - uses absolute layer_idx
                cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
                # Get unexpanded KV for cache
                kv_key = key_states[:, :: self.num_key_value_groups, :, :]
                kv_value = value_states[:, :: self.num_key_value_groups, :, :]

                torch_tensors = [TorchTTNNTensor(kv_key), TorchTTNNTensor(kv_value)]
                orig_shapes = [kv_key.shape, kv_value.shape]

                torch_tensors = [
                    torch_tensor.to_torch[: orig_shape[0], : orig_shape[1], : orig_shape[2], : orig_shape[3]]
                    for orig_shape, torch_tensor in zip(orig_shapes, torch_tensors)
                ]

                cached_key, cached_value = past_key_values.update(
                    *torch_tensors,
                    self.layer_idx,  # Standard cache uses absolute layer_idx
                    cache_kwargs,
                )
                cached_key, cached_value = [TorchTTNNTensor(cached_key), TorchTTNNTensor(cached_value)]
                cached_key = ttnn.to_device(cached_key.to_ttnn, self.device)
                cached_value = ttnn.to_device(cached_value.to_ttnn, self.device)
                cached_key = self._maybe_all_gather(cached_key)
                cached_value = self._maybe_all_gather(cached_value)

                # Expand cached KV for attention
                key_states = self._repeat_kv(cached_key, self.num_key_value_groups)
                value_states = self._repeat_kv(cached_value, self.num_key_value_groups)

        # Compute attention
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

        # Reshape output: [batch, seq_len, num_heads, head_dim] -> [batch, seq_len, hidden_size]
        # Use actual tensor shape after SDPA
        attn_shape = list(attn_output.shape)
        attn_batch = attn_shape[0]
        attn_seq = attn_shape[1]
        attn_output = ttnn.reshape(attn_output, (attn_batch, attn_seq, self.num_attention_heads * self.head_dim))

        # Apply Q gating: attn_output = attn_output * sigmoid(gate)
        # PyTorch: attn_output = attn_output.reshape(*input_shape, -1).contiguous() * torch.sigmoid(gate)
        if gate is not None:
            gate_sigmoid = ttnn.sigmoid(gate)
            attn_output = ttnn.mul(attn_output, gate_sigmoid)

        # Output projection
        attn_output = self.o_proj(attn_output)

        return attn_output, None

    def _forward_decode_paged(
        self,
        hidden_states: ttnn.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[ttnn.Tensor],
        past_key_values,
        cache_position: Optional[torch.LongTensor],
    ) -> tuple[ttnn.Tensor, Optional[torch.Tensor]]:
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
        # _project_qkv returns [B, H, S, D]:
        #   Q : [B, num_attention_heads, 1, head_dim]
        #   K : [B, num_attention_heads, 1, head_dim] (already expanded)
        #   V : [B, num_attention_heads, 1, head_dim] (already expanded)
        # gate: [B, S, num_heads * head_dim] or None

        layer_idx = self.layer_idx

        # Get unexpanded KV for cache update
        kv_key = key_states[:, :: self.num_key_value_groups, :, :]
        kv_value = value_states[:, :: self.num_key_value_groups, :, :]

        # --- resolve cache position to a 1-D torch int32 tensor [batch] ---
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

        mesh_mapper = ttnn.ReplicateTensorToMesh(self.device) if self.device.get_num_devices() > 1 else None

        # 1-D [batch_size] tensor for paged_update_cache & paged_sdpa_decode
        cur_pos_tt = ttnn.from_torch(
            cache_position_tensor,
            device=self.device,
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=mesh_mapper,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        # --- permute B H S D -> S B H D (the layout paged kernels expect) ---
        query_states_paged = ttnn.permute(query_states, (2, 0, 1, 3))
        kv_key = ttnn.permute(kv_key, (2, 0, 1, 3))
        kv_value = ttnn.permute(kv_value, (2, 0, 1, 3))

        # --- multi-device: convert all-gathered topology -> replicated ---
        if self.device.get_num_devices() > 1:
            query_states_paged = self._to_replicated(query_states_paged)
            kv_key = self._to_replicated(kv_key)
            kv_value = self._to_replicated(kv_value)

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

        # --- update the on-device paged KV cache ---
        # paged_update_on_device handles _seq_lengths and _seen_tokens updates internally
        past_key_values.paged_update_on_device(
            kv_key,
            kv_value,
            layer_idx=layer_idx,
            current_pos=cur_pos_tt,
        )
        ttnn.deallocate(kv_key)
        ttnn.deallocate(kv_value)

        # --- paged SDPA decode (Q stays in DRAM) ---
        # Note: For GQA, the paged attention handles the KV head expansion internally
        attn_output = past_key_values.paged_sdpa_decode(
            query_states_paged,
            layer_idx,
            current_pos=cur_pos_tt,
            scale=self.scaling,
            program_config=self.sdpa.program_config,
            compute_kernel_config=self.sdpa.compute_kernel_config,
        )
        # attn_output: [1, B, H, head_dim]

        # --- convert back to [B, S, H*D] for the output projection ---
        attn_output = ttnn.permute(attn_output, (1, 0, 2, 3))  # [B, 1, H, head_dim]
        # Use actual tensor shape after attention processing
        attn_shape = list(attn_output.shape)
        attn_batch = attn_shape[0]
        attn_seq = attn_shape[1]
        attn_output = ttnn.reshape(attn_output, (attn_batch, attn_seq, self.num_attention_heads * self.head_dim))

        # Apply Q gating: attn_output = attn_output * sigmoid(gate)
        if gate is not None:
            gate_sigmoid = ttnn.sigmoid(gate)
            attn_output = ttnn.mul(attn_output, gate_sigmoid)

        attn_output = self.o_proj(attn_output)

        return attn_output, None

    def forward(
        self,
        hidden_states: ttnn.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[ttnn.Tensor] = None,
        past_key_values=None,
        cache_position: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> tuple[ttnn.Tensor, Optional[torch.Tensor]]:
        """Forward pass through Qwen3 full attention.

        Args:
            hidden_states: Input tensor [batch, seq_len, hidden_size]
            position_embeddings: Tuple of (cos, sin) for RoPE
            attention_mask: Optional attention mask
            past_key_values: Optional KV cache (TTNNQwenPagedAttentionKVCache or DynamicCache)
            cache_position: Optional cache position tensor
            position_ids: Optional position IDs (unused, for API compatibility)
            **kwargs: Additional arguments for forward compatibility

        Returns:
            Tuple of (attention_output, None)
        """
        # CPU fallback for debugging
        use_cpu_fallback = os.environ.get("TT_QWEN_CPU_FULL_ATTN", "0").lower() in ("1", "true", "yes")
        if use_cpu_fallback:
            # Log only once per layer instance
            if not getattr(self, "_cpu_fallback_logged", False):
                print(f"[DEBUG] TT_QWEN_CPU_FULL_ATTN=1: Using PyTorch for full attention (layer {self.layer_idx})")
                self._cpu_fallback_logged = True

            # Helper to convert any tensor type to PyTorch, handling multi-device tensors
            def to_pytorch(t):
                if t is None:
                    return None

                # Get the underlying TTNN tensor and distributed config if it's a TorchTTNNTensor wrapper
                ttnn_t = None
                dist_config = None
                if hasattr(t, "ttnn_tensor") and t.ttnn_tensor is not None:
                    ttnn_t = t.ttnn_tensor
                    if hasattr(t, "ttnn_distributed_tensor_config"):
                        dist_config = t.ttnn_distributed_tensor_config
                elif isinstance(t, ttnn.Tensor):
                    ttnn_t = t

                # If we have a TTNN tensor, convert it with mesh composer
                if ttnn_t is not None:
                    mesh_composer = None
                    is_replicated = False
                    try:
                        dev = ttnn_t.device()
                        if dev is not None and hasattr(dev, "get_num_devices") and dev.get_num_devices() > 1:
                            # Use the tensor's configured mesh_composer if available
                            if dist_config is not None and hasattr(dist_config, "mesh_composer"):
                                mesh_composer = dist_config.mesh_composer
                                # Check if it's replicated (post_process_fn is set to slice first element)
                                if hasattr(dist_config, "post_process_fn") and dist_config.post_process_fn is not None:
                                    is_replicated = True
                            else:
                                # Default to concat on dim=-1 (col-sharded is most common for hidden_states)
                                mesh_composer = ttnn.ConcatMeshToTensor(dev, dim=-1)
                    except Exception:
                        pass
                    result = ttnn.to_torch(ttnn_t, mesh_composer=mesh_composer)
                    # Apply post-processing for replicated tensors
                    if (
                        dist_config is not None
                        and hasattr(dist_config, "post_process")
                        and callable(dist_config.post_process)
                    ):
                        result = dist_config.post_process(result)
                    return result

                # If the TorchTTNNTensor has a PyTorch elem, try to use it
                if hasattr(t, "elem") and t.elem is not None:
                    return t.elem

                # Fallback - just return as-is (already PyTorch tensor)
                return t

            # Convert hidden_states to PyTorch
            hs_pt = to_pytorch(hidden_states)

            # Convert position_embeddings tuple (cos, sin) to PyTorch
            if isinstance(position_embeddings, (tuple, list)):
                pos_emb_pt = tuple(to_pytorch(p) for p in position_embeddings)
            else:
                pos_emb_pt = to_pytorch(position_embeddings)

            # Convert cache_position to PyTorch
            cache_pos_pt = to_pytorch(cache_position)

            # Convert attention_mask to PyTorch
            attn_mask_pt = to_pytorch(attention_mask)

            return self._fallback_torch_layer(
                hs_pt,
                position_embeddings=pos_emb_pt,
                attention_mask=attn_mask_pt,
                past_key_values=None,  # Can't use TTNN paged cache with PyTorch
                cache_position=cache_pos_pt,
                **kwargs,
            )

        seq_length = hidden_states.shape[1]

        use_paged = isinstance(past_key_values, TTNNQwenPagedAttentionKVCache) or isinstance(
            past_key_values, TTNNPagedAttentionKVCache
        )

        if use_paged and seq_length == 1:
            ttnn_output, _ = self._forward_decode_paged(
                hidden_states,
                position_embeddings,
                attention_mask,
                past_key_values,
                cache_position,
            )
        else:
            ttnn_output, _ = self._forward_prefill(
                hidden_states,
                position_embeddings,
                attention_mask,
                past_key_values,
                cache_position,
            )

        return ttnn_output, None


class TTNNQwen3LinearAttention(TTNNModule):
    """TTNN-accelerated Linear Attention (DeltaNet/Mamba-style) for Qwen3.5-35B-A3B.

    This module handles 30/40 layers in Qwen3.5 that use linear attention instead
    of full attention. Linear attention uses state-space model computation:
    - NO KV cache (state is computed directly each step)
    - O(n) complexity vs O(n^2) for full attention
    - Uses gating mechanisms (beta, g) and 1D convolutions

    Key components from Qwen3_5MoeGatedDeltaNet:
    - in_proj_qkv: Projects hidden_states to Q, K, V
    - in_proj_z: Projects for gating
    - in_proj_a, in_proj_b: Projects for alpha/beta gates
    - conv1d: Causal convolution for sequential processing
    - norm: RMS norm with gating
    - out_proj: Output projection

    Implementation Strategy (Hybrid):
    - TTNN acceleration for linear projections (in_proj_qkv, in_proj_z, in_proj_a, in_proj_b, out_proj)
    - PyTorch fallback for DeltaNet kernel (conv1d, gating, chunk_gated_delta_rule)
    - Feature-flagged via TTNN_LINEAR_ATTN_PROJECTIONS env var (default: enabled)
    """

    def __init__(self, config=None, distributed: bool = True):
        super().__init__()
        self.config = config
        self.distributed = distributed

        # Layer dimensions (will be set from torch layer)
        self.hidden_size = None
        self.num_v_heads = None
        self.num_k_heads = None
        self.head_k_dim = None
        self.head_v_dim = None
        self.key_dim = None
        self.value_dim = None
        self.conv_kernel_size = None
        self.layer_idx = None

        # TTNN linear projections
        self.in_proj_qkv = None  # TTNNLinear: hidden_size -> key_dim * 2 + value_dim
        self.in_proj_z = None  # TTNNLinear: hidden_size -> value_dim
        self.in_proj_a = None  # TTNNLinear: hidden_size -> num_v_heads
        self.in_proj_b = None  # TTNNLinear: hidden_size -> num_v_heads
        self.out_proj = None  # TTNNLinear: value_dim -> hidden_size

        # DeltaNet kernel parameters (kept on PyTorch)
        self.conv1d = None  # PyTorch Conv1d for causal convolution
        self.dt_bias = None  # Time step bias
        self.A_log = None  # A parameter (log space)
        self.norm = None  # RMSNorm with gating

        # Feature flag for TTNN projections (default: enabled)
        self.use_ttnn_projections = os.environ.get("TTNN_LINEAR_ATTN_PROJECTIONS", "1") == "1"

    @classmethod
    def from_torch(cls, torch_layer, distributed: bool = True):
        """Create TTNNQwen3LinearAttention from PyTorch Qwen3_5MoeGatedDeltaNet.

        Args:
            torch_layer: PyTorch Qwen3_5MoeGatedDeltaNet layer
            distributed: Whether to use distributed linear layers (default True for T3K)

        Returns:
            TTNNQwen3LinearAttention instance
        """
        config = torch_layer.config if hasattr(torch_layer, "config") else None
        new_layer = cls(config, distributed=distributed)
        new_layer._fallback_torch_layer = torch_layer

        # Extract layer dimensions
        new_layer.hidden_size = torch_layer.hidden_size
        new_layer.num_v_heads = torch_layer.num_v_heads
        new_layer.num_k_heads = torch_layer.num_k_heads
        new_layer.head_k_dim = torch_layer.head_k_dim
        new_layer.head_v_dim = torch_layer.head_v_dim
        new_layer.key_dim = torch_layer.key_dim
        new_layer.value_dim = torch_layer.value_dim
        new_layer.conv_kernel_size = torch_layer.conv_kernel_size
        new_layer.layer_idx = torch_layer.layer_idx

        # Choose linear layer classes based on distributed mode
        # Input projections: replicated input -> col-sharded weights -> sharded output
        # Output projection: sharded input -> row-sharded weights -> replicated output (with all-reduce)
        # Note: in_proj_a and in_proj_b have small output dims (num_v_heads=4) that can't be sharded,
        # so they use non-sharded linear layers
        LinearClsIn = TTNNLinearIReplicatedWColSharded if distributed else TTNNLinear
        LinearClsOut = TTNNLinearIReplicatedWColSharded if distributed else TTNNLinear
        LinearClsSmall = TTNNLinear  # Always non-sharded for small projections

        # Create TTNN linear projections
        # These take replicated input (full hidden_states) and produce sharded output
        new_layer.in_proj_qkv = LinearClsIn.from_torch(torch_layer.in_proj_qkv)
        new_layer.in_proj_z = LinearClsIn.from_torch(torch_layer.in_proj_z)
        # in_proj_a and in_proj_b have tiny output dims (num_v_heads=4) that can't be col-sharded
        # Keep as PyTorch layers to avoid distributed weight replication issues
        # (TTNNLinear doesn't replicate weights across mesh devices, causing garbage on non-device-0)
        new_layer.in_proj_a = torch_layer.in_proj_a  # PyTorch nn.Linear
        new_layer.in_proj_b = torch_layer.in_proj_b  # PyTorch nn.Linear
        new_layer.out_proj = LinearClsOut.from_torch(torch_layer.out_proj)

        # Keep DeltaNet kernel components as references (not TTNN)
        new_layer.conv1d = torch_layer.conv1d
        new_layer.dt_bias = torch_layer.dt_bias
        new_layer.A_log = torch_layer.A_log
        new_layer.norm = torch_layer.norm

        # Store kernel functions
        new_layer.causal_conv1d_fn = torch_layer.causal_conv1d_fn
        new_layer.causal_conv1d_update = torch_layer.causal_conv1d_update
        new_layer.chunk_gated_delta_rule = torch_layer.chunk_gated_delta_rule
        new_layer.recurrent_gated_delta_rule = torch_layer.recurrent_gated_delta_rule
        new_layer.activation = torch_layer.activation

        return new_layer

    def deallocate_weights_impl(self):
        """Deallocate TTNN weights from device.

        Note: in_proj_a and in_proj_b are PyTorch layers, not TTNN, so we skip them.
        """
        if self.in_proj_qkv is not None:
            self.in_proj_qkv.deallocate_weights()
        if self.in_proj_z is not None:
            self.in_proj_z.deallocate_weights()
        # in_proj_a and in_proj_b are PyTorch nn.Linear layers, not TTNN modules
        # They don't have deallocate_weights() method
        if self.out_proj is not None:
            self.out_proj.deallocate_weights()

    def set_output_tensors_config_impl(self, output_tensors):
        """Set output tensor config for col-sharded output.

        The out_proj output is col-sharded (each device has [batch, seq, hidden_size/8]).
        We need to use ConcatMeshToTensor on dim=-1 to concatenate the shards.
        """

        def set_col_sharded_config(e):
            if isinstance(e, TorchTTNNTensor) and e.ttnn_tensor is not None:
                if self._is_distributed and self.device is not None:
                    # Use ConcatMeshToTensor on dim=-1 only (not batch dim)
                    # This concatenates the col-sharded output from all devices
                    mesh_composer = ttnn.ConcatMeshToTensor(self.device, dim=-1)
                    mesh_mapper = ttnn.ShardTensorToMesh(self.device, dim=-1)

                    def logical_shape_for_col_sharded(shape):
                        """Compute logical shape by multiplying last dim by num_devices."""
                        shape_list = list(shape)
                        num_devices = self.device.get_num_devices()
                        shape_list[-1] = shape_list[-1] * num_devices
                        return tuple(shape_list)

                    config = DistributedTensorConfig(
                        mesh_mapper=mesh_mapper,
                        mesh_composer=mesh_composer,
                        logical_shape_fn=logical_shape_for_col_sharded,
                    )
                    e.set_distributed_tensor_config(config)
            return e

        # Use the default config from parent if not distributed
        if not self._is_distributed:
            return super().set_output_tensors_config_impl(output_tensors)

        return tree_map(set_col_sharded_config, output_tensors)

    @property
    def _is_distributed(self):
        """Check if running in distributed mode with CCL manager.

        Returns True only if:
        1. Layer was created with distributed=True (from_torch)
        2. Device state has a CCL manager for all-gather operations
        """
        return (
            getattr(self, "distributed", True)  # Check distributed flag from from_torch
            and self.device_state is not None
            and hasattr(self.device_state, "ccl_manager")
            and self.device_state.ccl_manager is not None
        )

    def _maybe_all_gather(self, tensor):
        """All-gather tensor across mesh devices if in distributed mode."""
        if not self._is_distributed:
            return tensor
        t = tensor
        gathered = ttnn.all_gather(
            t,
            dim=-1,
            num_links=1,
            topology=ttnn.Topology.Linear,
        )
        # Synchronize to ensure all-gather completes before returning
        ttnn.synchronize_device(self.device)
        return gathered

    def _is_tensor_replicated(self, tensor) -> bool:
        """Check if tensor is replicated across devices (vs sharded).

        Returns True if tensor uses ReplicateTensorToMesh, False otherwise.
        This allows auto-detection of the correct conversion mode.
        """
        if tensor is None:
            return False

        # Check if tensor has distributed config indicating replication
        if hasattr(tensor, "ttnn_distributed_tensor_config"):
            config = tensor.ttnn_distributed_tensor_config
            if config is not None:
                # KEY FIX: If tensor has a logical_shape_fn, physical shape differs from logical
                # This means it's sharded (each device has a portion of the data)
                # ShardTensorToMesh returns CppTensorToMesh whose name doesn't contain "Shard",
                # so we must check logical_shape_fn instead of relying on mapper type name
                if config.logical_shape_fn is not None:
                    return False  # Sharded, not replicated

                if hasattr(config, "mesh_mapper"):
                    mapper = config.mesh_mapper
                    # Check if it's a ReplicateTensorToMesh mapper
                    mapper_type = type(mapper).__name__
                    if "Replicate" in mapper_type:
                        return True
                    # If it's Shard, return False
                    if "Shard" in mapper_type:
                        return False
                # Config exists but doesn't indicate replication
                return False

        # NO CONFIG CASE: Need to check physical shape directly
        # For TorchTTNNTensor, we must access the underlying ttnn_tensor shape
        # (NOT the .shape property which returns logical shape)

        physical_shape = None

        if hasattr(tensor, "ttnn_tensor") and tensor.ttnn_tensor is not None:
            # TorchTTNNTensor with underlying ttnn tensor - get PHYSICAL shape
            physical_shape = tuple(int(i) for i in tensor.ttnn_tensor.shape)
        elif isinstance(tensor, ttnn.Tensor):
            # Raw ttnn.Tensor - shape is already physical
            physical_shape = tuple(int(i) for i in tensor.shape)
        else:
            # Regular torch tensor or no way to get physical shape
            if hasattr(tensor, "shape") and tensor.shape is not None:
                physical_shape = tuple(tensor.shape)

        if physical_shape is not None and len(physical_shape) >= 1 and self.device is not None:
            num_devices = self.device.get_num_devices() if hasattr(self.device, "get_num_devices") else 1
            if num_devices > 1:
                last_dim = physical_shape[-1]
                # If last dim equals hidden_size, tensor is replicated (full data per device)
                if last_dim == self.hidden_size:
                    return True
                # If last dim equals hidden_size/num_devices, tensor is sharded
                elif last_dim == self.hidden_size // num_devices:
                    return False

        # Unable to determine, default to sharded (safer for distributed ops)
        return False

    def _to_pytorch(self, tensor, replicated=None):
        """Convert TTNN tensor to PyTorch, handling multi-device meshes.

        Args:
            tensor: TTNN or TorchTTNNTensor to convert
            replicated: If True, tensor is replicated (same data on all devices),
                        so we take from first device instead of concatenating.
                        If None, auto-detect based on tensor config.
                        Use True for tensors after all_gather operations.
        """
        # Auto-detect if not specified
        if replicated is None:
            replicated = self._is_tensor_replicated(tensor)
        if tensor is None:
            return None

        # Get original batch size BEFORE conversion to handle multi-device slicing correctly
        original_batch_size = 1
        if hasattr(tensor, "shape") and tensor.shape is not None:
            shape = tensor.shape
            if len(shape) > 0:
                original_batch_size = shape[0]

        # Handle raw ttnn.Tensor objects directly (e.g., from TTNN projections)
        if isinstance(tensor, ttnn.Tensor):
            try:
                device = tensor.device()
                if device is not None and hasattr(device, "get_num_devices") and device.get_num_devices() > 1:
                    num_devices = device.get_num_devices()
                    if replicated:
                        # For replicated tensors, concat on dim=0 then take first batch
                        pt_tensor = ttnn.to_torch(tensor, mesh_composer=ttnn.ConcatMeshToTensor(device, dim=0))
                        batch_per_device = pt_tensor.shape[0] // num_devices
                        return pt_tensor[:batch_per_device]
                    else:
                        # For sharded tensors, concat on last dim to get full tensor
                        return ttnn.to_torch(tensor, mesh_composer=ttnn.ConcatMeshToTensor(device, dim=-1))
                else:
                    return ttnn.to_torch(tensor)
            except Exception as e:
                import logging

                logging.warning(f"TTNNQwen3LinearAttention._to_pytorch: Error converting raw ttnn.Tensor: {e}")
                return ttnn.to_torch(tensor)

        if hasattr(tensor, "to_torch") and callable(tensor.to_torch):
            # Check if tensor is on multi-device mesh
            try:
                device = getattr(tensor, "device", None)
                if device is not None:
                    if callable(device):
                        device = device()
                    if device is not None:
                        num_devices = 1
                        mesh_shape = getattr(device, "shape", None)
                        if mesh_shape is not None and hasattr(mesh_shape, "num_devices"):
                            num_devices = mesh_shape.num_devices
                        elif hasattr(device, "get_num_devices"):
                            num_devices = device.get_num_devices()

                        if num_devices > 1:
                            if replicated:
                                # For replicated tensors (after all-gather), take from first device
                                # Using ConcatMeshToTensor on dim=0 and then slicing gives us one copy
                                pt_tensor = ttnn.to_torch(tensor, mesh_composer=ttnn.ConcatMeshToTensor(device, dim=0))
                                return pt_tensor[0:original_batch_size]  # Use original batch size for slicing
                            else:
                                # For sharded tensors, concatenate along last dim
                                return ttnn.to_torch(tensor, mesh_composer=ttnn.ConcatMeshToTensor(device, dim=-1))
            except Exception as e:
                import logging

                logging.warning(f"TTNNQwen3LinearAttention._to_pytorch: Error during mesh handling: {e}")
            return tensor.to_torch()
        elif isinstance(tensor, TorchTTNNTensor):
            # For TorchTTNNTensor, use the underlying ttnn_tensor if available
            if hasattr(tensor, "ttnn_tensor") and tensor.ttnn_tensor is not None:
                # Get batch size from TorchTTNNTensor shape
                if hasattr(tensor, "shape") and tensor.shape is not None and len(tensor.shape) > 0:
                    original_batch_size = tensor.shape[0]
                device = tensor.ttnn_tensor.device() if hasattr(tensor.ttnn_tensor, "device") else None
                if device is not None and hasattr(device, "get_num_devices") and device.get_num_devices() > 1:
                    if replicated:
                        pt_tensor = ttnn.to_torch(
                            tensor.ttnn_tensor, mesh_composer=ttnn.ConcatMeshToTensor(device, dim=0)
                        )
                        return pt_tensor[0:original_batch_size]  # Use original batch size for slicing
                    else:
                        return ttnn.to_torch(tensor.ttnn_tensor, mesh_composer=ttnn.ConcatMeshToTensor(device, dim=-1))
            return tensor.torch_tensor
        elif hasattr(tensor, "ttnn_tensor"):
            # Get batch size from wrapper object
            if hasattr(tensor, "shape") and tensor.shape is not None and len(tensor.shape) > 0:
                original_batch_size = tensor.shape[0]
            mesh_composer = None
            if hasattr(tensor.ttnn_tensor, "device") and tensor.ttnn_tensor.device() is not None:
                device = tensor.ttnn_tensor.device()
                if hasattr(device, "get_num_devices") and device.get_num_devices() > 1:
                    if replicated:
                        pt_tensor = ttnn.to_torch(
                            tensor.ttnn_tensor, mesh_composer=ttnn.ConcatMeshToTensor(device, dim=0)
                        )
                        return pt_tensor[0:original_batch_size]  # Use original batch size for slicing
                    else:
                        mesh_composer = ttnn.ConcatMeshToTensor(device, dim=-1)
            return ttnn.to_torch(tensor.ttnn_tensor, mesh_composer=mesh_composer)
        return tensor

    def _to_ttnn(self, tensor):
        """Convert PyTorch tensor to TTNN tensor on device."""
        if tensor is None:
            return None
        if self.device is None:
            return tensor
        mesh_mapper = ttnn.ReplicateTensorToMesh(self.device) if self.device.get_num_devices() > 1 else None
        return ttnn.from_torch(
            tensor,
            device=self.device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=mesh_mapper,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def _to_pytorch_replicated(self, tensor):
        """Convert a tensor that's known to be replicated (identical on all devices).

        After all_gather, all devices have identical data. This method properly
        handles the extraction without relying on auto-detection.
        """
        if tensor is None:
            return None
        if isinstance(tensor, ttnn.Tensor):
            device = tensor.device()
            if device is not None and device.get_num_devices() > 1:
                # After all_gather, all devices have identical data.
                # Use ConcatMeshToTensor with dim=0 and take first element (all devices have identical data)
                mesh_composer = ttnn.ConcatMeshToTensor(device, dim=0)
                pt_tensor = ttnn.to_torch(tensor, mesh_composer=mesh_composer)
                return pt_tensor[0].unsqueeze(0).contiguous()  # [0] removes batch, unsqueeze adds it back
            return ttnn.to_torch(tensor).contiguous()
        elif isinstance(tensor, TorchTTNNTensor):
            return self._to_pytorch_replicated(tensor.ttnn_tensor)
        elif hasattr(tensor, "ttnn_tensor") and tensor.ttnn_tensor is not None:
            return self._to_pytorch_replicated(tensor.ttnn_tensor)
        # Already PyTorch tensor
        if hasattr(tensor, "contiguous"):
            return tensor.contiguous()
        return tensor

    def forward(
        self,
        hidden_states: torch.Tensor,
        cache_params=None,
        cache_position: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Forward pass through Qwen3 linear attention with TTNN-accelerated projections.

        Uses TTNN for linear projections and PyTorch for DeltaNet kernel.

        Args:
            hidden_states: Input tensor [batch, seq_len, hidden_size]
            cache_params: Optional Qwen3_5MoeDynamicCache for recurrent state
            cache_position: Optional position tensor for decode
            attention_mask: Optional attention mask
            **kwargs: Additional arguments for forward compatibility

        Returns:
            Output tensor [batch, seq_len, hidden_size]
        """
        # DEBUG: Compare INPUT with CPU reference at the START of forward
        import os

        if os.environ.get("DEBUG_LINEAR_ATTN_INPUT", "0") == "1":
            # Get input as PyTorch
            if hasattr(hidden_states, "elem") and hidden_states.elem is not None:
                hs_input = hidden_states.elem
                input_source = "elem"
            elif hasattr(hidden_states, "ttnn_tensor") and hidden_states.ttnn_tensor is not None:
                hs_input = self._to_pytorch(hidden_states)
                input_source = "ttnn_tensor"
            else:
                hs_input = hidden_states
                input_source = "raw"

            # Check config
            config_info = "None"
            if hasattr(hidden_states, "ttnn_distributed_tensor_config"):
                cfg = hidden_states.ttnn_distributed_tensor_config
                if cfg is not None:
                    config_info = f"logical_shape_fn={cfg.logical_shape_fn is not None}, mapper={type(cfg.mesh_mapper).__name__ if cfg.mesh_mapper else 'None'}"

            print(f"[INPUT DEBUG] Layer {self.layer_idx}:")
            print(f"  source={input_source}, shape={hs_input.shape}, config={config_info}")
            print(f"  type={type(hs_input).__name__}")

        # If TTNN projections disabled or no device, use pure PyTorch fallback
        if not self.use_ttnn_projections or self.device is None:
            # Layer 0: input from embedding is REPLICATED (full hidden_size on each device)
            # Layers 1-39: input from previous MoE is COL-SHARDED (hidden_size/num_devices)
            # Auto-detect via _is_tensor_replicated() to handle both cases correctly
            hidden_states_pt = self._to_pytorch(hidden_states)  # Auto-detect via _is_tensor_replicated
            cache_position_pt = self._to_pytorch(cache_position)  # Auto-detect
            attention_mask_pt = self._to_pytorch(attention_mask)  # Auto-detect

            output = self._fallback_torch_layer(
                hidden_states_pt,
                cache_params=cache_params,
                cache_position=cache_position_pt,
                attention_mask=attention_mask_pt,
            )

            # When using pure PyTorch fallback, return raw PyTorch tensor
            # Don't wrap in TorchTTNNTensor to avoid distributed config issues
            # The output is a regular CPU tensor that will flow through subsequent layers
            return output

        # === HYBRID FORWARD: TTNN projections + PyTorch DeltaNet kernel ===

        # ALL-GATHER INPUT IF SHARDED: Ensure we have full hidden_size before projections
        # The projections expect replicated input [batch, seq, hidden_size]
        # If input is col-sharded (from previous MoE), all-gather it first
        if self._is_distributed and not self._is_tensor_replicated(hidden_states):
            hidden_states = self._maybe_all_gather(hidden_states)

        # Apply mask to hidden states (from PyTorch implementation)
        try:
            from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import apply_mask_to_padding_states

            # After all-gather, input is REPLICATED (full hidden_size on each device)
            # Use explicit replicated=True since we just all-gathered
            hidden_states_pt = self._to_pytorch(hidden_states, replicated=True)
            attention_mask_pt = self._to_pytorch(attention_mask)  # Auto-detect
            hidden_states_masked = apply_mask_to_padding_states(hidden_states_pt, attention_mask_pt)
            # Convert back to TTNN for projections
            hidden_states_ttnn = self._to_ttnn(hidden_states_masked)
        except ImportError:
            hidden_states_ttnn = hidden_states

        batch_size, seq_len, _ = hidden_states_ttnn.shape

        # Linear attention uses a different cache format than paged attention
        # Check if cache_params is compatible with linear attention (has_previous_state, conv_states, recurrent_states)
        is_linear_attn_cache = (
            cache_params is not None
            and hasattr(cache_params, "has_previous_state")
            and hasattr(cache_params, "conv_states")
            and hasattr(cache_params, "recurrent_states")
        )

        use_precomputed_states = (
            is_linear_attn_cache and cache_params.has_previous_state and seq_len == 1 and cache_position is not None
        )

        # Get cache states if available (only for linear attention-compatible caches)
        # Use .get() to safely access - during first forward pass, these may not exist yet
        conv_state = None
        recurrent_state = None
        if is_linear_attn_cache:
            conv_state = cache_params.conv_states.get(self.layer_idx)
            recurrent_state = cache_params.recurrent_states.get(self.layer_idx)

        # === TTNN Linear Projections ===
        # Ensure tile layout for TTNN operations
        if hidden_states_ttnn.layout != ttnn.TILE_LAYOUT:
            hidden_states_ttnn = ttnn.to_layout(
                hidden_states_ttnn, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG
            )

        # Project: in_proj_qkv -> [batch, seq, key_dim * 2 + value_dim]
        mixed_qkv_ttnn = self.in_proj_qkv(hidden_states_ttnn)

        # Project: in_proj_z -> [batch, seq, value_dim]
        z_ttnn = self.in_proj_z(hidden_states_ttnn)

        # All-gather for distributed mode (results are replicated - same data on all devices)
        mixed_qkv_ttnn = self._maybe_all_gather(mixed_qkv_ttnn)
        z_ttnn = self._maybe_all_gather(z_ttnn)

        # === Convert to PyTorch for DeltaNet kernel ===
        # Use explicit replicated conversion since all_gather makes them replicated
        # The _to_pytorch_replicated method properly handles multi-device extraction
        mixed_qkv = self._to_pytorch_replicated(mixed_qkv_ttnn)
        z = self._to_pytorch_replicated(z_ttnn)

        # DEBUG: Compare with PyTorch reference
        import os

        if os.environ.get("DEBUG_LINEAR_ATTN", "0") == "1":
            # Get hidden_states as PyTorch - auto-detect replicated vs sharded
            if (
                isinstance(hidden_states, ttnn.Tensor)
                or hasattr(hidden_states, "ttnn_tensor")
                or hasattr(hidden_states, "to_ttnn")
            ):
                hs_pt = self._to_pytorch(hidden_states)  # Auto-detect via _is_tensor_replicated
            else:
                hs_pt = hidden_states

            # Run PyTorch reference
            mixed_qkv_ref = self._fallback_torch_layer.in_proj_qkv(hs_pt)
            z_ref = self._fallback_torch_layer.in_proj_z(hs_pt)

            print(f"[DEBUG] hidden_states shape for reference: {hs_pt.shape}")
            print(f"[DEBUG] mixed_qkv shape: TTNN={mixed_qkv.shape}, PyTorch={mixed_qkv_ref.shape}")
            print(f"[DEBUG] z shape: TTNN={z.shape}, PyTorch={z_ref.shape}")
            print(f"[DEBUG] mixed_qkv TTNN sample: {mixed_qkv[0,0,:5].tolist()}")
            print(f"[DEBUG] mixed_qkv PyTorch sample: {mixed_qkv_ref[0,0,:5].tolist()}")

            # Max absolute difference
            diff = (mixed_qkv.float() - mixed_qkv_ref.float()).abs()
            print(f"[DEBUG] mixed_qkv max diff: {diff.max().item():.6f}, mean: {diff.mean().item():.6f}")

            z_diff = (z.float() - z_ref.float()).abs()
            print(f"[DEBUG] z max diff: {z_diff.max().item():.6f}, mean: {z_diff.mean().item():.6f}")

        # Get hidden_states as PyTorch tensor for small projections
        # in_proj_a and in_proj_b are PyTorch nn.Linear layers (not TTNN) to avoid
        # distributed weight replication issues - they have tiny output dims (num_v_heads=4)
        # hidden_states_ttnn was converted via _to_ttnn which replicates, so use replicated conversion
        hidden_states_pt = self._to_pytorch_replicated(hidden_states_ttnn)
        b = self.in_proj_b(hidden_states_pt).contiguous()  # PyTorch nn.Linear -> [batch, seq, num_v_heads]
        a = self.in_proj_a(hidden_states_pt).contiguous()  # PyTorch nn.Linear -> [batch, seq, num_v_heads]

        # DEBUG: Compare a and b projections with PyTorch reference
        if os.environ.get("DEBUG_LINEAR_ATTN", "0") == "1":
            # Get hs_pt for reference if not already available
            if "hs_pt" not in dir():
                if (
                    isinstance(hidden_states, ttnn.Tensor)
                    or hasattr(hidden_states, "ttnn_tensor")
                    or hasattr(hidden_states, "to_ttnn")
                ):
                    hs_pt = self._to_pytorch(hidden_states)  # Auto-detect via _is_tensor_replicated
                else:
                    hs_pt = hidden_states
            a_ref = self._fallback_torch_layer.in_proj_a(hs_pt)
            b_ref = self._fallback_torch_layer.in_proj_b(hs_pt)
            print(f"[DEBUG] a max diff: {(a.float() - a_ref.float()).abs().max().item():.6f}")
            print(f"[DEBUG] b max diff: {(b.float() - b_ref.float()).abs().max().item():.6f}")

        # Correct batch_size based on actual converted tensor shapes (not input shape which may be inflated)
        # The _to_pytorch with replicated=True extracts data for a single batch from all devices
        actual_batch_size = mixed_qkv.shape[0]
        if actual_batch_size != batch_size:
            batch_size = actual_batch_size

        # If tensors have 8x batch from mesh replication, take first slice
        if mixed_qkv.shape[0] > batch_size:
            mixed_qkv = mixed_qkv[:batch_size]
            z = z[:batch_size]
            b = b[:batch_size]
            a = a[:batch_size]

        # Reshape z for gated norm: [batch, seq, num_v_heads, head_v_dim]
        # Call contiguous() to ensure memory layout is correct for DeltaNet kernel
        z = z.reshape(batch_size, seq_len, -1, self.head_v_dim).contiguous()

        # Transpose mixed_qkv for conv1d: [batch, key_dim * 2 + value_dim, seq]
        # Call contiguous() after transpose to ensure memory layout is correct
        mixed_qkv = mixed_qkv.transpose(1, 2).contiguous()

        # === PyTorch DeltaNet Kernel ===
        if use_precomputed_states:
            # Decode path: use causal_conv1d_update
            mixed_qkv = self.causal_conv1d_update(
                mixed_qkv,
                conv_state,
                self.conv1d.weight.squeeze(1),
                self.conv1d.bias,
                self.activation,
            )
        else:
            # Prefill path: use causal_conv1d_fn or fallback
            if is_linear_attn_cache:
                conv_state = F.pad(mixed_qkv, (self.conv_kernel_size - mixed_qkv.shape[-1], 0))
                cache_params.conv_states[self.layer_idx] = conv_state

            if self.causal_conv1d_fn is not None:
                mixed_qkv = self.causal_conv1d_fn(
                    x=mixed_qkv,
                    weight=self.conv1d.weight.squeeze(1),
                    bias=self.conv1d.bias,
                    activation=self.activation,
                    seq_idx=None,
                )
            else:
                mixed_qkv = F.silu(self.conv1d(mixed_qkv)[:, :, :seq_len])

        # Transpose back: [batch, seq, key_dim * 2 + value_dim]
        mixed_qkv = mixed_qkv.transpose(1, 2).contiguous()

        # Split into Q, K, V
        query, key, value = torch.split(
            mixed_qkv,
            [self.key_dim, self.key_dim, self.value_dim],
            dim=-1,
        )

        # Reshape for multi-head attention - ensure contiguous memory for DeltaNet kernel
        query = query.reshape(batch_size, seq_len, -1, self.head_k_dim).contiguous()
        key = key.reshape(batch_size, seq_len, -1, self.head_k_dim).contiguous()
        value = value.reshape(batch_size, seq_len, -1, self.head_v_dim).contiguous()

        # Compute gating parameters
        beta = b.sigmoid()
        g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)

        # Repeat Q, K for value heads if needed
        if self.num_v_heads // self.num_k_heads > 1:
            query = query.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)
            key = key.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)

        # Run delta rule kernel
        if not use_precomputed_states:
            core_attn_out, last_recurrent_state = self.chunk_gated_delta_rule(
                query,
                key,
                value,
                g=g,
                beta=beta,
                initial_state=None,
                output_final_state=is_linear_attn_cache,
                use_qk_l2norm_in_kernel=True,
            )
        else:
            core_attn_out, last_recurrent_state = self.recurrent_gated_delta_rule(
                query,
                key,
                value,
                g=g,
                beta=beta,
                initial_state=recurrent_state,
                output_final_state=is_linear_attn_cache,
                use_qk_l2norm_in_kernel=True,
            )

        # Update cache (only for linear attention-compatible caches)
        if is_linear_attn_cache:
            cache_params.recurrent_states[self.layer_idx] = last_recurrent_state

        # DEBUG: Print core_attn_out info after DeltaNet kernel
        if os.environ.get("DEBUG_LINEAR_ATTN", "0") == "1":
            print(f"[DEBUG] core_attn_out shape: {core_attn_out.shape}, sample: {core_attn_out[0,0,:3].tolist()}")

        # Apply gated RMS norm
        core_attn_out = core_attn_out.reshape(-1, self.head_v_dim)
        z = z.reshape(-1, self.head_v_dim)
        core_attn_out = self.norm(core_attn_out, z)
        core_attn_out = core_attn_out.reshape(batch_size, seq_len, -1)

        # === TTNN Output Projection ===
        # Convert back to TTNN for output projection
        core_attn_out_ttnn = self._to_ttnn(core_attn_out)
        if core_attn_out_ttnn.layout != ttnn.TILE_LAYOUT:
            core_attn_out_ttnn = ttnn.to_layout(
                core_attn_out_ttnn, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG
            )

        output_ttnn = self.out_proj(core_attn_out_ttnn)

        # DEBUG: Compare out_proj with PyTorch reference
        if os.environ.get("DEBUG_LINEAR_ATTN", "0") == "1":
            out_ref = self._fallback_torch_layer.out_proj(core_attn_out)
            out_ttnn_pt = self._to_pytorch(output_ttnn, replicated=False)  # col-sharded output
            print(f"[DEBUG] out_proj max diff: {(out_ttnn_pt.float() - out_ref.float()).abs().max().item():.6f}")

        # NOTE: Do NOT all_gather here. Output should stay col-sharded to match:
        # 1. Full attention output (also col-sharded)
        # 2. MoE input expectation (does all_gather internally)

        # DEBUG: Compare FULL LAYER OUTPUT with CPU fallback
        if os.environ.get("DEBUG_LINEAR_ATTN_OUTPUT", "0") == "1":
            # Get the input as PyTorch tensor
            hs_input = self._to_pytorch(hidden_states)

            # Run CPU fallback for comparison - may return 1 or 2 values
            cpu_result = self._fallback_torch_layer(
                hs_input,
                cache_params=None,
                cache_position=cache_position.cpu() if cache_position is not None else None,
                attention_mask=self._to_pytorch(attention_mask) if attention_mask is not None else None,
            )
            cpu_output = cpu_result[0] if isinstance(cpu_result, tuple) else cpu_result

            # Get TTNN output as PyTorch
            ttnn_output = self._to_pytorch(output_ttnn, replicated=False)  # col-sharded

            # Compare
            diff = (ttnn_output.float() - cpu_output.float()).abs()
            print(f"[LAYER OUTPUT DEBUG] Layer {self.layer_idx}:")
            print(f"  TTNN shape: {ttnn_output.shape}, CPU shape: {cpu_output.shape}")
            print(f"  Max diff: {diff.max().item():.6f}, Mean diff: {diff.mean().item():.6f}")
            if diff.max().item() > 1.0:
                print(f"  TTNN sample: {ttnn_output[0,0,:5].tolist()}")
                print(f"  CPU sample: {cpu_output[0,0,:5].tolist()}")

        # out_proj returns TorchTTNNTensor already, return it directly
        return output_ttnn
