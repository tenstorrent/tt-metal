# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Gemma4 Attention implementations for TTNN.

This module provides TTNN-accelerated attention mechanisms specific to Gemma4:
- TTNNGemma4Attention: Unified attention class for both sliding and global layers
- TTNNGemma4PagedAttentionKVCache: Dual paged KV cache routing sliding/global layers

Gemma4 attention has two variants:
- Sliding layers: 32 Q heads, 16 KV heads, head_dim=256, separate K/V projections
- Global layers: 32 Q heads, 4 KV heads, head_dim=512, K=V sharing (v_proj is None)

Both variants use per-head Q/K/V RMSNorm (no 1/sqrt(d) scaling) and RoPE.
"""

from typing import Optional

import torch

import ttnn
from models.experimental.tt_symbiote.core.module import TTNNModule
from models.experimental.tt_symbiote.core.tensor import TorchTTNNTensor
from models.experimental.tt_symbiote.modules.attention import (
    TTNNPagedAttentionKVCache,
    PagedAttentionConfig,
    TTNNSDPAAttention,
)
from models.experimental.tt_symbiote.modules.linear import (
    TTNNLinear,
    TTNNLinearIReplicatedWColSharded,
    TTNNLinearIColShardedWAllReduced,
)
from models.experimental.tt_symbiote.modules.normalization import TTNNLocalRMSNorm
from models.experimental.tt_symbiote.modules.rope import (
    BailingRotarySetup,
)
from models.experimental.tt_symbiote.core.run_config import trace_enabled

try:
    from transformers.cache_utils import Cache
except ImportError:
    Cache = object


def _reverse_permute_weight(tensor: torch.Tensor, n_heads: int, head_dim: int) -> torch.Tensor:
    """Permute Q/K weight from HF split-half to Meta interleaved layout.

    Full rotary: standard permutation matching tt-transformers reverse_permute.
    Reorders from [first_half, second_half] to interleaved [x0, x_{d/2}, x1, x_{d/2+1}, ...].
    """
    dim1, dim2 = tensor.shape
    return tensor.view(n_heads, 2, dim1 // n_heads // 2, dim2).transpose(1, 2).reshape(dim1, dim2)


def _reverse_permute_norm_weight(weight: torch.Tensor, head_dim: int) -> torch.Tensor:
    """Permute per-head norm weight from split-half to interleaved layout.

    When Q/K projection weights are permuted via _reverse_permute_weight, the
    per-head norm weights (q_norm, k_norm) must be permuted identically so that
    the element-wise scaling aligns with the reordered head dimensions.

    Transforms [w0, w1, ..., w_{d/2-1}, w_{d/2}, ...] to [w0, w_{d/2}, w1, w_{d/2+1}, ...].
    """
    return weight.view(2, head_dim // 2).T.reshape(-1)


def _deinterleave_heads(tensor, head_dim):
    """Convert from interleaved [a0,b0,a1,b1,...] to split-half [a0,a1,...,b0,b1,...] format.

    Applied per-head on the last dimension. Used to un-permute V states that
    inherited the k_proj weight permutation in K=V sharing layers.
    """
    shape = list(tensor.shape)
    new_shape = shape[:-1] + [head_dim // 2, 2]
    tensor = ttnn.reshape(tensor, new_shape)
    ndim = len(new_shape)
    perm = list(range(ndim - 2)) + [ndim - 1, ndim - 2]
    tensor = ttnn.permute(tensor, perm)
    tensor = ttnn.reshape(tensor, shape)
    return tensor


@trace_enabled
class TTNNGemma4Attention(TTNNModule):
    """TTNN-accelerated Attention for Gemma4.

    Handles BOTH sliding and global attention layers. The from_torch method
    inspects hf_attn.is_sliding to determine the variant.

    Sliding layers:
    - 32 Q heads, 16 KV heads, head_dim=256
    - Separate q_proj, k_proj, v_proj
    - Q-norm, K-norm, V-norm (per-head RMSNorm)

    Global layers:
    - 32 Q heads, 4 KV heads, head_dim=512
    - q_proj and k_proj only; v_proj is None (K=V sharing)
    - After k_proj, the shared output gets k_norm -> K and v_norm -> V

    Both use scaling=1.0 (norms replace 1/sqrt(d) scaling) and full RoPE.
    """

    # Class-level cache for BailingRotarySetup instances, shared across layers
    # with the same (device_id, head_dim, rope_theta) to avoid OOM from
    # per-layer cos/sin cache allocation (~16-32MB each × 60 layers = ~1.1GB).
    _shared_rotary_setups = {}

    def __init__(self):
        super().__init__()
        self.num_attention_heads = None
        self.num_key_value_heads = None
        self.num_key_value_groups = None
        self.head_dim = None
        self.hidden_size = None
        self.scaling = None
        self.is_causal = True
        self.is_sliding = None
        self.layer_idx = None
        self.sliding_window = None

        self.q_proj = None
        self.k_proj = None
        self.v_proj = None  # None for global layers (K=V sharing)
        self.o_proj = None
        self.q_norm = None
        self.k_norm = None
        self.v_norm = None
        self.rope = None
        self.sdpa = None
        self.core_grid = None

    @classmethod
    def from_torch(cls, hf_attn, distributed: bool = True):
        """Create TTNNGemma4Attention from HuggingFace Gemma4TextAttention.

        Args:
            hf_attn: HuggingFace Gemma4TextAttention module
            distributed: Whether to use distributed linear layers (default True for T3K)

        Returns:
            TTNNGemma4Attention instance
        """
        new_attn = cls()
        new_attn._fallback_torch_layer = hf_attn

        # Extract configuration
        config = hf_attn.config
        new_attn.is_sliding = hf_attn.is_sliding
        new_attn.layer_idx = hf_attn.layer_idx
        new_attn.num_attention_heads = config.num_attention_heads  # 32
        # num_key_value_heads is a local variable in HF's __init__, not stored as self attr.
        # Replicate the HF logic: use num_global_key_value_heads for global (k=v) layers,
        # otherwise use config.num_key_value_heads.
        use_alternative_attention = getattr(config, "attention_k_eq_v", False) and not new_attn.is_sliding
        num_kv_heads = config.num_global_key_value_heads if use_alternative_attention else config.num_key_value_heads
        new_attn.num_key_value_heads = num_kv_heads
        new_attn.num_key_value_groups = new_attn.num_attention_heads // new_attn.num_key_value_heads
        new_attn.head_dim = hf_attn.head_dim  # 256 sliding, 512 global
        new_attn.hidden_size = config.hidden_size

        # Gemma4 uses per-head norms instead of 1/sqrt(d) scaling.
        # Read scaling from the HF module if available, otherwise default to 1.0.
        new_attn.scaling = getattr(hf_attn, "scaling", 1.0)

        # Store sliding window size for decode mask construction.
        # Sliding layers enforce a window; global layers use None (full context).
        new_attn.sliding_window = getattr(hf_attn, "sliding_window", None)

        # Choose linear layer class for output projection
        LinearClsOut = TTNNLinearIReplicatedWColSharded if distributed else TTNNLinear

        # Permute Q/K weights from HF split-half to Meta interleaved layout
        # before creating TTNN linear layers. This is required for
        # ttnn.experimental.rotary_embedding_llama which expects interleaved
        # cos/sin pairs [c0, c0, c1, c1, ...].
        q_weight = _reverse_permute_weight(
            hf_attn.q_proj.weight.data.clone(), new_attn.num_attention_heads, new_attn.head_dim
        )
        k_weight = _reverse_permute_weight(
            hf_attn.k_proj.weight.data.clone(), new_attn.num_key_value_heads, new_attn.head_dim
        )

        # Permute per-head Q/K norm weights to match the interleaved layout.
        # The norm weight is [head_dim] and applied element-wise after projection,
        # so it must follow the same split-half -> interleaved reordering.
        if hasattr(hf_attn.q_norm, "weight") and hf_attn.q_norm.weight is not None:
            hf_attn.q_norm.weight.data = _reverse_permute_norm_weight(hf_attn.q_norm.weight.data, new_attn.head_dim)
        if hasattr(hf_attn.k_norm, "weight") and hf_attn.k_norm.weight is not None:
            hf_attn.k_norm.weight.data = _reverse_permute_norm_weight(hf_attn.k_norm.weight.data, new_attn.head_dim)

        # Fused QKV projection: concatenate Q, K, V weights into single matmul.
        # Sliding layers: Q + K + V; Global layers: Q + K only (K=V sharing).
        # Uses TTNNLinearIColShardedWAllReduced: input col-sharded -> matmul -> all_reduce
        # replaces 3 separate matmuls + 4 all_gather CCL ops with 1 matmul + 2 CCL ops.
        q_size = new_attn.num_attention_heads * new_attn.head_dim
        kv_size = new_attn.num_key_value_heads * new_attn.head_dim

        has_v_proj = hf_attn.v_proj is not None
        new_attn._has_v_proj = has_v_proj

        if has_v_proj:
            v_weight = hf_attn.v_proj.weight.data.clone()
            fused_weight = torch.cat([q_weight, k_weight, v_weight], dim=0)
        else:
            fused_weight = torch.cat([q_weight, k_weight], dim=0)

        fused_linear = torch.nn.Linear(new_attn.hidden_size, fused_weight.shape[0], bias=False)
        fused_linear.weight.data = fused_weight
        new_attn.qkv_proj = TTNNLinearIColShardedWAllReduced.from_torch(fused_linear)
        new_attn._q_size = q_size
        new_attn._kv_size = kv_size

        # Keep individual proj references as None (fused into qkv_proj)
        new_attn.q_proj = None
        new_attn.k_proj = None
        new_attn.v_proj = None

        # O projection: row-parallel with reduce_scatter
        new_attn.o_proj = LinearClsOut.from_torch(hf_attn.o_proj)

        # Per-head Q/K/V norms (Gemma4RMSNorm instances, weights already permuted above)
        new_attn.q_norm = TTNNLocalRMSNorm.from_torch(hf_attn.q_norm)
        new_attn.k_norm = TTNNLocalRMSNorm.from_torch(hf_attn.k_norm)
        if hasattr(hf_attn, "v_norm") and hf_attn.v_norm is not None:
            # Gemma4RMSNorm(with_scale=False) doesn't store dim; stash it for _infer_dim
            if not hasattr(hf_attn.v_norm, "weight") or hf_attn.v_norm.weight is None:
                hf_attn.v_norm._norm_dim = new_attn.head_dim
            new_attn.v_norm = TTNNLocalRMSNorm.from_torch(hf_attn.v_norm)
        else:
            new_attn.v_norm = None

        # RoPE is handled by BailingRotarySetup (initialized in move_weights_to_device_impl)

        # SDPA for attention computation
        new_attn.sdpa = TTNNSDPAAttention()

        # Default core grid (will be updated in move_weights_to_device_impl)
        new_attn.core_grid = ttnn.CoreGrid(y=8, x=8)

        return new_attn

    def move_weights_to_device_impl(self):
        """Initialize SDPA config and move weights to device."""
        super().move_weights_to_device_impl()

        # Query grid dynamically from device
        grid = self.device.compute_with_storage_grid_size()
        self.core_grid = ttnn.CoreGrid(y=grid.y, x=grid.x)

        if self.sdpa.program_config is None:
            # Configure SDPA for the appropriate head_dim
            # Use smaller chunk sizes for larger head_dim to avoid L1 pressure
            chunk_size = 64 if self.head_dim >= 512 else 128
            self.sdpa.program_config = ttnn.SDPAProgramConfig(
                compute_with_storage_grid_size=(self.core_grid.x, self.core_grid.y),
                q_chunk_size=chunk_size,
                k_chunk_size=chunk_size,
                exp_approx_mode=False,
            )
            self.sdpa.compute_kernel_config = ttnn.init_device_compute_kernel_config(
                self.device.arch(),
                math_fidelity=ttnn.MathFidelity.HiFi4,
                math_approx_mode=False,
                fp32_dest_acc_en=False,
                packer_l1_acc=False,
            )

        # Pre-allocate replicated decode cur_pos buffer for trace safety.
        if self.device.get_num_devices() > 1:
            mesh_mapper = ttnn.ReplicateTensorToMesh(self.device)
            self._decode_cur_pos = ttnn.from_torch(
                torch.zeros(1, dtype=torch.int32),
                device=self.device,
                dtype=ttnn.int32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                mesh_mapper=mesh_mapper,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

        # Initialize BailingRotarySetup for on-device RoPE (trace-safe).
        # Pre-computes cos/sin caches and transformation matrices on device,
        # avoiding frozen position_embeddings during trace replay.
        # SHARED across layers with the same config to avoid OOM — only 2 unique
        # configs exist (sliding head_dim=256, global head_dim=512).
        config = self._fallback_torch_layer.config
        layer_type = "sliding_attention" if self.is_sliding else "full_attention"
        rope_params = config.rope_parameters[layer_type]
        rope_theta = rope_params["rope_theta"]  # 10_000 sliding, 1_000_000 global
        partial_rotary_factor = rope_params.get("partial_rotary_factor", 1.0)  # 1.0 sliding, 0.25 global
        setup_key = (id(self.device), self.head_dim, rope_theta, partial_rotary_factor)
        if setup_key not in TTNNGemma4Attention._shared_rotary_setups:
            # Gemma4 follows the HuggingFace Gemma3 convention: inv_freq is
            # computed over the full head_dim and only the first rotary_dim
            # frequencies are used.  This differs from the standard Phi/Ling
            # convention where inv_freq is computed over rotary_dim only.
            TTNNGemma4Attention._shared_rotary_setups[setup_key] = BailingRotarySetup(
                device=self.device,
                head_dim=self.head_dim,
                max_seq_len=min(getattr(config, "max_position_embeddings", 8192), 2048),
                rope_theta=rope_theta,
                partial_rotary_factor=partial_rotary_factor,
                use_head_dim_for_freq=True,
            )
        self._rotary_setup = TTNNGemma4Attention._shared_rotary_setups[setup_key]
        self._rotary_dim = self._rotary_setup.rotary_dim  # 256 sliding, 128 global

    @property
    def _is_distributed(self):
        return (
            self.device_state is not None
            and hasattr(self.device_state, "ccl_manager")
            and self.device_state.ccl_manager is not None
        )

    def _maybe_all_gather(self, tensor):
        """All-gather tensor across mesh devices if in distributed mode.

        Uses Ring topology for trace compatibility — Linear topology may
        allocate dynamic intermediate buffers that are not pinned by trace
        capture, causing address aliasing on replay.
        """
        if not self._is_distributed:
            return tensor
        gathered = ttnn.all_gather(
            tensor,
            dim=-1,
            num_links=1,
            cluster_axis=1,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            topology=ttnn.Topology.Ring,
        )
        return gathered

    def _is_tensor_replicated(self, tensor) -> bool:
        """Check if tensor is replicated across devices (vs sharded)."""
        if tensor is None:
            return True

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
                    return True
                elif last_dim == self.hidden_size / num_devices:
                    return False

        return False

    def _repeat_kv(self, hidden_states: ttnn.Tensor, n_rep: int) -> ttnn.Tensor:
        """Repeat KV heads to match Q heads for GQA.

        Uses repeat_interleave for correct GQA head ordering.
        """
        if n_rep == 1:
            return hidden_states
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

    def _apply_per_head_norm(self, states, norm_module, batch_size, seq_length, num_heads, head_dim):
        """Apply per-head RMSNorm by flattening to [batch*seq*heads, head_dim].

        Args:
            states: Tensor of shape [batch, seq, num_heads, head_dim]
            norm_module: TTNNLocalRMSNorm instance
            batch_size: Batch size
            seq_length: Sequence length
            num_heads: Number of heads
            head_dim: Head dimension

        Returns:
            Normalized tensor of shape [batch, seq, num_heads, head_dim]
        """
        orig_shape = states.shape
        # Flatten to 2D for rms_norm: [batch * seq * num_heads, head_dim]
        states = ttnn.reshape(states, (batch_size * seq_length * num_heads, head_dim))
        states = norm_module(states)
        states = ttnn.reshape(states, orig_shape)
        return states

    def _project_qkv(self, hidden_states, batch_size, seq_length, apply_rope=False, for_decode=False):
        """Project hidden states to Q, K, V via fused QKV matmul and apply per-head norms.

        Uses a single fused QKV projection (TTNNLinearIColShardedWAllReduced)
        instead of 3 separate projections + all_gathers. Input arrives col-sharded
        from the norm; the fused linear does matmul + reduce_scatter + all_gather
        in 1 matmul + 2 CCL ops (vs 3 matmuls + 4 CCL ops before).

        For decode (S=1, for_decode=True), uses an optimized reshape path that
        goes directly from [B, 1, proj_size] to [num_heads, head_dim] for norm,
        skipping the intermediate [B, S, H, D] reshape. This saves 2 reshape ops
        per projection (6 total for Q/K/V).

        Args:
            hidden_states: Input tensor [batch, seq_len, hidden_size]
            batch_size: Batch dimension
            seq_length: Sequence length
            apply_rope: Unused, kept for signature compatibility (always False)
            for_decode: If True, skip the permute to [B,H,S,D] and return
                [B,S,H,D] directly (avoids 3 trace ops when S=1).

        Returns:
            Tuple of (query_states, key_states, value_states) in
            [batch, num_heads, seq_len, head_dim] layout (or [B,S,H,D] if for_decode).
        """
        if hidden_states.layout != ttnn.TILE_LAYOUT:
            hidden_states = ttnn.to_layout(hidden_states, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        # Fused QKV: single matmul + all_reduce produces replicated output
        qkv_states = self.qkv_proj(hidden_states)

        # Slice into Q, K, V components
        q_size = self._q_size
        kv_size = self._kv_size
        query_states = ttnn.slice(qkv_states, [0, 0, 0], [batch_size, seq_length, q_size])
        key_states = ttnn.slice(qkv_states, [0, 0, q_size], [batch_size, seq_length, q_size + kv_size])
        if self._has_v_proj:
            value_states = ttnn.slice(
                qkv_states, [0, 0, q_size + kv_size], [batch_size, seq_length, q_size + 2 * kv_size]
            )
        else:
            # Global layers (K=V sharing): clone K output for V path
            value_states = ttnn.clone(key_states, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(qkv_states)

        if for_decode and seq_length == 1:
            # Optimized decode path: skip intermediate [B,S,H,D] reshape.
            # Go directly from [B, 1, proj_size] to [num_heads, head_dim] for norm,
            # then reshape to [B, 1, H, D]. Saves 2 reshapes per projection.
            query_states = ttnn.reshape(query_states, (self.num_attention_heads, self.head_dim))
            key_states = ttnn.reshape(key_states, (self.num_key_value_heads, self.head_dim))
            value_states = ttnn.reshape(value_states, (self.num_key_value_heads, self.head_dim))

            if not self._has_v_proj:
                value_states = _deinterleave_heads(value_states, self.head_dim)

            query_states = self.q_norm(query_states)
            key_states = self.k_norm(key_states)
            if self.v_norm is not None:
                value_states = self.v_norm(value_states)

            query_states = ttnn.reshape(query_states, (batch_size, seq_length, self.num_attention_heads, self.head_dim))
            key_states = ttnn.reshape(key_states, (batch_size, seq_length, self.num_key_value_heads, self.head_dim))
            value_states = ttnn.reshape(value_states, (batch_size, seq_length, self.num_key_value_heads, self.head_dim))
        else:
            # Prefill path: standard reshape chain
            query_states = ttnn.reshape(query_states, (batch_size, seq_length, self.num_attention_heads, self.head_dim))
            key_states = ttnn.reshape(key_states, (batch_size, seq_length, self.num_key_value_heads, self.head_dim))
            value_states = ttnn.reshape(value_states, (batch_size, seq_length, self.num_key_value_heads, self.head_dim))

            if not self._has_v_proj:
                value_states = _deinterleave_heads(value_states, self.head_dim)

            query_states = self._apply_per_head_norm(
                query_states,
                self.q_norm,
                batch_size,
                seq_length,
                self.num_attention_heads,
                self.head_dim,
            )
            key_states = self._apply_per_head_norm(
                key_states,
                self.k_norm,
                batch_size,
                seq_length,
                self.num_key_value_heads,
                self.head_dim,
            )
            if self.v_norm is not None:
                value_states = self._apply_per_head_norm(
                    value_states,
                    self.v_norm,
                    batch_size,
                    seq_length,
                    self.num_key_value_heads,
                    self.head_dim,
                )

            # Permute to [batch, num_heads, seq_len, head_dim] for prefill attention
            query_states = ttnn.permute(query_states, (0, 2, 1, 3))
            key_states = ttnn.permute(key_states, (0, 2, 1, 3))
            value_states = ttnn.permute(value_states, (0, 2, 1, 3))

        return query_states, key_states, value_states

    def _apply_rotary_embedding_llama(
        self, query_states, key_states, cos, sin, trans_mat, is_decode_mode, batch_size=None
    ):
        """Apply rotary_embedding_llama with partial-RoPE optimization.

        The TTNN rotary_embedding_llama kernel requires head_dim <= 256.
        Three paths:
        1. head_dim <= 256: Direct single kernel call (sliding layers).
        2. head_dim > 256 AND rotary_dim <= 256: Partial RoPE — slice only the
           rotary dims (128 for Gemma 4 global), apply one kernel call, concat
           with untouched pass-through dims. 1.57x decode speedup.
        3. head_dim > 256 AND rotary_dim > 256: Chunked fallback — split into
           256-dim chunks and apply RoPE to each (future-proofing).

        In decode mode, the kernel requires HEIGHT_SHARDED inputs, so tensors
        are sharded before RoPE and un-sharded after.
        """
        max_rope_dim = 256

        # --- Path 1: head_dim fits kernel limit (sliding layers, head_dim=256) ---
        if self.head_dim <= max_rope_dim:
            query_states = ttnn.experimental.rotary_embedding_llama(
                query_states, cos, sin, trans_mat, is_decode_mode=is_decode_mode
            )
            key_states = ttnn.experimental.rotary_embedding_llama(
                key_states, cos, sin, trans_mat, is_decode_mode=is_decode_mode
            )
            return query_states, key_states

        # --- Path 2: Partial RoPE (global layers, rotary_dim=128 <= 256) ---
        rotary_dim = self._rotary_dim
        if rotary_dim <= max_rope_dim:
            # Slice rotary portion and pass-through portion
            q_rot = query_states[:, :, :, :rotary_dim]
            q_pass = query_states[:, :, :, rotary_dim:]
            k_rot = key_states[:, :, :, :rotary_dim]
            k_pass = key_states[:, :, :, rotary_dim:]

            # Slice cos/sin to actual rotary dims only
            cos_rot = cos[:, :, :, :rotary_dim]
            sin_rot = sin[:, :, :, :rotary_dim]

            # For decode mode, shard rotary-dim tensors to L1 (kernel requirement)
            if is_decode_mode and batch_size is not None:
                batch_grid = ttnn.num_cores_to_corerangeset(
                    batch_size, self.device.compute_with_storage_grid_size(), True
                )
                shard_mem = ttnn.create_sharded_memory_config(
                    shape=(ttnn.TILE_SIZE, rotary_dim),
                    core_grid=batch_grid,
                    strategy=ttnn.ShardStrategy.HEIGHT,
                    orientation=ttnn.ShardOrientation.ROW_MAJOR,
                    use_height_and_width_as_shard_shape=True,
                )
                q_rot = ttnn.to_memory_config(q_rot, shard_mem)
                k_rot = ttnn.to_memory_config(k_rot, shard_mem)
                cos_rot = ttnn.to_memory_config(cos_rot, shard_mem)
                sin_rot = ttnn.to_memory_config(sin_rot, shard_mem)

            # Single RoPE kernel call per Q/K (128 dims fits kernel limit)
            q_rot = ttnn.experimental.rotary_embedding_llama(
                q_rot, cos_rot, sin_rot, trans_mat, is_decode_mode=is_decode_mode
            )
            k_rot = ttnn.experimental.rotary_embedding_llama(
                k_rot, cos_rot, sin_rot, trans_mat, is_decode_mode=is_decode_mode
            )

            # Unshard back to DRAM after RoPE
            if is_decode_mode and batch_size is not None:
                q_rot = ttnn.to_memory_config(q_rot, ttnn.DRAM_MEMORY_CONFIG)
                k_rot = ttnn.to_memory_config(k_rot, ttnn.DRAM_MEMORY_CONFIG)

            # Concat rotated portion with untouched pass-through
            query_states = ttnn.concat([q_rot, q_pass], dim=-1)
            key_states = ttnn.concat([k_rot, k_pass], dim=-1)
            return query_states, key_states

        # --- Path 3: Chunked fallback for rotary_dim > 256 (future-proofing) ---
        num_chunks = (self.head_dim + max_rope_dim - 1) // max_rope_dim
        q_chunks = []
        k_chunks = []

        shard_mem = None
        if is_decode_mode and batch_size is not None:
            batch_grid = ttnn.num_cores_to_corerangeset(batch_size, self.device.compute_with_storage_grid_size(), True)
            shard_mem = ttnn.create_sharded_memory_config(
                shape=(ttnn.TILE_SIZE, max_rope_dim),
                core_grid=batch_grid,
                strategy=ttnn.ShardStrategy.HEIGHT,
                orientation=ttnn.ShardOrientation.ROW_MAJOR,
                use_height_and_width_as_shard_shape=True,
            )

        for i in range(num_chunks):
            start = i * max_rope_dim
            end = min((i + 1) * max_rope_dim, self.head_dim)
            q_chunk = query_states[:, :, :, start:end]
            k_chunk = key_states[:, :, :, start:end]
            cos_chunk = cos[:, :, :, start:end]
            sin_chunk = sin[:, :, :, start:end]

            if shard_mem is not None:
                q_chunk = ttnn.to_memory_config(q_chunk, shard_mem)
                k_chunk = ttnn.to_memory_config(k_chunk, shard_mem)
                cos_chunk = ttnn.to_memory_config(cos_chunk, shard_mem)
                sin_chunk = ttnn.to_memory_config(sin_chunk, shard_mem)

            q_chunk = ttnn.experimental.rotary_embedding_llama(
                q_chunk, cos_chunk, sin_chunk, trans_mat, is_decode_mode=is_decode_mode
            )
            k_chunk = ttnn.experimental.rotary_embedding_llama(
                k_chunk, cos_chunk, sin_chunk, trans_mat, is_decode_mode=is_decode_mode
            )

            if shard_mem is not None:
                q_chunk = ttnn.to_memory_config(q_chunk, ttnn.DRAM_MEMORY_CONFIG)
                k_chunk = ttnn.to_memory_config(k_chunk, ttnn.DRAM_MEMORY_CONFIG)

            q_chunks.append(q_chunk)
            k_chunks.append(k_chunk)

        query_states = ttnn.concat(q_chunks, dim=-1)
        key_states = ttnn.concat(k_chunks, dim=-1)
        return query_states, key_states

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

        # Project Q/K/V with per-head norms (no RoPE yet)
        query_states, key_states, value_states = self._project_qkv(hidden_states, batch_size, seq_length)

        # Apply on-device RoPE via BailingRotarySetup (trace-safe)
        seq_len = query_states.shape[2]
        cos, sin = self._rotary_setup.get_cos_sin_for_prefill(seq_len)
        trans_mat = self._rotary_setup.get_trans_mat(is_decode=False)

        if isinstance(query_states, ttnn.Tensor) and query_states.dtype != ttnn.bfloat16:
            query_states = ttnn.typecast(query_states, ttnn.bfloat16)
        if isinstance(key_states, ttnn.Tensor) and key_states.dtype != ttnn.bfloat16:
            key_states = ttnn.typecast(key_states, ttnn.bfloat16)

        query_states, key_states = self._apply_rotary_embedding_llama(
            query_states, key_states, cos, sin, trans_mat, is_decode_mode=False
        )

        # Expand KV to match Q heads (GQA)
        key_states = self._repeat_kv(key_states, self.num_key_value_groups)
        value_states = self._repeat_kv(value_states, self.num_key_value_groups)

        use_paged = isinstance(past_key_values, TTNNGemma4PagedAttentionKVCache) or isinstance(
            past_key_values, TTNNPagedAttentionKVCache
        )

        if past_key_values is not None:
            cache_layer_idx = self.layer_idx

            if use_paged:
                # Get unexpanded KV for cache storage
                kv_key = key_states[:, :: self.num_key_value_groups, :, :]
                kv_value = value_states[:, :: self.num_key_value_groups, :, :]

                past_key_values.paged_fill_on_device(
                    kv_key,
                    kv_value,
                    layer_idx=cache_layer_idx,
                    batch_idx=0,
                )
            else:
                # Standard cache path (non-paged)
                cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
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
                    self.layer_idx,
                    cache_kwargs,
                )
                cached_key, cached_value = [TorchTTNNTensor(cached_key), TorchTTNNTensor(cached_value)]
                cached_key = ttnn.to_device(cached_key.to_ttnn, self.device)
                cached_value = ttnn.to_device(cached_value.to_ttnn, self.device)
                cached_key = self._maybe_all_gather(cached_key)
                cached_value = self._maybe_all_gather(cached_value)

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

        # Reshape output: [batch, seq_len, num_heads, head_dim] -> [batch, seq_len, hidden_dim]
        attn_shape = list(attn_output.shape)
        attn_batch = attn_shape[0]
        attn_seq = attn_shape[1]
        attn_output = ttnn.reshape(attn_output, (attn_batch, attn_seq, self.num_attention_heads * self.head_dim))

        # Output projection
        attn_output = self.o_proj(attn_output)

        return attn_output, None

    def _get_cur_pos_device_tensor(self, cache_position, past_key_values, layer_idx, batch_size):
        """Extract cur_pos as an on-device int32 tensor.

        Like Ling's pattern: the model wrapper (TTNNGemma4TextModel.call)
        always provides cache_position as an on-device ttnn.Tensor(int32).
        TracedRun manages the trace buffer; we only copy into
        _decode_cur_pos for a stable address on multi-device.
        """
        cp = cache_position
        if isinstance(cp, TorchTTNNTensor):
            cp = cp.ttnn_tensor

        # Flatten and slice to batch_size
        if len(cp.shape) > 1:
            total_elems = 1
            for d in cp.shape:
                total_elems *= d
            cp = ttnn.reshape(cp, (total_elems,))
        if cp.shape[0] > batch_size:
            cp = ttnn.slice(cp, [0], [batch_size])

        # On multi-device, copy into pre-allocated buffer for trace-stable address
        if self._decode_cur_pos is not None:
            ttnn.copy(cp, self._decode_cur_pos)
            return self._decode_cur_pos
        return cp

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
        [batch, heads, seq, head_dim] (B H S D).
        """
        batch_size, seq_length = hidden_states.shape[0], hidden_states.shape[1]

        layer_idx = self.layer_idx

        # Always resolve position from cache_position kwarg. This ensures
        # _decode_cur_pos is updated during ALL phases (warmup, capture, replay).
        # During replay, pre_trace_execute pre-copies the correct value
        # to the trace kwarg buffer before execute_trace, so the baked-in copy
        # (from kwarg buffer to _decode_cur_pos) uses the correct position.
        cur_pos_tt = self._get_cur_pos_device_tensor(cache_position, past_key_values, layer_idx, batch_size)

        # Project Q/K/V with per-head norms (no RoPE yet).
        # for_decode=True skips the [B,H,S,D] permute — returns [B,S,H,D].
        query_states, key_states, value_states = self._project_qkv(
            hidden_states, batch_size, seq_length, for_decode=True
        )

        # Reshape Q/K/V from [B, S, H, D] to [1, B, H, D] for decode RoPE.
        # With for_decode=True and S=1: [1,1,H,D] -> [1,1,H,D] (same values, no permute needed).
        query_states = ttnn.reshape(query_states, (1, batch_size, self.num_attention_heads, self.head_dim))
        key_states = ttnn.reshape(key_states, (1, batch_size, self.num_key_value_heads, self.head_dim))
        value_states = ttnn.reshape(value_states, (1, batch_size, self.num_key_value_heads, self.head_dim))

        # Typecast to bfloat16 for rotary_embedding_llama
        if isinstance(query_states, ttnn.Tensor) and query_states.dtype != ttnn.bfloat16:
            query_states = ttnn.typecast(query_states, ttnn.bfloat16)
        if isinstance(key_states, ttnn.Tensor) and key_states.dtype != ttnn.bfloat16:
            key_states = ttnn.typecast(key_states, ttnn.bfloat16)

        # On-device RoPE for decode (trace-safe) via BailingRotarySetup
        cos_ttnn, sin_ttnn = self._rotary_setup.get_cos_sin_for_decode(cur_pos_tt)

        if self.head_dim <= 256:
            # Standard sharded decode RoPE path (head_dim fits kernel limit)
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
        else:
            # Split decode RoPE for head_dim > 256 (kernel limit workaround).
            # Global layers (head_dim=512): split into 256-dim chunks, apply
            # RoPE to each, concat back. Skip sharding for simplicity — only
            # 10 out of 60 layers use this path.
            trans_mat = self._rotary_setup.get_trans_mat_decode_sharded(batch_size)
            query_states, key_states = self._apply_rotary_embedding_llama(
                query_states, key_states, cos_ttnn, sin_ttnn, trans_mat, is_decode_mode=True, batch_size=batch_size
            )

        # query_states/key_states are now in [1, B, H, D] — the S B H D layout
        # that paged kernels expect. After RoPE they stay in their current
        # memory config (HEIGHT_SHARDED L1 for head_dim<=256 path, DRAM for
        # the split path). No copy-to-replicated needed — the all_gather in
        # _project_qkv already produced full tensors on each device.
        query_states_paged = query_states
        kv_key = key_states
        kv_value = value_states

        # Typecast value_states to bfloat16 if needed (for paged cache)
        if isinstance(kv_value, ttnn.Tensor) and kv_value.dtype != ttnn.bfloat16:
            kv_value = ttnn.typecast(kv_value, ttnn.bfloat16)

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
            sliding_window=self.sliding_window,
        )
        # attn_output: [1, B, H, head_dim]

        # HEIGHT_SHARDED for nlp_concat_heads_decode
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
            num_heads=self.num_attention_heads,
        )

        # Output projection runs on width-sharded tensor before slice
        attn_output = self.o_proj(attn_output)

        if batch_size < 32:
            attn_output = ttnn.slice(attn_output, [0, 0, 0, 0], [1, 1, batch_size, attn_output.shape[-1]])

        attn_output = ttnn.reshape(attn_output, (batch_size, seq_length, -1))

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
        """Forward pass through Gemma4 attention.

        Args:
            hidden_states: Input tensor [batch, seq_len, hidden_size]
            position_embeddings: Tuple of (cos, sin) for RoPE
            attention_mask: Optional attention mask
            past_key_values: Optional KV cache (TTNNGemma4PagedAttentionKVCache or DynamicCache)
            cache_position: Optional cache position tensor
            position_ids: Optional position IDs (unused, for API compatibility)
            **kwargs: Additional arguments for forward compatibility

        Returns:
            Tuple of (attention_output, None)
        """
        seq_length = hidden_states.shape[1]

        use_paged = isinstance(past_key_values, TTNNGemma4PagedAttentionKVCache) or isinstance(
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


class TTNNGemma4PagedAttentionKVCache(Cache):
    """Dual paged KV cache for Gemma4 that routes sliding/global layers to separate caches.

    Gemma4 has two types of attention layers with different KV configurations:
    - Sliding layers: 16 KV heads, head_dim=256, with sliding window
    - Global layers: 4 KV heads, head_dim=512, full context

    This class wraps two TTNNPagedAttentionKVCache instances and routes
    operations to the correct cache based on layer_idx.
    """

    def __init__(self, text_config, global_layer_indices, device=None):
        """Initialize dual paged KV cache.

        Args:
            text_config: Gemma4TextConfig with model dimensions
            global_layer_indices: Set or list of layer indices that use global attention
            device: TTNN device or mesh device
        """
        try:
            super().__init__(layers=[])
        except Exception:
            super().__init__()

        self.text_config = text_config
        self.global_layer_indices = set(global_layer_indices)

        # Build layer routing: layer_idx -> ('sliding'|'global', cache_layer_idx)
        self._routing = {}
        sliding_idx = 0
        global_idx = 0
        for layer_idx in range(text_config.num_hidden_layers):
            if layer_idx in self.global_layer_indices:
                self._routing[layer_idx] = ("global", global_idx)
                global_idx += 1
            else:
                self._routing[layer_idx] = ("sliding", sliding_idx)
                sliding_idx += 1

        num_sliding = text_config.num_hidden_layers - len(self.global_layer_indices)
        num_global = len(self.global_layer_indices)

        # Sliding: only needs ceil(window/block_size) blocks per sequence.
        # window=1024, block_size=64 -> 16 blocks. Use 32 for headroom.
        sliding_config = PagedAttentionConfig(block_size=64, max_num_blocks=32)
        # Global: needs ceil(max_seq_len/block_size) blocks. For typical
        # inference (prompt + 128-256 new tokens), 64 blocks = 4096 tokens.
        global_config = PagedAttentionConfig(block_size=64, max_num_blocks=64)

        # Sliding cache: more KV heads, smaller head_dim
        self.sliding_cache = TTNNPagedAttentionKVCache(
            num_layers=num_sliding,
            num_kv_heads=text_config.num_key_value_heads,  # 16
            head_dim=text_config.head_dim,  # 256
            config=sliding_config,
        )

        # Global cache: fewer KV heads, larger head_dim
        # Use global-specific config attributes if available, otherwise derive from config
        global_num_kv_heads = getattr(
            text_config,
            "num_global_key_value_heads",
            getattr(text_config, "global_num_key_value_heads", 4),
        )
        global_head_dim = getattr(
            text_config,
            "global_head_dim",
            getattr(text_config, "head_dim_global", 512),
        )

        self.global_cache = TTNNPagedAttentionKVCache(
            num_layers=num_global,
            num_kv_heads=global_num_kv_heads,  # 4
            head_dim=global_head_dim,  # 512
            config=global_config,
        )

        self._device = device

    def _get_cache_and_idx(self, layer_idx: int):
        """Route layer_idx to the correct sub-cache and cache-local index.

        Returns:
            Tuple of (cache_instance, cache_local_layer_idx)
        """
        cache_type, cache_idx = self._routing[layer_idx]
        if cache_type == "global":
            return self.global_cache, cache_idx
        else:
            return self.sliding_cache, cache_idx

    def to_device(self, device) -> "TTNNGemma4PagedAttentionKVCache":
        """Move both sub-caches to device."""
        self._device = device
        self.sliding_cache.to_device(device)
        self.global_cache.to_device(device)
        return self

    def paged_fill_on_device(
        self,
        key_states: ttnn.Tensor,
        value_states: ttnn.Tensor,
        layer_idx: int,
        batch_idx: int = 0,
    ):
        """Fill KV cache for a layer, routing to the correct sub-cache."""
        cache, cache_idx = self._get_cache_and_idx(layer_idx)
        cache.paged_fill_on_device(key_states, value_states, cache_idx, batch_idx)

    def paged_update_on_device(
        self,
        key_states: ttnn.Tensor,
        value_states: ttnn.Tensor,
        layer_idx: int,
        current_pos: ttnn.Tensor,
    ):
        """Update KV cache for a layer, routing to the correct sub-cache."""
        cache, cache_idx = self._get_cache_and_idx(layer_idx)
        cache.paged_update_on_device(key_states, value_states, cache_idx, current_pos)

    def update_seq_length(self, layer_idx: int, seq_len: int = 1) -> None:
        """Increment Python-side sequence counters, routing to the correct sub-cache."""
        cache, cache_idx = self._get_cache_and_idx(layer_idx)
        cache.update_seq_length(cache_idx, seq_len)

    def paged_sdpa_decode(
        self,
        query: ttnn.Tensor,
        layer_idx: int,
        current_pos: ttnn.Tensor,
        scale: float = 1.0,
        program_config=None,
        compute_kernel_config=None,
        sliding_window: int | None = None,
    ) -> ttnn.Tensor:
        """Decode using paged KV cache, routing to the correct sub-cache."""
        cache, cache_idx = self._get_cache_and_idx(layer_idx)
        return cache.paged_sdpa_decode(
            query,
            cache_idx,
            current_pos,
            scale,
            program_config,
            compute_kernel_config,
            sliding_window=sliding_window,
        )

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[dict] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Update CPU-side KV cache (DynamicCache compatibility)."""
        cache, cache_idx = self._get_cache_and_idx(layer_idx)
        return cache.update(key_states, value_states, cache_idx, cache_kwargs)

    def get_seq_length(self, layer_idx: int = 0) -> int:
        """Get sequence length for a layer, routing to the correct sub-cache."""
        if layer_idx in self._routing:
            cache, cache_idx = self._get_cache_and_idx(layer_idx)
            return cache.get_seq_length(cache_idx)
        # Default: return sliding cache layer 0
        return self.sliding_cache.get_seq_length(0)

    def get_max_cache_shape(self) -> Optional[int]:
        """Return the max cache shape (minimum of both sub-caches)."""
        sliding_max = self.sliding_cache.get_max_cache_shape()
        global_max = self.global_cache.get_max_cache_shape()
        if sliding_max is None:
            return global_max
        if global_max is None:
            return sliding_max
        return min(sliding_max, global_max)

    def reset(self) -> None:
        """Reset both sub-caches for a new generation turn."""
        self.sliding_cache.reset()
        self.global_cache.reset()
