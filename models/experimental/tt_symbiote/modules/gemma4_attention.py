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

import os
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
    TTNNLinearGemma4IColShardedWAllReduced,
    TTNNLinearGemma4IReplicatedWColSharded,
)
from models.experimental.tt_symbiote.modules.normalization import TTNNLocalRMSNorm
from models.experimental.tt_symbiote.core.run_config import trace_enabled
from models.experimental.tt_symbiote.core.module import DeviceArch, MeshShapeToDeviceArch

try:
    from transformers.cache_utils import Cache
except ImportError:
    Cache = object


class CompatNlpCreateQkvHeadsDecode:
    """Architecture-aware wrapper for ``ttnn.experimental.nlp_create_qkv_heads_decode``.

    On Wormhole (T3K, etc.) the original kernel is correct and is invoked directly.

    On Blackhole (QB2) the kernel drops every odd-indexed head (PCC=0 at heads
    1, 3, 5, ... while heads 0, 2, 4, ... remain at PCC=0.9999), causing model
    decode output to collapse to multilingual gibberish.  We sidestep the bug by
    splitting the fused QKV tensor with ``ttnn.slice`` + ``ttnn.reshape``, which
    are both correct on Blackhole.

    Output interface matches the original op: HEIGHT_SHARDED tensors with shape
    ``[1, B, num_heads, head_dim]`` (Q) and ``[1, B, num_kv_heads, head_dim]`` (K, V).
    """

    @staticmethod
    def _current_arch() -> DeviceArch | None:
        return MeshShapeToDeviceArch.get(os.environ.get("MESH_DEVICE"))

    @staticmethod
    def split(qkv_4d: ttnn.Tensor, num_heads: int, num_kv_heads: int, memory_config):
        """Run the original op on Wormhole, manual slice+reshape on Blackhole."""
        arch = CompatNlpCreateQkvHeadsDecode._current_arch()
        if arch == DeviceArch.QB2:
            return CompatNlpCreateQkvHeadsDecode._manual_split(qkv_4d, num_heads, num_kv_heads, memory_config)
        return ttnn.experimental.nlp_create_qkv_heads_decode(
            qkv_4d,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            memory_config=memory_config,
        )

    @staticmethod
    def _manual_split(qkv_4d: ttnn.Tensor, num_heads: int, num_kv_heads: int, memory_config):
        # qkv_4d shape: (1, 1, B, fused_size) where fused_size == (num_heads + 2*num_kv_heads) * head_dim
        batch_size = qkv_4d.shape[2]
        fused_size = qkv_4d.shape[3]
        head_dim = fused_size // (num_heads + 2 * num_kv_heads)
        q_size = num_heads * head_dim
        kv_size = num_kv_heads * head_dim

        q = ttnn.slice(qkv_4d, [0, 0, 0, 0], [1, 1, batch_size, q_size])
        k = ttnn.slice(qkv_4d, [0, 0, 0, q_size], [1, 1, batch_size, q_size + kv_size])
        v = ttnn.slice(qkv_4d, [0, 0, 0, q_size + kv_size], [1, 1, batch_size, q_size + 2 * kv_size])

        q = ttnn.reshape(q, (1, batch_size, num_heads, head_dim))
        k = ttnn.reshape(k, (1, batch_size, num_kv_heads, head_dim))
        v = ttnn.reshape(v, (1, batch_size, num_kv_heads, head_dim))

        # Convert to the requested HEIGHT_SHARDED memory_config so callers see the
        # same output layout the original op produces.
        q = ttnn.to_memory_config(q, memory_config)
        k = ttnn.to_memory_config(k, memory_config)
        v = ttnn.to_memory_config(v, memory_config)
        return q, k, v


@trace_enabled
class TTNNGemma4Attention(TTNNModule):
    """Base class for Gemma4 attention. Use from_torch() as a factory.

    Returns TTNNGemma4SlidingAttention (is_sliding=True, head_dim=256) or
    TTNNGemma4GlobalAttention (is_sliding=False, head_dim=512) depending on
    the HuggingFace attention layer passed to from_torch().

    Subclasses implement:
    - _sdpa_transpose_output: controls SDPA output shape
    - _concat_attn_output: variant-specific head-concat op (nlp_concat_heads vs reshape)
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
        """Factory: returns TTNNGemma4SlidingAttention or TTNNGemma4GlobalAttention.

        Call on the base class; dispatches based on hf_attn.is_sliding.
        """
        if cls is TTNNGemma4Attention:
            if hf_attn.is_sliding:
                return TTNNGemma4SlidingAttention.from_torch(hf_attn, distributed=distributed)
            else:
                return TTNNGemma4GlobalAttention.from_torch(hf_attn, distributed=distributed)
        raise TypeError(
            f"{cls.__name__}: subclasses must implement from_torch calling _build_common, "
            f"not call super().from_torch()"
        )

    @classmethod
    def _build_common(cls, hf_attn, num_kv_heads: int, distributed: bool = True):
        """Build shared Gemma4 attention state. Called by subclass from_torch methods."""
        new_attn = cls()
        new_attn._fallback_torch_layer = hf_attn

        config = hf_attn.config
        new_attn.is_sliding = hf_attn.is_sliding
        new_attn.layer_idx = hf_attn.layer_idx
        new_attn.num_attention_heads = config.num_attention_heads  # 32
        assert (
            new_attn.num_attention_heads <= 32
        ), f"nlp_create_qkv_heads supports num_heads<=32, got {new_attn.num_attention_heads}"
        new_attn.num_key_value_heads = num_kv_heads
        new_attn.num_key_value_groups = new_attn.num_attention_heads // new_attn.num_key_value_heads
        new_attn.head_dim = hf_attn.head_dim  # 256 sliding, 512 global
        new_attn.hidden_size = config.hidden_size
        new_attn.scaling = getattr(hf_attn, "scaling", 1.0)
        new_attn.sliding_window = getattr(hf_attn, "sliding_window", None)

        LinearClsOut = TTNNLinearGemma4IReplicatedWColSharded if distributed else TTNNLinear

        q_weight = hf_attn.q_proj.weight.data.clone()
        k_weight = hf_attn.k_proj.weight.data.clone()

        q_size = new_attn.num_attention_heads * new_attn.head_dim
        kv_size = new_attn.num_key_value_heads * new_attn.head_dim

        has_v_proj = hf_attn.v_proj is not None
        new_attn._has_v_proj = True
        new_attn._is_kv_sharing = not has_v_proj

        if has_v_proj:
            v_weight = hf_attn.v_proj.weight.data.clone()
        else:
            # K=V sharing for global layers: duplicate K weight as V.
            v_weight = k_weight.clone()

        fused_weight = torch.cat([q_weight, k_weight, v_weight], dim=0)
        fused_linear = torch.nn.Linear(new_attn.hidden_size, fused_weight.shape[0], bias=False)
        fused_linear.weight.data = fused_weight
        new_attn.qkv_proj = TTNNLinearGemma4IColShardedWAllReduced.from_torch(fused_linear)
        new_attn._q_size = q_size
        new_attn._kv_size = kv_size

        new_attn.q_proj = None
        new_attn.k_proj = None
        new_attn.v_proj = None

        new_attn.o_proj = LinearClsOut.from_torch(hf_attn.o_proj)

        new_attn.q_norm = TTNNLocalRMSNorm.from_torch(hf_attn.q_norm)
        new_attn.k_norm = TTNNLocalRMSNorm.from_torch(hf_attn.k_norm)
        if hasattr(hf_attn, "v_norm") and hf_attn.v_norm is not None:
            if not hasattr(hf_attn.v_norm, "weight") or hf_attn.v_norm.weight is None:
                hf_attn.v_norm._norm_dim = new_attn.head_dim
            new_attn.v_norm = TTNNLocalRMSNorm.from_torch(hf_attn.v_norm)
        else:
            new_attn.v_norm = None

        new_attn.sdpa = TTNNSDPAAttention()
        new_attn.core_grid = ttnn.CoreGrid(y=8, x=8)

        return new_attn

    def move_weights_to_device_impl(self):
        """Initialize SDPA config and move weights to device."""
        super().move_weights_to_device_impl()

        # Query grid dynamically from device
        grid = self.device.compute_with_storage_grid_size()
        self.core_grid = ttnn.CoreGrid(y=grid.y, x=grid.x)

        if not hasattr(self.sdpa, "decode_program_config"):
            # Prefill: auto-derive (no program_config), matching reference prefill.py
            self.sdpa.program_config = None

            # Decode: sliding default is None; TTNNGemma4GlobalAttention overrides this.
            self.sdpa.decode_program_config = None
            self.sdpa.compute_kernel_config = ttnn.init_device_compute_kernel_config(
                self.device.arch(),
                math_fidelity=ttnn.MathFidelity.HiFi4,
                math_approx_mode=False,
                fp32_dest_acc_en=False,
                packer_l1_acc=False,
            )

        # Enable fp32 destination accumulation on linear projections.
        # Default matmul uses HiFi2 + bf16 dest accumulation, which causes
        # significant precision loss for global layers where o_proj has a
        # 16384-element inner dimension (vs 8192 for sliding layers).
        linear_compute_config = ttnn.init_device_compute_kernel_config(
            self.device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )
        self.qkv_proj.compute_kernel_config = linear_compute_config
        self.o_proj.compute_kernel_config = linear_compute_config

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

        # RoPE cos/sin caches are now managed at the model level
        # (TTNNGemma4TextModel) and passed to attention as arguments.
        # No BailingRotarySetup needed per-layer.
        # HF's Gemma4TextRotaryEmbedding returns full-width cos/sin with
        # identity values for non-rotary dims, so no partial RoPE handling needed.

    @property
    def _sdpa_transpose_output(self) -> bool:
        raise NotImplementedError(f"{type(self).__name__} must implement _sdpa_transpose_output")

    def _concat_attn_output(self, attn_output: ttnn.Tensor) -> ttnn.Tensor:
        raise NotImplementedError(f"{type(self).__name__} must implement _concat_attn_output")

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
        """Apply per-head RMSNorm directly on 4D [B, S, H, D] input.

        ttnn.rms_norm normalizes over the last dimension for any input rank.
        Prerequisite: verified by test_change1_rms_norm_4d.py (PCC >= 0.999 for
        both 4D prefill and 2D decode inputs with [1, D] weight).
        Eliminates 6 reshape ops (3 flatten + 3 restore) across Q/K/V norm calls.
        """
        return norm_module(states)

    def _project_qkv(self, hidden_states, batch_size, seq_length, for_decode=False):
        """Project hidden states to Q, K, V via fused QKV matmul and apply per-head norms.

        Uses a single fused QKV projection (TTNNLinearIColShardedWAllReduced)
        instead of 3 separate projections + all_gathers. Input arrives col-sharded
        from the norm; the fused linear does matmul + reduce_scatter + all_gather
        in 1 matmul + 2 CCL ops (vs 3 matmuls + 4 CCL ops before).

        Head-split uses ``ttnn.experimental.nlp_create_qkv_heads(_decode)``
        unconditionally (no fallback). The decode variant emits
        [1, B, nH, D] HEIGHT_SHARDED; the prefill variant emits [B, nH, S, D].
        Per-head Q/K/V RMSNorms operate on dim=-1 (head_dim) and are
        layout-invariant.

        UAF hazard (decode and prefill paths): rank-3 -> rank-4 ttnn.reshape on
        TILE_LAYOUT DRAM_INTERLEAVED is metadata-only (a view). The
        ttnn.deallocate(qkv_states) MUST happen AFTER the fused op consumes
        qkv_4d; freeing the source before the op reads the view is a silent
        use-after-free. See research_topics.md:765 for the generic class.

        Args:
            hidden_states: Input tensor [batch, seq_len, hidden_size]
            batch_size: Batch dimension
            seq_length: Sequence length
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

        q_size = self._q_size
        kv_size = self._kv_size

        if for_decode and seq_length == 1:
            # Fused decode head-split. Op signature: input [1, 1, B, fused],
            # output Q [1, B, nQ, D] / K,V [1, B, nKV, D] HEIGHT_SHARDED.
            qkv_4d = ttnn.reshape(qkv_states, (1, 1, batch_size, q_size + 2 * kv_size))
            # HEIGHT_SHARDED output mem-config — same idiom as the K/V reshard
            # block in _forward_decode_paged.
            tile_size = 32
            shard_h = ((self.num_key_value_heads + tile_size - 1) // tile_size) * tile_size
            assert batch_size <= 8, f"HEIGHT_SHARDED memcfg core_grid=(y=1, x={batch_size}) exceeds T3K 8x8 grid"
            hs_mem_cfg = ttnn.create_sharded_memory_config(
                shape=(shard_h, self.head_dim),
                core_grid=ttnn.CoreGrid(y=1, x=batch_size),
                strategy=ttnn.ShardStrategy.HEIGHT,
                orientation=ttnn.ShardOrientation.ROW_MAJOR,
                use_height_and_width_as_shard_shape=True,
            )
            q, k, v = CompatNlpCreateQkvHeadsDecode.split(
                qkv_4d,
                num_heads=self.num_attention_heads,
                num_kv_heads=self.num_key_value_heads,
                memory_config=hs_mem_cfg,
            )
            ttnn.deallocate(qkv_4d)
            ttnn.deallocate(qkv_states)  # UAF guard: see _project_qkv docstring (research_topics.md:765).
            # DRAM-out for paged_sdpa_decode requirement (Q must be DRAM when not sharded).
            q_interleaved = ttnn.sharded_to_interleaved(q, ttnn.DRAM_MEMORY_CONFIG)
            ttnn.deallocate(q)
            # Reshard K/V back to interleaved: ttnn.rms_norm doesn't accept HEIGHT_SHARDED inputs (attention_1d.py:618).
            k_interleaved = ttnn.sharded_to_interleaved(k, ttnn.L1_MEMORY_CONFIG)
            ttnn.deallocate(k)
            v_interleaved = ttnn.sharded_to_interleaved(v, ttnn.L1_MEMORY_CONFIG)
            ttnn.deallocate(v)
            query_states = self._apply_per_head_norm(
                q_interleaved,
                self.q_norm,
                batch_size,
                seq_length,
                self.num_attention_heads,
                self.head_dim,
            )
            key_states = self._apply_per_head_norm(
                k_interleaved,
                self.k_norm,
                batch_size,
                seq_length,
                self.num_key_value_heads,
                self.head_dim,
            )
            if self.v_norm is not None:
                value_states = self._apply_per_head_norm(
                    v_interleaved,
                    self.v_norm,
                    batch_size,
                    seq_length,
                    self.num_key_value_heads,
                    self.head_dim,
                )
            else:
                value_states = v_interleaved
            return query_states, key_states, value_states
        else:
            # Fused prefill head-split. rank-3 [B, S, fused] -> rank-4 [B, 1, S, fused] required by op.
            qkv_4d = ttnn.reshape(qkv_states, (batch_size, 1, seq_length, q_size + 2 * kv_size))
            q, k, v = ttnn.experimental.nlp_create_qkv_heads(
                qkv_4d,
                num_heads=self.num_attention_heads,
                num_kv_heads=self.num_key_value_heads,
                # transpose_k_heads=False: required for paged_sdpa downstream layout
                # (see nlp_create_qkv_heads.cpp:31).
                transpose_k_heads=False,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            ttnn.deallocate(qkv_4d)
            ttnn.deallocate(qkv_states)  # UAF guard: see _project_qkv docstring (research_topics.md:765).
            # Apply per-head norms on [B, nH, S, D]; norms are dim=-1 invariant.
            query_states = self._apply_per_head_norm(
                q,
                self.q_norm,
                batch_size,
                seq_length,
                self.num_attention_heads,
                self.head_dim,
            )
            key_states = self._apply_per_head_norm(
                k,
                self.k_norm,
                batch_size,
                seq_length,
                self.num_key_value_heads,
                self.head_dim,
            )
            if self.v_norm is not None:
                value_states = self._apply_per_head_norm(
                    v,
                    self.v_norm,
                    batch_size,
                    seq_length,
                    self.num_key_value_heads,
                    self.head_dim,
                )
            else:
                value_states = v
            return query_states, key_states, value_states

    def _apply_rope(self, query_states, key_states, cos, sin, token_index=None):
        """Apply rotary position embedding to Q and K.

        HF's Gemma4TextRotaryEmbedding returns full-width cos/sin (head_dim-wide)
        with identity values (cos=1, sin=0) for non-rotary dimensions. This
        naturally produces pass-through for non-rotary dims, so no split-apply-concat
        is needed. Matches the sdjordjevic reference implementation.
        """
        orig_q_shape = query_states.shape
        orig_k_shape = key_states.shape

        query_states = ttnn.experimental.rotary_embedding(query_states, cos, sin, token_index)
        key_states = ttnn.experimental.rotary_embedding(key_states, cos, sin, token_index)

        # Restore dim-2 if the kernel padded it to TILE_HEIGHT
        if query_states.shape[2] != orig_q_shape[2]:
            query_states = query_states[:, :, : orig_q_shape[2], :]
        if key_states.shape[2] != orig_k_shape[2]:
            key_states = key_states[:, :, : orig_k_shape[2], :]

        return query_states, key_states

    def _cpu_sdpa_decode(
        self,
        query_states: ttnn.Tensor,
        past_key_values,
        layer_idx: int,
        cur_pos_tt: ttnn.Tensor,
    ) -> ttnn.Tensor:
        """CPU fp32 SDPA fallback for decode, bypassing the paged SDPA kernel.

        Reads Q and KV cache from device, computes attention in fp32 on CPU
        via torch.nn.functional.scaled_dot_product_attention, and sends the
        result back to device. Activated by TT_GEMMA4_CPU_SDPA=1 for global
        layers to work around bf16 precision loss in the paged SDPA kernel.
        """
        if not getattr(self, "_cpu_sdpa_logged", False):
            print(f"[CPU-SDPA] Layer {layer_idx}: Using CPU fp32 attention (TT_GEMMA4_CPU_SDPA=1)")
            self._cpu_sdpa_logged = True

        device = query_states.device()
        num_devices = device.get_num_devices() if hasattr(device, "get_num_devices") else 1

        # Q: [1, B, H, D] replicated -> read to CPU
        if num_devices > 1:
            q_cpu = ttnn.to_torch(query_states, mesh_composer=ttnn.ConcatMeshToTensor(device, dim=0))
            q_cpu = q_cpu[:1]  # take first replica
        else:
            q_cpu = ttnn.to_torch(query_states)
        # q_cpu: [1, B, num_q_heads, head_dim]

        # Get the correct sub-cache and cache-local index
        cache, cache_idx = past_key_values._get_cache_and_idx(layer_idx)

        # K/V cache: [max_blocks, nkv, block_size, head_dim] replicated
        k_cache_tt = cache._tt_key_cache[cache_idx]
        v_cache_tt = cache._tt_value_cache[cache_idx]

        if num_devices > 1:
            k_cache_cpu = ttnn.to_torch(k_cache_tt, mesh_composer=ttnn.ConcatMeshToTensor(device, dim=0))
            k_cache_cpu = k_cache_cpu[: k_cache_tt.shape[0]]
            v_cache_cpu = ttnn.to_torch(v_cache_tt, mesh_composer=ttnn.ConcatMeshToTensor(device, dim=0))
            v_cache_cpu = v_cache_cpu[: v_cache_tt.shape[0]]
        else:
            k_cache_cpu = ttnn.to_torch(k_cache_tt)
            v_cache_cpu = ttnn.to_torch(v_cache_tt)
        # k/v_cache_cpu: [max_blocks, nkv, block_size, head_dim]

        # cur_pos: scalar position for batch=1 decode
        if num_devices > 1:
            cur_pos_cpu = ttnn.to_torch(cur_pos_tt, mesh_composer=ttnn.ConcatMeshToTensor(device, dim=0))
            cur_pos_cpu = cur_pos_cpu[:1]
        else:
            cur_pos_cpu = ttnn.to_torch(cur_pos_tt)
        cur_pos = int(cur_pos_cpu.flatten()[0].item())

        # Reshape cache from [max_blocks, nkv, block_size, hd] -> [nkv, seq, hd]
        # permute(1,0,2,3) -> [nkv, max_blocks, block_size, hd]
        # reshape -> [nkv, max_blocks*block_size, hd]
        # slice to cur_pos+1 valid positions
        nkv = k_cache_cpu.shape[1]
        hd = k_cache_cpu.shape[3]
        k_seq = k_cache_cpu.permute(1, 0, 2, 3).reshape(nkv, -1, hd)[:, : cur_pos + 1, :]
        v_seq = v_cache_cpu.permute(1, 0, 2, 3).reshape(nkv, -1, hd)[:, : cur_pos + 1, :]

        # GQA expansion: [nkv, seq, hd] -> [num_q_heads, seq, hd]
        n_rep = self.num_attention_heads // self.num_key_value_heads
        if n_rep > 1:
            k_seq = k_seq.repeat_interleave(n_rep, dim=0)
            v_seq = v_seq.repeat_interleave(n_rep, dim=0)

        # Prepare for SDPA: Q [B, H, 1, D], K [B, H, S, D], V [B, H, S, D]
        batch_size = q_cpu.shape[1]
        q_sdpa = q_cpu.squeeze(0).permute(0, 1, 2).unsqueeze(2).float()  # [B, H, 1, D]
        k_sdpa = k_seq.unsqueeze(0).expand(batch_size, -1, -1, -1).float()  # [B, H, S, D]
        v_sdpa = v_seq.unsqueeze(0).expand(batch_size, -1, -1, -1).float()  # [B, H, S, D]

        # fp32 SDPA on CPU
        attn_out = torch.nn.functional.scaled_dot_product_attention(
            q_sdpa,
            k_sdpa,
            v_sdpa,
            scale=self.scaling,
            is_causal=False,
        )
        # attn_out: [B, H, 1, D]

        # Reshape back to [1, B, H, D] for downstream concat_heads
        attn_out = attn_out.squeeze(2).unsqueeze(0).to(torch.bfloat16)  # [1, B, H, D]

        # Send back to device as replicated tensor in TILE_LAYOUT
        mesh_mapper = ttnn.ReplicateTensorToMesh(device) if num_devices > 1 else None
        attn_output = ttnn.from_torch(
            attn_out,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            mesh_mapper=mesh_mapper,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        return attn_output

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

        # Apply HF-style RoPE. cos/sin come from the model-level cache
        # (passed via position_embeddings from TTNNGemma4TextModel).
        cos, sin = position_embeddings

        if isinstance(query_states, ttnn.Tensor) and query_states.dtype != ttnn.bfloat16:
            query_states = ttnn.typecast(query_states, ttnn.bfloat16)
        if isinstance(key_states, ttnn.Tensor) and key_states.dtype != ttnn.bfloat16:
            key_states = ttnn.typecast(key_states, ttnn.bfloat16)

        query_states, key_states = self._apply_rope(query_states, key_states, cos, sin)

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

        attn_output = self.sdpa(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0,
            scaling=self.scaling,
            is_causal=self.is_causal,
            transpose_output=self._sdpa_transpose_output,
        )

        attn_output = self._concat_attn_output(attn_output)

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
        **kwargs,
    ) -> tuple[ttnn.Tensor, Optional[torch.Tensor]]:
        """Decode path using paged attention with on-device KV cache.

        TTNN paged kernels require tensors in [1, batch, heads, head_dim]
        layout (S B H D) whereas _project_qkv returns the standard
        [batch, heads, seq, head_dim] (B H S D).
        """
        batch_size, seq_length = hidden_states.shape[0], hidden_states.shape[1]

        layer_idx = self.layer_idx

        cur_pos_tt = self._get_cur_pos_device_tensor(cache_position, past_key_values, layer_idx, batch_size)

        # Project Q/K/V with per-head norms (no RoPE yet).
        # for_decode=True skips the [B,H,S,D] permute — returns [B,S,H,D].
        query_states, key_states, value_states = self._project_qkv(
            hidden_states, batch_size, seq_length, for_decode=True
        )
        assert (
            query_states.shape[0] == 1 and query_states.shape[1] == batch_size
        ), f"_project_qkv(for_decode=True) must return [1, {batch_size}, H, D]; got {query_states.shape}"

        # Typecast to bfloat16 for rotary_embedding
        if isinstance(query_states, ttnn.Tensor) and query_states.dtype != ttnn.bfloat16:
            query_states = ttnn.typecast(query_states, ttnn.bfloat16)
        if isinstance(key_states, ttnn.Tensor) and key_states.dtype != ttnn.bfloat16:
            key_states = ttnn.typecast(key_states, ttnn.bfloat16)

        # On-device RoPE for decode via ttnn.embedding lookup (trace-safe).
        # cos_cache/sin_cache are 2D [max_seq_len, head_dim] from model-level cache.
        cos_cache, sin_cache = position_embeddings  # Passed from model via decoder layer

        # rope_position_idx: [1, 32] uint32 tensor for ttnn.embedding lookup
        rope_position_idx = kwargs.get("rope_position_idx")

        # Gather position-specific cos/sin via ttnn.embedding
        cos_pos = ttnn.embedding(rope_position_idx, cos_cache, layout=ttnn.TILE_LAYOUT)
        sin_pos = ttnn.embedding(rope_position_idx, sin_cache, layout=ttnn.TILE_LAYOUT)
        cos_pos = ttnn.reshape(cos_pos, (1, 1, cos_pos.shape[-2], cos_pos.shape[-1]))
        sin_pos = ttnn.reshape(sin_pos, (1, 1, sin_pos.shape[-2], sin_pos.shape[-1]))

        # Apply RoPE with token_index=0 (gathered cache already has the right position data)
        query_states, key_states = self._apply_rope(query_states, key_states, cos_pos, sin_pos, token_index=0)

        # query_states/key_states are now in [1, B, H, D] — the S B H D layout
        # that paged kernels expect.
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
            use_height_and_width_as_shard_shape=True,
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

        # Paged SDPA decode (or CPU fp32 fallback for all layers)
        use_cpu_sdpa = os.environ.get("TT_GEMMA4_CPU_SDPA", "0").lower() in ("1", "true", "yes")
        if use_cpu_sdpa:
            attn_output = self._cpu_sdpa_decode(
                query_states_paged,
                past_key_values,
                layer_idx,
                cur_pos_tt,
            )
        else:
            attn_output = past_key_values.paged_sdpa_decode(
                query_states_paged,
                layer_idx,
                current_pos=cur_pos_tt,
                scale=self.scaling,
                program_config=self.sdpa.decode_program_config,
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
                **kwargs,
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


class TTNNGemma4SlidingAttention(TTNNGemma4Attention):
    """Sliding-window attention for Gemma4 (32Q/16KV, head_dim=256, sliding_window=1024).

    Uses nlp_concat_heads for output concat: CB = H*D*128 = 1,048,576 B < L1 limit.
    Created by TTNNGemma4Attention.from_torch() when hf_attn.is_sliding is True.
    """

    @classmethod
    def from_torch(cls, hf_attn, distributed: bool = True):
        assert hf_attn.is_sliding, (
            f"TTNNGemma4SlidingAttention.from_torch expects is_sliding=True, " f"got is_sliding={hf_attn.is_sliding}"
        )
        return cls._build_common(hf_attn, num_kv_heads=hf_attn.config.num_key_value_heads, distributed=distributed)

    @property
    def module_name(self) -> str:
        base = self._unique_name or f"{self.__class__.__name__}_{id(self)}"
        return f"{base}[sliding]"

    @property
    def _sdpa_transpose_output(self) -> bool:
        return False

    def _concat_attn_output(self, attn_output: ttnn.Tensor) -> ttnn.Tensor:
        # Fused permute+reshape: [B, H, S, D] → [B, 1, S, H*D] → [B, S, H*D]
        attn_output_4d = ttnn.experimental.nlp_concat_heads(
            attn_output,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(attn_output)
        assert (
            attn_output_4d.shape[1] == 1
        ), f"nlp_concat_heads output expected unit dim-1, got shape {attn_output_4d.shape}"
        return ttnn.squeeze(attn_output_4d, 1)


class TTNNGemma4GlobalAttention(TTNNGemma4Attention):
    """Global (full-context) attention for Gemma4 (32Q/4KV, head_dim=512, K=V sharing).

    Uses ttnn.reshape for output concat: nlp_concat_heads CB = 2,097,152 B > L1 limit.
    Created by TTNNGemma4Attention.from_torch() when hf_attn.is_sliding is False.
    """

    @classmethod
    def from_torch(cls, hf_attn, distributed: bool = True):
        assert not hf_attn.is_sliding, (
            f"TTNNGemma4GlobalAttention.from_torch expects is_sliding=False, " f"got is_sliding={hf_attn.is_sliding}"
        )
        config = hf_attn.config
        use_alternative_attention = getattr(config, "attention_k_eq_v", False)
        num_kv_heads = config.num_global_key_value_heads if use_alternative_attention else config.num_key_value_heads
        return cls._build_common(hf_attn, num_kv_heads=num_kv_heads, distributed=distributed)

    @property
    def module_name(self) -> str:
        base = self._unique_name or f"{self.__class__.__name__}_{id(self)}"
        return f"{base}[global]"

    @property
    def _sdpa_transpose_output(self) -> bool:
        return True

    def _concat_attn_output(self, attn_output: ttnn.Tensor) -> ttnn.Tensor:
        # nlp_concat_heads CB = 2,097,152 B > L1 limit 1,499,136 B for head_dim=512.
        attn_shape = list(attn_output.shape)
        return ttnn.reshape(
            attn_output,
            (attn_shape[0], attn_shape[1], self.num_attention_heads * self.head_dim),
        )

    def move_weights_to_device_impl(self):
        super().move_weights_to_device_impl()
        # Global attention (head_dim=512) requires explicit SDPA program config for decode.
        self.sdpa.decode_program_config = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=(8, 4),
            q_chunk_size=32,
            k_chunk_size=64,
            exp_approx_mode=False,
        )


class TTNNGemma4PagedAttentionKVCache(Cache):
    """Dual paged KV cache for Gemma4 that routes sliding/global layers to separate caches.

    Gemma4 has two types of attention layers with different KV configurations:
    - Sliding layers: 16 KV heads, head_dim=256, with sliding window
    - Global layers: 4 KV heads, head_dim=512, full context

    This class wraps two TTNNPagedAttentionKVCache instances and routes
    operations to the correct cache based on layer_idx.
    """

    def __init__(
        self, text_config, global_layer_indices, device=None, sliding_max_num_blocks=32, global_max_num_blocks=64
    ):
        """Initialize dual paged KV cache.

        Args:
            text_config: Gemma4TextConfig with model dimensions
            global_layer_indices: Set or list of layer indices that use global attention
            device: TTNN device or mesh device
            sliding_max_num_blocks: Max blocks for sliding-window layers (default 32 = 2048 tokens).
                Use 16 (1024 tokens) for memory-constrained smoke tests.
            global_max_num_blocks: Max blocks for global-attention layers (default 64 = 4096 tokens).
                Use 32 (2048 tokens) for memory-constrained smoke tests.
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
        # window=1024, block_size=64 -> 16 blocks. Default 32 for headroom.
        sliding_config = PagedAttentionConfig(block_size=64, max_num_blocks=sliding_max_num_blocks)
        # Global: needs ceil(max_seq_len/block_size) blocks. For typical
        # inference (prompt + 128-256 new tokens), 64 blocks = 4096 tokens.
        global_config = PagedAttentionConfig(block_size=64, max_num_blocks=global_max_num_blocks)

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
        self._sliding_window = getattr(text_config, "sliding_window", 1024)

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
