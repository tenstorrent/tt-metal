# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Rotary Position Embedding (RoPE) implementations for TTNN."""

from typing import Any, Tuple, Union
import torch
import torch.nn as nn

import ttnn
from models.experimental.tt_symbiote.core.module import TTNNModule


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


class TorchRotaryPositionEmbedding(nn.Module):
    """PyTorch implementation of Rotary Position Embedding."""

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        unsqueeze_dim: int = 1,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Applies Rotary Position Embedding to the query and key tensors.

        Args:
            q: The query tensor.
            k: The key tensor.
            cos: The cosine part of the rotary embedding.
            sin: The sine part of the rotary embedding.
            unsqueeze_dim: The dimension along which to unsqueeze cos and sin.

        Returns:
            Tuple of (rotated_query, rotated_key)
        """
        cos = cos.unsqueeze(unsqueeze_dim)
        sin = sin.unsqueeze(unsqueeze_dim)

        # Keep half or full tensor for later concatenation
        rotary_dim = cos.shape[-1]
        q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
        k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]

        # Apply rotary embeddings
        q_embed = (q_rot * cos) + (rotate_half(q_rot) * sin)
        k_embed = (k_rot * cos) + (rotate_half(k_rot) * sin)

        # Concatenate back to full shape
        q_embed = torch.cat([q_embed, q_pass], dim=-1)
        k_embed = torch.cat([k_embed, k_pass], dim=-1)
        return q_embed, k_embed


class TTNNRotaryPositionEmbedding(TTNNModule):
    """TTNN-accelerated Rotary Position Embedding."""

    def forward(
        self,
        q: ttnn.Tensor,
        k: ttnn.Tensor,
        cos: ttnn.Tensor,
        sin: ttnn.Tensor,
    ) -> Tuple[ttnn.Tensor, ttnn.Tensor]:
        """
        Forward pass through RoPE layer.

        Args:
            q: Query tensor
            k: Key tensor
            cos: Cosine position embeddings
            sin: Sine position embeddings
            fused_sequence_threshold: Sequence length threshold for using fused operation (default: 128)

        Returns:
            Tuple of (rotated_query, rotated_key)
        """

        if q.layout != ttnn.TILE_LAYOUT:
            q = ttnn.to_layout(q, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        if k.layout != ttnn.TILE_LAYOUT:
            k = ttnn.to_layout(k, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        if cos.layout != ttnn.TILE_LAYOUT:
            cos = ttnn.to_layout(cos, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        if sin.layout != ttnn.TILE_LAYOUT:
            sin = ttnn.to_layout(sin, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        if len(sin.shape) == 3:
            sin = ttnn.unsqueeze(sin, dim=0)
        if len(cos.shape) == 3:
            cos = ttnn.unsqueeze(cos, dim=0)
        # Infer configuration from inputs
        batch_size, n_q_heads, seq_len, head_dim = q.shape
        batch_size2, n_k_heads, seq_len2, head_dim2 = k.shape
        assert seq_len == seq_len2, "Query and Key sequence lengths must match."
        assert batch_size == batch_size2, "Query and Key batch sizes must match."
        assert head_dim == head_dim2, "Query and Key head dimensions must match."

        # Store original dimensions
        original_head_dim = head_dim
        original_seq_len = seq_len
        rotary_dim = cos.shape[-1]

        # Handle partial rotary embedding (when cos/sin dim < head_dim)
        if rotary_dim < head_dim:
            # Split q and k into rotary and pass-through portions
            # q/k: [batch, heads, seq, head_dim] -> [batch, heads, seq, rotary_dim] + [batch, heads, seq, head_dim-rotary_dim]
            q_rot = q[:, :, :, :rotary_dim]
            q_pass = q[:, :, :, rotary_dim:]
            k_rot = k[:, :, :, :rotary_dim]
            k_pass = k[:, :, :, rotary_dim:]

            # Pad cos/sin to match tile boundaries if needed
            padded_rotary_dim = rotary_dim
            if rotary_dim % 32 != 0:
                padded_rotary_dim = ((rotary_dim + 31) // 32) * 32
                cos = ttnn.pad(cos, [1, 1, cos.shape[-2], padded_rotary_dim], [0, 0, 0, 0], 0.0)
                sin = ttnn.pad(sin, [1, 1, sin.shape[-2], padded_rotary_dim], [0, 0, 0, 0], 0.0)
                # Also pad q_rot and k_rot
                q_rot = ttnn.pad(q_rot, [batch_size, n_q_heads, seq_len, padded_rotary_dim], [0, 0, 0, 0], 0.0)
                k_rot = ttnn.pad(k_rot, [batch_size2, n_k_heads, seq_len2, padded_rotary_dim], [0, 0, 0, 0], 0.0)

            # Apply rotation to rotary portion only
            q_rot_embedded = ttnn.experimental.rotary_embedding(q_rot, cos, sin)
            k_rot_embedded = ttnn.experimental.rotary_embedding(k_rot, cos, sin)

            # Slice back to original dimensions if padding occurred
            if q_rot_embedded.shape[-2] != seq_len:
                q_rot_embedded = q_rot_embedded[:, :, :seq_len, :]
            if k_rot_embedded.shape[-2] != seq_len:
                k_rot_embedded = k_rot_embedded[:, :, :seq_len, :]
            if padded_rotary_dim != rotary_dim:
                q_rot_embedded = q_rot_embedded[:, :, :, :rotary_dim]
                k_rot_embedded = k_rot_embedded[:, :, :, :rotary_dim]

            # Concatenate rotated and pass-through portions
            q_rotated = ttnn.concat([q_rot_embedded, q_pass], dim=-1)
            k_rotated = ttnn.concat([k_rot_embedded, k_pass], dim=-1)
        else:
            # Full rotary embedding - pad if needed for tile boundaries
            if rotary_dim != head_dim:
                cos = ttnn.pad(cos, [1, 1, cos.shape[-2], head_dim], [0, 0, 0, 0], 0.0)
                sin = ttnn.pad(sin, [1, 1, sin.shape[-2], head_dim], [0, 0, 0, 0], 0.0)

            q_rotated = ttnn.experimental.rotary_embedding(q, cos, sin)
            k_rotated = ttnn.experimental.rotary_embedding(k, cos, sin)

        # Slice back to original dimensions if padding occurred
        if q_rotated.shape[-1] != original_head_dim:
            q_rotated = q_rotated[:, :, :, :original_head_dim]
        if k_rotated.shape[-1] != original_head_dim:
            k_rotated = k_rotated[:, :, :, :original_head_dim]
        if q_rotated.shape[-2] != original_seq_len:
            q_rotated = q_rotated[:, :, :original_seq_len, :]
        if k_rotated.shape[-2] != original_seq_len:
            k_rotated = k_rotated[:, :, :original_seq_len, :]

        return q_rotated, k_rotated


class TTNNDistributedRotaryPositionEmbedding(TTNNModule):
    """TTNN-accelerated Rotary Position Embedding for distributed/mesh devices.

    Uses ttnn.experimental.rotary_embedding_llama which is optimized for
    multi-device tensor-parallel scenarios.
    """

    def move_weights_to_device_impl(self):
        # Cache key based on device and mode
        self._trans_mat_cache = {}
        for is_decode in [True, False]:
            cache_key = is_decode
            if cache_key not in self._trans_mat_cache:
                # Create transformation matrix: swaps pairs and negates for rotation
                dhead = ttnn.TILE_SIZE  # Assuming head_dim is equal to tile size for optimal performance
                trans_mat = torch.zeros(1, 1, dhead, dhead)
                trans_mat[..., torch.arange(0, dhead, 2), torch.arange(1, dhead, 2)] = 1
                trans_mat[..., torch.arange(1, dhead, 2), torch.arange(0, dhead, 2)] = -1

                # Convert to device tensor
                mesh_mapper = None
                if isinstance(self.device, ttnn._ttnn.multi_device.MeshDevice):
                    mesh_mapper = ttnn.ReplicateTensorToMesh(self.device)

                trans_mat_tensor = ttnn.from_torch(
                    trans_mat,
                    device=self.device,
                    layout=ttnn.TILE_LAYOUT,
                    dtype=ttnn.bfloat16,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    mesh_mapper=mesh_mapper,
                )
                self._trans_mat_cache[cache_key] = trans_mat_tensor

    def forward(
        self,
        q: ttnn.Tensor,
        k: ttnn.Tensor,
        cos: ttnn.Tensor,
        sin: ttnn.Tensor,
    ) -> Tuple[ttnn.Tensor, ttnn.Tensor]:
        """
        Forward pass through distributed RoPE layer.

        Args:
            q: Query tensor [batch, n_heads, seq_len, head_dim] or [batch, seq_len, n_heads, head_dim]
            k: Key tensor [batch, n_kv_heads, seq_len, head_dim] or [batch, seq_len, n_kv_heads, head_dim]
            cos: Cosine position embeddings [1, 1, seq_len, rotary_dim] or [1, batch, 1, rotary_dim]
            sin: Sine position embeddings [1, 1, seq_len, rotary_dim] or [1, batch, 1, rotary_dim]

        Returns:
            Tuple of (rotated_query, rotated_key)
        """
        # Ensure tensors are in TILE_LAYOUT and bfloat16
        if q.layout != ttnn.TILE_LAYOUT:
            q = ttnn.to_layout(q, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        if k.layout != ttnn.TILE_LAYOUT:
            k = ttnn.to_layout(k, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        if cos.layout != ttnn.TILE_LAYOUT:
            cos = ttnn.to_layout(cos, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        if sin.layout != ttnn.TILE_LAYOUT:
            sin = ttnn.to_layout(sin, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        # Ensure all tensors are BFLOAT16 (required by rotary_embedding_llama)
        if q.dtype != ttnn.bfloat16:
            q = ttnn.typecast(q, ttnn.bfloat16)
        if k.dtype != ttnn.bfloat16:
            k = ttnn.typecast(k, ttnn.bfloat16)
        if cos.dtype != ttnn.bfloat16:
            cos = ttnn.typecast(cos, ttnn.bfloat16)
        if sin.dtype != ttnn.bfloat16:
            sin = ttnn.typecast(sin, ttnn.bfloat16)

        # Get device and determine decode mode
        seq_len = q.shape[2] if len(q.shape) == 4 else q.shape[1]
        is_decode_mode = False  # (seq_len == 1)

        # Get transformation matrix
        trans_mat = self._trans_mat_cache[is_decode_mode]

        # Apply rotary embedding using distributed-optimized operation
        q_rotated = ttnn.experimental.rotary_embedding_llama(
            q,
            cos,
            sin,
            trans_mat,
            is_decode_mode=is_decode_mode,
        )

        k_rotated = ttnn.experimental.rotary_embedding_llama(
            k,
            cos,
            sin,
            trans_mat,
            is_decode_mode=is_decode_mode,
        )

        return q_rotated, k_rotated


def _get_rotation_transformation_mat(dhead: int) -> torch.Tensor:
    """Generate rotation transformation matrix for RoPE: [x1, x2] -> [-x2, x1]."""
    trans_mat = torch.zeros(1, 1, dhead, dhead)
    trans_mat[..., torch.arange(0, dhead, 2), torch.arange(1, dhead, 2)] = 1
    trans_mat[..., torch.arange(1, dhead, 2), torch.arange(0, dhead, 2)] = -1
    return trans_mat


def _compute_cos_sin_cache(
    head_dim: int,
    max_seq_len: int,
    rope_theta: float,
    partial_rotary_factor: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute cos/sin cache in Meta interleaved format, identity-padded for partial rotary."""
    rotary_dim = int(head_dim * partial_rotary_factor)
    inv_freq = 1.0 / (rope_theta ** (torch.arange(0, rotary_dim, 2).float() / rotary_dim))
    t = torch.arange(max_seq_len, dtype=inv_freq.dtype)
    freqs = torch.outer(t, inv_freq)

    emb = torch.cat((freqs, freqs), dim=-1)
    cos = emb.cos()
    sin = emb.sin()

    # Interleave pairs for Meta format
    cos = cos[:, : cos.shape[1] // 2]
    cos = torch.stack((cos, cos), dim=-1).flatten(-2)
    sin = sin[:, : sin.shape[1] // 2]
    sin = torch.stack((sin, sin), dim=-1).flatten(-2)

    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)

    # Identity-pad for partial rotary (cos=1, sin=0 means pass-through)
    if partial_rotary_factor < 1.0:
        pad_width = head_dim - rotary_dim
        cos_pad = torch.ones(cos.shape[0], cos.shape[1], cos.shape[2], pad_width)
        sin_pad = torch.zeros(sin.shape[0], sin.shape[1], sin.shape[2], pad_width)
        cos = torch.cat([cos, cos_pad], dim=-1)
        sin = torch.cat([sin, sin_pad], dim=-1)

    return cos, sin


class BailingRotarySetup:
    """Pre-computed RoPE cos/sin and transformation matrices with replicated topology."""

    def __init__(
        self,
        device: Any,
        head_dim: int,
        max_seq_len: int,
        rope_theta: float,
        partial_rotary_factor: float = 1.0,
        datatype: ttnn.DataType = ttnn.bfloat16,
    ) -> None:
        """Initialize with pre-computed cos/sin cache and transformation matrices."""
        self.device = device
        self.head_dim = head_dim
        self.rotary_dim = int(head_dim * partial_rotary_factor)
        self.max_seq_len = max_seq_len
        self.partial_rotary_factor = partial_rotary_factor
        self.datatype = datatype

        self.is_mesh_device = isinstance(device, ttnn._ttnn.multi_device.MeshDevice)
        self.num_devices = device.get_num_devices() if self.is_mesh_device else 1

        cos_cache_torch, sin_cache_torch = _compute_cos_sin_cache(
            head_dim=head_dim,
            max_seq_len=max_seq_len,
            rope_theta=rope_theta,
            partial_rotary_factor=partial_rotary_factor,
        )

        mesh_mapper = ttnn.ReplicateTensorToMesh(device) if self.is_mesh_device else None

        # TILE_LAYOUT cos/sin for prefill
        self.cos_cache = ttnn.from_torch(
            cos_cache_torch.to(torch.bfloat16),
            device=device,
            layout=ttnn.TILE_LAYOUT,
            dtype=datatype,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=mesh_mapper,
        )
        self.sin_cache = ttnn.from_torch(
            sin_cache_torch.to(torch.bfloat16),
            device=device,
            layout=ttnn.TILE_LAYOUT,
            dtype=datatype,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=mesh_mapper,
        )

        # ROW_MAJOR cos/sin for decode embedding lookup
        self.cos_cache_row_major = ttnn.from_torch(
            cos_cache_torch.squeeze(0).squeeze(0).to(torch.bfloat16),
            device=device,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=datatype,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=mesh_mapper,
        )
        self.sin_cache_row_major = ttnn.from_torch(
            sin_cache_torch.squeeze(0).squeeze(0).to(torch.bfloat16),
            device=device,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=datatype,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=mesh_mapper,
        )

        # Transformation matrices (TILE_SIZE x TILE_SIZE)
        trans_mat_decode_torch = _get_rotation_transformation_mat(ttnn.TILE_SIZE)
        self.trans_mat_decode = ttnn.from_torch(
            trans_mat_decode_torch.to(torch.bfloat16),
            device=device,
            layout=ttnn.TILE_LAYOUT,
            dtype=datatype,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=mesh_mapper,
        )
        self._trans_mat_decode_torch = trans_mat_decode_torch.to(torch.bfloat16)
        self._trans_mat_decode_sharded_cache = {}

        trans_mat_prefill_torch = _get_rotation_transformation_mat(ttnn.TILE_SIZE)
        self.trans_mat_prefill = ttnn.from_torch(
            trans_mat_prefill_torch.to(torch.bfloat16),
            device=device,
            layout=ttnn.TILE_LAYOUT,
            dtype=datatype,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=mesh_mapper,
        )

    def get_cos_sin_for_prefill(
        self,
        seq_len: int,
    ) -> Tuple[ttnn.Tensor, ttnn.Tensor]:
        """Get cos/sin sliced to seq_len for prefill. Returns [1, 1, seq_len, head_dim]."""
        if seq_len > self.max_seq_len:
            raise ValueError(
                f"Requested seq_len {seq_len} exceeds max_seq_len {self.max_seq_len}. "
                f"Reinitialize BailingRotarySetup with larger max_seq_len."
            )

        cos = self.cos_cache[:, :, :seq_len, :]
        sin = self.sin_cache[:, :, :seq_len, :]

        return cos, sin

    def get_cos_sin_for_decode(
        self,
        position_ids: Union[torch.Tensor, ttnn.Tensor],
    ) -> Tuple[ttnn.Tensor, ttnn.Tensor]:
        """Get cos/sin for decode via embedding lookup. Returns [1, batch, 1, head_dim]."""
        if isinstance(position_ids, ttnn.Tensor):
            if self.is_mesh_device:
                pos_torch = ttnn.to_torch(
                    position_ids,
                    mesh_composer=ttnn.ConcatMeshToTensor(self.device, dim=0),
                )
                pos_torch = pos_torch[: position_ids.shape[0]]
            else:
                pos_torch = ttnn.to_torch(position_ids)
        else:
            pos_torch = position_ids

        if len(pos_torch.shape) == 2:
            pos_torch = pos_torch.squeeze(0)

        batch_size = pos_torch.shape[0]
        pos_indices = pos_torch.reshape(1, batch_size).to(torch.int32)
        mesh_mapper = ttnn.ReplicateTensorToMesh(self.device) if self.is_mesh_device else None

        pos_ttnn = ttnn.from_torch(
            pos_indices,
            device=self.device,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=mesh_mapper,
        )

        cos = ttnn.embedding(
            pos_ttnn,
            self.cos_cache_row_major,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        sin = ttnn.embedding(
            pos_ttnn,
            self.sin_cache_row_major,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        # Reshape [1, batch, rotary_dim] -> [1, batch, 1, rotary_dim]
        cos = ttnn.unsqueeze_to_4D(cos)
        sin = ttnn.unsqueeze_to_4D(sin)
        cos = ttnn.transpose(cos, 1, 2)
        sin = ttnn.transpose(sin, 1, 2)

        return cos, sin

    def get_trans_mat(self, is_decode: bool = False) -> ttnn.Tensor:
        """Get the RoPE transformation matrix (decode or prefill)."""
        return self.trans_mat_decode if is_decode else self.trans_mat_prefill

    def get_trans_mat_decode_sharded(self, batch_size: int) -> ttnn.Tensor:
        """Get HEIGHT_SHARDED trans_mat for decode (lazily cached per batch_size)."""
        if batch_size not in self._trans_mat_decode_sharded_cache:
            trans_mat_torch = self._trans_mat_decode_torch.repeat(1, 1, batch_size, 1)
            batch_grid = ttnn.num_cores_to_corerangeset(batch_size, self.device.compute_with_storage_grid_size(), True)
            mem_config = ttnn.create_sharded_memory_config(
                shape=(ttnn.TILE_SIZE, ttnn.TILE_SIZE),
                core_grid=batch_grid,
                strategy=ttnn.ShardStrategy.HEIGHT,
                orientation=ttnn.ShardOrientation.ROW_MAJOR,
                use_height_and_width_as_shard_shape=True,
            )
            mesh_mapper = ttnn.ReplicateTensorToMesh(self.device) if self.is_mesh_device else None
            self._trans_mat_decode_sharded_cache[batch_size] = ttnn.from_torch(
                trans_mat_torch,
                device=self.device,
                layout=ttnn.TILE_LAYOUT,
                dtype=self.datatype,
                memory_config=mem_config,
                mesh_mapper=mesh_mapper,
            )
        return self._trans_mat_decode_sharded_cache[batch_size]
