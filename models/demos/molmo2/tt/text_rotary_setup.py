# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
TTNN-native Rotary Position Embeddings for Molmo2 Text Model.

Implements half-span RoPE using ttnn.experimental.rotary_embedding for
device-side computation (no transformation matrix).

Based on tt_transformers/tt/rope.py patterns.
"""

from typing import List, Optional, Tuple

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule


def compute_cos_sin_cache(
    head_dim: int,
    max_seq_len: int,
    theta: float = 1000000.0,
    dtype: torch.dtype = torch.float32,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute cos/sin cache for RoPE in HuggingFace format.

    Uses HF-style RoPE format for ttnn.experimental.rotary_embedding:
    [c0, c1, ..., c_{d/2-1}, c0, c1, ..., c_{d/2-1}]

    Args:
        head_dim: Dimension per head
        max_seq_len: Maximum sequence length
        theta: RoPE theta parameter
        dtype: Data type for output tensors

    Returns:
        Tuple of (cos_cache, sin_cache) each of shape [1, 1, max_seq_len, head_dim]
    """
    # Compute inverse frequencies
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))

    # Compute position embeddings
    t = torch.arange(max_seq_len, dtype=inv_freq.dtype)
    freqs = torch.outer(t, inv_freq)  # [max_seq_len, head_dim // 2]

    # HF format: concat freqs with itself
    # [c0, c1, ..., c_{d/2-1}, c0, c1, ..., c_{d/2-1}]
    cos = torch.cat([freqs.cos(), freqs.cos()], dim=-1)  # [max_seq_len, head_dim]
    sin = torch.cat([freqs.sin(), freqs.sin()], dim=-1)  # [max_seq_len, head_dim]

    # Reshape to [1, 1, max_seq_len, head_dim]
    cos = cos.unsqueeze(0).unsqueeze(0).to(dtype)
    sin = sin.unsqueeze(0).unsqueeze(0).to(dtype)

    return cos, sin


class TextRotarySetup(LightweightModule):
    """
    TTNN-native rotary position embeddings setup for Molmo2.

    Pre-computes cos/sin matrices on device for half-span RoPE
    using ttnn.experimental.rotary_embedding (no transformation matrix).
    """

    def __init__(
        self,
        mesh_device,
        head_dim: int = 128,
        max_seq_len: int = 8192,
        rope_theta: float = 1000000.0,
        batch_size: int = 1,
        datatype: ttnn.DataType = ttnn.bfloat16,
    ):
        """
        Initialize TextRotarySetup.

        Args:
            mesh_device: TTNN mesh device or single device
            head_dim: Dimension per head (128 for Molmo2)
            max_seq_len: Maximum sequence length
            rope_theta: RoPE theta parameter (1,000,000 for Molmo2)
            batch_size: Batch size
            datatype: Data type for tensors
        """
        super().__init__()

        self.mesh_device = mesh_device
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.rope_theta = rope_theta
        self.batch_size = batch_size
        self.datatype = datatype

        self.is_mesh_device = mesh_device.__class__.__name__ == "MeshDevice"
        self.mesh_mapper = ttnn.ReplicateTensorToMesh(mesh_device) if self.is_mesh_device else None

        # Compute cos/sin cache
        cos_cache, sin_cache = compute_cos_sin_cache(
            head_dim=head_dim,
            max_seq_len=max_seq_len,
            theta=rope_theta,
        )

        # Create cos/sin matrices for decode mode (TILE_LAYOUT for ttnn.experimental.rotary_embedding)
        # ttnn.experimental.rotary_embedding expects the FULL cache in shape [1, 1, max_seq_len, head_dim]
        # and performs position slicing internally using the token_index parameter
        self.cos_matrix = ttnn.from_torch(
            cos_cache,  # [1, 1, max_seq_len, head_dim]
            device=mesh_device,
            dtype=datatype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=self.mesh_mapper,
        )

        self.sin_matrix = ttnn.from_torch(
            sin_cache,  # [1, 1, max_seq_len, head_dim]
            device=mesh_device,
            dtype=datatype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=self.mesh_mapper,
        )

        # Create cos/sin matrices for prefill mode (TILE_LAYOUT)
        self.cos_matrix_prefill = ttnn.from_torch(
            cos_cache.squeeze(0),  # [1, max_seq_len, head_dim]
            device=mesh_device,
            dtype=datatype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=self.mesh_mapper,
        )

        self.sin_matrix_prefill = ttnn.from_torch(
            sin_cache.squeeze(0),  # [1, max_seq_len, head_dim]
            device=mesh_device,
            dtype=datatype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=self.mesh_mapper,
        )

        # Decode: embedding outputs from get_rot_mats_decode_traced; deallocate before next lookup
        self._decode_rot_cos_last: Optional[ttnn.Tensor] = None
        self._decode_rot_sin_last: Optional[ttnn.Tensor] = None

        # Half-span RoPE uses ttnn.experimental.rotary_embedding (no transformation matrix).
        # Return None for backward-compat callers that still pass transformation_mats.
        self._transformation_mat_decode = None
        self._transformation_mat_prefill = None

    def get_transformation_mats(self) -> dict:
        """
        Return placeholder dict for API compatibility. Half-span RoPE does not use transformation matrices.
        """
        return {
            "decode": self._transformation_mat_decode,
            "prefill": self._transformation_mat_prefill,
        }

    def get_rot_mats_prefill(
        self,
        seq_len: int,
        start_pos: int = 0,
    ) -> List[ttnn.Tensor]:
        """
        Get rotation matrices for prefill mode.

        Args:
            seq_len: Sequence length
            start_pos: Starting position

        Returns:
            List of [cos, sin] tensors for the sequence
        """
        # Slice the pre-computed matrices
        # cos_matrix_prefill shape: [1, max_seq_len, head_dim]
        end_pos = start_pos + seq_len

        # Create position indices tensor
        positions = torch.arange(start_pos, end_pos, dtype=torch.int32).unsqueeze(0)
        pos_ttnn = ttnn.from_torch(
            positions,
            device=self.mesh_device,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=self.mesh_mapper,
        )

        # Use embedding lookup to get cos/sin for positions
        cos = ttnn.embedding(
            pos_ttnn,
            self.cos_matrix_prefill,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )  # [1, seq_len, head_dim]

        sin = ttnn.embedding(
            pos_ttnn,
            self.sin_matrix_prefill,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )  # [1, seq_len, head_dim]

        ttnn.deallocate(pos_ttnn)

        # Reshape to [1, 1, seq_len, head_dim]
        cos = ttnn.unsqueeze_to_4D(cos)
        sin = ttnn.unsqueeze_to_4D(sin)

        return [cos, sin]

    def get_rot_mats_prefill_on_device(
        self,
        seq_len: int,
        start_pos: int = 0,
        pos_tensor: ttnn.Tensor = None,
        cos_out: ttnn.Tensor = None,
        sin_out: ttnn.Tensor = None,
    ) -> List[ttnn.Tensor]:
        """
        Get rotation matrices for prefill mode using pre-allocated tensors.

        For tracing support - uses pre-allocated position and output tensors.

        Args:
            seq_len: Sequence length
            start_pos: Starting position
            pos_tensor: Pre-allocated position tensor (will be updated)
            cos_out: Pre-allocated cos output tensor (optional)
            sin_out: Pre-allocated sin output tensor (optional)

        Returns:
            List of [cos, sin] tensors for the sequence
        """
        # Update position tensor if provided
        if pos_tensor is not None:
            positions = torch.arange(start_pos, start_pos + seq_len, dtype=torch.int32).unsqueeze(0)
            pos_new = ttnn.from_torch(
                positions,
                device=self.mesh_device,
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=self.mesh_mapper,
            )
            ttnn.copy(pos_new, pos_tensor)
            ttnn.deallocate(pos_new)

        # Use embedding lookup
        cos = ttnn.embedding(
            pos_tensor,
            self.cos_matrix_prefill,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        sin = ttnn.embedding(
            pos_tensor,
            self.sin_matrix_prefill,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        # Reshape to [1, 1, seq_len, head_dim]
        cos = ttnn.unsqueeze_to_4D(cos)
        sin = ttnn.unsqueeze_to_4D(sin)

        return [cos, sin]

    def allocate_prefill_tensors(
        self,
        seq_len: int,
    ) -> dict:
        """
        Allocate tensors needed for traced prefill.

        Args:
            seq_len: Sequence length

        Returns:
            Dict with pre-allocated tensors
        """
        # Allocate position tensor
        positions = torch.arange(0, seq_len, dtype=torch.int32).unsqueeze(0)
        pos_tensor = ttnn.from_torch(
            positions,
            device=self.mesh_device,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=self.mesh_mapper,
        )

        return {
            "pos_tensor": pos_tensor,
            "seq_len": seq_len,
        }

    def get_rot_mats_decode(
        self,
        position_idxs: torch.Tensor,
    ) -> List[ttnn.Tensor]:
        """
        Get rotation matrices for decode mode (host-side position).

        Uses the same on-device embedding lookup as get_rot_mats_decode_traced so attention
        can call rotary_embedding without a host read of the position scalar.

        Args:
            position_idxs: Scalar or 1-D tensor of current position(s) (batch 1 uses first element)

        Returns:
            List of [cos, sin] tensors shaped [1, 1, padded_batch, head_dim] for rotary_embedding
        """
        batch = self.batch_size
        pad_size = ((batch + 31) // 32) * 32 - batch
        pos_val = int(position_idxs.reshape(-1)[0].item())
        idx = torch.full((1, batch + pad_size), pos_val, dtype=torch.int32)
        rot_ttnn = ttnn.from_torch(
            idx,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=self.mesh_mapper,
        )
        mats = self.get_rot_mats_decode_traced(rot_ttnn)
        ttnn.deallocate(rot_ttnn)
        return mats

    def get_rot_mats_decode_traced(
        self,
        rot_idxs: ttnn.Tensor,
    ) -> List[ttnn.Tensor]:
        """
        Get cos/sin for decode via on-device embedding lookup into the RoPE table.

        rotary_embedding is then called without token_index (prefill-style), so no host read
        of current_pos is needed during trace capture.

        Args:
            rot_idxs: Device tensor [1, batch_padded] of sequence positions (uint32), same layout
                as allocate_decode_rot_idxs.

        Returns:
            List of [cos, sin] tensors shaped [1, 1, batch_padded, head_dim]
        """
        if self._decode_rot_cos_last is not None:
            ttnn.deallocate(self._decode_rot_cos_last)
            ttnn.deallocate(self._decode_rot_sin_last)
            self._decode_rot_cos_last = None
            self._decode_rot_sin_last = None

        cos = ttnn.embedding(
            rot_idxs,
            self.cos_matrix_prefill,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        sin = ttnn.embedding(
            rot_idxs,
            self.sin_matrix_prefill,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        cos = ttnn.unsqueeze_to_4D(cos)
        sin = ttnn.unsqueeze_to_4D(sin)

        self._decode_rot_cos_last = cos
        self._decode_rot_sin_last = sin
        return [cos, sin]

    def allocate_decode_rot_idxs(self, initial_pos: int = 0) -> ttnn.Tensor:
        """
        Allocate position index tensor for traced decode.

        Args:
            initial_pos: Initial position value

        Returns:
            Pre-allocated position index tensor on device
        """
        # Pad to multiple of 32
        batch = self.batch_size
        pad_size = ((batch + 31) // 32) * 32 - batch
        position_idxs = torch.full((1, batch + pad_size), initial_pos, dtype=torch.int32)

        return ttnn.from_torch(
            position_idxs,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=self.mesh_mapper,
        )
