# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
TTNN-native Rotary Position Embeddings for Molmo2 Text Model.

Implements RoPE using ttnn.experimental.rotary_embedding_llama for efficient
device-side computation without CPU roundtrips.

Based on tt_transformers/tt/rope.py patterns.
"""

from typing import List, Tuple

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule


def get_rot_transformation_mat(dhead: int = 32) -> torch.Tensor:
    """
    Generate the transformation matrix for RoPE.

    The transformation matrix is used by rotary_embedding_llama to apply
    the rotation operation efficiently.

    Args:
        dhead: Head dimension (typically 32 for tile-based ops)

    Returns:
        Transformation matrix of shape [1, 1, dhead, dhead]
    """
    # ROPE op uses a single tile
    dhead = 32
    rot_emb_matrix = torch.zeros(1, 1, dhead, dhead)
    rot_emb_matrix[..., torch.arange(0, dhead, 2), torch.arange(1, dhead, 2)] = 1
    rot_emb_matrix[..., torch.arange(1, dhead, 2), torch.arange(0, dhead, 2)] = -1
    return rot_emb_matrix


def compute_cos_sin_cache(
    head_dim: int,
    max_seq_len: int,
    theta: float = 1000000.0,
    dtype: torch.dtype = torch.float32,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute cos/sin cache for RoPE.

    Uses the standard RoPE formula from the Llama family.

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
    freqs = torch.outer(t, inv_freq)

    # Create [cos, cos] and [sin, sin] format for efficient computation
    emb = torch.cat((freqs, freqs), dim=-1)
    cos = emb.cos()
    sin = emb.sin()

    # Permute to meta format: interleave the values
    # Original: [pos, dim/2] -> [pos, dim] with interleaved
    cos_half = cos[:, : head_dim // 2]
    cos = torch.stack((cos_half, cos_half), dim=-1).flatten(-2)

    sin_half = sin[:, : head_dim // 2]
    sin = torch.stack((sin_half, sin_half), dim=-1).flatten(-2)

    # Reshape to [1, 1, max_seq_len, head_dim]
    cos = cos.unsqueeze(0).unsqueeze(0).to(dtype)
    sin = sin.unsqueeze(0).unsqueeze(0).to(dtype)

    return cos, sin


class TextRotarySetup(LightweightModule):
    """
    TTNN-native rotary position embeddings setup for Molmo2.

    Pre-computes cos/sin matrices and transformation matrices on device
    for efficient RoPE computation using ttnn.experimental.rotary_embedding_llama.
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

        # Create cos/sin matrices for decode mode (ROW_MAJOR for embedding lookup)
        self.cos_matrix = ttnn.from_torch(
            cos_cache.squeeze(0),  # [1, max_seq_len, head_dim]
            device=mesh_device,
            dtype=datatype,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=self.mesh_mapper,
        )

        self.sin_matrix = ttnn.from_torch(
            sin_cache.squeeze(0),  # [1, max_seq_len, head_dim]
            device=mesh_device,
            dtype=datatype,
            layout=ttnn.ROW_MAJOR_LAYOUT,
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

        # Create transformation matrices
        trans_mat_decode = get_rot_transformation_mat(dhead=ttnn.TILE_SIZE).repeat(1, 1, batch_size, 1)

        # Decode transformation matrix with sharding
        core_grid = ttnn.CoreCoord(8, 8)  # Standard grid for Wormhole
        batch_grid = ttnn.num_cores_to_corerangeset(batch_size, core_grid, row_wise=True)
        trans_mat_mem_config = ttnn.create_sharded_memory_config(
            shape=(ttnn.TILE_SIZE, ttnn.TILE_SIZE),
            core_grid=batch_grid,
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )

        self.transformation_mat_decode = ttnn.from_torch(
            trans_mat_decode,
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=datatype,
            memory_config=trans_mat_mem_config,
            mesh_mapper=self.mesh_mapper,
        )

        # Prefill transformation matrix (DRAM, full head_dim)
        trans_mat_prefill = get_rot_transformation_mat(dhead=head_dim)
        self.transformation_mat_prefill = ttnn.from_torch(
            trans_mat_prefill,
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=datatype,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=self.mesh_mapper,
        )

    def get_transformation_mats(self) -> dict:
        """
        Get both decode and prefill transformation matrices.

        Returns:
            Dict with 'decode' and 'prefill' transformation matrices
        """
        return {
            "decode": self.transformation_mat_decode,
            "prefill": self.transformation_mat_prefill,
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
        Get rotation matrices for decode mode.

        Args:
            position_idxs: Position indices tensor of shape [batch]

        Returns:
            List of [cos, sin] tensors for decode
        """
        batch = position_idxs.shape[0]
        position_idxs = position_idxs.reshape(1, batch)

        # Pad to multiple of 32
        pad_size = ((batch + 31) // 32) * 32 - batch
        if pad_size > 0:
            position_idxs = torch.nn.functional.pad(position_idxs, (0, pad_size), "constant", 0)

        rot_idxs = ttnn.from_torch(
            position_idxs,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=self.mesh_mapper,
        )

        # Embedding lookup for cos/sin
        cos = ttnn.embedding(
            rot_idxs,
            self.cos_matrix,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )  # [1, batch, head_dim]

        sin = ttnn.embedding(
            rot_idxs,
            self.sin_matrix,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )  # [1, batch, head_dim]

        ttnn.deallocate(rot_idxs)

        # Reshape for decode: [1, batch, 1, head_dim]
        cos = ttnn.unsqueeze_to_4D(cos)  # [1, 1, batch, head_dim]
        sin = ttnn.unsqueeze_to_4D(sin)

        cos = ttnn.transpose(cos, 1, 2)  # [1, batch, 1, head_dim]
        sin = ttnn.transpose(sin, 1, 2)

        # Shard to cores
        core_grid = ttnn.CoreCoord(8, 8)
        batch_grid = ttnn.num_cores_to_corerangeset(self.batch_size, core_grid, row_wise=True)
        mem_config = ttnn.create_sharded_memory_config(
            shape=(ttnn.TILE_SIZE, self.head_dim),
            core_grid=batch_grid,
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )

        if self.batch_size % ttnn.TILE_SIZE != 0:
            cos = cos[:, : self.batch_size, :, :]
            sin = sin[:, : self.batch_size, :, :]

        cos = ttnn.interleaved_to_sharded(cos, mem_config)
        sin = ttnn.interleaved_to_sharded(sin, mem_config)

        return [cos, sin]
