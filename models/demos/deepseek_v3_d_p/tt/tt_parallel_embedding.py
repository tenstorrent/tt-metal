# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""
TTNN implementation of Parallel Embedding for DeepSeek V3 prefill.

Sequence-parallel and tensor-parallel embedding that converts token IDs
into hidden representations distributed across a 2D mesh.

Weight Sharding:
    - Replicated across SP axis (mesh rows) — each SP device has full vocab
    - Sharded on emb_dim across TP axis (mesh columns)
    - Each device stores [vocab_size, emb_dim / tp_factor]

Forward:
    Input:  token_ids [1, 1, seq_len_per_chip] (SP-sharded by caller)
    Output: embeddings [1, 1, seq_len_per_chip, emb_dim / tp_factor] TILE_LAYOUT
"""

from pathlib import Path
from typing import Optional

import torch
from loguru import logger

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.demos.deepseek_v3_d_p.reference.deepseek_v3_config import DeepSeekV3Config


class TtParallelEmbedding(LightweightModule):
    """
    SP+TP parallel embedding for DeepSeek V3 prefill.

    No CCL operations are needed inside this module:
    - TP sharding: each device does an independent lookup on its weight slice
    - SP sharding: each device processes its own token chunk
    """

    @staticmethod
    def check_cache_complete(cache_path: Path) -> bool:
        """Check if embedding weight cache files exist."""
        if not list(cache_path.glob("embed_weight*.tensorbin")):
            logger.debug("TTNN cache missing: embed_weight")
            return False
        return True

    @staticmethod
    def _convert_and_cache_weight(
        torch_weight: torch.Tensor,
        vocab_size: int,
        emb_dim: int,
        mesh_device: ttnn.MeshDevice,
        tp_axis: int,
        dtype: ttnn.DataType,
        cache_path: Path | None,
        device: ttnn.MeshDevice | None = None,
    ):
        """
        Shared logic for converting embedding weight to ttnn with caching.

        Args:
            torch_weight: Weight tensor [vocab_size, emb_dim]
            vocab_size: Vocabulary size
            emb_dim: Embedding dimension
            mesh_device: Mesh device (for mesh_mapper)
            tp_axis: Tensor parallel axis
            dtype: Data type
            cache_path: Cache directory path
            device: None for cache-only, mesh_device for cache+load

        Returns:
            ttnn.Tensor if device is not None, else None
        """
        assert torch_weight.shape == (
            vocab_size,
            emb_dim,
        ), f"Weight shape mismatch: got {torch_weight.shape}, expected ({vocab_size}, {emb_dim})"

        shard_dims = [None, None]
        shard_dims[tp_axis] = -1  # shard emb_dim across TP axis

        mesh_mapper = ttnn.ShardTensor2dMesh(
            mesh_device,
            mesh_shape=mesh_device.shape,
            dims=tuple(shard_dims),
        )

        cache_file_name = str(cache_path / "embed_weight") if cache_path else None

        tt_weight = ttnn.as_tensor(
            torch_weight,
            mesh_mapper=mesh_mapper,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            dtype=dtype,
            memory_config=ttnn.DRAM_MEMORY_CONFIG if device else None,
            cache_file_name=cache_file_name,
        )

        if device is None:
            del tt_weight
            return None
        else:
            return tt_weight

    @staticmethod
    def build_ttnn_cache(
        torch_weight: torch.Tensor,
        vocab_size: int,
        emb_dim: int,
        mesh_device: ttnn.MeshDevice,
        cache_path: Path,
        tp_axis: int = 1,
        dtype: ttnn.DataType = ttnn.bfloat16,
    ):
        """Build TTNN cache for embedding weight without device copy."""
        TtParallelEmbedding._convert_and_cache_weight(
            torch_weight, vocab_size, emb_dim, mesh_device, tp_axis, dtype, cache_path, device=None
        )

    def __init__(
        self,
        mesh_device: ttnn.MeshDevice,
        vocab_size: int = DeepSeekV3Config.VOCAB_SIZE,
        emb_dim: int = DeepSeekV3Config.EMB_SIZE,
        torch_weight: torch.Tensor = None,
        sp_axis: int = 0,
        tp_axis: int = 1,
        dtype: ttnn.DataType = ttnn.bfloat16,
        weight_cache_path: Optional[Path] = None,
    ):
        """
        Initialize parallel embedding module.

        Args:
            mesh_device: TTNN mesh device
            vocab_size: Vocabulary size (default: 129280)
            emb_dim: Embedding dimension (default: 7168)
            torch_weight: Optional weight tensor [vocab_size, emb_dim].
                          If None, creates random weight for testing.
            sp_axis: Mesh axis for sequence parallelism (default: 0, rows)
            tp_axis: Mesh axis for tensor parallelism (default: 1, columns)
            dtype: Weight data type (default: bfloat16)
        """
        super().__init__()
        self.mesh_device = mesh_device
        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        self.sp_axis = sp_axis
        self.tp_axis = tp_axis
        self.dtype = dtype

        tp_factor = mesh_device.shape[tp_axis]
        assert emb_dim % tp_factor == 0, f"emb_dim ({emb_dim}) must be divisible by tp_factor ({tp_factor})"

        logger.debug(
            f"Initializing TtParallelEmbedding: vocab_size={vocab_size}, emb_dim={emb_dim}, "
            f"mesh_shape={mesh_device.shape}, tp_factor={tp_factor}"
        )

        self.weight_cache_path = weight_cache_path

        if torch_weight is not None:
            self.weight = self._create_weight_from_torch(torch_weight)
        else:
            self.weight = self._create_random_weight()

    def _create_weight_from_torch(self, torch_weight: torch.Tensor) -> ttnn.Tensor:
        """
        Convert torch embedding weight to TP-sharded ttnn tensor.

        Uses shared static method for conversion.

        Args:
            torch_weight: [vocab_size, emb_dim]

        Returns:
            Sharded ttnn tensor: each TP device gets [vocab_size, emb_dim / tp_factor]
        """
        # Use shared static method with device=self.mesh_device
        tt_weight = self._convert_and_cache_weight(
            torch_weight,
            self.vocab_size,
            self.emb_dim,
            self.mesh_device,
            self.tp_axis,
            self.dtype,
            self.weight_cache_path,
            device=self.mesh_device,  # Cache + load to device
        )

        logger.debug(f"Created sharded weight: {tt_weight.shape}")
        return tt_weight

    def _create_random_weight(self) -> ttnn.Tensor:
        """Create random sharded weight for testing."""
        torch_weight = torch.randn(self.vocab_size, self.emb_dim, dtype=torch.float32)
        return self._create_weight_from_torch(torch_weight)

    def forward(self, token_ids: ttnn.Tensor) -> ttnn.Tensor:
        """
        Embedding lookup producing TP-sharded output.

        Args:
            token_ids: [1, 1, seq_len_per_chip] uint32, already SP-sharded by caller.
                       seq_len_per_chip must be a multiple of TILE_SIZE (32).

        Returns:
            embeddings: [1, 1, seq_len_per_chip, emb_dim / tp_factor] TILE_LAYOUT
        """
        seq_len = token_ids.shape[-1]
        assert seq_len % ttnn.TILE_SIZE == 0, (
            f"seq_len ({seq_len}) must be a multiple of TILE_SIZE ({ttnn.TILE_SIZE}). "
            f"Pad input before calling forward."
        )

        logger.debug(f"Forward: token_ids shape={token_ids.shape}")

        embeddings = ttnn.embedding(
            token_ids,
            self.weight,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        logger.debug(f"Output: embeddings shape={embeddings.shape}")
        return embeddings
