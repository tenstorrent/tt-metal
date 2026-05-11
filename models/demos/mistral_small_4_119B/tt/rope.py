# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""First-class TT RoPE module for Mistral Small 4.

Provides :class:`TtMistral4Rotary` — a stateful class that holds
device-side cos/sin lookup tables and the transformation matrix,
and exposes helpers to convert position indices into the
``rope_tensors`` dict consumed by ``MistralSmall4MLA1D.forward_decode``
/ ``forward_prefill``.

Usage::

    rotary = TtMistral4Rotary(mesh_device, batch_size_per_row, hf_config)
    rope_tensors = rotary.get_rot_mats(position_ids)
    # pass rope_tensors to DecoderBlock2D.forward_decode / text_stack forward
"""

from __future__ import annotations

import torch
from transformers.models.mistral4.configuration_mistral4 import Mistral4Config

import ttnn
from models.demos.mistral_small_4_119B.tt_utils.config_helpers import find_largest_divisor


def _get_rot_transformation_mat() -> torch.Tensor:
    """32×32 rotation matrix used by ``ttnn.experimental.rotary_embedding_llama``."""
    dhead = 32
    mat = torch.zeros(1, 1, dhead, dhead)
    mat[..., torch.arange(0, dhead, 2), torch.arange(1, dhead, 2)] = 1
    mat[..., torch.arange(1, dhead, 2), torch.arange(0, dhead, 2)] = -1
    return mat


def get_cos_sin_matrix(hf_config: Mistral4Config) -> tuple[torch.Tensor, torch.Tensor]:
    """Build ``[1, 1, max_seq_len, dim]`` cos/sin in Meta-interleaved format.

    HuggingFace returns cos/sin in ``[r, r, …, i, i, …]`` order (halved dim).
    The ``rotary_embedding_llama`` op expects Meta-style ``[r, i, r, i, …]`` (interleaved),
    so we un-halve and re-interleave here.

    Returns:
        (cos, sin) each of shape ``[1, 1, max_seq_len, qk_rope_head_dim]``.
    """
    from transformers.models.mistral4.modeling_mistral4 import Mistral4RotaryEmbedding

    max_seq_len = getattr(hf_config, "max_seq_len", hf_config.max_position_embeddings)
    rope_dim = hf_config.qk_rope_head_dim

    rope = Mistral4RotaryEmbedding(hf_config)
    pos_ids = torch.arange(max_seq_len, dtype=torch.long).unsqueeze(0)
    dummy = torch.zeros(1, max_seq_len, hf_config.hidden_size, dtype=torch.bfloat16)
    cos_hf, sin_hf = rope(dummy, position_ids=pos_ids)

    cos_hf = cos_hf.squeeze(0).float()
    sin_hf = sin_hf.squeeze(0).float()

    half = rope_dim // 2
    cos = cos_hf[:, :half]
    cos = torch.stack((cos, cos), dim=-1).flatten(-2)
    sin = sin_hf[:, :half]
    sin = torch.stack((sin, sin), dim=-1).flatten(-2)

    return cos.unsqueeze(0).unsqueeze(0), sin.unsqueeze(0).unsqueeze(0)


class TtMistral4Rotary:
    """Stateful RoPE setup for Mistral Small 4 on TT devices.

    Holds device-side cos/sin lookup tables and transformation matrix.
    Call :meth:`get_rot_mats` per decode step with current position ids
    to produce the ``rope_tensors`` dict.
    """

    def __init__(
        self,
        mesh_device: ttnn.MeshDevice,
        batch_size_per_row: int,
        hf_config: Mistral4Config,
    ):
        self.mesh_device = mesh_device
        self.batch_size_per_row = batch_size_per_row
        self.hf_config = hf_config
        self.dim = hf_config.qk_rope_head_dim

        self.core_grid = mesh_device.compute_with_storage_grid_size()

        # ── Cos/Sin lookup tables (full sequence length) ─────────────
        cos_torch, sin_torch = get_cos_sin_matrix(hf_config)
        self.cos_matrix = ttnn.from_torch(
            cos_torch,
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )
        self.sin_matrix = ttnn.from_torch(
            sin_torch,
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )

        # ── Batch grid for HEIGHT-sharded decode tensors ─────────────
        num_cores = find_largest_divisor(batch_size_per_row, self.core_grid.x * self.core_grid.y)
        self.batch_grid = ttnn.num_cores_to_corerangeset(num_cores, self.core_grid, row_wise=True)

        # ── Transformation matrix (decode: repeated per batch) ───────
        trans_mat = _get_rot_transformation_mat().repeat(1, 1, batch_size_per_row, 1)
        trans_mat_mem = ttnn.create_sharded_memory_config(
            shape=(ttnn.TILE_SIZE, ttnn.TILE_SIZE),
            core_grid=self.batch_grid,
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
        self.transformation_mat = ttnn.from_torch(
            trans_mat,
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
            memory_config=trans_mat_mem,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )

        # ── Transformation matrix (prefill: single tile, DRAM) ───────
        self.transformation_mat_prefill = ttnn.from_torch(
            _get_rot_transformation_mat(),
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )

    # ── Position index helpers ───────────────────────────────────────

    def _position_idxs_to_tensor(
        self,
        position_idxs: torch.Tensor,
        *,
        dtype: ttnn.DataType,
        on_host: bool = False,
    ) -> ttnn.Tensor:
        """Convert 1-D position ids ``[batch]`` to device tensor with mesh sharding."""
        assert isinstance(position_idxs, torch.Tensor) and position_idxs.ndim == 1

        pos = position_idxs.clamp_min(0).unsqueeze(0)  # [1, batch]
        interleaved = pos.shape[1] == self.batch_size_per_row

        pad_size = ttnn.core.roundup(pos.shape[1], ttnn.TILE_SIZE) - pos.shape[1]
        pos = torch.nn.functional.pad(pos, (0, pad_size), "constant", 0)

        return ttnn.as_tensor(
            pos,
            dtype=dtype,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.ShardTensor2dMesh(
                self.mesh_device,
                dims=(None, None) if interleaved else (1, None),
                mesh_shape=tuple(self.mesh_device.shape),
            ),
            device=None if on_host else self.mesh_device,
            memory_config=None if on_host else ttnn.DRAM_MEMORY_CONFIG,
        )

    def get_position_idxs_tensor(self, position_idxs: torch.Tensor, on_host: bool = False) -> ttnn.Tensor:
        """Position ids as int32 tensor (for paged cache / attention mask)."""
        return self._position_idxs_to_tensor(position_idxs, dtype=ttnn.int32, on_host=on_host)

    def get_rot_idxs(self, position_idxs: torch.Tensor, on_host: bool = False) -> ttnn.Tensor:
        """Position ids as uint32 tensor (for embedding lookup)."""
        return self._position_idxs_to_tensor(position_idxs, dtype=ttnn.uint32, on_host=on_host)

    # ── Rotation matrices (decode) ───────────────────────────────────

    def get_rot_mats(
        self,
        position_idxs: torch.Tensor | ttnn.Tensor,
        return_rot_idxs: bool = False,
    ) -> dict[str, ttnn.Tensor] | tuple[dict[str, ttnn.Tensor], ttnn.Tensor]:
        """Produce ``rope_tensors`` dict for decode from position ids.

        Args:
            position_idxs: 1-D torch tensor ``[batch]`` or pre-computed ttnn rot_idxs ``[1, batch]``.
            return_rot_idxs: If True, also return the device-side rot_idxs tensor.

        Returns:
            ``{"cos_matrix": ..., "sin_matrix": ..., "trans_matrix": ...}``
        """
        if isinstance(position_idxs, torch.Tensor):
            rot_idxs = self.get_rot_idxs(position_idxs)
        else:
            rot_idxs = position_idxs
            assert len(rot_idxs.shape) == 2 and rot_idxs.shape[0] == 1

        if rot_idxs.device() != self.mesh_device:
            rot_idxs = ttnn.to_device(rot_idxs, self.mesh_device, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        cos = ttnn.embedding(rot_idxs, self.cos_matrix, layout=ttnn.TILE_LAYOUT)
        sin = ttnn.embedding(rot_idxs, self.sin_matrix, layout=ttnn.TILE_LAYOUT)

        cos = ttnn.unsqueeze_to_4D(cos)
        sin = ttnn.unsqueeze_to_4D(sin)

        cos = ttnn.transpose(cos, 1, 2)
        sin = ttnn.transpose(sin, 1, 2)

        if self.batch_size_per_row % ttnn.TILE_SIZE != 0:
            cos = cos[:, : self.batch_size_per_row, :, :]
            sin = sin[:, : self.batch_size_per_row, :, :]

        mem_config = ttnn.create_sharded_memory_config(
            shape=(ttnn.TILE_SIZE, self.dim),
            core_grid=self.batch_grid,
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
        cos = ttnn.to_memory_config(cos, mem_config)
        sin = ttnn.to_memory_config(sin, mem_config)

        result = {"cos_matrix": cos, "sin_matrix": sin, "trans_matrix": self.transformation_mat}
        if return_rot_idxs:
            return result, rot_idxs
        return result

    def get_rot_mats_from_rot_idxs(
        self,
        rot_idxs: ttnn.Tensor,
        return_rot_idxs: bool = False,
    ) -> dict[str, ttnn.Tensor] | tuple[dict[str, ttnn.Tensor], ttnn.Tensor]:
        """Pure-ttnn version of :meth:`get_rot_mats` for trace capture (no torch ops).

        Args:
            rot_idxs: Pre-computed ``[1, batch]`` uint32 tensor on device.
        """
        assert isinstance(rot_idxs, ttnn.Tensor)
        assert len(rot_idxs.shape) == 2 and rot_idxs.shape[0] == 1

        cos = ttnn.embedding(rot_idxs, self.cos_matrix, layout=ttnn.TILE_LAYOUT)
        sin = ttnn.embedding(rot_idxs, self.sin_matrix, layout=ttnn.TILE_LAYOUT)

        cos = ttnn.unsqueeze_to_4D(cos)
        sin = ttnn.unsqueeze_to_4D(sin)

        cos = ttnn.transpose(cos, 1, 2)
        sin = ttnn.transpose(sin, 1, 2)

        if self.batch_size_per_row % ttnn.TILE_SIZE != 0:
            cos = cos[:, : self.batch_size_per_row, :, :]
            sin = sin[:, : self.batch_size_per_row, :, :]

        mem_config = ttnn.create_sharded_memory_config(
            shape=(ttnn.TILE_SIZE, self.dim),
            core_grid=self.batch_grid,
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
        cos = ttnn.to_memory_config(cos, mem_config)
        sin = ttnn.to_memory_config(sin, mem_config)

        result = {"cos_matrix": cos, "sin_matrix": sin, "trans_matrix": self.transformation_mat}
        if return_rot_idxs:
            return result, rot_idxs
        return result

    # ── Prefill rotation tables ──────────────────────────────────────

    def get_rot_mats_prefill(self, seq_len: int | None = None) -> dict[str, ttnn.Tensor]:
        """Return cos/sin tables + transformation matrix for prefill mode.

        Args:
            seq_len: If given, trim tables to this length (must be ≤ max_position_embeddings).
        """
        max_seq = getattr(self.hf_config, "max_seq_len", self.hf_config.max_position_embeddings)
        if seq_len is not None:
            assert seq_len <= max_seq, f"seq_len {seq_len} > max {max_seq}"
            cos_torch, sin_torch = get_cos_sin_matrix(self.hf_config)
            cos_torch = cos_torch[..., :seq_len, :]
            sin_torch = sin_torch[..., :seq_len, :]
            cos = ttnn.from_torch(
                cos_torch,
                device=self.mesh_device,
                layout=ttnn.TILE_LAYOUT,
                dtype=ttnn.bfloat16,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
            )
            sin = ttnn.from_torch(
                sin_torch,
                device=self.mesh_device,
                layout=ttnn.TILE_LAYOUT,
                dtype=ttnn.bfloat16,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
            )
        else:
            cos = self.cos_matrix
            sin = self.sin_matrix

        return {"cos_matrix": cos, "sin_matrix": sin, "trans_matrix": self.transformation_mat_prefill}
