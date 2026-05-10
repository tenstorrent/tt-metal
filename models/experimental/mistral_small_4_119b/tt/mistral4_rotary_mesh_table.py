# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
On-mesh RoPE cos/sin tables for Mistral4 text (bring-up).

HF :class:`transformers.models.mistral4.modeling_mistral4.Mistral4RotaryEmbedding` is run
**once** at init to fill ``(num_positions, rope_dim)`` rows; weights live in DRAM and
prefill gathers per ``position_ids`` via :func:`ttnn.embedding` — no large cos/sin
``from_torch`` each forward.

``num_positions`` is capped by ``text_config.max_position_embeddings``; the default is
8192 so CPU table build and mesh DRAM stay modest — raise it for longer supported contexts.
Every index in ``position_ids`` must satisfy ``0 <= id < num_positions``.
"""

from __future__ import annotations

import torch
import ttnn

from models.common.lightweightmodule import LightweightModule


class TtMistral4RotaryEmbeddingMeshTable(LightweightModule):
    """
    Replicated TILE embedding tables ``(num_positions, rope_dim)`` for cos and sin.
    """

    def __init__(
        self,
        mesh_device,
        text_config,
        *,
        num_positions: int = 8192,
    ):
        super().__init__()
        try:
            from transformers.models.mistral4.modeling_mistral4 import Mistral4RotaryEmbedding
        except ImportError as exc:
            raise ImportError("TtMistral4RotaryEmbeddingMeshTable requires transformers with Mistral4.") from exc

        max_pos = int(getattr(text_config, "max_position_embeddings", 0) or 0)
        if max_pos <= 0:
            max_pos = num_positions
        self.num_positions = min(int(num_positions), max_pos)
        if self.num_positions < 1:
            raise ValueError("num_positions must be >= 1")

        hidden = int(getattr(text_config, "hidden_size", 0) or 0)
        if hidden <= 0:
            raise ValueError("text_config.hidden_size is required")

        x_dummy = torch.zeros(1, 1, hidden, dtype=torch.bfloat16)
        pos = torch.arange(self.num_positions, dtype=torch.long).unsqueeze(0)
        rotary = Mistral4RotaryEmbedding(text_config).eval().to(torch.bfloat16)
        with torch.no_grad():
            cos, sin = rotary(x_dummy, pos)
        cos = cos.to(dtype=torch.bfloat16).contiguous()
        sin = sin.to(dtype=torch.bfloat16).contiguous()
        if cos.dim() != 3 or cos.shape[0] != 1:
            raise ValueError(f"Unexpected HF cos shape {tuple(cos.shape)}")
        cos_2d = cos.squeeze(0).contiguous()
        sin_2d = sin.squeeze(0).contiguous()
        self.rope_dim = int(cos_2d.shape[-1])

        mapper = ttnn.ReplicateTensorToMesh(mesh_device=mesh_device)
        self.mesh_device = mesh_device
        self.cos_weight = ttnn.from_torch(
            cos_2d,
            device=mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=mapper,
        )
        self.sin_weight = ttnn.from_torch(
            sin_2d,
            device=mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=mapper,
        )

    def gather(self, position_ids: torch.Tensor) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        """
        Return cos/sin as ``[B, 1, S, rope_dim]`` TILE tensors for :func:`_apply_rotary_ttnn`.

        Caller must deallocate both tensors after the RoPE consumer finishes.
        """
        position_ids = position_ids.long().contiguous()
        if position_ids.dim() != 2:
            raise ValueError(f"position_ids must be rank-2 [batch, seq], got {tuple(position_ids.shape)}")
        if torch.any(position_ids < 0) or torch.any(position_ids >= self.num_positions):
            raise ValueError(
                f"position_ids must be in [0, {self.num_positions}); "
                f"got min={int(position_ids.min())} max={int(position_ids.max())}"
            )

        mapper = ttnn.ReplicateTensorToMesh(mesh_device=self.mesh_device)
        ids_tt = ttnn.from_torch(
            position_ids,
            device=self.mesh_device,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=mapper,
        )
        cos_flat = ttnn.embedding(
            ids_tt,
            self.cos_weight,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        sin_flat = ttnn.embedding(
            ids_tt,
            self.sin_weight,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ids_tt)

        b, s, d = int(cos_flat.shape[0]), int(cos_flat.shape[1]), int(cos_flat.shape[2])
        cos_4d = ttnn.reshape(cos_flat, (b, 1, s, d))
        sin_4d = ttnn.reshape(sin_flat, (b, 1, s, d))
        return cos_4d, sin_4d
