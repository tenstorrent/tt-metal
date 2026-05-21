# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.experimental.devstarl2_small.devstral_utils.pixtral_seq_chunk import vision_rope_memcfg
from models.tt_transformers.tt.common import precompute_mistral_vision_freqs


class TtPixtralRotaryEmbedding(LightweightModule):
    """TT version of HF ``PixtralRotaryEmbedding`` for vision position ids."""

    def __init__(self, mesh_device, config, datatype=ttnn.bfloat16):
        super().__init__()
        self.mesh_device = mesh_device
        self.config = config
        self.datatype = datatype
        self.is_mesh_device = isinstance(mesh_device, ttnn._ttnn.multi_device.MeshDevice)

        rope_type = config.rope_parameters["rope_type"]
        if rope_type != "default":
            raise ValueError(f"TtPixtralRotaryEmbedding supports rope_type='default' only, got {rope_type!r}.")

        image_size = config.image_size[0] if isinstance(config.image_size, (tuple, list)) else int(config.image_size)
        patch_size = config.patch_size[0] if isinstance(config.patch_size, (tuple, list)) else int(config.patch_size)
        max_patches_per_side = image_size // patch_size

        head_dim = getattr(config, "head_dim", None) or (config.hidden_size // config.num_attention_heads)
        rope_theta = float(config.rope_parameters["rope_theta"])
        cos_torch, sin_torch = precompute_mistral_vision_freqs(
            dim=int(head_dim),
            max_patches_per_side=int(max_patches_per_side),
            theta=rope_theta,
            scale_factor=None,
            orig_context_len=None,
        )

        mapper = ttnn.ReplicateTensorToMesh(mesh_device) if self.is_mesh_device else None
        max_table_rows = int(max_patches_per_side) * int(max_patches_per_side)
        rope_mem_config = vision_rope_memcfg(max_table_rows, int(head_dim))
        self.cos_matrix = ttnn.from_torch(
            cos_torch,
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=datatype,
            memory_config=rope_mem_config,
            mesh_mapper=mapper,
        )
        self.sin_matrix = ttnn.from_torch(
            sin_torch,
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=datatype,
            memory_config=rope_mem_config,
            mesh_mapper=mapper,
        )

    def _to_position_indices(self, position_ids):
        if len(position_ids.shape) == 1:
            return ttnn.reshape(position_ids, (1, position_ids.shape[0]))
        return position_ids

    def forward(self, x: ttnn.Tensor, position_ids):
        position_ids = self._to_position_indices(position_ids)
        seq_len = int(x.shape[-2])
        head_dim = int(
            getattr(self.config, "head_dim", None) or (self.config.hidden_size // self.config.num_attention_heads)
        )
        rope_mem_cfg = vision_rope_memcfg(seq_len, head_dim)
        if position_ids.memory_config().buffer_type != rope_mem_cfg.buffer_type:
            position_ids = ttnn.to_memory_config(position_ids, rope_mem_cfg)
        cos = ttnn.embedding(position_ids, self.cos_matrix, layout=ttnn.TILE_LAYOUT, memory_config=rope_mem_cfg)
        sin = ttnn.embedding(position_ids, self.sin_matrix, layout=ttnn.TILE_LAYOUT, memory_config=rope_mem_cfg)

        if x.dtype != cos.dtype:
            cos = ttnn.typecast(cos, dtype=x.dtype)
            sin = ttnn.typecast(sin, dtype=x.dtype)
        return cos, sin


__all__ = ["TtPixtralRotaryEmbedding"]
