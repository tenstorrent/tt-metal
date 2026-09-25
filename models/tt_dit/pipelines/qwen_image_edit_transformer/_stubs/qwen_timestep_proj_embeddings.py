# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Native TTNN port of the QwenImage timestep conditioning (`time_text_embed`,
QwenTimestepProjEmbeddings):

    temb = timestep_embedder(time_proj(timestep))  (+ addition_t_embedding(addition_t_cond))

Composes the graduated Timesteps (`timesteps`) and TimestepEmbedding (`timestep_embedding`) ports. `hidden_states` only sets the reference's
cast dtype; the ports keep float32 activations throughout.
"""

from __future__ import annotations

import torch

import ttnn
from models.tt_dit.pipelines.qwen_image_edit_transformer._stubs.timestep_embedding import TtTimestepEmbedding
from models.tt_dit.pipelines.qwen_image_edit_transformer._stubs.timesteps import TtTimesteps


def _replicated(t, device, dtype=ttnn.float32):
    kw = {"mesh_mapper": ttnn.ReplicateTensorToMesh(device)} if isinstance(device, ttnn.MeshDevice) else {}
    return ttnn.from_torch(t.contiguous(), dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device, **kw)


class TtQwenTimestepProjEmbeddings:
    def __init__(self, device, torch_module):
        self.device = device
        self.time_proj = TtTimesteps(device, torch_module.time_proj)
        self.timestep_embedder = TtTimestepEmbedding(device, torch_module.timestep_embedder)
        self.use_additional_t_cond = bool(getattr(torch_module, "use_additional_t_cond", False))
        if self.use_additional_t_cond:
            kw = {"mesh_mapper": ttnn.ReplicateTensorToMesh(device)} if isinstance(device, ttnn.MeshDevice) else {}
            self.add_table = ttnn.from_torch(
                torch_module.addition_t_embedding.weight.detach().to(torch.float32),
                dtype=ttnn.float32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=device,
                **kw,
            )

    def __call__(self, timestep, hidden_states=None, addition_t_cond=None, **_unused):
        cond = self.timestep_embedder(self.time_proj(timestep))
        if self.use_additional_t_cond:
            if addition_t_cond is None:
                raise ValueError("When additional_t_cond is True, addition_t_cond must be provided.")
            idx = addition_t_cond
            if not isinstance(idx, ttnn.Tensor):
                kw = (
                    {"mesh_mapper": ttnn.ReplicateTensorToMesh(self.device)}
                    if isinstance(self.device, ttnn.MeshDevice)
                    else {}
                )
                idx = ttnn.from_torch(
                    idx.to(torch.int32), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=self.device, **kw
                )
            add = ttnn.embedding(idx, self.add_table, layout=ttnn.TILE_LAYOUT)
            cond = ttnn.add(cond, ttnn.reshape(add, cond.shape))
        return cond


def build(device, torch_module=None):
    return TtQwenTimestepProjEmbeddings(device, torch_module)


def qwen_timestep_proj_embeddings(device, torch_module=None):
    return TtQwenTimestepProjEmbeddings(device, torch_module)
