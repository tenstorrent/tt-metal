# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Patch embedding: `MoonVision3dPatchEmbed` in the reference.

The reference is a Conv2d whose kernel equals its stride over patches the processor has already
cut, which makes it a matmul over each flattened 3x14x14 patch (`reference.functional.patches_to_rows`).
So `proj` is a Linear and `_prepare_torch_state` flattens the conv weight into it.

The 64x64 position table stays on the host. It is interpolated to the image's grid there
(`reference.functional.pos_embeds`) and uploaded with the patches, so `forward` only adds it.
"""

from __future__ import annotations

from pathlib import Path

import torch

import ttnn

from .linear import Linear
from .model_config import MoonViTConfig
from .module import Module


class TT_Patch_Embed(Module):
    def __init__(self, config: MoonViTConfig, *, mesh_device: ttnn.MeshDevice, compute_kernel_config) -> None:
        super().__init__()
        self.config = config
        self.proj = Linear(
            config.patch_dim,
            config.hidden_size,
            bias=config.patch_embed_proj_bias,
            mesh_device=mesh_device,
            compute_kernel_config=compute_kernel_config,
        )
        self.pos_emb_weight: torch.Tensor | None = None

    def _prepare_torch_state(self, state: dict[str, torch.Tensor]) -> None:
        weight = state.get("proj.weight")
        if weight is not None and weight.ndim > 2:
            state["proj.weight"] = weight.reshape(self.config.hidden_size, self.config.patch_dim)
        if "pos_emb.weight" in state:
            self.pos_emb_weight = state.pop("pos_emb.weight")

    def save(self, directory: str | Path, /, *, prefix: str = "") -> None:
        if self.pos_emb_weight is None:
            raise ValueError("pos_emb weight is unavailable; call load_torch_state_dict first")
        super().save(directory, prefix=prefix)
        torch.save(self.pos_emb_weight, Path(directory) / f"{prefix}pos_emb.weight.pt")

    def load(self, directory: str | Path, /, *, prefix: str = "") -> None:
        super().load(directory, prefix=prefix)
        self.pos_emb_weight = torch.load(Path(directory) / f"{prefix}pos_emb.weight.pt")

    def forward(self, patches: ttnn.Tensor, pos_emb: ttnn.Tensor) -> ttnn.Tensor:
        """patches [1, 1, S/sp, patch_dim], pos_emb [1, 1, S/sp, hidden] -> [1, 1, S/sp, hidden]."""
        raise NotImplementedError("TT_Patch_Embed.forward")
