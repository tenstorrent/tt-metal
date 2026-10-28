# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""The vision model: `KimiK3VisionModel` in the reference.

`TT_MoonViT_Vision_Tower` is `MoonViT3dPretrainedModel` (patch embed, 27 blocks, final norm, patch
merger). `TT_MoonViT_Vision_Model` adds the projector (`mm_projector`), so it maps image patches to
language-model embeddings.

One still image per call. Its patches are padded to `config.max_patches` rows on the host
(`reference.functional`), uploaded split along `sp_axis`, run through the tower, merger and
projector, and gathered back with the padding dropped. Weights are replicated and activations are
DRAM interleaved.

Child names match the reference, so `load_torch_state_dict(reference.state_dict())` loads it, and
`module.load_cached` caches it.
"""

from __future__ import annotations

import torch

import ttnn

from .ccl import ParallelFactor, TT_CCL
from .model_config import MoonViTConfig, VisionParallel, check_grid, compute_kernel_config, resolve_parallel
from .module import Module, ModuleList
from .patch_embed import TT_Patch_Embed
from .patch_merger import TT_Patch_Merger
from .projector import TT_Projector
from .rms_norm import RMSNorm
from .rope import RopeTensors, TT_Rope
from .vision_block import TT_Vision_Block


class TT_MoonViT_Encoder(Module):
    def __init__(
        self, config: MoonViTConfig, *, mesh_device: ttnn.MeshDevice, parallel: VisionParallel, compute_kernel_config
    ) -> None:
        super().__init__()
        self.blocks = ModuleList(
            TT_Vision_Block(
                config, mesh_device=mesh_device, parallel=parallel, compute_kernel_config=compute_kernel_config
            )
            for _ in range(config.num_hidden_layers)
        )
        self.final_layernorm = RMSNorm(
            config.hidden_size,
            eps=config.norm_eps,
            mesh_device=mesh_device,
            compute_kernel_config=compute_kernel_config,
        )

    def forward(self, x: ttnn.Tensor, rope: RopeTensors, logical_n: int) -> ttnn.Tensor:
        raise NotImplementedError("TT_MoonViT_Encoder.forward")


class TT_MoonViT_Vision_Tower(Module):
    def __init__(
        self, config: MoonViTConfig, *, mesh_device: ttnn.MeshDevice, parallel: VisionParallel, compute_kernel_config
    ) -> None:
        super().__init__()
        self.patch_embed = TT_Patch_Embed(config, mesh_device=mesh_device, compute_kernel_config=compute_kernel_config)
        self.encoder = TT_MoonViT_Encoder(
            config, mesh_device=mesh_device, parallel=parallel, compute_kernel_config=compute_kernel_config
        )
        self.merger = TT_Patch_Merger(config, mesh_device=mesh_device, parallel=parallel)

    def forward(
        self, patches: ttnn.Tensor, pos_emb: ttnn.Tensor, rope: RopeTensors, logical_n: int, h: int, w: int
    ) -> ttnn.Tensor:
        x = self.patch_embed(patches, pos_emb)
        x = self.encoder(x, rope, logical_n)
        return self.merger(x, h, w)


class TT_MoonViT_Vision_Model(Module):
    def __init__(
        self,
        config: MoonViTConfig,
        *,
        mesh_device: ttnn.MeshDevice,
        sequence_parallel: ParallelFactor | None = None,
        ccl: TT_CCL | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.mesh_device = mesh_device
        self.parallel = resolve_parallel(mesh_device, sequence_parallel, ccl, config)
        ckc = compute_kernel_config(mesh_device)
        self.rope = TT_Rope(config, mesh_device=mesh_device, parallel=self.parallel)
        self.vision_tower = TT_MoonViT_Vision_Tower(
            config, mesh_device=mesh_device, parallel=self.parallel, compute_kernel_config=ckc
        )
        self.mm_projector = TT_Projector(config, mesh_device=mesh_device, compute_kernel_config=ckc)

    def prepare_inputs(self, pixel_values: torch.Tensor, grid_thws: torch.Tensor) -> dict:
        """Host prep (`reference.functional.prepare_host_inputs`), then upload split along `sp_axis`.

        Returns `patches` [1, 1, S/sp, patch_dim], `pos_emb` [1, 1, S/sp, hidden], `rope`,
        `logical_n` (the unpadded length), and the grid `h`, `w`.
        """
        check_grid(grid_thws, self.config)
        raise NotImplementedError("TT_MoonViT_Vision_Model.prepare_inputs")

    def gather_output(self, out: ttnn.Tensor, *, logical_n: int) -> torch.Tensor:
        """Concatenate the shards along `sp_axis` and drop the padding -> (logical_n, C)."""
        raise NotImplementedError("TT_MoonViT_Vision_Model.gather_output")

    def forward(
        self, patches: ttnn.Tensor, pos_emb: ttnn.Tensor, rope: RopeTensors, logical_n: int, h: int, w: int
    ) -> ttnn.Tensor:
        x = self.vision_tower(patches, pos_emb, rope, logical_n, h, w)
        return self.mm_projector(x)

    def run(self, pixel_values: torch.Tensor, grid_thws: torch.Tensor) -> torch.Tensor:
        """Host in, host out: (h*w, 3, 14, 14), [[1, h, w]] -> (h*w/4, text_hidden_size)."""
        inputs = self.prepare_inputs(pixel_values, grid_thws)
        merged_n = inputs["logical_n"] // self.config.merge_area
        return self.gather_output(self.forward(**inputs), logical_n=merged_n)
