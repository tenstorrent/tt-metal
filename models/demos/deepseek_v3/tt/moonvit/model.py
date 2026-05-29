# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0
"""
Full MoonViT vision tower.

Forward pipeline (matches MoonVitPretrainedModel.forward + final_layernorm
+ KimiVLMultiModalProjector when `MoonViT` is constructed with a projector):

    pixel_patches (L, 3, 14, 14) host
        -> patch_embed     (Conv2d projection + 2D learned posemb add)  device
        -> 27 x encoder blocks
              (LN -> attn(cu_seqlens, 2D RoPE) -> residual; LN -> MLP -> residual)
        -> final_layernorm
        -> patch_merger (2x2 spatial concat)
        -> projector (LN -> Linear -> GELU -> Linear)
        -> vision tokens (L_new, text_hidden)

The patch_merger step runs on host (pure shape ops); everything else is
on device. cu_seqlens and the 2D-RoPE cos/sin are precomputed from grid_hws
host-side and pushed to device once per forward.

References:
  - `MoonVitPretrainedModel` and `KimiVLMultiModalProjector` in modeling_kimi_k25.py.
"""
from __future__ import annotations

from typing import List, Optional

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.demos.deepseek_v3.tt.moonvit.block import MoonVisionBlock
from models.demos.deepseek_v3.tt.moonvit.layernorm import MoonVisionLayerNorm
from models.demos.deepseek_v3.tt.moonvit.patch_embed import MoonVisionPatchEmbed, MoonVisionPatchProj
from models.demos.deepseek_v3.tt.moonvit.projector import MoonViTProjector
from models.demos.deepseek_v3.tt.moonvit.rope import Rope2DSetup


def _is_mesh_device(device) -> bool:
    return type(device).__name__ == "MeshDevice"


class MoonViT(LightweightModule):
    """Full MoonViT vision tower + optional projector composition."""

    def __init__(
        self,
        mesh_device,
        patch_embed: MoonVisionPatchEmbed,
        blocks: List[MoonVisionBlock],
        final_layernorm: MoonVisionLayerNorm,
        rope: Rope2DSetup,
        merge_kernel_size,
        projector: Optional[MoonViTProjector] = None,
        dtype=ttnn.bfloat16,
    ):
        super().__init__()
        self.device = mesh_device
        self.dtype = dtype
        self.patch_embed = patch_embed
        self.blocks = blocks
        self.final_layernorm = final_layernorm
        self.rope = rope
        self.merge_kernel_size = tuple(merge_kernel_size)
        self.projector = projector

    @classmethod
    def from_torch(
        cls,
        model_args,
        mesh_device,
        with_projector: bool = True,
        dtype=ttnn.bfloat16,
    ) -> "MoonViT":
        """Construct the full vision tower from a MoonViTModelArgs.

        Uses the reference factories to extract weights from the loaded
        HF Kimi-VL model. Loading happens lazily — first access to
        `model_args.hf_model` triggers the checkpoint download (or hits
        the local cache).
        """
        # Lazily resolve the HF model graph so the encoder.blocks list is in scope.
        from models.demos.deepseek_v3.tt.moonvit._references import _vision_tower

        vt = _vision_tower(model_args)
        ref_blocks = vt.encoder.blocks
        num_layers = len(ref_blocks)

        # Patch embed (Conv2d projection + learned 2D interp posemb).
        ref_patch_embed = model_args.reference_patch_embed()
        patch_embed = MoonVisionPatchEmbed.from_torch(mesh_device, ref_patch_embed, dtype=dtype)

        # Encoder blocks.
        blocks: List[MoonVisionBlock] = []
        for i, ref_layer in enumerate(ref_blocks):
            # Force sdpa to keep behavior consistent with our cu_seqlens-aware path.
            ref_layer.attn_implementation = "sdpa"
            blocks.append(
                MoonVisionBlock.from_torch(
                    mesh_device,
                    ref_layer,
                    hidden_size=model_args.hidden_size,
                    num_heads=model_args.num_attention_heads,
                    head_dim=model_args.head_dim,
                    dtype=dtype,
                )
            )

        # Final LayerNorm (encoder tail).
        ref_final_ln = model_args.reference_final_layernorm()
        final_ln = MoonVisionLayerNorm.from_torch(mesh_device, ref_final_ln, dtype=dtype)

        # 2D RoPE (used by every block).
        ref_rope = model_args.reference_rope_2d()
        rope = Rope2DSetup.from_torch(ref_rope)

        # Optional projector.
        projector: Optional[MoonViTProjector] = None
        if with_projector:
            ref_proj = model_args.reference_projector()
            projector = MoonViTProjector.from_torch(mesh_device, ref_proj, dtype=dtype)

        return cls(
            mesh_device=mesh_device,
            patch_embed=patch_embed,
            blocks=blocks,
            final_layernorm=final_ln,
            rope=rope,
            merge_kernel_size=model_args.merge_kernel_size,
            projector=projector,
            dtype=dtype,
        )

    # ------------------------------------------------------------------
    # Helpers

    def _build_cu_seqlens(self, grid_hws_pt: torch.Tensor) -> ttnn.Tensor:
        """Build cu_seqlens uint32 row-major tensor on device."""
        lengths = torch.cat([torch.zeros(1, dtype=grid_hws_pt.dtype), grid_hws_pt[:, 0] * grid_hws_pt[:, 1]])
        cu_pt = lengths.cumsum(dim=0, dtype=torch.int32).contiguous()
        mesh_mapper = ttnn.ReplicateTensorToMesh(self.device) if _is_mesh_device(self.device) else None
        return ttnn.from_torch(
            cu_pt,
            device=self.device,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=mesh_mapper,
        )

    def _push_pixel_patches(self, pixel_patches: torch.Tensor) -> ttnn.Tensor:
        """Flatten (L, 3, 14, 14) patches host-side and push to device."""
        x_flat = MoonVisionPatchProj.flatten_patches(pixel_patches.to(torch.bfloat16))
        L = x_flat.shape[0]
        x_4d = x_flat.view(1, 1, L, -1).contiguous()
        mesh_mapper = ttnn.ReplicateTensorToMesh(self.device) if _is_mesh_device(self.device) else None
        return ttnn.from_torch(
            x_4d,
            device=self.device,
            dtype=self.dtype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=mesh_mapper,
        )

    def _push_merged(self, merged_pt: torch.Tensor) -> ttnn.Tensor:
        """Push the host-computed patch_merger output to device.

        merged_pt has shape (L_new_total, kh*kw, vision_hidden); we reshape
        to (1, 1, L_new_total * kh*kw, vision_hidden) which is what the
        projector expects (LN-over-last-dim then reshape).
        """
        L_new, k_group, D = merged_pt.shape
        flat = merged_pt.reshape(L_new * k_group, D)
        x_4d = flat.view(1, 1, L_new * k_group, D).to(torch.bfloat16).contiguous()
        mesh_mapper = ttnn.ReplicateTensorToMesh(self.device) if _is_mesh_device(self.device) else None
        return ttnn.from_torch(
            x_4d,
            device=self.device,
            dtype=self.dtype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=mesh_mapper,
        )

    # ------------------------------------------------------------------
    # Forward

    def forward(
        self,
        pixel_patches: torch.Tensor,  # (L, 3, 14, 14) host
        grid_hws: torch.Tensor,  # (N, 2) host
    ) -> ttnn.Tensor:
        """End-to-end vision tower forward.

        Args:
            pixel_patches: host tensor of pre-cut patches.
            grid_hws: host tensor of per-image (H, W) shapes.

        Returns:
            device tensor. If projector is present, shape (1, 1, L_new, text_hidden);
            otherwise (1, 1, L, vision_hidden) — the pre-merge encoder output.
        """
        # ----- patch_embed (Conv2d proj + bicubic posemb) on device -----
        x_tt = self._push_pixel_patches(pixel_patches)
        x_tt = self.patch_embed(x_tt, grid_hws)  # (1, 1, L, hidden)

        # ----- encoder blocks share cu_seqlens and cos/sin -----
        cu_tt = self._build_cu_seqlens(grid_hws)
        # Use the first block's attention staging for cos/sin (they all use the same shapes).
        cos_pt, sin_pt = self.rope.get_cos_sin(grid_hws, dtype=torch.float32)
        cos_tt, sin_tt = self.blocks[0].attention.stage_cos_sin(cos_pt, sin_pt)

        for blk in self.blocks:
            x_tt = blk(x_tt, cu_tt, cos_tt, sin_tt)

        # ----- final layernorm on device -----
        x_tt = self.final_layernorm(x_tt)

        # ----- patch_merger on host -----
        # Pull encoder output to host as (L, vision_hidden).
        x_pt = ttnn.to_torch(
            x_tt,
            mesh_composer=ttnn.ConcatMeshToTensor(self.device, dim=0) if _is_mesh_device(self.device) else None,
        )
        if _is_mesh_device(self.device) and x_pt.shape[0] != 1:
            x_pt = x_pt[:1]
        L = pixel_patches.shape[0]
        x_pt_2d = x_pt.view(L, -1)
        from models.demos.deepseek_v3.tt.moonvit.patch_merger import patch_merger_per_image

        merged_list = patch_merger_per_image(x_pt_2d, grid_hws, self.merge_kernel_size)
        merged_pt = torch.cat(merged_list, dim=0)  # (L_new, kh*kw, vision_hidden)

        if self.projector is None:
            # Return the merger output for callers that want the un-projected form.
            return self._push_merged(merged_pt)

        # ----- projector on device -----
        merged_tt = self._push_merged(merged_pt)
        out_tt = self.projector(merged_tt)
        return out_tt


class DropInMoonViT(torch.nn.Module):
    """torch.nn.Module-compatible wrapper around the ttnn MoonViT vision tower.

    Faithfully mirrors HF `MoonVitPretrainedModel.forward(pixel_values, grid_hws)`:
    accepts host torch tensors and returns a list of per-image 3D tensors of
    shape ``(L_new_i, kh*kw, vision_hidden)``. The wrapped tt-metal MoonViT
    must be constructed WITHOUT the projector (``with_projector=False``); the
    HF projector is a separate module that consumers chain externally — see
    `DropInKimiVLMultiModalProjector` below.

    Use case: any HF code (e.g., `KimiVLForConditionalGeneration`) that
    delegates vision encoding to ``model.vision_tower(pixel_values, grid_hws)``
    can substitute an instance of this class in place of the HF tower with
    no other changes. Useful for A/B comparing our impl against HF in
    existing HF-based pipelines.
    """

    def __init__(self, tt_model: "MoonViT"):
        super().__init__()
        if tt_model.projector is not None:
            raise ValueError(
                "DropInMoonViT requires a MoonViT built with with_projector=False; "
                "got one with a projector. Build a vision-tower-only MoonViT via "
                "MoonViT.from_torch(..., with_projector=False) and use "
                "DropInKimiVLMultiModalProjector separately for the projector."
            )
        self.tt_model = tt_model
        self.mesh_device = tt_model.device
        self.merge_kernel_size = tt_model.merge_kernel_size

    def forward(
        self,
        pixel_values: torch.Tensor,
        grid_hws: torch.Tensor,
    ) -> List[torch.Tensor]:
        """Run the tt-metal vision tower; return outputs in HF list format.

        Args:
            pixel_values: host tensor of pre-cut patches, shape (L, 3, 14, 14)
                (as produced by HF KimiVLImageProcessor).
            grid_hws: host tensor of per-image (H, W) shapes, (N, 2).

        Returns:
            list of N torch CPU tensors, each of shape
            ``(L_new_i, kh*kw, vision_hidden)`` — same as HF `patch_merger`.
        """
        out_tt = self.tt_model(pixel_values, grid_hws)
        is_mesh = _is_mesh_device(self.mesh_device)
        out_pt = ttnn.to_torch(
            out_tt,
            mesh_composer=ttnn.ConcatMeshToTensor(self.mesh_device, dim=0) if is_mesh else None,
        )
        if is_mesh and out_pt.shape[0] != 1:
            out_pt = out_pt[:1]

        # MoonViT (without projector) yields (1, 1, L_new_total * kh*kw, vision_hidden).
        # Reshape to (L_new_total, kh*kw, vision_hidden) then split per-image.
        kh, kw = self.merge_kernel_size
        k_group = kh * kw
        vision_hidden = out_pt.shape[-1]
        merged = out_pt.view(-1, vision_hidden)
        l_new_total = merged.shape[0] // k_group
        merged = merged.view(l_new_total, k_group, vision_hidden)

        outputs: List[torch.Tensor] = []
        cursor = 0
        for h, w in grid_hws.tolist():
            new_h, new_w = int(h) // kh, int(w) // kw
            n = new_h * new_w
            outputs.append(merged[cursor : cursor + n].contiguous())
            cursor += n
        if cursor != l_new_total:
            raise RuntimeError(
                f"DropInMoonViT: per-image L_new sum ({cursor}) != L_new_total "
                f"({l_new_total}); grid_hws inconsistent with tt_model output."
            )
        return outputs


class DropInKimiVLMultiModalProjector(torch.nn.Module):
    """torch.nn.Module-compatible wrapper around the ttnn `MoonViTProjector`.

    Mirrors HF `KimiVLMultiModalProjector.forward(image_features)` exactly:
    accepts the list output of `DropInMoonViT` (or HF MoonVitPretrainedModel)
    and returns a single torch tensor of shape ``(L_new_total, text_hidden)``.
    """

    def __init__(self, tt_projector, mesh_device, dtype=ttnn.bfloat16):
        super().__init__()
        self.tt_projector = tt_projector
        self.mesh_device = mesh_device
        self.dtype = dtype

    def forward(self, image_features: List[torch.Tensor]) -> torch.Tensor:
        # Mirror HF: torch.cat along dim=0 of the list -> (L_new_total, kh*kw, vision_hidden).
        concat = torch.cat(image_features, dim=0)
        l_new, k_group, vision_hidden = concat.shape
        flat = concat.reshape(l_new * k_group, vision_hidden)
        x_4d = flat.view(1, 1, l_new * k_group, vision_hidden).to(torch.bfloat16).contiguous()
        is_mesh = _is_mesh_device(self.mesh_device)
        x_tt = ttnn.from_torch(
            x_4d,
            device=self.mesh_device,
            dtype=self.dtype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device) if is_mesh else None,
        )
        out_tt = self.tt_projector(x_tt)
        out_pt = ttnn.to_torch(
            out_tt,
            mesh_composer=ttnn.ConcatMeshToTensor(self.mesh_device, dim=0) if is_mesh else None,
        )
        if is_mesh and out_pt.shape[0] != 1:
            out_pt = out_pt[:1]
        return out_pt.view(l_new, -1)
