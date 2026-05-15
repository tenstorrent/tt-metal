# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""On-device port of the S2V motion-frame encoder.

The production Wan2.2-S2V-14B config uses ``FramePackMotioner`` from
``wan/modules/s2v/motioner.py``: three ``Conv3d`` layers with
``kernel == stride`` (patch-style projections):

  * ``proj``     — kernel (1, 2, 2), most-recent frame
  * ``proj_2x``  — kernel (2, 4, 4), next 2 frames
  * ``proj_4x``  — kernel (4, 8, 8), oldest 16 frames

Each maps ``in_channels=16`` motion-latent channels to ``inner_dim`` (5120
for the production checkpoint). Because stride equals kernel, every op is
an unfold + matmul — same pattern as :class:`WanPatchEmbed`.

The forward takes a motion-latent tensor, pads/zips into the three buckets,
runs each projection, and concatenates the token sequences. Rope-precompute
runs on host via :func:`s2v_rope.rope_precompute` and is uploaded once per
clip.

The alternative ``MotionerTransformers`` path (``enable_motioner=True``) is
not in scope; :class:`MotionerTransformersWan` is a placeholder that raises
on instantiation so the class hierarchy stays parallel to the reference.
"""

from __future__ import annotations

from typing import Iterable

import torch

import ttnn

from ....layers.embeddings import WanPatchEmbed
from ....layers.module import Module
from ....parallel.config import DiTParallelConfig
from ....utils.tensor import bf16_tensor
from .s2v_rope import rope_params, rope_precompute


def _patchify_for_unfolded_conv(x_BCTHW: torch.Tensor, patch_size: tuple[int, int, int]) -> torch.Tensor:
    """Reshape ``[B, C, T, H, W]`` into ``[1, B, N, pT*pH*pW*C]``.

    Mirrors :meth:`WanTransformer3DModel.preprocess_spatial_input_host`. Drops
    edge tokens when ``T/H/W`` isn't a multiple of the corresponding patch
    dim (the reference's ``Conv3d`` does the same — output size = ``T // pT``).
    """
    pT, pH, pW = patch_size
    B, C, T, H, W = x_BCTHW.shape
    # Trim edges to a multiple of patch_size — Conv3d with stride==kernel
    # silently drops the remainder, so we do the same.
    T_use, H_use, W_use = (T // pT) * pT, (H // pH) * pH, (W // pW) * pW
    x_BCTHW = x_BCTHW[:, :, :T_use, :H_use, :W_use]
    patch_T, patch_H, patch_W = T_use // pT, H_use // pH, W_use // pW
    N = patch_T * patch_H * patch_W
    x = x_BCTHW.reshape(B, C, patch_T, pT, patch_H, pH, patch_W, pW)
    x = x.permute(0, 2, 4, 6, 3, 5, 7, 1).reshape(1, B, N, pT * pH * pW * C)
    return x


class FramePackMotionerWan(Module):
    """On-device :class:`FramePackMotioner` from the WAN 2.2 S2V reference.

    Three patch-embedding projections with different temporal/spatial strides,
    one per ``zip_frame_buckets`` entry. Produces flat motion tokens on device
    and returns the per-token rope embedding as a CPU complex tensor (rope is
    applied downstream via the standard attention path).
    """

    def __init__(
        self,
        *,
        in_channels: int = 16,
        inner_dim: int = 5120,
        num_heads: int = 40,
        zip_frame_buckets: tuple[int, int, int] = (1, 2, 16),
        drop_mode: str = "padd",
        mesh_device: ttnn.MeshDevice,
        parallel_config: DiTParallelConfig,
    ) -> None:
        super().__init__()
        assert (
            inner_dim % num_heads == 0 and (inner_dim // num_heads) % 2 == 0
        ), "inner_dim must be divisible by num_heads and head_dim must be even"
        self.in_channels = in_channels
        self.inner_dim = inner_dim
        self.num_heads = num_heads
        self.head_dim = inner_dim // num_heads
        self.zip_frame_buckets = zip_frame_buckets
        self.drop_mode = drop_mode
        self.mesh_device = mesh_device
        self.parallel_config = parallel_config
        tp_axis = parallel_config.tensor_parallel.mesh_axis

        # Three patch-style projections matching the reference's
        # ``proj``/``proj_2x``/``proj_4x`` Conv3d layers (stride == kernel).
        self.proj = WanPatchEmbed(
            patch_size=(1, 2, 2),
            in_channels=in_channels,
            embed_dim=inner_dim,
            mesh_device=mesh_device,
            tp_mesh_axis=tp_axis,
        )
        self.proj_2x = WanPatchEmbed(
            patch_size=(2, 4, 4),
            in_channels=in_channels,
            embed_dim=inner_dim,
            mesh_device=mesh_device,
            tp_mesh_axis=tp_axis,
        )
        self.proj_4x = WanPatchEmbed(
            patch_size=(4, 8, 8),
            in_channels=in_channels,
            embed_dim=inner_dim,
            mesh_device=mesh_device,
            tp_mesh_axis=tp_axis,
        )

        # Rope frequencies — same construction as ``WanModel_S2V.__init__``.
        # Cached on host; the per-clip ``rope_precompute`` call uses these to
        # build motion-token rotaries.
        d = self.head_dim
        self.freqs = torch.cat(
            [
                rope_params(1024, d - 4 * (d // 6)),
                rope_params(1024, 2 * (d // 6)),
                rope_params(1024, 2 * (d // 6)),
            ],
            dim=1,
        )

    def _project(self, x_BCTHW: torch.Tensor, proj: WanPatchEmbed) -> ttnn.Tensor:
        """Host-patchify → upload → on-device matmul. Returns ``[1, B, N, dim]``."""
        x_1BNI = _patchify_for_unfolded_conv(x_BCTHW, proj.patch_size)
        x_dev = bf16_tensor(x_1BNI, device=self.mesh_device, layout=ttnn.TILE_LAYOUT)
        return proj(x_dev)

    def forward(
        self,
        motion_latents: torch.Tensor | Iterable[torch.Tensor],
        *,
        add_last_motion: int = 2,
    ) -> tuple[ttnn.Tensor, torch.Tensor]:
        """Run the three projections and concatenate.

        Args:
            motion_latents: CPU motion-latent tensor ``[B, C=16, T, H, W]``
                (or a single-element list of one). For single-clip the
                contents are typically zero.
            add_last_motion: production default is ``2``; other values are
                rejected (not in the HF prod path).

        Returns:
            ``(motion_tokens_dev, motion_rope_torch)``:
                * ``motion_tokens_dev`` — ``[1, B, N_motion, inner_dim]``
                  ttnn tensor, TP-fractured on D.
                * ``motion_rope_torch`` — ``[B, N_motion, num_heads,
                  head_dim/2]`` complex CPU tensor; applied downstream via
                  the standard rope plumbing.
        """
        # Production always passes add_last_motion=2; lower values are an
        # historical legacy of multi-clip / partial-context inference paths
        # the production pipeline never exercises.
        if add_last_motion != 2:
            raise NotImplementedError(f"add_last_motion={add_last_motion} is not in the HF prod path (expected 2)")

        if isinstance(motion_latents, torch.Tensor):
            motion_latents = [motion_latents]
        x = motion_latents[0]
        if x.dim() == 4:
            # [C, T, H, W] → [1, C, T, H, W]
            x = x.unsqueeze(0)
        B, C, T_motion, lat_h, lat_w = x.shape
        assert B == 1, "FramePackMotionerWan is single-batch in v1"
        assert C == self.in_channels, f"expected {self.in_channels} channels, got {C}"

        total_T = sum(self.zip_frame_buckets)
        padd_lat = torch.zeros(B, C, total_T, lat_h, lat_w, dtype=x.dtype, device=x.device)
        overlap = min(total_T, T_motion)
        if overlap > 0:
            padd_lat[:, :, -overlap:] = x[:, :, -overlap:]

        # Split the temporal axis as the reference does — bucket order is
        # [4x, 2x, post] (reverse of self.zip_frame_buckets).
        clean_4x, clean_2x, clean_post = padd_lat.split(list(self.zip_frame_buckets)[::-1], dim=2)

        # Patchify + on-device matmul for each bucket.
        post_tokens = self._project(clean_post, self.proj)  # [1, B, N_post, dim]
        twox_tokens = self._project(clean_2x, self.proj_2x)
        fourx_tokens = self._project(clean_4x, self.proj_4x)

        motion_tokens = ttnn.concat([post_tokens, twox_tokens, fourx_tokens], dim=2)

        # Rope precompute on host. Grid sizes mirror the reference (lines
        # 720-749 of motioner.py): negative ``f_o`` indicates motion frames
        # precede the noisy clip in time, and the ``t_f/t_h/t_w`` "range"
        # arguments give the upsampling factors used by the rope frequencies.
        zb = self.zip_frame_buckets
        start_post = -zb[0]
        end_post = start_post + zb[0]
        grid_post = [
            torch.tensor([start_post, 0, 0]).unsqueeze(0),
            torch.tensor([end_post, lat_h // 2, lat_w // 2]).unsqueeze(0),
            torch.tensor([zb[0], lat_h // 2, lat_w // 2]).unsqueeze(0),
        ]
        start_2x = -(zb[0] + zb[1])
        end_2x = start_2x + zb[1] // 2
        grid_2x = [
            torch.tensor([start_2x, 0, 0]).unsqueeze(0),
            torch.tensor([end_2x, lat_h // 4, lat_w // 4]).unsqueeze(0),
            torch.tensor([zb[1], lat_h // 2, lat_w // 2]).unsqueeze(0),
        ]
        start_4x = -(zb[0] + zb[1] + zb[2])
        end_4x = start_4x + zb[2] // 4
        grid_4x = [
            torch.tensor([start_4x, 0, 0]).unsqueeze(0),
            torch.tensor([end_4x, lat_h // 8, lat_w // 8]).unsqueeze(0),
            torch.tensor([zb[2], lat_h // 2, lat_w // 2]).unsqueeze(0),
        ]
        grid_sizes = [grid_post, grid_2x, grid_4x]

        N_motion = motion_tokens.shape[-2]
        rope_input = torch.zeros(B, N_motion, self.num_heads, self.head_dim, dtype=torch.float32)
        motion_rope = rope_precompute(rope_input, grid_sizes, self.freqs, start=None)
        return motion_tokens, motion_rope


class MotionerTransformersWan(Module):
    """Placeholder for the alternative ``MotionerTransformers`` path.

    Kept so the class hierarchy parallels ``wan/modules/s2v/motioner.py``,
    but ``enable_motioner=True`` is rejected by
    :class:`WanS2VTransformer3DModel.__init__`, so instantiation is never
    expected.
    """

    def __init__(self, *_args, **_kwargs) -> None:
        raise NotImplementedError("MotionerTransformersWan is not implemented; production uses FramePackMotionerWan")
