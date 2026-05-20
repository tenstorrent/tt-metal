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
for the motion tokens is done by the surrounding transformer's
``prepare_rope_features`` (which rebuilds rope from scratch using
``self.freqs`` as reference data); the motioner does not produce its own
rope.

The alternative ``MotionerTransformers`` path (``enable_motioner=True``) is
rejected by :class:`WanS2VTransformer3DModel.__init__` and is not in scope.
"""

from __future__ import annotations

from typing import Iterable

import torch

import ttnn

from .....layers.embeddings import WanPatchEmbed
from .....layers.module import Module
from .....parallel.config import DiTParallelConfig
from .....utils.tensor import bf16_tensor
from .rope_s2v import rope_params


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

        # Build padd_lat by concat instead of zero-alloc + slice-overwrite.
        # In the production multi-clip path, T_motion == total_T == 19, so the
        # full zero buffer was being allocated and then completely overwritten —
        # ~7.6 MB of pointless alloc + memset per clip. Concat skips that.
        total_T = sum(self.zip_frame_buckets)
        overlap = min(total_T, T_motion)
        zero_lead = total_T - overlap
        if zero_lead == 0:
            padd_lat = x[:, :, -overlap:]
        elif overlap == 0:
            padd_lat = torch.zeros(B, C, total_T, lat_h, lat_w, dtype=x.dtype, device=x.device)
        else:
            padd_lat = torch.cat(
                [
                    torch.zeros(B, C, zero_lead, lat_h, lat_w, dtype=x.dtype, device=x.device),
                    x[:, :, -overlap:],
                ],
                dim=2,
            )

        # Future optimization: replace the host-side `_patchify_for_unfolded_conv`
        # + `bf16_tensor` upload per bucket with `ttnn.experimental.conv3d` using
        # stride=kernel=patch_size. That requires a conv3d sweep on the three
        # motioner shapes ((16, 5120, (1,2,2)), (16, 5120, (2,4,4)),
        # (16, 5120, (4,8,8))) so they don't hit the `(in_c, 32, 1, 1, 1)`
        # hardcoded fallback. Tracked under WAN_S2V_PERF_CLEANUP.md item 4.

        # Split the temporal axis as the reference does — bucket order is
        # [4x, 2x, post] (reverse of self.zip_frame_buckets).
        clean_4x, clean_2x, clean_post = padd_lat.split(list(self.zip_frame_buckets)[::-1], dim=2)

        # Patchify + on-device matmul for each bucket.
        post_tokens = self._project(clean_post, self.proj)  # [1, B, N_post, dim]
        twox_tokens = self._project(clean_2x, self.proj_2x)
        fourx_tokens = self._project(clean_4x, self.proj_4x)

        return ttnn.concat([post_tokens, twox_tokens, fourx_tokens], dim=2)
