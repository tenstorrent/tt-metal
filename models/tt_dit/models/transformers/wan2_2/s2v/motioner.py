# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""On-device port of the S2V FramePackMotioner."""

from __future__ import annotations

import torch

import ttnn

from .....layers.embeddings import WanPatchEmbed
from .....layers.module import Module
from .....parallel.config import DiTParallelConfig
from .....utils.tensor import bf16_tensor


# Vendored from wan/modules/s2v/motioner.py — kept byte-identical to the ref.
def rope_params(max_seq_len, dim, theta=10000):
    assert dim % 2 == 0
    freqs = torch.outer(
        torch.arange(max_seq_len), 1.0 / torch.pow(theta, torch.arange(0, dim, 2).to(torch.float64).div(dim))
    )
    freqs = torch.polar(torch.ones_like(freqs), freqs)
    return freqs


class FramePackMotionerWan(Module):
    """On-device :class:`FramePackMotioner` from the WAN 2.2 S2V reference.

    Three patch-embedding projections with different temporal/spatial strides,
    one per ``zip_frame_buckets`` entry. After a single upload of the motion
    latents, the per-bucket temporal split, patchify reshape/permute, and
    matmul all happen on device.
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
        assert drop_mode == "padd", f"only drop_mode='padd' is supported, got {drop_mode!r}"
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
        # Cached on host; the per-clip ``rope_precompute`` call in the parent
        # transformer uses these to build motion-token rotaries.
        d = self.head_dim
        self.freqs = torch.cat(
            [
                rope_params(1024, d - 4 * (d // 6)),
                rope_params(1024, 2 * (d // 6)),
                rope_params(1024, 2 * (d // 6)),
            ],
            dim=1,
        )

    def _project(self, x_BCTHW_dev: ttnn.Tensor, proj: WanPatchEmbed) -> ttnn.Tensor:
        """Patchify + project on device. Mirrors HF ``Conv3d(stride=kernel=patch_size)``.

        Reshape+permute into the unfolded ``[1, B, N, pT*pH*pW*C]`` form that
        :class:`WanPatchEmbed` expects, decomposed into two ≤6D stages because
        the natural 8D permute exceeds ttnn's supported rank. Spatial/temporal
        remainders not divisible by the patch size are dropped, same as Conv3d.
        """
        pT, pH, pW = proj.patch_size
        B, C, T, H, W = x_BCTHW_dev.shape
        T_use, H_use, W_use = (T // pT) * pT, (H // pH) * pH, (W // pW) * pW
        patch_T, patch_H, patch_W = T_use // pT, H_use // pH, W_use // pW

        x = x_BCTHW_dev
        if (T_use, H_use, W_use) != (T, H, W):
            x = ttnn.slice(x, [0, 0, 0, 0, 0], [B, C, T_use, H_use, W_use])

        # Spatial fold: [B, C, T_use, H, W] → [B, T_use, patch_H, patch_W, pH*pW*C]
        x = ttnn.reshape(x, (B, C, T_use, patch_H, pH, W_use))
        x = ttnn.permute(x, (0, 2, 3, 4, 5, 1))
        x = ttnn.reshape(x, (B, T_use, patch_H, pH, patch_W, pW * C))
        x = ttnn.permute(x, (0, 1, 2, 4, 3, 5))
        x = ttnn.reshape(x, (B, T_use, patch_H, patch_W, pH * pW * C))

        # Temporal fold: → [1, B, patch_T*patch_H*patch_W, pT*pH*pW*C]
        x = ttnn.reshape(x, (B, patch_T, pT, patch_H, patch_W, pH * pW * C))
        x = ttnn.permute(x, (0, 1, 3, 4, 2, 5))
        x = ttnn.reshape(x, (1, B, patch_T * patch_H * patch_W, pT * pH * pW * C))

        return proj(ttnn.to_layout(x, ttnn.TILE_LAYOUT))

    def forward(self, motion_latents: torch.Tensor) -> ttnn.Tensor:
        """Run the three projections and concatenate.

        Args:
            motion_latents: ``[B=1, C=16, T, H, W]`` CPU latents. Single-clip
                production passes a zero tensor with ``T == sum(zip_frame_buckets)``.

        Returns:
            ``[1, B, N_motion, inner_dim]`` motion tokens, TP-fractured on the
            embed dim.
        """
        B, C, T_motion, lat_h, lat_w = motion_latents.shape
        assert B == 1, "FramePackMotionerWan is single-batch"
        assert C == self.in_channels, f"expected {self.in_channels} channels, got {C}"

        # Build padd_lat [B, C, total_T, H, W] on device. Production hits the
        # ``zero_lead == 0`` fast path (motion latents cover total_T); the other
        # branches handle short-clip / no-overlap edge cases.
        total_T = sum(self.zip_frame_buckets)
        overlap = min(total_T, T_motion)
        zero_lead = total_T - overlap

        if overlap > 0:
            x_used_dev = bf16_tensor(
                motion_latents[:, :, -overlap:].contiguous(),
                device=self.mesh_device,
                layout=ttnn.ROW_MAJOR_LAYOUT,
            )
        if zero_lead == 0:
            padd_lat_dev = x_used_dev
        else:
            zeros_dev = ttnn.zeros(
                [B, C, zero_lead, lat_h, lat_w],
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=self.mesh_device,
            )
            padd_lat_dev = zeros_dev if overlap == 0 else ttnn.concat([zeros_dev, x_used_dev], dim=2)

        # Reference splits along T as [4x, 2x, post] (reverse of zip_frame_buckets).
        t_4x, t_2x, _ = list(self.zip_frame_buckets)[::-1]
        clean_4x = ttnn.slice(padd_lat_dev, [0, 0, 0, 0, 0], [B, C, t_4x, lat_h, lat_w])
        clean_2x = ttnn.slice(padd_lat_dev, [0, 0, t_4x, 0, 0], [B, C, t_4x + t_2x, lat_h, lat_w])
        clean_post = ttnn.slice(padd_lat_dev, [0, 0, t_4x + t_2x, 0, 0], [B, C, total_T, lat_h, lat_w])

        return ttnn.concat(
            [
                self._project(clean_post, self.proj),
                self._project(clean_2x, self.proj_2x),
                self._project(clean_4x, self.proj_4x),
            ],
            dim=2,
        )
