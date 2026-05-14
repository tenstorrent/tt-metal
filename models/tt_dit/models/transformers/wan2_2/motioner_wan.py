# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""On-device port of the S2V motion-frame encoder.

The production Wan2.2-S2V-14B config uses ``FramePackMotioner``, defined at
``wan/modules/s2v/motioner.py`` in the reference repo. It is three ``Conv3d``
layers with ``kernel == stride`` (so they're patch-style projections):

  * ``proj``     — kernel (1, 2, 2), processes the most-recent frame
  * ``proj_2x``  — kernel (2, 4, 4), processes the next 2 frames
  * ``proj_4x``  — kernel (4, 8, 8), processes the oldest 16 frames

Each maps ``in_channels=16`` motion-latent channels to ``inner_dim`` (5120
for the production checkpoint). Because the strides equal the kernels, every
op is a standard unfold + matmul — same pattern as :class:`WanPatchEmbed`.

The forward takes a list of motion latents (or a stacked tensor), pads / zips
into the three buckets, runs each projection, and concatenates the token
sequences. Rope-precompute is done on host via the reference helper and
uploaded as a real tensor — it's invoked once per clip outside the denoise
loop.
"""

from __future__ import annotations

from typing import Iterable

import torch

import ttnn

from ....layers.embeddings import WanPatchEmbed
from ....layers.module import Module
from ....parallel.config import DiTParallelConfig
from ....utils.tensor import bf16_tensor


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

        # Rope frequencies — same construction as the reference's ``self.freqs``
        # in WanModel_S2V.__init__. Cached on host; the per-clip
        # ``rope_precompute`` call uses these to build motion token rotaries.
        # The reference's ``wan/__init__.py`` eagerly evaluates
        # ``torch.cuda.current_device()`` at import time; stub it for CPU.
        d = self.head_dim
        import sys
        import types

        if "flash_attn" not in sys.modules:
            mod = types.ModuleType("flash_attn")
            mod.flash_attn_func = None  # type: ignore[attr-defined]
            mod.flash_attn_qkvpacked_func = None  # type: ignore[attr-defined]
            sys.modules["flash_attn"] = mod
        if "decord" not in sys.modules:
            mod = types.ModuleType("decord")
            mod.VideoReader = None  # type: ignore[attr-defined]
            mod.cpu = lambda x=0: None  # type: ignore[attr-defined]
            sys.modules["decord"] = mod
        _orig_cuda_current = torch.cuda.current_device
        torch.cuda.current_device = lambda: 0  # type: ignore[assignment]
        try:
            from wan.modules.model import rope_params
        finally:
            torch.cuda.current_device = _orig_cuda_current  # type: ignore[assignment]

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
            motion_latents: CPU motion-latent tensor of shape ``[B, C=16, T, H, W]``
                (or a single-element list of one). For v1 single-clip the
                contents are typically zero; the projections still produce the
                learned-bias broadcast pattern.
            add_last_motion: matches the reference's flag. ``< 2`` with
                ``drop_mode == "drop"`` zeros out the most-recent slot.

        Returns:
            ``(motion_tokens_dev, motion_rope_torch)``:
                * ``motion_tokens_dev`` — ``[1, B, N_motion, inner_dim]`` ttnn
                  tensor (TP-fractured on D, matching the noisy patched tokens).
                * ``motion_rope_torch`` — ``[B, N_motion, num_heads,
                  head_dim/2]`` complex CPU tensor; applied in the standard
                  attention path via the existing rope plumbing.
        """
        import sys
        import types

        if "flash_attn" not in sys.modules:
            mod = types.ModuleType("flash_attn")
            mod.flash_attn_func = None  # type: ignore[attr-defined]
            mod.flash_attn_qkvpacked_func = None  # type: ignore[attr-defined]
            sys.modules["flash_attn"] = mod
        if "decord" not in sys.modules:
            mod = types.ModuleType("decord")
            mod.VideoReader = None  # type: ignore[attr-defined]
            mod.cpu = lambda x=0: None  # type: ignore[attr-defined]
            sys.modules["decord"] = mod
        _orig_cuda_current = torch.cuda.current_device
        torch.cuda.current_device = lambda: 0  # type: ignore[assignment]
        try:
            from wan.modules.s2v.s2v_utils import rope_precompute
        finally:
            torch.cuda.current_device = _orig_cuda_current  # type: ignore[assignment]

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
        if add_last_motion < 2 and self.drop_mode != "drop":
            zero_end_frame = sum(self.zip_frame_buckets[: len(self.zip_frame_buckets) - add_last_motion - 1])
            padd_lat[:, :, -zero_end_frame:] = 0

        # Split the temporal axis as the reference does — bucket order is
        # [4x, 2x, post] (reverse of self.zip_frame_buckets).
        clean_4x, clean_2x, clean_post = padd_lat.split(list(self.zip_frame_buckets)[::-1], dim=2)

        # Patchify + on-device matmul for each bucket.
        post_tokens = self._project(clean_post, self.proj)  # [1, B, N_post, dim]
        twox_tokens = self._project(clean_2x, self.proj_2x)
        fourx_tokens = self._project(clean_4x, self.proj_4x)

        if add_last_motion < 2 and self.drop_mode == "drop":
            if add_last_motion < 2:
                post_tokens = ttnn.slice(post_tokens, [0, 0, 0, 0], [1, 1, 0, post_tokens.shape[-1]])
            if add_last_motion < 1:
                twox_tokens = ttnn.slice(twox_tokens, [0, 0, 0, 0], [1, 1, 0, twox_tokens.shape[-1]])

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
