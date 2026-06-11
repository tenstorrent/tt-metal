# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Reusable fast image-encode optimizations for Wan2.2 I2V pipelines.

This packages the three image-encode wins first developed for the lightx2v
distill (``pipeline_wan_distill.py``) into a mixin so other I2V variants
(AniSora, LoRA, ...) can opt in without duplicating the logic:

1. **Truncated encode** — for I2V the conditioning image sits at frame 0 and the
   remaining pixel frames are zeros, which the causal Wan VAE encoder maps to a
   steady-state latent. Encoding only the first ``ENCODE_FRAMES`` (default 33)
   frames and replicating the tail latent is bit-equivalent to encoding all 81,
   and lands on the swept conv3d blocking entries instead of the slow default.
2. **Fast swept VAE encoder** — rebuild the encoder at the real resolution +
   truncated-T chunk size so it keys the swept conv3d blockings. Optionally cap
   ``T_out_block`` at 1 to avoid the temporal-blocking artifact few-step models
   are sensitive to.
3. **On-device conditioning assembly** — transfer only the conditioned frame(s)
   and build the zero frames directly on device via binary doubling, avoiding
   the large host->device transfer of a mostly-zero pixel video.

Each optimization is gated behind a per-model env flag (names are class
attributes so each pipeline gets its own switches). The mixin must precede
``WanPipelineI2V`` in the MRO so its ``_encode_image_condition`` /
``_encode_frames_for`` overrides win while ``super()`` still resolves to the
base host-build path.
"""
from __future__ import annotations

import os

import torch

import ttnn
from models.tt_dit.models.vae.vae_wan2_1 import WanEncoder
from models.tt_dit.utils import cache
from models.tt_dit.utils.conv3d import (
    conv3d_blocking_hash,
    conv_pad_height,
    conv_pad_in_channels,
    conv_pad_width,
    set_force_t_out_block_1,
)
from models.tt_dit.utils.tensor import bf16_tensor_2dshard


class FastImageEncodeMixin:
    """Mixin providing truncated / swept / on-device image-encode for I2V.

    Subclasses set the env-flag names and (optionally) override ``ENCODE_FRAMES``
    / ``ENCODER_T_CHUNK_BY_MESH``. Call :meth:`_maybe_install_fast_vae_encoder`
    at the end of ``__init__`` (after the base VAE encoder has been built).
    """

    # Pixel frames to VAE-encode for conditioning (rest are zeros -> steady-state
    # latent replicated downstream). 33 pixel frames -> 9 latent frames.
    ENCODE_FRAMES = 33
    # Forward chunk size per mesh shape; must match a swept (mesh, T) blocking table.
    ENCODER_T_CHUNK_BY_MESH = {(4, 8): 16}

    # Per-model env-flag names (override in the subclass).
    FAST_ENCODER_FLAG = "WAN_FAST_VAE_ENCODER"
    ENCODER_T_OUT_1_FLAG = "WAN_ENCODER_T_OUT_1"
    ONDEVICE_COND_FLAG = "WAN_ONDEVICE_COND"

    def _maybe_install_fast_vae_encoder(self) -> None:
        """Install the swept fast VAE encoder if ``FAST_ENCODER_FLAG`` is set."""
        if os.environ.get(self.FAST_ENCODER_FLAG, "0") == "1":
            self._install_fast_vae_encoder()

    def _encode_frames_for(self, num_frames: int, max_cond_pos: int) -> int:
        # Truncation is tied to the swept fast encoder (its chunk size matches the
        # swept (mesh, T) blocking tables), so gate it on the same flag. With the
        # flag unset, defer to the base (encode all frames) — keeping a clean
        # full-encode baseline for A/B validation.
        if os.environ.get(self.FAST_ENCODER_FLAG, "0") != "1":
            return super()._encode_frames_for(num_frames, max_cond_pos)
        # Encode only the first ~ENCODE_FRAMES pixel frames (but always enough to
        # cover the furthest conditioned frame); the rest are zeros -> steady-state
        # latent, replicated downstream by the base prepare_latents.
        return max(min(self.ENCODE_FRAMES, num_frames), min(max_cond_pos + 1, num_frames))

    def _build_fast_vae_encoder(self, *, force_t_out_block_1: bool = False) -> WanEncoder:
        """Build a VAE encoder at the real resolution + truncated-T chunk size so
        it keys the swept conv3d blockings (instead of the slow H/W=0 default).

        ``force_t_out_block_1`` caps every encoder conv's ``T_out_block`` at 1,
        which keeps the fast C/H/W blocking but avoids the temporal-blocking
        artifact few-step models are sensitive to. The blocking cache key
        (``conv3d_blocking_hash``) only depends on ``C_in_block``, which the cap
        doesn't change, so prepared weights are shared with the un-capped build.
        """
        mesh_shape = tuple(self.mesh_device.shape)
        chunk = self.ENCODER_T_CHUNK_BY_MESH.get(mesh_shape, self.ENCODE_FRAMES)

        set_force_t_out_block_1(force_t_out_block_1)
        try:
            enc = WanEncoder(
                base_dim=self._vae.config.base_dim,
                in_channels=self._vae.config.in_channels,
                z_dim=self._vae.config.z_dim,
                dim_mult=self._vae.config.dim_mult,
                num_res_blocks=self._vae.config.num_res_blocks,
                attn_scales=self._vae.config.attn_scales,
                temperal_downsample=self._vae.config.temperal_downsample,
                is_residual=self._vae.config.is_residual,
                mesh_device=self.mesh_device,
                ccl_manager=self.vae_ccl_manager,
                parallel_config=self.vae_parallel_config,
                height=self._height,
                width=self._width,
                encoder_t_chunk_size=chunk,
            )
        finally:
            set_force_t_out_block_1(False)

        # Cache prepared weights under a blocking-specific subfolder so we don't
        # reuse the slow-blocking weights the base loaded under "vae_encoder".
        blocking_key = conv3d_blocking_hash(enc)
        subfolder = f"vae_encoder_{blocking_key}" if blocking_key else "vae_encoder"
        cache.load_model(
            enc,
            model_name=os.path.basename(self.checkpoint_name),
            subfolder=subfolder,
            parallel_config=self.vae_parallel_config,
            mesh_shape=mesh_shape,
            get_torch_state_dict=lambda: self._vae.torch_state_dict(),
        )
        return enc

    def _install_fast_vae_encoder(self) -> None:
        mesh_shape = tuple(self.mesh_device.shape)
        # Drives the forward chunking (base prepare_latents reads this).
        self._encoder_t_chunk_size = self.ENCODER_T_CHUNK_BY_MESH.get(mesh_shape, self.ENCODE_FRAMES)
        force = os.environ.get(self.ENCODER_T_OUT_1_FLAG, "0") == "1"
        self.tt_vae_encoder = self._build_fast_vae_encoder(force_t_out_block_1=force)

    def _encode_image_condition(self, image_prompt, *, enc_frames, height, width, dtype, device):
        """On-device conditioning assembly (gated by ``ONDEVICE_COND_FLAG``).

        The base path builds the full ``enc_frames`` pixel video on the host
        (mostly zeros for I2V) and ships it to the device — at 720p that
        host->device transfer dominates the image-encode stage. Here we instead
        transfer only the conditioned frame(s) and build the zero frames directly
        on-device via binary doubling. Falls back to the base host-build path
        when the flag is unset.
        """
        if os.environ.get(self.ONDEVICE_COND_FLAG, "0") != "1":
            return super()._encode_image_condition(
                image_prompt, enc_frames=enc_frames, height=height, width=width, dtype=dtype, device=device
            )

        shard_mapping = {
            self.vae_parallel_config.height_parallel.mesh_axis: 2,
            self.vae_parallel_config.width_parallel.mesh_axis: 3,
        }
        h_factor = self.vae_parallel_config.height_parallel.factor * self.vae_scale_factor_spatial
        w_factor = self.vae_parallel_config.width_parallel.factor * self.vae_scale_factor_spatial

        # Per conditioned frame: preprocess on host -> pad -> shard to device.
        cond_by_pos: dict[int, ttnn.Tensor] = {}
        logical_h = logical_w = None
        for image, frame_pos in image_prompt:
            img = self.video_processor.preprocess(image, height=height, width=width).to(
                torch.device("cpu"), dtype=torch.float32
            )  # [B, C, H, W]
            frame_BTHWC = img.unsqueeze(2).permute(0, 2, 3, 4, 1)  # [B, 1, H, W, C]
            frame_BTHWC = conv_pad_in_channels(frame_BTHWC)
            frame_BTHWC, logical_h = conv_pad_height(frame_BTHWC, h_factor)
            frame_BTHWC, logical_w = conv_pad_width(frame_BTHWC, w_factor)
            cond_by_pos[frame_pos] = bf16_tensor_2dshard(
                frame_BTHWC,
                self.mesh_device,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                shard_mapping=shard_mapping,
            )

        # Assemble the enc_frames video on-device: conditioned frames at their
        # positions, zero frames everywhere else (built by doubling a single
        # zero frame so we never allocate/transfer the full pixel video).
        tt_zero_1: ttnn.Tensor | None = None

        def _zero_run(n: int) -> ttnn.Tensor:
            nonlocal tt_zero_1
            if tt_zero_1 is None:
                tt_zero_1 = ttnn.zeros_like(next(iter(cond_by_pos.values())))
            z = tt_zero_1
            built = 1
            while built * 2 <= n:
                z = ttnn.concat([z, z], dim=1)
                built *= 2
            if built < n:
                z = ttnn.concat([z, z[:, : n - built, :, :, :]], dim=1)
            return z

        segments: list[ttnn.Tensor] = []
        zero_start: int | None = None
        for i in range(enc_frames):
            if i in cond_by_pos:
                if zero_start is not None:
                    segments.append(_zero_run(i - zero_start))
                    zero_start = None
                segments.append(cond_by_pos[i])
            elif zero_start is None:
                zero_start = i
        if zero_start is not None:
            segments.append(_zero_run(enc_frames - zero_start))

        tt_video_BTHWC = segments[0] if len(segments) == 1 else ttnn.concat(segments, dim=1)

        encoded = self._vae_encode_to_torch(tt_video_BTHWC, logical_h, logical_w, dtype)

        # Free device tensors. When concat produced a fresh buffer the inputs are
        # safe to drop; guard the single-segment case where tt_video aliases an input.
        if tt_video_BTHWC is not None and len(segments) > 1:
            ttnn.deallocate(tt_video_BTHWC)
        for tt in cond_by_pos.values():
            if not (len(segments) == 1 and tt is segments[0]):
                ttnn.deallocate(tt)
        if tt_zero_1 is not None and not (len(segments) == 1 and tt_zero_1 is segments[0]):
            ttnn.deallocate(tt_zero_1)
        return encoded
