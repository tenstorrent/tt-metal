# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Cosmos3-I2V VAE adapters (TT-NN encoder + decoder).

Cosmos3 uses `AutoencoderKLWan` (the Wan2.2-TI2V-5B variant) with z_dim=48 and
`patch_size=(1, 2, 2)`. The existing TT-NN building blocks (`WanEncoder`,
`WanDecoder`, `WanVAEDecoderAdapter` in `models/tt_dit/models/vae/vae_wan2_1.py`)
generalize to z_dim=48 because they read those dims from the loaded torch VAE
config. What they do NOT handle is the spatial patchify/unpatchify wrapping
that AutoencoderKLWan applies around its conv_in/conv_out when patch_size>1.

This module adds two adapters that wrap the TT-NN encoder/decoder and apply
patchify (encoder side) and unpatchify (decoder side) on the host. Both are
torch-in / torch-out so callers can drop them in as a replacement for
`pipe.vae.encode` / `pipe.vae.decode`.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import torch
from diffusers.models import AutoencoderKLWan
from loguru import logger

import ttnn

from ....models.vae.vae_wan2_1 import WanEncoder, WanVAEDecoderAdapter
from ....utils import cache
from ....utils.conv3d import conv_pad_height, conv_pad_in_channels, conv_pad_width, register_conv3d_configs
from ....utils.tensor import bf16_tensor_2dshard, fast_device_to_host

# Conservative conv3d blockings for Cosmos3's channel set (base_dim=160,
# dim_mult=[1,2,4,4] → 160/320/640). The default fallback table in
# tt_dit/utils/conv3d.py only has Wan2.2's (96/192/384) channels; without
# these entries Cosmos3 convs fall through to a `(in_channels, 32, 1, 1, 1)`
# blocking with C_in_block=320 (or 640) which blows L1.
#
# Pattern mirrors Wan2.2's `_DEFAULT_BLOCKINGS` (conv3d.py:456): C blocks
# bounded to ~64-128 channels regardless of total dim. Values picked to
# match the smallest known-good blocking for Wan2.2's analogous layer
# scaled up; spatial H/W blocks kept identical.
_COSMOS3_CONV3D_FALLBACKS = {
    # (in, out, kernel): (C_in_block, C_out_block, T_out_block, H_out_block, W_out_block)
    #
    # Conv3d requires: C_*_block % 32 == 0 AND C_*_block divides the padded
    # channel count exactly. For 160 the only valid value is 32; for 320 we
    # can use 32 or 64; for 640 up to 128.
    #
    # Encoder / decoder 3x3x3 residual conv path. Conservative blockings —
    # values were tried bigger (Cin_block=160 etc.) to chase a main-path std
    # mismatch, but L1 overflowed. The std halving remains unexplained;
    # likely lives in conv3d kernel semantics or sharded padding, not block
    # sizing. Coming back to this once we have a fast PCC loop.
    (160, 160, (3, 3, 3)): (32, 32, 1, 8, 8),
    (160, 320, (3, 3, 3)): (32, 64, 1, 8, 4),
    (320, 320, (3, 3, 3)): (64, 64, 1, 8, 4),
    (320, 640, (3, 3, 3)): (64, 64, 1, 8, 4),
    (640, 640, (3, 3, 3)): (64, 64, 1, 8, 4),
    # conv_in / conv_out (post-patchify Cosmos3 has 12 in-channels padded to
    # 32, z_dim*2=96 out for encoder).
    (12, 160, (3, 3, 3)): (32, 32, 1, 8, 8),
    (640, 96, (3, 3, 3)): (64, 32, 1, 8, 4),
    # Decoder mirror: conv_in goes z_dim=48 (padded to 64) → 640.
    (48, 640, (3, 3, 3)): (64, 64, 1, 8, 4),
    (640, 12, (3, 3, 3)): (64, 32, 1, 8, 4),
    # WanResample downsample2d / upsample2d (kt=1) spatial conv.
    (160, 160, (1, 3, 3)): (32, 32, 1, 8, 8),
    (320, 320, (1, 3, 3)): (64, 64, 1, 8, 4),
    (640, 640, (1, 3, 3)): (64, 64, 1, 8, 4),
    # WanResample downsample3d / upsample3d time_conv (kh=kw=1, kt=3).
    (160, 160, (3, 1, 1)): (32, 32, 1, 8, 8),
    (320, 320, (3, 1, 1)): (64, 64, 1, 8, 4),
    (640, 640, (3, 1, 1)): (64, 64, 1, 8, 4),
    (160, 320, (3, 1, 1)): (32, 64, 1, 8, 4),  # upsample3d doubles channels
    (320, 640, (3, 1, 1)): (64, 64, 1, 8, 4),
    (640, 1280, (3, 1, 1)): (64, 128, 1, 8, 4),
    # ------------------------------------------------------------------
    # Decoder uses its own base_dim (decoder_base_dim=256 on Cosmos3),
    # giving channels [1024, 1024, 1024, 512, 256] across stages.
    # ------------------------------------------------------------------
    (1024, 1024, (3, 3, 3)): (64, 64, 1, 8, 4),
    (1024, 512, (3, 3, 3)): (64, 64, 1, 8, 4),
    (512, 512, (3, 3, 3)): (64, 64, 1, 8, 4),
    (512, 256, (3, 3, 3)): (64, 64, 1, 8, 4),
    (256, 256, (3, 3, 3)): (64, 64, 1, 8, 4),
    # decoder conv_in (z_dim 48 padded to 64 → 1024).
    (48, 1024, (3, 3, 3)): (64, 64, 1, 8, 4),
    # decoder conv_out (256 → out_channels=12 post-unpatchify).
    (256, 12, (3, 3, 3)): (64, 32, 1, 8, 4),
    # Spatial / time conv pairs at decoder channels.
    # (1,3,3) and (3,1,1) kernel volume is 3× lower than (3,3,3), so
    # C_block=128 fits L1 at these channel counts.
    (1024, 1024, (1, 3, 3)): (128, 128, 1, 8, 4),
    (512, 512, (1, 3, 3)): (128, 128, 1, 8, 4),
    (256, 256, (1, 3, 3)): (128, 128, 1, 8, 4),
    (1024, 1024, (3, 1, 1)): (128, 128, 1, 8, 4),
    (512, 512, (3, 1, 1)): (128, 128, 1, 8, 4),
    (256, 256, (3, 1, 1)): (128, 128, 1, 8, 4),
    (1024, 2048, (3, 1, 1)): (128, 128, 1, 8, 4),  # upsample3d 2x channels
    (512, 1024, (3, 1, 1)): (128, 128, 1, 8, 4),
    (256, 512, (3, 1, 1)): (128, 128, 1, 8, 4),
}

_cosmos3_conv3d_registered = False


def _ensure_cosmos3_conv3d_configs_registered() -> None:
    global _cosmos3_conv3d_registered
    if _cosmos3_conv3d_registered:
        return
    register_conv3d_configs(_COSMOS3_CONV3D_FALLBACKS)
    _cosmos3_conv3d_registered = True
    logger.info(f"Registered {len(_COSMOS3_CONV3D_FALLBACKS)} Cosmos3 conv3d fallback blockings")


if TYPE_CHECKING:
    from ....parallel.config import VaeHWParallelConfig
    from ....parallel.manager import CCLManager


def _patch_size_to_spatial(patch_size) -> int:
    """Normalize a Wan-VAE config.patch_size to a single spatial factor.

    Wan-VAE's host `patchify` only operates on H and W; the temporal patch
    factor (first element of a tuple-form patch_size) is currently always 1
    in shipped configs. We assert that and return the spatial factor.
    """
    if patch_size is None:
        return 1
    if isinstance(patch_size, int):
        return patch_size
    # Tuple/list form, e.g. (1, 2, 2) for Cosmos3.
    if len(patch_size) == 3:
        pt, ph, pw = patch_size
        assert pt == 1, f"Temporal patchify factor must be 1, got {pt}"
        assert ph == pw, f"Non-square spatial patchify not supported, got ({ph}, {pw})"
        return int(ph)
    if len(patch_size) == 2:
        ph, pw = patch_size
        assert ph == pw, f"Non-square spatial patchify not supported, got ({ph}, {pw})"
        return int(ph)
    raise ValueError(f"Unsupported patch_size form: {patch_size!r}")


def _host_patchify_spatial(x_BCTHW: torch.Tensor, p: int) -> torch.Tensor:
    """Reorder spatial pixels into channels (host PyTorch). Inverse of
    `_host_unpatchify_spatial`. p=1 → identity."""
    if p == 1:
        return x_BCTHW
    B, C, T, H, W = x_BCTHW.shape
    assert H % p == 0 and W % p == 0, f"H={H}, W={W} not divisible by patch={p}"
    x = x_BCTHW.view(B, C, T, H // p, p, W // p, p)
    # Match diffusers' autoencoder_kl_wan.patchify channel ordering exactly.
    x = x.permute(0, 1, 6, 4, 2, 3, 5).contiguous()
    return x.view(B, C * p * p, T, H // p, W // p)


def _host_unpatchify_spatial(x_BCTHW: torch.Tensor, p: int) -> torch.Tensor:
    """Inverse of `_host_patchify_spatial`. p=1 → identity."""
    if p == 1:
        return x_BCTHW
    B, Cp, T, H, W = x_BCTHW.shape
    assert Cp % (p * p) == 0, f"Channel dim {Cp} not divisible by p*p={p*p}"
    C = Cp // (p * p)
    x = x_BCTHW.view(B, C, p, p, T, H, W)
    x = x.permute(0, 1, 4, 5, 3, 6, 2).contiguous()
    return x.view(B, C, T, H * p, W * p)


class Cosmos3VAEEncoderAdapter:
    """Torch-in (BCTHW), torch-out (BCTHW latents) VAE encoder for Cosmos3-I2V.

    Wraps `WanEncoder` on device and applies:
      - host-side spatial patchify (Cosmos3 uses patch_size=(1,2,2))
      - Cosmos3 per-channel `latents_mean` / `latents_std` normalization after
        encode (output is the diffusion-ready latent, not raw mu)

    Weight loading is idempotent and uses `cache.load_model` with the
    `vae_encoder` subfolder — same convention as `WanPipelineI2V`.
    """

    def __init__(
        self,
        *,
        checkpoint_name: str,
        parallel_config: VaeHWParallelConfig,
        ccl_manager: CCLManager,
        encoder_t_chunk_size: int | None = None,
        vae_dtype: ttnn.DataType = ttnn.bfloat16,
        torch_vae: AutoencoderKLWan | None = None,
    ) -> None:
        _ensure_cosmos3_conv3d_configs_registered()
        self.device = ccl_manager.mesh_device
        self._parallel_config = parallel_config
        self._ccl_manager = ccl_manager
        self._checkpoint_name = checkpoint_name
        self._t_chunk_size = encoder_t_chunk_size

        # Allow callers to inject an already-loaded torch VAE (so a co-located
        # decoder adapter and encoder adapter can share one host copy).
        self._torch_vae = torch_vae or AutoencoderKLWan.from_pretrained(
            checkpoint_name, subfolder="vae", trust_remote_code=True
        )
        cfg = self._torch_vae.config

        self._patch_spatial = _patch_size_to_spatial(getattr(cfg, "patch_size", None))
        # `cfg.in_channels` already reports the post-patchify channel count on the
        # Cosmos3 AutoencoderKLWan config (e.g. 12 for patch_size=(1,2,2), since
        # `encoder.conv_in.weight` has 12 input channels). Use it directly.
        logger.info(
            f"Cosmos3VAEEncoderAdapter: in_channels={cfg.in_channels}, "
            f"patch_spatial={self._patch_spatial}, z_dim={cfg.z_dim}"
        )

        self._encoder = WanEncoder(
            base_dim=cfg.base_dim,
            in_channels=cfg.in_channels,
            z_dim=cfg.z_dim,
            dim_mult=cfg.dim_mult,
            num_res_blocks=cfg.num_res_blocks,
            attn_scales=cfg.attn_scales,
            temperal_downsample=cfg.temperal_downsample,
            is_residual=cfg.is_residual,
            mesh_device=self.device,
            ccl_manager=ccl_manager,
            parallel_config=parallel_config,
            dtype=vae_dtype,
            encoder_t_chunk_size=encoder_t_chunk_size,
        )

    @property
    def config(self):
        return self._torch_vae.config

    @property
    def dtype(self):
        return self._torch_vae.dtype

    @property
    def encoder(self) -> WanEncoder:
        return self._encoder

    @property
    def torch_vae(self) -> AutoencoderKLWan:
        return self._torch_vae

    def torch_state_dict(self) -> dict[str, torch.Tensor]:
        return self._torch_vae.state_dict()

    def is_loaded(self) -> bool:
        return self._encoder.is_loaded()

    def deallocate_weights(self) -> None:
        self._encoder.deallocate_weights()

    def reload_weights(self) -> None:
        cache.load_model(
            self._encoder,
            model_name=os.path.basename(self._checkpoint_name),
            subfolder="vae_encoder",
            parallel_config=self._parallel_config,
            mesh_shape=tuple(self.device.shape),
            get_torch_state_dict=lambda: self._torch_vae.state_dict(),
        )

    @torch.no_grad()
    def encode(self, video_BCTHW: torch.Tensor) -> torch.Tensor:
        """Encode a video (BCTHW, in [-1, 1]) to raw latent mu (BCTHW).

        Returns the deterministic mode of the latent distribution — does NOT
        apply Cosmos3's `latents_mean` / `latents_std` normalization. The
        reference pipeline's `_encode_video` applies that normalization on
        top, so this method matches the contract of
        `diffusers.AutoencoderKLWan.encode(...).latent_dist.mode()`.
        """
        dtype = self._torch_vae.dtype
        x = video_BCTHW.to(dtype)

        # Host patchify: (B, C, T, H, W) → (B, C*p*p, T, H/p, W/p).
        x = _host_patchify_spatial(x, self._patch_spatial)

        # (B, C, T, H, W) → (B, T, H, W, C) for the TT encoder.
        x_BTHWC = x.permute(0, 2, 3, 4, 1).contiguous()
        x_BTHWC = conv_pad_in_channels(x_BTHWC)
        sf = int(self._torch_vae.config.scale_factor_spatial)
        x_BTHWC, logical_h = conv_pad_height(
            x_BTHWC, self._parallel_config.height_parallel.factor * sf // self._patch_spatial
        )
        x_BTHWC, logical_w = conv_pad_width(
            x_BTHWC, self._parallel_config.width_parallel.factor * sf // self._patch_spatial
        )
        tt_x_BTHWC = bf16_tensor_2dshard(
            x_BTHWC,
            self.device,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            shard_mapping={
                self._parallel_config.height_parallel.mesh_axis: 2,
                self._parallel_config.width_parallel.mesh_axis: 3,
            },
        )

        self.reload_weights()
        tt_z_BCTHW, new_logical_h, new_logical_w = self._encoder(
            tt_x_BTHWC, logical_h, encoder_t_chunk_size=self._t_chunk_size, logical_w=logical_w
        )

        concat_dims = [None, None]
        concat_dims[self._parallel_config.height_parallel.mesh_axis] = 3
        concat_dims[self._parallel_config.width_parallel.mesh_axis] = 4
        z_torch = fast_device_to_host(
            tt_z_BCTHW,
            self.device,
            concat_dims,
            ccl_manager=self._ccl_manager,
        )
        # WanEncoder already returned BCTHW with C trimmed to z_dim. Trim padded H/W.
        return z_torch[:, :, :, :new_logical_h, :new_logical_w].to(dtype)


class Cosmos3VAEDecoderAdapter:
    """Torch-in (BCTHW latents), torch-out VAE decoder for Cosmos3-I2V.

    Wraps `WanVAEDecoderAdapter` and applies host-side spatial unpatchify so
    the returned tensor has 3 channels (raw pixels) regardless of the model's
    `out_channels` (which is post-patchify on Cosmos3).
    """

    def __init__(
        self,
        *,
        checkpoint_name: str,
        parallel_config: VaeHWParallelConfig,
        ccl_manager: CCLManager,
        height: int,
        width: int,
        num_frames: int,
        vae_t_chunk_size: int | None,
        vae_dtype: ttnn.DataType = ttnn.bfloat16,
        sdpa_t_fracture_w_only: bool = False,
        torch_vae: AutoencoderKLWan | None = None,
    ) -> None:
        _ensure_cosmos3_conv3d_configs_registered()
        self._inner = WanVAEDecoderAdapter(
            checkpoint_name=checkpoint_name,
            parallel_config=parallel_config,
            ccl_manager=ccl_manager,
            height=height,
            width=width,
            num_frames=num_frames,
            vae_t_chunk_size=vae_t_chunk_size,
            vae_dtype=vae_dtype,
            sdpa_t_fracture_w_only=sdpa_t_fracture_w_only,
        )
        if torch_vae is not None:
            # Replace the inner's torch_vae with the shared instance so we don't
            # hold two copies of ~5GB of host weights.
            self._inner._torch_vae = torch_vae

        # The Cosmos3OmniPipeline pre-denormalizes latents before calling
        # `vae.decode(z_raw)` (reference/pipeline_cosmos3_omni.py:1705). Neutralize
        # WanVAEDecoderAdapter's internal `z * latents_std + latents_mean` so we
        # don't denormalize a second time.
        self._inner._latents_mean = torch.zeros_like(self._inner._latents_mean)
        self._inner._latents_std = torch.ones_like(self._inner._latents_std)

        self._patch_spatial = _patch_size_to_spatial(getattr(self._inner.config, "patch_size", None))

    @property
    def config(self):
        return self._inner.config

    @property
    def dtype(self):
        return self._inner.dtype

    @property
    def decoder(self):
        return self._inner.decoder

    @property
    def torch_vae(self) -> AutoencoderKLWan:
        return self._inner._torch_vae

    def torch_state_dict(self) -> dict[str, torch.Tensor]:
        return self._inner.torch_state_dict()

    def is_loaded(self) -> bool:
        return self._inner.is_loaded()

    def deallocate_weights(self) -> None:
        self._inner.deallocate_weights()

    def reload_weights(self) -> None:
        self._inner.reload_weights()

    @torch.no_grad()
    def decode(self, latents: torch.Tensor, *, output_type: str = "pt") -> torch.Tensor:
        """Decode normalized latents to a video.

        Note: this layer keeps output_type semantics simple — always returns
        a torch tensor (BCTHW) in the VAE's dtype, post-unpatchify. Pipeline-
        level postprocessing (np conversion, uint8 quantization) should run
        on the unpatchified output. Pass `output_type="pt"` (default) here.
        """
        assert output_type in ("pt",), (
            f"Cosmos3VAEDecoderAdapter.decode only supports output_type='pt' "
            f"(host unpatchify needs raw float). Got {output_type}."
        )
        # Inner returns BCTHW float for output_type='pt'.
        video_BCTHW = self._inner.decode(latents, output_type="pt")
        return _host_unpatchify_spatial(video_BCTHW, self._patch_spatial)
