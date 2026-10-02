# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""On-device VAE for the Qwen-Image-Edit pipeline.

``DeviceVAE`` is a drop-in for the reference ``AutoencoderKLQwenImage`` that runs the Qwen-Image VAE
on the Galaxy. It exposes the slice of the HF VAE surface the edit pipeline touches -- ``.config``,
``.dtype``, ``.encode(...)`` (``.latent_dist.mode()``) and ``.decode(...)`` (``.sample``) -- so the
reference host loop can drive it unchanged.

* **decode** reuses the shared, already-validated ``QwenImageVAEDecoderAdapter`` (a WAN-architecture
  decoder). The reference pipeline unnormalizes latents (``* std + mean``) *before* calling decode, so
  we feed the decoder network directly and skip the adapter's own normalization step.
* **encode** wraps the shared ``WanEncoder`` (the Qwen-Image VAE is WAN-architecture). It returns the
  distribution *mode* (mean); the reference pipeline applies the ``(z - mean) / std`` normalization
  afterwards, so the encoder emits raw latents (no folded norm).

Nothing in the shared tt_dit tree is modified; this composes the existing shared modules.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from diffusers.models.autoencoders.autoencoder_kl_qwenimage import AutoencoderKLQwenImage

import ttnn
from models.tt_dit.models.vae.vae_qwenimage import QwenImageVAEDecoderAdapter
from models.tt_dit.models.vae.vae_wan2_1 import WanEncoder
from models.tt_dit.parallel.config import ParallelFactor, VaeHWParallelConfig
from models.tt_dit.utils.conv3d import conv_pad_height, conv_pad_in_channels, conv_pad_width
from models.tt_dit.utils.tensor import typed_tensor_2dshard

if TYPE_CHECKING:
    from models.tt_dit.parallel.manager import CCLManager


class _ModeDist:
    """Minimal stand-in for a DiagonalGaussianDistribution exposing only ``mode()``."""

    def __init__(self, latents: torch.Tensor) -> None:
        self._latents = latents

    def mode(self) -> torch.Tensor:
        return self._latents

    def sample(self, _generator=None) -> torch.Tensor:  # noqa: ANN001 - argmax path is what the edit pipeline uses
        return self._latents


class _EncoderOutput:
    def __init__(self, latents: torch.Tensor) -> None:
        self.latent_dist = _ModeDist(latents)


class _DecoderOutput:
    def __init__(self, sample: torch.Tensor) -> None:
        self.sample = sample


class DeviceVAE:
    """Device-backed replacement for ``AutoencoderKLQwenImage`` (encode + decode on the Galaxy)."""

    def __init__(
        self,
        *,
        checkpoint_name: str,
        mesh_device: ttnn.MeshDevice,
        ccl_manager: CCLManager,
        height_axis: int = 0,
        width_axis: int = 1,
        device_encode: bool = True,
    ) -> None:
        torch_vae = AutoencoderKLQwenImage.from_pretrained(checkpoint_name, subfolder="vae")
        assert isinstance(torch_vae, AutoencoderKLQwenImage)

        # HF-visible surface.
        self.config = torch_vae.config
        self.dtype = torch_vae.dtype

        self._mesh = mesh_device
        self._h_axis = height_axis
        self._w_axis = width_axis
        self._device_encode = device_encode
        self._torch_vae = torch_vae  # kept for host encode fallback + config

        self._vae_parallel_config = VaeHWParallelConfig(
            height_parallel=ParallelFactor(factor=tuple(mesh_device.shape)[height_axis], mesh_axis=height_axis),
            width_parallel=ParallelFactor(factor=tuple(mesh_device.shape)[width_axis], mesh_axis=width_axis),
        )

        # --- decoder (reuse the shared, validated adapter) ---
        self._dec_adapter = QwenImageVAEDecoderAdapter(
            checkpoint_name=checkpoint_name,
            parallel_config=self._vae_parallel_config,
            ccl_manager=ccl_manager,
            use_torch=False,
        )
        self._dec_adapter.reload_weights()
        # Drive the underlying WanDecoder directly: unlike the shared adapter's prepare_input /
        # postprocess_output (height-only), we need width padding + width trim for width_factor > 1.
        self._wan_decoder = self._dec_adapter._decoder.wan_decoder  # noqa: SLF001

        # --- encoder (wrap the shared WanEncoder) ---
        self._encoder: WanEncoder | None = None
        if device_encode:
            c = torch_vae.config
            self._encoder = WanEncoder(
                base_dim=c.base_dim,
                in_channels=3,
                z_dim=c.z_dim,
                dim_mult=c.dim_mult,
                num_res_blocks=c.num_res_blocks,
                attn_scales=c.attn_scales or [],
                temperal_downsample=c.temperal_downsample,
                is_residual=bool(getattr(c, "is_residual", False)),
                mesh_device=mesh_device,
                ccl_manager=ccl_manager,
                parallel_config=self._vae_parallel_config,
                dtype=ttnn.bfloat16,
            )
            self._encoder.load_torch_state_dict(torch_vae.state_dict())

    # ------------------------------------------------------------------ encode
    @torch.no_grad()
    def encode(self, image: torch.Tensor):  # noqa: ANN201
        """``image``: [B, 3, T, H, W] (reference passes T=1). Returns an object with ``.latent_dist``."""
        if not self._device_encode:
            vae_dtype = next(self._torch_vae.parameters()).dtype
            return self._torch_vae.encode(image.to(vae_dtype))  # diffusers AutoencoderKLOutput

        # retrieve_latents() reads ``.latent_dist.mode()`` off the returned object directly.
        return _EncoderOutput(self._encode_device(image))

    def _encode_device(self, image: torch.Tensor) -> torch.Tensor:
        # BCTHW -> BTHWC, pad channels / height / width, shard across (H,W).
        x = image.to(torch.float32).permute(0, 2, 3, 4, 1)
        x = conv_pad_in_channels(x)
        hf_factor = self._vae_parallel_config.height_parallel.factor * 8
        wf_factor = self._vae_parallel_config.width_parallel.factor * 8
        x, logical_h = conv_pad_height(x, hf_factor)
        x, logical_w = conv_pad_width(x, wf_factor)
        tt_in = typed_tensor_2dshard(
            x,
            self._mesh,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            shard_mapping={self._h_axis: 2, self._w_axis: 3},
            dtype=ttnn.bfloat16,
        )
        tt_out, new_logical_h, new_logical_w = self._encoder(tt_in, logical_h, logical_w=logical_w)

        concat_dims = [None, None]
        concat_dims[self._h_axis] = 3  # BCTHW: H at dim 3
        concat_dims[self._w_axis] = 4  # BCTHW: W at dim 4
        latents = ttnn.to_torch(
            tt_out,
            mesh_composer=ttnn.ConcatMesh2dToTensor(self._mesh, mesh_shape=tuple(self._mesh.shape), dims=concat_dims),
        )
        # Trim conv padding back to logical content.
        latents = latents[:, : self.config.z_dim, :, :new_logical_h, :new_logical_w]
        return latents

    # ------------------------------------------------------------------ decode
    @torch.no_grad()
    def decode(self, latents: torch.Tensor, return_dict: bool = False):  # noqa: ANN201
        """``latents``: [B, z, 1, H, W], already unnormalized by the reference loop. Returns ``.sample`` 5D."""
        # BCTHW -> BTHWC, pad channels + height + width, shard across (H, W). Mirrors the validated
        # WanDecoder test prep so width_factor > 1 reconstructs correctly.
        x = latents.to(torch.float32).permute(0, 2, 3, 4, 1)  # [B, T=1, H, W, C]
        x = conv_pad_in_channels(x)
        x, logical_h = conv_pad_height(x, self._vae_parallel_config.height_parallel.factor)
        x, logical_w = conv_pad_width(x, self._vae_parallel_config.width_parallel.factor)
        tt_in = typed_tensor_2dshard(
            x,
            self._mesh,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            shard_mapping={self._h_axis: 2, self._w_axis: 3},
            dtype=ttnn.bfloat16,
        )

        tt_out, new_logical_h, new_logical_w = self._wan_decoder(tt_in, logical_h, logical_w=logical_w)

        concat_dims = [None, None]
        concat_dims[self._h_axis] = 3  # BCTHW: H at dim 3
        concat_dims[self._w_axis] = 4  # BCTHW: W at dim 4
        sample = ttnn.to_torch(
            tt_out,
            mesh_composer=ttnn.ConcatMesh2dToTensor(self._mesh, mesh_shape=tuple(self._mesh.shape), dims=concat_dims),
        )
        # Trim conv padding back to the real decoded extent (image space).
        sample = sample[:, :3, :, :new_logical_h, :new_logical_w]

        if return_dict:
            return _DecoderOutput(sample)
        return (sample,)
