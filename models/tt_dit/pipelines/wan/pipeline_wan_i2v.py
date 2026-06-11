# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

# Adapted from https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/wan/pipeline_wan.py

from __future__ import annotations

import os
from typing import List, NamedTuple, Optional, Union

import torch
from PIL import Image

import ttnn

from ...models.vae.vae_wan2_1 import WanEncoder
from ...utils import cache
from ...utils.conv3d import conv_pad_height, conv_pad_in_channels, conv_pad_width
from ...utils.tensor import bf16_tensor_2dshard, fast_device_to_host, unflatten
from .pipeline_wan import WanPipeline, WanPipelineConfig

_DEFAULT_I2V_CHECKPOINT = "Wan-AI/Wan2.2-I2V-A14B-Diffusers"


class ImagePrompt(NamedTuple):
    image: Image.Image
    frame_pos: int


class WanPipelineI2V(WanPipeline):
    def __init__(self, *, device: ttnn.MeshDevice, config: WanPipelineConfig) -> None:
        # initialize without warmup; we warm up below with a sample image_prompt.
        super().__init__(device=device, config=config, run_warmup=False)

        self.tt_vae_encoder = WanEncoder(
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
        )

        cache.load_model(
            self.tt_vae_encoder,
            model_name=os.path.basename(self.checkpoint_name),
            subfolder="vae_encoder",
            parallel_config=self.vae_parallel_config,
            mesh_shape=tuple(self.mesh_device.shape),
            get_torch_state_dict=lambda: self._vae.torch_state_dict(),
        )

        # Pixel-frame chunk size passed to the VAE encoder forward. Default 4
        # matches the encoder's own forward default (preserves base behavior);
        # subclasses (e.g. the distill) override this together with a real-H/W
        # encoder build to hit the swept conv3d blockings.
        self._encoder_t_chunk_size = 4

        # warmup buffers with a sample image_prompt sized to the target resolution.
        self(
            prompts=["warmup"],
            image_prompt=Image.new("RGB", (self._width, self._height)),
            num_inference_steps=2,
            guidance_scale=2 if config.cfg_enabled else 1,
            guidance_scale_2=2 if config.cfg_enabled else 1,
        )

    @classmethod
    def create_pipeline(
        cls,
        *,
        mesh_device: ttnn.MeshDevice,
        checkpoint_name: str = _DEFAULT_I2V_CHECKPOINT,
        height: int = 480,
        width: int = 832,
        num_frames: int = 81,
        cfg_enabled: bool = True,
        pipeline_class: type[WanPipeline] | None = None,
    ) -> WanPipelineI2V:
        config = WanPipelineConfig.default(
            mesh_shape=mesh_device.shape,
            checkpoint_name=checkpoint_name,
            height=height,
            width=width,
            num_frames=num_frames,
            cfg_enabled=cfg_enabled,
            model_type="i2v",
        )
        pipeline_class_ = pipeline_class or cls
        return pipeline_class_(device=mesh_device, config=config)

    def get_model_input(self, latents, cond_latents):
        """
        Adapter function to enable I2V. For base T2V, just return the latents.
        """
        latents = super().get_model_input(latents, None)
        z_dim = self._vae.config.z_dim
        t_size = latents.shape[-1]
        model_input = ttnn.concat(
            [unflatten(latents, -1, (t_size // z_dim, -1)), unflatten(cond_latents, -1, (t_size // z_dim, -1))],
            dim=-1,
        )
        return ttnn.reshape(model_input, (*tuple(latents.shape)[:-1], -1))

    def _encode_frames_for(self, num_frames: int, max_cond_pos: int) -> int:
        """Number of pixel frames to VAE-encode for image conditioning.

        Base encodes all ``num_frames``. Subclasses may truncate: for I2V the
        conditioning frames beyond the first are zeros, which the causal Wan VAE
        encoder maps to a steady-state latent, so the tail latent frames can be
        replicated instead of encoded (see ``prepare_latents``). The returned
        value must be ``>= max_cond_pos + 1`` so every conditioned frame is
        actually encoded.
        """
        del max_cond_pos
        return num_frames

    def prepare_latents(
        self,
        batch_size: int,
        image_prompt: Union[ImagePrompt, Image.Image, List[ImagePrompt]],
        num_channels_latents: int = 16,
        height: int = 480,
        width: int = 832,
        num_frames: int = 81,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        assert batch_size == 1, "Only batch size 1 is currently supported for I2V"

        if isinstance(image_prompt, ImagePrompt):
            image_prompt = [image_prompt]
        elif isinstance(image_prompt, Image.Image):
            image_prompt = [ImagePrompt(image=image_prompt, frame_pos=0)]

        latents, _ = super().prepare_latents(
            batch_size=batch_size,
            num_channels_latents=num_channels_latents,
            height=height,
            width=width,
            num_frames=num_frames,
            dtype=dtype,
            device=device,
        )

        latent_shape = latents.shape

        # Number of pixel frames to actually encode (subclasses may truncate;
        # base encodes all num_frames). The mask still spans the full num_frames.
        max_cond_pos = max(frame_pos for _, frame_pos in image_prompt)
        enc_frames = self._encode_frames_for(num_frames, max_cond_pos)

        # Initialize mask (spans the full num_frames; cheap host op).
        msk = torch.zeros(batch_size, num_frames, latent_shape[-2], latent_shape[-1])
        inserted_frames = set()
        for _, frame_pos in image_prompt:
            assert (
                frame_pos not in inserted_frames
            ), f"Frame position {frame_pos} already processed. Please remove duplicate frame positions."
            inserted_frames.add(frame_pos)
            msk[:, frame_pos, :, :] = 1

        # VAE-encode the conditioning frames -> latent (device->host, sliced to
        # logical H/W, cast to ``dtype``). Overridable: the base builds the full
        # pixel video on host and transfers it; subclasses may assemble the zero
        # frames on-device to avoid the large host->device transfer.
        encoded_video_torch = self._encode_image_condition(
            image_prompt, enc_frames=enc_frames, height=height, width=width, dtype=dtype, device=device
        )

        latents_mean = (
            torch.tensor(self._vae.config.latents_mean)
            .view(1, self._vae.config.z_dim, 1, 1, 1)
            .to(encoded_video_torch.device, encoded_video_torch.dtype)
        )
        latents_std = 1.0 / torch.tensor(self._vae.config.latents_std).view(1, self._vae.config.z_dim, 1, 1, 1).to(
            encoded_video_torch.device, encoded_video_torch.dtype
        )

        encoded_video_torch = (encoded_video_torch - latents_mean) * latents_std

        # If we truncated the pixel-frame encode (enc_frames < num_frames), the
        # remaining latent frames correspond to all-zero conditioning, which the
        # causal encoder produces as a steady-state latent. Replicate the last
        # encoded latent frame to fill the full latent-frame count. No-op for the
        # base path where enc_frames == num_frames.
        num_latent_frames = (num_frames - 1) // self.vae_scale_factor_temporal + 1
        encoded_latent_frames = encoded_video_torch.shape[2]
        if encoded_latent_frames < num_latent_frames:
            encoded_video_torch = torch.cat(
                [
                    encoded_video_torch,
                    encoded_video_torch[:, :, -1:, :, :].expand(
                        -1, -1, num_latent_frames - encoded_latent_frames, -1, -1
                    ),
                ],
                dim=2,
            )

        # Finalize mask setup into the latent space
        msk = torch.concat([torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]], dim=1)
        msk = msk.view(1, msk.shape[1] // 4, 4, latent_shape[-2], latent_shape[-1])
        msk = msk.transpose(1, 2)

        tt_y = torch.cat([msk, encoded_video_torch], dim=1)

        return latents, tt_y

    def _vae_encode_to_torch(self, tt_video_condition_BTHWC, logical_h, logical_w, dtype):
        """Run the VAE encoder on an already-padded/sharded BTHWC pixel video
        and bring the latent back to host (sliced to logical H/W, cast to
        ``dtype``). Shared by the host-build and on-device-build paths."""
        encoded_video_BCTHW, new_logical_h, new_logical_w = self.tt_vae_encoder(
            tt_video_condition_BTHWC, logical_h, logical_w=logical_w, encoder_t_chunk_size=self._encoder_t_chunk_size
        )
        concat_dims = [None, None]
        concat_dims[self.vae_parallel_config.height_parallel.mesh_axis] = 3
        concat_dims[self.vae_parallel_config.width_parallel.mesh_axis] = 4
        encoded_video_torch = fast_device_to_host(
            encoded_video_BCTHW,
            self.mesh_device,
            concat_dims,
            ccl_manager=self.vae_ccl_manager,
        )
        encoded_video_torch = encoded_video_torch[:, :, :, :new_logical_h, :new_logical_w]
        return encoded_video_torch.to(dtype=dtype)

    def _encode_image_condition(self, image_prompt, *, enc_frames, height, width, dtype, device):
        """Build the conditioning pixel video on host, transfer it to device,
        and VAE-encode it. Base/T2V behavior: allocates the full ``enc_frames``
        pixel video (mostly zeros) on the host and ships it to the device.

        Subclasses may override to assemble the zero frames directly on-device
        and only transfer the conditioned frame(s), avoiding the large
        host->device transfer (see ``WanDistillPipelineI2V``)."""
        video_condition = None
        for image, frame_pos in image_prompt:
            image = self.video_processor.preprocess(image, height=height, width=width).to(device, dtype=torch.float32)
            if video_condition is None:
                video_condition = image.new_zeros(image.shape[0], image.shape[1], enc_frames, height, width)
            if self._expand_timesteps:  # Unverified code path
                video_condition = image_prompt.unqueeze(2)
                assert len(image_prompt) == 1, "Only single image conditioning is supported for expand timesteps"
            else:
                video_condition[:, :, frame_pos, :, :] = image

        tt_video_condition_BTHWC = video_condition.permute(0, 2, 3, 4, 1)
        tt_video_condition_BTHWC = conv_pad_in_channels(tt_video_condition_BTHWC)
        tt_video_condition_BTHWC, logical_h = conv_pad_height(
            tt_video_condition_BTHWC, self.vae_parallel_config.height_parallel.factor * self.vae_scale_factor_spatial
        )
        tt_video_condition_BTHWC, logical_w = conv_pad_width(
            tt_video_condition_BTHWC, self.vae_parallel_config.width_parallel.factor * self.vae_scale_factor_spatial
        )
        tt_video_condition_BTHWC = bf16_tensor_2dshard(
            tt_video_condition_BTHWC,
            self.mesh_device,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            shard_mapping={
                self.vae_parallel_config.height_parallel.mesh_axis: 2,
                self.vae_parallel_config.width_parallel.mesh_axis: 3,
            },
        )
        return self._vae_encode_to_torch(tt_video_condition_BTHWC, logical_h, logical_w, dtype)
