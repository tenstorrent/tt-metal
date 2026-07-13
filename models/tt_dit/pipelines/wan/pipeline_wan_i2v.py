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

        # Initialize mask
        msk = torch.zeros(batch_size, num_frames, latent_shape[-2], latent_shape[-1])
        inserted_frames = set()
        ## Encode image
        # Convert image to tensor
        video_condition = None
        for image, frame_pos in image_prompt:
            assert (
                frame_pos not in inserted_frames
            ), f"Frame position {frame_pos} already processed. Please remove duplicate frame positions."
            inserted_frames.add(frame_pos)
            image = self.video_processor.preprocess(image, height=height, width=width).to(device, dtype=torch.float32)

            if video_condition is None:
                video_condition = image.new_zeros(image.shape[0], image.shape[1], num_frames, height, width)
            if self._expand_timesteps:  # Unverified code path
                video_condition = image_prompt.unqueeze(2)
                assert len(image_prompt) == 1, "Only single image conditioning is supported for expand timesteps"
            else:
                video_condition[:, :, frame_pos, :, :] = image
            msk[:, frame_pos, :, :] = 1

        # Convert to ttnn
        tt_video_condition_BTHWC = video_condition.permute(0, 2, 3, 4, 1)
        # --> Do the required padding and sharding. Verify if this is needed.
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

        encoded_video_BCTHW, new_logical_h, new_logical_w = self.tt_vae_encoder(
            tt_video_condition_BTHWC, logical_h, logical_w=logical_w
        )

        # convert to torch
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
        encoded_video_torch = encoded_video_torch.to(dtype=dtype)

        latents_mean = (
            torch.tensor(self._vae.config.latents_mean)
            .view(1, self._vae.config.z_dim, 1, 1, 1)
            .to(encoded_video_torch.device, encoded_video_torch.dtype)
        )
        latents_std = 1.0 / torch.tensor(self._vae.config.latents_std).view(1, self._vae.config.z_dim, 1, 1, 1).to(
            encoded_video_torch.device, encoded_video_torch.dtype
        )

        encoded_video_torch = (encoded_video_torch - latents_mean) * latents_std

        # Finalize mask setup into the latent space
        msk = torch.concat([torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]], dim=1)
        msk = msk.view(1, msk.shape[1] // 4, 4, latent_shape[-2], latent_shape[-1])
        msk = msk.transpose(1, 2)

        tt_y = torch.cat([msk, encoded_video_torch], dim=1)

        return latents, tt_y
