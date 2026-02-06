# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

# Adapted from https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/wan/pipeline_wan.py

from typing import List, NamedTuple, Optional, Union

import PIL
import torch
from diffusers.schedulers import UniPCMultistepScheduler

import ttnn

from ...models.vae.vae_wan2_1 import WanEncoder
from ...utils.conv3d import conv_pad_height, conv_pad_in_channels
from ...utils.tensor import bf16_tensor_2dshard
from .pipeline_wan import WanPipeline


class ImagePrompt(NamedTuple):
    image: PIL.Image.Image
    frame_pos: int


class WanPipelineI2V(WanPipeline):
    def __init__(self, *args, **kwargs):
        # Update I2V specific defaults
        if "checkpoint_name" not in kwargs:
            kwargs["checkpoint_name"] = "Wan-AI/Wan2.2-I2V-A14B-Diffusers"
        if "scheduler" not in kwargs:
            kwargs["scheduler"] = UniPCMultistepScheduler.from_pretrained(
                kwargs["checkpoint_name"], subfolder="scheduler", trust_remote_code=True
            )

        super().__init__(*args, model_type="i2v", **kwargs)

        self.tt_encoder = WanEncoder(
            base_dim=self.vae.config.base_dim,
            in_channels=self.vae.config.in_channels,
            z_dim=self.vae.config.z_dim,
            dim_mult=self.vae.config.dim_mult,
            num_res_blocks=self.vae.config.num_res_blocks,
            attn_scales=self.vae.config.attn_scales,
            temperal_downsample=self.vae.config.temperal_downsample,
            is_residual=self.vae.config.is_residual,
            mesh_device=self.mesh_device,
            ccl_manager=self.vae_ccl_manager,
            parallel_config=self.vae_parallel_config,
        )

        self.tt_encoder.load_state_dict(self.vae.state_dict())

    @staticmethod
    def create_pipeline(*args, **kwargs):
        # Update I2V specific defaults
        if "checkpoint_name" not in kwargs:
            kwargs["checkpoint_name"] = "Wan-AI/Wan2.2-I2V-A14B-Diffusers"
        if "scheduler" not in kwargs:
            kwargs["scheduler"] = UniPCMultistepScheduler.from_pretrained(
                kwargs["checkpoint_name"], subfolder="scheduler", trust_remote_code=True
            )
        return WanPipeline.create_pipeline(*args, pipeline_class=WanPipelineI2V, **kwargs)

    def get_model_input(self, latents, cond_latents):
        """
        Adapter function to enable I2V. For base T2V, just return the latents.
        """
        # Reshape to make the channel last
        U, B, NPad, T_size = latents.shape
        # break out the channels for processing
        latents = latents.reshape(U, B, NPad, T_size // self.vae.config.z_dim, -1)
        cond_latents = cond_latents.reshape(U, B, NPad, T_size // self.vae.config.z_dim, -1)

        # concatenate the latents and cond_latents
        model_input = torch.cat([latents, cond_latents], dim=-1).reshape(U, B, NPad, -1)
        return model_input

    def prepare_latents(
        self,
        batch_size: int,
        image_prompt: Union[ImagePrompt, PIL.Image.Image, List[ImagePrompt]],
        num_channels_latents: int = 16,
        height: int = 480,
        width: int = 832,
        num_frames: int = 81,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        latents: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        assert batch_size == 1, "Only batch size 1 is currently supported for I2V"

        if isinstance(image_prompt, ImagePrompt):
            image_prompt = [image_prompt]
        elif isinstance(image_prompt, PIL.Image.Image):
            image_prompt = [ImagePrompt(image=image_prompt, frame_pos=0)]

        latents, _ = super().prepare_latents(
            batch_size=batch_size,
            num_channels_latents=num_channels_latents,
            height=height,
            width=width,
            num_frames=num_frames,
            dtype=dtype,
            device=device,
            latents=latents,
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
            if self.config.expand_timesteps:  # Unverified code path
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
        tt_video_condition_BTHWC = bf16_tensor_2dshard(
            tt_video_condition_BTHWC,
            self.mesh_device,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            shard_mapping={
                self.vae_parallel_config.height_parallel.mesh_axis: 2,
                self.vae_parallel_config.width_parallel.mesh_axis: 3,
            },
        )

        encoded_video_BCTHW, new_logical_h = self.tt_encoder(tt_video_condition_BTHWC, logical_h)

        # convert to torch
        concat_dims = [None, None]
        concat_dims[self.vae_parallel_config.height_parallel.mesh_axis] = 3
        concat_dims[self.vae_parallel_config.width_parallel.mesh_axis] = 4
        encoded_video_torch = ttnn.to_torch(
            encoded_video_BCTHW,
            mesh_composer=ttnn.ConcatMesh2dToTensor(
                self.mesh_device, mesh_shape=tuple(self.mesh_device.shape), dims=concat_dims
            ),
        )
        encoded_video_torch = encoded_video_torch[:, :, :, :new_logical_h, :]
        encoded_video_torch = encoded_video_torch.to(dtype=dtype)

        latents_mean = (
            torch.tensor(self.vae.config.latents_mean)
            .view(1, self.vae.config.z_dim, 1, 1, 1)
            .to(encoded_video_torch.device, encoded_video_torch.dtype)
        )
        latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(1, self.vae.config.z_dim, 1, 1, 1).to(
            encoded_video_torch.device, encoded_video_torch.dtype
        )

        encoded_video_torch = (encoded_video_torch - latents_mean) * latents_std

        # Finalize mask setup into the latent space
        msk = torch.concat([torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]], dim=1)
        msk = msk.view(1, msk.shape[1] // 4, 4, latent_shape[-2], latent_shape[-1])
        msk = msk.transpose(1, 2)

        tt_y = torch.cat([msk, encoded_video_torch], dim=1)

        return latents, tt_y
