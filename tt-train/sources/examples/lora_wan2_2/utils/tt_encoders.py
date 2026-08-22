# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Wan VAE encoder and UMT5 text encoder on the mesh, via tt_dit. See README.md."""

from __future__ import annotations

import torch
import ttnn
from diffusers.models import AutoencoderKLWan

from models.tt_dit.models.vae.vae_wan2_1 import WanEncoder
from models.tt_dit.parallel.config import (
    DiTParallelConfig,
    EncoderParallelConfig,
    ParallelFactor,
    VaeHWParallelConfig,
)
from models.tt_dit.parallel.manager import CCLManager
from models.tt_dit.pipelines.wan.text_encoder import TextEncoder
from models.tt_dit.utils.conv3d import conv_pad_height, conv_pad_in_channels, conv_pad_width
from models.tt_dit.utils.tensor import fast_device_to_host, typed_tensor_2dshard

# Encoder downsamples 8x spatially, so input padding is a multiple of factor * 8.
_ENCODER_SPATIAL_MULTIPLIER = 8

_H_AXIS, _W_AXIS = 0, 1


def open_mesh(mesh_shape) -> ttnn.MeshDevice:
    shape = tuple(mesh_shape)
    if shape[0] * shape[1] > 1:
        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D, ttnn.FabricReliabilityMode.STRICT_INIT)
    return ttnn.open_mesh_device(ttnn.MeshShape(*shape))


def close_mesh(mesh_device: ttnn.MeshDevice) -> None:
    shape = tuple(mesh_device.shape)
    ttnn.close_mesh_device(mesh_device)
    if shape[0] * shape[1] > 1:
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


def make_ccl_manager(mesh_device: ttnn.MeshDevice, num_links: int = 2) -> CCLManager:
    return CCLManager(mesh_device, topology=ttnn.Topology.Linear, num_links=num_links)


def _vae_parallel_config(mesh_device: ttnn.MeshDevice) -> VaeHWParallelConfig:
    shape = tuple(mesh_device.shape)
    return VaeHWParallelConfig(
        height_parallel=ParallelFactor(factor=shape[_H_AXIS], mesh_axis=_H_AXIS),
        width_parallel=ParallelFactor(factor=shape[_W_AXIS], mesh_axis=_W_AXIS),
    )


class WanVAEEncoderTT:
    def __init__(
        self,
        *,
        checkpoint_name: str,
        mesh_device: ttnn.MeshDevice,
        ccl_manager: CCLManager,
        height: int,
        width: int,
        num_frames: int,
        dtype: ttnn.DataType = ttnn.bfloat16,
    ) -> None:
        self.device = mesh_device
        self._ccl_manager = ccl_manager
        self._parallel_config = _vae_parallel_config(mesh_device)

        self._torch_vae = AutoencoderKLWan.from_pretrained(checkpoint_name, subfolder="vae", trust_remote_code=True)
        cfg = self._torch_vae.config
        self.z_dim = cfg.z_dim
        self._dtype = dtype

        self._ctor_t_chunk = num_frames if num_frames < 4 else 4
        self._fwd_t_chunk = None if num_frames < 4 else 4

        self._encoder = WanEncoder(
            base_dim=cfg.base_dim,
            in_channels=cfg.in_channels,
            z_dim=cfg.z_dim,
            dim_mult=cfg.dim_mult,
            num_res_blocks=cfg.num_res_blocks,
            attn_scales=cfg.attn_scales,
            temperal_downsample=cfg.temperal_downsample,
            is_residual=cfg.is_residual,
            mesh_device=mesh_device,
            ccl_manager=ccl_manager,
            parallel_config=self._parallel_config,
            dtype=dtype,
            height=height,
            width=width,
            encoder_t_chunk_size=self._ctor_t_chunk,
        )
        self._encoder.load_torch_state_dict(self._torch_vae.state_dict())

        view = (1, cfg.z_dim, 1, 1, 1)
        self._latents_mean = torch.tensor(cfg.latents_mean, dtype=torch.float32).view(view)
        self._latents_std = torch.tensor(cfg.latents_std, dtype=torch.float32).view(view)

    @property
    def config(self):
        return self._torch_vae.config

    def _prepare_input(self, video_BCTHW: torch.Tensor):
        x = video_BCTHW.permute(0, 2, 3, 4, 1)
        x = conv_pad_in_channels(x)
        h_factor = self._parallel_config.height_parallel.factor * _ENCODER_SPATIAL_MULTIPLIER
        w_factor = self._parallel_config.width_parallel.factor * _ENCODER_SPATIAL_MULTIPLIER
        x, logical_h = conv_pad_height(x, h_factor)
        x, logical_w = conv_pad_width(x, w_factor)

        tt_x = typed_tensor_2dshard(
            x,
            self.device,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            shard_mapping={
                self._parallel_config.height_parallel.mesh_axis: 2,
                self._parallel_config.width_parallel.mesh_axis: 3,
            },
            dtype=self._dtype,
        )
        return tt_x, logical_h, logical_w

    @torch.no_grad()
    def encode(self, video_BCTHW: torch.Tensor) -> torch.Tensor:
        tt_x, logical_h, logical_w = self._prepare_input(video_BCTHW.float())

        tt_latent_BCTHW, new_logical_h, new_logical_w = self._encoder(
            tt_x, logical_h, encoder_t_chunk_size=self._fwd_t_chunk, logical_w=logical_w
        )

        # BCTHW: height on dim 3, width on dim 4.
        concat_dims = [None, None]
        concat_dims[self._parallel_config.height_parallel.mesh_axis] = 3
        concat_dims[self._parallel_config.width_parallel.mesh_axis] = 4
        latent = fast_device_to_host(
            tt_latent_BCTHW,
            self.device,
            concat_dims,
            ccl_manager=self._ccl_manager,
            dtype=torch.float32,
        )

        latent = latent[:, : self.z_dim, :, :new_logical_h, :new_logical_w]
        latent = (latent - self._latents_mean) * (1.0 / self._latents_std)
        return latent.squeeze(0).contiguous()


class WanTextEncoderTT:
    def __init__(
        self,
        *,
        checkpoint_name: str,
        mesh_device: ttnn.MeshDevice,
        ccl_manager: CCLManager,
        max_sequence_length: int = 512,
        tp_axis: int = 1,
    ) -> None:
        self.device = mesh_device
        shape = tuple(mesh_device.shape)
        self._tp_axis = tp_axis
        self._dp_axis = 1 - tp_axis
        self._batch_group = shape[self._dp_axis]

        encoder_parallel_config = EncoderParallelConfig(
            tensor_parallel=ParallelFactor(factor=shape[tp_axis], mesh_axis=tp_axis)
        )
        dit_parallel_config = DiTParallelConfig(
            cfg_parallel=ParallelFactor(factor=1, mesh_axis=0),
            tensor_parallel=ParallelFactor(factor=shape[tp_axis], mesh_axis=tp_axis),
            sequence_parallel=ParallelFactor(factor=shape[self._dp_axis], mesh_axis=self._dp_axis),
        )

        self._encoder = TextEncoder(
            checkpoint_name=checkpoint_name,
            device=mesh_device,
            ccl_manager=ccl_manager,
            encoder_parallel_config=encoder_parallel_config,
            dit_parallel_config=dit_parallel_config,
            max_sequence_length=max_sequence_length,
        )
        self._encoder.prepare()

    @torch.no_grad()
    def encode(self, captions: list[str]) -> dict[str, torch.Tensor]:
        out: dict[str, torch.Tensor] = {}
        group = max(1, self._batch_group)
        for start in range(0, len(captions), group):
            chunk = captions[start : start + group]
            # TextEncoder shards the prompt batch over the non-TP axis; pad to fill it.
            padded = chunk + [" "] * (group - len(chunk))
            tt_embeds = self._encoder._encode(padded, num_videos_per_prompt=1)
            embeds = fast_device_to_host(tt_embeds, self.device, [None, None], dtype=torch.float32)
            embeds = embeds.reshape(-1, embeds.shape[-2], embeds.shape[-1])
            for i, cap in enumerate(chunk):
                out[cap] = embeds[i].contiguous()
        return out
