# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING

import ttnn

from ...models.vae.vae_wan2_1 import WanDecoder, WanEncoder
from ...parallel.config import VaeHWParallelConfig, VAEParallelConfig
from ...parallel.manager import CCLManager
from ...utils.conv3d import conv_pad_height, conv_pad_in_channels
from ...utils.tensor import bf16_tensor_2dshard

if TYPE_CHECKING:
    from collections.abc import Sequence

    import torch


class QwenImageVaeDecoder:
    def __init__(
        self,
        *,
        base_dim: int = 96,
        z_dim: int = 16,
        dim_mult: Sequence[int] = (1, 2, 4, 4),
        num_res_blocks: int = 2,
        temperal_downsample: Sequence[bool] = (False, True, True),
        non_linearity: str = "silu",
        parallel_config: VAEParallelConfig | None,
        device: ttnn.MeshDevice,
        ccl_manager: CCLManager | None,
    ) -> None:
        self.params = {
            "base_dim": base_dim,
            "z_dim": z_dim,
            "dim_mult": dim_mult,
            "num_res_blocks": num_res_blocks,
            "temperal_downsample": temperal_downsample,
            "mesh_device": device,
            "ccl_manager": ccl_manager,
            "parallel_config": parallel_config,
        }

        self.wan_decoder = WanDecoder(**self.params)

        # TODO: Remove when WanDecoder is migrated to tt-dit Modules framework.
        self._is_loaded = False

    def forward(self, x: ttnn.Tensor, logical_h: int) -> ttnn.Tensor:
        return self.wan_decoder(x, logical_h=logical_h)

    def load_torch_state_dict(self, state_dict: dict[str, torch.Tensor]) -> None:
        if self.wan_decoder is None:
            self.wan_decoder = WanDecoder(**self.params)
        self.wan_decoder.load_state_dict(state_dict)
        self._is_loaded = True

    def deallocate_weights(self) -> None:
        # Hack. Move Wan to Module framework and implement deallocate_weights.
        del self.wan_decoder
        self.wan_decoder = None
        self._is_loaded = False

    def is_loaded(self) -> bool:
        return self._is_loaded

    def is_wan_based(self) -> bool:
        return True

    def prepare_input(self, torch_latents: torch.Tensor) -> tuple[ttnn.Tensor, int]:
        torch_latents = torch_latents.unsqueeze(1)
        torch_latents = conv_pad_in_channels(torch_latents)
        torch_latents, logical_h = conv_pad_height(
            torch_latents, self.wan_decoder.parallel_config.height_parallel.factor
        )
        tt_latents = bf16_tensor_2dshard(
            torch_latents,
            self.wan_decoder.mesh_device,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            shard_mapping={
                self.wan_decoder.parallel_config.height_parallel.mesh_axis: 2,
                self.wan_decoder.parallel_config.width_parallel.mesh_axis: 3,
            },
        )
        return tt_latents, logical_h

    def postprocess_output(self, tt_latents: ttnn.Tensor, logical_h: int) -> torch.Tensor:
        concat_dims = [None, None]
        concat_dims[self.wan_decoder.parallel_config.height_parallel.mesh_axis] = 3
        concat_dims[self.wan_decoder.parallel_config.width_parallel.mesh_axis] = 4
        decoded_output = ttnn.to_torch(
            tt_latents,
            mesh_composer=ttnn.ConcatMesh2dToTensor(
                self.wan_decoder.mesh_device, mesh_shape=tuple(self.wan_decoder.mesh_device.shape), dims=concat_dims
            ),
        ).squeeze(
            2
        )  # remove the temporal dimension
        return decoded_output


class QwenImageVaeEncoder:
    """On-device VAE encoder for QwenImage, wrapping WanEncoder."""

    def __init__(
        self,
        *,
        base_dim: int = 96,
        z_dim: int = 16,
        dim_mult: Sequence[int] = (1, 2, 4, 4),
        num_res_blocks: int = 2,
        attn_scales: Sequence[float] = (),
        temperal_downsample: Sequence[bool] = (False, True, True),
        is_residual: bool = False,
        parallel_config: VaeHWParallelConfig,
        device: ttnn.MeshDevice,
        ccl_manager: CCLManager | None,
    ) -> None:
        self.wan_encoder = WanEncoder(
            base_dim=base_dim,
            in_channels=3,
            z_dim=z_dim,
            dim_mult=dim_mult,
            num_res_blocks=num_res_blocks,
            attn_scales=list(attn_scales),
            temperal_downsample=list(temperal_downsample),
            is_residual=is_residual,
            mesh_device=device,
            ccl_manager=ccl_manager,
            parallel_config=parallel_config,
        )
        self._parallel_config = parallel_config
        self._device = device
        self._is_loaded = False

    def load_torch_state_dict(self, state_dict: dict[str, torch.Tensor]) -> None:
        self.wan_encoder.load_state_dict(state_dict)
        self._is_loaded = True

    def deallocate_weights(self) -> None:
        del self.wan_encoder
        self.wan_encoder = None
        self._is_loaded = False

    def is_loaded(self) -> bool:
        return self._is_loaded

    def prepare_input(self, image_BCHW: torch.Tensor, height_parallel_factor: int) -> tuple[ttnn.Tensor, int]:
        """Convert image tensor (B,C,H,W) to device-ready (B,T,H,W,C) format."""
        image_BTHWC = image_BCHW.unsqueeze(2).permute(0, 2, 3, 4, 1).to(torch.float32)
        image_BTHWC = conv_pad_in_channels(image_BTHWC)
        image_BTHWC, logical_h = conv_pad_height(image_BTHWC, height_parallel_factor)
        tt_image = bf16_tensor_2dshard(
            image_BTHWC,
            self._device,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            shard_mapping={
                self._parallel_config.height_parallel.mesh_axis: 2,
                self._parallel_config.width_parallel.mesh_axis: 3,
            },
        )
        return tt_image, logical_h

    def forward(self, tt_image: ttnn.Tensor, logical_h: int) -> tuple[ttnn.Tensor, int]:
        return self.wan_encoder(tt_image, logical_h)

    def postprocess_output(self, tt_latents: ttnn.Tensor, logical_h: int) -> torch.Tensor:
        """Convert device output (B,C,T,H,W) back to torch, trimming padding."""
        concat_dims = [None, None]
        concat_dims[self._parallel_config.height_parallel.mesh_axis] = 3
        concat_dims[self._parallel_config.width_parallel.mesh_axis] = 4
        latents = ttnn.to_torch(
            tt_latents,
            mesh_composer=ttnn.ConcatMesh2dToTensor(
                self._device, mesh_shape=tuple(self._device.shape), dims=concat_dims
            ),
        )
        return latents[:, :, :, :logical_h, :]
