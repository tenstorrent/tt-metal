# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from diffusers.models.autoencoders.autoencoder_kl_qwenimage import AutoencoderKLQwenImage

import ttnn

from ...models.vae.vae_wan2_1 import WanDecoder
from ...parallel.config import VaeHWParallelConfig, VAEParallelConfig
from ...parallel.manager import CCLManager
from ...utils.conv3d import conv_pad_height, conv_pad_in_channels
from ...utils.tensor import bf16_tensor_2dshard
from ...utils.tracing import Tracer

if TYPE_CHECKING:
    from collections.abc import Sequence


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

    def forward(self, x: ttnn.Tensor, logical_h: int) -> tuple[ttnn.Tensor, int]:
        output, new_logical_h, _new_logical_w = self.wan_decoder(x, logical_h=logical_h)
        return output, new_logical_h

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
        )  # remove the temporal dimension — output is (B, C, H, W)
        decoded_output = decoded_output[:, :, :logical_h, :]
        return decoded_output


class QwenImageVAEDecoderAdapter:
    """Torch-in (NHWC), torch-out (BCHW) VAE decoder for the QwenImage VAE.

    Applies per-channel scaling/shift inversion before decoding. Supports both the PyTorch and
    TT-NN implementations; the TT-NN backend supports tracing and dynamic load / unload of weights
    via ``deallocate_weights``/``reload_weights``.
    """

    def __init__(
        self,
        *,
        checkpoint_name: str,
        parallel_config: VaeHWParallelConfig,
        ccl_manager: CCLManager,
        use_torch: bool,
    ) -> None:
        torch_vae = AutoencoderKLQwenImage.from_pretrained(checkpoint_name, subfolder="vae")
        assert isinstance(torch_vae, AutoencoderKLQwenImage)

        self.device = ccl_manager.mesh_device
        # Per-channel scaling (z_dim,)
        self._latents_scaling = 1.0 / torch.tensor(torch_vae.config.latents_std)
        self._latents_shift = torch.tensor(torch_vae.config.latents_mean)

        if use_torch:
            self._torch_vae = torch_vae
            self._decoder = None
            self._tracer = None
            self._decoder_state_dict = None
        else:
            self._torch_vae = None
            self._decoder = QwenImageVaeDecoder(
                base_dim=torch_vae.config.base_dim,
                z_dim=torch_vae.config.z_dim,
                dim_mult=torch_vae.config.dim_mult,
                num_res_blocks=torch_vae.config.num_res_blocks,
                temperal_downsample=torch_vae.config.temperal_downsample,
                device=self.device,
                parallel_config=parallel_config,
                ccl_manager=ccl_manager,
            )
            self._tracer = Tracer(self._decoder.forward, device=self.device, prep_run=False)
            self._decoder_state_dict = torch_vae.state_dict()

    def is_loaded(self) -> bool:
        return self._torch_vae is not None or self._decoder.is_loaded()

    def deallocate_weights(self) -> None:
        if self._decoder is not None:
            self._decoder.deallocate_weights()

    def reload_weights(self) -> None:
        if self._decoder is None or self._decoder.is_loaded():
            return
        self._decoder.load_torch_state_dict(self._decoder_state_dict)

    @torch.no_grad()
    def decode(self, latents: torch.Tensor, *, traced: bool) -> torch.Tensor:
        latents = latents / self._latents_scaling + self._latents_shift

        if self._torch_vae is not None:
            return self._torch_vae.decode(latents.permute(0, 3, 1, 2).unsqueeze(2)).sample[:, :, 0]

        tt_latents, logical_h = self._decoder.prepare_input(latents)
        forward = self._tracer if traced else self._decoder.forward
        tt_out, logical_h = forward(tt_latents, logical_h)
        return self._decoder.postprocess_output(tt_out, logical_h)
