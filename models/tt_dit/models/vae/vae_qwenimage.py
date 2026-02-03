# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING

import ttnn

from ...parallel.config import VAEParallelConfig
from ...parallel.manager import CCLManager
from ...models.vae.vae_wan2_1 import WanDecoder
from ...utils.conv3d import conv_pad_in_channels, conv_pad_height
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
