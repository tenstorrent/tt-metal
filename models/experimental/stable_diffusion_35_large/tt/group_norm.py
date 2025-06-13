# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import ttnn
from loguru import logger

from .utils import from_torch_fast

if TYPE_CHECKING:
    import torch


@dataclass
class TtGroupNormParameters:
    weight: torch.Tensor
    bias: torch.Tensor
    device: ttnn.Device

    @classmethod
    def from_torch(
        cls,
        state: dict[str, torch.Tensor],
        *,
        device: ttnn.Device,
    ) -> TtGroupNormParameters:
        return cls(
            weight=state["weight"],
            bias=state["bias"],
            device=device,
        )


@dataclass
class TtGroupNormPreparation:
    batch_size: int
    input_width: int
    input_height: int
    num_channels: int
    weight: ttnn.Tensor
    bias: ttnn.Tensor
    mask: ttnn.Tensor
    memory_config: ttnn.MemoryConfig
    core_grid: ttnn.CoreGrid


class TtGroupNorm:
    def __init__(self, parameters: TtGroupNormParameters, *, eps: float, num_groups: int) -> None:
        self._eps = eps
        self._num_groups = num_groups
        self._parameters = parameters
        self._device = parameters.device
        self._preparation = None

    def _prepare(
        self,
        *,
        batch_size: int,
        input_width: int,
        input_height: int,
        num_channels: int,
    ) -> TtGroupNormPreparation:
        if self._preparation is not None:
            if (
                self._preparation.batch_size == batch_size
                and self._preparation.input_width == input_width
                and self._preparation.input_height == input_height
                and self._preparation.num_channels == num_channels
            ):
                return self._preparation
            logger.warning("shape of group norm input changed")

        num_groups = self._num_groups

        k_device = 256 * self._device.core_grid.x * self._device.core_grid.y
        self._inplace = input_width * input_height <= k_device  # a heuristic

        if self._inplace:
            (
                memory_config,
                core_grid,
            ) = ttnn.determine_expected_group_norm_sharded_config_and_grid_size(
                device=self._device,
                num_channels=num_channels,
                num_groups=num_groups,
                input_nhw=batch_size * input_height * input_width,
                is_height_sharded=False,
            )
            self._num_out_blocks = 1

            # if input_memory_config.memory_layout == ttnn.TensorMemoryLayout.BLOCK_SHARDED:
            #     grid_y = self.group_norm_core_grid.y
            # elif input_memory_config.memory_layout == ttnn.TensorMemoryLayout.HEIGHT_SHARDED:
            #     grid_y = 1
            # else:
            #     grid_y = int(self.group_norm_core_grid.x * self.group_norm_core_grid.y)
        else:
            # https://github.com/tenstorrent/tt-metal/issues/22149#issuecomment-2884093864
            h = num_channels // num_groups * num_groups
            assert h % 32 == 0
            grid_y = self._device.core_grid.y
            while h // grid_y % 32 != 0:
                grid_y -= 1

            core_grid = ttnn.CoreGrid(x=self._device.core_grid.x, y=grid_y)
            k_grid = 256 * core_grid.x * core_grid.y
            self._num_out_blocks = -(-input_width * input_height // k_grid)  # a heuristic
            memory_config = ttnn.DRAM_MEMORY_CONFIG

        torch_weight = ttnn.create_group_norm_weight_bias_rm(
            self._parameters.weight,
            num_channels,
            core_grid.y,
        )
        torch_bias = ttnn.create_group_norm_weight_bias_rm(
            self._parameters.bias,
            num_channels,
            core_grid.y,
        )
        torch_mask = ttnn.create_group_norm_input_mask(
            num_channels,
            num_groups,
            core_grid.y,
        )

        self._preparation = TtGroupNormPreparation(
            batch_size=batch_size,
            input_width=input_width,
            input_height=input_height,
            num_channels=num_channels,
            weight=from_torch_fast(
                torch_weight,
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=self._device,
            ),
            bias=from_torch_fast(
                torch_bias,
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=self._device,
            ),
            mask=from_torch_fast(
                torch_mask,
                dtype=ttnn.bfloat8_b,
                layout=ttnn.TILE_LAYOUT,
                device=self._device,
            ),
            memory_config=memory_config,
            core_grid=core_grid,
        )

        return self._preparation

    def __call__(self, x: ttnn.Tensor, *, memory_config: ttnn.MemoryConfig | None = None) -> ttnn.Tensor:
        [batch_size, height, width, channels] = list(x.shape)

        prep = self._prepare(
            batch_size=batch_size,
            input_width=width,
            input_height=height,
            num_channels=channels,
        )

        x = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
        x = x.reshape([batch_size, 1, width * height, channels])

        if self._inplace:
            x = ttnn.to_memory_config(x, prep.memory_config)
            x = ttnn.reallocate(x)
        else:
            x = ttnn.tilize_with_zero_padding(x, use_multicore=True)

        x = ttnn.group_norm(
            x,
            weight=prep.weight,
            bias=prep.bias,
            input_mask=prep.mask,
            num_groups=self._num_groups,
            epsilon=self._eps,
            core_grid=prep.core_grid,
            # memory_config=memory_config if memory_config is not None else prep.memory_config,
            inplace=self._inplace,
            num_out_blocks=self._num_out_blocks,
        )

        # to_layout does not work with block sharded tensors
        if memory_config is None:
            memory_config = ttnn.DRAM_MEMORY_CONFIG
        x = ttnn.to_memory_config(x, memory_config)
        x = ttnn.to_layout(x, ttnn.TILE_LAYOUT)

        return x.reshape([batch_size, height, width, channels])
