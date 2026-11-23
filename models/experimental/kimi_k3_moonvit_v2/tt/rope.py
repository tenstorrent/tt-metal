# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""2D RoPE tables on device: `Rope2DPosEmbRepeated` + `apply_rope` in the reference.

The reference rotates adjacent (x0, x1) pairs by a complex multiply in fp32. That equals
`x * cos + (x @ trans_mat) * sin` with cos/sin repeated per pair and `trans_mat` mapping
(x0, x1) -> (-x1, x0), which is what `ttnn.experimental.rotary_embedding_llama` computes
(`host_prep.rope_cos_sin` / `rotate_pairs`). The tables are shared by every layer
and head, so they are built once per image. Their rows are in window order, like the activations
(`host_prep.to_window_order`).

The tables are `DeviceInput`s: written in place for each image, so a trace can replay them.
"""

from __future__ import annotations

from typing import NamedTuple

import torch

import ttnn

from . import host_prep
from .device_input import DeviceInput
from .model_config import MoonViTConfig, VisionParallel


class RopeTensors(NamedTuple):
    cos: ttnn.Tensor  # [1, 1, S/sp, head_dim], split along the sequence like the activations
    sin: ttnn.Tensor
    trans_mat: ttnn.Tensor  # [1, 1, 32, 32], replicated


class TT_Rope:
    def __init__(self, config: MoonViTConfig, *, mesh_device: ttnn.MeshDevice, parallel: VisionParallel) -> None:
        table_shape = (1, 1, config.max_patches, config.head_dim)
        self.cos = DeviceInput(table_shape, mesh_device=mesh_device, sp_axis=parallel.sp_axis)
        self.sin = DeviceInput(table_shape, mesh_device=mesh_device, sp_axis=parallel.sp_axis)
        self.trans_mat = DeviceInput((1, 1, 32, 32), mesh_device=mesh_device, sp_axis=None)

    def write(self, cos: torch.Tensor, sin: torch.Tensor) -> RopeTensors:
        """Fill the tables from `host_prep.rope_tables` (or `prepare_host_inputs`) cos/sin."""
        return self._tensors(self.cos.write(cos), self.sin.write(sin))

    def write_rows(self, cos: torch.Tensor, sin: torch.Tensor) -> RopeTensors:
        """Fill the first rows of the tables from cos/sin [n, head_dim] (`host_fill.rope_window_rows`); the rest are
        zero, as `host_prep.rope_tables` pads them."""
        return self._tensors(self.cos.write_rows(cos), self.sin.write_rows(sin))

    def tensors(self) -> RopeTensors:
        """The tables as the last write left them."""
        return RopeTensors(self.cos.tensor, self.sin.tensor, self.trans_mat.tensor)

    def _tensors(self, cos: ttnn.Tensor, sin: ttnn.Tensor) -> RopeTensors:
        if self.trans_mat.tensor is None:
            self.trans_mat.write(host_prep.rot_transformation_mat())
        return RopeTensors(cos, sin, self.trans_mat.tensor)
