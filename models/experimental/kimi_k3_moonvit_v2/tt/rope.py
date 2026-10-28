# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""2D RoPE tables on device: `Rope2DPosEmbRepeated` + `apply_rope` in the reference.

The reference rotates adjacent (x0, x1) pairs by a complex multiply in fp32. That equals
`x * cos + (x @ trans_mat) * sin` with cos/sin repeated per pair and `trans_mat` mapping
(x0, x1) -> (-x1, x0), which is what `ttnn.experimental.rotary_embedding_llama` computes
(`reference.functional.rope_cos_sin` / `rotate_pairs`). The tables are shared by every layer
and head, so they are built once per image. Their rows are in window order, like the activations
(`reference.functional.to_window_order`).
"""

from __future__ import annotations

from typing import NamedTuple

import ttnn

from .model_config import MoonViTConfig, VisionParallel


class RopeTensors(NamedTuple):
    cos: ttnn.Tensor  # [1, 1, S/sp, head_dim], split along the sequence like the activations
    sin: ttnn.Tensor
    trans_mat: ttnn.Tensor  # [1, 1, 32, 32], replicated


class TT_Rope:
    def __init__(self, config: MoonViTConfig, *, mesh_device: ttnn.MeshDevice, parallel: VisionParallel) -> None:
        self.config = config
        self.mesh_device = mesh_device
        self.parallel = parallel

    def get_rot_mats(self, h: int, w: int) -> RopeTensors:
        """`rope_cos_sin(config, h, w)` in window order, padded to `max_patches` rows and uploaded,
        plus the `rot_transformation_mat` tile, all from `reference.functional`."""
        raise NotImplementedError("TT_Rope.get_rot_mats")
