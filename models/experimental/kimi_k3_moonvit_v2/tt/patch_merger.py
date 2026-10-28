# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Patch merger: `tpool_patch_merger` in the reference. No weights.

Each 2x2 window of patches becomes one token of the window's four patches concatenated, in
(row, column) order. With one still image the temporal mean is a no-op. The encoder output is split
along `sp_axis` in raster order, so the two image rows of a window can sit on different devices.
"""

from __future__ import annotations

import ttnn

from .model_config import MoonViTConfig, VisionParallel
from .module import Module


class TT_Patch_Merger(Module):
    def __init__(self, config: MoonViTConfig, *, mesh_device: ttnn.MeshDevice, parallel: VisionParallel) -> None:
        super().__init__()
        self.config = config
        self.mesh_device = mesh_device
        self.parallel = parallel

    def forward(self, x: ttnn.Tensor, h: int, w: int) -> ttnn.Tensor:
        """[1, 1, S/sp, hidden] -> [1, 1, S/(4*sp), 4*hidden]."""
        raise NotImplementedError("TT_Patch_Merger.forward")
