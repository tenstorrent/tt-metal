# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Patch merger: `tpool_patch_merger` in the reference. No weights.

Each 2x2 window of patches becomes one token of the window's four patches concatenated, in
(row, column) order. With one still image the temporal mean is a no-op. The rows arrive in window
order (`host_prep.to_window_order`), so a window is four consecutive rows and the merge
is a reshape on each device: no collective, and the output is in the reference's token order.
Each device's share of `max_patches` is a whole number of windows (`resolve_parallel` checks it).
"""

from __future__ import annotations

import ttnn

from .model_config import MoonViTConfig
from .module import Module


class TT_Patch_Merger(Module):
    def __init__(self, config: MoonViTConfig) -> None:
        super().__init__()
        self.config = config

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        """[1, 1, S/sp, hidden] in window order -> [1, 1, S/(4*sp), 4*hidden]."""
        raise NotImplementedError("TT_Patch_Merger.forward")
