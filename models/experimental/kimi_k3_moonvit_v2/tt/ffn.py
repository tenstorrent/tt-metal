# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The MoonViT2 vision FFN on ttnn.

`TtVisionFfn` is not implemented yet; it takes the host `VisionFfnWeights` and converts them itself.
Its golden is `TorchVisionFfn` in `reference/ffn.py`, which tests build directly.
"""

from __future__ import annotations

from typing import Protocol

from models.experimental.kimi_k3_moonvit_v2.reference.weights import VisionFfnWeights
from models.experimental.kimi_k3_moonvit_v2.reference.config import KimiK3VisionConfig


class VisionFfn(Protocol):
    """What a vision block needs from its FFN, and nothing more."""

    def forward(self, x):
        """`[1, 1, S, VT_HIDDEN_SIZE]` in, the same shape out."""
        ...


# ttnn implementation of `VisionFfn`; not written yet.
TtVisionFfn = None


def build_vision_ffn(mesh_device, weights: VisionFfnWeights, cfg: type = KimiK3VisionConfig) -> VisionFfn:
    if TtVisionFfn is None:
        raise NotImplementedError("TtVisionFfn is not implemented yet")
    return TtVisionFfn(mesh_device, weights, cfg)
