# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The 4 residual streams across a pipeline boundary: the engine's D2D socket ships ONE ``[1, 1, S_l, W]`` tensor
per chunk (``prefill_runner.py``, width = hidden * multiplier), so the streams are packed stream-major on the last
dim -- ``W = 4 * D_l`` on each chip -- and unpacked on the receiver. No reshuffle across chips: each chip's four
``D_l`` slices stay its own."""

from __future__ import annotations

import torch

HC = 4


def pack_streams_torch(streams: list[torch.Tensor]) -> torch.Tensor:
    assert len(streams) == HC
    return torch.cat(list(streams), dim=-1)


def unpack_streams_torch(packed: torch.Tensor) -> list[torch.Tensor]:
    d = packed.shape[-1]
    assert d % HC == 0
    return list(packed.split(d // HC, dim=-1))


def pack_streams(streams: list):
    import ttnn

    assert len(streams) == HC
    return ttnn.concat(list(streams), dim=3)


def unpack_streams(packed) -> list:
    import ttnn

    d = packed.shape[-1]
    assert d % HC == 0
    dl = d // HC
    return [ttnn.slice(packed, [0, 0, 0, h * dl], [1, 1, packed.shape[2], (h + 1) * dl]) for h in range(HC)]
