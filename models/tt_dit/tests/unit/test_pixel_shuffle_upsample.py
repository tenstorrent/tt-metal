# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""The T-slabbed pixel-shuffle upsample must equal the whole-volume upsample.

At 6s the last deterministic upsample's projection is a single ~10 GiB buffer that OOMs, so the
upsample slabs over source frames when the projection would exceed CHUNK_BYTES. Slabbing is only
memory bookkeeping -- each source frame maps to its own output frames -- so the result must be
identical to running the volume whole. Forced here by shrinking CHUNK_BYTES so a small volume slabs.
"""

from __future__ import annotations

import pytest
import torch

import ttnn

from ...models.vae import diffvae_ltx
from ...models.vae.diffvae_ltx import LinearPixelShuffleUpsample
from ...utils.check import assert_quality


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize(
    "stride, reduction",
    [((2, 2, 2), 2), ((1, 2, 2), 2), ((2, 1, 1), 2)],
    ids=["s222", "s122", "s211"],
)
def test_upsample_chunked_matches_whole(*, mesh_device, stride, reduction, monkeypatch):
    in_channels = 64
    t, h, w = 8, 8, 8  # h*w = 64, a multiple of TILE, so a frame boundary is sliceable
    up = LinearPixelShuffleUpsample(in_channels, stride, reduction, mesh_device=mesh_device)

    torch.manual_seed(0)
    up.load_state_dict(
        {
            "proj.weight": torch.randn(up.proj_out_channels, in_channels) * 0.1,
            "proj.bias": torch.randn(up.proj_out_channels) * 0.1,
        }
    )
    x = torch.randn(t * h * w, in_channels)

    def run() -> tuple[torch.Tensor, tuple[int, int, int]]:
        x_tt = ttnn.from_torch(x, device=mesh_device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
        out, dims = up(x_tt, dims=(t, h, w))
        return ttnn.to_torch(out).float(), dims

    # Whole: a huge budget keeps it single-shot.
    monkeypatch.setattr(diffvae_ltx, "CHUNK_BYTES", 1 << 40)
    whole, dims_whole = run()

    # Chunked: a budget of ~3 frames' projection forces multi-slab (slabs of 3, 3, 2 over t=8).
    monkeypatch.setattr(diffvae_ltx, "CHUNK_BYTES", 3 * h * w * up.proj_out_channels * 2)
    chunked, dims_chunked = run()

    assert dims_chunked == dims_whole, f"{dims_chunked} != {dims_whole}"
    assert tuple(chunked.shape) == tuple(whole.shape), f"{tuple(chunked.shape)} != {tuple(whole.shape)}"
    assert_quality(whole, chunked, pcc=0.9999)
