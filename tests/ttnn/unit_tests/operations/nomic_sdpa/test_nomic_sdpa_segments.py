# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""nomic_sdpa segmented mode: packed rows of several texts, each attending only its own keys.

q, k, v are (R, 12, W, 64): every row holds texts side by side, each starting on a tile, and runs of tiles no text
fills. Each text's output must match a float64 softmax over that text's keys only; the tile padding past a text's
real keys holds NaN, which must take no part. Slots no text fills must come out finite.
"""

from __future__ import annotations

import math
import random

import pytest
import torch
import ttnn

from ttnn.operations.nomic_sdpa import nomic_sdpa

HEADS, HEAD_DIM = 12, 64
TILE = 32


def _layout(rows, width, seed):
    """One list a row of (start, length): texts of real-looking lengths placed on tiles, edge lengths included, and
    leftover tiles at row ends."""
    rng = random.Random(seed)
    lengths = [1, 31, 32, 33, 512, 287, 260, 419, 112, 64, 200, 333, 450, 96, 5, 160]
    out = []
    for _ in range(rows):
        texts, at = [], 0
        while True:
            length = min(rng.choice(lengths), width)
            span = math.ceil(length / TILE) * TILE
            if at + span > width or (texts and rng.random() < 0.15):
                break
            texts.append((at, length))
            at += span
        out.append(texts)
    return out


def _inputs(rows, width, segments, seed=0):
    g = torch.Generator().manual_seed(seed)
    shape = (rows, HEADS, width, HEAD_DIM)
    q = torch.randn(shape, generator=g) * (2.0 / 8.0)  # the model's score spread at scale 1 (q pre-scaled)
    k = torch.randn(shape, generator=g)
    v = torch.randn(shape, generator=g)
    for r, texts in enumerate(segments):  # NaN in each text's tile padding: keys no query may read
        for start, length in texts:
            end = start + math.ceil(length / TILE) * TILE
            k[r, :, start + length : end] = float("nan")
            v[r, :, start + length : end] = float("nan")
    return q.bfloat16(), k.bfloat16(), v.bfloat16()


def _reference(q, k, v, segments):
    """(R, W, 768) float64 for the texts' slots (NaN elsewhere): softmax over each text's own keys."""
    rows, heads, width, dim = q.shape
    out = torch.full((rows, width, heads * dim), float("nan"), dtype=torch.float64)
    for r, texts in enumerate(segments):
        for start, length in texts:
            sl = slice(start, start + length)
            qq, kk, vv = q[r, :, sl].double(), k[r, :, sl].double(), v[r, :, sl].double()
            o = torch.softmax(qq @ kk.transpose(-1, -2), dim=-1) @ vv  # (heads, length, dim)
            out[r, sl] = o.permute(1, 0, 2).reshape(length, heads * dim)
    return out


@pytest.mark.parametrize(
    "rows, width, buffer",
    [(8, 512, "l1"), (4, 1024, "l1"), (4, 1024, "dram"), (16, 256, "dram"), (2, 1024, "l1")],
    ids=["8x512-l1", "4x1024-l1", "4x1024-dram", "16x256-dram", "2x1024-l1"],
)
def test_segments_attend_only_their_own_keys(device, rows, width, buffer):
    config = ttnn.L1_MEMORY_CONFIG if buffer == "l1" else ttnn.DRAM_MEMORY_CONFIG
    segments = _layout(rows, width, seed=rows * width)
    q, k, v = _inputs(rows, width, segments)
    tq, tk, tv = (
        ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=config)
        for t in (q, k, v)
    )

    out = nomic_sdpa(tq, tk, tv, scale=1.0, segments=segments, memory_config=config)

    got = ttnn.to_torch(out).double().reshape(rows, width, HEADS * HEAD_DIM)
    assert torch.isfinite(got).all(), "non-finite output (a text read NaN keys, or a filler slot overflowed)"
    ref = _reference(q, k, v, segments)
    keep = torch.isfinite(ref)
    a, b = got[keep], ref[keep]
    rel_rms = float((a - b).pow(2).mean().sqrt() / b.pow(2).mean().sqrt())
    pcc = float(torch.corrcoef(torch.stack([a, b]))[0, 1])
    assert pcc > 0.9995 and rel_rms < 0.03, f"PCC {pcc:.6f}, relative RMS {rel_rms:.4f}"
    for tensor in (tq, tk, tv, out):
        ttnn.deallocate(tensor)


def test_segments_are_read_per_call(device):
    """One shape, two layouts: the unit table is per call, never compiled into the cached program."""
    rows, width = 4, 1024
    config = ttnn.L1_MEMORY_CONFIG
    for seed in (1, 2):
        segments = _layout(rows, width, seed=seed)
        q, k, v = _inputs(rows, width, segments, seed=seed)
        tq, tk, tv = (
            ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=config)
            for t in (q, k, v)
        )
        got = ttnn.to_torch(nomic_sdpa(tq, tk, tv, segments=segments, memory_config=config)).double()
        got = got.reshape(rows, width, HEADS * HEAD_DIM)
        ref = _reference(q, k, v, segments)
        keep = torch.isfinite(ref)
        rel_rms = float((got[keep] - ref[keep]).pow(2).mean().sqrt() / ref[keep].pow(2).mean().sqrt())
        assert rel_rms < 0.03, f"layout {seed}: relative RMS {rel_rms:.4f}"


def test_malformed_segments_are_rejected(device, expect_error):
    q = ttnn.from_torch(torch.zeros(1, HEADS, 512, HEAD_DIM).bfloat16(), layout=ttnn.TILE_LAYOUT, device=device)
    with expect_error(ValueError, "start on a tile"):
        nomic_sdpa(q, q, q, segments=[[(0, 40), (40, 10)]])
    with expect_error(ValueError, "outside"):
        nomic_sdpa(q, q, q, segments=[[(0, 600)]])
    with expect_error(ValueError, "exclusive"):
        nomic_sdpa(q, q, q, key_lengths=[100], segments=[[(0, 100)]])
