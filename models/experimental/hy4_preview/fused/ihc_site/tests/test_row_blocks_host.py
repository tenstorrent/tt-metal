# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Row blocks of the iHC site's hand-over (kernels/ihc_site.cpp HY4_IHC_ROW_BLOCK, op.py NormOut.block), on the
host: the writes the kernel issues, replayed on bytes. Two sites of K cores each hold the 8 token rows of their block
as rows pages ([8, 32] tiles, two [8, 16] halves each); every core writes each half of each of its tiles to
``staging + k * 2 * shard + block * 256 + (2 t + h) * 512``. The staging must then be the bytes of x [16, C] as
[16, 32] row tiles (fused/projections layout.to_row_tiles): what the broadcast sends and a 16-row unit reads.
No device."""

import numpy as np
import torch

from models.experimental.hy4_preview.fused.ihc_site import layout

K, C = 16, 6144
HALF = 256


def tile_bytes(m):
    """A [32, w] BF16 page matrix -> the bytes of its [32, 32] tiles (faces: row half x column half, row major)."""
    v = m.to(torch.bfloat16).reshape(2, 16, m.shape[1] // 32, 2, 16).permute(2, 0, 3, 1, 4).contiguous()
    return v.view(torch.int16).numpy().tobytes()


def row_tiles16(x):
    """x [16, n] -> the bytes of its [16, 32] BF16 tiles (two faces of [16, 16])."""
    v = x.to(torch.bfloat16).reshape(16, x.shape[1] // 32, 2, 16).permute(1, 2, 0, 3).contiguous()
    return v.view(torch.int16).numpy().tobytes()


def test_block_pieces():
    g = torch.Generator().manual_seed(5)
    x = torch.randn(16, C, generator=g).to(torch.bfloat16).float()
    py = C // 128 // K  # rows pages of a core
    shard = py * 2048  # a core's BF16 rows of one block
    staging = np.zeros(K * 2 * shard, dtype=np.uint8)
    written = np.zeros(K * 2 * shard, dtype=np.int32)
    for block in range(2):
        pages = layout.rows_to_pages(x[8 * block : 8 * block + 8])  # [32, C / 4]
        per = pages.shape[1] // K
        for k in range(K):
            src = np.frombuffer(tile_bytes(pages[:, k * per : (k + 1) * per]), dtype=np.uint8)
            assert src.size == shard
            dst = k * 2 * shard + block * HALF
            for t in range(4 * py):
                for h in range(2):
                    a, b = (2 * t + h) * HALF, dst + (2 * t + h) * 2 * HALF
                    assert a % 16 == b % 16  # (a NoC write keeps the offset within the 16-byte word)
                    staging[b : b + HALF] = src[a : a + HALF]
                    written[b : b + HALF] += 1
    assert (written == 1).all(), "every byte of the staging written exactly once"
    assert staging.tobytes() == row_tiles16(x), "the staging is x as [16, 32] row tiles"
