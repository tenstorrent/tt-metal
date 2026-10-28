# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for ``ttnn.experimental.fused_partial_rope`` with ROW_MAJOR inputs and per-row cos/sin.

Each ``head_dim`` block of a row keeps its leading ``head_dim - rope_dim`` channels and rotates the
trailing ``rope_dim``::

    out[r, b, Hd-Rd:] = x_rope * cos[r] + (x_rope @ trans_mat) * sin[r]

where ``trans_mat`` is a 32x32 matrix applied to every 32-channel chunk of ``x_rope``. These cases
cover verifying several draft tokens in one step (DSpark): ``T = 5`` rows at different positions,
so cos/sin carry one row per input row and every input is ROW_MAJOR.

Run::

    pytest tests/ttnn/unit_tests/operations/experimental/transformer/test_fused_partial_rope.py
"""

from __future__ import annotations

import pytest
import torch

import ttnn
from models.common.utility_functions import comp_pcc

TILE = 32
ROWS = 5
HEAD_DIM = 512
ROPE_DIM = 64


def _reference(x, cos, sin, trans_mat, head_dim, rope_dim):
    rows, d = x.shape
    blocks = x.float().reshape(rows, d // head_dim, head_dim)
    rope = blocks[..., head_dim - rope_dim :]
    rotated = (rope.reshape(rows, -1, rope_dim // TILE, TILE) @ trans_mat.float()).reshape(rope.shape)
    out = blocks.clone()
    out[..., head_dim - rope_dim :] = rope * cos.float()[:, None, :] + rotated * sin.float()[:, None, :]
    return out.reshape(rows, d)


def _sharded_config(shard_shape, cores, strategy):
    grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(cores - 1, 0))})
    return ttnn.create_sharded_memory_config(
        shape=shard_shape,
        core_grid=grid,
        strategy=strategy,
        use_height_and_width_as_shard_shape=True,
    )


@pytest.mark.parametrize("trans_mat_layout", [ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT], ids=["tm_rm", "tm_tile"])
@pytest.mark.parametrize(
    "rows, d, shard_shape, cores, strategy, cos_rows",
    [
        # Verify query [T*H, Dh]: one token's H head rows per core, each head row has its token's position.
        (ROWS * 64, HEAD_DIM, (64, HEAD_DIM), ROWS, ttnn.ShardStrategy.HEIGHT, ROWS * 64),
        # Verify latent kv [T, Dh]: all T rows on one core.
        (ROWS, HEAD_DIM, (ROWS, HEAD_DIM), 1, ttnn.ShardStrategy.HEIGHT, ROWS),
        # Width-sharded: every core holds all T rows and a column slice; only the last core has rope.
        (ROWS, HEAD_DIM, (ROWS, HEAD_DIM // 4), 4, ttnn.ShardStrategy.WIDTH, ROWS),
        # Two head blocks per row, width-sharded so each core straddles nope and rope columns.
        (ROWS, 2 * HEAD_DIM, (ROWS, 2 * HEAD_DIM // 8), 8, ttnn.ShardStrategy.WIDTH, ROWS),
        # A single ROW_MAJOR cos/sin row broadcast across every input row.
        (ROWS, HEAD_DIM, (ROWS, HEAD_DIM), 1, ttnn.ShardStrategy.HEIGHT, 1),
        (ROWS, HEAD_DIM, (ROWS, HEAD_DIM // 4), 4, ttnn.ShardStrategy.WIDTH, 1),
    ],
    ids=["q_height", "kv_height", "kv_width", "two_heads_width", "bcast_height", "bcast_width"],
)
def test_fused_partial_rope_row_major(device, rows, d, shard_shape, cores, strategy, cos_rows, trans_mat_layout):
    torch.manual_seed(0)
    x = torch.randn(rows, d, dtype=torch.bfloat16)
    angles = torch.rand(cos_rows, ROPE_DIM) * 6.28
    cos = torch.cos(angles).to(torch.bfloat16)
    sin = torch.sin(angles).to(torch.bfloat16)
    trans_mat = torch.zeros(TILE, TILE, dtype=torch.bfloat16)
    trans_mat[torch.arange(1, TILE, 2), torch.arange(0, TILE, 2)] = -1
    trans_mat[torch.arange(0, TILE, 2), torch.arange(1, TILE, 2)] = 1

    tt_x = ttnn.from_torch(
        x.reshape(1, 1, rows, d),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=_sharded_config(shard_shape, cores, strategy),
    )
    tt_cos, tt_sin = (
        ttnn.from_torch(
            t.reshape(1, 1, cos_rows, ROPE_DIM),
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        for t in (cos, sin)
    )
    tt_trans_mat = ttnn.from_torch(
        trans_mat.reshape(1, 1, TILE, TILE),
        dtype=ttnn.bfloat16,
        layout=trans_mat_layout,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    out = ttnn.experimental.fused_partial_rope(tt_x, tt_cos, tt_sin, tt_trans_mat, ROPE_DIM, head_dim=HEAD_DIM)
    assert out.layout == ttnn.ROW_MAJOR_LAYOUT
    got = ttnn.to_torch(out).reshape(rows, d).float()

    expected = _reference(x, cos.expand(rows, -1), sin.expand(rows, -1), trans_mat, HEAD_DIM, ROPE_DIM)
    passed, pcc = comp_pcc(expected, got, 0.999)
    assert passed, f"PCC {pcc}"
    # The nope channels pass through bit-exact.
    nope = expected.reshape(rows, -1, HEAD_DIM)[..., : HEAD_DIM - ROPE_DIM]
    assert torch.equal(got.reshape(rows, -1, HEAD_DIM)[..., : HEAD_DIM - ROPE_DIM], nope)
