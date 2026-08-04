# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Independence of ``ttnn.rand`` along the tile WIDTH axis.

The existing rand suite pins range, determinism, cache behaviour and mesh sharding, all of which
hold. None of it pins *independence between elements*, and on Blackhole the width axis is not
independent:

* ~~within every 32-wide tile, columns 24..31 are byte-identical to columns 0..7~~ -- FIXED: the
  kernel now consumes several PRNG values per element, so the sliding window advances past its own
  width and no two columns are byte-identical. That test is no longer xfail;
* the remaining correlation is NOT fixed. The PRNG is a sliding window over one stream -- element
  (read t, lane i) carries stream[t + i] -- so columns stay correlated in value and the argmax over
  a wide row still coincides far more often than IID allows (cross-position max |r| 0.618 against
  0.035 for a host control). Any combination of extra reads is still a function of t + i; removing
  it needs a counter-based RNG keyed on the element's own position.

Both are properties of the op (they are present in the raw ``ttnn.rand`` output, independent of
the other axis extent and of the seed), so they belong in the rand suite rather than in a
consumer's tests. Measured 2026-07-25 on P150x4, at width 256 and at width 262144 alike.

The per-core seed was ruled out as the cause: cores are seeded ``seed + i``, and replacing that
with an avalanche mix left both metrics unchanged (cross-core rows already measure clean --
max |r| 5.7 vs 6.5 sigma before/after, both against a 3.6 sigma host control). The defect is
inside a single core's tile, in the SFPU PRNG path used by ``compute_uniform.cpp``.

Marked xfail(strict) rather than deleted so the suite records the defect and flips to a failure
the moment it is fixed.
"""

import pytest
import torch

import ttnn

TILE = 32
SEED = 48291


def _rand_rows(device, height, width, seed=SEED):
    """Return ``ttnn.rand((height, width))`` transposed to ``[width, height]`` host rows."""
    tensor = ttnn.rand(
        (height, width),
        device=device,
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        low=0.0,
        high=1.0,
        seed=seed,
    )
    host = ttnn.to_torch(tensor).float()
    tensor.deallocate(True)
    return host.t().contiguous()


@pytest.mark.parametrize("width", [256, 2048])
def test_rand_columns_are_distinct(device, width):
    """No two columns of a single rand draw may be byte-identical."""
    columns = _rand_rows(device, height=2048, width=width)
    seen, duplicates = {}, []
    for index in range(columns.shape[0]):
        key = hash(columns[index].numpy().tobytes())
        if key in seen:
            duplicates.append((index, seen[key]))
        else:
            seen[key] = index
    assert not duplicates, (
        f"{len(duplicates)} of {columns.shape[0]} columns duplicate an earlier column; "
        f"first pairs {duplicates[:6]}, offsets {sorted({i - j for i, j in duplicates})}"
    )


@pytest.mark.xfail(
    strict=True,
    reason="the PRNG is a sliding window over one stream, so columns stay correlated in value even "
    "now that none are byte-identical: cross-position max |r| is 0.618 against 0.035 for host IID, "
    "and argmax coincides far more often than IID allows. Removing this needs a counter-based RNG "
    "keyed on each element's position; consuming more PRNG values only dilutes it",
)
def test_rand_columns_do_not_share_an_argmax(device):
    """Independent columns almost never pick the same row as their maximum.

    With ``height`` rows and ``n`` columns of IID noise, the expected number of colliding
    argmaxes is ``C(n, 2) / height`` -- under 0.5 here. Any sizeable tie group is a dependency
    between columns. This is the functional form of the defect: for a Gumbel-max sampler it
    turns independent rare flips into synchronized bursts of the same token.
    """
    height, width = 16384, 256
    columns = _rand_rows(device, height=height, width=width)
    counts = torch.bincount(columns.argmax(dim=-1))
    distinct = int((counts > 0).sum())
    expected_collisions = width * (width - 1) / 2 / height
    assert distinct >= width - 4, (
        f"only {distinct}/{width} columns have a distinct argmax (largest tie group "
        f"{int(counts.max())}); IID would collide about {expected_collisions:.2f} times"
    )


def test_rand_rows_across_tiles_are_distinct(device):
    """Control: the CROSS-TILE axis is fine, so the defect is not the per-core seeding.

    This is the arm that would fail if ``seed + i`` per core were the problem. It passes, which
    is what rules that hypothesis out and localises the defect inside a single tile.
    """
    height, width = 2048, 2048
    tensor = ttnn.rand(
        (height, width), device=device, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, low=0.0, high=1.0, seed=SEED
    )
    host = ttnn.to_torch(tensor).float()
    tensor.deallocate(True)
    rows = host[::TILE].contiguous()  # one row per tile-row -> different tiles, different cores
    counts = torch.bincount(rows.argmax(dim=-1))
    assert int(counts.max()) <= 2, f"cross-tile rows share an argmax (largest group {int(counts.max())})"
