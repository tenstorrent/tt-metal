# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Independence of ``ttnn.rand`` along the tile WIDTH axis.

The existing rand suite pins range, determinism, cache behaviour and mesh sharding, all of which
hold. None of it pins *independence between elements*, and on Blackhole the width axis is not
independent:

* ~~within every 32-wide tile, columns 24..31 are byte-identical to columns 0..7~~ -- FIXED: the
  kernel now consumes several PRNG values per element, so the sliding window advances past its own
  width and no two columns are byte-identical;
* ~~columns stay correlated in value (sliding window over one stream: element (read t, lane i)
  carries stream[t + i]; cross-position max |r| 0.618 against 0.035 for a host control)~~ --
  FIXED by the lane-salted counter RNG (tt-metal#52024, tracked as #52014): each lane's stream is
  keyed on its own position, and the argmax-collision gate below passes on P150x4.

Both were properties of the op (present in the raw ``ttnn.rand`` output, independent of the other
axis extent and of the seed), so they live in the rand suite rather than in a consumer's tests.
The tests stay as regression gates: they flip back to failures if cross-lane correlation returns.
"""

import pytest
import torch

import ttnn
from models.common.utility_functions import run_for_blackhole

TILE = 32
SEED = 48291

pytestmark = run_for_blackhole("This test targets the Blackhole SFPU PRNG implementation")


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
        key = columns[index].numpy().tobytes()
        if key in seen:
            duplicates.append((index, seen[key]))
        else:
            seen[key] = index
    assert not duplicates, (
        f"{len(duplicates)} of {columns.shape[0]} columns duplicate an earlier column; "
        f"first pairs {duplicates[:6]}, offsets {sorted({i - j for i, j in duplicates})}"
    )


def test_rand_columns_do_not_share_an_argmax(device):
    """Independent columns almost never pick the same row as their maximum.

    With ``height`` rows and ``n`` columns of IID noise, the expected number of colliding
    argmaxes is ``C(n, 2) / height`` -- about 2 here. Any sizeable tie group is a dependency
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
