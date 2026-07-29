# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Extended (verifier-authored) coverage for `ttnn.operations.tilize.tilize`.

Deliberately small — it only covers the gaps found during the verification code
review, everything else is already covered by the acceptance suite
(`test_tilize.py`) and the golden suite (`eval/golden_tests/tilize/`):

1. **High rank** — the program folds every leading dim into one row axis, so it
   is rank-agnostic by construction. `SUPPORTED["rank"]` claims 5 and 6 beyond
   the golden TARGET's [2,3,4]; this is the evidence for that claim.
2. **Awkward `Wt`** — the planner picks `chunk_wt` as the largest divisor of
   `Wt` (never skipping 5 or 7, unlike the C++ `find_max_divisor`). `Wt ∈
   {5, 7}` on a wide-short shape is where a divisor bug collapses parallelism
   or breaks the `chunk_wt | Wt` invariant.
3. **`Wt` far above `WT_CHUNK_MAX`** — the per-core CB must stay bounded by a
   constant in W, and the width split must still tile the grid exactly.
4. **Depth-1 on the zero-copy sharded path** — `use_double_buffer=False` with
   same-spec L1 shards. The golden harness never forwards `use_double_buffer`,
   so nothing else exercises depth-1 against an aliased CB.

No PCC-tolerance games: tilize is value-preserving, so every case here is
checked with `torch.equal` (bit-identity).
"""

from __future__ import annotations

import pytest
import torch

import ttnn
from ttnn.operations.tilize import tilize
from ttnn.operations.tilize.tilize_program_descriptor import (
    L1_CB_BUDGET_BYTES,
    build_plan,
)


def _roundtrip(device, shape, *, use_multicore=True, use_double_buffer=True, memory_config=None):
    torch.manual_seed(0)
    torch_input = torch.randn(shape).bfloat16()
    tt_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=memory_config if memory_config is not None else ttnn.DRAM_MEMORY_CONFIG,
    )
    tt_output = tilize(
        tt_input,
        memory_config,
        use_multicore=use_multicore,
        use_double_buffer=use_double_buffer,
    )
    assert tt_output.layout == ttnn.TILE_LAYOUT
    return torch_input, ttnn.to_torch(tt_output)


# ---------------------------------------------------------------------------
# 1. High rank — the fold is rank-agnostic
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "shape",
    [
        pytest.param((2, 2, 2, 32, 64), id="rank5"),
        pytest.param((2, 2, 2, 2, 32, 32), id="rank6"),
    ],
)
@pytest.mark.parametrize("use_multicore", [False, True], ids=["single_core", "multi_core"])
def test_tilize_high_rank(device, shape, use_multicore):
    expected, actual = _roundtrip(device, shape, use_multicore=use_multicore)
    assert torch.equal(expected, actual), "high-rank tilize must be bit-exact"


# ---------------------------------------------------------------------------
# 2/3. Width-split edge cases: awkward and very large Wt
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "shape",
    [
        pytest.param((1, 1, 32, 160), id="wt5_wide_short"),  # Wt = 5 (prime)
        pytest.param((1, 1, 32, 224), id="wt7_wide_short"),  # Wt = 7 (prime)
        pytest.param((1, 1, 64, 2016), id="wt63_odd"),  # Wt = 63 = 7*9
        pytest.param((1, 1, 32, 8192), id="wt256_wide"),  # Wt = 256 >> WT_CHUNK_MAX
    ],
)
def test_tilize_width_split_edges(device, shape):
    expected, actual = _roundtrip(device, shape)
    assert torch.equal(expected, actual), "width-split tilize must be bit-exact"


@pytest.mark.parametrize(
    "shape",
    [
        pytest.param((1, 1, 32, 160), id="wt5"),
        pytest.param((1, 1, 32, 8192), id="wt256"),
        pytest.param((1, 1, 2048, 2048), id="square64"),
    ],
)
def test_tilize_plan_invariants(device, shape):
    """Host-planner invariants that a wrong split would silently break."""
    torch.manual_seed(0)
    tt_input = ttnn.from_torch(
        torch.randn(shape).bfloat16(),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    tt_output = ttnn.allocate_tensor_on_device(
        ttnn.Shape(list(shape)), ttnn.bfloat16, ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG
    )
    plan = build_plan(tt_input, tt_output, device, use_multicore=True)

    grid = device.compute_with_storage_grid_size()
    assert plan["wt"] % plan["chunk_wt"] == 0, "chunk_wt must divide Wt exactly"
    # A0: every core with work is launched, and no more.
    assert plan["ncores"] == min(
        grid.x * grid.y, plan["total_tiles"]
    ), f"expected {min(grid.x * grid.y, plan['total_tiles'])} cores, got {plan['ncores']}"
    # Per-core CB L1 is bounded by a constant, not by W.
    assert plan["cb_bytes_per_core"] <= L1_CB_BUDGET_BYTES
    # The 2D split covers the tile grid exactly once.
    covered = sum(unit["row_count"] * unit["chunk_count"] for unit in plan["work"])
    assert covered * plan["chunk_wt"] == plan["total_tiles"], "the 2D split must tile the grid exactly"


# ---------------------------------------------------------------------------
# 4. Depth-1 on the zero-copy (aliased-CB) sharded path
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("use_double_buffer", [False, True], ids=["depth1", "depth2"])
def test_tilize_sharded_double_buffer(device, use_double_buffer):
    """Same-spec L1 shards: depth is structurally 1 (the CB *is* the shard), so
    `use_double_buffer` must be inert rather than doubling the aliased CB."""
    mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 0))}),
            (128, 64),
            ttnn.ShardOrientation.ROW_MAJOR,
        ),
    )
    expected, actual = _roundtrip(
        device,
        (1, 1, 512, 64),
        use_double_buffer=use_double_buffer,
        memory_config=mem_config,
    )
    assert torch.equal(expected, actual), "sharded tilize must be bit-exact at either CB depth"
