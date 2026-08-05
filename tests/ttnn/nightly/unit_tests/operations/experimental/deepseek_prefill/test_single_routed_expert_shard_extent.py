# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Regression test for the ND-shard grid extent.

The kernels address a DRAM ND-sharded weight tensor by forming the linear
ROUND_ROBIN_1D shard id as ``krow * extent + my_nt``, where ``extent`` is the number of
shards along N. That extent is ``ceil(N_tiles / per_core_N)``, and since
``per_core_N = ceil(N_tiles / GRID_X)`` it equals GRID_X only when that division is
tight. Every shipped model shape happens to be tight (extent 11 = GRID_X), so passing
GRID_X went unnoticed; it is wrong for 55 of the first 299 tile counts.

emb=512 (16 tiles) / hidden=384 (12 tiles) mismatches on BOTH tensors: gate/up gets
per_core_N=2 and extent 6, down gets per_core_N=2 and extent 8 — neither is 11. With
GRID_X substituted for the extent, this shape produced PCC -0.001 / 0.000 (random
output) on the ND-sharded path while the interleaved path stayed correct, since
interleaved indexes ``row * N_tiles_full + col`` and never uses the shard grid.

Both weight layouts and both x layouts are covered so the interleaved path acts as the
control: a regression in the shard-id arithmetic must fail ND-shard while interleaved
still passes.
"""

import pytest

from tests.ttnn.nightly.unit_tests.operations.experimental.deepseek_prefill.test_single_routed_expert import (
    run_single_routed_expert,
)


@pytest.mark.parametrize("x_row_major", [False, True], ids=["x_tile", "x_rm"])
@pytest.mark.parametrize("weights_dram_sharded", [False, True], ids=["w_interleaved", "w_ndshard"])
def test_single_routed_expert_shard_extent_ne_grid_x(device, x_row_major, weights_dram_sharded):
    """emb/hidden chosen so the shard-grid N extent != GRID_X for gate/up AND down."""
    run_single_routed_expert(
        device,
        allocated_tokens=256,
        emb_dim=512,
        hidden_dim=384,
        active_tokens=128,
        x_row_major=x_row_major,
        weights_dram_sharded=weights_dram_sharded,
    )
