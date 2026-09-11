# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Regression test for the FFN's weight reads at a shape where the N split is NOT tight.

The op splits N across GRID_X = 11 cores as per_core_N = ceil(N_tiles / GRID_X), so the padded
width per_core_N * GRID_X exceeds the real N unless that division comes out tight. It is tight for
every shipped model shape, which hides the padded case: the last cores hold a partially valid slice
or none at all, and a weight read that does not clip its tile-column window to the real N walks off
the tensor.

emb=512 (16 tiles) / hidden=384 (12 tiles) is padded on BOTH tensors -- gate/up gets per_core_N=2
over 6 populated cores, down per_core_N=2 over 8 -- so three and five cores respectively own a
window entirely past N. Under DRAM ND-sharded weights that window is also where a coalesced shard
run would over-read, which the interleaved path cannot expose because it addresses one tile at a
time. Both weight layouts and both x layouts run, so interleaved acts as the control: a regression
in the run clipping must fail ND-shard while interleaved still passes.
"""

import pytest

from models.common.utility_functions import is_blackhole
from tests.ttnn.nightly.unit_tests.operations.experimental.deepseek_prefill.test_single_routed_expert import (
    run_single_routed_expert,
)


@pytest.mark.parametrize("x_row_major", [False, True], ids=["x_tile", "x_rm"])
@pytest.mark.parametrize("weights_dram_sharded", [False, True], ids=["w_interleaved", "w_ndshard"])
@pytest.mark.skipif(not is_blackhole(), reason="device-side count-aware sparsity is Blackhole-only")
def test_single_routed_expert_padded_n_split(device, x_row_major, weights_dram_sharded):
    """emb/hidden chosen so per_core_N * GRID_X overshoots N for gate/up AND down."""
    run_single_routed_expert(
        device,
        allocated_tokens=256,
        emb_dim=512,
        hidden_dim=384,
        active_tokens=128,
        x_row_major=x_row_major,
        weights_dram_sharded=weights_dram_sharded,
    )
