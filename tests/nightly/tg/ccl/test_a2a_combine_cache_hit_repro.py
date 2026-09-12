# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Standalone repro for #47108: all_to_all_combine produces WRONG output (all zeros)
on a program-cache HIT when the SAME input tensors are reused across iterations.

Filed on Wormhole Galaxy 8x4 (gengelage/all_to_all_combine_bug_repro). Existing
nightlies reallocate tensors every iteration and therefore stay green.

Run on Galaxy / TG 8x4:
  export TT_METAL_HOME=$(git rev-parse --show-toplevel)
  pytest -svv tests/nightly/tg/ccl/test_a2a_combine_cache_hit_repro.py
"""

import pytest
import ttnn

from tests.nightly.t3000.ccl.test_a2a_combine_cache_hit_repro import (
    run_a2a_combine_static_buffer_cache_hit_repro,
)


@pytest.mark.parametrize(
    "device_params",
    [
        {
            "dispatch_core_axis": ttnn.DispatchCoreAxis.COL,
            "reliability_mode": ttnn.FabricReliabilityMode.RELAXED_INIT,
            "fabric_config": ttnn.FabricConfig.FABRIC_1D,
        },
    ],
    ids=["fabric_1d_line"],
    indirect=True,
)
@pytest.mark.parametrize(
    "mesh_shape, mesh_device",
    [pytest.param((8, 4), (8, 4), id="8x4_grid")],
    indirect=["mesh_device"],
)
@pytest.mark.parametrize("axis", [1])
@pytest.mark.parametrize("batches_per_device", [8])
@pytest.mark.parametrize("seq", [2])
@pytest.mark.parametrize("local_reduce", [False, True], ids=["dense", "sparse"])
@pytest.mark.parametrize("experts", [256])
@pytest.mark.parametrize("select_experts_k", [8])
@pytest.mark.parametrize("hidden_size", [7168])
@pytest.mark.parametrize("num_iters", [3])
@pytest.mark.parametrize("num_links", [4])
def test_a2a_combine_cache_hit_repro(
    mesh_device,
    mesh_shape,
    axis,
    batches_per_device,
    seq,
    local_reduce,
    experts,
    select_experts_k,
    hidden_size,
    num_iters,
    num_links,
):
    run_a2a_combine_static_buffer_cache_hit_repro(
        mesh_device,
        mesh_shape,
        axis,
        batches_per_device,
        seq,
        local_reduce,
        experts,
        select_experts_k,
        hidden_size,
        num_iters,
        num_links,
    )
