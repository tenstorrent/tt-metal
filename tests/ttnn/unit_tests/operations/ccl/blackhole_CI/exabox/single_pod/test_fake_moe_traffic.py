# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Real-MoE-wire reduce-to-one chain on the (4, 2) per-rank submesh.

Exercises the demo's actual `ReduceToOneB1` op (3-level tree, 8 → 1) under
`FABRIC_2D_TORUS_Y` + `fabric_router_config(15232)` + `worker_l1_size=1431568`
— the exact configuration the deepseek_v3_b1 single-pod demo runs under.

`num_blocks` parametrize:
  - `1block`:  one ReduceToOneB1 call per rank (per-token wire workload).
  - `10blocks`: ten back-to-back ReduceToOneB1 calls per rank, mirroring a
                10-token decode of the 10-MoE-stage single-pod pipeline.

Launch (16 ranks, 4 hosts × 4 procs):
  scripts/run_chain_test.sh \\
    'test_fake_moe_chain_real_reduce_to_one_4x2_single_pod[1block-h7168-root_0_1-1link-4x2_grid-fabric_2d_torus_y]'
"""

from __future__ import annotations

import time

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.deepseek_v3_b1.demo.pipeline import (
    create_fabric_router_config,
)
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_pcc
from tests.ttnn.unit_tests.operations.ccl.blackhole_CI.exabox.single_pod._fake_moe_helpers import (
    _setup_sub_devices,
    _teardown_sub_devices,
    make_reduce_to_one_b1_inputs,
    step_reduce_to_one_b1,
)

# (4, 2) is the per-rank submesh shape used by every MoE rank in the
# blitz_decode single-pod descriptor.
PER_RANK_MESH_SHAPE = (4, 2)


@pytest.mark.requires_device(["QUAD_BH"])
@pytest.mark.parametrize(
    "device_params, topology",
    [
        pytest.param(
            {
                "fabric_config": ttnn.FabricConfig.FABRIC_2D_TORUS_Y,
                "fabric_router_config": create_fabric_router_config(15232),
                "worker_l1_size": 1431568,
            },
            ttnn.Topology.Linear,
            id="fabric_2d_torus_y",
        ),
    ],
    indirect=["device_params"],
)
@pytest.mark.parametrize("mesh_device", [pytest.param(PER_RANK_MESH_SHAPE, id="4x2_grid")], indirect=True)
@pytest.mark.parametrize("num_links", [1], ids=["1link"])
@pytest.mark.parametrize("root_coord", [pytest.param((0, 1), id="root_0_1")])  # corner row required by torus mode
@pytest.mark.parametrize("hidden", [pytest.param(7168, id="h7168")])  # demo's ACTIVATION_DIM
@pytest.mark.parametrize("num_blocks", [1, 10], ids=["1block", "10blocks"])
def test_fake_moe_chain_real_reduce_to_one_4x2_single_pod(
    mesh_device,
    topology,
    num_links,
    root_coord,
    hidden,
    num_blocks,
):
    """Run `ReduceToOneB1` `num_blocks` times back-to-back on the (4, 2)
    per-rank submesh. Input is `sender` replicated on every device; the
    op leaves `8 × sender` on the root device after each call."""
    torch.manual_seed(0)
    sender = torch.randn(1, hidden, dtype=torch.bfloat16)

    # 8 worker cores from the optimal DRAM-bank mapping (matches the demo's
    # MoE shard_grid; ReduceToOneB1 expects width sharding across exactly 8 cores).
    compute_cores = mesh_device.get_optimal_dram_bank_to_logical_worker_assignment(ttnn.NOC.NOC_0)
    shard_grid = ttnn.CoreRangeSet({ttnn.CoreRange(c, c) for c in compute_cores[:8]})

    sub_device_id, _, sub_device_manager = _setup_sub_devices(mesh_device)
    try:
        bundle = make_reduce_to_one_b1_inputs(sender, mesh_device, shard_grid=shard_grid, is_torus=True)

        block_durations_s = []
        tt_root = None
        for i in range(num_blocks):
            t0 = time.perf_counter()
            tt_root = step_reduce_to_one_b1(
                bundle["input_tensor"],
                bundle["intermediate_tensor"],
                bundle["output_tensor"],
                bundle["semaphores"],
                root_coord,
                exit_coord=root_coord,
                is_torus=True,
            )
            ttnn.synchronize_device(mesh_device)
            block_durations_s.append(time.perf_counter() - t0)
            logger.info(
                "[reduce_to_one] block {}/{} on (4,2) took {:.3f}s",
                i + 1,
                num_blocks,
                block_durations_s[-1],
            )

        # Verify the root device after the final iteration only — every
        # iteration produces the same value (8 × sender), so checking once
        # at the end is sufficient.
        mesh_cols = mesh_device.shape[1]
        root_idx = root_coord[0] * mesh_cols + root_coord[1]
        all_devs_torch = ttnn.to_torch(tt_root, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))
        per_device = all_devs_torch.reshape(mesh_device.shape[0] * mesh_cols, 1, hidden)
        root_out = per_device[root_idx]

        expected = (sender.float() * 8.0).bfloat16()
        eq, mess = comp_pcc(root_out, expected, 0.998)
        assert eq, f"root device PCC failed after {num_blocks} block(s): {mess}"

        logger.info(
            "[reduce_to_one] {} blocks complete; total {:.3f}s, mean per-block {:.3f}s",
            num_blocks,
            sum(block_durations_s),
            sum(block_durations_s) / len(block_durations_s),
        )
    finally:
        _teardown_sub_devices(mesh_device, sub_device_manager)
