# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""``fused_experts`` with its TP all-reduce fused in, against the same op followed by
``width_sharded_all_reduce``.

Each chip of a 1xR line gets its own token row, so the per-chip partials differ and the sum is
non-trivial. The fused output must match the two-op path on every chip (within bf16 rounding of one
sum) and be bit-identical across the chips of the ring.
"""

import pytest
import torch

import ttnn
from models.common.utility_functions import comp_pcc
from tests.ttnn.nightly.unit_tests.operations.experimental.test_fused_experts import (
    _expert_weights,
    _shared_expert_weights,
)

TILE = 32
DOWN_CORES = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 7))})


@pytest.mark.timeout(600)
@pytest.mark.parametrize("mesh_device", [(8, 4)], indirect=True)
@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True)
@pytest.mark.parametrize(
    "hidden, intermediate",
    [(4096, 512), (5120, 576)],
    ids=("v4_tp4", "v41_tp4"),
)
@pytest.mark.parametrize("num_links", (1, 2), ids=lambda n: f"links{n}")
@pytest.mark.parametrize("iterations", (3,))
def test_fused_experts_all_reduce(mesh_device, hidden, intermediate, num_links, iterations):
    submesh = mesh_device.create_submesh(ttnn.MeshShape(1, 4), ttnn.MeshCoordinate(0, 0))
    ring = submesh.get_num_devices()
    grid = submesh.compute_with_storage_grid_size()
    grid_y = 8 if (intermediate // TILE) % 16 == 0 else 9
    if grid.x < 13 or grid.y < grid_y:
        pytest.skip(f"needs the 12x{grid_y} expert grid plus a free sender column, got {grid.x}x{grid.y}")

    torch.manual_seed(0)
    num_experts, top_k = 16, 6
    limit, scaling, eps = 7.0, 2.5, 1e-20
    ids = torch.arange(top_k, dtype=torch.int64).reshape(1, 1, 1, top_k)
    scores = torch.rand((1, 1, 1, num_experts), dtype=torch.bfloat16).float() + 0.5
    x = (torch.rand((ring, 1, 1, hidden), dtype=torch.bfloat16) - 0.5).float()

    _, _, gate_up_tt, down_tt = _expert_weights(submesh, hidden, intermediate, num_experts)
    _, _, shared_gate_up_tt, shared_down_tt = _shared_expert_weights(submesh, hidden, intermediate)

    x_tt = ttnn.from_torch(
        x,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=submesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensorToMesh(submesh, dim=0),
    )
    ids_tt = ttnn.from_torch(
        ids.to(torch.int32),
        dtype=ttnn.uint16,
        layout=ttnn.TILE_LAYOUT,
        device=submesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    scores_tt = ttnn.from_torch(
        scores,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=submesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    out_mem = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(DOWN_CORES, [1, hidden // 16], ttnn.ShardOrientation.COL_MAJOR),
    )

    def run(**all_reduce):
        return ttnn.experimental.deepseek.moe.fused_experts(
            x_tt,
            routing_indices=ids_tt,
            routing_scores=scores_tt,
            gate_up_weights=gate_up_tt,
            down_weights=down_tt,
            num_experts=top_k,
            intermediate_size=intermediate,
            swiglu_limit=limit,
            top_k=top_k,
            routed_scaling_factor=scaling,
            routing_eps=eps,
            shared_gate_up_weights=shared_gate_up_tt,
            shared_down_weights=shared_down_tt,
            memory_config=out_mem,
            **all_reduce,
        )

    def per_chip(t):
        return ttnn.to_torch(t, mesh_composer=ttnn.ConcatMeshToTensor(submesh, dim=0)).float().reshape(ring, hidden)

    partial = run()
    expected = per_chip(
        ttnn.experimental.deepseek.width_sharded_all_reduce(
            partial, cluster_axis=1, num_links=num_links, topology=ttnn.Topology.Linear
        )
    )
    assert torch.allclose(per_chip(partial).sum(0).expand(ring, -1), expected, rtol=2e-2, atol=2e-2)

    # Several back-to-back calls exercise the semaphore reset and the persistent scratch.
    for i in range(iterations):
        fused = run(all_reduce_cluster_axis=1, all_reduce_num_links=num_links, all_reduce_topology=ttnn.Topology.Linear)
        assert fused.memory_config() == out_mem
        got = per_chip(fused)
        for chip in range(1, ring):
            assert torch.equal(got[0], got[chip]), f"iteration {i}: chip {chip} differs from chip 0"
        passing, msg = comp_pcc(expected, got, pcc=0.9999)
        assert passing, f"iteration {i}: fused all-reduce vs two-op path: {msg}"
        torch.testing.assert_close(got, expected, rtol=1e-2, atol=1e-2)
