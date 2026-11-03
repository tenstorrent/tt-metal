# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Laguna's token-dispatch routing metadata (tt/routing_setup.py) against a torch reference, every chip of a 1 x D
mesh with one-chip dispatch groups. Each chip gets its own random top-k indices."""

import pytest
import torch

import ttnn
from models.demos.deepseek_v3_d_p.tt.moe.init_helpers import ExpertMapping
from models.demos.laguna.tt.routing_setup import LagunaRoutingSetup


def _reference(indices, table, experts_per_chip):
    """indices [T, K] of one chip, table [E + 1] of its dispatch group -> (histogram, region offsets), both [E]."""
    E = table.shape[-1] - 1
    hist = torch.bincount(indices.flatten().long(), minlength=E)[:E]
    hist[table[:E] < 0] = 0
    padded = (hist + 31) // 32 * 32
    region = torch.zeros(E, dtype=torch.int64)
    for g in range(0, E, experts_per_chip):
        group = padded[g : g + experts_per_chip]
        region[g : g + experts_per_chip] = torch.cumsum(group, 0) - group
    return hist, region


@pytest.mark.parametrize("tokens", [256, 2048])
@pytest.mark.parametrize("num_experts, top_k", [(256, 8)])
def test_routing_setup_matches_reference(tokens, num_experts, top_k):
    num_devices = min(4, ttnn.get_num_devices())
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, num_devices))
    try:
        torch.manual_seed(0)
        experts_per_chip = num_experts // num_devices
        table = ExpertMapping.create_dispatch_table(
            num_routed_experts=num_experts, dispatch_group_size=1, num_dispatch_groups=num_devices
        )
        indices = torch.stack(
            [torch.stack([torch.randperm(num_experts)[:top_k] for _ in range(tokens)]) for _ in range(num_devices)]
        ).to(torch.int32)
        tt_indices = ttnn.from_torch(
            indices,
            dtype=ttnn.uint16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=0),
        )
        out = LagunaRoutingSetup(mesh, table, experts_per_chip)(tt_indices, num_experts, top_k)

        def chips(t):
            return ttnn.to_torch(t, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0)).reshape(num_devices, -1)

        hist, offsets, totals, region, all_offsets = (
            chips(t).long()
            for t in (
                out.expert_histograms,
                out.global_dispatch_offsets,
                out.total_counts_per_expert,
                out.expert_region_offsets,
                out.all_global_dispatch_offsets,
            )
        )
        for d in range(num_devices):
            ref_hist, ref_region = _reference(indices[d], table[d], experts_per_chip)
            assert torch.equal(hist[d], ref_hist), f"chip {d}: histogram"
            assert torch.equal(totals[d], ref_hist), f"chip {d}: totals"
            assert torch.equal(region[d], ref_region), f"chip {d}: expert region offsets"
            assert torch.equal(offsets[d], ref_region), f"chip {d}: dispatch offsets"
            assert torch.equal(all_offsets[d], ref_region), f"chip {d}: all dispatch offsets"
    finally:
        ttnn.close_mesh_device(mesh)
