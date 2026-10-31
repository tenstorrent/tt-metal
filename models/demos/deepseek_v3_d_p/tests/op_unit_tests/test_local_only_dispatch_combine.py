# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Coverage for independent size-one MoE dispatch groups on a 1xN mesh."""

import os

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.deepseek_v3_d_p.tt.moe.init_helpers import (
    ExpertMapping,
    create_fabric_router_config,
    get_dispatch_input_mesh_mapper,
    get_ep_mesh_composer,
    get_gate_outputs,
)
from models.demos.deepseek_v3_d_p.tt.moe.tt_combine import TtCombineModule
from models.demos.deepseek_v3_d_p.tt.moe.tt_dispatch import TtDispatchModule
from models.demos.deepseek_v3_d_p.tt.moe.tt_moe_routing_setup import TtMoERoutingSetup

SEQ_LEN = 256
EMB_DIM = 256
NUM_EXPERTS = 8
TOP_K = 2
NUM_DISPATCH_GROUPS = 2
DISPATCH_GROUP_SIZE = 1
EXPERTS_PER_CHIP = NUM_EXPERTS // NUM_DISPATCH_GROUPS
MAX_DISPATCH_BUFFER_TOKENS = SEQ_LEN * TOP_K
RUN_DEVICE_TEST = os.environ.get("TT_RUN_LOCAL_ONLY_MOE_DEVICE_TEST") == "1"


def _balanced_inputs():
    torch.manual_seed(42)
    x = torch.randn((DISPATCH_GROUP_SIZE, SEQ_LEN, EMB_DIM), dtype=torch.bfloat16)
    weights = torch.ones((DISPATCH_GROUP_SIZE, SEQ_LEN, TOP_K), dtype=torch.bfloat16)
    indices = torch.arange(SEQ_LEN * TOP_K, dtype=torch.int32).reshape(DISPATCH_GROUP_SIZE, SEQ_LEN, TOP_K)
    indices %= NUM_EXPERTS
    return x, weights, indices


def test_local_only_expert_mapping_and_row_plan():
    """Each column owns a disjoint expert row and needs no cross-device offset."""
    _, _, indices = _balanced_inputs()
    dispatch_table = ExpertMapping.create_dispatch_table(
        num_routed_experts=NUM_EXPERTS,
        dispatch_group_size=DISPATCH_GROUP_SIZE,
        num_dispatch_groups=NUM_DISPATCH_GROUPS,
    )

    assert dispatch_table.tolist() == [
        [0, 0, 0, 0, -1, -1, -1, -1],
        [-1, -1, -1, -1, 0, 0, 0, 0],
    ]

    offsets, counts, region_offsets, per_source_counts = get_gate_outputs(
        indices,
        DISPATCH_GROUP_SIZE,
        NUM_EXPERTS,
        EXPERTS_PER_CHIP,
        SEQ_LEN,
        TOP_K,
        expert_dispatch_table=dispatch_table,
    )

    # With one source in each independent group there is no source-prefix term.
    assert torch.equal(offsets, region_offsets)
    assert torch.equal(counts, per_source_counts)

    expected_dense_counts = torch.bincount(indices.flatten(), minlength=NUM_EXPERTS).to(torch.int32)
    for group in range(NUM_DISPATCH_GROUPS):
        group_mask = dispatch_table[group] >= 0
        assert torch.equal(counts[group, 0, group_mask], expected_dense_counts[group_mask])
        assert torch.count_nonzero(counts[group, 0, ~group_mask]) == 0

        padded_rows = sum(
            ttnn.TILE_SIZE * ((int(count) + ttnn.TILE_SIZE - 1) // ttnn.TILE_SIZE)
            for count in counts[group, 0, group_mask]
        )
        assert padded_rows <= MAX_DISPATCH_BUFFER_TOKENS


def test_unified_routed_expert_chunk_override_binding_contract():
    """Both per-expert and composite bindings expose the same optional override."""
    for operation_name in (
        "unified_routed_expert_ffn",
        "unified_routed_expert_moe",
        "unified_routed_expert_moe_stacked",
    ):
        operation = getattr(ttnn.experimental.deepseek_prefill, operation_name)
        generated_signature = operation.function.__call__.__doc__
        assert "chunk_m_tiles_override: int | None = None" in generated_signature
    assert "BFLOAT16 (BF16 intermediates/output)" in (
        ttnn.experimental.deepseek_prefill.unified_routed_expert_ffn.__doc__ or ""
    )


@pytest.mark.parametrize(
    "mesh_device, device_params",
    [
        pytest.param(
            (1, 2),
            {
                "fabric_config": ttnn.FabricConfig.FABRIC_1D,
                # This test is Blackhole-only. Keep collection hardware-free by
                # using the BH constant rather than probing the current arch.
                "fabric_router_config": create_fabric_router_config(max_payload_size=14 * 1024),
            },
            id="local-groups-1x2",
        )
    ],
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.skipif(
    not RUN_DEVICE_TEST,
    reason="set TT_RUN_LOCAL_ONLY_MOE_DEVICE_TEST=1 to open the 1x2 mesh",
)
def test_local_only_dispatch_combine_roundtrip(mesh_device):
    """A 1x2 mesh dispatches and combines locally with one expert group per column."""
    x, weights, indices = _balanced_inputs()
    dispatch_table = ExpertMapping.create_dispatch_table(
        num_routed_experts=NUM_EXPERTS,
        dispatch_group_size=DISPATCH_GROUP_SIZE,
        num_dispatch_groups=NUM_DISPATCH_GROUPS,
    )
    ref_offsets, ref_counts, _, _ = get_gate_outputs(
        indices,
        DISPATCH_GROUP_SIZE,
        NUM_EXPERTS,
        EXPERTS_PER_CHIP,
        SEQ_LEN,
        TOP_K,
        expert_dispatch_table=dispatch_table,
    )

    input_mapper = get_dispatch_input_mesh_mapper(mesh_device, sp_axis=0)
    tt_x = ttnn.from_torch(
        x,
        mesh_mapper=input_mapper,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        dtype=ttnn.bfloat16,
    )
    tt_weights = ttnn.from_torch(
        weights,
        mesh_mapper=input_mapper,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        dtype=ttnn.bfloat16,
    )
    tt_indices = ttnn.from_torch(
        indices,
        mesh_mapper=input_mapper,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        dtype=ttnn.uint16,
    )

    routing_setup = TtMoERoutingSetup(
        mesh_device=mesh_device,
        expert_dispatch_table=dispatch_table,
        num_links=1,
        experts_per_chip=EXPERTS_PER_CHIP,
    )
    tt_offsets, tt_counts, tt_region_offsets, _ = routing_setup(
        ttnn_top_k_experts_indices=indices,
        num_routed_experts=NUM_EXPERTS,
        seq_len_per_chip=SEQ_LEN,
        num_experts_per_tok=TOP_K,
    )

    ep_composer = get_ep_mesh_composer(mesh_device)
    host_offsets = ttnn.to_torch(ttnn.unsqueeze_to_4D(tt_offsets), mesh_composer=ep_composer).squeeze(2)
    host_counts = ttnn.to_torch(ttnn.unsqueeze_to_4D(tt_counts), mesh_composer=ep_composer).squeeze(2)
    assert torch.equal(host_offsets.to(torch.int32), ref_offsets)
    assert torch.equal(host_counts.to(torch.int32), ref_counts)

    dispatch = TtDispatchModule(
        mesh_device=mesh_device,
        dispatch_group_size=DISPATCH_GROUP_SIZE,
        experts_per_chip=EXPERTS_PER_CHIP,
        num_routed_experts=NUM_EXPERTS,
        num_experts_per_tok=TOP_K,
        metadata_len=5,
        max_dispatch_buffer_token_size=MAX_DISPATCH_BUFFER_TOKENS,
        seq_len_per_chip=SEQ_LEN,
        emb_dim=EMB_DIM,
        cluster_axis=0,
        num_links=1,
        topology=ttnn.Topology.Linear,
    )
    tt_dispatch_table = TtDispatchModule.shard_expert_dispatch_table(mesh_device, dispatch_table, dispatch_axis=0)
    dispatched_buffer, metadata = dispatch(tt_x, tt_weights, tt_indices, tt_offsets, tt_dispatch_table)

    combine = TtCombineModule(
        mesh_device=mesh_device,
        dispatch_group_size=DISPATCH_GROUP_SIZE,
        num_dispatch_groups=NUM_DISPATCH_GROUPS,
        experts_per_chip=EXPERTS_PER_CHIP,
        num_experts_per_tok=TOP_K,
        seq_len_per_chip=SEQ_LEN,
        cluster_axis=0,
        num_links=1,
        topology=ttnn.Topology.Linear,
        init_zeros=False,
    )
    output = combine(dispatched_buffer, metadata, tt_counts, tt_region_offsets)
    ttnn.synchronize_device(mesh_device)

    composed = ttnn.to_torch(output, mesh_composer=ep_composer, dtype=torch.bfloat16)
    assert composed.shape == (NUM_DISPATCH_GROUPS, DISPATCH_GROUP_SIZE, SEQ_LEN, TOP_K, EMB_DIM)
    selected = torch.empty_like(x.unsqueeze(2).expand(-1, -1, TOP_K, -1))
    experts_per_group = NUM_EXPERTS // NUM_DISPATCH_GROUPS
    for token in range(SEQ_LEN):
        for topk in range(TOP_K):
            group = int(indices[0, token, topk]) // experts_per_group
            selected[0, token, topk] = composed[group, 0, token, topk]

    expected = x.unsqueeze(2).expand_as(selected)
    passed, pcc = comp_pcc(expected.float(), selected.float(), pcc=0.9999)
    logger.info(f"local-only dispatch/combine selected-row PCC={pcc:.8f}")
    assert passed, f"selected-row PCC {pcc:.8f} is below 0.9999"
    torch.testing.assert_close(selected, expected, atol=1e-2, rtol=1e-2)
