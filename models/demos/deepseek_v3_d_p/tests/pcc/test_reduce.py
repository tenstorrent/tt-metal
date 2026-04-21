# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""
Test for post-combine reduce module in isolation.

This test verifies that the TTNN reduce module produces the same output as the
PyTorch reference implementation when reducing sparse combine outputs.

Uses synthetic sparse inputs to isolate the reduce operation from dispatch/combine.
"""

import pytest
import torch
from loguru import logger
from tracy import signpost

import ttnn
from models.demos.deepseek_v3_d_p.reference.tt.moe.reduce import TorchReduceModule
from models.demos.deepseek_v3_d_p.tt.moe.init_helpers import (
    create_sparse_combine_output,
    extract_mesh_config,
    get_tp_mesh_composer,
    initialize_test_inputs,
)
from models.demos.deepseek_v3_d_p.tt.moe.tt_prefill_transformer import TT_PREFILL_TRANSFORMER_L1_SMALL
from models.demos.deepseek_v3_d_p.tt.moe.tt_reduce import TtReduceModule
from tests.ttnn.utils_for_testing import comp_pcc


@pytest.mark.parametrize("use_weights", [True, False], ids=["weighted", "unweighted"])
@pytest.mark.parametrize(
    "seq_len, emb_dim, topk",
    [
        (32, 2048, 8),
        (3200, 7 * 1024, 8),  # DeepSeek values
    ],
)
@pytest.mark.parametrize(
    "mesh_device, device_params",
    [
        pytest.param(
            (4, 1),
            {"fabric_config": ttnn.FabricConfig.FABRIC_1D, "l1_small_size": TT_PREFILL_TRANSFORMER_L1_SMALL},
            marks=pytest.mark.requires_mesh_topology(mesh_shape=(4, 1), topology="linear"),
            id="linear-4",
        ),
        pytest.param(
            (4, 2),
            {"fabric_config": ttnn.FabricConfig.FABRIC_1D, "l1_small_size": TT_PREFILL_TRANSFORMER_L1_SMALL},
            marks=pytest.mark.requires_mesh_topology(mesh_shape=(4, 2), topology="mesh-4x2"),
            id="mesh-4x2",
        ),
    ],
    indirect=["mesh_device", "device_params"],
)
def test_ttnn_reduce(
    mesh_device,
    seq_len,
    emb_dim,
    topk,
    use_weights,
):
    """Test TTNN reduce module in isolation using synthetic sparse inputs."""
    torch.manual_seed(42)

    signpost(f"reduce-{mesh_device.shape}-seq{seq_len}-{'weighted' if use_weights else 'unweighted'}")

    # Set topology and num_links
    topology = ttnn.Topology.Linear
    num_links = 1

    num_devices = mesh_device.get_num_devices()
    mesh_config = extract_mesh_config(mesh_device)
    dispatch_group_size = mesh_config.dispatch_group_size
    num_dispatch_groups = mesh_config.num_dispatch_groups

    logger.debug(f"Testing with {mesh_device.shape=}, {num_devices=} {dispatch_group_size=} {num_dispatch_groups=}")

    ttnn.visualize_mesh_device(mesh_device)

    # Create synthetic sparse combine output
    torch_combine_output = create_sparse_combine_output(
        num_chips=dispatch_group_size,
        seq_len=seq_len,
        topk=topk,
        emb_dim=emb_dim,
        sparsity=0.75,
    )
    logger.debug(f"Created sparse combine output: {torch_combine_output.shape}")

    num_routed_experts = 64

    # Create random gate weights for weighted reduce (if enabled)
    torch_gate_weights = None
    if use_weights:
        _, torch_gate_weights, _ = initialize_test_inputs(
            dispatch_group_size=dispatch_group_size,
            seq_len_per_chip=seq_len,
            emb_dim=emb_dim,
            num_routed_experts=num_routed_experts,
            num_experts_per_tok=topk,
            max_dispatched_tokens_per_expert=1000,
            validate=False,
            skip_x_initialization=True,
        )
        logger.debug(f"Created gate weights: {torch_gate_weights.shape}")

    # Create indices and dispatch table for the reduce kernel.
    # Use a dispatch table where ALL experts are valid (no -1 entries) so
    # the kernel processes every expert slot — matching the torch reference.
    # Non-local expert skipping correctness is tested separately in
    # test_deepseek_moe_post_combine_reduce.py::test_skip_nonlocal_experts.
    torch_indices = torch.randint(0, num_routed_experts, (dispatch_group_size, seq_len, topk), dtype=torch.int32)
    expert_dispatch_table = torch.zeros((num_dispatch_groups, num_routed_experts), dtype=torch.int32)

    # Compute reference output using torch
    torch_reduce = TorchReduceModule(
        topk_dim=2,  # topk is dim 2 in [chips, seq, topk, hidden]
    )
    torch_shards = torch_reduce(torch_combine_output, weights=torch_gate_weights)
    logger.debug(f"Torch reference output: {len(torch_shards)} shards of shape {torch_shards[0].shape}")

    # Convert to TTNN tensor distributed across mesh
    # For reduce_scatter, we need each chip to have its portion of the input
    # Shape transformation: [num_reduce_chips, seq, topk, hidden] -> per-chip [seq, topk, hidden]

    mesh_mapper = ttnn.ShardTensor2dMesh(
        mesh_device,
        mesh_shape=mesh_device.shape,
        dims=(0, 2),  # Shard batch within dispatch group; shard topk across dispatch groups
    )

    tt_combine_output = ttnn.from_torch(
        torch_combine_output,
        mesh_mapper=mesh_mapper,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        dtype=ttnn.bfloat16,
    )
    logger.debug(f"{tt_combine_output.shape=}")

    # Convert gate weights to TTNN tensor with same sharding as combine_output (if enabled)
    tt_gate_weights = None
    if use_weights:
        tt_gate_weights = ttnn.from_torch(
            torch_gate_weights,
            mesh_mapper=mesh_mapper,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh_device,
            dtype=ttnn.bfloat16,
        )
        logger.debug(f"{tt_gate_weights.shape=}")

    # Convert indices and dispatch table to TTNN tensors
    indices_mapper = ttnn.ShardTensor2dMesh(
        mesh_device,
        mesh_shape=mesh_device.shape,
        dims=(0, None),
    )
    tt_indices = ttnn.from_torch(
        torch_indices,
        mesh_mapper=indices_mapper,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        dtype=ttnn.int32,
    )
    tt_expert_dispatch_table = ttnn.from_torch(
        expert_dispatch_table,
        mesh_mapper=ttnn.ShardTensor2dMesh(
            mesh_device,
            mesh_shape=mesh_device.shape,
            dims=(None, 0),
        ),
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        dtype=ttnn.int32,
    )

    # Run TTNN reduce
    # NOTE: TTNN adds a batch dim, so [seq, topk, hidden] becomes [1, seq, topk, hidden]
    # topk is at dim=2 in the 4D tensor
    tt_reduce = TtReduceModule(
        mesh_device=mesh_device,
        topk_dim=2,  # topk is dim 2 in [1, seq, topk, hidden]
        cluster_axis=1,
        num_links=num_links,
        topology=topology,
    )

    tt_output = tt_reduce(
        tt_combine_output,
        weights=tt_gate_weights,
        indices=tt_indices,
        expert_dispatch_table=tt_expert_dispatch_table,
    )
    logger.debug(f"{tt_output.shape=}")

    composer = get_tp_mesh_composer(mesh_device)
    tt_host = ttnn.to_torch(tt_output, mesh_composer=composer, dtype=torch.bfloat16)
    logger.debug(f"{tt_host.shape=}")
    threshold = 0.999
    _, pcc = comp_pcc(torch_shards.float(), tt_host.float())

    logger.debug(f"TTNN reduce operation matches torch reference! (PCC={pcc:.6f})")
    assert pcc > threshold, f"PCC {pcc:.6f} below threshold {threshold}"
