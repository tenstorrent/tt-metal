# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

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
from models.demos.deepseek_v3_d_p.tt.moe.init_helpers import extract_mesh_config, get_tp_mesh_composer
from models.demos.deepseek_v3_d_p.tt.moe.tt_reduce import TtReduceModule
from tests.ttnn.utils_for_testing import comp_pcc


def create_random_weights(
    num_chips: int,
    seq_len: int,
    topk: int,
    seed: int = 123,
) -> torch.Tensor:
    """
    Create random gate weights for testing weighted reduce.

    Args:
        num_chips: Number of chips in the reduction group
        seq_len: Sequence length per chip
        topk: Number of expert slots per token
        seed: Random seed for reproducibility

    Returns:
        Weights tensor of shape [num_chips, seq_len, topk]
    """
    torch.manual_seed(seed)
    # Random weights in [0, 1], normalized so each token's topk sums to 1
    weights = torch.rand(num_chips, seq_len, topk, dtype=torch.bfloat16)
    weights = weights / weights.sum(dim=-1, keepdim=True)
    return weights


def create_sparse_combine_output(
    num_chips: int,
    seq_len: int,
    topk: int,
    hidden_dim: int,
    sparsity: float = 0.75,
    seed: int = 42,
) -> torch.Tensor:
    """
    Create synthetic sparse combine output for testing.

    In real MoE, combine output is sparse because each chip only has valid data
    for tokens routed to its local experts. This function simulates that sparsity.

    Args:
        num_chips: Number of chips in the reduction group
        seq_len: Sequence length per chip
        topk: Number of expert slots per token
        hidden_dim: Hidden dimension
        sparsity: Fraction of positions that are zero (default 0.75)
        seed: Random seed for reproducibility

    Returns:
        Sparse tensor of shape [num_chips, seq_len, topk, hidden_dim]
    """
    torch.manual_seed(seed)

    # Create random data
    data = torch.randn(num_chips, seq_len, topk, hidden_dim, dtype=torch.bfloat16)

    # Apply sparsity mask (zero out random positions in topk dimension)
    mask = torch.rand(num_chips, seq_len, topk, 1) > sparsity
    data = data * mask.to(torch.bfloat16)

    return data


@pytest.mark.parametrize("use_weights", [True, False], ids=["weighted", "unweighted"])
@pytest.mark.parametrize(
    "seq_len, hidden_dim, topk",
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
            {"fabric_config": ttnn.FabricConfig.FABRIC_1D},
            marks=pytest.mark.requires_mesh_topology(mesh_shape=(8, 1), topology="linear"),
            id="linear-4",
        ),
        pytest.param(
            (4, 2),
            {"fabric_config": ttnn.FabricConfig.FABRIC_1D},
            marks=pytest.mark.requires_mesh_topology(mesh_shape=(4, 2), topology="mesh-4x2"),
            id="mesh-4x2",
        ),
    ],
    indirect=["mesh_device", "device_params"],
)
def test_ttnn_reduce(
    mesh_device,
    seq_len,
    hidden_dim,
    topk,
    use_weights,
):
    """Test TTNN reduce module in isolation using synthetic sparse inputs."""

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
        hidden_dim=hidden_dim,
        sparsity=0.75,
        seed=42,
    )
    logger.debug(f"Created sparse combine output: {torch_combine_output.shape}")

    # Create random weights for weighted reduce (if enabled)
    torch_weights = None
    if use_weights:
        torch_weights = create_random_weights(
            num_chips=dispatch_group_size,
            seq_len=seq_len,
            topk=topk,
            seed=123,
        )
        logger.debug(f"Created weights: {torch_weights.shape}")

    # Compute reference output using torch
    torch_reduce = TorchReduceModule(
        topk_dim=1,  # topk is dim 1 in [seq, topk, hidden]
    )
    torch_shards = torch_reduce(torch_combine_output, weights=torch_weights)
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
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        dtype=ttnn.bfloat16,
    )
    logger.debug(f"{tt_combine_output.shape=}")

    # Convert weights to TTNN tensor with same sharding as combine_output (if enabled)
    tt_weights = None
    if use_weights:
        tt_weights = ttnn.from_torch(
            torch_weights,
            mesh_mapper=mesh_mapper,
            layout=ttnn.ROW_MAJOR_LAYOUT,  # Will be converted to TILE inside reduce module
            device=mesh_device,
            dtype=ttnn.bfloat16,
        )
        logger.debug(f"{tt_weights.shape=}")

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

    tt_output = tt_reduce(tt_combine_output, weights=tt_weights)
    logger.debug(f"{tt_output.shape=}")

    composer = get_tp_mesh_composer(mesh_device)
    tt_host = ttnn.to_torch(tt_output, mesh_composer=composer, dtype=torch.bfloat16)
    logger.debug(f"{tt_host.shape=}")
    threshold = 0.999
    _, pcc = comp_pcc(torch_shards.float(), tt_host.float())

    logger.debug(f"TTNN reduce operation matches torch reference! (PCC={pcc:.6f})")
    assert pcc > threshold, f"PCC {pcc:.6f} below threshold {threshold}"
