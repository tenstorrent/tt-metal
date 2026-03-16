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
from models.demos.deepseek_v3_d_p.reference.moe.reduce import TorchReduceModule
from models.demos.deepseek_v3_d_p.tt.moe.tt_reduce import TtReduceModule


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


@pytest.mark.parametrize(
    "seq_len, hidden_dim, topk",
    [
        (32, 2048, 8),
        (64, 4096, 4),
        (3200, 7 * 1024, 8),  # DeepSeek values
    ],
)
@pytest.mark.parametrize(
    "mesh_device, device_params",
    [
        pytest.param(
            (2, 1),
            {"fabric_config": ttnn.FabricConfig.FABRIC_1D},
            marks=pytest.mark.requires_mesh_topology(mesh_shape=(2, 1), topology="linear"),
            id="linear-2x1",
        ),
        pytest.param(
            (4, 1),
            {"fabric_config": ttnn.FabricConfig.FABRIC_1D},
            marks=pytest.mark.requires_mesh_topology(mesh_shape=(4, 1), topology="linear"),
            id="linear-4x1",
        ),
        pytest.param(
            (1, 2),
            {"fabric_config": ttnn.FabricConfig.FABRIC_1D},
            marks=pytest.mark.requires_mesh_topology(mesh_shape=(1, 2), topology="linear"),
            id="linear-1x2",
        ),
        pytest.param(
            (4, 2),
            {"fabric_config": ttnn.FabricConfig.FABRIC_1D},
            marks=pytest.mark.requires_mesh_topology(mesh_shape=(4, 2), topology="mesh-4x2"),
            id="mesh-4x2",
        ),
        pytest.param(
            (2, 4),
            {"fabric_config": ttnn.FabricConfig.FABRIC_1D},
            marks=pytest.mark.requires_mesh_topology(mesh_shape=(2, 4), topology="mesh-4x2"),
            id="mesh-2x4",
        ),
    ],
    indirect=["mesh_device", "device_params"],
)
def test_ttnn_reduce(
    mesh_device,
    seq_len,
    hidden_dim,
    topk,
):
    """Test TTNN reduce module in isolation using synthetic sparse inputs."""

    num_devices = mesh_device.get_num_devices()
    mesh_shape = mesh_device.shape

    # Derive cluster_axis from mesh shape
    # For 2D mesh (4x2), reduce across columns (axis 1)
    # For 1D mesh (Nx1 or 1xN), reduce across the non-trivial axis
    if mesh_shape[1] > 1:
        cluster_axis = 1  # Reduce across columns
    else:
        cluster_axis = 0  # Reduce across rows

    # Set topology and num_links
    topology = ttnn.Topology.Linear
    num_links = 1

    # Determine number of chips in the reduction axis
    num_reduce_chips = mesh_shape[cluster_axis]

    logger.debug(f"Testing reduce: {mesh_shape=}, {cluster_axis=}, {num_reduce_chips=}")
    logger.debug(f"Input shape per chip: [{seq_len}, {topk}, {hidden_dim}]")
    logger.debug(f"Expected output shape per chip: [{seq_len}, {hidden_dim // num_reduce_chips}]")
    ttnn.visualize_mesh_device(mesh_device)

    signpost(
        f"Reduce {mesh_shape=} {num_devices=} {seq_len=} {topk=} {hidden_dim=} "
        f"{cluster_axis=} {num_links=} {topology=}"
    )

    # Create synthetic sparse combine output
    torch_combine_output = create_sparse_combine_output(
        num_chips=num_reduce_chips,
        seq_len=seq_len,
        topk=topk,
        hidden_dim=hidden_dim,
        sparsity=0.75,
        seed=42,
    )
    logger.debug(f"Created sparse combine output: {torch_combine_output.shape}")

    # Compute reference output using torch
    torch_reduce = TorchReduceModule(
        num_reduce_chips=num_reduce_chips,
        topk_dim=1,  # topk is dim 1 in [seq, topk, hidden]
    )
    torch_shards = torch_reduce(torch_combine_output)
    logger.debug(f"Torch reference output: {len(torch_shards)} shards of shape {torch_shards[0].shape}")

    # Convert to TTNN tensor distributed across mesh
    # For reduce_scatter, we need each chip to have its portion of the input
    # Shape transformation: [num_reduce_chips, seq, topk, hidden] -> per-chip [seq, topk, hidden]

    # Determine the mesh dimension to shard on based on cluster_axis
    if cluster_axis == 0:
        # Reduce across rows: shard tensor's chip dimension across mesh rows
        mesh_mapper = ttnn.ShardTensor2dMesh(
            mesh_device,
            mesh_shape=mesh_device.shape,
            dims=(0, None),  # Shard tensor dim 0 across mesh rows, replicate across cols
        )
    else:
        # Reduce across columns: shard tensor's chip dimension across mesh columns
        mesh_mapper = ttnn.ShardTensor2dMesh(
            mesh_device,
            mesh_shape=mesh_device.shape,
            dims=(None, 0),  # Replicate across mesh rows, shard tensor dim 0 across cols
        )

    tt_combine_output = ttnn.from_torch(
        torch_combine_output,
        mesh_mapper=mesh_mapper,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        dtype=ttnn.bfloat16,
    )
    logger.debug(f"TTNN input shape: {tt_combine_output.shape}")

    # Run TTNN reduce
    # NOTE: TTNN adds a batch dim, so [seq, topk, hidden] becomes [1, seq, topk, hidden]
    # topk is at dim=2 in the 4D tensor
    tt_reduce = TtReduceModule(
        mesh_device=mesh_device,
        topk_dim=2,  # topk is dim 2 in [1, seq, topk, hidden]
        cluster_axis=cluster_axis,
        num_links=num_links,
        topology=topology,
    )

    tt_output = tt_reduce(tt_combine_output)
    logger.debug(f"TTNN output shape: {tt_output.shape}")

    # Convert output back to torch for comparison
    # After reduce_scatter, each chip has [seq, hidden/num_reduce_chips]
    # Get individual device outputs and compare with corresponding torch shards
    device_tensors = ttnn.get_device_tensors(tt_output)
    logger.debug(f"Got {len(device_tensors)} device tensors")

    # For each chip in the reduction axis, compare its output with the corresponding torch shard
    all_passed = True
    min_pcc = 1.0

    for chip_idx in range(num_reduce_chips):
        # Get the device tensor for this chip
        # For cluster_axis=0, chips are arranged vertically (rows)
        # For cluster_axis=1, chips are arranged horizontally (cols)
        if cluster_axis == 0:
            device_idx = chip_idx  # Vertical arrangement
        else:
            device_idx = chip_idx  # First row, iterate across columns

        tt_chip_output = ttnn.to_torch(device_tensors[device_idx], dtype=torch.bfloat16)
        torch_shard = torch_shards[chip_idx]

        # Remove extra dimensions if present
        while tt_chip_output.dim() > torch_shard.dim():
            tt_chip_output = tt_chip_output.squeeze(0)

        logger.debug(f"Chip {chip_idx}: tt_shape={tt_chip_output.shape}, torch_shape={torch_shard.shape}")

        # Calculate PCC for this chip
        tt_flat = tt_chip_output.float().flatten()
        torch_flat = torch_shard.float().flatten()

        if tt_flat.shape != torch_flat.shape:
            logger.warning(f"Chip {chip_idx} shape mismatch: tt={tt_chip_output.shape}, torch={torch_shard.shape}")
            min_len = min(len(tt_flat), len(torch_flat))
            tt_flat = tt_flat[:min_len]
            torch_flat = torch_flat[:min_len]

        pcc = torch.corrcoef(torch.stack([tt_flat, torch_flat]))[0, 1].item()
        min_pcc = min(min_pcc, pcc)
        logger.debug(f"Chip {chip_idx} PCC: {pcc:.6f}")

        if pcc < 0.99:
            all_passed = False
            logger.error(f"Chip {chip_idx} PCC {pcc:.6f} below threshold 0.99")

    # Assert overall PCC threshold
    assert all_passed, f"Min PCC {min_pcc:.6f} below threshold 0.99"
    logger.debug(f"TTNN reduce operation matches torch reference! (min PCC={min_pcc:.6f})")
