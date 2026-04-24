# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""
Minimal single-device, single-expert test for TtRoutedExpert profiling.

The simplest scenario: 1 chip, 1 expert, minimal dimensions.
"""

import pytest
import torch
from loguru import logger
from tracy import signpost

import ttnn
from models.demos.deepseek_v3_d_p.reference.tt.moe.expert import TorchExpert
from models.demos.deepseek_v3_d_p.tt.moe.tt_routed_expert import TtRoutedExpert
from tests.ttnn.utils_for_testing import comp_pcc


@pytest.mark.parametrize(
    "use_routed_matmul",
    [
        pytest.param(False, id="stock-matmul"),
        pytest.param(True, id="routed-matmul"),
    ],
)
@pytest.mark.parametrize(
    "num_tokens, emb_dim, hidden_dim",
    [
        (1024, 7168, 2048),  # DeepSeek V3 dims, 1K tokens
        (2048, 7168, 2048),  # DeepSeek V3 dims, 2K tokens
        (4096, 7168, 2048),  # DeepSeek V3 dims, 4K tokens
        (8192, 7168, 2048),  # DeepSeek V3 dims, 8K tokens
        # 25K-ish — rounded down to the nearest multiple of MAX_EXPERT_LENGTH (2048)
        # so the chunk loop's last iteration stays tile-aligned. Plain 25000 is not
        # a multiple of 32 and ttnn.narrow rejects the ragged tail.
        (24576, 7168, 2048),  # DeepSeek V3 dims, 24K tokens (12 × 2048)
    ],
    ids=["ds-v3-1k", "ds-v3-2k", "ds-v3-4k", "ds-v3-8k", "ds-v3-24k"],
)
@pytest.mark.parametrize(
    "mesh_device, device_params",
    [
        pytest.param(
            1,
            {"fabric_config": ttnn.FabricConfig.DISABLED},
            id="single-chip",
        ),
    ],
    indirect=["mesh_device", "device_params"],
)
def test_single_routed_expert(
    mesh_device,
    device_params,
    num_tokens: int,
    emb_dim: int,
    hidden_dim: int,
    use_routed_matmul: bool,
):
    """
    Simplest test: 1 chip, 1 expert.

    Perfect for profiling the core FFN computation without any mesh complexity.
    """
    # Stock matmul only supports up to 4K tokens; 8K and 25K require routed-matmul.
    if not use_routed_matmul and num_tokens > 4096:
        pytest.skip("stock-matmul is only validated up to 4K tokens")

    experts_per_chip = 1

    signpost(f"SingleRoutedExpert {num_tokens=} {emb_dim=} {hidden_dim=}")

    logger.debug(f"Testing single routed expert: {num_tokens=}, {emb_dim=}, {hidden_dim=}")
    logger.debug(f"Mesh: {mesh_device.shape}, num_devices={mesh_device.get_num_devices()}")

    # Create random weights
    torch.manual_seed(42)
    weights = {
        "gate_proj": torch.randn(hidden_dim, emb_dim, dtype=torch.float32) * 0.02,
        "up_proj": torch.randn(hidden_dim, emb_dim, dtype=torch.float32) * 0.02,
        "down_proj": torch.randn(emb_dim, hidden_dim, dtype=torch.float32) * 0.02,
    }

    # Create torch reference
    torch_expert = TorchExpert(emb_dim, hidden_dim, weights)

    # Create random input: (experts_per_chip, num_tokens, emb_dim)
    torch_input = torch.randn(experts_per_chip, num_tokens, emb_dim, dtype=torch.float32)
    logger.debug(f"Input shape: {torch_input.shape}")

    # Run torch reference
    logger.debug("Running torch reference...")
    with torch.no_grad():
        torch_output = torch_expert(torch_input[0])  # Process first (only) expert's tokens
    logger.debug(f"Torch output shape: {torch_output.shape}")

    # Create TTNN input
    tt_input = ttnn.from_torch(
        torch_input,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        dtype=ttnn.bfloat8_b,
    )
    logger.debug(f"TTNN input shape: {tt_input.shape}")

    # Build fake guard tensors to exercise the routed-matmul kernel path.
    # ROW_MAJOR_LAYOUT uint32 chosen for its intent: metadata tables indexed by
    # small integer ids, not tile-aligned data. Note: the current kernel guard
    # does a naive single-bank NoC read, which for DRAM-INTERLEAVED buffers only
    # returns meaningful data for the element(s) that happen to land in bank 0.
    # As a stub we fill the whole table uniformly so any byte offset the kernel
    # reads lands on the correct value. Upgrading the guard to InterleavedAddrGen
    # for per-index reads is tracked by the TODO in guard.h.
    global_expert_idx_table = None
    expert_token_counts_tt = None
    if use_routed_matmul:
        pad = 32  # fill size — must be >= one DRAM transaction (32 uint32s)

        # Identity fill for the table and uniform num_tokens for counts are both
        # correct for any indexing the kernel performs today (since the guard
        # reads a single uint32's worth of data that must equal the expected
        # value regardless of which bank element it reads).
        global_table_torch = torch.zeros((pad,), dtype=torch.int32)  # all locals → global 0
        token_counts_torch = torch.full((pad,), num_tokens, dtype=torch.int32)

        global_expert_idx_table = ttnn.from_torch(
            global_table_torch,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )
        expert_token_counts_tt = ttnn.from_torch(
            token_counts_torch,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )
        ttnn.synchronize_device(mesh_device)
        logger.debug(f"Fake guard tensors created (identity table, counts={num_tokens})")

    # Create TtRoutedExpert
    logger.debug("Creating TtRoutedExpert...")
    tt_expert = TtRoutedExpert(
        mesh_device=mesh_device,
        experts_per_chip=experts_per_chip,
        emb_dim=emb_dim,
        hidden_dim=hidden_dim,
        max_tokens=num_tokens,
        torch_weights=[weights],  # List with single expert weights
        activations_dtype=ttnn.bfloat8_b,
        weights_dtype=ttnn.bfloat4_b,
    )

    # Run TTNN forward
    logger.debug("Running TTNN forward...")
    tt_output = tt_expert(
        tt_input,
        global_expert_idx_table=global_expert_idx_table,
        expert_token_counts=expert_token_counts_tt,
    )
    logger.debug(f"TTNN output shape: {tt_output.shape}")

    # Convert back to torch for comparison
    tt_output_torch = ttnn.to_torch(
        tt_output,
        mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0),
    )
    # Extract the single expert output: (experts_per_chip, num_tokens, emb_dim) -> (num_tokens, emb_dim)
    tt_output_single = tt_output_torch[0]
    logger.debug(f"TTNN output (torch) shape: {tt_output_single.shape}")

    # Compare PCC
    _, pcc = comp_pcc(torch_output, tt_output_single)
    logger.debug(f"PCC: {pcc:.6f}")

    # Validate
    pcc_threshold = 0.97
    assert pcc >= pcc_threshold, f"PCC {pcc:.6f} below threshold {pcc_threshold}"
    assert not torch.isnan(tt_output_torch).any(), "Output contains NaN"
    assert not torch.isinf(tt_output_torch).any(), "Output contains Inf"

    logger.debug("Test PASSED!")
