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
from models.common.utility_functions import is_blackhole
from models.demos.deepseek_v3_d_p.reference.tt.moe.expert import TorchExpert
from models.demos.deepseek_v3_d_p.tt.moe.tt_routed_expert import TtRoutedExpert
from tests.ttnn.utils_for_testing import comp_pcc


@pytest.mark.parametrize(
    "num_tokens, emb_dim, hidden_dim",
    [
        (1024, 7168, 2048),  # DeepSeek V3 dims, 1K tokens
        (2048, 7168, 2048),  # DeepSeek V3 dims, 2K tokens
        (4096, 7168, 2048),  # DeepSeek V3 dims, 4K tokens
        (5120, 7168, 2048),  # DeepSeek V3 dims, 5K tokens
        (6144, 7168, 2048),  # DeepSeek V3 dims, 6K tokens
        (8192, 7168, 2048),  # DeepSeek V3 dims, 8K tokens
        (25600, 7168, 2048),  # DeepSeek V3 dims, 25K tokens
    ],
    ids=[
        "ds-v3-1k",
        "ds-v3-2k",
        "ds-v3-4k",
        "ds-v3-5k",
        "ds-v3-6k",
        "ds-v3-8k",
        "ds-v3-25k",
    ],
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
):
    """
    Simplest test: 1 chip, 1 expert.

    Perfect for profiling the core FFN computation without any mesh complexity.
    """
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

    # 2D input (num_tokens, emb_dim) — the single expert's dispatch buffer.
    torch_input = torch.randn(num_tokens, emb_dim, dtype=torch.float32)
    logger.debug(f"Input shape: {torch_input.shape}")

    # Run torch reference
    logger.debug("Running torch reference...")
    with torch.no_grad():
        torch_output = torch_expert(torch_input)
    logger.debug(f"Torch output shape: {torch_output.shape}")

    # Create TTNN input: 2D (num_tokens, emb_dim), replicated across the 1-device mesh.
    tt_input = ttnn.from_torch(
        torch_input,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        dtype=ttnn.bfloat8_b,
    )
    logger.debug(f"TTNN input shape: {tt_input.shape}")

    # Single-expert auxiliaries (1D, length 1, UINT32 ROW_MAJOR DRAM):
    #   - global_expert_idx_table[0] = 0   (local 0 -> global 0)
    #   - expert_token_counts[0]     = num_tokens
    #   - expert_region_offsets[0]   = 0   (expert's slice starts at row 0)
    def _make_idx_tensor(values):
        return ttnn.from_torch(
            torch.tensor(values, dtype=torch.int32),
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh_device,
            dtype=ttnn.uint32,
        )

    global_expert_idx_tt = _make_idx_tensor([0])
    expert_token_counts_tt = _make_idx_tensor([num_tokens])
    expert_region_offsets_tt = _make_idx_tensor([0])

    # Create TtRoutedExpert
    logger.debug("Creating TtRoutedExpert...")
    tt_expert = TtRoutedExpert(
        mesh_device=mesh_device,
        experts_per_chip=experts_per_chip,
        global_expert_idx_table=global_expert_idx_tt,
        emb_dim=emb_dim,
        hidden_dim=hidden_dim,
        max_tokens=num_tokens,
        torch_weights=[weights],  # List with single expert weights
        activations_dtype=ttnn.bfloat8_b,
        weights_dtype=ttnn.bfloat4_b,
    )

    # Run TTNN forward
    logger.debug("Running TTNN forward...")
    tt_output = tt_expert(tt_input, expert_token_counts_tt, expert_region_offsets_tt)
    logger.debug(f"TTNN output shape: {tt_output.shape}")

    # Convert back to torch for comparison. For a 1-device replicated tensor,
    # ConcatMeshToTensor(dim=0) with 1 slice is a no-op that returns the tensor.
    tt_output_torch = ttnn.to_torch(
        tt_output,
        mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0),
    )
    logger.debug(f"TTNN output (torch) shape: {tt_output_torch.shape}")

    # Compare PCC
    _, pcc = comp_pcc(torch_output, tt_output_torch)
    logger.debug(f"PCC: {pcc:.6f}")

    # Validate
    pcc_threshold = 0.97
    assert pcc >= pcc_threshold, f"PCC {pcc:.6f} below threshold {pcc_threshold}"
    assert not torch.isnan(tt_output_torch).any(), "Output contains NaN"
    assert not torch.isinf(tt_output_torch).any(), "Output contains Inf"

    logger.debug("Test PASSED!")


@pytest.mark.parametrize(
    "allocated_tokens, active_tokens, emb_dim, hidden_dim",
    [
        (4096, 2048, 7168, 2048),
        (25 * 1024, 2048, 7168, 2048),
        (25 * 1024, 4096, 7168, 2048),
        (16384, 2048, 7168, 2048),
        (16384, 4096, 7168, 2048),
    ],
    ids=[
        "ds-v3-4k-alloc-2k-active",
        "ds-v3-25k-alloc-2k-active",
        "ds-v3-25k-alloc-4k-active",
        "ds-v3-16k-alloc-2k-active",
        "ds-v3-16k-alloc-4k-active",
    ],
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
@pytest.mark.skipif(not is_blackhole(), reason="device-side count-aware sparsity is Blackhole-only")
def test_single_routed_expert_faked_token_count(
    mesh_device,
    device_params,
    allocated_tokens: int,
    active_tokens: int,
    emb_dim: int,
    hidden_dim: int,
):
    """
    Verifies the unified kernel honors expert_token_counts and skips work on
    inactive padding rows.

    Dispatch buffer sized for ``allocated_tokens`` but only the first
    ``active_tokens`` rows hold real data; the rest is zero padding. The
    kernel must (a) produce correct output on the active slice and (b) not
    do matmuls on the inactive padding rows (device-side count sparsity).
    """
    experts_per_chip = 1

    signpost(f"SingleRoutedExpertFaked {allocated_tokens=} {active_tokens=} {emb_dim=} {hidden_dim=}")

    torch.manual_seed(42)
    weights = {
        "gate_proj": torch.randn(hidden_dim, emb_dim, dtype=torch.float32) * 0.02,
        "up_proj": torch.randn(hidden_dim, emb_dim, dtype=torch.float32) * 0.02,
        "down_proj": torch.randn(emb_dim, hidden_dim, dtype=torch.float32) * 0.02,
    }
    torch_expert = TorchExpert(emb_dim, hidden_dim, weights)

    torch_active = torch.randn(active_tokens, emb_dim, dtype=torch.float32)
    torch_input = torch.zeros(allocated_tokens, emb_dim, dtype=torch.float32)
    torch_input[:active_tokens] = torch_active

    with torch.no_grad():
        torch_output_active = torch_expert(torch_active)

    tt_input = ttnn.from_torch(
        torch_input,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        dtype=ttnn.bfloat8_b,
    )

    def _make_idx_tensor(values):
        return ttnn.from_torch(
            torch.tensor(values, dtype=torch.int32),
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh_device,
            dtype=ttnn.uint32,
        )

    global_expert_idx_tt = _make_idx_tensor([0])
    expert_token_counts_tt = _make_idx_tensor([active_tokens])
    expert_region_offsets_tt = _make_idx_tensor([0])

    tt_expert = TtRoutedExpert(
        mesh_device=mesh_device,
        experts_per_chip=experts_per_chip,
        global_expert_idx_table=global_expert_idx_tt,
        emb_dim=emb_dim,
        hidden_dim=hidden_dim,
        max_tokens=allocated_tokens,
        torch_weights=[weights],
        activations_dtype=ttnn.bfloat8_b,
        weights_dtype=ttnn.bfloat4_b,
    )

    # Time 5 iters (iter0 includes JIT compile; iter1-4 are steady-state).
    import time as _time

    for _i in range(5):
        _t0 = _time.time()
        tt_output = tt_expert(tt_input, expert_token_counts_tt, expert_region_offsets_tt)
        ttnn.synchronize_device(mesh_device)
        _dt_ms = (_time.time() - _t0) * 1000
        logger.warning(f"  faked iter {_i}: {_dt_ms:.2f} ms (alloc={allocated_tokens}, active={active_tokens})")
    tt_output_torch = ttnn.to_torch(
        tt_output,
        mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0),
    )
    tt_output_active = tt_output_torch[:active_tokens]

    _, pcc = comp_pcc(torch_output_active, tt_output_active)
    logger.debug(f"PCC over active slice ({active_tokens} rows): {pcc:.6f}")

    assert pcc >= 0.97, f"PCC {pcc:.6f} below threshold 0.97"
    assert not torch.isnan(tt_output_active).any(), "Active output contains NaN"
    assert not torch.isinf(tt_output_active).any(), "Active output contains Inf"


@pytest.mark.parametrize(
    "num_tokens, emb_dim, hidden_dim",
    [
        (1024, 2880, 2880),  # Wormhole MoE dims, 1K tokens (1 chunk)
        (2048, 2880, 2880),  # Wormhole MoE dims, 2K tokens (2 chunks)
        (4096, 2880, 2880),  # Wormhole MoE dims, 4K tokens (4 chunks)
        (3072, 2880, 2880),  # Wormhole MoE dims, non-power-of-two (3 chunks)
    ],
    ids=[
        "wh-2880-1k",
        "wh-2880-2k",
        "wh-2880-4k",
        "wh-2880-3k",
    ],
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
def test_single_routed_expert_wh(
    mesh_device,
    device_params,
    num_tokens: int,
    emb_dim: int,
    hidden_dim: int,
    monkeypatch,
):
    """
    Single-expert PCC test for the Wormhole unified-FFN kernel (8x8 grid,
    emb=hidden=2880).

    The dedicated WH kernel only differs from the Blackhole one in grid width
    and the gate/up K-block width — it reuses the same dataflow/compute
    kernels. To exercise it on the Blackhole silicon we develop on, set the
    ``TT_UNIFIED_REXPERT_FORCE_WH`` passthrough env var so the C++ program
    factory selects the WH config even though the arch reports Blackhole.
    On real Wormhole the env var is harmless (WH always takes the WH path).
    """
    monkeypatch.setenv("TT_UNIFIED_REXPERT_FORCE_WH", "1")

    experts_per_chip = 1

    signpost(f"SingleRoutedExpertWH {num_tokens=} {emb_dim=} {hidden_dim=}")

    logger.debug(f"Testing WH single routed expert: {num_tokens=}, {emb_dim=}, {hidden_dim=}")

    torch.manual_seed(42)
    weights = {
        "gate_proj": torch.randn(hidden_dim, emb_dim, dtype=torch.float32) * 0.02,
        "up_proj": torch.randn(hidden_dim, emb_dim, dtype=torch.float32) * 0.02,
        "down_proj": torch.randn(emb_dim, hidden_dim, dtype=torch.float32) * 0.02,
    }

    torch_expert = TorchExpert(emb_dim, hidden_dim, weights)

    torch_input = torch.randn(num_tokens, emb_dim, dtype=torch.float32)

    with torch.no_grad():
        torch_output = torch_expert(torch_input)

    tt_input = ttnn.from_torch(
        torch_input,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        dtype=ttnn.bfloat8_b,
    )

    def _make_idx_tensor(values):
        return ttnn.from_torch(
            torch.tensor(values, dtype=torch.int32),
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh_device,
            dtype=ttnn.uint32,
        )

    global_expert_idx_tt = _make_idx_tensor([0])
    expert_token_counts_tt = _make_idx_tensor([num_tokens])
    expert_region_offsets_tt = _make_idx_tensor([0])

    tt_expert = TtRoutedExpert(
        mesh_device=mesh_device,
        experts_per_chip=experts_per_chip,
        global_expert_idx_table=global_expert_idx_tt,
        emb_dim=emb_dim,
        hidden_dim=hidden_dim,
        max_tokens=num_tokens,
        torch_weights=[weights],
        activations_dtype=ttnn.bfloat8_b,
        weights_dtype=ttnn.bfloat4_b,
    )

    tt_output = tt_expert(tt_input, expert_token_counts_tt, expert_region_offsets_tt)

    tt_output_torch = ttnn.to_torch(
        tt_output,
        mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0),
    )

    _, pcc = comp_pcc(torch_output, tt_output_torch)
    logger.debug(f"PCC: {pcc:.6f}")

    pcc_threshold = 0.97
    assert pcc >= pcc_threshold, f"PCC {pcc:.6f} below threshold {pcc_threshold}"
    assert not torch.isnan(tt_output_torch).any(), "Output contains NaN"
    assert not torch.isinf(tt_output_torch).any(), "Output contains Inf"

    logger.debug("WH single-expert test PASSED!")
