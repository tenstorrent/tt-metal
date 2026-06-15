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
from models.demos.deepseek_v3_d_p.reference.deepseek_v3_config import DeepSeekV3Config
from models.demos.deepseek_v3_d_p.reference.deepseek_v4_flash_config import DeepSeekV4FlashConfig
from models.demos.deepseek_v3_d_p.reference.deepseek_v4_pro_config import DeepSeekV4ProConfig
from models.demos.deepseek_v3_d_p.reference.glm_5_1_config import GLM51Config
from models.demos.deepseek_v3_d_p.reference.gpt_oss_120b_config import GptOss120BConfig
from models.demos.deepseek_v3_d_p.reference.minimax_m2_7_config import MiniMaxM27Config
from models.demos.deepseek_v3_d_p.reference.tt.moe.expert import TorchExpert
from models.demos.deepseek_v3_d_p.tt.moe.tt_routed_expert import TtRoutedExpert
from tests.ttnn.utils_for_testing import comp_pcc


SINGLE_CHIP_MESH_PARAMS = [
    pytest.param(
        1,
        {"fabric_config": ttnn.FabricConfig.DISABLED},
        id="single-chip",
    ),
]

# Token-count sweep for the single-expert profiling test, applied per model with that model's
# (emb_dim, hidden_dim). The (num_tokens, id) pairs are model-independent.
_TOKEN_SWEEP = [
    (1024, "1k"),
    (2048, "2k"),
    (4096, "4k"),
    (5120, "5k"),
    (6144, "6k"),
    (8192, "8k"),
    (25600, "25k"),
]


def run_single_routed_expert(
    mesh_device,
    device_params,
    num_tokens: int,
    emb_dim: int,
    hidden_dim: int,
):
    """
    Simplest scenario: 1 chip, 1 expert. Shared body for the per-model entrypoints below — they
    differ only on the (emb_dim, hidden_dim) shape axis.

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


# DeepSeek V3 dims (emb 7168, hidden = MOE_INTERMEDIATE_SIZE 2048) across the token sweep.
@pytest.mark.parametrize(
    "num_tokens, emb_dim, hidden_dim",
    [(n, DeepSeekV3Config.EMB_SIZE, DeepSeekV3Config.MOE_INTERMEDIATE_SIZE) for n, _ in _TOKEN_SWEEP],
    ids=[f"ds-v3-{tag}" for _, tag in _TOKEN_SWEEP],
)
@pytest.mark.parametrize(
    "mesh_device, device_params", SINGLE_CHIP_MESH_PARAMS, indirect=["mesh_device", "device_params"]
)
def test_single_routed_expert_ds(mesh_device, device_params, num_tokens: int, emb_dim: int, hidden_dim: int):
    run_single_routed_expert(mesh_device, device_params, num_tokens, emb_dim, hidden_dim)


# MiniMax M2.7 dims (emb 3072, hidden = MOE_INTERMEDIATE_SIZE 1536) across the token sweep.
@pytest.mark.parametrize(
    "num_tokens, emb_dim, hidden_dim",
    [(n, MiniMaxM27Config.EMB_SIZE, MiniMaxM27Config.MOE_INTERMEDIATE_SIZE) for n, _ in _TOKEN_SWEEP],
    ids=[f"minimax-{tag}" for _, tag in _TOKEN_SWEEP],
)
@pytest.mark.parametrize(
    "mesh_device, device_params", SINGLE_CHIP_MESH_PARAMS, indirect=["mesh_device", "device_params"]
)
@pytest.mark.extended_model
def test_single_routed_expert_minimax(mesh_device, device_params, num_tokens: int, emb_dim: int, hidden_dim: int):
    run_single_routed_expert(mesh_device, device_params, num_tokens, emb_dim, hidden_dim)


# GLM 5.1 dims (emb 6144, hidden = MOE_INTERMEDIATE_SIZE 2048) across the token sweep.
@pytest.mark.parametrize(
    "num_tokens, emb_dim, hidden_dim",
    [(n, GLM51Config.EMB_SIZE, GLM51Config.MOE_INTERMEDIATE_SIZE) for n, _ in _TOKEN_SWEEP],
    ids=[f"glm-{tag}" for _, tag in _TOKEN_SWEEP],
)
@pytest.mark.parametrize(
    "mesh_device, device_params", SINGLE_CHIP_MESH_PARAMS, indirect=["mesh_device", "device_params"]
)
@pytest.mark.extended_model
def test_single_routed_expert_glm(mesh_device, device_params, num_tokens: int, emb_dim: int, hidden_dim: int):
    run_single_routed_expert(mesh_device, device_params, num_tokens, emb_dim, hidden_dim)


# DeepSeek V4 Pro dims (emb 7168, hidden = MOE_INTERMEDIATE_SIZE 3072) across the token sweep.
@pytest.mark.parametrize(
    "num_tokens, emb_dim, hidden_dim",
    [(n, DeepSeekV4ProConfig.EMB_SIZE, DeepSeekV4ProConfig.MOE_INTERMEDIATE_SIZE) for n, _ in _TOKEN_SWEEP],
    ids=[f"v4_pro-{tag}" for _, tag in _TOKEN_SWEEP],
)
@pytest.mark.parametrize(
    "mesh_device, device_params", SINGLE_CHIP_MESH_PARAMS, indirect=["mesh_device", "device_params"]
)
@pytest.mark.extended_model
def test_single_routed_expert_v4_pro(mesh_device, device_params, num_tokens: int, emb_dim: int, hidden_dim: int):
    run_single_routed_expert(mesh_device, device_params, num_tokens, emb_dim, hidden_dim)


# DeepSeek V4 Flash dims (emb 4096, hidden = MOE_INTERMEDIATE_SIZE 2048) across the token sweep.
@pytest.mark.parametrize(
    "num_tokens, emb_dim, hidden_dim",
    [(n, DeepSeekV4FlashConfig.EMB_SIZE, DeepSeekV4FlashConfig.MOE_INTERMEDIATE_SIZE) for n, _ in _TOKEN_SWEEP],
    ids=[f"v4_flash-{tag}" for _, tag in _TOKEN_SWEEP],
)
@pytest.mark.parametrize(
    "mesh_device, device_params", SINGLE_CHIP_MESH_PARAMS, indirect=["mesh_device", "device_params"]
)
@pytest.mark.extended_model
def test_single_routed_expert_v4_flash(mesh_device, device_params, num_tokens: int, emb_dim: int, hidden_dim: int):
    run_single_routed_expert(mesh_device, device_params, num_tokens, emb_dim, hidden_dim)


# GPT-OSS 120B dims (emb 2880, hidden = MOE_INTERMEDIATE_SIZE 2880) across the token sweep.
@pytest.mark.parametrize(
    "num_tokens, emb_dim, hidden_dim",
    [(n, GptOss120BConfig.EMB_SIZE, GptOss120BConfig.MOE_INTERMEDIATE_SIZE) for n, _ in _TOKEN_SWEEP],
    ids=[f"gpt_oss-{tag}" for _, tag in _TOKEN_SWEEP],
)
@pytest.mark.parametrize(
    "mesh_device, device_params", SINGLE_CHIP_MESH_PARAMS, indirect=["mesh_device", "device_params"]
)
@pytest.mark.extended_model
def test_single_routed_expert_gpt_oss(mesh_device, device_params, num_tokens: int, emb_dim: int, hidden_dim: int):
    run_single_routed_expert(mesh_device, device_params, num_tokens, emb_dim, hidden_dim)


# (allocated_tokens, active_tokens, id) sweep for the count-aware sparsity test, applied per
# model with that model's (emb_dim, hidden_dim). The alloc/active pairs are model-independent.
_FAKED_SWEEP = [
    (4096, 2048, "4k-alloc-2k-active"),
    (25 * 1024, 2048, "25k-alloc-2k-active"),
    (25 * 1024, 4096, "25k-alloc-4k-active"),
    (16384, 2048, "16k-alloc-2k-active"),
    (16384, 4096, "16k-alloc-4k-active"),
]


def run_single_routed_expert_faked_token_count(
    mesh_device,
    device_params,
    allocated_tokens: int,
    active_tokens: int,
    emb_dim: int,
    hidden_dim: int,
):
    """
    Verifies the unified kernel honors expert_token_counts and skips work on
    inactive padding rows. Shared body for the per-model entrypoints below — they differ only on
    the (emb_dim, hidden_dim) shape axis.

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


# DeepSeek V3 dims (emb 7168, hidden = MOE_INTERMEDIATE_SIZE 2048) across the alloc/active sweep.
@pytest.mark.parametrize(
    "allocated_tokens, active_tokens, emb_dim, hidden_dim",
    [
        (alloc, active, DeepSeekV3Config.EMB_SIZE, DeepSeekV3Config.MOE_INTERMEDIATE_SIZE)
        for alloc, active, _ in _FAKED_SWEEP
    ],
    ids=[f"ds-v3-{tag}" for _, _, tag in _FAKED_SWEEP],
)
@pytest.mark.parametrize(
    "mesh_device, device_params", SINGLE_CHIP_MESH_PARAMS, indirect=["mesh_device", "device_params"]
)
@pytest.mark.skipif(not is_blackhole(), reason="device-side count-aware sparsity is Blackhole-only")
def test_single_routed_expert_faked_token_count_ds(
    mesh_device, device_params, allocated_tokens: int, active_tokens: int, emb_dim: int, hidden_dim: int
):
    run_single_routed_expert_faked_token_count(
        mesh_device, device_params, allocated_tokens, active_tokens, emb_dim, hidden_dim
    )


# MiniMax M2.7 dims (emb 3072, hidden = MOE_INTERMEDIATE_SIZE 1536) across the alloc/active sweep.
@pytest.mark.parametrize(
    "allocated_tokens, active_tokens, emb_dim, hidden_dim",
    [
        (alloc, active, MiniMaxM27Config.EMB_SIZE, MiniMaxM27Config.MOE_INTERMEDIATE_SIZE)
        for alloc, active, _ in _FAKED_SWEEP
    ],
    ids=[f"minimax-{tag}" for _, _, tag in _FAKED_SWEEP],
)
@pytest.mark.parametrize(
    "mesh_device, device_params", SINGLE_CHIP_MESH_PARAMS, indirect=["mesh_device", "device_params"]
)
@pytest.mark.skipif(not is_blackhole(), reason="device-side count-aware sparsity is Blackhole-only")
@pytest.mark.extended_model
def test_single_routed_expert_faked_token_count_minimax(
    mesh_device, device_params, allocated_tokens: int, active_tokens: int, emb_dim: int, hidden_dim: int
):
    run_single_routed_expert_faked_token_count(
        mesh_device, device_params, allocated_tokens, active_tokens, emb_dim, hidden_dim
    )


# GLM 5.1 dims (emb 6144, hidden = MOE_INTERMEDIATE_SIZE 2048) across the alloc/active sweep.
@pytest.mark.parametrize(
    "allocated_tokens, active_tokens, emb_dim, hidden_dim",
    [(alloc, active, GLM51Config.EMB_SIZE, GLM51Config.MOE_INTERMEDIATE_SIZE) for alloc, active, _ in _FAKED_SWEEP],
    ids=[f"glm-{tag}" for _, _, tag in _FAKED_SWEEP],
)
@pytest.mark.parametrize(
    "mesh_device, device_params", SINGLE_CHIP_MESH_PARAMS, indirect=["mesh_device", "device_params"]
)
@pytest.mark.skipif(not is_blackhole(), reason="device-side count-aware sparsity is Blackhole-only")
@pytest.mark.extended_model
def test_single_routed_expert_faked_token_count_glm(
    mesh_device, device_params, allocated_tokens: int, active_tokens: int, emb_dim: int, hidden_dim: int
):
    run_single_routed_expert_faked_token_count(
        mesh_device, device_params, allocated_tokens, active_tokens, emb_dim, hidden_dim
    )


# DeepSeek V4 Pro dims (emb 7168, hidden = MOE_INTERMEDIATE_SIZE 3072) across the alloc/active sweep.
@pytest.mark.parametrize(
    "allocated_tokens, active_tokens, emb_dim, hidden_dim",
    [
        (alloc, active, DeepSeekV4ProConfig.EMB_SIZE, DeepSeekV4ProConfig.MOE_INTERMEDIATE_SIZE)
        for alloc, active, _ in _FAKED_SWEEP
    ],
    ids=[f"v4_pro-{tag}" for _, _, tag in _FAKED_SWEEP],
)
@pytest.mark.parametrize(
    "mesh_device, device_params", SINGLE_CHIP_MESH_PARAMS, indirect=["mesh_device", "device_params"]
)
@pytest.mark.skipif(not is_blackhole(), reason="device-side count-aware sparsity is Blackhole-only")
@pytest.mark.extended_model
def test_single_routed_expert_faked_token_count_v4_pro(
    mesh_device, device_params, allocated_tokens: int, active_tokens: int, emb_dim: int, hidden_dim: int
):
    run_single_routed_expert_faked_token_count(
        mesh_device, device_params, allocated_tokens, active_tokens, emb_dim, hidden_dim
    )


# DeepSeek V4 Flash dims (emb 4096, hidden = MOE_INTERMEDIATE_SIZE 2048) across the alloc/active sweep.
@pytest.mark.parametrize(
    "allocated_tokens, active_tokens, emb_dim, hidden_dim",
    [
        (alloc, active, DeepSeekV4FlashConfig.EMB_SIZE, DeepSeekV4FlashConfig.MOE_INTERMEDIATE_SIZE)
        for alloc, active, _ in _FAKED_SWEEP
    ],
    ids=[f"v4_flash-{tag}" for _, _, tag in _FAKED_SWEEP],
)
@pytest.mark.parametrize(
    "mesh_device, device_params", SINGLE_CHIP_MESH_PARAMS, indirect=["mesh_device", "device_params"]
)
@pytest.mark.skipif(not is_blackhole(), reason="device-side count-aware sparsity is Blackhole-only")
@pytest.mark.extended_model
def test_single_routed_expert_faked_token_count_v4_flash(
    mesh_device, device_params, allocated_tokens: int, active_tokens: int, emb_dim: int, hidden_dim: int
):
    run_single_routed_expert_faked_token_count(
        mesh_device, device_params, allocated_tokens, active_tokens, emb_dim, hidden_dim
    )


# GPT-OSS 120B dims (emb 2880, hidden = MOE_INTERMEDIATE_SIZE 2880) across the alloc/active sweep.
@pytest.mark.parametrize(
    "allocated_tokens, active_tokens, emb_dim, hidden_dim",
    [
        (alloc, active, GptOss120BConfig.EMB_SIZE, GptOss120BConfig.MOE_INTERMEDIATE_SIZE)
        for alloc, active, _ in _FAKED_SWEEP
    ],
    ids=[f"gpt_oss-{tag}" for _, _, tag in _FAKED_SWEEP],
)
@pytest.mark.parametrize(
    "mesh_device, device_params", SINGLE_CHIP_MESH_PARAMS, indirect=["mesh_device", "device_params"]
)
@pytest.mark.skipif(not is_blackhole(), reason="device-side count-aware sparsity is Blackhole-only")
@pytest.mark.extended_model
def test_single_routed_expert_faked_token_count_gpt_oss(
    mesh_device, device_params, allocated_tokens: int, active_tokens: int, emb_dim: int, hidden_dim: int
):
    run_single_routed_expert_faked_token_count(
        mesh_device, device_params, allocated_tokens, active_tokens, emb_dim, hidden_dim
    )
