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
from models.demos.deepseek_v3_d_p.reference.kimi_k2_6_config import KimiK26Config
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
    (5120, "5k"),
    (25600, "25k"),
]


def run_single_routed_expert(
    mesh_device,
    device_params,
    num_tokens: int,
    emb_dim: int,
    hidden_dim: int,
    x_row_major: bool = False,
):
    """
    Simplest scenario: 1 chip, 1 expert. Shared body for the per-model entrypoints below — they
    differ only on the (emb_dim, hidden_dim) shape axis and the x input layout.

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
    # The composite op branches on x layout: ROW_MAJOR is bf16 (tilized and bf8-packed
    # inside the op), TILE is consumed directly as bf8. Pair the dtype with the layout so
    # each variation drives its real device path.
    tt_input = ttnn.from_torch(
        torch_input,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        layout=ttnn.ROW_MAJOR_LAYOUT if x_row_major else ttnn.TILE_LAYOUT,
        device=mesh_device,
        dtype=ttnn.bfloat16 if x_row_major else ttnn.bfloat8_b,
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
        activation=ttnn.RoutedExpertActivation.Silu,
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


# Per-model dims as (id_prefix, config, extended_model), each run at its own (emb_dim,
# MOE_INTERMEDIATE_SIZE). DeepSeek V3 is the baseline and runs by default; every other model is
# gated behind @pytest.mark.extended_model.
SINGLE_EXPERT_MODELS = [
    ("dsv3", DeepSeekV3Config, False),
    ("minimax_m27", MiniMaxM27Config, True),
    ("glm_51", GLM51Config, True),
    ("dsv4_pro", DeepSeekV4ProConfig, True),
    ("dsv4_flash", DeepSeekV4FlashConfig, True),
    ("gptoss_120b", GptOss120BConfig, True),
    ("kimi_k26", KimiK26Config, True),
]


# Registry of currently-failing single-routed-expert cases, strict-xfail'd on blackhole by
# _xfail_blackhole_cases so CI stays green while linked issues are worked on. Each entry is
# (id_substrings, reason): a case is marked when every substring is present in its param id, which is
# how a case gets pinned to a specific layout (the id carries "-x_rm-"/"-x_tile-") and/or token count.
# Blackhole-only: these program-creation asserts depend on the BH grid (GRID_X=11), so the same dims
# pass on other arches where an unconditional strict xfail would flip CI red on XPASS.
_BLACKHOLE_XFAIL = [
    # gpt-oss emb=2880 -> K_gate_tiles=90. The factory's adaptive L1 guard narrows in0_block_w_gu to a
    # divisor of 90 for the RM path (bf16 x-staging always inflates the CB footprint past the guard)
    # and for the TILE path at 1k/25k, but not at the 5k footprint, where it stays 16 and 90 % 16 != 0
    # -> TT_FATAL. So only the TILE path at 5120-token alloc fails (dense -5k plus faked -5k-alloc-*).
    (
        ("-x_tile-", "gptoss_120b-5k"),
        "gpt-oss K_gate_tiles=90 not divisible by in0_block_w_gu=16 (TILE path, 5k alloc)",
    ),
]


def single_routed_expert_token_sweep_params():
    """Build the per-model (num_tokens, emb_dim, hidden_dim) parametrization over _TOKEN_SWEEP.
    Non-baseline models carry the extended_model marker so they stay gated as before; currently-failing
    cases are xfail'd per-arch by _xfail_blackhole_cases, not here."""
    params = []
    for name, config, extended in SINGLE_EXPERT_MODELS:
        for num_tokens, tag in _TOKEN_SWEEP:
            test_id = f"{name}-{tag}"
            marks = (pytest.mark.extended_model,) if extended else ()
            params.append(
                pytest.param(num_tokens, config.EMB_SIZE, config.MOE_INTERMEDIATE_SIZE, marks=marks, id=test_id)
            )
    return params


@pytest.fixture(autouse=True)
def _xfail_blackhole_cases(request, silicon_arch_name):
    """Strict-xfail the _BLACKHOLE_XFAIL cases (both the dense and faked-count tests). An entry fires
    when all of its id substrings appear in the case's param id, so a single list drives layout- and
    token-count-specific marks without a per-case predicate. Blackhole-only, per the list's contract."""
    if silicon_arch_name != "blackhole":
        return
    callspec = getattr(request.node, "callspec", None)
    if callspec is None:
        return
    for substrings, reason in _BLACKHOLE_XFAIL:
        if all(s in callspec.id for s in substrings):
            request.applymarker(pytest.mark.xfail(reason=reason, strict=True))
            break


@pytest.mark.parametrize("num_tokens, emb_dim, hidden_dim", single_routed_expert_token_sweep_params())
@pytest.mark.parametrize(
    "mesh_device, device_params", SINGLE_CHIP_MESH_PARAMS, indirect=["mesh_device", "device_params"]
)
@pytest.mark.parametrize("x_row_major", [True, False], ids=["x_rm", "x_tile"])
def test_single_routed_expert_models(
    mesh_device, device_params, num_tokens: int, emb_dim: int, hidden_dim: int, x_row_major: bool
):
    run_single_routed_expert(mesh_device, device_params, num_tokens, emb_dim, hidden_dim, x_row_major)


# (allocated_tokens, active_tokens, id) sweep for the count-aware sparsity test, applied per
# model with that model's (emb_dim, hidden_dim). The alloc/active pairs are model-independent.
_FAKED_SWEEP = [
    (1024, 0, "1k-alloc-0k-active"),
    (5120, 161, "5k-alloc-161-active"),
    (5120, 256, "5k-alloc-256-active"),
    (25600, 4096, "25k-alloc-4k-active"),
]


def run_single_routed_expert_faked_token_count(
    mesh_device,
    device_params,
    allocated_tokens: int,
    active_tokens: int,
    emb_dim: int,
    hidden_dim: int,
    x_row_major: bool = False,
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

    # ROW_MAJOR is bf16 (tilized + bf8-packed in-op), TILE is consumed directly as bf8.
    # With active_tokens < allocated_tokens the ROW_MAJOR variant is what exercises the
    # reader's clamp-read of the inactive rows past the runtime count.
    tt_input = ttnn.from_torch(
        torch_input,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        layout=ttnn.ROW_MAJOR_LAYOUT if x_row_major else ttnn.TILE_LAYOUT,
        device=mesh_device,
        dtype=ttnn.bfloat16 if x_row_major else ttnn.bfloat8_b,
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
        activation=ttnn.RoutedExpertActivation.Silu,
    )

    tt_output = tt_expert(tt_input, expert_token_counts_tt, expert_region_offsets_tt)

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


def single_routed_expert_faked_params():
    """Build the per-model (allocated_tokens, active_tokens, emb_dim, hidden_dim) parametrization
    over _FAKED_SWEEP. Reuses SINGLE_EXPERT_MODELS, so non-baseline models stay gated behind the
    extended_model marker exactly as the separate tests were."""
    params = []
    for name, config, extended in SINGLE_EXPERT_MODELS:
        for alloc, active, tag in _FAKED_SWEEP:
            test_id = f"{name}-{tag}"
            marks = (pytest.mark.extended_model,) if extended else ()
            params.append(
                pytest.param(
                    alloc,
                    active,
                    config.EMB_SIZE,
                    config.MOE_INTERMEDIATE_SIZE,
                    marks=marks,
                    id=test_id,
                )
            )
    return params


@pytest.mark.parametrize("allocated_tokens, active_tokens, emb_dim, hidden_dim", single_routed_expert_faked_params())
@pytest.mark.parametrize(
    "mesh_device, device_params", SINGLE_CHIP_MESH_PARAMS, indirect=["mesh_device", "device_params"]
)
@pytest.mark.parametrize("x_row_major", [True, False], ids=["x_rm", "x_tile"])
@pytest.mark.skipif(not is_blackhole(), reason="device-side count-aware sparsity is Blackhole-only")
def test_single_routed_expert_faked_token_count_models(
    mesh_device,
    device_params,
    allocated_tokens: int,
    active_tokens: int,
    emb_dim: int,
    hidden_dim: int,
    x_row_major: bool,
):
    run_single_routed_expert_faked_token_count(
        mesh_device, device_params, allocated_tokens, active_tokens, emb_dim, hidden_dim, x_row_major
    )
