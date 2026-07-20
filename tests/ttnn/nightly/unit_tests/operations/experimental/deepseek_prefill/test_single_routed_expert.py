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


def run_single_routed_expert(
    mesh_device,
    device_params,
    allocated_tokens: int,
    emb_dim: int,
    hidden_dim: int,
    active_tokens: int = None,
    x_row_major: bool = False,
):
    """
    Simplest scenario: 1 chip, 1 expert. Shared body for the per-model entrypoints below — they
    differ only on the (emb_dim, hidden_dim) shape axis and the x input layout.

    The single expert's dispatch buffer is sized for ``allocated_tokens`` but only the first
    ``active_tokens`` rows hold real data; the rest is zero padding. When ``active_tokens`` is
    None it defaults to ``allocated_tokens`` (a fully-active buffer), which is the plain
    profiling case. With ``active_tokens < allocated_tokens`` this exercises device-side
    count-aware sparsity: the kernel must (a) produce correct output on the active slice and
    (b) not do matmuls on the inactive padding rows. The ROW_MAJOR variant additionally drives
    the reader's clamp-read of the inactive rows past the runtime count.
    """
    if active_tokens is None:
        active_tokens = allocated_tokens
    experts_per_chip = 1

    signpost(f"SingleRoutedExpert {allocated_tokens=} {active_tokens=} {emb_dim=} {hidden_dim=}")

    logger.debug(f"Testing single routed expert: {allocated_tokens=}, {active_tokens=}, {emb_dim=}, {hidden_dim=}")
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

    # 2D input (allocated_tokens, emb_dim) — the single expert's dispatch buffer. The first
    # active_tokens rows hold real data; the rest is zero padding (a no-op when active==allocated).
    torch_active = torch.randn(active_tokens, emb_dim, dtype=torch.float32)
    torch_input = torch.zeros(allocated_tokens, emb_dim, dtype=torch.float32)
    torch_input[:active_tokens] = torch_active
    logger.debug(f"Input shape: {torch_input.shape}")

    # Run torch reference over the active slice only.
    logger.debug("Running torch reference...")
    with torch.no_grad():
        torch_output_active = torch_expert(torch_active)
    logger.debug(f"Torch output shape: {torch_output_active.shape}")

    # Create TTNN input: 2D (allocated_tokens, emb_dim), replicated across the 1-device mesh.
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
    #   - global_expert_idx_table[0] = 0             (local 0 -> global 0)
    #   - expert_token_counts[0]     = active_tokens (runtime count; drives count sparsity)
    #   - expert_region_offsets[0]   = 0             (expert's slice starts at row 0)
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

    # Create TtRoutedExpert
    logger.debug("Creating TtRoutedExpert...")
    tt_expert = TtRoutedExpert(
        mesh_device=mesh_device,
        experts_per_chip=experts_per_chip,
        global_expert_idx_table=global_expert_idx_tt,
        emb_dim=emb_dim,
        hidden_dim=hidden_dim,
        max_tokens=allocated_tokens,
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
    tt_output_active = tt_output_torch[:active_tokens]

    # Compare PCC over the active slice.
    _, pcc = comp_pcc(torch_output_active, tt_output_active)
    logger.debug(f"PCC over active slice ({active_tokens} rows): {pcc:.6f}")

    # Validate
    pcc_threshold = 0.97
    assert pcc >= pcc_threshold, f"PCC {pcc:.6f} below threshold {pcc_threshold}"
    assert not torch.isnan(tt_output_active).any(), "Active output contains NaN"
    assert not torch.isinf(tt_output_active).any(), "Active output contains Inf"

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


# Registry of currently-failing single-routed-expert cases -> xfail reason (with tracking issue), so
# CI stays green while linked issues are worked on. Applied strict, only on blackhole, by
# _xfail_blackhole (these cases pass on other arches, where an unconditional strict xfail would turn
# CI red on XPASS). Each key is a space-separated set of id tokens that must ALL appear in the param
# id, so a case can be scoped by any combination of layout ("x_tile"/"x_rm") and model/isl id.
#   - gptoss_120b on the TILE x layout: the factory picks an in0_block_w_gu that does not divide
#     K_gate_tiles for gptoss dims (K_gate_tiles % in0_block_w_gu == 0 TT_FATAL). The ROW_MAJOR x
#     layout passes, so scope the xfail to x_tile only.
_XFAIL = {
    "x_tile gptoss_120b": "gptoss_120b K_gate tiling: factory picks in0_block_w_gu that does not divide K_gate_tiles on the TILE x layout (unified_routed_expert_ffn_program_factory.cpp:404)",
}


@pytest.fixture(autouse=True)
def _xfail_blackhole(request, silicon_arch_name):
    """Strict-xfail the _XFAIL cases only on blackhole: the K_gate / L1 issues are blackhole-specific
    and these cases pass on other arches, where an unconditional strict xfail would turn CI red on
    XPASS. A case matches when every whitespace-separated token of the key appears in the param id."""
    if silicon_arch_name != "blackhole":
        return
    callspec = getattr(request.node, "callspec", None)
    if callspec is None:
        return
    for key, reason in _XFAIL.items():
        if all(token in callspec.id for token in key.split()):
            request.applymarker(pytest.mark.xfail(reason=reason, strict=True))
            break


# All active-token sweeps run against a fixed 5K allocated buffer: 5120 dispatch rows with only the
# first `active_tokens` holding real data; the rest is zero padding.
_ISL_ALLOCATED_TOKENS = 5120

# Functional sweep: a few active-token counts across every model, on both WH and BH. Two off-tile-
# boundary primes (one below 256, one above 3K) plus a round 1K.
_ISL_FUNCTIONAL_SWEEP = [251, 1024, 3001]

# Exhaustive sweep: the full fill range from empty to fully-packed, restricted to the two largest
# models to keep runtime bounded. Blackhole-only, since active < allocated exercises device-side
# count-aware sparsity (skipping matmuls on the inactive padding rows), which is a Blackhole feature.
_ISL_EXHAUSTIVE_SWEEP = [0, 128, 256, 512, 1024, 2048, 4096, 5120]
_ISL_EXHAUSTIVE_MODELS = ("kimi_k26", "glm_51")


def _isl_params(active_sweep, only_models=None):
    """Build the per-model (allocated_tokens, active_tokens, emb_dim, hidden_dim) parametrization over
    `active_sweep`, all against the fixed _ISL_ALLOCATED_TOKENS buffer. Reuses SINGLE_EXPERT_MODELS so
    non-baseline models stay gated behind the extended_model marker; `only_models` restricts to a
    subset of model names."""
    params = []
    for name, config, extended in SINGLE_EXPERT_MODELS:
        if only_models is not None and name not in only_models:
            continue
        for active in active_sweep:
            marks = (pytest.mark.extended_model,) if extended else ()
            params.append(
                pytest.param(
                    _ISL_ALLOCATED_TOKENS,
                    active,
                    config.EMB_SIZE,
                    config.MOE_INTERMEDIATE_SIZE,
                    marks=marks,
                    id=f"{name}-isl-{active}",
                )
            )
    return params


@pytest.mark.parametrize("allocated_tokens, active_tokens, emb_dim, hidden_dim", _isl_params(_ISL_FUNCTIONAL_SWEEP))
@pytest.mark.parametrize(
    "mesh_device, device_params", SINGLE_CHIP_MESH_PARAMS, indirect=["mesh_device", "device_params"]
)
@pytest.mark.parametrize("x_row_major", [True, False], ids=["x_rm", "x_tile"])
def test_single_routed_expert_functional(
    mesh_device,
    device_params,
    allocated_tokens: int,
    active_tokens: int,
    emb_dim: int,
    hidden_dim: int,
    x_row_major: bool,
):
    run_single_routed_expert(
        mesh_device,
        device_params,
        allocated_tokens,
        emb_dim,
        hidden_dim,
        active_tokens=active_tokens,
        x_row_major=x_row_major,
    )


@pytest.mark.parametrize(
    "allocated_tokens, active_tokens, emb_dim, hidden_dim",
    _isl_params(_ISL_EXHAUSTIVE_SWEEP, only_models=_ISL_EXHAUSTIVE_MODELS),
)
@pytest.mark.parametrize(
    "mesh_device, device_params", SINGLE_CHIP_MESH_PARAMS, indirect=["mesh_device", "device_params"]
)
@pytest.mark.parametrize("x_row_major", [True, False], ids=["x_rm", "x_tile"])
@pytest.mark.skipif(not is_blackhole(), reason="device-side count-aware sparsity is Blackhole-only")
def test_single_routed_expert_isl_sweep(
    mesh_device,
    device_params,
    allocated_tokens: int,
    active_tokens: int,
    emb_dim: int,
    hidden_dim: int,
    x_row_major: bool,
):
    run_single_routed_expert(
        mesh_device,
        device_params,
        allocated_tokens,
        emb_dim,
        hidden_dim,
        active_tokens=active_tokens,
        x_row_major=x_row_major,
    )
