# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""
Minimal single-device, single-expert test for TtRoutedExpert profiling.

The simplest scenario: 1 chip, 1 expert, minimal dimensions.
"""

import pathlib
import re

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
from models.demos.deepseek_v3_d_p.reference.kimi_k3_config import KimiK3Config
from models.demos.deepseek_v3_d_p.reference.minimax_m2_7_config import MiniMaxM27Config
from models.demos.deepseek_v3_d_p.reference.tt.moe.expert import (
    ACTIVATION_CLAMPED_SILU_GLU,
    ACTIVATION_SILU,
    ACTIVATION_SITU,
    CLAMPED_SILU_GLU_LIMIT,
    TorchExpert,
)
from models.demos.deepseek_v3_d_p.tt.moe.tt_routed_expert import TtRoutedExpert
from tests.ttnn.utils_for_testing import comp_pcc
from tests.ttnn.nightly.unit_tests.operations.experimental.deepseek_prefill import ci_pruning


SINGLE_CHIP_MESH_PARAMS = [
    pytest.param(
        1,
        {"fabric_config": ttnn.FabricConfig.DISABLED},
        id="single-chip",
    ),
]

# Device activation -> the TorchExpert reference that must match it. Keeping the pairing
# in one place stops a case from measuring one activation against another's golden.
# SwiGluOai is absent on purpose: its reference lives in test_swigluoai_routed_expert.py.
_TORCH_ACTIVATION = {
    ttnn.RoutedExpertActivation.Silu: ACTIVATION_SILU,
    ttnn.RoutedExpertActivation.SituGlu: ACTIVATION_SITU,
    ttnn.RoutedExpertActivation.ClampedSiluGlu: ACTIVATION_CLAMPED_SILU_GLU,
}

# Kimi K3 SiTU-GLU betas. The device kernel bakes SituGluConfigKimi; these must match it,
# or the reference silently grades against a different activation.
_SITU_BETA_GATE = KimiK3Config.ACTIVATION_SITU_BETA
_SITU_BETA_UP = KimiK3Config.ACTIVATION_SITU_LINEAR_BETA


def run_single_routed_expert(
    device,
    allocated_tokens: int,
    emb_dim: int,
    hidden_dim: int,
    active_tokens: int = None,
    x_row_major: bool = False,
    activation=None,
    weight_scale: float = 0.02,
    weights_dtype=ttnn.bfloat4_b,
    pcc_threshold: float = 0.97,
    min_cap_frac: tuple[float, float] | None = None,
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

    ``activation`` selects the fused kernel's activation and the matching torch reference;
    it defaults to SiLU (the DeepSeek path all pre-existing cases measure).

    ``weight_scale`` sets the gate/up/down init std. The default keeps the gate/up matmul
    outputs near O(1); raising it pushes them into the saturating region of the tanh-capped
    activations, which is the only way a SiTU-GLU case actually exercises the caps rather
    than their near-linear middle.

    ``weights_dtype`` defaults to the production bfloat4_b; ``pcc_threshold`` defaults to
    the usual 0.97. Both are parameters because saturating a capped activation costs
    accuracy under bf4 specifically: saturation turns the output nearly two-level, so bf4
    weight error stops passing through proportionally and instead flips whole terms of the
    down matmul. That is a property of bf4 + saturation rather than of the kernel (SiTU-GLU
    holds ~0.999 at every scale with bf8/bf16 weights, and the SFPU op passes its own
    saturation-coverage test), so a saturated bf4 case needs its own bar to stay meaningful.

    ``min_cap_frac`` is a ``(gate, up)`` pair of minimum fractions of activation inputs that
    must land past their respective cap (beta for SiTU-GLU, ±limit for clamped SiLU-GLU, where
    each element is the floor for both of that half's tails). Set it on any case whose point is the saturated
    region: without it, a change to ``weight_scale``, the dims or the seed would quietly drop
    the case back into the near-linear middle of both tanhs while still passing.
    """
    if active_tokens is None:
        active_tokens = allocated_tokens
    if activation is None:
        activation = ttnn.RoutedExpertActivation.Silu
    torch_activation = _TORCH_ACTIVATION.get(activation)
    if torch_activation is None:
        raise ValueError(f"no torch reference for {activation}; supported: {list(_TORCH_ACTIVATION)}")
    experts_per_chip = 1

    signpost(f"SingleRoutedExpert {allocated_tokens=} {active_tokens=} {emb_dim=} {hidden_dim=} {activation=}")

    logger.debug(f"Testing single routed expert: {allocated_tokens=}, {active_tokens=}, {emb_dim=}, {hidden_dim=}")
    logger.debug(f"Mesh: {device.shape}, num_devices={device.get_num_devices()}")

    # Create random weights
    torch.manual_seed(42)
    weights = {
        "gate_proj": torch.randn(hidden_dim, emb_dim, dtype=torch.float32) * weight_scale,
        "up_proj": torch.randn(hidden_dim, emb_dim, dtype=torch.float32) * weight_scale,
        "down_proj": torch.randn(emb_dim, hidden_dim, dtype=torch.float32) * weight_scale,
    }

    # Create torch reference
    torch_expert = TorchExpert(
        emb_dim,
        hidden_dim,
        weights,
        activation=torch_activation,
        situ_beta=_SITU_BETA_GATE,
        situ_linear_beta=_SITU_BETA_UP,
    )

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
        if torch_activation == ACTIVATION_SITU and min_cap_frac is not None:
            # How far into each tanh cap the inputs actually reach, so a saturation case can't
            # silently degrade into a near-linear run. Only the cases that assert it pay for the
            # two extra host matmuls (~225 GFLOP at the 5120-token shape, and the perf harness
            # calls this body once per iteration).
            gate_out = torch.nn.functional.linear(torch_active, weights["gate_proj"])
            up_out = torch.nn.functional.linear(torch_active, weights["up_proj"])
            gate_frac = (gate_out.abs() > _SITU_BETA_GATE).float().mean().item()
            up_frac = (up_out.abs() > _SITU_BETA_UP).float().mean().item()
            logger.info(
                f"SiTU-GLU cap coverage: |gate|>{_SITU_BETA_GATE}: {gate_frac:.1%}, "
                f"|up|>{_SITU_BETA_UP}: {up_frac:.1%}"
            )
            gate_min, up_min = min_cap_frac
            assert gate_frac >= gate_min, f"gate cap coverage {gate_frac:.1%} below {gate_min:.1%}"
            assert up_frac >= up_min, f"up cap coverage {up_frac:.1%} below {up_min:.1%}"
        elif torch_activation == ACTIVATION_CLAMPED_SILU_GLU and min_cap_frac is not None:
            # Each tail on its own count, never an |x| aggregate: a clamp applied to one side
            # only would otherwise pass on the other side's coverage. gate<-L is included even
            # though the kernel does not clamp there, so the stimulus keeps reaching the region
            # where a two-sided gate clamp would differ.
            gate_out = torch.nn.functional.linear(torch_active, weights["gate_proj"])
            up_out = torch.nn.functional.linear(torch_active, weights["up_proj"])
            lim = CLAMPED_SILU_GLU_LIMIT
            gate_min, up_min = min_cap_frac
            tails = [
                (f"gate>{lim}", (gate_out > lim).float().mean().item(), gate_min),
                (f"gate<-{lim}", (gate_out < -lim).float().mean().item(), gate_min),
                (f"up>{lim}", (up_out > lim).float().mean().item(), up_min),
                (f"up<-{lim}", (up_out < -lim).float().mean().item(), up_min),
            ]
            logger.info("clamped SiLU-GLU cap coverage: " + ", ".join(f"{name}: {frac:.1%}" for name, frac, _ in tails))
            for name, frac, floor in tails:
                assert frac >= floor, f"{name} coverage {frac:.1%} below {floor:.1%}"
    logger.debug(f"Torch output shape: {torch_output_active.shape}")

    # Create TTNN input: 2D (allocated_tokens, emb_dim), replicated across the 1-device mesh.
    # The composite op branches on x layout: ROW_MAJOR is bf16 (tilized and bf8-packed
    # inside the op), TILE is consumed directly as bf8. Pair the dtype with the layout so
    # each variation drives its real device path.
    tt_input = ttnn.from_torch(
        torch_input,
        mesh_mapper=ttnn.ReplicateTensorToMesh(device),
        layout=ttnn.ROW_MAJOR_LAYOUT if x_row_major else ttnn.TILE_LAYOUT,
        device=device,
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
            device=device,
            dtype=ttnn.uint32,
        )

    global_expert_idx_tt = _make_idx_tensor([0])
    expert_token_counts_tt = _make_idx_tensor([active_tokens])
    expert_region_offsets_tt = _make_idx_tensor([0])

    # Create TtRoutedExpert
    logger.debug("Creating TtRoutedExpert...")
    tt_expert = TtRoutedExpert(
        mesh_device=device,
        experts_per_chip=experts_per_chip,
        global_expert_idx_table=global_expert_idx_tt,
        emb_dim=emb_dim,
        hidden_dim=hidden_dim,
        max_tokens=allocated_tokens,
        torch_weights=[weights],  # List with single expert weights
        activations_dtype=ttnn.bfloat8_b,
        weights_dtype=weights_dtype,
        activation=activation,
    )

    # Run TTNN forward
    logger.debug("Running TTNN forward...")
    tt_output = tt_expert(tt_input, expert_token_counts_tt, expert_region_offsets_tt)
    logger.debug(f"TTNN output shape: {tt_output.shape}")

    # Convert back to torch for comparison. For a 1-device replicated tensor,
    # ConcatMeshToTensor(dim=0) with 1 slice is a no-op that returns the tensor.
    tt_output_torch = ttnn.to_torch(
        tt_output,
        mesh_composer=ttnn.ConcatMeshToTensor(device, dim=0),
    )
    logger.debug(f"TTNN output (torch) shape: {tt_output_torch.shape}")
    tt_output_active = tt_output_torch[:active_tokens]

    # Compare PCC over the active slice.
    _, pcc = comp_pcc(torch_output_active, tt_output_active)
    logger.debug(f"PCC over active slice ({active_tokens} rows): {pcc:.6f}")

    # Validate
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
# Kimi K3 is deliberately absent: _isl_params below takes config.EMB_SIZE as the routed-expert K
# axis, which holds only for models with no pre-projection. K3's LatentMoE projects 7168 -> 3584
# first, so adding it here would silently run it at 2x its real K. Its sweeps are below instead.


# Registry of currently-failing single-routed-expert cases -> xfail reason (with tracking issue), so
# CI stays green while linked issues are worked on. Applied strict, only on blackhole, by
# _xfail_blackhole (these cases pass on other arches, where an unconditional strict xfail would turn
# CI red on XPASS). Each key is a space-separated set of id tokens that must ALL appear in the param
# id, so a case can be scoped by any combination of layout ("x_tile"/"x_rm") and model/isl id.
# Empty: the program factory now snaps in0_block_w_gu to a divisor of K_gate_tiles on every path
# (not just when the L1 guard fires), so the prior gptoss_120b TILE-layout K_gate failure is fixed.
_XFAIL = {}


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

# Functional sweep: a few active-token counts across every model
_ISL_FUNCTIONAL_SWEEP = [251, 768, 3001]

# Exhaustive sweep: the full range from empty to fully-packed
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


@pytest.mark.uncollect_if(pred=ci_pruning.tiled_x_input)
@pytest.mark.parametrize("allocated_tokens, active_tokens, emb_dim, hidden_dim", _isl_params(_ISL_FUNCTIONAL_SWEEP))
@pytest.mark.parametrize("x_row_major", [True, False], ids=["x_rm", "x_tile"])
def test_single_routed_expert_functional(
    device,
    allocated_tokens: int,
    active_tokens: int,
    emb_dim: int,
    hidden_dim: int,
    x_row_major: bool,
):
    run_single_routed_expert(
        device,
        allocated_tokens,
        emb_dim,
        hidden_dim,
        active_tokens=active_tokens,
        x_row_major=x_row_major,
    )


@pytest.mark.uncollect_if(pred=ci_pruning.tiled_x_input)
@pytest.mark.parametrize(
    "allocated_tokens, active_tokens, emb_dim, hidden_dim",
    _isl_params(_ISL_EXHAUSTIVE_SWEEP, only_models=_ISL_EXHAUSTIVE_MODELS),
)
@pytest.mark.parametrize("x_row_major", [True, False], ids=["x_rm", "x_tile"])
@pytest.mark.skipif(not is_blackhole(), reason="device-side count-aware sparsity is Blackhole-only")
def test_single_routed_expert_isl_sweep(
    device,
    allocated_tokens: int,
    active_tokens: int,
    emb_dim: int,
    hidden_dim: int,
    x_row_major: bool,
):
    run_single_routed_expert(
        device,
        allocated_tokens,
        emb_dim,
        hidden_dim,
        active_tokens=active_tokens,
        x_row_major=x_row_major,
    )


# Kimi K3 token sweep. Unlike the ISL sweeps above (fixed 5K buffer, varying active count,
# which measures count-aware sparsity) this runs allocated == active, so each case is a
# fully-packed buffer at that token count -- the shape the DRAM-bandwidth / utilization /
# raw-time numbers are read off. 32 is one tile-row, 5120 the prefill dispatch width.
_K3_TOKEN_SWEEP = [32, 64, 128, 256, 512, 1024, 2048, 5120]


@pytest.mark.parametrize("num_tokens", _K3_TOKEN_SWEEP, ids=[f"t{t}" for t in _K3_TOKEN_SWEEP])
@pytest.mark.parametrize("x_row_major", [True, False], ids=["x_rm", "x_tile"])
@pytest.mark.extended_model
@pytest.mark.skipif(not is_blackhole(), reason="SiTU-GLU routed expert is Blackhole-only")
def test_single_routed_expert_k3_sweep(device, num_tokens: int, x_row_major: bool):
    """Kimi K3 routed expert: SiTU-GLU activation at the post-projection dims.

    K3's LatentMoE down-projects the embedding (EMB_SIZE -> ROUTED_EXPERT_HIDDEN_SIZE)
    before the routed experts, so the op's K axis is ROUTED_EXPERT_HIDDEN_SIZE=3584 with
    hidden 3072 -- the projection itself is a separate op and is not measured here.
    """
    run_single_routed_expert(
        device,
        num_tokens,
        KimiK3Config.ROUTED_EXPERT_HIDDEN_SIZE,
        KimiK3Config.MOE_INTERMEDIATE_SIZE,
        x_row_major=x_row_major,
        activation=ttnn.RoutedExpertActivation.SituGlu,
    )


# (gate, up) minimum cap-coverage floors, set just under the measured coverage at each
# weight_scale so a case fails loudly if it ever stops exercising the caps. Measured at the
# 512-token / 3584x3072 shape with seed 42: 65.6% / 0.5% at scale 0.15, 86.7% / 29.6% at 0.4.
# Scale 0.15 grazes the up cap by design -- it is the case where the gate cap carries the
# result on its own, which is exactly what makes it distinct from sat_deep.
_SAT_PARTIAL_CAP_FRAC = (0.60, 0.004)
_SAT_DEEP_CAP_FRAC = (0.80, 0.25)

# Saturation cases: (weight_scale, weights_dtype, pcc_threshold, min_cap_frac).
#
# bf8 isolates the activation -- it holds ~0.999 no matter how deep the saturation, so a
# regression in either tanh cap shows up immediately against the 0.97 bar.
#
# bf4 is the production weight dtype and is kept here deliberately, at a threshold matched
# to its measured floor (0.976 at scale 0.15, 0.968 at 0.4). Saturation makes the output
# near two-level, so bf4 weight error stops passing through proportionally and instead
# flips whole terms of the down matmul; that cost is real and is what K3 prefill will see,
# so it is measured rather than excluded. The bar still catches a genuine regression -- it
# sits just under the measured value, not at an arbitrary low number.
_K3_SATURATION_CASES = [
    pytest.param(0.15, ttnn.bfloat8_b, 0.97, _SAT_PARTIAL_CAP_FRAC, id="sat_partial-bf8"),
    pytest.param(0.4, ttnn.bfloat8_b, 0.97, _SAT_DEEP_CAP_FRAC, id="sat_deep-bf8"),
    pytest.param(0.15, ttnn.bfloat4_b, 0.97, _SAT_PARTIAL_CAP_FRAC, id="sat_partial-bf4"),
    pytest.param(0.4, ttnn.bfloat4_b, 0.96, _SAT_DEEP_CAP_FRAC, id="sat_deep-bf4"),
]

_K3_SATURATION_TOKENS = 512


@pytest.mark.parametrize("weight_scale, weights_dtype, pcc_threshold, min_cap_frac", _K3_SATURATION_CASES)
@pytest.mark.extended_model
@pytest.mark.skipif(not is_blackhole(), reason="SiTU-GLU routed expert is Blackhole-only")
def test_single_routed_expert_k3_saturated(
    device, weight_scale: float, weights_dtype, pcc_threshold: float, min_cap_frac: tuple[float, float]
):
    """Kimi K3 SiTU-GLU driven into the saturating region of both tanh caps.

    The sweep above runs at the default weight scale, where gate/up land around O(1) --
    well inside the near-linear middle of tanh(x/4) and tanh(x/25). A kernel that dropped
    either cap entirely would still pass there. These cases scale the weights so the caps
    carry the result (sat_deep reaches ~87% of gate and ~30% of up past their beta), across
    both the production bf4 weights and bf8 -- see _K3_SATURATION_CASES for why both.
    """
    run_single_routed_expert(
        device,
        _K3_SATURATION_TOKENS,
        KimiK3Config.ROUTED_EXPERT_HIDDEN_SIZE,
        KimiK3Config.MOE_INTERMEDIATE_SIZE,
        activation=ttnn.RoutedExpertActivation.SituGlu,
        weight_scale=weight_scale,
        weights_dtype=weights_dtype,
        pcc_threshold=pcc_threshold,
        min_cap_frac=min_cap_frac,
    )


# DeepSeek-V4 clamped SiLU-GLU. The functional sweep above runs both V4 shapes on the SiLU path
# at the default weight scale, where gate/up land near O(1) and never reach the limit of 10, so
# a kernel that dropped either clamp would pass there. These cases scale the weights until the
# clamps carry the result and assert the coverage.
#
# Scales differ per model because coverage tracks gate_out's std, which grows as sqrt(emb_dim).
# Measured at 512 tokens / seed 42, all four tails within 0.1pp: Pro 2.5% at 0.06 and 16.2% at
# 0.12; Flash 2.5% at 0.08 and 14.8% at 0.15.
_DSV4_PARTIAL_CAP_FRAC = (0.02, 0.02)
_DSV4_PRO_DEEP_CAP_FRAC = (0.14, 0.14)
_DSV4_FLASH_DEEP_CAP_FRAC = (0.12, 0.12)

# (config, weight_scale, weights_dtype, pcc_threshold, min_cap_frac).
#
# bf8 measures 0.99929-0.99943 and a dropped gate clamp gives 0.9936, so the partial cases need
# a bar inside (0.9936, 0.99929) and take 0.998; the file's usual 0.97 could not separate them.
# The deep cases only have to clear a 0.949 failure floor and take 0.99.
#
# bf4 is the production dtype and measures 0.9753-0.9767, i.e. its quantization error exceeds a
# dropped-clamp perturbation at the partial scale, so bf4 runs only at the deep scale where
# clamping dominates.
_DSV4_CLAMP_CASES = [
    pytest.param(DeepSeekV4ProConfig, 0.06, ttnn.bfloat8_b, 0.998, _DSV4_PARTIAL_CAP_FRAC, id="pro_partial-bf8"),
    pytest.param(DeepSeekV4FlashConfig, 0.08, ttnn.bfloat8_b, 0.998, _DSV4_PARTIAL_CAP_FRAC, id="flash_partial-bf8"),
    pytest.param(DeepSeekV4ProConfig, 0.12, ttnn.bfloat8_b, 0.99, _DSV4_PRO_DEEP_CAP_FRAC, id="pro_deep-bf8"),
    pytest.param(DeepSeekV4FlashConfig, 0.15, ttnn.bfloat8_b, 0.99, _DSV4_FLASH_DEEP_CAP_FRAC, id="flash_deep-bf8"),
    pytest.param(DeepSeekV4ProConfig, 0.12, ttnn.bfloat4_b, 0.97, _DSV4_PRO_DEEP_CAP_FRAC, id="pro_deep-bf4"),
    pytest.param(DeepSeekV4FlashConfig, 0.15, ttnn.bfloat4_b, 0.97, _DSV4_FLASH_DEEP_CAP_FRAC, id="flash_deep-bf4"),
]

_DSV4_CLAMP_TOKENS = 512


@pytest.mark.parametrize("config, weight_scale, weights_dtype, pcc_threshold, min_cap_frac", _DSV4_CLAMP_CASES)
# Both layouts: row-major is what production feeds the routed expert, and it tilizes inside the
# per-chunk loop, between BINARY_ACT_INIT() and the BINARY_ACT_TILE calls.
@pytest.mark.parametrize("x_row_major", [True, False], ids=["x_rm", "x_tile"])
@pytest.mark.extended_model
@pytest.mark.skipif(not is_blackhole(), reason="clamped SiLU-GLU routed expert is Blackhole-only")
def test_single_routed_expert_dsv4_clamped(
    device,
    config,
    weight_scale: float,
    weights_dtype,
    pcc_threshold: float,
    min_cap_frac: tuple[float, float],
    x_row_major: bool,
):
    """DeepSeek-V4 routed expert at V4 Pro and V4 Flash dims, with both clamps exercised.

    Catches a dropped clamp. Does not catch a gate clamped at both ends, at any threshold:
    silu(x) is within 4.5e-4 of zero for x <= -10, so clamping the gate's lower tail moves each
    element by at most 4.5e-4 * |up|, which vanishes into bf8 rounding of an O(10-100) FFN
    output. That case is covered by the op's elementwise test,
    tests/ttnn/unit_tests/operations/eltwise/test_clamped_silu_glu_sfpu.py.
    """
    run_single_routed_expert(
        device,
        _DSV4_CLAMP_TOKENS,
        config.EMB_SIZE,
        config.MOE_INTERMEDIATE_SIZE,
        x_row_major=x_row_major,
        activation=ttnn.RoutedExpertActivation.ClampedSiluGlu,
        weight_scale=weight_scale,
        weights_dtype=weights_dtype,
        pcc_threshold=pcc_threshold,
        min_cap_frac=min_cap_frac,
    )


# ClampedSiluGluConfigDsV4::limit is compile-time only, so it is unreachable from Python and a
# drift against the model configs would still score full PCC against a golden built from the
# Python constant. Parsing the header is the only check short of exporting it through the binding.
_KERNEL_HEADER = (
    pathlib.Path(__file__).parents[7]
    / "tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_clamped_silu_glu.h"
)
_KERNEL_LIMIT_RE = re.compile(r"struct\s+ClampedSiluGluConfigDsV4\s*\{[^}]*?\bfloat\s+limit\s*=\s*([0-9.]+)f", re.S)


@pytest.mark.parametrize("config", [DeepSeekV4ProConfig, DeepSeekV4FlashConfig], ids=["dsv4_pro", "dsv4_flash"])
def test_dsv4_clamp_limit_matches_kernel(config):
    """Host-only: the model config, the torch reference and the kernel all use one limit."""
    assert _KERNEL_HEADER.is_file(), f"kernel header not found at {_KERNEL_HEADER}; path is stale"
    match = _KERNEL_LIMIT_RE.search(_KERNEL_HEADER.read_text())
    assert match is not None, f"could not find ClampedSiluGluConfigDsV4::limit in {_KERNEL_HEADER}"
    assert config.SWIGLU_LIMIT == CLAMPED_SILU_GLU_LIMIT
    assert config.SWIGLU_LIMIT == float(match.group(1))


def test_dsv4_activation_enum_exposed():
    """Host-only: the enumerator reaches Python, at the value the program cache keys on."""
    activation = ttnn.RoutedExpertActivation
    assert hasattr(activation, "ClampedSiluGlu"), "enum is missing the ClampedSiluGlu variant"
    # Appended, not renumbered: the values are part of the program-cache key.
    assert activation.Silu.value == 0
    assert activation.SwiGluOai.value == 1
    assert activation.SituGlu.value == 2
    assert activation.ClampedSiluGlu.value == 3
