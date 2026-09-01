# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Minimal single-device, single-expert test for the hybrid routed-expert dispatch.

The third of the trio: test_single_routed_expert.py drives TtRoutedExpert onto the composite,
test_moe_fused_swiglu.py calls the fused op directly, and this one drives TtRoutedExpert with a
hybrid_token_threshold, so both ops are reachable through the module and a failure here is the
split rather than either op.

The threshold is not a test axis. It comes from each model's own
ROUTED_EXPERT_HYBRID_TOKEN_THRESHOLD, read the way tt_prefill_block reads it, so a case grades the
band the shipped config actually routes that token count to -- `count <= threshold` to
moe_fused_swiglu, the rest to unified_routed_expert_moe. Models that ship no threshold are absent:
with none, TtRoutedExpert is single-op and test_single_routed_expert already covers it.

The measured crossover the thresholds encode is gated in test_routed_expert_crossover_perf.py.
"""

from typing import Optional

import pytest
import torch
from loguru import logger
from tracy import signpost

import ttnn
from models.common.utility_functions import is_blackhole
from models.demos.deepseek_v3_d_p.reference.kimi_k3_config import KimiK3Config
from models.demos.deepseek_v3_d_p.reference.tt.moe.expert import ACTIVATION_SILU, ACTIVATION_SITU, TorchExpert
from models.demos.deepseek_v3_d_p.tt.moe.tt_routed_expert import TtRoutedExpert
from tests.ttnn.utils_for_testing import comp_pcc
from tests.ttnn.nightly.unit_tests.operations.experimental.deepseek_prefill import ci_pruning
from tests.ttnn.nightly.unit_tests.operations.experimental.deepseek_prefill.test_single_routed_expert import (
    _ISL_ALLOCATED_TOKENS,
    _ISL_EXHAUSTIVE_MODELS,
    _ISL_EXHAUSTIVE_SWEEP,
    _ISL_FUNCTIONAL_SWEEP,
    SINGLE_EXPERT_MODELS,
)

SINGLE_CHIP_MESH_PARAMS = [
    pytest.param(
        1,
        {"fabric_config": ttnn.FabricConfig.DISABLED},
        id="single-chip",
    ),
]

# Device activation -> the TorchExpert reference that must match it, so a case cannot grade one
# activation against another's golden.
_TORCH_ACTIVATION = {
    ttnn.RoutedExpertActivation.Silu: ACTIVATION_SILU,
    ttnn.RoutedExpertActivation.SituGlu: ACTIVATION_SITU,
}

# The device kernel bakes SituGluConfigKimi; these must match it or the reference silently grades
# against a different activation.
_SITU_BETA_GATE = KimiK3Config.ACTIVATION_SITU_BETA
_SITU_BETA_UP = KimiK3Config.ACTIVATION_SITU_LINEAR_BETA


def _threshold_of(config) -> Optional[int]:
    """The hybrid split the model ships, read as tt_prefill_block reads it. A config without one
    runs single-op, which TtRoutedExpert spells `None`."""
    return getattr(config, "ROUTED_EXPERT_HYBRID_TOKEN_THRESHOLD", None)


def run_routed_expert_hybrid(
    device,
    allocated_tokens: int,
    emb_dim: int,
    hidden_dim: int,
    threshold: Optional[int],
    active_tokens: int = None,
    x_row_major: bool = True,
    activation=None,
    weight_scale: float = 0.02,
    weights_dtype=ttnn.bfloat4_b,
    pcc_threshold: float = 0.97,
):
    """
    One chip, one expert, TtRoutedExpert(hybrid_token_threshold=threshold).

    The expert's dispatch buffer is sized for ``allocated_tokens`` but only the first
    ``active_tokens`` rows hold real data; the rest is zero padding. ``active_tokens`` defaults to
    ``allocated_tokens``.

    With one expert the threshold selects the band and nothing else: ``active_tokens <= threshold``
    sends it to moe_fused_swiglu and anything above to unified_routed_expert_moe, and the band that
    does not claim it still launches and skips it. Both are graded against the same TorchExpert
    reference, so a band that claims an expert it should not -- or drops one it should -- shows up
    as the output holding the raw allocation instead of the expert's result.
    """
    if active_tokens is None:
        active_tokens = allocated_tokens
    if activation is None:
        activation = ttnn.RoutedExpertActivation.Silu
    torch_activation = _TORCH_ACTIVATION.get(activation)
    if torch_activation is None:
        raise ValueError(f"no torch reference for {activation}; supported: {list(_TORCH_ACTIVATION)}")
    owner = "fused" if threshold is not None and active_tokens <= threshold else "composite"

    signpost(
        f"RoutedExpertHybrid {allocated_tokens=} {active_tokens=} {emb_dim=} {hidden_dim=} "
        f"{threshold=} {owner=} {activation=}"
    )

    torch.manual_seed(42)
    # torch.nn.Linear convention (out_features, in_features), which is what TorchExpert reads.
    weights = {
        "gate_proj": torch.randn(hidden_dim, emb_dim, dtype=torch.float32) * weight_scale,
        "up_proj": torch.randn(hidden_dim, emb_dim, dtype=torch.float32) * weight_scale,
        "down_proj": torch.randn(emb_dim, hidden_dim, dtype=torch.float32) * weight_scale,
    }

    torch_active = torch.randn(active_tokens, emb_dim, dtype=torch.float32)
    torch_input = torch.zeros(allocated_tokens, emb_dim, dtype=torch.float32)
    torch_input[:active_tokens] = torch_active

    with torch.no_grad():
        torch_expert = TorchExpert(
            emb_dim,
            hidden_dim,
            weights,
            activation=torch_activation,
            situ_beta=_SITU_BETA_GATE,
            situ_linear_beta=_SITU_BETA_UP,
        )
        torch_output_active = torch_expert(torch_active)

    # TtRoutedExpert branches on x layout: ROW_MAJOR is bf16 (tilized and bf8-packed inside the op),
    # TILE is consumed directly as bf8. Pair dtype with layout so each variant drives its real path.
    tt_input = ttnn.from_torch(
        torch_input,
        mesh_mapper=ttnn.ReplicateTensorToMesh(device),
        layout=ttnn.ROW_MAJOR_LAYOUT if x_row_major else ttnn.TILE_LAYOUT,
        device=device,
        dtype=ttnn.bfloat16 if x_row_major else ttnn.bfloat8_b,
    )

    # Single-expert auxiliaries: local 0 -> global 0, region starts at row 0, and the runtime count
    # is what both bands read to decide whether this expert is theirs.
    def idx_tensor(values):
        return ttnn.from_torch(
            torch.tensor(values, dtype=torch.int32), layout=ttnn.ROW_MAJOR_LAYOUT, device=device, dtype=ttnn.uint32
        )

    tt_expert = TtRoutedExpert(
        mesh_device=device,
        experts_per_chip=1,
        global_expert_idx_table=idx_tensor([0]),
        emb_dim=emb_dim,
        hidden_dim=hidden_dim,
        max_tokens=allocated_tokens,
        torch_weights=[weights],
        activations_dtype=ttnn.bfloat8_b,
        weights_dtype=weights_dtype,
        activation=activation,
        hybrid_token_threshold=threshold,
    )
    tt_output = tt_expert(tt_input, idx_tensor([active_tokens]), idx_tensor([0]))

    # For a 1-device replicated tensor, ConcatMeshToTensor(dim=0) with 1 slice returns the tensor.
    tt_output_torch = ttnn.to_torch(tt_output, mesh_composer=ttnn.ConcatMeshToTensor(device, dim=0))

    if active_tokens == 0:
        # A zero count is skipped by BOTH bands, so no row is written and the output holds whatever
        # the allocation held before. There is nothing to grade: reaching here IS the assertion.
        return

    tt_output_active = tt_output_torch[:active_tokens].float()
    _, pcc = comp_pcc(torch_output_active, tt_output_active)
    logger.debug(f"PCC over active slice ({active_tokens} rows, {owner} band): {pcc:.6f}")

    assert not torch.isnan(tt_output_active).any(), f"Active output contains NaN ({owner} band)"
    assert not torch.isinf(tt_output_active).any(), f"Active output contains Inf ({owner} band)"
    assert pcc >= pcc_threshold, f"PCC {pcc:.6f} below threshold {pcc_threshold} ({owner} band)"


# Registry of currently-failing hybrid cases -> xfail reason (with tracking issue), so CI stays green
# while linked issues are worked on. Applied strict, only on blackhole, by _xfail_blackhole (these
# cases pass on other arches, where an unconditional strict xfail would turn CI red on XPASS). Each
# key is a space-separated set of id tokens that must ALL appear in the param id.
_XFAIL = {}


@pytest.fixture(autouse=True)
def _xfail_blackhole(request, silicon_arch_name):
    """Strict-xfail the _XFAIL cases only on blackhole. A case matches when every
    whitespace-separated token of the key appears in the param id."""
    if silicon_arch_name != "blackhole":
        return
    callspec = getattr(request.node, "callspec", None)
    if callspec is None:
        return
    for key, reason in _XFAIL.items():
        if all(token in callspec.id for token in key.split()):
            request.applymarker(pytest.mark.xfail(reason=reason, strict=True))
            break


# The composite consumes a TILE activation buffer IN PLACE and hands it straight back as the
# output. A hybrid that runs both bands then calls moe_fused_swiglu with an output aliasing its own
# activations, which the op rejects -- at validation, so the case fails whichever band owns the
# expert and even at count 0. Unreachable when the threshold drops the composite entirely
# (fused_only allocates fresh), and unreachable in production, where the routed expert is always fed
# row-major; ci_pruning.tiled_x_input prunes these in CI for that reason.
_TILE_X_OUTPUT_ALIAS = (
    "hybrid on TILE x: the composite writes in place, so the fused band's output aliases its "
    "activations and trips the op's alias guard"
)


@pytest.fixture(autouse=True)
def _xfail_tile_x_with_live_composite(request):
    """Strict-xfail the TILE-x cases whose threshold leaves the composite a band to run, so the day
    the aliasing is fixed these XPASS and the marker comes out."""
    params = getattr(getattr(request.node, "callspec", None), "params", {})
    threshold = params.get("threshold")
    if params.get("x_row_major", True) or threshold is None:
        return
    if threshold < _ISL_ALLOCATED_TOKENS:
        request.applymarker(pytest.mark.xfail(reason=_TILE_X_OUTPUT_ALIAS, strict=True))


def _isl_params(active_sweep, only_models=None):
    """Per-model dims and shipped threshold crossed with a token sweep, all against the fixed
    _ISL_ALLOCATED_TOKENS buffer. Reuses SINGLE_EXPERT_MODELS so non-baseline models stay gated
    behind the extended_model marker; `only_models` restricts to a subset of model names.

    A model with no threshold is dropped rather than run at `None`: that is the single-op path
    test_single_routed_expert already grades, and it would not exercise a split at all.

    Kimi K3 has no case in this file at all: it is not in SINGLE_EXPERT_MODELS (EMB_SIZE would run
    it at 2x its real K, since LatentMoE projects 7168 -> 3584 first), and it does not enable the
    split either, so there would be nothing here for it to grade.
    """
    params = []
    for name, config, extended in SINGLE_EXPERT_MODELS:
        if only_models is not None and name not in only_models:
            continue
        threshold = _threshold_of(config)
        if threshold is None:
            continue
        for active in active_sweep:
            params.append(
                pytest.param(
                    _ISL_ALLOCATED_TOKENS,
                    active,
                    config.EMB_SIZE,
                    config.MOE_INTERMEDIATE_SIZE,
                    threshold,
                    marks=pytest.mark.extended_model if extended else (),
                    # "-t" keeps ids collision-free under -k: "512" is a substring of "5120".
                    id=f"{name}-t{active}",
                )
            )
    return params


@pytest.mark.parametrize(
    "mesh_device, device_params", SINGLE_CHIP_MESH_PARAMS, indirect=["mesh_device", "device_params"]
)
@pytest.mark.uncollect_if(pred=ci_pruning.tiled_x_input)
@pytest.mark.parametrize(
    "allocated_tokens, active_tokens, emb_dim, hidden_dim, threshold", _isl_params(_ISL_FUNCTIONAL_SWEEP)
)
@pytest.mark.parametrize("x_row_major", [True, False], ids=["x_rm", "x_tile"])
@pytest.mark.skipif(not is_blackhole(), reason="the hybrid dispatch is Blackhole-only")
def test_tt_routed_expert_hybrid_functional(
    mesh_device,
    device_params,
    allocated_tokens: int,
    active_tokens: int,
    emb_dim: int,
    hidden_dim: int,
    threshold: Optional[int],
    x_row_major: bool,
):
    """Per-model dims on the ragged token counts, over both x layouts."""
    run_routed_expert_hybrid(
        mesh_device,
        allocated_tokens,
        emb_dim,
        hidden_dim,
        threshold,
        active_tokens=active_tokens,
        x_row_major=x_row_major,
    )


@pytest.mark.parametrize(
    "mesh_device, device_params", SINGLE_CHIP_MESH_PARAMS, indirect=["mesh_device", "device_params"]
)
@pytest.mark.uncollect_if(pred=ci_pruning.tiled_x_input)
@pytest.mark.parametrize(
    "allocated_tokens, active_tokens, emb_dim, hidden_dim, threshold",
    _isl_params(_ISL_EXHAUSTIVE_SWEEP, only_models=_ISL_EXHAUSTIVE_MODELS),
)
@pytest.mark.parametrize("x_row_major", [True, False], ids=["x_rm", "x_tile"])
@pytest.mark.skipif(not is_blackhole(), reason="the hybrid dispatch is Blackhole-only")
def test_tt_routed_expert_hybrid_isl_sweep(
    mesh_device,
    device_params,
    allocated_tokens: int,
    active_tokens: int,
    emb_dim: int,
    hidden_dim: int,
    threshold: Optional[int],
    x_row_major: bool,
):
    """The aligned sweep, which straddles each model's threshold in both directions: kimi_k26's
    sentinel keeps every count fused, glm_51's 1792 puts 1024 and below in the fused band and 2048
    and above in the composite's."""
    run_routed_expert_hybrid(
        mesh_device,
        allocated_tokens,
        emb_dim,
        hidden_dim,
        threshold,
        active_tokens=active_tokens,
        x_row_major=x_row_major,
    )
