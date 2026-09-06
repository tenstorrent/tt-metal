# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""
Minimal single-device, single-expert test for the moe_fused_swiglu op.

The op counterpart to test_single_routed_expert.py: that one drives TtRoutedExpert (and so the
composite unified_routed_expert_moe), this one calls moe_fused_swiglu directly, so a failure here
is the op rather than the module wiring it. run_moe_fused_swiglu is the shared body, imported by
test_moe_fused_swiglu_perf.py so a perf case also grades correctness.
"""

import pytest
import torch
from loguru import logger
from tracy import signpost

import ttnn
from models.common.utility_functions import is_blackhole
from models.demos.deepseek_v3_d_p.reference.kimi_k3_config import KimiK3Config
from models.demos.deepseek_v3_d_p.reference.tt.moe.expert import ACTIVATION_SILU, ACTIVATION_SITU, TorchExpert
from tests.ttnn.utils_for_testing import comp_pcc
from tests.ttnn.nightly.unit_tests.operations.experimental.deepseek_prefill import ci_pruning
from tests.ttnn.nightly.unit_tests.operations.experimental.deepseek_prefill.test_single_routed_expert import (
    SINGLE_EXPERT_MODELS,
)

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

# Pinned rather than left to the device grid: the op's blocking is a function of the grid, so a
# fixed rectangle is what keeps the perf baselines in the sibling file comparable across boards.
GRID = ttnn.CoreCoord(11, 8)

_ISL_ALLOCATED_TOKENS = 5120
# Deliberately not tile-aligned and not powers of two: these drive the ragged tail of the token
# axis, which the aligned sweep below never reaches.
_ISL_FUNCTIONAL_SWEEP = [251, 768, 3001]
_ISL_EXHAUSTIVE_SWEEP = [0, 128, 256, 512, 1024, 2048, 4096, 5120]
_ISL_EXHAUSTIVE_MODELS = ("kimi_k26", "glm_51")


def run_moe_fused_swiglu(
    device,
    allocated_tokens: int,
    emb_dim: int,
    hidden_dim: int,
    active_tokens: int = None,
    x_row_major: bool = True,
    activation=None,
    weight_scale: float = 0.02,
    weights_dtype=ttnn.bfloat4_b,
    pcc_threshold: float = 0.97,
    core_grid=GRID,
):
    """
    One chip, one expert, moe_fused_swiglu called directly.

    The expert's dispatch buffer is sized for ``allocated_tokens`` but only the first
    ``active_tokens`` rows hold real data; the rest is zero padding. ``active_tokens`` defaults to
    ``allocated_tokens``. Below it, this exercises the op's device-side count sparsity: the counts
    vector is DEVICE-resident, so the kernel reads its own row budget and must (a) be correct on the
    active slice and (b) not matmul the padding.

    ``activation`` defaults to SiLU. ``weight_scale`` sets the gate/up/down init std; the default
    keeps the matmul outputs near O(1), which is the near-linear middle of the tanh-capped SiTU-GLU
    activation rather than its caps.
    """
    if active_tokens is None:
        active_tokens = allocated_tokens
    if activation is None:
        activation = ttnn.RoutedExpertActivation.Silu
    torch_activation = _TORCH_ACTIVATION.get(activation)
    if torch_activation is None:
        raise ValueError(f"no torch reference for {activation}; supported: {list(_TORCH_ACTIVATION)}")

    signpost(f"MoeFusedSwiGlu {allocated_tokens=} {active_tokens=} {emb_dim=} {hidden_dim=} {activation=}")

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

    # The op addresses weights as [K, N] tile pages, the transpose of the Linear convention above.
    def to_device(tensor, dtype, layout):
        return ttnn.from_torch(
            tensor.contiguous(), dtype=dtype, layout=layout, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )

    w_gate = to_device(weights["gate_proj"].T, weights_dtype, ttnn.TILE_LAYOUT)
    w_up = to_device(weights["up_proj"].T, weights_dtype, ttnn.TILE_LAYOUT)
    w_down = to_device(weights["down_proj"].T, weights_dtype, ttnn.TILE_LAYOUT)

    # ROW_MAJOR x is bf16 and tilized inside the op (the Blackhole production fast path); TILE x is
    # consumed directly as bf8. Pair dtype with layout so each variant drives its real device path.
    x = to_device(
        torch_input.reshape(1, 1, allocated_tokens, emb_dim),
        ttnn.bfloat16 if x_row_major else ttnn.bfloat8_b,
        ttnn.ROW_MAJOR_LAYOUT if x_row_major else ttnn.TILE_LAYOUT,
    )

    # Single expert: local 0 -> global 0, its region starts at row 0, and the runtime count is what
    # drives the sparsity.
    def idx_tensor(values):
        return to_device(torch.tensor(values, dtype=torch.int32), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)

    idx = idx_tensor([0])
    counts = idx_tensor([active_tokens])

    output = ttnn.experimental.deepseek_prefill.moe_fused_swiglu(
        x,
        [w_gate],
        [w_up],
        [w_down],
        counts,
        idx,
        input_m_tiles=allocated_tokens // 32,
        core_grid=core_grid,
        activation=activation,
    )
    tt_output = ttnn.to_torch(output)[0, 0]

    if active_tokens == 0:
        # A zero count skips the expert, so no row is written and the freshly allocated output holds
        # whatever DRAM held before. There is nothing to grade: reaching here IS the assertion.
        return

    tt_output_active = tt_output[:active_tokens].float()
    _, pcc = comp_pcc(torch_output_active, tt_output_active)
    logger.debug(f"PCC over active slice ({active_tokens} rows): {pcc:.6f}")

    assert not torch.isnan(tt_output_active).any(), "Active output contains NaN"
    assert not torch.isinf(tt_output_active).any(), "Active output contains Inf"
    assert pcc >= pcc_threshold, f"PCC {pcc:.6f} below threshold {pcc_threshold}"


# Known-unfittable (grid, dims, layout) combinations -> reason. Strict, so making one fit turns CI
# red on XPASS rather than leaving a stale entry. Each key is a space-separated set of id tokens that
# must ALL appear in the param id.
# Empty: the `depth_x` 2 -> 1 fallback now serves tiled x as well as row-major, and the reader drops
# its cross-M-block prefetch there rather than aiming at a slot that does not exist, so dsv4_pro on
# TILE x (emb 7168 / hidden 3072, the only shape that overshot at 11x8) fits.
_XFAIL = {}


@pytest.fixture(autouse=True)
def _xfail_known_unfittable(request):
    """Strict-xfail the _XFAIL cases. A case matches when every whitespace-separated token of the key
    appears in the param id."""
    callspec = getattr(request.node, "callspec", None)
    if callspec is None:
        return
    for key, reason in _XFAIL.items():
        if all(token in callspec.id for token in key.split()):
            request.applymarker(pytest.mark.xfail(reason=reason, strict=True))
            break


def _skip_if_grid_too_small(device):
    available = device.compute_with_storage_grid_size()
    if GRID.x > available.x or GRID.y > available.y:
        pytest.skip(f"requested {GRID.y}x{GRID.x} grid exceeds available {available.y}x{available.x}")


def _isl_params(active_sweep, only_models=None):
    """Per-model dims crossed with a token sweep, taking the routed-expert K axis from EMB_SIZE.

    Kimi K3 is absent for the same reason as in test_single_routed_expert: its LatentMoE projects
    7168 -> 3584 first, so EMB_SIZE would run it at 2x its real K. Its case is separate below.
    """
    params = []
    for name, config, extended in SINGLE_EXPERT_MODELS:
        if only_models is not None and name not in only_models:
            continue
        for active in active_sweep:
            params.append(
                pytest.param(
                    _ISL_ALLOCATED_TOKENS,
                    active,
                    config.EMB_SIZE,
                    config.MOE_INTERMEDIATE_SIZE,
                    marks=pytest.mark.extended_model if extended else (),
                    # "-t" keeps ids collision-free under -k: "512" is a substring of "5120".
                    id=f"{name}-t{active}",
                )
            )
    return params


@pytest.mark.uncollect_if(pred=ci_pruning.tiled_x_input)
@pytest.mark.parametrize("allocated_tokens, active_tokens, emb_dim, hidden_dim", _isl_params(_ISL_FUNCTIONAL_SWEEP))
@pytest.mark.parametrize("x_row_major", [True, False], ids=["x_rm", "x_tile"])
@pytest.mark.skipif(not is_blackhole(), reason="moe_fused_swiglu is Blackhole-only")
def test_moe_fused_swiglu_functional(
    device, allocated_tokens: int, active_tokens: int, emb_dim: int, hidden_dim: int, x_row_major: bool
):
    """Per-model dims on the ragged token counts, over both x layouts."""
    _skip_if_grid_too_small(device)
    run_moe_fused_swiglu(
        device, allocated_tokens, emb_dim, hidden_dim, active_tokens=active_tokens, x_row_major=x_row_major
    )


@pytest.mark.parametrize(
    "allocated_tokens, active_tokens, emb_dim, hidden_dim",
    _isl_params(_ISL_EXHAUSTIVE_SWEEP, only_models=_ISL_EXHAUSTIVE_MODELS),
)
@pytest.mark.skipif(not is_blackhole(), reason="moe_fused_swiglu is Blackhole-only")
def test_moe_fused_swiglu_isl_sweep(device, allocated_tokens: int, active_tokens: int, emb_dim: int, hidden_dim: int):
    """The aligned sweep the perf baselines are keyed on, x_rm only (the production path)."""
    _skip_if_grid_too_small(device)
    run_moe_fused_swiglu(device, allocated_tokens, emb_dim, hidden_dim, active_tokens=active_tokens, x_row_major=True)


@pytest.mark.uncollect_if(pred=ci_pruning.tiled_x_input)
@pytest.mark.parametrize("active_tokens", _ISL_EXHAUSTIVE_SWEEP, ids=[f"t{t}" for t in _ISL_EXHAUSTIVE_SWEEP])
@pytest.mark.parametrize("x_row_major", [True, False], ids=["x_rm", "x_tile"])
@pytest.mark.skipif(not is_blackhole(), reason="SiTU-GLU is Blackhole-only")
def test_moe_fused_swiglu_k3_situ(device, active_tokens: int, x_row_major: bool):
    """Kimi K3: SiTU-GLU at the post-projection dims, so K is ROUTED_EXPERT_HIDDEN_SIZE."""
    _skip_if_grid_too_small(device)
    run_moe_fused_swiglu(
        device,
        _ISL_ALLOCATED_TOKENS,
        KimiK3Config.ROUTED_EXPERT_HIDDEN_SIZE,
        KimiK3Config.MOE_INTERMEDIATE_SIZE,
        active_tokens=active_tokens,
        x_row_major=x_row_major,
        activation=ttnn.RoutedExpertActivation.SituGlu,
    )
