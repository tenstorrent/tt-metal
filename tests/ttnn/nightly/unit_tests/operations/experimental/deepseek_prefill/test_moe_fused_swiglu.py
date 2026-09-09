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
from models.demos.deepseek_v3_d_p.reference.tt.moe.expert import (
    ACTIVATION_SILU,
    ACTIVATION_SITU,
    ACTIVATION_SWIGLUOAI,
    TorchExpert,
    apply_glu_activation,
)
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
    ttnn.RoutedExpertActivation.SwiGluOai: ACTIVATION_SWIGLUOAI,
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
# Token counts whose LAST M-block is SHORT, i.e. m_eff < M_BLOCK. Every count in the sweep above
# lands on a full 8-tile block, and the exhaustive sweep below reaches m_eff 4 on ONE model, so
# without these no other shape ever runs a reduced m_eff. That is not hypothetical: a cb_h capacity
# bug that only misfired at m_t 3-4 reached two production shapes through this exact hole.
# `m_eff_min` is pow2_ceil(OUT_SUBBLOCK_H_GU) and so shape-dependent, which is why these are token
# counts rather than target m_eff values -- each shape rounds them to whatever its geometry allows.
#   67  -> m_t 3: one short block, and m_t < m_eff so the padded tile-rows are live too.
#   289 -> m_t 10: a FULL block then a 2-tile tail, a different path from a lone short block
#          because the tail reuses CB slots the full block already cycled.
_ISL_SHORT_BLOCK_SWEEP = [67, 289]
_ISL_EXHAUSTIVE_SWEEP = [0, 128, 256, 512, 1024, 2048, 4096, 5120]
# "kimi_k26" used to sit here and matched nothing: SINGLE_EXPERT_MODELS calls that shape
# kimi_k2_7, so the sweep silently ran ONE model for however long the name was stale.
_ISL_EXHAUSTIVE_MODELS = ("kimi_k2_7", "glm_51")


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
    gate_bias=None,
    up_bias=None,
    down_bias=None,
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
        if gate_bias is None:
            torch_expert = TorchExpert(
                emb_dim,
                hidden_dim,
                weights,
                activation=torch_activation,
                situ_beta=_SITU_BETA_GATE,
                situ_linear_beta=_SITU_BETA_UP,
            )
            torch_output_active = torch_expert(torch_active)
        else:
            # TorchExpert carries no bias, and the bias terms sit outside apply_glu_activation's
            # contract anyway -- gate/up before the activation, down after the down matmul -- so a
            # biased case builds the reference explicitly around that one shared activation.
            gb = ttnn.to_torch(gate_bias).reshape(-1)[:hidden_dim].float()
            ub = ttnn.to_torch(up_bias).reshape(-1)[:hidden_dim].float()
            db = ttnn.to_torch(down_bias).reshape(-1)[:emb_dim].float()
            gate_out = torch_active @ weights["gate_proj"].T + gb
            up_out = torch_active @ weights["up_proj"].T + ub
            activated = apply_glu_activation(
                gate_out,
                up_out,
                activation=torch_activation,
                situ_beta=_SITU_BETA_GATE,
                situ_linear_beta=_SITU_BETA_UP,
            )
            torch_output_active = activated @ weights["down_proj"].T + db

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
        gate_biases=None if gate_bias is None else [gate_bias],
        up_biases=None if up_bias is None else [up_bias],
        down_biases=None if down_bias is None else [down_bias],
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
_XFAIL = {
    # 7168x3072 stopped fitting when the gate/up accumulators went bf16: cb_layout needs 1_694_592
    # bytes against a 1_461_248 budget at 11x8, so the op TT_FATALs in the program factory before it
    # runs -- every token count, both x layouts. The L1 went to phase_cb_alias(), which is now dead
    # for EVERY shape (cb_gather_gate is bf16 while cb_h_slice and cb_out_tiles are bfp8, so the
    # three views can never agree on page size); recovering it is worth ~233 KB here.
    # Strict on purpose: this turns red on XPASS the moment the shape fits again, which is the
    # signal that the threshold removed from DeepSeekV4ProConfig can be re-derived.
    "dsv4_pro": "moe_fused_swiglu exceeds CB L1 on 7168x3072 under bf16 gate/up accumulators",
}


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
@pytest.mark.parametrize(
    "allocated_tokens, active_tokens, emb_dim, hidden_dim",
    _isl_params(_ISL_FUNCTIONAL_SWEEP + _ISL_SHORT_BLOCK_SWEEP),
)
@pytest.mark.parametrize("x_row_major", [True, False], ids=["x_rm", "x_tile"])
@pytest.mark.skipif(not is_blackhole(), reason="moe_fused_swiglu is Blackhole-only")
def test_moe_fused_swiglu_functional(
    device, allocated_tokens: int, active_tokens: int, emb_dim: int, hidden_dim: int, x_row_major: bool
):
    """Per-model dims on the ragged token counts and the short-last-block counts, both x layouts."""
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
