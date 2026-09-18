# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Single-device PCC test for HybridRoutedExpertFfn, the one-dispatch routed expert.

The same cases as test_single_routed_expert.py, driven onto the union op instead of the module:
the sweeps, dims and saturation cases are imported from there rather than copied, so the two
files cannot drift apart.

The op is the union of moe_fused_swiglu and unified_routed_expert_ffn -- both implementations
compiled into one binary per RISC-V, run as ordered passes over the same grid, with each expert
claimed by whichever half its active-token count selects. One expert per case, as in the
reference: the half that does not claim it still launches on every core and sweeps the counts, so
a single-expert case exercises both halves and the barrier between them.

Graded against TorchExpert, not against either shipping op, so the bar does not move when the
halves disagree numerically -- the merged compute binary runs with bfp8_pack_precise, which the
fused half requires and the unified op alone does not use.

The clamped SiLU-GLU cases from the reference have no counterpart here: the fused half implements
Silu, SituGlu and SwiGluOai, so a clamped activation would leave every below-threshold expert
unserved and the op rejects it outright.

The op is not wired into any model; nothing here should run in CI.
"""

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import is_blackhole
from models.demos.deepseek_v3_d_p.reference.deepseek_v3_config import DeepSeekV3Config
from models.demos.deepseek_v3_d_p.reference.kimi_k3_config import KimiK3Config
from models.demos.deepseek_v3_d_p.reference.tt.moe.expert import ACTIVATION_SITU, TorchExpert
from models.demos.deepseek_v3_d_p.tt.moe.tt_routed_expert import TtRoutedExpert
from tests.ttnn.utils_for_testing import comp_pcc
from tests.ttnn.nightly.unit_tests.operations.experimental.deepseek_prefill import ci_pruning
from tests.ttnn.nightly.unit_tests.operations.experimental.deepseek_prefill.test_single_routed_expert import (
    _ISL_ALLOCATED_TOKENS,
    _ISL_EXHAUSTIVE_MODELS,
    _ISL_EXHAUSTIVE_SWEEP,
    _ISL_FUNCTIONAL_SWEEP,
    _K3_SATURATION_CASES,
    _K3_TOKEN_SWEEP,
    _SITU_BETA_GATE,
    _SITU_BETA_UP,
    _TORCH_ACTIVATION,
    _isl_params,
    reshard_expert_weights_nd,
)

pytestmark = pytest.mark.uncollect_if(pred=ci_pruning.no_production_counterpart)

# Which half serves an expert. Fixed rather than read from each model's own
# ROUTED_EXPERT_HYBRID_TOKEN_THRESHOLD, because the models that carry one all carry 320 and the
# baseline model (dsv3) carries none at all -- and a case with no threshold is not this op. At 320
# the exhaustive sweep puts 0/128/256 on the fused half and 512 upward on the unified one, so both
# halves are graded across the sweep.
_THRESHOLD = 320


# Cases the union op cannot serve, strict-xfailed so they turn red the day they start working.
# A case matches when every whitespace-separated token of the key appears in the param id.
#
# dsv4_pro is the widest shape here (emb 7168, hidden 3072) and the only one that overflows: the
# union overlays both halves' circular buffers on ONE arena, which at the default worker_l1_size
# is 1,412,096 B after the op's scratch margin, and the fused half alone wants 1,418,112 B of it
# in ROW_MAJOR. TILE fits, because that layout carries no tilize buffers.
_XFAIL = {
    "dsv4_pro x_rm": "fused half needs 1418112 B of CB L1, union arena is 1412096 B (issue #56752)",
}


@pytest.fixture(autouse=True)
def _xfail_unsupported(request):
    """Apply _XFAIL to the cases whose ids match."""
    callspec = getattr(request.node, "callspec", None)
    if callspec is None:
        return
    for key, reason in _XFAIL.items():
        if all(token in callspec.id for token in key.split()):
            request.applymarker(pytest.mark.xfail(reason=reason, strict=True))
            break


def _idx_tensor(device, values):
    return ttnn.from_torch(
        torch.tensor(values, dtype=torch.int32),
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        dtype=ttnn.uint32,
    )


def run_hybrid_routed_expert(
    device,
    allocated_tokens: int,
    emb_dim: int,
    hidden_dim: int,
    active_tokens: int = None,
    x_row_major: bool = False,
    weights_dram_sharded: bool = False,
    activation=None,
    weight_scale: float = 0.02,
    weights_dtype=ttnn.bfloat4_b,
    pcc_threshold: float = 0.97,
    min_cap_frac: tuple[float, float] | None = None,
    threshold: int = _THRESHOLD,
):
    """One expert over an ``allocated_tokens`` region with ``active_tokens`` live rows, served by
    one half of the union op and swept over by the other.

    Signature mirrors ``run_single_routed_expert`` so the imported case matrices apply unchanged;
    see that docstring for what each knob is for. ``threshold`` is the one addition -- it decides
    which half owns the expert.
    """
    if active_tokens is None:
        active_tokens = allocated_tokens
    if activation is None:
        activation = ttnn.RoutedExpertActivation.Silu
    torch_activation = _TORCH_ACTIVATION.get(activation)
    if torch_activation is None:
        raise ValueError(f"no torch reference for {activation}; supported: {list(_TORCH_ACTIVATION)}")

    torch.manual_seed(42)
    weights = {
        "gate_proj": torch.randn(hidden_dim, emb_dim, dtype=torch.float32) * weight_scale,
        "up_proj": torch.randn(hidden_dim, emb_dim, dtype=torch.float32) * weight_scale,
        "down_proj": torch.randn(emb_dim, hidden_dim, dtype=torch.float32) * weight_scale,
    }

    torch_active = torch.randn(active_tokens, emb_dim, dtype=torch.float32)
    torch_input = torch.zeros(allocated_tokens, emb_dim, dtype=torch.float32)
    torch_input[:active_tokens] = torch_active

    torch_expert = TorchExpert(
        emb_dim,
        hidden_dim,
        weights,
        activation=torch_activation,
        situ_beta=_SITU_BETA_GATE,
        situ_linear_beta=_SITU_BETA_UP,
    )
    with torch.no_grad():
        torch_output_active = torch_expert(torch_active)
        if min_cap_frac is not None:
            # Same guard as the reference: without it a change to weight_scale, the dims or the
            # seed would quietly drop a saturation case back into the near-linear middle of both
            # tanhs while still passing.
            if torch_activation != ACTIVATION_SITU:
                raise ValueError(f"min_cap_frac given for {activation}, which defines no cap to measure")
            gate_out = torch.nn.functional.linear(torch_active, weights["gate_proj"])
            up_out = torch.nn.functional.linear(torch_active, weights["up_proj"])
            gate_min, up_min = min_cap_frac
            gate_frac = (gate_out.abs() > _SITU_BETA_GATE).float().mean().item()
            up_frac = (up_out.abs() > _SITU_BETA_UP).float().mean().item()
            logger.info(
                f"SiTU-GLU cap coverage: |gate|>{_SITU_BETA_GATE}: {gate_frac:.1%}, "
                f"|up|>{_SITU_BETA_UP}: {up_frac:.1%}"
            )
            assert gate_frac >= gate_min, f"gate cap coverage {gate_frac:.1%} below {gate_min:.1%}"
            assert up_frac >= up_min, f"up cap coverage {up_frac:.1%} below {up_min:.1%}"

    idx_tt = _idx_tensor(device, [0])
    counts_tt = _idx_tensor(device, [active_tokens])
    offsets_tt = _idx_tensor(device, [0])

    # TtRoutedExpert is the weight holder only; the op is called directly, because the module's
    # forward is where the two-op fallback lives and this file is about the union op.
    tt_expert = TtRoutedExpert(
        mesh_device=device,
        experts_per_chip=1,
        global_expert_idx_table=idx_tt,
        emb_dim=emb_dim,
        hidden_dim=hidden_dim,
        max_tokens=allocated_tokens,
        torch_weights=[weights],
        activations_dtype=ttnn.bfloat8_b,
        weights_dtype=weights_dtype,
        activation=activation,
    )
    if weights_dram_sharded:
        reshard_expert_weights_nd(tt_expert, device)

    # ROW_MAJOR is bf16 and tilized inside the op; TILE is consumed directly as bf8. Pair the
    # dtype with the layout so each variation drives its real device path.
    tt_input = ttnn.from_torch(
        torch_input,
        mesh_mapper=ttnn.ReplicateTensorToMesh(device),
        layout=ttnn.ROW_MAJOR_LAYOUT if x_row_major else ttnn.TILE_LAYOUT,
        device=device,
        dtype=ttnn.bfloat16 if x_row_major else ttnn.bfloat8_b,
    )

    tt_output = ttnn.experimental.deepseek_prefill.hybrid_routed_expert_moe(
        tt_input,
        offsets_tt,
        counts_tt,
        idx_tt,
        tt_expert.gate_projs,
        tt_expert.up_projs,
        tt_expert.down_projs,
        max_dispatched_tokens_per_expert=allocated_tokens,
        hybrid_token_threshold=threshold,
        compute_kernel_config=tt_expert.compute_kernel_config,
        activation=activation,
    )
    tt_output_torch = ttnn.to_torch(tt_output, mesh_composer=ttnn.ConcatMeshToTensor(device, dim=0))
    tt_output_active = tt_output_torch[:active_tokens]

    half = "fused" if active_tokens <= threshold else "unified"
    _, pcc = comp_pcc(torch_output_active, tt_output_active)
    logger.debug(f"PCC over active slice ({active_tokens} rows, {half} half): {pcc:.6f}")

    assert pcc >= pcc_threshold, f"PCC {pcc:.6f} below threshold {pcc_threshold} ({half} half)"
    assert not torch.isnan(tt_output_active).any(), "Active output contains NaN"
    assert not torch.isinf(tt_output_active).any(), "Active output contains Inf"


@pytest.mark.uncollect_if(pred=ci_pruning.tiled_x_input)
@pytest.mark.parametrize("allocated_tokens, active_tokens, emb_dim, hidden_dim", _isl_params(_ISL_FUNCTIONAL_SWEEP))
@pytest.mark.parametrize("x_row_major", [True, False], ids=["x_rm", "x_tile"])
@pytest.mark.skipif(not is_blackhole(), reason="the routed expert is Blackhole-only")
def test_hybrid_routed_expert_functional(
    device,
    allocated_tokens: int,
    active_tokens: int,
    emb_dim: int,
    hidden_dim: int,
    x_row_major: bool,
):
    run_hybrid_routed_expert(
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
@pytest.mark.parametrize("weights_dram_sharded", [False, True], ids=["w_interleaved", "w_ndshard"])
@pytest.mark.skipif(not is_blackhole(), reason="device-side count-aware sparsity is Blackhole-only")
def test_hybrid_routed_expert_isl_sweep(
    device,
    allocated_tokens: int,
    active_tokens: int,
    emb_dim: int,
    hidden_dim: int,
    x_row_major: bool,
    weights_dram_sharded: bool,
):
    run_hybrid_routed_expert(
        device,
        allocated_tokens,
        emb_dim,
        hidden_dim,
        active_tokens=active_tokens,
        x_row_major=x_row_major,
        weights_dram_sharded=weights_dram_sharded,
    )


@pytest.mark.uncollect_if(pred=ci_pruning.tiled_x_input)
@pytest.mark.parametrize("num_tokens", _K3_TOKEN_SWEEP, ids=[f"t{t}" for t in _K3_TOKEN_SWEEP])
@pytest.mark.parametrize("x_row_major", [True, False], ids=["x_rm", "x_tile"])
@pytest.mark.extended_model
@pytest.mark.skipif(not is_blackhole(), reason="SiTU-GLU routed expert is Blackhole-only")
def test_hybrid_routed_expert_k3_sweep(device, num_tokens: int, x_row_major: bool):
    """Fully-packed buffer at each token count, as in the reference's K3 sweep."""
    run_hybrid_routed_expert(
        device,
        num_tokens,
        KimiK3Config.ROUTED_EXPERT_HIDDEN_SIZE,
        KimiK3Config.MOE_INTERMEDIATE_SIZE,
        x_row_major=x_row_major,
        activation=ttnn.RoutedExpertActivation.SituGlu,
        # This sweep runs allocated == active, so the threshold has to scale with the case: the op
        # rejects one at or above max_dispatched_tokens_per_expert, where no expert could ever
        # reach the unified half. Half the count keeps both halves reachable at every size.
        threshold=max(1, num_tokens // 2),
    )


@pytest.mark.parametrize("weight_scale, weights_dtype, pcc_threshold, min_cap_frac", _K3_SATURATION_CASES)
@pytest.mark.extended_model
@pytest.mark.skipif(not is_blackhole(), reason="SiTU-GLU routed expert is Blackhole-only")
def test_hybrid_routed_expert_k3_saturated(
    device,
    weight_scale: float,
    weights_dtype,
    pcc_threshold: float,
    min_cap_frac,
):
    """SiTU-GLU driven into its caps, on the half the count selects."""
    run_hybrid_routed_expert(
        device,
        _ISL_ALLOCATED_TOKENS,
        KimiK3Config.ROUTED_EXPERT_HIDDEN_SIZE,
        KimiK3Config.MOE_INTERMEDIATE_SIZE,
        active_tokens=_ISL_ALLOCATED_TOKENS,
        x_row_major=True,
        activation=ttnn.RoutedExpertActivation.SituGlu,
        weight_scale=weight_scale,
        weights_dtype=weights_dtype,
        pcc_threshold=pcc_threshold,
        min_cap_frac=min_cap_frac,
    )


# Every model that measured a crossover runs 256 routed experts over 8 chips.
_MODEL_EXPERTS_PER_CHIP = 32


@pytest.mark.skipif(not is_blackhole(), reason="the routed expert is Blackhole-only")
def test_hybrid_routed_expert_config_fits_default_ring(device):
    """The union program's kernel config has to fit the ring a device gets by default.

    Sized at a model's expert count, not the one expert every case above uses, because the two are
    not the same program: the reader carries three per-expert weight addresses in its runtime args
    and the writer a fourth, so the config grows about 2 KB between one expert and a real
    32-expert layer, against a ring that does not move. It clears by a few hundred bytes, so any
    growth in either half's kernel text breaks it -- and without this that break lands in a
    32-device model test rather than here.

    One weight tensor is shared by all 32 experts on purpose: this grades the program's size, not
    its output, and 32 distinct copies of the real shape cost gigabytes of host memory.
    """
    emb_dim, hidden_dim = DeepSeekV3Config.EMB_SIZE, DeepSeekV3Config.MOE_INTERMEDIATE_SIZE
    torch.manual_seed(42)

    shared = {
        "gate_proj": torch.randn(hidden_dim, emb_dim, dtype=torch.float32) * 0.02,
        "up_proj": torch.randn(hidden_dim, emb_dim, dtype=torch.float32) * 0.02,
        "down_proj": torch.randn(emb_dim, hidden_dim, dtype=torch.float32) * 0.02,
    }
    counts = [32] * 24 + [512] * 8  # straddles the threshold, so neither half is optimised away
    offsets, running = [], 0
    for c in counts:
        offsets.append(running)
        running += c
    assert running <= _ISL_ALLOCATED_TOKENS

    x = ttnn.from_torch(
        torch.randn(_ISL_ALLOCATED_TOKENS, emb_dim, dtype=torch.float32),
        mesh_mapper=ttnn.ReplicateTensorToMesh(device),
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        dtype=ttnn.bfloat16,
    )
    idx_tt = _idx_tensor(device, list(range(_MODEL_EXPERTS_PER_CHIP)))
    tt_expert = TtRoutedExpert(
        mesh_device=device,
        experts_per_chip=_MODEL_EXPERTS_PER_CHIP,
        global_expert_idx_table=idx_tt,
        emb_dim=emb_dim,
        hidden_dim=hidden_dim,
        max_tokens=_ISL_ALLOCATED_TOKENS,
        torch_weights=[shared] * _MODEL_EXPERTS_PER_CHIP,
        activation=ttnn.RoutedExpertActivation.Silu,
    )

    try:
        ttnn.experimental.deepseek_prefill.hybrid_routed_expert_moe(
            x,
            _idx_tensor(device, offsets),
            _idx_tensor(device, counts),
            idx_tt,
            tt_expert.gate_projs,
            tt_expert.up_projs,
            tt_expert.down_projs,
            max_dispatched_tokens_per_expert=_ISL_ALLOCATED_TOKENS,
            hybrid_token_threshold=_THRESHOLD,
            compute_kernel_config=tt_expert.compute_kernel_config,
            activation=ttnn.RoutedExpertActivation.Silu,
        )
    except RuntimeError as exc:
        if "kernel config buffer" not in str(exc):
            raise
        pytest.fail(
            "the union program no longer fits the default kernel-config ring, so the op is broken "
            "on any device opened at the default worker_l1_size. Recover the bytes in kernel text "
            f"-- see the rules in hybrid_llk_shims.hpp for what may be out-of-lined.\n{exc}"
        )
