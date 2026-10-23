# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Device perf gate for the three routed-expert ops on a MULTI-expert load: unified_routed_expert_ffn,
moe_fused_swiglu and hybrid_routed_expert_ffn, each run on the same experts, x and weights and
measured with the real-time program profiler.

Every other perf gate runs one expert, so each op's per-dispatch cost -- the launch, and for the
hybrid the unified pass that finds nothing to do plus the grid barrier between its passes -- lands
on that one expert. Production runs 32 experts per chip, and these cases spread that cost over the
experts the way the model does.

Every count is at or under the 320-token threshold both models ship, so the hybrid serves all of
them from its fused half, and moe_fused_swiglu is its like-for-like reference: the gap between the
two is what the union costs on top of the op it runs. That only holds if the swiglu op is driven as
TtRoutedExpert's two-op forward drives it -- region offsets, x read at each expert's offset, one
shared bf8 output and the fused compute config -- which is what the hybrid's fused half is built
from, and not as test_moe_fused_swiglu_perf.py drives it.

Needs a host-IOMMU runner, hence requires_host_iommu: on Blackhole the profiler's D2H socket uses
64-bit PCIe addressing, which requires IOMMU with no hugepage fallback (realtime_profiler_manager.cpp).
"""

import pytest
import torch

import ttnn
from tt_py_test_utils_common.utility_functions import is_blackhole, skip_with_llk_assert, skip_with_watcher
from models.demos.deepseek_v3_d_p.reference.tt.moe.expert import TorchExpert
from models.demos.deepseek_v3_d_p.tt.moe.tt_routed_expert import (
    TtRoutedExpert,
    _fused_compute_config,
    routed_expert_weight_memory_config,
)
from models.demos.deepseek_v3_d_p.utils.smbus_telemetry import is_p150
from tests.ttnn.profiling.realtime_profiler_utils import assert_op_duration_merged, require_realtime_profiler
from tests.ttnn.utils_for_testing import comp_pcc
from tests.ttnn.nightly.unit_tests.operations.experimental.deepseek_prefill import ci_pruning
from tests.ttnn.nightly.unit_tests.operations.experimental.deepseek_prefill.test_single_routed_expert import (
    _ISL_ALLOCATED_TOKENS,
    _ISL_EXHAUSTIVE_MODELS,
    SINGLE_EXPERT_MODELS,
)

pytestmark = pytest.mark.uncollect_if(pred=ci_pruning.hybrid_op)

# RT records carry kernel sources, not an OP CODE, so identify each op by its kernel directory.
_OP_KERNEL_DIRS = {
    "unified": "/unified_routed_expert_ffn/",
    "swiglu": "/moe_fused_swiglu/",
    "hybrid": "/hybrid_routed_expert_ffn/",
}

# Median of this many dispatches, as in the single-expert gates.
_ITERS = 3

# Log every program in the profiled window -- use when recalibrating.
_VERBOSE = False

_MARGIN = 0.03

# Per-expert token counts, packed back to back from row 0 of the _ISL_ALLOCATED_TOKENS buffer.
#   8x32 and 32x32 differ only in expert count, so their difference is the per-expert cost and the
#   rest is per dispatch.
#   32xmixed is the production expert count over the spread of counts the fused half serves.
_CASES: dict[str, list[int]] = {
    "8x32": [32] * 8,
    "32x32": [32] * 32,
    "32xmixed": [32, 64, 96, 128, 160, 192, 256, 320] * 4,
}

# Device duration in ns per (op, model, case), x_rm layout, weights DRAM-interleaved: the midpoint of
# three sweeps on a BH p150b (2026-10-23), each case a median of _ITERS dispatches, cross-sweep
# spread at most 0.6%. Recalibrate on the perf runner (DDR-speed dependent): each case logs an
# "RT-CAL" line in this dict's format.
_EXPECTED_NS: dict[tuple[str, str, str], int] = {
    ("unified", "kimi_k2_7", "8x32"): 927_263,
    ("unified", "kimi_k2_7", "32x32"): 3_614_198,
    ("unified", "kimi_k2_7", "32xmixed"): 3_712_893,
    ("unified", "glm_53", "8x32"): 802_784,
    ("unified", "glm_53", "32x32"): 3_183_627,
    ("unified", "glm_53", "32xmixed"): 3_267_190,
    ("swiglu", "kimi_k2_7", "8x32"): 579_264,
    ("swiglu", "kimi_k2_7", "32x32"): 2_304_662,
    ("swiglu", "kimi_k2_7", "32xmixed"): 3_074_287,
    ("swiglu", "glm_53", "8x32"): 495_122,
    ("swiglu", "glm_53", "32x32"): 1_971_322,
    ("swiglu", "glm_53", "32xmixed"): 2_758_621,
    ("hybrid", "kimi_k2_7", "8x32"): 623_495,
    ("hybrid", "kimi_k2_7", "32x32"): 2_463_880,
    ("hybrid", "kimi_k2_7", "32xmixed"): 3_249_681,
    ("hybrid", "glm_53", "8x32"): 543_061,
    ("hybrid", "glm_53", "32x32"): 2_141_537,
    ("hybrid", "glm_53", "32xmixed"): 2_908_282,
}

# Same measurement and key with the weights DRAM ND-sharded, the placement TtRoutedExpert picks on
# Blackhole by default.
_NDSHARD_EXPECTED_NS: dict[tuple[str, str, str], int] = {
    ("unified", "kimi_k2_7", "8x32"): 732_453,
    ("unified", "kimi_k2_7", "32x32"): 2_907_988,
    ("unified", "kimi_k2_7", "32xmixed"): 3_130_069,
    ("unified", "glm_53", "8x32"): 637_844,
    ("unified", "glm_53", "32x32"): 2_574_734,
    ("unified", "glm_53", "32xmixed"): 2_740_784,
    ("swiglu", "kimi_k2_7", "8x32"): 491_906,
    ("swiglu", "kimi_k2_7", "32x32"): 1_959_636,
    ("swiglu", "kimi_k2_7", "32xmixed"): 2_734_973,
    ("swiglu", "glm_53", "8x32"): 430_887,
    ("swiglu", "glm_53", "32x32"): 1_717_338,
    ("swiglu", "glm_53", "32xmixed"): 2_429_721,
    ("hybrid", "kimi_k2_7", "8x32"): 540_548,
    ("hybrid", "kimi_k2_7", "32x32"): 2_135_313,
    ("hybrid", "kimi_k2_7", "32xmixed"): 2_792_500,
    ("hybrid", "glm_53", "8x32"): 456_733,
    ("hybrid", "glm_53", "32x32"): 1_796_511,
    ("hybrid", "glm_53", "32xmixed"): 2_518_752,
}


def _baseline_or_skip(table, key, label: str):
    """The case's baseline, or a skip when its table has no entry. Borrowing the other placement's
    number is not an option -- the two tables exist because the placements measure differently."""
    expected_ns = table.get(key)
    if expected_ns is None:
        pytest.skip(f"no baseline for {label}; add {key} to its table -- see the comment there")
    return expected_ns


def _perf_params():
    params = []
    for name, config, _extended in SINGLE_EXPERT_MODELS:
        if name not in _ISL_EXHAUSTIVE_MODELS:
            continue
        for case in _CASES:
            for op in _OP_KERNEL_DIRS:
                params.append(
                    pytest.param(
                        name,
                        case,
                        op,
                        config.ROUTED_EXPERT_HYBRID_TOKEN_THRESHOLD,
                        config.EMB_SIZE,
                        config.MOE_INTERMEDIATE_SIZE,
                        id=f"{name}-{case}-{op}-perf",
                    )
                )
    return params


def _build(device, emb_dim: int, hidden_dim: int, counts: list[int], threshold: int, weights_dram_sharded: bool):
    """Every op's forward over one shared load, built OUTSIDE the measured callable so the profiled
    window carries forwards rather than weight uploads.

    One weight set, copied per expert on the device: converting 32 experts on the host would
    dominate the run, and DRAM has no cache for identical contents to hit, so each expert still
    reads its own buffer. Each forward PCC-checks every expert's region on its own, so a case that
    gets fast by dropping an expert fails on correctness rather than passing the band."""
    torch.manual_seed(42)
    weights = {
        "gate_proj": torch.randn(hidden_dim, emb_dim, dtype=torch.float32) * 0.02,
        "up_proj": torch.randn(hidden_dim, emb_dim, dtype=torch.float32) * 0.02,
        "down_proj": torch.randn(emb_dim, hidden_dim, dtype=torch.float32) * 0.02,
    }
    offsets = [sum(counts[:e]) for e in range(len(counts))]
    total = sum(counts)
    assert total <= _ISL_ALLOCATED_TOKENS

    torch_input = torch.zeros(_ISL_ALLOCATED_TOKENS, emb_dim, dtype=torch.float32)
    torch_input[:total] = torch.randn(total, emb_dim, dtype=torch.float32)
    with torch.no_grad():
        # The FFN is row-wise and every expert has the same weights, so one pass grades them all.
        want = TorchExpert(emb_dim, hidden_dim, weights)(torch_input[:total])

    def idx_tensor(values):
        return ttnn.from_torch(
            torch.tensor(values, dtype=torch.int32), layout=ttnn.ROW_MAJOR_LAYOUT, device=device, dtype=ttnn.uint32
        )

    idx_tt = idx_tensor(list(range(len(counts))))
    counts_tt = idx_tensor(counts)
    offsets_tt = idx_tensor(offsets)

    # Interleaved here whatever the case's placement: an interleaved tensor clones, and the
    # ND-sharded copies are resharded from it below.
    tt_expert = TtRoutedExpert(
        mesh_device=device,
        experts_per_chip=1,
        global_expert_idx_table=idx_tensor([0]),
        emb_dim=emb_dim,
        hidden_dim=hidden_dim,
        max_tokens=_ISL_ALLOCATED_TOKENS,
        torch_weights=[weights],
        activations_dtype=ttnn.bfloat8_b,
        weights_dtype=ttnn.bfloat4_b,
        activation=ttnn.RoutedExpertActivation.Silu,
        weights_dram_nd_sharded=False,
    )

    def per_expert(base):
        if weights_dram_sharded:
            config = routed_expert_weight_memory_config(device, base.shape[-1], dram_nd_sharded=True)
            return [ttnn.to_memory_config(base, config) for _ in counts]
        return [base] + [ttnn.clone(base) for _ in counts[1:]]

    gates = per_expert(tt_expert.gate_projs[0])
    ups = per_expert(tt_expert.up_projs[0])
    downs = per_expert(tt_expert.down_projs[0])
    compute_config = tt_expert.compute_kernel_config
    activation = ttnn.RoutedExpertActivation.Silu

    tt_input = ttnn.from_torch(
        torch_input,
        mesh_mapper=ttnn.ReplicateTensorToMesh(device),
        layout=ttnn.ROW_MAJOR_LAYOUT,  # x_rm: the Blackhole fused-tilize production fast path
        device=device,
        dtype=ttnn.bfloat16,
    )

    # The arguments TtRoutedExpert's forward hands each op; see the module docstring for why the
    # swiglu op takes the two-op forward's rather than its own perf gate's.
    def unified():
        return ttnn.experimental.deepseek_prefill.unified_routed_expert_moe(
            tt_input,
            offsets_tt,
            counts_tt,
            idx_tt,
            gates,
            ups,
            downs,
            max_dispatched_tokens_per_expert=_ISL_ALLOCATED_TOKENS,
            compute_kernel_config=compute_config,
            activation=activation,
            min_active_tokens=0,
        )

    def swiglu():
        output = ttnn.empty(
            tt_input.shape,
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.experimental.deepseek_prefill.moe_fused_swiglu(
            tt_input,
            gates,
            ups,
            downs,
            counts_tt,
            idx_tt,
            input_m_tiles=_ISL_ALLOCATED_TOKENS // ttnn.TILE_SIZE,
            core_grid=ttnn.UNIFIED_ROUTED_EXPERT_CORE_GRID,
            compute_kernel_config=_fused_compute_config(compute_config),
            activation=activation,
            output=output,
            expert_region_offsets=offsets_tt,
            read_x_at_offset=True,
            max_active_tokens=threshold,
        )
        return output

    def hybrid():
        return ttnn.experimental.deepseek_prefill.hybrid_routed_expert_moe(
            tt_input,
            offsets_tt,
            counts_tt,
            idx_tt,
            gates,
            ups,
            downs,
            max_dispatched_tokens_per_expert=_ISL_ALLOCATED_TOKENS,
            hybrid_token_threshold=threshold,
            compute_kernel_config=compute_config,
            activation=activation,
        )

    def checked(forward, op: str):
        def run():
            got = ttnn.to_torch(forward(), mesh_composer=ttnn.ConcatMeshToTensor(device, dim=0))
            for e, (off, cnt) in enumerate(zip(offsets, counts)):
                rows = got[off : off + cnt]
                assert not torch.isnan(rows).any(), f"{op}: expert {e} output contains NaN"
                _, pcc = comp_pcc(want[off : off + cnt], rows)
                assert pcc >= 0.97, f"{op}: expert {e} (rows {off}..{off + cnt}) PCC {pcc:.6f}"

        return run

    # Construction dispatches device work of its own. Drain it here or its tail lands inside the
    # profiler window.
    ttnn.synchronize_device(device)
    return {
        "unified": checked(unified, "unified"),
        "swiglu": checked(swiglu, "swiglu"),
        "hybrid": checked(hybrid, "hybrid"),
    }


@pytest.mark.parametrize("model_name, case, op, threshold, emb_dim, hidden_dim", _perf_params())
@pytest.mark.parametrize("weights_dram_sharded", [False, True], ids=["w_interleaved", "w_ndshard"])
@pytest.mark.requires_host_iommu
@pytest.mark.skipif(not is_blackhole(), reason="the routed expert is Blackhole-only")
@pytest.mark.skipif(not is_p150(), reason="perf baselines are P150-specific; skip on any other board")
@skip_with_llk_assert("No need to verify LLK asserts for performance tests.")
@skip_with_watcher("Watcher perturbs kernel timing; perf checks are not meaningful with it enabled.")
def test_routed_expert_multi_expert_perf(
    device,
    model_name: str,
    case: str,
    op: str,
    threshold: int,
    emb_dim: int,
    hidden_dim: int,
    weights_dram_sharded: bool,
):
    require_realtime_profiler("routed expert multi-expert perf checks")
    counts = _CASES[case]
    assert all(c <= threshold for c in counts), f"{case} puts an expert above {model_name}'s threshold {threshold}"

    table = _NDSHARD_EXPECTED_NS if weights_dram_sharded else _EXPECTED_NS
    placement = "w_ndshard" if weights_dram_sharded else "w_interleaved"
    label = f'{placement} ("{op}", "{model_name}", "{case}")'
    expected_ns = _baseline_or_skip(table, (op, model_name, case), label)

    forwards = _build(device, emb_dim, hidden_dim, counts, threshold, weights_dram_sharded)
    assert_op_duration_merged(
        device,
        forwards[op],
        _OP_KERNEL_DIRS[op],
        expected_ns=expected_ns,
        margin=_MARGIN,
        label=label,
        iters=_ITERS,
        verbose=_VERBOSE,
    )
