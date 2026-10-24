# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Device perf gate for UnifiedRoutedExpertFfn: per-case device duration over the kimi/glm ISL sweep
on the x_rm production path, measured with the real-time program profiler.

Needs a host-IOMMU runner, hence requires_host_iommu: on Blackhole the profiler's D2H socket uses
64-bit PCIe addressing, which requires IOMMU with no hugepage fallback (realtime_profiler_manager.cpp).
"""

import pytest

import ttnn
from tt_py_test_utils_common.utility_functions import is_blackhole, skip_with_llk_assert, skip_with_watcher
from models.demos.deepseek_v3_d_p.utils.smbus_telemetry import is_p150
from tests.ttnn.profiling.realtime_profiler_utils import assert_op_duration_merged, require_realtime_profiler
from models.demos.deepseek_v3_d_p.reference.kimi_k3_config import KimiK3Config
from tests.ttnn.nightly.unit_tests.operations.experimental.deepseek_prefill.test_single_routed_expert import (
    _ISL_ALLOCATED_TOKENS,
    _ISL_EXHAUSTIVE_MODELS,
    _ISL_PERF_SWEEP,
    SINGLE_EXPERT_MODELS,
    run_single_routed_expert,
)

# RT records carry kernel sources, not an OP CODE, so identify the op by its kernel directory.
_OP_KERNEL_DIR = "/unified_routed_expert_ffn/"

# Median of this many dispatches. A single dispatch throws occasional >3% flyers, so per-case margins
# turn into whack-a-mole; the median collapses them and keeps _MARGIN meaningful above 256.
_ITERS = 3

# Log every program in the profiled window, per iteration — use when recalibrating.
_VERBOSE = False

_MARGIN = 0.03
# active=0 is ~4us of launch overhead; margin=1.0 zeroes the floor and leaves only a ceiling.
_CEILING_ONLY = 1.0
# At 128/256 of 5120 allocated rows the kernel skips most chunks, so fixed overhead dominates and the
# median still has a long right tail (glm-256 spans 192-208us over 15 runs); >=512 holds inside 1.5%.
_LOW_ISL_MARGIN = 0.08

# Device duration in ns per (model, active), x_rm layout: the midpoint of three sweeps on a BH p150b
# (2026-10-23), each case a median of _ITERS dispatches. Recalibrate on the perf runner (DDR-speed
# dependent): each case logs an "RT-CAL" line in this dict's format, so one run regenerates the table.
_EXPECTED_NS: dict[tuple[str, int], int] = {
    ("kimi_k2_7", 0): 1_966,
    ("kimi_k2_7", 32): 124_359,
    ("kimi_k2_7", 64): 124_565,
    ("kimi_k2_7", 96): 125_633,
    ("kimi_k2_7", 128): 126_939,
    ("kimi_k2_7", 256): 130_984,
    ("kimi_k2_7", 512): 149_362,
    ("kimi_k2_7", 768): 199_282,
    ("kimi_k2_7", 1024): 260_971,
    ("kimi_k2_7", 2048): 507_555,
    ("kimi_k2_7", 4096): 1_007_220,
    ("kimi_k2_7", 5120): 1_252_930,
    ("glm_53", 0): 1_945,
    ("glm_53", 32): 102_729,
    ("glm_53", 64): 103_136,
    ("glm_53", 96): 104_414,
    ("glm_53", 128): 105_323,
    ("glm_53", 256): 109_037,
    ("glm_53", 512): 132_175,
    ("glm_53", 768): 175_030,
    ("glm_53", 1024): 229_006,
    ("glm_53", 2048): 444_156,
    ("glm_53", 4096): 877_486,
    ("glm_53", 5120): 1_092_959,
}

# Same measurement and key as _EXPECTED_NS, with the weights DRAM ND-sharded: a core fetches its
# whole K-row weight slice in ONE NoC request instead of one per tile. Its own table because the gain
# lands where this op is weight-read bound -- 1.21-1.29x at 128 active tokens, 1.19-1.26x at 256, 1.08x
# at 512, flat from 1024 up -- so the interleaved bands reject the low-ISL cases outright. Same
# calibration as above.
_NDSHARD_EXPECTED_NS: dict[tuple[str, int], int] = {
    ("kimi_k2_7", 0): 1_894,
    ("kimi_k2_7", 32): 98_741,
    ("kimi_k2_7", 64): 97_564,
    ("kimi_k2_7", 96): 98_584,
    ("kimi_k2_7", 128): 98_536,
    ("kimi_k2_7", 256): 103_627,
    ("kimi_k2_7", 512): 137_798,
    ("kimi_k2_7", 768): 200_584,
    ("kimi_k2_7", 1024): 264_731,
    ("kimi_k2_7", 2048): 509_620,
    ("kimi_k2_7", 4096): 1_001_749,
    ("kimi_k2_7", 5120): 1_248_580,
    ("glm_53", 0): 1_882,
    ("glm_53", 32): 86_813,
    ("glm_53", 64): 86_176,
    ("glm_53", 96): 86_820,
    ("glm_53", 128): 86_805,
    ("glm_53", 256): 91_855,
    ("glm_53", 512): 122_194,
    ("glm_53", 768): 173_301,
    ("glm_53", 1024): 232_407,
    ("glm_53", 2048): 447_034,
    ("glm_53", 4096): 877_213,
    ("glm_53", 5120): 1_091_630,
}


# Kimi K3 runs SiTU-GLU at the post-projection dims, so its K axis is ROUTED_EXPERT_HIDDEN_SIZE and
# it cannot be driven from SINGLE_EXPERT_MODELS (which reads config.EMB_SIZE). Same calibration as
# above.
#
# K3's per_core_N_d is 11 -- PRIME, so the down matmul has no divisor reaching a 3-tile subblock and
# rests on the ragged-tail path to avoid running 1x1, which costs 1.20-1.24x from 512 up.
_K3_SITU_EXPECTED_NS: dict[int, int] = {
    0: 1_939,
    32: 106_453,
    64: 106_530,
    96: 107_186,
    128: 107_728,
    256: 110_347,
    512: 148_823,
    768: 216_073,
    1024: 274_782,
    2048: 541_065,
    4096: 1_070_632,
    5120: 1_334_741,
}

# Kimi K3 SiTU-GLU counterpart of _NDSHARD_EXPECTED_NS: 1.12x at 128, 1.13x at 256, flat from 1024.
_K3_SITU_NDSHARD_EXPECTED_NS: dict[int, int] = {
    0: 1_894,
    32: 96_487,
    64: 96_415,
    96: 96_821,
    128: 96_426,
    256: 97_405,
    512: 145_427,
    768: 215_173,
    1024: 279_044,
    2048: 544_120,
    4096: 1_076_451,
    5120: 1_343_579,
}

# K3's DRAM weight read is 18.58 MB against a ~117 us floor, so its knee sits later than kimi_k2_7's
# or glm_53's: 512 is the first case where compute starts to cover the read, and 768 still spread
# 3.7-4.1% across three sweeps, which the usual 3% would not hold. Both keep _LOW_ISL_MARGIN;
# everything past 768 holds inside 3%.
_K3_KNEE_TOKENS = 768


def _baseline_or_skip(table, key, label: str):
    """The case's baseline, or a skip when its table has no entry. Borrowing the other placement's
    number is not an option -- the two tables exist because the placements measure differently."""
    expected_ns = table.get(key)
    if expected_ns is None:
        pytest.skip(f"no baseline for {label}; add {key} to its table -- see the comment there")
    return expected_ns


def _margin_for(active: int) -> float:
    return _CEILING_ONLY if active == 0 else _LOW_ISL_MARGIN if active <= 256 else _MARGIN


def _margin_for_k3(active: int) -> float:
    return _CEILING_ONLY if active == 0 else _LOW_ISL_MARGIN if active <= _K3_KNEE_TOKENS else _MARGIN


def _perf_params():
    """Dims and margin per (model, active) over the exhaustive ISL sweep, dims from
    SINGLE_EXPERT_MODELS. The baseline is not carried here -- it is keyed on the weight placement
    too, so the test body picks the table. No extended_model mark: the markers below already scope
    where these run."""
    params = []
    for name, config, _extended in SINGLE_EXPERT_MODELS:
        if name not in _ISL_EXHAUSTIVE_MODELS:
            continue
        for active in _ISL_PERF_SWEEP:
            params.append(
                pytest.param(
                    name,
                    active,
                    config.EMB_SIZE,
                    config.MOE_INTERMEDIATE_SIZE,
                    _margin_for(active),
                    # "-perf" keeps ids collision-free under -k: "512-perf" is not in "5120-perf".
                    id=f"{name}-isl-{active}-perf",
                )
            )
    return params


@pytest.mark.parametrize("model_name, active_tokens, emb_dim, hidden_dim, margin", _perf_params())
@pytest.mark.parametrize("weights_dram_sharded", [False, True], ids=["w_interleaved", "w_ndshard"])
@pytest.mark.requires_host_iommu
@pytest.mark.skipif(not is_blackhole(), reason="the measured fused FFN path is Blackhole-only")
@pytest.mark.skipif(not is_p150(), reason="perf baselines are P150-specific; skip on any other board")
@skip_with_llk_assert("No need to verify LLK asserts for performance tests.")
@skip_with_watcher("Watcher perturbs kernel timing; perf checks are not meaningful with it enabled.")
def test_single_routed_expert_perf(
    device,
    model_name: str,
    active_tokens: int,
    emb_dim: int,
    hidden_dim: int,
    margin: float,
    weights_dram_sharded: bool,
):
    require_realtime_profiler("single routed expert perf checks")

    table = _NDSHARD_EXPECTED_NS if weights_dram_sharded else _EXPECTED_NS
    placement = "w_ndshard" if weights_dram_sharded else "w_interleaved"
    label = f'{placement} ("{model_name}", {active_tokens})'
    expected_ns = _baseline_or_skip(table, (model_name, active_tokens), label)

    # run_single_routed_expert also PCC-checks, so a case that gets fast by computing the wrong thing
    # fails on correctness rather than passing the band.
    assert_op_duration_merged(
        device,
        lambda: run_single_routed_expert(
            device,
            _ISL_ALLOCATED_TOKENS,
            emb_dim,
            hidden_dim,
            active_tokens=active_tokens,
            x_row_major=True,  # x_rm: the Blackhole fused-tilize production fast path
            weights_dram_sharded=weights_dram_sharded,
        ),
        _OP_KERNEL_DIR,
        expected_ns=expected_ns,
        margin=margin,
        label=label,
        iters=_ITERS,
        verbose=_VERBOSE,
    )


@pytest.mark.parametrize(
    "active_tokens, margin",
    [pytest.param(active, _margin_for_k3(active), id=f"k3-isl-{active}-perf") for active in _ISL_PERF_SWEEP],
)
@pytest.mark.parametrize("weights_dram_sharded", [False, True], ids=["w_interleaved", "w_ndshard"])
@pytest.mark.requires_host_iommu
@pytest.mark.skipif(not is_blackhole(), reason="SiTU-GLU is Blackhole-only")
@pytest.mark.skipif(not is_p150(), reason="perf baselines are P150-specific; skip on any other board")
@skip_with_llk_assert("No need to verify LLK asserts for performance tests.")
@skip_with_watcher("Watcher perturbs kernel timing; perf checks are not meaningful with it enabled.")
def test_single_routed_expert_k3_perf(device, active_tokens: int, margin: float, weights_dram_sharded: bool):
    """Kimi K3 routed expert (SiTU-GLU) device duration over the same ISL sweep as above."""
    require_realtime_profiler("single routed expert perf checks")

    table = _K3_SITU_NDSHARD_EXPECTED_NS if weights_dram_sharded else _K3_SITU_EXPECTED_NS
    placement = "w_ndshard" if weights_dram_sharded else "w_interleaved"
    label = f'{placement} ("kimi_k3", {active_tokens})'
    expected_ns = _baseline_or_skip(table, active_tokens, label)

    assert_op_duration_merged(
        device,
        lambda: run_single_routed_expert(
            device,
            _ISL_ALLOCATED_TOKENS,
            KimiK3Config.ROUTED_EXPERT_HIDDEN_SIZE,
            KimiK3Config.MOE_INTERMEDIATE_SIZE,
            active_tokens=active_tokens,
            x_row_major=True,
            activation=ttnn.RoutedExpertActivation.SituGlu,
            weights_dram_sharded=weights_dram_sharded,
        ),
        _OP_KERNEL_DIR,
        expected_ns=expected_ns,
        margin=margin,
        label=label,
        iters=_ITERS,
        verbose=_VERBOSE,
    )
