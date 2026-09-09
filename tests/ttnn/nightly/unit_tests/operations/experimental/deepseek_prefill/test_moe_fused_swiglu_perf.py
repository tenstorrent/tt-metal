# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Device perf gate for moe_fused_swiglu: per-case device duration over the kimi/glm ISL sweep on
the x_rm production path, measured with the real-time program profiler.

The op counterpart to test_single_routed_expert_perf.py, which gates the composite over the same
sweep. Both share one harness, so a regression in either op is attributable to that op.

Needs a host-IOMMU runner, hence requires_host_iommu: on Blackhole the profiler's D2H socket uses
64-bit PCIe addressing, which requires IOMMU with no hugepage fallback (realtime_profiler_manager.cpp).
"""

import pytest

import ttnn
from models.common.utility_functions import is_blackhole, skip_with_llk_assert, skip_with_watcher
from models.demos.deepseek_v3_d_p.utils.smbus_telemetry import is_p150
from tests.ttnn.profiling.realtime_profiler_utils import assert_op_duration_merged, require_realtime_profiler
from models.demos.deepseek_v3_d_p.reference.kimi_k3_config import KimiK3Config
from tests.ttnn.nightly.unit_tests.operations.experimental.deepseek_prefill.test_single_routed_expert import (
    SINGLE_EXPERT_MODELS,
)
from tests.ttnn.nightly.unit_tests.operations.experimental.deepseek_prefill.test_moe_fused_swiglu import (
    _ISL_ALLOCATED_TOKENS,
    _ISL_EXHAUSTIVE_MODELS,
    _ISL_EXHAUSTIVE_SWEEP,
    run_moe_fused_swiglu,
)

# RT records carry kernel sources, not an OP CODE, so identify the op by its kernel directory.
_OP_KERNEL_DIR = "/moe_fused_swiglu/"

# Median of this many dispatches. A single dispatch throws occasional flyers, so per-case margins
# turn into whack-a-mole; the median collapses them.
_ITERS = 3

# Log every program in the profiled window, per iteration — use when recalibrating.
_VERBOSE = False

_MARGIN = 0.03
# active=0 is launch overhead only; margin=1.0 zeroes the floor and leaves only a ceiling.
_CEILING_ONLY = 1.0
# Below the knee the kernel skips most chunks, so fixed overhead dominates and the median keeps a
# long right tail. 512 is inside the knee rather than past it: it is the first count where compute
# starts to cover the DRAM weight read. Everything past it holds inside 1% cross-sweep, while 256
# spreads 3-5%, which is what _LOW_ISL_MARGIN covers.
_LOW_ISL_MARGIN = 0.08
_KNEE_TOKENS = 512

# Device duration in ns per (model, active), x_rm layout, 11x8 grid: per-cell MEDIAN OVER THREE
# sweeps on a BH p150b (2026-09-08). Recalibrate on the perf runner (DDR-speed dependent): each case
# logs an "RT-CAL" line in this dict's format, so one run regenerates the table -- but centre it over
# several sweeps rather than copying one, because ISL 256 carries a 3-5% cross-sweep spread.
#
# Re-cut for bf16 gate/up accumulators. The cost climbs with token count -- +5% at 128 to +24% at
# 5120 -- because those two accumulators went from bfp8 to 2 B/value, so the column all-to-all moves
# ~1.9x the payload and the penalty is per-token rather than fixed. It also moved every crossover
# against the composite one sweep step earlier; the model configs' hybrid thresholds follow.
_EXPECTED_NS: dict[tuple[str, int], int] = {
    ("kimi_k2_7", 0): 2_786,
    ("kimi_k2_7", 128): 99_784,
    ("kimi_k2_7", 256): 128_139,
    ("kimi_k2_7", 512): 228_098,
    ("kimi_k2_7", 1024): 413_117,
    ("kimi_k2_7", 2048): 782_755,
    ("kimi_k2_7", 4096): 1_520_810,
    ("kimi_k2_7", 5120): 1_892_193,
    ("glm_51", 0): 2_744,
    ("glm_51", 128): 89_676,
    ("glm_51", 256): 116_975,
    ("glm_51", 512): 206_125,
    ("glm_51", 1024): 377_474,
    ("glm_51", 2048): 718_394,
    ("glm_51", 4096): 1_404_145,
    ("glm_51", 5120): 1_747_664,
}

# Kimi K3 runs SiTU-GLU at the post-projection dims, so its K axis is ROUTED_EXPERT_HIDDEN_SIZE and
# it cannot be driven from SINGLE_EXPERT_MODELS (which reads config.EMB_SIZE). Same measurement.
_K3_SITU_EXPECTED_NS: dict[int, int] = {
    0: 2_764,
    128: 89_949,
    256: 126_833,
    512: 231_401,
    1024: 435_460,
    2048: 844_275,
    4096: 1_661_876,
    5120: 2_068_973,
}


def _margin_for(active: int) -> float:
    return _CEILING_ONLY if active == 0 else _LOW_ISL_MARGIN if active <= _KNEE_TOKENS else _MARGIN


def _perf_params():
    """Baseline and margin per (model, active) over the exhaustive ISL sweep, dims from
    SINGLE_EXPERT_MODELS. No extended_model mark: the markers below already scope where these run."""
    params = []
    for name, config, _extended in SINGLE_EXPERT_MODELS:
        if name not in _ISL_EXHAUSTIVE_MODELS:
            continue
        for active in _ISL_EXHAUSTIVE_SWEEP:
            params.append(
                pytest.param(
                    name,
                    active,
                    config.EMB_SIZE,
                    config.MOE_INTERMEDIATE_SIZE,
                    _EXPECTED_NS[(name, active)],
                    _margin_for(active),
                    # "-perf" keeps ids collision-free under -k: "512-perf" is not in "5120-perf".
                    id=f"{name}-isl-{active}-perf",
                )
            )
    return params


@pytest.mark.parametrize("model_name, active_tokens, emb_dim, hidden_dim, expected_ns, margin", _perf_params())
@pytest.mark.requires_host_iommu
@pytest.mark.skipif(not is_blackhole(), reason="moe_fused_swiglu is Blackhole-only")
@pytest.mark.skipif(not is_p150(), reason="perf baselines are P150-specific; skip on any other board")
@skip_with_llk_assert("No need to verify LLK asserts for performance tests.")
@skip_with_watcher("Watcher perturbs kernel timing; perf checks are not meaningful with it enabled.")
def test_moe_fused_swiglu_perf(
    device,
    model_name: str,
    active_tokens: int,
    emb_dim: int,
    hidden_dim: int,
    expected_ns: int,
    margin: float,
):
    require_realtime_profiler("moe_fused_swiglu perf checks")

    # run_moe_fused_swiglu also PCC-checks, so a case that gets fast by computing the wrong thing
    # fails on correctness rather than passing the band.
    assert_op_duration_merged(
        device,
        lambda: run_moe_fused_swiglu(
            device,
            _ISL_ALLOCATED_TOKENS,
            emb_dim,
            hidden_dim,
            active_tokens=active_tokens,
            x_row_major=True,  # x_rm: the Blackhole fused-tilize production fast path
        ),
        _OP_KERNEL_DIR,
        expected_ns=expected_ns,
        margin=margin,
        label=f'("{model_name}", {active_tokens})',
        iters=_ITERS,
        verbose=_VERBOSE,
    )


@pytest.mark.parametrize(
    "active_tokens, expected_ns, margin",
    [
        pytest.param(active, _K3_SITU_EXPECTED_NS[active], _margin_for(active), id=f"k3-isl-{active}-perf")
        for active in _ISL_EXHAUSTIVE_SWEEP
    ],
)
@pytest.mark.requires_host_iommu
@pytest.mark.skipif(not is_blackhole(), reason="SiTU-GLU is Blackhole-only")
@pytest.mark.skipif(not is_p150(), reason="perf baselines are P150-specific; skip on any other board")
@skip_with_llk_assert("No need to verify LLK asserts for performance tests.")
@skip_with_watcher("Watcher perturbs kernel timing; perf checks are not meaningful with it enabled.")
def test_moe_fused_swiglu_k3_perf(device, active_tokens: int, expected_ns: int, margin: float):
    """Kimi K3 (SiTU-GLU) device duration over the same ISL sweep as above."""
    require_realtime_profiler("moe_fused_swiglu perf checks")

    assert_op_duration_merged(
        device,
        lambda: run_moe_fused_swiglu(
            device,
            _ISL_ALLOCATED_TOKENS,
            KimiK3Config.ROUTED_EXPERT_HIDDEN_SIZE,
            KimiK3Config.MOE_INTERMEDIATE_SIZE,
            active_tokens=active_tokens,
            x_row_major=True,
            activation=ttnn.RoutedExpertActivation.SituGlu,
        ),
        _OP_KERNEL_DIR,
        expected_ns=expected_ns,
        margin=margin,
        label=f'("kimi_k3", {active_tokens})',
        iters=_ITERS,
        verbose=_VERBOSE,
    )
