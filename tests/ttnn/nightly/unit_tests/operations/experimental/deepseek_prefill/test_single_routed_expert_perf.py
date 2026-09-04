# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Device perf gate for UnifiedRoutedExpertFfn: per-case device duration over the kimi/glm ISL sweep
on the x_rm production path, measured with the real-time program profiler.

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
    _ISL_ALLOCATED_TOKENS,
    _routed_expert_k,
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

# Device duration in ns per (model, active), x_rm layout: midpoint of min/max over 3 sweeps
# on a BH p150b (2026-09-04), each sweep itself a median of 3 dispatches. Cross-sweep
# spread was <=0.5% on every case except active=0 (1.7%, ~50 ns, ceiling-only band).
# Recalibrate on the perf runner (DDR-speed dependent): each case logs an
# "RT-CAL" line in this dict's format, so one run regenerates the table.
_EXPECTED_NS: dict[tuple[str, int], int] = {
    ("kimi_k26", 0): 3_080,
    ("kimi_k26", 128): 123_946,
    ("kimi_k26", 256): 127_376,
    ("kimi_k26", 512): 162_862,
    ("kimi_k26", 1024): 294_228,
    ("kimi_k26", 2048): 579_661,
    ("kimi_k26", 4096): 1_150_627,
    ("kimi_k26", 5120): 1_437_053,
    ("glm_51", 0): 3_133,
    ("glm_51", 128): 109_978,
    ("glm_51", 256): 113_223,
    ("glm_51", 512): 145_161,
    ("glm_51", 1024): 257_773,
    ("glm_51", 2048): 508_241,
    ("glm_51", 4096): 1_009_661,
    ("glm_51", 5120): 1_260_985,
}


# Kimi K3 keeps its own baseline table: SiTU-GLU is calibrated separately from the SiLU models, and
# its knee sits at a different token count (see _K3_KNEE_TOKENS). Same measurement as _EXPECTED_NS:
# midpoint of min/max over 3 sweeps, each a median of 3 dispatches, x_rm layout, on a BH p150b
# (2026-09-04). Flat to ~256 tokens (the op sits on its DRAM weight-read
# floor), linear in tokens past that.
_K3_SITU_EXPECTED_NS: dict[int, int] = {
    0: 3_092,
    128: 117_154,
    256: 118_893,
    512: 179_315,
    1024: 343_033,
    2048: 671_470,
    4096: 1_325_686,
    5120: 1_658_646,
}

# K3's DRAM weight read is 18.58 MB against a ~117 us floor, so its knee sits a token count later
# than kimi_k26's or glm_51's: 512 is the first case where compute starts to cover the read, and it
# inherits the long right tail _LOW_ISL_MARGIN exists for (2% cross-sweep spread at 512 against
# 0.1% at 128 and 256). Everything past the knee holds inside the usual 3%.
_K3_KNEE_TOKENS = 512


def _margin_for(active: int) -> float:
    return _CEILING_ONLY if active == 0 else _LOW_ISL_MARGIN if active <= 256 else _MARGIN


def _margin_for_k3(active: int) -> float:
    return _CEILING_ONLY if active == 0 else _LOW_ISL_MARGIN if active <= _K3_KNEE_TOKENS else _MARGIN


def _perf_params():
    """Baseline and margin per (model, active), dims from SINGLE_EXPERT_MODELS. No extended_model
    mark: the markers below already scope where these run.

    _EXPECTED_NS is the source of truth for what this gate covers -- a case with no baseline has
    nothing to assert against, and the worker sweep's model and ISL lists grow independently of it."""
    dims = {name: (config, activation) for name, config, _extended, activation in SINGLE_EXPERT_MODELS}
    params = []
    for name, active in sorted(_EXPECTED_NS):
        config, activation = dims[name]
        params.append(
            pytest.param(
                name,
                active,
                _routed_expert_k(config),
                config.MOE_INTERMEDIATE_SIZE,
                activation,
                _EXPECTED_NS[(name, active)],
                _margin_for(active),
                # "-perf" keeps ids collision-free under -k: "512-perf" is not in "5120-perf".
                id=f"{name}-isl-{active}-perf",
            )
        )
    return params


@pytest.mark.parametrize(
    "model_name, active_tokens, emb_dim, hidden_dim, activation, expected_ns, margin", _perf_params()
)
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
    activation,
    expected_ns: int,
    margin: float,
):
    require_realtime_profiler("single routed expert perf checks")

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
            activation=activation,
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
        pytest.param(active, _K3_SITU_EXPECTED_NS[active], _margin_for_k3(active), id=f"k3-isl-{active}-perf")
        for active in sorted(_K3_SITU_EXPECTED_NS)
    ],
)
@pytest.mark.requires_host_iommu
@pytest.mark.skipif(not is_blackhole(), reason="SiTU-GLU is Blackhole-only")
@pytest.mark.skipif(not is_p150(), reason="perf baselines are P150-specific; skip on any other board")
@skip_with_llk_assert("No need to verify LLK asserts for performance tests.")
@skip_with_watcher("Watcher perturbs kernel timing; perf checks are not meaningful with it enabled.")
def test_single_routed_expert_k3_perf(device, active_tokens: int, expected_ns: int, margin: float):
    """Kimi K3 routed expert (SiTU-GLU) device duration over the same ISL sweep as above."""
    require_realtime_profiler("single routed expert perf checks")

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
        ),
        _OP_KERNEL_DIR,
        expected_ns=expected_ns,
        margin=margin,
        label=f'("kimi_k3", {active_tokens})',
        iters=_ITERS,
        verbose=_VERBOSE,
    )
