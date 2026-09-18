# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Device perf gate for HybridRoutedExpertFfn: per-case device duration over the kimi/glm ISL
sweep on the x_rm production path, measured with the real-time program profiler.

The same cases as test_routed_expert_crossover_perf.py -- same models, same sweep, same weight
placements, same margins -- so the union op's number sits next to the best-of-both number the two
ops it replaces achieve on exactly the same case.

One expert over an _ISL_ALLOCATED_TOKENS region with `active_tokens` live rows, and the model's
own threshold decides which half serves it. The other half still launches on every core and
sweeps the counts, so each case carries the union's fixed cost -- which is the point: the op is a
net device-time regression against the ops it replaces, and it exists for the combine overlap
rather than for device time. A band here catches drift in the op; it says nothing about whether
the op is worth dispatching.

Needs a host-IOMMU runner, hence requires_host_iommu: on Blackhole the profiler's D2H socket uses
64-bit PCIe addressing, which requires IOMMU with no hugepage fallback (realtime_profiler_manager.cpp).
"""

from typing import Optional

import pytest

from models.common.utility_functions import is_blackhole, skip_with_llk_assert, skip_with_watcher
from models.demos.deepseek_v3_d_p.utils.smbus_telemetry import is_p150
from tests.ttnn.profiling.realtime_profiler_utils import assert_op_duration_merged, require_realtime_profiler
from tests.ttnn.nightly.unit_tests.operations.experimental.deepseek_prefill import ci_pruning
from tests.ttnn.nightly.unit_tests.operations.experimental.deepseek_prefill.test_single_routed_expert import (
    _ISL_ALLOCATED_TOKENS,
    _ISL_EXHAUSTIVE_MODELS,
    _ISL_EXHAUSTIVE_SWEEP,
    SINGLE_EXPERT_MODELS,
)
from tests.ttnn.nightly.unit_tests.operations.experimental.deepseek_prefill.test_hybrid_routed_expert import (
    run_hybrid_routed_expert,
)

pytestmark = pytest.mark.uncollect_if(pred=ci_pruning.no_production_counterpart)

# RT records carry kernel sources, not an OP CODE, so identify the op by its kernel directory.
_OP_KERNEL_DIR = "/hybrid_routed_expert_ffn/"

# Median of this many dispatches. A single dispatch throws occasional flyers, so per-case margins
# turn into whack-a-mole; the median collapses them.
_ITERS = 3

# Log every program in the profiled window -- use when recalibrating.
_VERBOSE = False

_MARGIN = 0.03
# active=0 is launch overhead only; margin=1.0 zeroes the floor and leaves only a ceiling.
_CEILING_ONLY = 1.0
# Below its knee the op skips most chunks, fixed overhead dominates and the median keeps a long
# right tail.
_LOW_ISL_MARGIN = 0.08
_KNEE_TOKENS = 512

# Device duration in ns per (model, active), x_rm layout: the MIDPOINT of three sweeps on a BH
# p150b (2026-09-18), each case a median of _ITERS dispatches. Three, not one, because the
# low counts carry real cross-sweep spread -- up to 4.0% at active=256 -- and a single sweep can
# land at either end of it, which no margin then holds. Recalibrate on the perf runner (it is
# DDR-speed dependent): each case logs an "RT-CAL" line in this dict's format.
#
# Every entry is above the best-of-both number the two ops this one replaces reach on the same
# case in test_routed_expert_crossover_perf.py -- by ~8-15 us across the sweep, and by 8.0 us at
# active=0, where no expert does any work at all and the whole difference is the second pass plus
# the grid barrier.
_EXPECTED_NS: dict[tuple[str, int], int] = {
    ("kimi_k2_7", 0): 11_039,
    ("kimi_k2_7", 128): 104_367,
    ("kimi_k2_7", 256): 126_942,
    ("kimi_k2_7", 512): 171_510,
    ("kimi_k2_7", 768): 235_003,
    ("kimi_k2_7", 1024): 304_004,
    ("kimi_k2_7", 2048): 590_696,
    ("kimi_k2_7", 4096): 1_166_144,
    ("kimi_k2_7", 5120): 1_453_411,
    ("glm_51", 0): 11_034,
    ("glm_51", 128): 93_162,
    ("glm_51", 256): 120_377,
    ("glm_51", 512): 152_494,
    ("glm_51", 768): 207_245,
    ("glm_51", 1024): 267_217,
    ("glm_51", 2048): 518_599,
    ("glm_51", 4096): 1_022_134,
    ("glm_51", 5120): 1_269_726,
}

# Same measurement and key with the weights DRAM ND-sharded. Its own table because the placement
# moves the op: a core fetches its whole K-row weight slice in one NoC request instead of one per
# tile, which the op is only bound by at the low counts.
_NDSHARD_EXPECTED_NS: dict[tuple[str, int], int] = {
    ("kimi_k2_7", 0): 11_123,
    ("kimi_k2_7", 128): 94_706,
    ("kimi_k2_7", 256): 120_486,
    ("kimi_k2_7", 512): 166_315,
    ("kimi_k2_7", 768): 234_897,
    ("kimi_k2_7", 1024): 304_249,
    ("kimi_k2_7", 2048): 591_246,
    ("kimi_k2_7", 4096): 1_167_110,
    ("kimi_k2_7", 5120): 1_459_315,
    ("glm_51", 0): 11_103,
    ("glm_51", 128): 85_575,
    ("glm_51", 256): 104_698,
    ("glm_51", 512): 148_123,
    ("glm_51", 768): 206_711,
    ("glm_51", 1024): 266_964,
    ("glm_51", 2048): 516_896,
    ("glm_51", 4096): 1_020_632,
    ("glm_51", 5120): 1_274_889,
}


def _threshold_of(config) -> Optional[int]:
    """The hybrid split the model ships, read as tt_prefill_block reads it."""
    return getattr(config, "ROUTED_EXPERT_HYBRID_TOKEN_THRESHOLD", None)


def _margin_for(active: int) -> float:
    return _CEILING_ONLY if active == 0 else _LOW_ISL_MARGIN if active <= _KNEE_TOKENS else _MARGIN


def _baseline_or_skip(table, key, label: str):
    """The case's baseline, or a skip when its table has no entry. Borrowing the other placement's
    number is not an option -- the two tables exist because the placements measure differently."""
    expected_ns = table.get(key)
    if expected_ns is None:
        pytest.skip(f"no baseline for {label}; add {key} to its table -- see the comment there")
    return expected_ns


def _perf_params():
    """Dims, threshold and margin per (model, active) over the exhaustive ISL sweep, from
    SINGLE_EXPERT_MODELS. The baseline is not carried here -- it is keyed on the weight placement
    too, so the test body picks the table."""
    params = []
    for name, config, _extended in SINGLE_EXPERT_MODELS:
        if name not in _ISL_EXHAUSTIVE_MODELS:
            continue
        threshold = _threshold_of(config)
        for active in _ISL_EXHAUSTIVE_SWEEP:
            params.append(
                pytest.param(
                    name,
                    active,
                    threshold,
                    config.EMB_SIZE,
                    config.MOE_INTERMEDIATE_SIZE,
                    _margin_for(active),
                    # "-perf" keeps ids collision-free under -k: "512-perf" is not in "5120-perf".
                    id=f"{name}-isl-{active}-perf",
                )
            )
    return params


@pytest.mark.parametrize("model_name, active_tokens, threshold, emb_dim, hidden_dim, margin", _perf_params())
@pytest.mark.parametrize("weights_dram_sharded", [False, True], ids=["w_interleaved", "w_ndshard"])
@pytest.mark.requires_host_iommu
@pytest.mark.skipif(not is_blackhole(), reason="the routed expert is Blackhole-only")
@pytest.mark.skipif(not is_p150(), reason="perf baselines are P150-specific; skip on any other board")
@skip_with_llk_assert("No need to verify LLK asserts for performance tests.")
@skip_with_watcher("Watcher perturbs kernel timing; perf checks are not meaningful with it enabled.")
def test_hybrid_routed_expert_perf(
    device,
    model_name: str,
    active_tokens: int,
    threshold: Optional[int],
    emb_dim: int,
    hidden_dim: int,
    margin: float,
    weights_dram_sharded: bool,
):
    require_realtime_profiler("hybrid routed expert perf checks")
    assert threshold is not None, f"{model_name} ships no hybrid threshold; this op needs one"

    table = _NDSHARD_EXPECTED_NS if weights_dram_sharded else _EXPECTED_NS
    placement = "w_ndshard" if weights_dram_sharded else "w_interleaved"
    label = f'{placement} ("{model_name}", {active_tokens})'
    expected_ns = _baseline_or_skip(table, (model_name, active_tokens), label)

    # run_hybrid_routed_expert also PCC-checks, so a case that gets fast by computing the wrong
    # thing fails on correctness rather than passing the band.
    assert_op_duration_merged(
        device,
        lambda: run_hybrid_routed_expert(
            device,
            _ISL_ALLOCATED_TOKENS,
            emb_dim,
            hidden_dim,
            active_tokens=active_tokens,
            x_row_major=True,  # x_rm: the Blackhole fused-tilize production fast path
            weights_dram_sharded=weights_dram_sharded,
            threshold=threshold,
        ),
        _OP_KERNEL_DIR,
        expected_ns=expected_ns,
        margin=margin,
        label=label,
        iters=_ITERS,
        verbose=_VERBOSE,
    )
