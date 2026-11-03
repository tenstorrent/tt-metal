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
    _ISL_EXHAUSTIVE_SWEEP,
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

# Device duration in ns per (model, active), x_rm layout: ONE sweep on a BH p150b (2026-09-15),
# each case a median of _ITERS dispatches. Recalibrate on the perf runner (DDR-speed dependent):
# each case logs an "RT-CAL" line in this dict's format, so one run regenerates the table.
_EXPECTED_NS: dict[tuple[str, int], int] = {
    ("kimi_k2_7", 0): 3_128,
    ("kimi_k2_7", 128): 123_789,
    ("kimi_k2_7", 256): 127_608,
    ("kimi_k2_7", 512): 156_199,
    ("kimi_k2_7", 768): 210_454,
    ("kimi_k2_7", 1024): 269_360,
    ("kimi_k2_7", 2048): 531_826,
    ("kimi_k2_7", 4096): 1_099_947,
    ("kimi_k2_7", 5120): 1_299_384,
    ("glm_53", 0): 3_093,
    ("glm_53", 128): 112_470,
    ("glm_53", 256): 116_603,
    ("glm_53", 512): 143_111,
    ("glm_53", 768): 188_893,
    ("glm_53", 1024): 239_785,
    ("glm_53", 2048): 458_928,
    ("glm_53", 4096): 905_267,
    ("glm_53", 5120): 1_133_054,
}

# Same measurement and key as _EXPECTED_NS, with the weights DRAM ND-sharded: a core fetches its
# whole K-row weight slice in ONE NoC request instead of one per tile. Its own table because the gain
# lands where this op is weight-read bound -- 1.15x at 128 active tokens, 1.10x at 256, 1.03x at 512, flat from 1024 up -- so the interleaved
# bands reject the low-ISL cases outright.
#
# ONE sweep on a BH p150b (2026-09-15), each case a median of _ITERS dispatches. The interleaved
# table above is the midpoint of THREE sweeps; these carry no cross-sweep spread, so the low-ISL
# entries are the thin ones -- which is what _LOW_ISL_MARGIN is there to absorb.
_NDSHARD_EXPECTED_NS: dict[tuple[str, int], int] = {
    ("kimi_k2_7", 0): 3_028,
    ("kimi_k2_7", 128): 103_905,
    ("kimi_k2_7", 256): 111_867,
    ("kimi_k2_7", 512): 146_189,
    ("kimi_k2_7", 768): 203_973,
    ("kimi_k2_7", 1024): 268_324,
    ("kimi_k2_7", 2048): 531_273,
    ("kimi_k2_7", 4096): 1_078_113,
    ("kimi_k2_7", 5120): 1_295_258,
    ("glm_53", 0): 3_076,
    ("glm_53", 128): 96_160,
    ("glm_53", 256): 104_446,
    ("glm_53", 512): 132_152,
    ("glm_53", 768): 186_862,
    ("glm_53", 1024): 234_693,
    ("glm_53", 2048): 459_744,
    ("glm_53", 4096): 906_926,
    ("glm_53", 5120): 1_133_381,
}


# Kimi K3 runs SiTU-GLU at the post-projection dims, so its K axis is ROUTED_EXPERT_HIDDEN_SIZE and
# it cannot be driven from SINGLE_EXPERT_MODELS (which reads config.EMB_SIZE). x_rm layout on a BH
# p150b (2026-09-15), each case a median of _ITERS dispatches.
#
# K3's per_core_N_d is 11 -- PRIME, so the down matmul has no divisor reaching a 3-tile subblock and
# rests on the ragged-tail path to avoid running 1x1, which costs 1.20-1.24x from 512 up. One sweep,
# not the midpoint of three.
_K3_SITU_EXPECTED_NS: dict[int, int] = {
    0: 3_099,
    128: 108_502,
    256: 109_698,
    512: 140_293,
    768: 191_000,
    1024: 242_771,
    2048: 475_196,
    4096: 936_373,
    5120: 1_170_397,
}

# Kimi K3 SiTU-GLU counterpart of _NDSHARD_EXPECTED_NS: 1.16x at 128, 1.14x at 256, flat from 1024.
_K3_SITU_NDSHARD_EXPECTED_NS: dict[int, int] = {
    0: 3_106,
    128: 93_428,
    256: 96_435,
    512: 132_767,
    768: 191_000,
    1024: 241_474,
    2048: 474_603,
    4096: 936_801,
    5120: 1_168_421,
}

# K3's DRAM weight read is 18.58 MB against a ~117 us floor, so its knee sits a token count later
# than kimi_k2_7's or glm_53's: 512 is the first case where compute starts to cover the read, and it
# keeps _LOW_ISL_MARGIN because the long right tail that margin exists for is a run-to-run effect,
# not one three consecutive sweeps expose. Everything past the knee holds inside the usual 3%.
_K3_KNEE_TOKENS = 512


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
        for active in _ISL_EXHAUSTIVE_SWEEP:
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
    [pytest.param(active, _margin_for_k3(active), id=f"k3-isl-{active}-perf") for active in _ISL_EXHAUSTIVE_SWEEP],
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
