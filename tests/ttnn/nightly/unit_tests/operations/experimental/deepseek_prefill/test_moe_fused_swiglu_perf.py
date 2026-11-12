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
from tt_py_test_utils_common.utility_functions import is_blackhole, skip_with_llk_assert, skip_with_watcher
from models.demos.deepseek_v3_d_p.utils.smbus_telemetry import is_p150
from tests.ttnn.profiling.realtime_profiler_utils import assert_op_duration_merged, require_realtime_profiler
from models.demos.deepseek_v3_d_p.reference.kimi_k3_config import KimiK3Config
from tests.ttnn.nightly.unit_tests.operations.experimental.deepseek_prefill.test_single_routed_expert import (
    _ISL_PERF_SWEEP,
    SINGLE_EXPERT_MODELS,
)
from tests.ttnn.nightly.unit_tests.operations.experimental.deepseek_prefill.test_moe_fused_swiglu import (
    _ISL_ALLOCATED_TOKENS,
    _ISL_EXHAUSTIVE_MODELS,
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
# starts to cover the DRAM weight read, and its cross-sweep spread (1.6% on kimi_k26) leaves too
# little headroom under _MARGIN to gate on. Everything past it holds inside 0.5%.
_LOW_ISL_MARGIN = 0.08
_KNEE_TOKENS = 512

# Device duration in ns per (model, active), x_rm layout: the midpoint of three sweeps on a BH p150b
# (2026-11-02), each case a median of _ITERS dispatches. Recalibrate on the perf runner (DDR-speed
# dependent): each case logs an "RT-CAL" line in this dict's format, so one run regenerates the table.
_EXPECTED_NS: dict[tuple[str, int], int] = {
    ("kimi_k2_7", 0): 2_045,
    ("kimi_k2_7", 32): 76_368,
    ("kimi_k2_7", 64): 79_122,
    ("kimi_k2_7", 96): 86_564,
    ("kimi_k2_7", 128): 86_710,
    ("kimi_k2_7", 256): 109_696,
    ("kimi_k2_7", 512): 184_738,
    ("kimi_k2_7", 768): 255_205,
    ("kimi_k2_7", 1024): 324_807,
    ("kimi_k2_7", 2048): 602_993,
    ("kimi_k2_7", 4096): 1_157_884,
    ("kimi_k2_7", 5120): 1_436_687,
    ("glm_53", 0): 2_058,
    ("glm_53", 32): 65_531,
    ("glm_53", 64): 69_422,
    ("glm_53", 96): 75_582,
    ("glm_53", 128): 76_005,
    ("glm_53", 256): 99_674,
    ("glm_53", 512): 169_466,
    ("glm_53", 768): 234_072,
    ("glm_53", 1024): 298_588,
    ("glm_53", 2048): 558_096,
    ("glm_53", 4096): 1_076_104,
    ("glm_53", 5120): 1_334_726,
}

# Same measurement and key as _EXPECTED_NS, with the weights DRAM ND-sharded: a core fetches its
# whole K-row weight slice in ONE NoC request instead of one per tile. Its own table because the gain
# lands where this op is weight-read bound -- 1.16-1.17x at 128, 1.12x at 256, and still 1.006-1.009x
# at 5120 -- so the interleaved bands reject the low-ISL cases outright. Same calibration as above.
_NDSHARD_EXPECTED_NS: dict[tuple[str, int], int] = {
    ("kimi_k2_7", 0): 2_079,
    ("kimi_k2_7", 32): 64_854,
    ("kimi_k2_7", 64): 67_741,
    ("kimi_k2_7", 96): 73_127,
    ("kimi_k2_7", 128): 73_600,
    ("kimi_k2_7", 256): 101_770,
    ("kimi_k2_7", 512): 181_649,
    ("kimi_k2_7", 768): 252_740,
    ("kimi_k2_7", 1024): 320_104,
    ("kimi_k2_7", 2048): 595_918,
    ("kimi_k2_7", 4096): 1_155_489,
    ("kimi_k2_7", 5120): 1_432_536,
    ("glm_53", 0): 2_106,
    ("glm_53", 32): 56_380,
    ("glm_53", 64): 59_623,
    ("glm_53", 96): 64_469,
    ("glm_53", 128): 65_568,
    ("glm_53", 256): 89_622,
    ("glm_53", 512): 159_695,
    ("glm_53", 768): 224_244,
    ("glm_53", 1024): 289_223,
    ("glm_53", 2048): 548_068,
    ("glm_53", 4096): 1_065_590,
    ("glm_53", 5120): 1_324_822,
}

# Kimi K3 runs SiTU-GLU at the post-projection dims, so its K axis is ROUTED_EXPERT_HIDDEN_SIZE and
# it cannot be driven from SINGLE_EXPERT_MODELS (which reads config.EMB_SIZE). Same measurement.
_K3_SITU_EXPECTED_NS: dict[int, int] = {
    0: 1_762,
    32: 69_903,
    64: 75_006,
    96: 82_900,
    128: 85_842,
    256: 120_747,
    512: 221_013,
    768: 310_053,
    1024: 399_767,
    2048: 758_652,
    4096: 1_476_747,
    5120: 1_835_451,
}

# Kimi K3 SiTU-GLU counterpart of _NDSHARD_EXPECTED_NS: 1.07x at 128, 1.08x at 256 and 512.
_K3_SITU_NDSHARD_EXPECTED_NS: dict[int, int] = {
    0: 1_782,
    32: 63_495,
    64: 66_969,
    96: 79_675,
    128: 80_023,
    256: 111_640,
    512: 205_237,
    768: 295_267,
    1024: 384_594,
    2048: 744_304,
    4096: 1_462_008,
    5120: 1_820_239,
}


def _baseline_or_skip(table, key, label: str):
    """The case's baseline, or a skip when its table has no entry. Borrowing the other placement's
    number is not an option -- the two tables exist because the placements measure differently."""
    expected_ns = table.get(key)
    if expected_ns is None:
        pytest.skip(f"no baseline for {label}; add {key} to its table -- see the comment there")
    return expected_ns


def _margin_for(active: int) -> float:
    return _CEILING_ONLY if active == 0 else _LOW_ISL_MARGIN if active <= _KNEE_TOKENS else _MARGIN


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
    margin: float,
    weights_dram_sharded: bool,
):
    require_realtime_profiler("moe_fused_swiglu perf checks")

    table = _NDSHARD_EXPECTED_NS if weights_dram_sharded else _EXPECTED_NS
    placement = "w_ndshard" if weights_dram_sharded else "w_interleaved"
    label = f'{placement} ("{model_name}", {active_tokens})'
    expected_ns = _baseline_or_skip(table, (model_name, active_tokens), label)

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
    [pytest.param(active, _margin_for(active), id=f"k3-isl-{active}-perf") for active in _ISL_PERF_SWEEP],
)
@pytest.mark.parametrize("weights_dram_sharded", [False, True], ids=["w_interleaved", "w_ndshard"])
@pytest.mark.requires_host_iommu
@pytest.mark.skipif(not is_blackhole(), reason="SiTU-GLU is Blackhole-only")
@pytest.mark.skipif(not is_p150(), reason="perf baselines are P150-specific; skip on any other board")
@skip_with_llk_assert("No need to verify LLK asserts for performance tests.")
@skip_with_watcher("Watcher perturbs kernel timing; perf checks are not meaningful with it enabled.")
def test_moe_fused_swiglu_k3_perf(device, active_tokens: int, margin: float, weights_dram_sharded: bool):
    """Kimi K3 (SiTU-GLU) device duration over the same ISL sweep as above."""
    require_realtime_profiler("moe_fused_swiglu perf checks")

    table = _K3_SITU_NDSHARD_EXPECTED_NS if weights_dram_sharded else _K3_SITU_EXPECTED_NS
    placement = "w_ndshard" if weights_dram_sharded else "w_interleaved"
    label = f'{placement} ("kimi_k3", {active_tokens})'
    expected_ns = _baseline_or_skip(table, active_tokens, label)

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
            weights_dram_sharded=weights_dram_sharded,
        ),
        _OP_KERNEL_DIR,
        expected_ns=expected_ns,
        margin=margin,
        label=label,
        iters=_ITERS,
        verbose=_VERBOSE,
    )
