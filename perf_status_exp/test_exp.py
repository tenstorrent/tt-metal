"""Scratch (merge-proto): end-to-end RT device time over a dense ISL range, no zones. Not committed.

Same measurement as the perf gates (x_rm, 5120 allocated, median of 3, RT profiler, PCC checked by
the run_* helpers); band disabled so every case logs its RT-CAL line. ISLs from SWEEP_ISL="lo:hi"
(inclusive, default 64:128). Filter op/model/layout with -k.
"""

import os

import pytest

from tests.ttnn.profiling.realtime_profiler_utils import assert_op_duration_merged, require_realtime_profiler
from tests.ttnn.nightly.unit_tests.operations.experimental.deepseek_prefill.test_single_routed_expert import (
    _ISL_ALLOCATED_TOKENS,
    SINGLE_EXPERT_MODELS,
    run_single_routed_expert,
)
from tests.ttnn.nightly.unit_tests.operations.experimental.deepseek_prefill.test_moe_fused_swiglu import (
    run_moe_fused_swiglu,
)

_SPEC = os.environ.get("SWEEP_ISL", "64:128")
if ":" in _SPEC:
    _LO, _HI = (int(v) for v in _SPEC.split(":"))
    _ISLS = list(range(_LO, _HI + 1))
else:
    _ISLS = [int(v) for v in _SPEC.split(",")]
_CFG = {name: config for name, config, _ in SINGLE_EXPERT_MODELS}
_OPS = {
    "moe_fused_swiglu": (run_moe_fused_swiglu, "/moe_fused_swiglu/"),
    "single_routed_expert": (run_single_routed_expert, "/unified_routed_expert_ffn/"),
}


@pytest.mark.parametrize("active_tokens", _ISLS, ids=lambda a: f"isl-{a}-perf")
@pytest.mark.parametrize("model_name", ["glm_53", "kimi_k2_7"])
@pytest.mark.parametrize("weights_dram_sharded", [False, True], ids=["w_interleaved", "w_ndshard"])
@pytest.mark.parametrize("op", list(_OPS))
def test_isl_sweep(device, op, weights_dram_sharded, model_name, active_tokens):
    require_realtime_profiler("isl sweep")
    run_fn, kernel_dir = _OPS[op]
    config = _CFG[model_name]
    placement = "w_ndshard" if weights_dram_sharded else "w_interleaved"
    assert_op_duration_merged(
        device,
        lambda: run_fn(
            device,
            _ISL_ALLOCATED_TOKENS,
            config.EMB_SIZE,
            config.MOE_INTERMEDIATE_SIZE,
            active_tokens=active_tokens,
            x_row_major=True,
            weights_dram_sharded=weights_dram_sharded,
        ),
        kernel_dir,
        expected_ns=1,
        margin=1e12,  # no band: log only
        label=f'{placement} ("{model_name}", {active_tokens})',
        iters=3,
    )
