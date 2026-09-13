# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""SCRATCH: device time for BOTH routed-expert ops under BOTH weight placements.

Four quadrants over the (model, ISL) grid of the branch-vs-main table:
unified_routed_expert_ffn and moe_fused_swiglu, each with DRAM-interleaved and DRAM ND-sharded
weights. One case per (op, placement, model); the ISL sweep runs inside it.

Activation is left at each runner's default (SiLU) for every model, matching the reference table:
driving a model with its real activation against a SiLU-measured column reports a difference that
is the activation, not the placement.

ND-sharded cases use TtRoutedExpert.dram_nd_shard_spec — the placement the model ships — so the
widths are the ones production reads, not ones chosen to flatter a kernel.
"""

import os
import statistics

import pytest
from loguru import logger

from models.common.utility_functions import is_blackhole, skip_with_llk_assert, skip_with_watcher
from models.demos.deepseek_v3_d_p.utils.smbus_telemetry import is_p150
from tests.ttnn.profiling.realtime_profiler_utils import profile_realtime_program_merged, require_realtime_profiler
from tests.ttnn.nightly.unit_tests.operations.experimental.deepseek_prefill.test_single_routed_expert import (
    _ISL_ALLOCATED_TOKENS,
    run_single_routed_expert,
)
from tests.ttnn.nightly.unit_tests.operations.experimental.deepseek_prefill.test_moe_fused_swiglu import (
    run_moe_fused_swiglu,
)

_ITERS = 3
_OUT = os.environ.get("NDPT_OUT", "/tmp/ndshard_perftable")
_ISLS = [int(a) for a in os.environ.get("NDPT_ISLS", "0,64,128,256,512,1024,2048,4096,5120").split(",")]

# (name, emb, hidden) — the reference table's grid.
_MODELS = [
    ("kimi_k26", 7168, 2048),
    ("glm_51", 6144, 2048),
    ("gptoss_120b", 2880, 2880),
    ("minimax_m3", 6144, 3072),
    ("kimi_k3", 3584, 3072),
    ("dsv4_pro", 7168, 3072),
    ("dsv4_flash", 4096, 2048),
]

# RT records carry kernel sources, not an op code, so each op is identified by its kernel directory.
_OPS = {
    "ure": ("/unified_routed_expert_ffn/", run_single_routed_expert),
    "fused": ("/moe_fused_swiglu/", run_moe_fused_swiglu),
}

_PARAMS = [
    pytest.param(op, sharded, name, emb, hidden, id=f"{op}-{'ndshard' if sharded else 'interleaved'}-{name}")
    for op in _OPS
    for sharded in (False, True)
    for name, emb, hidden in _MODELS
]


def _measure_ns(device, kernel_dir, run_fn):
    def run_all():
        for _ in range(_ITERS):
            run_fn()

    _, per_program = profile_realtime_program_merged(device, run_all)
    matched = [
        e["duration_ns"]
        for e in per_program.values()
        if any(kernel_dir in s.replace("\\", "/") for s in e["kernel_sources"])
    ]
    assert len(matched) == _ITERS, f"expected {_ITERS} programs matching {kernel_dir}, got {len(matched)}"
    return statistics.median(matched)


@pytest.mark.parametrize("op, sharded, name, emb, hidden", _PARAMS)
@pytest.mark.skipif(
    not os.environ.get("RE_NDSHARD_PERFTABLE"),
    reason="measurement harness, not a gate; opt in with RE_NDSHARD_PERFTABLE=1",
)
@pytest.mark.requires_host_iommu
@pytest.mark.skipif(not is_blackhole(), reason="both routed-expert paths are Blackhole-only")
@pytest.mark.skipif(not is_p150(), reason="timings are P150-specific")
@pytest.mark.timeout(0)
@skip_with_llk_assert("No need to verify LLK asserts for performance tests.")
@skip_with_watcher("Watcher perturbs kernel timing.")
def test_ndshard_perftable(device, op, sharded, name, emb, hidden):
    require_realtime_profiler("ND-shard vs interleaved perf table")
    os.makedirs(_OUT, exist_ok=True)
    kernel_dir, runner = _OPS[op]
    placement = "ndshard" if sharded else "interleaved"
    rows = []
    for active in _ISLS:
        try:
            ns = _measure_ns(
                device,
                kernel_dir,
                lambda: runner(
                    device,
                    _ISL_ALLOCATED_TOKENS,
                    emb,
                    hidden,
                    active_tokens=active,
                    x_row_major=True,
                    weights_dram_sharded=sharded,
                ),
            )
            logger.info(f"NDPT {op} {placement} {name} {emb}x{hidden} isl-{active}: {ns / 1000:.2f} us")
        except Exception as exc:
            ns = None
            logger.error(f"NDPT {op} {placement} {name} isl-{active}: FAILED {type(exc).__name__}: {exc}")
        rows.append((active, ns))
        with open(os.path.join(_OUT, f"{op}_{placement}_{name}.csv"), "w") as fh:
            fh.write("op,placement,model,emb,hidden,active,ns\n")
            for a, v in rows:
                fh.write(f"{op},{placement},{name},{emb},{hidden},{a},{'' if v is None else round(v)}\n")
