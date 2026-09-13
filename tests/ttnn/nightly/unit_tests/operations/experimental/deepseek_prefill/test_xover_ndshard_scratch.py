# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""SCRATCH: fused-vs-composite crossover under BOTH weight placements.

test_crossover_remeasure_scratch.py with a DRAM ND-sharded axis added, because the threshold a model
ships has to match the placement it runs: ND sharding speeds the two ops by different amounts at
different counts, so the count where they trade places moves with it.

Two things this keeps from the original, both load-bearing for a threshold:

* Each model drives its REAL activation. The ops implement activations differently, so a SiLU-measured
  crossover does not apply to a SwiGluOai or SituGlu model.
* gpt-oss is absent. Its experts carry FFN biases and run_single_routed_expert takes none, so the only
  comparison available here is the bias-free one its config already records as the wrong basis.

The sweep is dense through the crossover region on purpose. The composite's cost is flat inside an M
chunk while the fused op's rises with the count, so the two cross more than once; a coarse grid reads
a single crossing off a sawtooth and lands in the wrong place.
"""

import os
import statistics

import pytest
from loguru import logger

import ttnn
from models.common.utility_functions import is_blackhole, skip_with_llk_assert, skip_with_watcher
from models.demos.deepseek_v3_d_p.reference.deepseek_v4_flash_config import DeepSeekV4FlashConfig
from models.demos.deepseek_v3_d_p.reference.deepseek_v4_pro_config import DeepSeekV4ProConfig
from models.demos.deepseek_v3_d_p.reference.glm_5_1_config import GLM51Config
from models.demos.deepseek_v3_d_p.reference.kimi_k2_7_config import KimiK27Config
from models.demos.deepseek_v3_d_p.reference.kimi_k3_config import KimiK3Config
from models.demos.deepseek_v3_d_p.reference.minimax_m3_config import MiniMaxM3Config
from models.demos.deepseek_v3_d_p.utils.smbus_telemetry import is_p150
from tests.ttnn.profiling.realtime_profiler_utils import profile_realtime_program_merged, require_realtime_profiler
from tests.ttnn.nightly.unit_tests.operations.experimental.deepseek_prefill.test_moe_fused_swiglu import (
    _ISL_ALLOCATED_TOKENS,
    run_moe_fused_swiglu,
)
from tests.ttnn.nightly.unit_tests.operations.experimental.deepseek_prefill.test_single_routed_expert import (
    run_single_routed_expert,
)

_RE_DIR = "/unified_routed_expert_ffn/"
_SWIGLU_DIR = "/moe_fused_swiglu/"
_ITERS = 3
_OUT = os.environ.get("XND_OUT", "/tmp/xover_nd")
_ACTIVES = [
    int(a)
    for a in os.environ.get(
        "XND_ACTIVES", "96,128,160,192,224,256,288,320,384,448,512,576,640,704,768,832,896,1024"
    ).split(",")
]

_MODELS = [
    (
        "minimax_m3",
        MiniMaxM3Config.EMB_SIZE,
        MiniMaxM3Config.MOE_INTERMEDIATE_SIZE,
        ttnn.RoutedExpertActivation.SwiGluOai,
    ),
    (
        "dsv4_pro",
        DeepSeekV4ProConfig.EMB_SIZE,
        DeepSeekV4ProConfig.MOE_INTERMEDIATE_SIZE,
        ttnn.RoutedExpertActivation.Silu,
    ),
    (
        "dsv4_flash",
        DeepSeekV4FlashConfig.EMB_SIZE,
        DeepSeekV4FlashConfig.MOE_INTERMEDIATE_SIZE,
        ttnn.RoutedExpertActivation.Silu,
    ),
    ("kimi_k2_7", KimiK27Config.EMB_SIZE, KimiK27Config.MOE_INTERMEDIATE_SIZE, ttnn.RoutedExpertActivation.Silu),
    ("glm_51", GLM51Config.EMB_SIZE, GLM51Config.MOE_INTERMEDIATE_SIZE, ttnn.RoutedExpertActivation.Silu),
    (
        "kimi_k3",
        KimiK3Config.ROUTED_EXPERT_HIDDEN_SIZE,
        KimiK3Config.MOE_INTERMEDIATE_SIZE,
        ttnn.RoutedExpertActivation.SituGlu,
    ),
]

_OPS = {
    "composite": (_RE_DIR, run_single_routed_expert),
    "fused": (_SWIGLU_DIR, run_moe_fused_swiglu),
}

_PARAMS = [
    pytest.param(
        op, sharded, name, emb, hidden, activation, id=f"{op}-{'ndshard' if sharded else 'interleaved'}-{name}"
    )
    for op in _OPS
    for sharded in (False, True)
    for name, emb, hidden, activation in _MODELS
]


def _measure_ns(device, run_fn, kernel_dir):
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


@pytest.mark.parametrize("op, sharded, name, emb, hidden, activation", _PARAMS)
@pytest.mark.skipif(
    not os.environ.get("RE_XOVER_NDSHARD"),
    reason="measurement harness, not a gate; opt in with RE_XOVER_NDSHARD=1",
)
@pytest.mark.requires_host_iommu
@pytest.mark.skipif(not is_blackhole(), reason="both routed-expert paths are Blackhole-only")
@pytest.mark.skipif(not is_p150(), reason="timings are P150-specific")
@pytest.mark.timeout(0)
@skip_with_llk_assert("No need to verify LLK asserts for performance tests.")
@skip_with_watcher("Watcher perturbs kernel timing.")
def test_xover_nd(device, op, sharded, name, emb, hidden, activation):
    require_realtime_profiler("crossover under both weight placements")
    os.makedirs(_OUT, exist_ok=True)
    kernel_dir, runner = _OPS[op]
    placement = "ndshard" if sharded else "interleaved"
    rows = []
    for active in _ACTIVES:
        try:
            ns = _measure_ns(
                device,
                lambda: runner(
                    device,
                    _ISL_ALLOCATED_TOKENS,
                    emb,
                    hidden,
                    active_tokens=active,
                    x_row_major=True,
                    activation=activation,
                    weights_dram_sharded=sharded,
                ),
                kernel_dir,
            )
            logger.info(f"XND {op} {placement} {name} {emb}x{hidden} isl-{active}: {ns / 1000:.2f} us")
        except Exception as exc:
            ns = None
            logger.error(f"XND {op} {placement} {name} isl-{active}: FAILED {type(exc).__name__}: {exc}")
        rows.append((active, ns))
        with open(os.path.join(_OUT, f"{op}_{placement}_{name}.csv"), "w") as fh:
            fh.write("op,placement,model,emb,hidden,active,ns\n")
            for a, v in rows:
                fh.write(f"{op},{placement},{name},{emb},{hidden},{a},{'' if v is None else round(v)}\n")
