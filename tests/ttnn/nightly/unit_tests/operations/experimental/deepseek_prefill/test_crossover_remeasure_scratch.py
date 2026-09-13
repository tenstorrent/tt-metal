# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""SCRATCH: re-measure the fused-vs-composite crossover after the bf16 gate/up accumulators.

One test per (op, model) looping the ISL sweep -- the `device` fixture is function-scoped, so a test
per cell would open and close the device per cell. Both sides come from the shared runners, so every
cell is PCC-graded on the same path the perf gates measure; only the expected-ns band is dropped,
because the point is to produce the crossover rather than check one.

Bias-free by construction: `run_single_routed_expert` takes no bias arguments, so a biased model
(gpt-oss) cannot be compared apples-to-apples here and is deliberately absent.
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
_OUT = os.environ.get("XOVER_OUT", "/tmp/xover")
_ACTIVES = [int(a) for a in os.environ.get("XOVER_ACTIVES", "128,256,512,1024,2048,4096,5120").split(",")]

# (name, emb, hidden, activation). gpt-oss is absent: its experts carry FFN biases, and the
# composite runner cannot take them, so a comparison here would be the bias-free one the model
# config already records as the wrong basis for its threshold.
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
    (
        "kimi_k2_7",
        KimiK27Config.EMB_SIZE,
        KimiK27Config.MOE_INTERMEDIATE_SIZE,
        ttnn.RoutedExpertActivation.Silu,
    ),
    (
        "glm_51",
        GLM51Config.EMB_SIZE,
        GLM51Config.MOE_INTERMEDIATE_SIZE,
        ttnn.RoutedExpertActivation.Silu,
    ),
    (
        "kimi_k3",
        KimiK3Config.ROUTED_EXPERT_HIDDEN_SIZE,
        KimiK3Config.MOE_INTERMEDIATE_SIZE,
        ttnn.RoutedExpertActivation.SituGlu,
    ),
]
_PARAMS = [pytest.param(*m, id=m[0]) for m in _MODELS]


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


def _sweep(device, tag, name, emb, hidden, kernel_dir, dispatch):
    os.makedirs(_OUT, exist_ok=True)
    rows = []
    for active in _ACTIVES:
        try:
            ns = _measure_ns(device, lambda: dispatch(active), kernel_dir)
            logger.info(f"XOVER {tag} {name} {emb}x{hidden} isl-{active}: {ns / 1000:.1f} us")
        except Exception as exc:
            ns = None
            logger.error(f"XOVER {tag} {name} isl-{active}: FAILED {type(exc).__name__}: {exc}")
        rows.append((active, ns))
        with open(os.path.join(_OUT, f"{tag}_{name}.csv"), "w") as fh:
            fh.write("op,model,emb,hidden,active,ns\n")
            for a, v in rows:
                fh.write(f"{tag},{name},{emb},{hidden},{a},{'' if v is None else round(v)}\n")


@pytest.mark.parametrize("name, emb, hidden, activation", _PARAMS)
@pytest.mark.skipif(
    not os.environ.get("RE_XOVER_REMEASURE"),
    reason="measurement harness, not a gate; opt in with RE_XOVER_REMEASURE=1",
)
@pytest.mark.requires_host_iommu
@pytest.mark.skipif(not is_blackhole(), reason="both routed-expert fused paths are Blackhole-only")
@pytest.mark.skipif(not is_p150(), reason="timings are P150-specific")
@pytest.mark.timeout(0)
@skip_with_llk_assert("No need to verify LLK asserts for performance tests.")
@skip_with_watcher("Watcher perturbs kernel timing.")
def test_xover_composite(device, name, emb, hidden, activation):
    require_realtime_profiler("crossover re-measure")
    _sweep(
        device,
        "composite",
        name,
        emb,
        hidden,
        _RE_DIR,
        lambda a: run_single_routed_expert(
            device, _ISL_ALLOCATED_TOKENS, emb, hidden, active_tokens=a, x_row_major=True, activation=activation
        ),
    )


@pytest.mark.parametrize("name, emb, hidden, activation", _PARAMS)
@pytest.mark.skipif(
    not os.environ.get("RE_XOVER_REMEASURE"),
    reason="measurement harness, not a gate; opt in with RE_XOVER_REMEASURE=1",
)
@pytest.mark.requires_host_iommu
@pytest.mark.skipif(not is_blackhole(), reason="both routed-expert fused paths are Blackhole-only")
@pytest.mark.skipif(not is_p150(), reason="timings are P150-specific")
@pytest.mark.timeout(0)
@skip_with_llk_assert("No need to verify LLK asserts for performance tests.")
@skip_with_watcher("Watcher perturbs kernel timing.")
def test_xover_fused(device, name, emb, hidden, activation):
    require_realtime_profiler("crossover re-measure")
    _sweep(
        device,
        "fused",
        name,
        emb,
        hidden,
        _SWIGLU_DIR,
        lambda a: run_moe_fused_swiglu(
            device, _ISL_ALLOCATED_TOKENS, emb, hidden, active_tokens=a, x_row_major=True, activation=activation
        ),
    )
