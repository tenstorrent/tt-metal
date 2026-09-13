# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""SCRATCH: reproduce the branch-vs-main perf table over its (model, ISL) grid.

Activation is deliberately left at run_single_routed_expert's default (SiLU) for every model: the
main-side reference column was measured that way, and driving a model with its real activation
against a SiLU-measured baseline reports a regression that is purely the activation difference.
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

_RE_DIR = "/unified_routed_expert_ffn/"
_ITERS = 3
_OUT = os.environ.get("PT_OUT", "/tmp/perftable")
_ISLS = [int(a) for a in os.environ.get("PT_ISLS", "0,64,128,256,512,1024,2048,4096,5120").split(",")]

_MODELS = [
    ("kimi_k26", 7168, 2048),
    ("glm_51", 6144, 2048),
    ("gptoss_120b", 2880, 2880),
    ("minimax_m3", 6144, 3072),
    ("kimi_k3", 3584, 3072),
    ("dsv4_pro", 7168, 3072),
    ("dsv4_flash", 4096, 2048),
]
_PARAMS = [pytest.param(*m, id=m[0]) for m in _MODELS]


def _measure_ns(device, run_fn):
    def run_all():
        for _ in range(_ITERS):
            run_fn()

    _, per_program = profile_realtime_program_merged(device, run_all)
    matched = [
        e["duration_ns"]
        for e in per_program.values()
        if any(_RE_DIR in s.replace("\\", "/") for s in e["kernel_sources"])
    ]
    assert len(matched) == _ITERS, f"expected {_ITERS} programs matching {_RE_DIR}, got {len(matched)}"
    return statistics.median(matched)


@pytest.mark.parametrize("name, emb, hidden", _PARAMS)
@pytest.mark.skipif(
    not os.environ.get("RE_PERFTABLE"),
    reason="measurement harness, not a gate; opt in with RE_PERFTABLE=1",
)
@pytest.mark.requires_host_iommu
@pytest.mark.skipif(not is_blackhole(), reason="the fused routed-expert path is Blackhole-only")
@pytest.mark.skipif(not is_p150(), reason="timings are P150-specific")
@pytest.mark.timeout(0)
@skip_with_llk_assert("No need to verify LLK asserts for performance tests.")
@skip_with_watcher("Watcher perturbs kernel timing.")
def test_perftable(device, name, emb, hidden):
    require_realtime_profiler("branch-vs-main perf table")
    os.makedirs(_OUT, exist_ok=True)
    rows = []
    for active in _ISLS:
        try:
            ns = _measure_ns(
                device,
                lambda: run_single_routed_expert(
                    device, _ISL_ALLOCATED_TOKENS, emb, hidden, active_tokens=active, x_row_major=True
                ),
            )
            logger.info(f"PT {name} {emb}x{hidden} isl-{active}: {ns / 1000:.2f} us")
        except Exception as exc:
            ns = None
            logger.error(f"PT {name} isl-{active}: FAILED {type(exc).__name__}: {exc}")
        rows.append((active, ns))
        with open(os.path.join(_OUT, f"{name}.csv"), "w") as fh:
            fh.write("model,emb,hidden,active,ns\n")
            for a, v in rows:
                fh.write(f"{name},{emb},{hidden},{a},{'' if v is None else round(v)}\n")
