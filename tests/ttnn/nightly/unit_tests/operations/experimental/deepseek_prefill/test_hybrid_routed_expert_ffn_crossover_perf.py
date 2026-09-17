# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Union op against the two-op forward, over the crossover gate's sweep.

Same load, models and ISL points as test_routed_expert_crossover_perf.py: one expert, `active`
live rows against the same 5120-row region, at each model's shipped
ROUTED_EXPERT_HYBRID_TOKEN_THRESHOLD. The difference is what is compared -- that gate measures
each op alone and keeps the faster; this measures the two arrangements a layer can actually
dispatch:

  two-op : unified_routed_expert_moe then moe_fused_swiglu, as TtRoutedExpert dispatches them
  one-op : hybrid_routed_expert_moe, both halves in one program

Device duration only. Reporting harness, not a gate -- it prints the pair and the ratio so a
regression is visible, but nothing here asserts a bound.
"""

import statistics

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import is_blackhole
from models.demos.deepseek_v3_d_p.reference.glm_5_1_config import GLM51Config
from models.demos.deepseek_v3_d_p.reference.kimi_k2_7_config import KimiK27Config
from models.demos.deepseek_v3_d_p.tt.moe.tt_routed_expert import TtRoutedExpert
from tests.ttnn.profiling.realtime_profiler_utils import (
    profile_realtime_program_merged,
    require_realtime_profiler,
)
from tests.ttnn.nightly.unit_tests.operations.experimental.deepseek_prefill import ci_pruning
from tests.ttnn.nightly.unit_tests.operations.experimental.deepseek_prefill.test_single_routed_expert import (
    _ISL_ALLOCATED_TOKENS,
    _ISL_EXHAUSTIVE_SWEEP,
)

pytestmark = pytest.mark.uncollect_if(pred=ci_pruning.no_production_counterpart)

_ITERS = 3
_MODELS = {"kimi_k2_7": KimiK27Config, "glm_51": GLM51Config}

# RT records carry kernel sources, not an op code, so each arrangement is identified by the
# kernel directories its programs come from.
_TWO_OP_DIRS = ("/unified_routed_expert_ffn/", "/moe_fused_swiglu/")
_ONE_OP_DIRS = ("/hybrid_routed_expert_ffn/",)


def _idx_tensor(device, values):
    return ttnn.from_torch(
        torch.tensor(values, dtype=torch.int32), layout=ttnn.ROW_MAJOR_LAYOUT, device=device, dtype=ttnn.uint32
    )


@pytest.mark.skipif(not is_blackhole(), reason="the routed expert is Blackhole-only")
@pytest.mark.parametrize("model", list(_MODELS), ids=list(_MODELS))
@pytest.mark.parametrize("active", _ISL_EXHAUSTIVE_SWEEP)
def test_union_vs_two_op_crossover(device, model: str, active: int):
    require_realtime_profiler("the union-vs-two-op crossover comparison")

    config = _MODELS[model]
    emb_dim, hidden_dim = config.EMB_SIZE, config.MOE_INTERMEDIATE_SIZE
    threshold = config.ROUTED_EXPERT_HYBRID_TOKEN_THRESHOLD
    torch.manual_seed(42)

    weights = [
        {
            "gate_proj": torch.randn(hidden_dim, emb_dim, dtype=torch.float32) * 0.02,
            "up_proj": torch.randn(hidden_dim, emb_dim, dtype=torch.float32) * 0.02,
            "down_proj": torch.randn(emb_dim, hidden_dim, dtype=torch.float32) * 0.02,
        }
    ]
    torch_input = torch.zeros(_ISL_ALLOCATED_TOKENS, emb_dim, dtype=torch.float32)
    if active:
        torch_input[:active] = torch.randn(active, emb_dim, dtype=torch.float32)

    x = ttnn.from_torch(
        torch_input,
        mesh_mapper=ttnn.ReplicateTensorToMesh(device),
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        dtype=ttnn.bfloat16,
    )
    idx_tt = _idx_tensor(device, [0])
    counts_tt = _idx_tensor(device, [active])
    offsets_tt = _idx_tensor(device, [0])

    module = TtRoutedExpert(
        mesh_device=device,
        experts_per_chip=1,
        global_expert_idx_table=idx_tt,
        emb_dim=emb_dim,
        hidden_dim=hidden_dim,
        max_tokens=_ISL_ALLOCATED_TOKENS,
        torch_weights=weights,
        activation=ttnn.RoutedExpertActivation.Silu,
        hybrid_token_threshold=threshold,
    )

    def run_two_op():
        return module(x, counts_tt, offsets_tt)

    def run_one_op():
        return ttnn.experimental.deepseek_prefill.hybrid_routed_expert_moe(
            x,
            offsets_tt,
            counts_tt,
            idx_tt,
            module.gate_projs,
            module.up_projs,
            module.down_projs,
            max_dispatched_tokens_per_expert=_ISL_ALLOCATED_TOKENS,
            hybrid_token_threshold=threshold,
            compute_kernel_config=module.compute_kernel_config,
            activation=ttnn.RoutedExpertActivation.Silu,
        )

    def measure(run_fn, kernel_dirs, label):
        run_fn()  # program cache + JIT

        def run_all():
            for _ in range(_ITERS):
                run_fn()

        _, programs = profile_realtime_program_merged(device, run_all)
        # Only this arrangement's programs; a window also carries unrelated ones and can pick up
        # a record from the warm-up dispatch, so take the last _ITERS batches.
        mine = [
            p["duration_ns"]
            for p in programs.values()
            if any(d in src for src in p["kernel_sources"] for d in kernel_dirs)
        ]
        per_iter = len(mine) // _ITERS if _ITERS else 0
        assert per_iter >= 1, f"{label}: expected at least {_ITERS} programs in the window, saw {len(mine)}"
        # Each iteration dispatches `per_iter` programs back to back; the arrangement's cost is
        # their sum, and the median over iterations drops the occasional flyer.
        totals = [sum(mine[i * per_iter : (i + 1) * per_iter]) for i in range(_ITERS)]
        return statistics.median(totals), per_iter

    two_op_ns, two_op_programs = measure(run_two_op, _TWO_OP_DIRS, "two-op")
    one_op_ns, one_op_programs = measure(run_one_op, _ONE_OP_DIRS, "one-op")

    logger.info(
        f"XOVER {model} active={active} threshold={threshold} "
        f"two_op={two_op_ns:.0f}ns({two_op_programs}p) one_op={one_op_ns:.0f}ns({one_op_programs}p) "
        f"ratio={one_op_ns / two_op_ns:.3f}"
    )
    assert one_op_programs == 1, f"the union op must dispatch one program, saw {one_op_programs}"
