# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Per-expert cost of the union op against the standalone op that would have served that expert.

Both halves of the union op run on the whole 88-core rectangle, one pass after the other, so an
expert routed to a pass should cost what it costs in that op's own perf gate -- the load, the
grid and the blocking are identical. What the union op adds on top is the other pass's empty
sweep (it reads the counts and skips every expert) plus one grid barrier.

The single-expert load matches test_moe_fused_swiglu_perf.py and test_single_routed_expert_perf.py
exactly: one expert, `active_tokens` live rows against the same 5120-row region, so the numbers
here are directly comparable to those tables.
"""

import statistics

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import is_blackhole
from models.demos.deepseek_v3_d_p.reference.glm_5_1_config import GLM51Config
from models.demos.deepseek_v3_d_p.tt.moe.tt_routed_expert import TtRoutedExpert
from tests.ttnn.profiling.realtime_profiler_utils import (
    profile_realtime_program_merged,
    require_realtime_profiler,
)
from tests.ttnn.nightly.unit_tests.operations.experimental.deepseek_prefill import ci_pruning

pytestmark = pytest.mark.uncollect_if(pred=ci_pruning.no_production_counterpart)

_ALLOCATED_TOKENS = 5120
_THRESHOLD = 512
_ITERS = 3

# RT records carry kernel sources, not an op code, so identify the op by its kernel directory.
_KERNEL_DIR = "/hybrid_routed_expert_ffn/"

# The standalone per-op baselines this is measured against, copied from the existing perf gates so
# a drift in either shows up here as well. glm_51, interleaved weights, BH p150b 2026-09-15.
# test_moe_fused_swiglu_perf.py::_EXPECTED_NS -- that op alone.
_FUSED_BASELINE_NS = {128: 86_753, 256: 106_704, 512: 181_892}
# test_routed_expert_crossover_perf.py::_EXPECTED_NS, which is deliberately best-of-BOTH-ops and
# not keyed by the winner. Usable as a unified baseline only because unified is the faster op at
# every count here; at a crossover count it would be the other op's number.
_UNIFIED_BASELINE_NS = {1024: 257_645, 2048: 507_790, 4096: 1_005_379}

# What the union op may add over the standalone op for one expert: the other pass's empty sweep
# plus one grid barrier. Measured 1.01-1.11x, so this catches a real regression without tripping
# on run-to-run spread.
_MAX_PER_EXPERT_RATIO = 1.25


def _idx_tensor(device, values):
    return ttnn.from_torch(
        torch.tensor(values, dtype=torch.int32), layout=ttnn.ROW_MAJOR_LAYOUT, device=device, dtype=ttnn.uint32
    )


@pytest.mark.skipif(not is_blackhole(), reason="the routed expert is Blackhole-only")
@pytest.mark.parametrize("active", sorted({*_FUSED_BASELINE_NS, *_UNIFIED_BASELINE_NS}))
def test_union_op_per_expert_cost(device, active: int):
    require_realtime_profiler("the per-expert union-op cost comparison")

    emb_dim = GLM51Config.EMB_SIZE
    hidden_dim = GLM51Config.MOE_INTERMEDIATE_SIZE
    torch.manual_seed(42)

    weights = [
        {
            "gate_proj": torch.randn(hidden_dim, emb_dim, dtype=torch.float32) * 0.02,
            "up_proj": torch.randn(hidden_dim, emb_dim, dtype=torch.float32) * 0.02,
            "down_proj": torch.randn(emb_dim, hidden_dim, dtype=torch.float32) * 0.02,
        }
    ]

    torch_input = torch.zeros(_ALLOCATED_TOKENS, emb_dim, dtype=torch.float32)
    torch_input[:active] = torch.randn(active, emb_dim, dtype=torch.float32)
    tt_input = ttnn.from_torch(
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
        max_tokens=_ALLOCATED_TOKENS,
        torch_weights=weights,
        activation=ttnn.RoutedExpertActivation.Silu,
    )

    def run_union():
        return ttnn.experimental.deepseek_prefill.hybrid_routed_expert_moe(
            tt_input,
            offsets_tt,
            counts_tt,
            idx_tt,
            module.gate_projs,
            module.up_projs,
            module.down_projs,
            max_dispatched_tokens_per_expert=_ALLOCATED_TOKENS,
            hybrid_token_threshold=_THRESHOLD,
            compute_kernel_config=module.compute_kernel_config,
            activation=ttnn.RoutedExpertActivation.Silu,
        )

    run_union()  # program cache + JIT

    def run_all():
        for _ in range(_ITERS):
            run_union()

    # Identify the op by its kernel directory and take every match in the window: records arrive
    # asynchronously, so a window also carries unrelated programs and can pick up a record from
    # the warm-up dispatch. One match per iteration is the check that the number is attributable.
    _, programs = profile_realtime_program_merged(device, run_all)
    samples = [p["duration_ns"] for p in programs.values() if any(_KERNEL_DIR in src for src in p["kernel_sources"])]
    # Records come back in dispatch order, and the warm-up dispatch's can still land in this
    # window, so keep the last _ITERS rather than demanding an exact count.
    assert len(samples) >= _ITERS, f"expected at least {_ITERS} union-op programs in the window, saw {len(samples)}"
    measured = statistics.median(samples[-_ITERS:])

    half = "fused" if active <= _THRESHOLD else "unified"
    baseline = (_FUSED_BASELINE_NS if half == "fused" else _UNIFIED_BASELINE_NS)[active]
    overhead = measured - baseline
    ratio = measured / baseline
    logger.info(
        f"active={active} -> {half} half: union {measured:.0f} ns vs standalone {baseline} ns "
        f"(+{overhead:.0f} ns, {ratio:.2f}x)"
    )
    assert ratio <= _MAX_PER_EXPERT_RATIO, (
        f"union op costs {ratio:.2f}x the standalone {half} op for one expert at active={active} "
        f"({measured:.0f} ns vs {baseline} ns), over the {_MAX_PER_EXPERT_RATIO}x bound"
    )
