# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""One dispatch against two: the merged routed-expert op versus the two-op forward it replaces.

Same experts, same counts, same threshold, same weights. The two-op forward gives each op the whole
88-core rectangle in turn; the merged op runs both halves concurrently on disjoint rows of it, so
this measures what the single dispatch actually costs, not just that it is one program.

Two numbers per arrangement, because device duration alone flatters the two-op forward: it
excludes the turnaround between one program finishing and the next launching. The profiler records
carry start/end timestamps on one device clock, so the turnaround is measured rather than assumed.
It is ~130 ns for the two-op forward (both its programs fit the kernel-config ring at once, so the
next launch is staged while the current one runs) and ~6.4 us for the union op (its config is more
than half the ring, so every dispatch stalls on the previous program's workers first).

Neither number models the overlap with combine, which is the reason the union op exists. A
regression here still has to be weighed against that.
"""

import statistics

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import is_blackhole
from models.demos.deepseek_v3_d_p.reference.deepseek_v3_config import DeepSeekV3Config
from models.demos.deepseek_v3_d_p.tt.moe.tt_routed_expert import TtRoutedExpert
from tests.ttnn.profiling.realtime_profiler_utils import (
    profile_realtime_program,
    require_realtime_profiler,
)
from tests.ttnn.nightly.unit_tests.operations.experimental.deepseek_prefill import ci_pruning
from tests.ttnn.nightly.unit_tests.operations.experimental.deepseek_prefill.test_hybrid_routed_expert_ffn_port import (
    _UNION_WORKER_L1_SIZE,
)

# Not wired into any model yet.
pytestmark = pytest.mark.uncollect_if(pred=ci_pruning.no_production_counterpart)

_ALLOCATED_TOKENS = 5120
_THRESHOLD = 512
_ITERS = 5

# Expert-count is the axis that can break parity: BOTH arrangements sweep every expert twice (once
# per op / once per pass) and skip the ones outside their band, so the skip cost grows with the
# expert count. The distributions cover the three shapes that matter -- work on both sides of the
# threshold, everything on the fused side (unified sweeps empty), everything on the unified side
# (fused sweeps empty).
_DISTRIBUTIONS = {
    "4x_mixed": [0, 251, 1024, 3001],
    "8x_mixed": [0, 128, 251, 400, 1024, 2048, 3001, 4096],
    "8x_all_fused": [0, 64, 128, 192, 251, 320, 400, 480],
    "8x_all_unified": [0, 1024, 1536, 2048, 2560, 3001, 3584, 4096],
}


# What the single dispatch may cost against the two it replaces, as an ABSOLUTE budget rather than
# a ratio. The union op's overhead is per-DISPATCH, not per-expert: one kernel-config ring stall
# (its config is more than half the ring, so every launch waits for the previous program's workers)
# plus one grid barrier. Measured +6.2, +7.3, +6.9, +7.1 us at 1, 2, 4 and 8 experts -- flat while
# the work grows 16x. A ratio bound would therefore be wrong in both directions: far too tight on a
# small layer and far too slack on a large one, where it would wave through hundreds of us. An
# absolute bound gets relatively STRICTER as the layer grows, which is what catches a real kernel
# regression. Worst observed across every case here is +16.5 us on period.
#
# Both sides are measured in the SAME run against the same device, so this needs no per-machine
# recalibration -- only re-derivation if the dispatch overhead itself changes.
_MAX_DISPATCH_OVERHEAD_NS = 30_000


# Period also carries host time -- two Python op calls against one -- which no bound here
# controls and which grows noisier with the workload. So period passes on either the absolute
# budget or a small relative slop, while DEVICE duration, the signal actually being claimed, is
# held to the absolute budget alone. A real kernel regression shows up in device duration; a
# 5 ms case drifting 0.7% on period does not.
_MAX_PERIOD_RATIO = 1.01


def _assert_within_bound(one_op: float, two_op: float, label: str, what: str) -> None:
    delta = one_op - two_op
    if delta <= _MAX_DISPATCH_OVERHEAD_NS:
        return
    ratio = one_op / two_op
    assert what == "period" and ratio <= _MAX_PERIOD_RATIO, (
        f"[{label}] one-op {what} costs {delta:+.0f} ns over the two-op forward "
        f"({one_op:.0f} ns vs {two_op:.0f} ns, {ratio:.4f}x), past the "
        f"{_MAX_DISPATCH_OVERHEAD_NS} ns per-dispatch budget"
    )


def _idx_tensor(device, values):
    return ttnn.from_torch(
        torch.tensor(values, dtype=torch.int32), layout=ttnn.ROW_MAJOR_LAYOUT, device=device, dtype=ttnn.uint32
    )


def _measure_pair(device, active_counts, threshold: int):
    """Both arrangements on one load: (two_op_ns, one_op_ns, two_op_period, one_op_period).

    Same inputs, same weights, same threshold -- the only difference is whether the layer is
    dispatched as two programs or one. The pair is measured in one process against one device, so
    the ratio between them needs no absolute calibration and does not drift with the machine.
    """
    require_realtime_profiler("the one-op vs two-op routed-expert comparison")

    emb_dim = DeepSeekV3Config.EMB_SIZE
    hidden_dim = DeepSeekV3Config.MOE_INTERMEDIATE_SIZE
    experts_per_chip = len(active_counts)
    torch.manual_seed(42)

    weights = [
        {
            "gate_proj": torch.randn(hidden_dim, emb_dim, dtype=torch.float32) * 0.02,
            "up_proj": torch.randn(hidden_dim, emb_dim, dtype=torch.float32) * 0.02,
            "down_proj": torch.randn(emb_dim, hidden_dim, dtype=torch.float32) * 0.02,
        }
        for _ in range(experts_per_chip)
    ]

    total_rows = _ALLOCATED_TOKENS * experts_per_chip
    torch_input = torch.zeros(total_rows, emb_dim, dtype=torch.float32)
    for e, count in enumerate(active_counts):
        base = e * _ALLOCATED_TOKENS
        torch_input[base : base + count] = torch.randn(count, emb_dim, dtype=torch.float32)

    # ROW_MAJOR: the layout production feeds the routed expert.
    tt_input = ttnn.from_torch(
        torch_input,
        mesh_mapper=ttnn.ReplicateTensorToMesh(device),
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        dtype=ttnn.bfloat16,
    )
    idx_tt = _idx_tensor(device, list(range(experts_per_chip)))
    counts_tt = _idx_tensor(device, active_counts)
    offsets_tt = _idx_tensor(device, [e * _ALLOCATED_TOKENS for e in range(experts_per_chip)])

    module = TtRoutedExpert(
        mesh_device=device,
        experts_per_chip=experts_per_chip,
        global_expert_idx_table=idx_tt,
        emb_dim=emb_dim,
        hidden_dim=hidden_dim,
        max_tokens=_ALLOCATED_TOKENS,
        torch_weights=weights,
        activation=ttnn.RoutedExpertActivation.Silu,
        hybrid_token_threshold=threshold,
    )

    def run_two_ops():
        return module(tt_input, counts_tt, offsets_tt)

    def run_one_op():
        return ttnn.experimental.deepseek_prefill.hybrid_routed_expert_moe(
            tt_input,
            offsets_tt,
            counts_tt,
            idx_tt,
            module.gate_projs,
            module.up_projs,
            module.down_projs,
            max_dispatched_tokens_per_expert=_ALLOCATED_TOKENS,
            hybrid_token_threshold=threshold,
            compute_kernel_config=module.compute_kernel_config,
            activation=ttnn.RoutedExpertActivation.Silu,
        )

    def measure(run_fn, kernel_dirs, label, expected_programs):
        run_fn()  # program cache + JIT

        def run_all():
            for _ in range(_ITERS):
                run_fn()

        _, records = profile_realtime_program(device, run_all, collect_all=True)
        chips = {r["chip_id"] for r in records}
        assert len(chips) == 1, f"{label}: timestamps are per-chip clocks, saw {chips}"
        # One dispatch can emit several records under one runtime_id, so a program's span runs from
        # its earliest start to its latest end.
        spans: dict = {}
        for r in records:
            if not r["runtime_id"]:
                continue
            e = spans.setdefault(
                r["runtime_id"], {"start": r["start_ns"], "end": r["end_ns"], "src": r["kernel_sources"]}
            )
            e["start"] = min(e["start"], r["start_ns"])
            e["end"] = max(e["end"], r["end_ns"])
        # Only this arrangement's own programs: a window also carries unrelated ones, and records
        # arrive asynchronously, so the warm-up dispatch can land here too.
        mine = sorted(
            (e for e in spans.values() if any(d in src for src in e["src"] for d in kernel_dirs)),
            key=lambda e: e["start"],
        )
        per_iter = len(mine) // _ITERS
        assert per_iter == expected_programs, (
            f"{label}: expected {expected_programs} program(s) per iteration, saw {per_iter} "
            f"({len(mine)} matches over {_ITERS} iterations)"
        )
        mine = mine[-_ITERS * per_iter :]

        # Device duration: the programs' own time, excluding every turnaround between them.
        totals = [sum(m["end"] - m["start"] for m in mine[i * per_iter : (i + 1) * per_iter]) for i in range(_ITERS)]
        # Period: start of one iteration's first program to the start of the next's, so the
        # turnaround this arrangement actually pays is inside it.
        periods = [mine[i * per_iter]["start"] - mine[(i - 1) * per_iter]["start"] for i in range(1, _ITERS)]
        gaps = [b["start"] - a["end"] for a, b in zip(mine, mine[1:])]
        median, period = statistics.median(totals), statistics.median(periods)
        logger.info(
            f"{label}: device {median:.0f} ns, period {period:.0f} ns, "
            f"mean dispatch turnaround {statistics.mean(gaps):.0f} ns over {_ITERS} runs"
        )
        return median, period

    two_op_ns, two_op_period = measure(run_two_ops, ("/unified_routed_expert_ffn/", "/moe_fused_swiglu/"), "two-op", 2)
    one_op_ns, one_op_period = measure(run_one_op, ("/hybrid_routed_expert_ffn/",), "one-op", 1)
    return two_op_ns, one_op_ns, two_op_period, one_op_period


@pytest.mark.skipif(not is_blackhole(), reason="the routed expert is Blackhole-only")
@pytest.mark.parametrize("device_params", [{"worker_l1_size": _UNION_WORKER_L1_SIZE}], indirect=True)
@pytest.mark.parametrize("distribution", list(_DISTRIBUTIONS), ids=list(_DISTRIBUTIONS))
def test_one_op_vs_two_ops(device, distribution: str):
    two_op_ns, one_op_ns, two_op_period, one_op_period = _measure_pair(device, _DISTRIBUTIONS[distribution], _THRESHOLD)
    logger.info(
        f"RESULT[{distribution}] one-op {one_op_ns:.0f} ns vs two-op {two_op_ns:.0f} ns "
        f"-> delta {one_op_ns - two_op_ns:+.0f} ns, {one_op_ns / two_op_ns:.3f}x | "
        f"period {one_op_period:.0f} vs {two_op_period:.0f} ns, {one_op_period / two_op_period:.3f}x"
    )
    _assert_within_bound(one_op_ns, two_op_ns, distribution, "device")
    _assert_within_bound(one_op_period, two_op_period, distribution, "period")


# Experts per chip, with a FIXED per-expert load so total work scales linearly with the count
# while the number of dispatches does not. The barrier is one grid rendezvous per dispatch, so
# if the union op's overhead is the barrier it stays flat across this series; if it is per-expert
# (a skip sweep that the two-op forward does not also pay) it grows with N.
#
# Counts alternate below/above the threshold so BOTH halves carry real work at every N.
_EXPERTS_PER_CHIP_SWEEP = [1, 2, 4, 8]
_LOW_COUNT = 128
_HIGH_COUNT = 1024


def _alternating_counts(n_experts: int) -> list:
    return [(_LOW_COUNT if e % 2 == 0 else _HIGH_COUNT) for e in range(n_experts)]


@pytest.mark.skipif(not is_blackhole(), reason="the routed expert is Blackhole-only")
@pytest.mark.parametrize("device_params", [{"worker_l1_size": _UNION_WORKER_L1_SIZE}], indirect=True)
@pytest.mark.parametrize("n_experts", _EXPERTS_PER_CHIP_SWEEP)
def test_overhead_vs_experts_per_chip(device, n_experts: int):
    """Does the union op's overhead scale with experts per chip, or is it per dispatch?"""
    counts = _alternating_counts(n_experts)
    two_op_ns, one_op_ns, two_op_period, one_op_period = _measure_pair(device, counts, _THRESHOLD)
    logger.info(
        f"NEXPERTS n={n_experts} counts={counts} "
        f"two_op={two_op_ns:.0f}ns one_op={one_op_ns:.0f}ns "
        f"delta={one_op_ns - two_op_ns:+.0f}ns ratio={one_op_ns / two_op_ns:.4f} | "
        f"period ratio={one_op_period / two_op_period:.4f}"
    )
    _assert_within_bound(one_op_ns, two_op_ns, f"n={n_experts}", "device")
    _assert_within_bound(one_op_period, two_op_period, f"n={n_experts}", "period")
