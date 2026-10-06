# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""
Device time of ttnn.practice_routed_expert next to the production unified_routed_expert_moe. Both run
one Kimi K3 expert on the same inputs, in the production formats (bf8 activations, bf4 weights, DRAM
interleaved), timed by the real-time program profiler. No baseline is asserted yet: each case logs the
median of ITERS runs.
"""

import statistics

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import is_blackhole
from models.demos.deepseek_v3_d_p.tt.moe.tt_routed_expert import COMPUTE_KERNEL_CONFIG_LOFI
from tests.ttnn.profiling.realtime_profiler_utils import profile_realtime_program_merged, require_realtime_profiler
from tests.ttnn.unit_tests.operations.practice_routed_expert.test_practice_routed_expert import (
    K3_EMB,
    K3_HIDDEN,
    make_inputs,
    to_device,
    torch_reference,
)
from tests.ttnn.utils_for_testing import comp_pcc

ITERS = 3
TOKENS = [32, 128, 512, 2048, 5120]
PCC_THRESHOLD = 0.96

# The profiler names a program by its kernel sources, not by its op, so each op is found by the
# directory its kernels live in.
KERNEL_DIRS = {
    "practice": "/practice_routed_expert/",
    "production": "/unified_routed_expert_ffn/",
}


def production_op(device, tokens):
    """unified_routed_expert_moe with one expert owning every row: offset 0, count = tokens."""

    def uint32_vector(value):
        return ttnn.from_torch(
            torch.tensor([value], dtype=torch.int32), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device
        )

    offsets, counts, expert_ids = uint32_vector(0), uint32_vector(tokens), uint32_vector(0)

    def run(x, w_gate, w_up, w_down):
        return ttnn.experimental.deepseek_prefill.unified_routed_expert_moe(
            x,
            offsets,
            counts,
            expert_ids,
            [w_gate],
            [w_up],
            [w_down],
            max_dispatched_tokens_per_expert=tokens,
            compute_kernel_config=COMPUTE_KERNEL_CONFIG_LOFI,
            activation=ttnn.RoutedExpertActivation.SituGlu,
        )

    return run


def median_device_us(device, run, kernel_dir):
    """assert_op_duration_merged without its baseline band, which needs numbers the practice op does not have yet."""

    def run_all():
        for _ in range(ITERS):
            run()

    _, programs = profile_realtime_program_merged(device, run_all)
    durations = [
        program["duration_ns"]
        for program in programs.values()
        if any(kernel_dir in source for source in program["kernel_sources"])
    ]
    assert len(durations) == ITERS, f"expected one {kernel_dir} program per run, got {len(durations)} in {ITERS} runs"
    return statistics.median(durations) / 1000


@pytest.mark.parametrize("tokens", TOKENS, ids=[f"{t}tok" for t in TOKENS])
@pytest.mark.parametrize("impl", ["practice", "production"])
@pytest.mark.skipif(not is_blackhole(), reason="SiTU-GLU's SFPU op is Blackhole-only")
def test_practice_routed_expert_perf(device, impl, tokens):
    require_realtime_profiler("practice routed expert perf")

    x, weights = make_inputs(tokens, K3_EMB, K3_HIDDEN, gate_up_std=1.2)
    inputs = to_device(x, weights, ttnn.bfloat8_b, ttnn.bfloat4_b, device)
    op = ttnn.practice_routed_expert if impl == "practice" else production_op(device, tokens)

    # Checked before timing so an op cannot look fast by computing the wrong thing. Read right away:
    # the production op writes its output over x, which the timed runs then reuse.
    _, pcc = comp_pcc(torch_reference(x, weights), ttnn.to_torch(op(*inputs)))
    assert pcc >= PCC_THRESHOLD, f"{impl}: PCC {pcc:.5f} below {PCC_THRESHOLD}"

    us = median_device_us(device, lambda: op(*inputs), KERNEL_DIRS[impl])
    logger.info(f"practice_routed_expert perf: {impl} {tokens} tokens = {us:.1f} us")
