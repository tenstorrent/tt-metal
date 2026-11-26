# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Supplementary checks for gen_qkv_causal_conv1d_silu (DO NOT DELETE).

- BB-07 program-cache hit path: a second call with NEW operand tensors of the same spec reuses the
  cached descriptor (only buffer addresses are rewritten) and must still produce the correct,
  bit-identical result. The acceptance file's test_repeat_is_bit_identical checks bit-identity but
  not the second call's correctness against the reference.
- Regime pinning the acceptance file does not cover: the TILE regime with a multi-tile-row block
  (block_rows >= 2: several untilized tile-rows copied into one cb_rows slot before the windowed
  tilize). The default plan never selects it (OVERLAP_BLOCK_ROWS = 1), but any block_rows perf
  co-tune does, so it is pinned here on golden cell conv1d-005.

All cases are golden cells (BB-22), run through the acceptance file's own `_run` / `_check_case`.
"""

import pytest
import torch
import ttnn

from ttnn.operations.gen_qkv_causal_conv1d_silu import get_last_plan, plan_limits
from tests.ttnn.unit_tests.operations.gen_qkv_causal_conv1d_silu.test_gen_qkv_causal_conv1d_silu import (
    PHASE0_CASES,
    _check_case,
    _run,
)


@pytest.mark.parametrize("case_index", [0, 4], ids=["rm_conv1d-001", "tile_conv1d-005"])
def test_repeat_is_bit_identical_bb06(device, case_index):
    case = PHASE0_CASES[case_index].values[0]
    first, host, device_inputs = _run(device, case)
    second, _, _ = _run(device, case)  # new operand tensors, same spec -> program-cache hit
    _check_case(case, second, host, device_inputs)
    for a, b in zip(first, second):
        assert torch.equal(ttnn.to_torch(a), ttnn.to_torch(b)), "repeat call is not bit-identical"


def test_tile_multirow_block_pinned(device):
    """conv1d-005 (TILE, T 64) on one core per channel group with block_rows 2: one block of two
    untilized tile-rows per core."""
    case = PHASE0_CASES[4].values[0]
    with plan_limits(max_block_rows=2, max_cores_per_group=1):
        outputs, host, device_inputs = _run(device, case)
        plan = get_last_plan()
    assert plan.regime == "tile_untilize_staged_window"
    assert plan.block_rows == 2
    assert plan.cores_per_group == 1
    _check_case(case, outputs, host, device_inputs)
