# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Supplementary checks for gen_qkv_causal_conv1d_silu (DO NOT DELETE — implementer artifacts).

test_repeat_is_bit_identical in the acceptance file cannot run: it parametrizes with pytest.param
objects (which pytest unpacks to the dict) and then calls `case.values[0]` on that dict
(TypeError before the op is called). The test below is the same BB-06 check on the same two
golden cells (conv1d-001 RM, conv1d-005 TILE), using the acceptance file's own `_run` helper.

Also: the program-cache (BB-07) hit path — a second call with new tensors of the same spec must
reuse the cached descriptor and still produce the correct (bit-identical) result.
"""

import pytest
import torch
import ttnn

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
