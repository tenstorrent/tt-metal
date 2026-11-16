# SPDX-FileCopyrightText: (c) 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Refinement 27 under a ttnn trace: the packed rows live in one L1 tensor that is rewritten in place before every
replay, the states are updated in place, a matmul follows the op inside the trace. Every replay must match, to the
bit, the untraced split DRAM chain on the same data (run without the profiler)."""

import pytest
import torch

import ttnn
from ttnn.operations.gated_delta_net import gated_delta_net
from tests.ttnn.unit_tests.operations.gated_delta_net.test_gated_delta_net import ORDER, compute_config, make_inputs
from tests.ttnn.unit_tests.operations.gated_delta_net.test_gated_delta_net_packed import (
    DRAM,
    L1,
    TOKEN_INPUTS,
    chip,
    pack,
    upload,
)

TRACE = {"trace_region_size": 16 * 1024 * 1024}
BF16 = ttnn.bfloat16


@pytest.mark.use_module_device(TRACE)
@pytest.mark.parametrize(
    "inputs, replays, token_memory",
    [
        pytest.param(chip(2048), 8, L1, id="prefill_t2048_l1_x8"),
        pytest.param(chip(8, 8), 64, L1, id="decode_b8_l1_x64"),
        pytest.param(chip(1), 16, L1, id="decode_b1_l1_x16"),
        pytest.param(chip(32, 32), 8, L1, id="decode_b32_l1_x8"),
        pytest.param(chip(128), 8, DRAM, id="prefill_t128_dram_x8"),
    ],
)
def test_packed_trace_replays(device, inputs, replays, token_memory):
    torch.manual_seed(11)
    steps = [make_inputs(inputs) for _ in range(replays + 1)]  # compile run, then one step per replay
    first = steps[0]
    config = compute_config()
    weights = [upload(first[name], device, BF16) for name in ORDER[6:]]

    conv_ref = upload(first["conv_state"], device, BF16)
    rec_ref = upload(first["recurrent_state"], device, ttnn.float32)
    want = []
    for step in steps:
        tokens = [upload(step[name], device, BF16) for name in TOKEN_INPUTS]
        out, conv_ref, rec_ref = gated_delta_net(*tokens, conv_ref, rec_ref, *weights, compute_kernel_config=config)
        want.append(ttnn.to_torch(out))
        for t in (*tokens, out):
            ttnn.deallocate(t)
    want_states = (ttnn.to_torch(conv_ref), ttnn.to_torch(rec_ref))

    conv_state = upload(first["conv_state"], device, BF16)
    rec_state = upload(first["recurrent_state"], device, ttnn.float32)
    packed = upload(pack(first), device, BF16, token_memory)
    mm_a = upload(torch.randn(512, 2048), device, BF16)
    mm_b = upload(torch.randn(2048, 2048), device, BF16)

    def body():
        out, _, _ = gated_delta_net(
            packed,
            None,
            None,
            None,
            conv_state,
            rec_state,
            *weights,
            compute_kernel_config=config,
            conv_state_out=conv_state,
            recurrent_state_out=rec_state,
            memory_config=DRAM,
        )
        return out, ttnn.matmul(mm_a, mm_b)

    def rewrite(step):
        host = ttnn.from_torch(pack(steps[step]), dtype=BF16, layout=ttnn.TILE_LAYOUT)
        ttnn.copy_host_to_device_tensor(host, packed)

    out, mm = body()
    assert torch.equal(ttnn.to_torch(out), want[0]), "compile run differs"
    mm_want = ttnn.to_torch(mm)
    ttnn.deallocate(out)
    ttnn.deallocate(mm)

    # The capture records the programs without running them: the states stay at step 0's.
    tid = ttnn.begin_trace_capture(device, cq_id=0)
    out, mm = body()
    ttnn.end_trace_capture(device, tid, cq_id=0)
    for step in range(1, replays + 1):
        rewrite(step)
        ttnn.execute_trace(device, tid, cq_id=0, blocking=False)
        assert torch.equal(ttnn.to_torch(out), want[step]), f"output differs at replay {step}"
        assert torch.equal(ttnn.to_torch(mm), mm_want), f"the following matmul differs at replay {step}"
    assert torch.equal(ttnn.to_torch(conv_state), want_states[0]), "conv_state differs"
    assert torch.equal(ttnn.to_torch(rec_state), want_states[1]), "recurrent_state differs"
    ttnn.release_trace(device, tid)
