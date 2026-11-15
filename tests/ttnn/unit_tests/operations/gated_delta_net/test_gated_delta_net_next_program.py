# SPDX-FileCopyrightText: (c) 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Refinement 16: the op must leave every core's NoC counters quiescent. A chain of in-place calls, each followed by
a full-grid matmul whose kernels barrier on their own NoC atomics, must not hang (run without the profiler: its
end-of-kernel flush hides the defect), and must match the default chain to the bit."""

import pytest
import torch

import ttnn
from ttnn.operations.gated_delta_net import gated_delta_net
from tests.ttnn.unit_tests.operations.gated_delta_net.test_gated_delta_net import ORDER, compute_config, make_inputs

CALLS = 12
TOKEN_INPUTS = ("mixed_qkv", "z", "b", "a")


def _upload(tensor, device, dtype):
    return ttnn.from_torch(
        tensor, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )


SHAPES = [pytest.param(((1, t, 2560), (1, 12, 128, 128)), id=f"prefill_t{t}") for t in (64, 128, 256, 512, 640)] + [
    pytest.param(((1, b, 2560), (b, 12, 128, 128)), id=f"decode_b{b}") for b in (1, 8, 16)
]


@pytest.mark.parametrize("inputs", SHAPES)
def test_in_place_then_matmul(device, inputs):
    torch.manual_seed(0)
    steps = [make_inputs(inputs) for _ in range(CALLS)]
    first = steps[0]
    weights = {
        name: _upload(first[name], device, ttnn.float32 if name == "recurrent_state" else ttnn.bfloat16)
        for name in ORDER
    }
    tokens = [{name: _upload(step[name], device, ttnn.bfloat16) for name in TOKEN_INPUTS} for step in steps]
    mm_a = _upload(torch.randn(512, 2048), device, ttnn.bfloat16)
    mm_b = _upload(torch.randn(2048, 2048), device, ttnn.bfloat16)
    config = compute_config()

    def call(step, conv_state, recurrent_state, **outs):
        args = {**weights, **tokens[step], "conv_state": conv_state, "recurrent_state": recurrent_state}
        return gated_delta_net(*(args[name] for name in ORDER), compute_kernel_config=config, **outs)

    conv_state = _upload(first["conv_state"], device, ttnn.bfloat16)
    recurrent_state = _upload(first["recurrent_state"], device, ttnn.float32)
    in_place_out = []
    for step in range(CALLS):
        out, _, _ = call(
            step, conv_state, recurrent_state, conv_state_out=conv_state, recurrent_state_out=recurrent_state
        )
        ttnn.experimental.minimal_matmul(mm_a, mm_b)
        ttnn.matmul(mm_a, mm_b)
        in_place_out.append(out)
    ttnn.synchronize_device(device)

    conv_ref, recurrent_ref = weights["conv_state"], weights["recurrent_state"]
    for step in range(CALLS):
        out, conv_ref, recurrent_ref = call(step, conv_ref, recurrent_ref)
        assert torch.equal(ttnn.to_torch(in_place_out[step]), ttnn.to_torch(out)), f"output differs at call {step}"
    assert torch.equal(ttnn.to_torch(conv_state), ttnn.to_torch(conv_ref)), "conv_state differs"
    assert torch.equal(ttnn.to_torch(recurrent_state), ttnn.to_torch(recurrent_ref)), "recurrent_state differs"
