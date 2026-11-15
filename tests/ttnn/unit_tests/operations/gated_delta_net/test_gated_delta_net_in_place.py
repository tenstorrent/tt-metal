# SPDX-FileCopyrightText: (c) 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Refinement 10: caller-provided state outputs. A chain of calls that updates conv_state / recurrent_state in
place must match the default chain (new state tensors each call) to the bit, on every kernel path; provided outputs
of the wrong form, or aliasing another input, are refused."""

import pytest
import torch

import ttnn
from ttnn.operations.gated_delta_net import gated_delta_net
from tests.ttnn.unit_tests.operations.gated_delta_net.test_gated_delta_net import ORDER, compute_config, make_inputs

CALLS = 8
TOKEN_INPUTS = ("mixed_qkv", "z", "b", "a")


def _upload(tensor, device, dtype):
    return ttnn.from_torch(
        tensor, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )


def _device_inputs(tensors, device, token_dtype):
    return {
        name: _upload(tensors[name], device, ttnn.float32 if name == "recurrent_state" else token_dtype)
        for name in ORDER
    }


SHAPES = [
    pytest.param(((1, 1, 2560), (1, 12, 128, 128)), id="decode_b1"),
    pytest.param(((1, 8, 2560), (8, 12, 128, 128)), id="decode_b8"),
    pytest.param(((1, 16, 2560), (16, 12, 128, 128)), id="decode_b16"),
    pytest.param(((1, 40, 2560), (40, 12, 128, 128)), id="decode_b40"),
    pytest.param(((1, 33, 2560), (1, 12, 128, 128)), id="prefill_t33"),
    pytest.param(((1, 96, 2560), (1, 12, 128, 128)), id="prefill_t96"),
    pytest.param(((1, 4096, 2560), (1, 12, 128, 128)), id="prefill_t4096"),
    pytest.param(((1, 80, 2560), (2, 12, 128, 128)), id="batched_b2_t40"),
    pytest.param(((1, 4, 2560), (2, 12, 128, 128)), id="batched_b2_t2"),
]


@pytest.mark.parametrize("token_dtype", [ttnn.bfloat16, ttnn.float32], ids=["bf16", "f32"])
@pytest.mark.parametrize("inputs", SHAPES)
def test_in_place_matches_default(device, inputs, token_dtype):
    torch.manual_seed(0)
    first = make_inputs(inputs)
    steps = [first] + [make_inputs(inputs) for _ in range(CALLS - 1)]
    weights = _device_inputs(first, device, token_dtype)
    tokens = [{name: _upload(step[name], device, token_dtype) for name in TOKEN_INPUTS} for step in steps]
    config = compute_config()

    def call(step, conv_state, recurrent_state, **outs):
        args = {**weights, **tokens[step], "conv_state": conv_state, "recurrent_state": recurrent_state}
        return gated_delta_net(*(args[name] for name in ORDER), compute_kernel_config=config, **outs)

    default_out = []
    conv_state, recurrent_state = weights["conv_state"], weights["recurrent_state"]
    for step in range(CALLS):
        out, conv_state, recurrent_state = call(step, conv_state, recurrent_state)
        default_out.append(ttnn.to_torch(out))
    default_states = (ttnn.to_torch(conv_state), ttnn.to_torch(recurrent_state))

    conv_state = _upload(first["conv_state"], device, token_dtype)
    recurrent_state = _upload(first["recurrent_state"], device, ttnn.float32)
    for step in range(CALLS):
        out, cs, rs = call(
            step, conv_state, recurrent_state, conv_state_out=conv_state, recurrent_state_out=recurrent_state
        )
        assert cs.buffer_address() == conv_state.buffer_address()
        assert rs.buffer_address() == recurrent_state.buffer_address()
        assert torch.equal(ttnn.to_torch(out), default_out[step]), f"output differs at call {step}"
    assert torch.equal(ttnn.to_torch(conv_state), default_states[0]), "conv_state differs"
    assert torch.equal(ttnn.to_torch(recurrent_state), default_states[1]), "recurrent_state differs"


def test_distinct_outputs_and_refusals(device, expect_error):
    torch.manual_seed(0)
    inputs = ((1, 1, 2560), (1, 12, 128, 128))
    tensors = make_inputs(inputs)
    args = _device_inputs(tensors, device, ttnn.bfloat16)
    ordered = [args[name] for name in ORDER]
    config = compute_config()
    out, cs, rs = gated_delta_net(*ordered, compute_kernel_config=config)

    cs_buf = _upload(torch.zeros_like(tensors["conv_state"]), device, ttnn.bfloat16)
    rs_buf = _upload(torch.zeros_like(tensors["recurrent_state"]), device, ttnn.float32)
    out2, cs2, rs2 = gated_delta_net(
        *ordered, compute_kernel_config=config, conv_state_out=cs_buf, recurrent_state_out=rs_buf
    )
    assert cs2.buffer_address() == cs_buf.buffer_address() and rs2.buffer_address() == rs_buf.buffer_address()
    for got, ref in ((out2, out), (cs2, cs), (rs2, rs)):
        assert torch.equal(ttnn.to_torch(got), ttnn.to_torch(ref))

    bad = [
        {"recurrent_state_out": _upload(torch.zeros(1, 12, 128, 64), device, ttnn.float32)},  # shape
        {"conv_state_out": _upload(torch.zeros_like(tensors["conv_state"]), device, ttnn.float32)},  # dtype
        {"recurrent_state_out": rs_buf, "conv_state_out": args["mixed_qkv"]},  # another input
        {
            "conv_state_out": ttnn.from_torch(
                tensors["conv_state"],
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )
        },  # memory config
    ]
    for outs in bad:
        with expect_error(ValueError, "gated_delta_net"):
            gated_delta_net(*ordered, compute_kernel_config=config, **outs)
