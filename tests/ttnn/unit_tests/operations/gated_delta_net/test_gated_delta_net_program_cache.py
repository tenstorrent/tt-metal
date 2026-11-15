# SPDX-FileCopyrightText: (c) 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Refinement 13: the program descriptor is built once per cache key and re-sent with new buffer addresses. Every call
on a warm cache must match, to the bit, the same call on a cold cache: alternating shapes (keys do not collide), the
same shape with other tensors, in-place and caller-provided state outputs."""

import importlib

import torch

import ttnn
from ttnn.operations.gated_delta_net import gated_delta_net
from tests.ttnn.unit_tests.operations.gated_delta_net.test_gated_delta_net import ORDER, compute_config, make_inputs

gdn_pd = importlib.import_module("ttnn.operations.gated_delta_net.gated_delta_net_program_descriptor")

DECODE = ((1, 1, 2560), (1, 12, 128, 128))
PREFILL = ((1, 128, 2560), (1, 12, 128, 128))
SMALL = ((1, 64, 384), (1, 4, 32, 32))


def _upload(tensors, device, token_dtype=ttnn.bfloat16):
    return [
        ttnn.from_torch(
            tensors[name],
            dtype=ttnn.float32 if name == "recurrent_state" else token_dtype,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        for name in ORDER
    ]


def _run(args, **outs):
    return [ttnn.to_torch(t) for t in gated_delta_net(*args, compute_kernel_config=compute_config(), **outs)]


def _assert_equal(got, want, label):
    for g, w, name in zip(got, want, ("output", "conv_state_out", "recurrent_state_out")):
        assert torch.equal(g, w), f"{label}: {name} differs from the cold-cache call"


def test_alternating_shapes(device):
    shapes = (DECODE, PREFILL, SMALL)
    data = {}
    cold = {}
    for idx, shape in enumerate(shapes):
        torch.manual_seed(idx)
        data[shape] = make_inputs(shape)
        gdn_pd.clear_program_cache()
        cold[shape] = _run(_upload(data[shape], device))
    gdn_pd.clear_program_cache()
    for rnd in range(3):
        for shape in shapes:
            _assert_equal(_run(_upload(data[shape], device)), cold[shape], f"round {rnd} shape {shape}")
        assert gdn_pd.program_cache_size() == len(shapes)


def test_same_shape_other_tensors_and_in_place(device):
    for shape in (DECODE, PREFILL):
        torch.manual_seed(7)
        first, second = make_inputs(shape), make_inputs(shape)
        gdn_pd.clear_program_cache()
        cold_first = _run(_upload(first, device))
        gdn_pd.clear_program_cache()
        cold_second = _run(_upload(second, device))

        _assert_equal(_run(_upload(first, device)), cold_first, f"{shape} other tensors")
        args = _upload(second, device)
        _assert_equal(_run(args, conv_state_out=args[4], recurrent_state_out=args[5]), cold_second, f"{shape} in place")
        assert torch.equal(ttnn.to_torch(args[4]), cold_second[1]) and torch.equal(
            ttnn.to_torch(args[5]), cold_second[2]
        ), f"{shape}: in-place states not updated"
        # Caller-provided distinct outputs, twice: the second call hits the entry the first one built.
        for _ in range(2):
            args = _upload(first, device)
            outs = _upload(
                {**first, "conv_state": first["conv_state"] * 0, "recurrent_state": first["recurrent_state"] * 0},
                device,
            )
            got = _run(args, conv_state_out=outs[4], recurrent_state_out=outs[5])
            _assert_equal(got, cold_first, f"{shape} provided outputs")
        # Alternating with the in-place key of the same shape.
        args = _upload(first, device)
        _assert_equal(_run(args, conv_state_out=args[4], recurrent_state_out=args[5]), cold_first, f"{shape} in place")


def test_aliasing_refused_on_cache_hit(device, expect_error):
    """T = 3: mixed_qkv has the form of conv_state, so only the per-call aliasing check can refuse it as an output."""
    shape = ((1, 3, 384), (1, 4, 32, 32))
    torch.manual_seed(3)
    tensors = make_inputs(shape)
    args = _upload(tensors, device)
    spare = _upload(tensors, device)
    _run(args, conv_state_out=spare[4])  # builds the entry for a distinct conv_state_out
    with expect_error(ValueError, "gated_delta_net"):
        gated_delta_net(*args, compute_kernel_config=compute_config(), conv_state_out=args[0])
