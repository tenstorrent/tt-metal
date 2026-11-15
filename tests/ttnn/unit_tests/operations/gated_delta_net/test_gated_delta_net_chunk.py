# SPDX-FileCopyrightText: (c) 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Regime-pinned tests: R-CHUNK (one closed-form chunk step per tile row, state carried in fp32 across
chunks) vs the token step on the same multi-token inputs. CHUNK_MIN_TOKENS pins the regime: 2 = every
multi-token call chunked (the default), a huge value = token steps. Covers T < 32, T not a multiple of 32,
batched calls whose sequence boundary falls inside a tile row, calls that start mid-sequence from a
non-zero state and history, inf padding, and a state carried over several chunked calls."""

import pytest
import torch

import ttnn
from ttnn.operations.gated_delta_net import gated_delta_net
from ttnn.operations.gated_delta_net import gated_delta_net_program_descriptor as gdn_pd
from tests.ttnn.unit_tests.operations.gated_delta_net.test_gated_delta_net import (
    ORDER,
    check_call,
    compute_config,
    make_inputs,
    reference,
    to_device,
)

TOKEN_STEP = 1 << 30


@pytest.mark.parametrize("chunk_min_tokens", [2, TOKEN_STEP], ids=["chunk", "token_step"])
@pytest.mark.parametrize(
    "inputs",
    [
        pytest.param(((1, 2, 256), (1, 3, 32, 64)), id="t2"),
        pytest.param(((1, 5, 256), (1, 3, 32, 64)), id="t5"),
        pytest.param(((1, 32, 256), (1, 3, 32, 64)), id="t32"),
        pytest.param(((1, 33, 256), (1, 3, 32, 64)), id="t33"),
        pytest.param(((1, 100, 256), (1, 3, 32, 64)), id="t100"),
        pytest.param(((1, 40, 320), (1, 2, 64, 32)), id="k_gt_v"),
        pytest.param(((1, 10, 256), (2, 3, 32, 64)), id="batched_b2_t5"),
        pytest.param(((1, 66, 256), (2, 3, 32, 64)), id="batched_b2_t33"),
        pytest.param(((1, 5, 1760), (1, 33, 32, 32)), id="hv33"),
        pytest.param(((1, 70, 2560), (1, 12, 128, 128)), id="chip_t70"),
    ],
)
@pytest.mark.parametrize("poison", [None, float("inf")], ids=["clean", "inf_padding"])
def test_chunk_vs_token_step(device, monkeypatch, chunk_min_tokens, inputs, poison):
    monkeypatch.setattr(gdn_pd, "CHUNK_MIN_TOKENS", chunk_min_tokens)
    torch.manual_seed(5)
    tensors = make_inputs(inputs)
    expected = reference(*(tensors[name] for name in ORDER))
    returned = gated_delta_net(*to_device(tensors, device, poison=poison), compute_kernel_config=compute_config())
    check_call(returned, expected, inputs)


def test_chunked_state_carry(device):
    """Three chunked calls (T = 45, 32, 7) then a decode call, each from the previous call's states."""
    torch.manual_seed(9)
    head = (1, 3, 32, 64)
    channels = 256
    calls = [45, 32, 7, 1]
    full = make_inputs(((1, sum(calls), channels), head))
    hist, state = full["conv_state"], full["recurrent_state"]
    hist_dev, state_dev = None, None
    start = 0
    for tokens in calls:
        part = dict(full)
        for name in ("mixed_qkv", "z", "b", "a"):
            part[name] = full[name][:, start : start + tokens]
        part["conv_state"], part["recurrent_state"] = hist, state
        expected = reference(*(part[name] for name in ORDER))
        dev = to_device(part, device)
        if hist_dev is not None:
            dev[ORDER.index("conv_state")], dev[ORDER.index("recurrent_state")] = hist_dev, state_dev
        _, hist_dev, state_dev = check_call(
            gated_delta_net(*dev, compute_kernel_config=compute_config()), expected, ((1, tokens, channels), head)
        )
        hist, state = expected[1], expected[2]
        start += tokens
