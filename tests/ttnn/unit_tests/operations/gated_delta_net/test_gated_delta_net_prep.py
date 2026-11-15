# SPDX-FileCopyrightText: (c) 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Regime-pinned tests for R-PREP (prep cores feed the state cores' inboxes; the state cores run only the chunk
step's state part). PREP_MIN_TILE_ROWS = 1 forces R-PREP on short calls that the default keeps local, so a
sequence of one tile row (one prep core per group, one inbox slot), batched sequences sharing a tile row, a
partial last tile row, inf padding and one sequence's tile rows spread over several prep cores are all covered;
R-PREP off (PREP_OFFLOAD = False) is the local chunk step on the same inputs. The regime predicate is checked on
the descriptor helper."""

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


def test_prep_predicate(device):
    budget = gdn_pd.l1_cb_budget(device)
    # model_prefill_4096 (bf16): 2 pieces, every idle core preps, the key head's 3 value heads per prep group.
    assert gdn_pd._prep_layout(1, 12, 4, 4096, 4, 4, 2, True, ttnn.bfloat16, 110 - 24, budget)[2] == 3
    # Short calls (2 to 16 tile rows): one value head per prep group (Refinement 11).
    for tokens in (64, 96, 128, 512):
        assert gdn_pd._prep_layout(1, 12, 4, tokens, 4, 4, 2, True, ttnn.bfloat16, 86, budget)[2] == 1
    # Too few tile rows, not chunked, one piece per pair, or no idle core: local.
    assert gdn_pd._prep_layout(1, 12, 4, 32, 4, 4, 2, True, ttnn.bfloat16, 86, budget) is None
    assert gdn_pd._prep_layout(1, 12, 4, 4096, 4, 4, 2, False, ttnn.bfloat16, 86, budget) is None
    assert gdn_pd._prep_layout(1, 12, 4, 4096, 4, 4, 1, True, ttnn.bfloat16, 98, budget) is None
    assert gdn_pd._prep_layout(1, 12, 4, 4096, 4, 4, 2, True, ttnn.bfloat16, 3, budget) is None


@pytest.mark.parametrize("offload", [True, False], ids=["prep", "local"])
@pytest.mark.parametrize(
    "inputs",
    [
        pytest.param(((1, 2, 256), (1, 3, 32, 64)), id="t2"),
        pytest.param(((1, 33, 256), (1, 3, 32, 64)), id="t33"),
        pytest.param(((1, 200, 256), (1, 3, 32, 64)), id="t200"),
        pytest.param(((1, 66, 256), (2, 3, 32, 64)), id="batched_b2_t33"),
        pytest.param(((1, 70, 2560), (1, 12, 128, 128)), id="chip_t70"),
    ],
)
@pytest.mark.parametrize("poison", [None, float("inf")], ids=["clean", "inf_padding"])
def test_prep_vs_local(device, monkeypatch, offload, inputs, poison):
    monkeypatch.setattr(gdn_pd, "PREP_MIN_TILE_ROWS", 1)
    monkeypatch.setattr(gdn_pd, "PREP_OFFLOAD", offload)
    torch.manual_seed(11)
    tensors = make_inputs(inputs)
    expected = reference(*(tensors[name] for name in ORDER))
    returned = gated_delta_net(*to_device(tensors, device, poison=poison), compute_kernel_config=compute_config())
    check_call(returned, expected, inputs)


def test_prep_state_carry(device, monkeypatch):
    """Two R-PREP calls (T = 100, 40) then a decode call, each from the previous call's states."""
    monkeypatch.setattr(gdn_pd, "PREP_MIN_TILE_ROWS", 1)
    torch.manual_seed(13)
    head = (1, 3, 32, 64)
    channels = 256
    calls = [100, 40, 1]
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
