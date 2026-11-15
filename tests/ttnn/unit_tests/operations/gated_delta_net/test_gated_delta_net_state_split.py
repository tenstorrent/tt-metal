# SPDX-FileCopyrightText: (c) 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Refinement 15: R-QKCORE pair cores whose reader writes part of the final state (STATE_SPLIT tiles, every state of
STATE_WRITE_NOC = 1 on the swapped rows, the non-last units' states under STATE_READER_EARLY_UNITS). The same tiles
reach the same pages from the other RISC, so every output must match the default build bit for bit, in place too."""

import pytest
import torch

import ttnn
from ttnn.operations.gated_delta_net import gated_delta_net
from ttnn.operations.gated_delta_net import gated_delta_net_program_descriptor as gdn_pd
from tests.ttnn.unit_tests.operations.gated_delta_net.test_gated_delta_net import ORDER, compute_config, make_inputs
from tests.ttnn.unit_tests.operations.gated_delta_net.test_gated_delta_net_bcast import _to_device

LEVERS = {
    "split2": {"STATE_SPLIT": 2},
    "split1_noc1": {"STATE_SPLIT": 1, "STATE_WRITE_NOC": 1},  # every tile on the reader except swapped rows' writers
    "noc1_early": {"STATE_WRITE_NOC": 1, "STATE_READER_EARLY_UNITS": True},
    "split3_early": {"STATE_SPLIT": 3, "STATE_READER_EARLY_UNITS": True},
}


def _run(device, tensors, in_place):
    device_inputs = _to_device(tensors, device, None, ttnn.bfloat16)
    kwargs = {}
    if in_place:
        kwargs = {
            "conv_state_out": device_inputs[ORDER.index("conv_state")],
            "recurrent_state_out": device_inputs[ORDER.index("recurrent_state")],
        }
    outs = gated_delta_net(*device_inputs, compute_kernel_config=compute_config(), **kwargs)
    return [ttnn.to_torch(t).to(torch.float32) for t in outs]


@pytest.mark.parametrize("lever", list(LEVERS))
@pytest.mark.parametrize(
    "inputs",
    [
        pytest.param(((1, 8, 2560), (8, 12, 128, 128)), id="b8"),  # one unit per pair core
        pytest.param(((1, 16, 2560), (16, 12, 128, 128)), id="b16"),  # two units per pair core
        pytest.param(((1, 27, 2560), (27, 12, 128, 128)), id="b27"),  # ragged last core
    ],
)
@pytest.mark.parametrize("in_place", [False, True], ids=["new", "in_place"])
def test_state_split_bit_identical(device, monkeypatch, inputs, lever, in_place):
    torch.manual_seed(3)
    tensors = make_inputs(inputs)
    for name in ORDER:
        if name != "recurrent_state":
            tensors[name] = tensors[name].to(torch.bfloat16).to(torch.float32)
    expected = _run(device, tensors, in_place)
    for name, value in LEVERS[lever].items():
        monkeypatch.setattr(gdn_pd, name, value)
    got = _run(device, tensors, in_place)
    for a, b, label in zip(got, expected, ("output", "conv_state_out", "recurrent_state_out")):
        assert torch.equal(a, b), f"{label}: {lever} differs from the default build"
