# SPDX-FileCopyrightText: (c) 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Refinement 20: decode R-QKCORE state cores take A_log, dt_bias, the b / a gate tile and norm_weight from their key
head's q core (SMALL_INPUTS_MCAST, on by default) instead of reading DRAM. Same values in the rows compute uses, so every
output must match the build with the knob off bit for bit, in place and with poisoned tile padding too."""

import math

import pytest
import torch

import ttnn
from ttnn.operations.gated_delta_net import gated_delta_net
from ttnn.operations.gated_delta_net import gated_delta_net_program_descriptor as gdn_pd
from tests.ttnn.unit_tests.operations.gated_delta_net.test_gated_delta_net import ORDER, compute_config, make_inputs
from tests.ttnn.unit_tests.operations.gated_delta_net.test_gated_delta_net_bcast import _to_device


def _run(device, tensors, in_place, token_dtype, poison):
    device_inputs = _to_device(tensors, device, poison, token_dtype)
    kwargs = {}
    if in_place:
        kwargs = {
            "conv_state_out": device_inputs[ORDER.index("conv_state")],
            "recurrent_state_out": device_inputs[ORDER.index("recurrent_state")],
        }
    outs = gated_delta_net(*device_inputs, compute_kernel_config=compute_config(), **kwargs)
    return [ttnn.to_torch(t).to(torch.float32) for t in outs]


@pytest.mark.parametrize(
    "inputs, token_dtype, poison",
    [
        pytest.param(((1, 1, 2560), (1, 12, 128, 128)), ttnn.bfloat16, None, id="b1"),  # QK_CORES_VSPLIT pieces
        pytest.param(((1, 1, 2560), (1, 12, 128, 128)), ttnn.float32, None, id="b1_f32"),
        pytest.param(((1, 8, 2560), (8, 12, 128, 128)), ttnn.bfloat16, None, id="b8"),  # one unit per pair core
        pytest.param(((1, 8, 2560), (8, 12, 128, 128)), ttnn.bfloat16, math.inf, id="b8_inf_pad"),
        pytest.param(((1, 16, 2560), (16, 12, 128, 128)), ttnn.bfloat16, None, id="b16"),  # two units per pair core
        pytest.param(((1, 27, 2560), (27, 12, 128, 128)), ttnn.bfloat16, None, id="b27"),  # ragged last core
    ],
)
@pytest.mark.parametrize("in_place", [False, True], ids=["new", "in_place"])
def test_small_mcast_bit_identical(device, monkeypatch, inputs, token_dtype, poison, in_place):
    torch.manual_seed(7)
    tensors = make_inputs(inputs)
    if token_dtype == ttnn.bfloat16:
        for name in ORDER:
            if name != "recurrent_state":
                tensors[name] = tensors[name].to(torch.bfloat16).to(torch.float32)
    got = _run(device, tensors, in_place, token_dtype, poison)
    monkeypatch.setattr(gdn_pd, "SMALL_INPUTS_MCAST", False)
    expected = _run(device, tensors, in_place, token_dtype, poison)
    for a, b, label in zip(got, expected, ("output", "conv_state_out", "recurrent_state_out")):
        assert torch.equal(a, b), f"{label}: SMALL_INPUTS_MCAST differs from the DRAM reads"
