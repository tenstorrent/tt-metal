# SPDX-FileCopyrightText: (c) 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Refinement 19: R-QKCORE pair cores whose writer reads S0 (S0_WRITER, off by default) instead of the reader. Same
tiles into the same cb_state_in slots, so every output must match the default build bit for bit, in place too (the
writer reads a unit's S0 before it writes that unit's final state to the same pages)."""

import pytest
import torch

from ttnn.operations.gated_delta_net import gated_delta_net_program_descriptor as gdn_pd
from tests.ttnn.unit_tests.operations.gated_delta_net.test_gated_delta_net import ORDER, make_inputs
from tests.ttnn.unit_tests.operations.gated_delta_net.test_gated_delta_net_state_split import _run

LEVERS = {
    "at_start": {"S0_WRITER": True, "S0_WRITER_AFTER_INPUTS": False},
    "after_inputs": {"S0_WRITER": True},
    "keep_noc": {"S0_WRITER": True, "S0_WRITER_KEEP_NOC": True},
    "rows_3_4": {"S0_WRITER": True, "S0_WRITER_ROWS": (3, 4), "S0_WRITER_KEEP_NOC": True},
}


@pytest.mark.parametrize("lever", list(LEVERS))
@pytest.mark.parametrize(
    "inputs",
    [
        pytest.param(((1, 1, 2560), (1, 12, 128, 128)), id="b1"),  # QK_CORES_VSPLIT state cores
        pytest.param(((1, 8, 2560), (8, 12, 128, 128)), id="b8"),  # one unit per pair core
        pytest.param(((1, 16, 2560), (16, 12, 128, 128)), id="b16"),  # two units per pair core
        pytest.param(((1, 27, 2560), (27, 12, 128, 128)), id="b27"),  # 3-unit cores keep S0 on the reader
    ],
)
@pytest.mark.parametrize("in_place", [False, True], ids=["new", "in_place"])
def test_s0_writer_bit_identical(device, monkeypatch, inputs, lever, in_place):
    torch.manual_seed(5)
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
