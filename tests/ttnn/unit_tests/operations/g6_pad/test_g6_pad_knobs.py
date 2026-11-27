# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Knob sweeps for profiling g6_pad (run under scripts/run_safe_pytest.sh --profile).

Each case turns one host knob, clears the plan/descriptor caches and checks bit equality, so a
profiled row is both a timing and a correctness witness for that knob value.
"""

import pytest
import torch

import ttnn
from ttnn.operations.g6_pad import g6_pad
from ttnn.operations.g6_pad import g6_pad_program_descriptor as desc

MASK_FILL = -1.7014118346046923e38

SHAPES = [
    pytest.param((1, 1, 2, 62080), [[0, 0], [0, 0], [0, 30], [0, 0]], 0.0, id="p2"),
    pytest.param((1, 1, 8, 62080), [[0, 0], [0, 0], [0, 8], [0, 0]], 0, id="p3"),
    pytest.param((1, 1, 8, 62080), [[0, 0], [0, 0], [0, 24], [0, 0]], 0.0, id="p4"),
    pytest.param((16, 1, 1536), [[0, 0], [0, 31], [0, 0]], 0.0, id="16x1x1536"),
    pytest.param((1, 1, 196, 196), [[0, 60], [0, 60]], MASK_FILL, id="hw_196"),
]


def _torch_pad(padding):
    flat = []
    for low, high in reversed(padding):
        flat += [low, high]
    return flat


@pytest.mark.parametrize(
    "noc0_share", [(0, 1), (1, 3), (2, 5), (1, 2)], ids=["noc0_0", "noc0_1of3", "noc0_2of5", "noc0_1of2"]
)
@pytest.mark.parametrize("shape, padding, value", SHAPES)
def test_band_noc0_share(device, monkeypatch, shape, padding, value, noc0_share):
    monkeypatch.setattr(desc, "BAND_NOC0_SHARE", noc0_share)
    monkeypatch.setattr(desc, "_PLAN_CACHE", {})
    monkeypatch.setattr(desc, "_DESCRIPTOR_CACHE", {})
    torch.manual_seed(0)
    torch_input = (torch.randn(shape) * 100).to(torch.bfloat16)
    x = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    x = ttnn.fill_implicit_tile_padding(x, 1000.0)
    expected = torch.nn.functional.pad(torch_input, _torch_pad(padding), mode="constant", value=value)
    out = ttnn.to_torch(g6_pad(x, padding, value=value))
    assert torch.equal(out.view(torch.int16), expected.view(torch.int16))
