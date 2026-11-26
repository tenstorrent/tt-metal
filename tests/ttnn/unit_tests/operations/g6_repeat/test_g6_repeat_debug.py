# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Deterministic debug tests for g6_repeat.

ttnn.fill_implicit_tile_padding(x, nan) writes 0x7F80 (+inf) into the padding on
this build, so the acceptance and golden NaN-poison cases fail their harness
precondition before the op runs. These tests poison the padding of x another way:
a (1, 1, 32, W) tile tensor whose rows 1..31 hold the poison, viewed as
(1, 1, 1, W) over the same buffer, so x's tile padding is exactly that poison.
"""

import pytest
import torch
import ttnn

from ttnn.operations.g6_repeat import g6_repeat, plan_report


def _poisoned_row_tensor(device, w, poison_bits):
    torch.manual_seed(42)
    full = torch.randn((1, 1, 32, w), dtype=torch.float32).to(torch.bfloat16)
    bits = full.view(torch.int16).clone()
    signed = poison_bits - 0x10000 if poison_bits >= 0x8000 else poison_bits
    bits[..., 1:, :] = signed
    full = bits.view(torch.bfloat16)
    t32 = ttnn.from_torch(
        full, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    x = ttnn.reshape(t32, ttnn.Shape([1, 1, 1, w]), ttnn.Shape([1, 1, 32, w]))
    assert x.buffer_address() == t32.buffer_address(), "precondition: the view must share x's buffer"
    padding = x.cpu().to_torch_with_padded_shape()[..., 1:, :].contiguous().view(torch.int16)
    assert bool((padding == signed).all()), "precondition: x's tile padding does not hold the poison"
    return full[..., :1, :], x


@pytest.mark.parametrize("poison_bits", [0x7FC1, 0xFFC5, 0x447A], ids=["qnan_payload", "neg_qnan_payload", "p1000"])
@pytest.mark.parametrize("w, r", [(256, 16), (64, 17), (96, 31)], ids=["w256_r16", "w64_r17", "w96_r31"])
def test_poisoned_padding_via_view(device, poison_bits, w, r):
    row, x = _poisoned_row_tensor(device, w, poison_bits)
    out = g6_repeat(x, [1, 1, r, 1])
    got = ttnn.to_torch(out).view(torch.int16)
    want = row.repeat(1, 1, r, 1).view(torch.int16)
    assert torch.equal(got, want), f"{int((got != want).sum())} elements differ"


def test_tile_position_encoding(device):
    """Each element of x's row encodes its column: a misplaced face row shows its source."""
    w, r = 128, 20
    row = torch.arange(w, dtype=torch.float32).reshape(1, 1, 1, w).to(torch.bfloat16)
    x = ttnn.from_torch(
        row, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    got = ttnn.to_torch(g6_repeat(x, [1, 1, r, 1]))
    assert torch.equal(got.view(torch.int16), row.repeat(1, 1, r, 1).view(torch.int16))


def test_plan_report_p1(device):
    x = ttnn.from_torch(
        torch.zeros((1, 1, 1, 256), dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    plan = plan_report(x, [1, 1, 16, 1])
    assert plan["regime"] == "column_split_row_broadcast"
    assert plan["num_cores"] == 8
    assert plan["block_width_tiles"] == 1
    assert plan["num_blocks"] == 1
    assert [c["start_tile_col"] for c in plan["cores"]] == list(range(8))
