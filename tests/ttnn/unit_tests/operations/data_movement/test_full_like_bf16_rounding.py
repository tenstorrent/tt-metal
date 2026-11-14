# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Test for issue #63672: ttnn.full_like and ttnn.fill truncate bfloat16 fill values
while ttnn.full rounds them.

The bug: full_like_impl passed the raw float fill_value to ttnn::fill without
rounding to bfloat16 first. The SFPU FILL kernel stores the fp32 bit pattern
directly into dst_reg, which truncates (not rounds) when the Dest is 16-bit.

The fix: round to bfloat16 on the host in full_like_impl before calling ttnn::fill,
matching the pattern already used in full_impl (PR #58440).
"""

import pytest
import torch
import struct

import ttnn

torch.manual_seed(0)


def _bf16_hex(f):
    """Return the bfloat16 hex representation of a float (for debugging)."""
    bytes_f32 = struct.pack(">f", f)
    bf16_bits = int.from_bytes(bytes_f32[:2], "big")
    return f"0x{bf16_bits:04x}"


@pytest.mark.parametrize(
    "fill_value",
    [
        0.1,
        0.3,
        1.0 / 3.0,
        -0.1,
        1.2,
        2.00781250,
        2.00830080,
        2.02343750,
        -3.9921875,
        3.14159,
    ],
)
def test_full_like_bf16_rounding(device, fill_value):
    """ttnn.full_like must round bfloat16 fill values (ties to even), not truncate them.

    Exercises the fixed full_like_impl code path: device TILE bfloat16 tensor,
    where the fill value goes through the on-device SFPU FILL kernel.
    """
    torch_input = torch.zeros([1, 1, 32, 32], dtype=torch.bfloat16)
    torch_ref = torch.full_like(torch_input, fill_value)

    tt_input = ttnn.from_torch(torch_input, layout=ttnn.TILE_LAYOUT, device=device)
    tt_output = ttnn.full_like(tt_input, fill_value=fill_value)
    tt_output_host = ttnn.to_torch(tt_output)

    assert tt_output_host.shape == torch_ref.shape
    assert torch.equal(tt_output_host, torch_ref), (
        f"full_like bfloat16 rounding mismatch for fill_value={fill_value}: "
        f"ttnn={_bf16_hex(tt_output_host.flatten()[0].item())}, "
        f"torch={_bf16_hex(torch_ref.flatten()[0].item())}"
    )


@pytest.mark.parametrize(
    "fill_value",
    [0.1, 0.3, 1.2, -0.1],
)
def test_full_like_bf16_rounding_larger_tensor(device, fill_value):
    """Same rounding check on a multi-tile tensor to ensure the fix works at scale."""
    torch_input = torch.zeros([2, 3, 64, 128], dtype=torch.bfloat16)
    torch_ref = torch.full_like(torch_input, fill_value)

    tt_input = ttnn.from_torch(torch_input, layout=ttnn.TILE_LAYOUT, device=device)
    tt_output = ttnn.full_like(tt_input, fill_value=fill_value)
    tt_output_host = ttnn.to_torch(tt_output)

    assert torch.equal(tt_output_host, torch_ref), (
        f"full_like bfloat16 rounding mismatch (large tensor) for fill_value={fill_value}"
    )


def test_full_vs_full_like_consistency(device):
    """ttnn.full and ttnn.full_like must produce the same bfloat16 values for the same fill_value.

    Before the fix, full rounded but full_like truncated, so they disagreed by 1 ULP
    for about half of all non-representable bfloat16 values.
    """
    fill_value = 0.1
    shape = [1, 1, 32, 32]

    tt_full = ttnn.full(shape=shape, fill_value=fill_value, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    torch_input = torch.zeros(shape, dtype=torch.bfloat16)
    tt_input = ttnn.from_torch(torch_input, layout=ttnn.TILE_LAYOUT, device=device)
    tt_full_like = ttnn.full_like(tt_input, fill_value=fill_value)

    full_result = ttnn.to_torch(tt_full)
    full_like_result = ttnn.to_torch(tt_full_like)

    assert torch.equal(full_result, full_like_result), (
        f"ttnn.full and ttnn.full_like disagree on bfloat16 fill value {fill_value}: "
        f"full={_bf16_hex(full_result.flatten()[0].item())}, "
        f"full_like={_bf16_hex(full_like_result.flatten()[0].item())}"
    )
