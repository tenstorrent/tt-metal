# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""A bfloat16 NaN does not survive device compute: it comes back as an infinity.

Two independent losses, both at a datum-format boundary and neither in ttnn:

  read side   a bfloat16 NaN widened to float32 arrives as an infinity, so it never reaches
              arithmetic as a NaN at all. Widening is exact by construction, 0x7FC0 should
              become 0x7FC00000, but the mantissa is dropped:

                  typecast(bf16 0x7FC0 -> fp32)  ->  0x7F800000   (+inf)

  write side  the SFPU canonicalises a NaN result to 0x7F800001, with the payload in the
              lowest mantissa bit. float32 keeps it; bfloat16 has 7 mantissa bits and
              truncates it away, leaving the bare infinity pattern:

                  fp32 multiply of NaN  ->  0x7F800001   (NaN)
                  narrowed to bf16      ->  0x7F80       (inf)

              For contrast, torch narrowing the same 0x7F800001 to bfloat16 gives 0xFFFF,
              a NaN. Canonicalising to 0x7FC00000 instead would make the narrowing safe.

The tests below assert the behaviour we want, not the behaviour we have, and are marked
xfail(strict=True) so that fixing either layer turns them into failures rather than silent
passes. When that happens, also delete the bfloat16 skip in
test_backward_relu6.py::test_bw_relu6_boundaries, which exists only because of this.
"""

import pytest
import torch

import ttnn


def _bf16_from_bits(shape, bits):
    signed = bits - 0x10000 if bits > 0x7FFF else bits
    return torch.tensor([signed], dtype=torch.int16).view(torch.bfloat16).repeat(shape.numel()).reshape(shape)


@pytest.mark.xfail(strict=True, reason="a bfloat16 NaN widens to an infinity; the mantissa is dropped on read")
@pytest.mark.parametrize("nan_bits", (0x7FC0, 0xFFC0), ids=["positive_nan", "negative_nan"])
def test_bfloat16_nan_survives_widening_to_float32(device, nan_bits):
    """Widening bfloat16 to float32 appends zero bits, so a NaN must stay a NaN."""
    shape = torch.Size([1, 1, 32, 32])
    host = _bf16_from_bits(shape, nan_bits)
    assert torch.isnan(host).all(), "the host tensor must start as NaN"

    on_device = ttnn.from_torch(host, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    assert torch.isnan(ttnn.to_torch(on_device)).all(), "a round trip with no compute must preserve it"

    widened = ttnn.to_torch(ttnn.typecast(on_device, ttnn.float32))
    assert torch.isnan(widened).all(), f"widened to 0x{widened.view(torch.int32)[0, 0, 0, 0].item() & 0xFFFFFFFF:08x}"


@pytest.mark.xfail(strict=True, reason="the SFPU puts the NaN payload in bit 0, which bfloat16 truncates away")
def test_bfloat16_nan_survives_narrowing_from_float32(device):
    """A float32 NaN result narrowed to bfloat16 must stay a NaN."""
    shape = torch.Size([1, 1, 32, 32])
    host = torch.full(shape, float("nan"), dtype=torch.float32)

    on_device = ttnn.from_torch(host, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    computed = ttnn.multiply(on_device, 1.0)
    assert torch.isnan(ttnn.to_torch(computed)).all(), "the float32 result itself must be NaN"

    narrowed = ttnn.to_torch(ttnn.typecast(computed, ttnn.bfloat16))
    assert torch.isnan(narrowed).all(), f"narrowed to 0x{narrowed.view(torch.int16)[0, 0, 0, 0].item() & 0xFFFF:04x}"


@pytest.mark.xfail(strict=True, reason="a bfloat16 NaN operand becomes an infinity in every eltwise op")
@pytest.mark.parametrize(
    "op",
    (
        pytest.param(lambda x: ttnn.multiply(x, 1.0), id="multiply"),
        pytest.param(lambda x: ttnn.add(x, 0.0), id="add"),
        pytest.param(ttnn.neg, id="neg"),
        pytest.param(ttnn.abs, id="abs"),
    ),
)
def test_bfloat16_nan_propagates_through_eltwise(device, op):
    """Every eltwise op must propagate a NaN operand rather than returning an infinity."""
    shape = torch.Size([1, 1, 32, 32])
    host = _bf16_from_bits(shape, 0x7FC0)

    on_device = ttnn.from_torch(host, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    result = ttnn.to_torch(op(on_device))

    assert torch.isnan(result).all(), f"returned 0x{result.view(torch.int16)[0, 0, 0, 0].item() & 0xFFFF:04x}"


@pytest.mark.xfail(strict=True, reason="a NaN the device produces itself is lost the same way in bfloat16")
def test_bfloat16_nan_produced_on_device(device):
    """0 / 0 must be NaN in bfloat16, as it is in float32."""
    shape = torch.Size([1, 1, 32, 32])
    zeros = ttnn.from_torch(
        torch.zeros(shape, dtype=torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
    )

    result = ttnn.to_torch(ttnn.divide(zeros, zeros))

    assert torch.isnan(result).all(), f"returned 0x{result.view(torch.int16)[0, 0, 0, 0].item() & 0xFFFF:04x}"


def test_float32_nan_propagates_through_eltwise(device):
    """The float32 control: the same ops keep a NaN, so this is a bfloat16 problem only."""
    shape = torch.Size([1, 1, 32, 32])
    host = torch.full(shape, float("nan"), dtype=torch.float32)
    on_device = ttnn.from_torch(host, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)

    for op in (lambda x: ttnn.multiply(x, 1.0), lambda x: ttnn.add(x, 0.0), ttnn.neg, ttnn.abs):
        assert torch.isnan(ttnn.to_torch(op(on_device))).all(), "float32 must propagate NaN"
