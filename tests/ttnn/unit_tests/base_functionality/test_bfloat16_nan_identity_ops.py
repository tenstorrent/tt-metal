# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""A bfloat16 NaN must survive ops that neither convert a format nor compute, because the identity pack keeps it.
Narrowing conversions such as typecast(float32, bfloat16) are out of scope, since rounding maps NaN to infinity.
Denormals and -0 are not asserted: MOVA2D in an A2D datacopy flushes them upstream of the packer."""

import pytest
import torch

import ttnn
from models.common.utility_functions import run_for_blackhole

# Canonical quiet NaN, smallest payload, and a negative NaN, so sign and payload bugs both show.
NAN_BITS = [0x7FC0, 0x7F81, 0xFFC0]


def bf16_tile(bits, device, shape=(1, 1, 32, 32)):
    """Return a device tensor whose first element holds `bits` and the rest +0."""
    raw = torch.zeros(shape, dtype=torch.int16)
    raw[0, 0, 0, 0] = bits - (1 << 16) if bits >= 0x8000 else bits
    return ttnn.from_torch(raw.view(torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)


def first_bits(tensor):
    return int(ttnn.to_torch(tensor).view(torch.int16).flatten()[0].item()) & 0xFFFF


def classify(bits):
    exponent, mantissa = bits & 0x7F80, bits & 0x007F
    if exponent == 0x7F80:
        return "NaN" if mantissa else "Inf"
    if exponent == 0:
        return "zero" if mantissa == 0 else "denormal"
    return "normal"


@run_for_blackhole()
@pytest.mark.parametrize("nan_bits", NAN_BITS, ids=lambda b: f"0x{b:04X}")
@pytest.mark.parametrize(
    "op_name",
    ["to_layout_row_major", "to_layout_tile", "neg", "abs", "typecast_bfloat16", "clone"],
)
def test_identity_op_preserves_nan(device, op_name, nan_bits):
    """Each op gets its own tensor, because several ttnn ops deallocate their input."""
    ops = {
        "to_layout_row_major": lambda t: ttnn.to_layout(t, ttnn.ROW_MAJOR_LAYOUT),
        "to_layout_tile": lambda t: ttnn.to_layout(ttnn.to_layout(t, ttnn.ROW_MAJOR_LAYOUT), ttnn.TILE_LAYOUT),
        "neg": ttnn.neg,
        "abs": ttnn.abs,
        "typecast_bfloat16": lambda t: ttnn.typecast(t, ttnn.bfloat16),
        "clone": lambda t: ttnn.clone(t, dtype=ttnn.bfloat16),
    }
    got = first_bits(ops[op_name](bf16_tile(nan_bits, device)))

    if op_name == "abs":
        # Assert the payload only; abs leaves the NaN sign bit alone on this hardware.
        assert classify(got) == "NaN" and (got & 0x7FFF) == (
            nan_bits & 0x7FFF
        ), f"abs(0x{nan_bits:04X}) returned 0x{got:04X}"
        return

    expected = nan_bits ^ 0x8000 if op_name == "neg" else nan_bits
    assert got == expected, f"{op_name}(0x{nan_bits:04X}) returned 0x{got:04X}, expected 0x{expected:04X}"


@run_for_blackhole()
def test_full_like_nan_fill_stays_nan(device):
    """Issue #40503: the NaN fill value must not be lost on the way out."""
    reference = ttnn.from_torch(
        torch.zeros(1, 1, 32, 32, dtype=torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
    )
    got = first_bits(ttnn.full_like(reference, float("nan")))
    assert classify(got) == "NaN", f"full_like(nan) returned 0x{got:04X}"


@run_for_blackhole()
@pytest.mark.parametrize("input_layout", [ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT])
@pytest.mark.parametrize("output_layout", [ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT])
def test_to_layout_nan_not_mangled(device, input_layout, output_layout):
    """Issue #31406, with a shape that is not a multiple of the tile size."""
    torch_input = torch.full((17, 37), 0.05, dtype=torch.bfloat16)
    torch_input[0, 0] = float("nan")
    tensor = ttnn.from_torch(torch_input, device=device, layout=input_layout)
    result = ttnn.to_torch(ttnn.to_layout(tensor, layout=output_layout))
    assert torch.isnan(result[0, 0]), f"NaN became {float(result[0, 0])}"


@run_for_blackhole()
def test_every_bfloat16_encoding_through_to_layout(device):
    """All 65536 encodings through to_layout: no NaN and no normal value may change."""
    raw = torch.arange(65536, dtype=torch.int32) & 0xFFFF
    raw = torch.where(raw >= 0x8000, raw - (1 << 16), raw).to(torch.int16).reshape(1, 1, 256, 256)
    tensor = ttnn.from_torch(raw.view(torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    out = ttnn.to_torch(ttnn.to_layout(tensor, ttnn.ROW_MAJOR_LAYOUT)).view(torch.int16).flatten()
    source = raw.flatten()

    changed = {"NaN": [], "normal": []}
    for src, got in zip(source.tolist(), out.tolist()):
        src, got = src & 0xFFFF, got & 0xFFFF
        kind = classify(src)
        if kind in changed and src != got:
            changed[kind].append((src, got))

    assert not changed["normal"], f"{len(changed['normal'])} normal encodings changed, e.g. {changed['normal'][:4]}"
    assert not changed["NaN"], f"{len(changed['NaN'])} of 254 NaN encodings changed, e.g. " + ", ".join(
        f"0x{s:04X}->0x{g:04X}" for s, g in changed["NaN"][:4]
    )


@run_for_blackhole()
def test_narrowing_pack_is_unaffected(device):
    """The identity pack's read raw bit must not leak into a bfloat8_b matmul pack via reconfig.

    Default bfp8 pack rounds twice and differs from the precise typecast path (P300: 7851 of 65536 differ with and without the fix, 0 if the bit leaks).
    """
    torch.manual_seed(1234)
    a = torch.randn(1, 1, 256, 256, dtype=torch.bfloat16)
    b = torch.randn(1, 1, 256, 256, dtype=torch.bfloat16)

    def to_device(x):
        return ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    direct = ttnn.to_torch(ttnn.matmul(to_device(a), to_device(b), dtype=ttnn.bfloat8_b))
    precise = ttnn.to_torch(ttnn.typecast(ttnn.matmul(to_device(a), to_device(b), dtype=ttnn.bfloat16), ttnn.bfloat8_b))

    assert not torch.equal(direct, precise), (
        "matmul with a bfloat8_b output is bit-identical to the bfp8_pack_precise path, which means "
        "the packer read Dst raw for a conversion that needs rounding"
    )
