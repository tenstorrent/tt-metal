# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""A bfloat16 NaN must survive an op that neither converts a format nor computes anything.

Before the packer selects the identity path for same-format packs, every op in this file returns an
Infinity of the same sign instead, which is issue #31406 for `to_layout` and #40503 for `full_like`.
Genuine narrowing conversions such as `typecast(float32, bfloat16)` are out of scope: their early
format conversion has to round, and rounding maps NaN to infinity for an 8-bit exponent.

Denormals and minus zero are deliberately not asserted here. Both are still flushed, and
`tech_reports/Handling_Special_Value/special_values.md` documents denormals as flushed globally.
"""

import pytest
import torch

import ttnn
from models.common.utility_functions import run_for_blackhole

# Three of the 254 bfloat16 NaN encodings: the canonical quiet NaN, the smallest payload, and a
# negative one, so that a sign-preserving bug and a payload-dropping bug are both visible.
NAN_BITS = [0x7FC0, 0x7F81, 0xFFC0]


def bf16_tile(bits, device, shape=(1, 1, 32, 32)):
    """A fresh device tensor whose first element holds `bits` and whose remainder is +0."""
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
    """Each op is given a tensor of its own, because several ttnn ops deallocate their input."""
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
        # The payload has to survive, and it is the payload this is about. The sign bit of a NaN is
        # left alone by abs on this hardware, which is a separate question from the pack path.
        assert classify(got) == "NaN" and (got & 0x7FFF) == (
            nan_bits & 0x7FFF
        ), f"abs(0x{nan_bits:04X}) returned 0x{got:04X}"
        return

    expected = nan_bits ^ 0x8000 if op_name == "neg" else nan_bits
    assert got == expected, f"{op_name}(0x{nan_bits:04X}) returned 0x{got:04X}, expected 0x{expected:04X}"


@run_for_blackhole()
def test_full_like_nan_fill_stays_nan(device):
    """Issue #40503: the fill value reaches the SFPU intact and is lost on the way out."""
    reference = ttnn.from_torch(
        torch.zeros(1, 1, 32, 32, dtype=torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
    )
    got = first_bits(ttnn.full_like(reference, float("nan")))
    assert classify(got) == "NaN", f"full_like(nan) returned 0x{got:04X}"


@run_for_blackhole()
@pytest.mark.parametrize("input_layout", [ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT])
@pytest.mark.parametrize("output_layout", [ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT])
def test_to_layout_nan_not_mangled(device, input_layout, output_layout):
    """Issue #31406, with its own shape, which is deliberately not a multiple of the tile size."""
    torch_input = torch.full((17, 37), 0.05, dtype=torch.bfloat16)
    torch_input[0, 0] = float("nan")
    tensor = ttnn.from_torch(torch_input, device=device, layout=input_layout)
    result = ttnn.to_torch(ttnn.to_layout(tensor, layout=output_layout))
    assert torch.isnan(result[0, 0]), f"NaN became {float(result[0, 0])}"


@run_for_blackhole()
def test_every_bfloat16_encoding_through_to_layout(device):
    """All 65536 encodings in one tensor: no NaN may change, and no normal value may change."""
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
    """Selecting the identity pack path must not leak into a pack that does convert.

    The packer's read-raw bit lives in a register that `set_packer_config` writes whole and
    `reconfig_packer_data_format` writes through a mask, so a kernel that initialises the packer for
    an identity pack and later reconfigures it to a block-float output can carry the bit into that
    output unless the mask covers it. `matmul` with a `bfloat8_b` output is such a kernel.

    The observable: the default pack for a `bfloat8_b` output rounds twice, once into the packer's
    block-float intermediate and once into the output, whereas `ttnn.typecast` sets
    `bfp8_pack_precise` and rounds once. So the two disagree by about a tenth of the elements. If the
    read-raw bit leaks in, the default path stops rounding early and becomes the precise one, and the
    two become bit-identical. Measured on Blackhole P300: 7851 of 65536 elements differ both on the
    base commit and with the complete fix, and 0 differ when the mask is left out of the fix.

    This test therefore locks the scope of the NaN fix rather than any particular rounding mode. If
    `bfp8_pack_precise` ever becomes the default, delete it rather than papering over it.
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
