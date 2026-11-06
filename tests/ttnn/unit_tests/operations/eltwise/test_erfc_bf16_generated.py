# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Generated BF16 erfc cases, beside the unary category suites."""

import torch
import pytest
import ttnn
from tests.ttnn.unit_tests.operations.eltwise.eltwise_test_utils import (
    generate_bfloat16_bits,
    to_tt_tensor,
    SMALLEST_NORMAL_BF16,
)


@pytest.mark.parametrize("generated_first", [True, False])
def test_erfc_with_sqrt_in_sfpu_chain(device, generated_first):
    """erfc chained with stock sqrt, both orders: each step stores what its op stores standalone."""
    generated = (ttnn.UnaryWithParam(ttnn.UnaryOpType.ERFC), lambda tensor: ttnn.erfc(tensor))
    stock = (ttnn.UnaryWithParam(ttnn.UnaryOpType.SQRT), lambda tensor: ttnn.sqrt(tensor))
    first, second = (generated, stock) if generated_first else (stock, generated)
    input_tensor = generate_bfloat16_bits(dtype=torch.bfloat16)
    tt_in = to_tt_tensor(input_tensor, device)

    def words(tensor):
        return ttnn.to_torch(tensor).view(torch.int16)

    for step, standalone in (generated, stock):
        assert torch.equal(
            words(ttnn.unary_chain(tt_in, [step])), words(standalone(tt_in))
        ), f"{step} is not its op's kernel"

    result = words(ttnn.unary_chain(tt_in, [first[0], second[0]]))
    middle = ttnn.to_torch(first[1](tt_in))
    expected = words(second[1](to_tt_tensor(middle, device)))
    # A finite normal first result sits in DEST exactly as stored, so both runs feed the second op the same input.
    checked = torch.isfinite(middle) & (middle.abs() >= SMALLEST_NORMAL_BF16)
    assert checked.sum() >= 64
    mismatched = checked & (result != expected)
    assert not mismatched.any(), (
        f"{int(mismatched.sum())} of {int(checked.sum())} lanes differ from the standalone second op, "
        f"first at its input {middle[mismatched][0].item()!r}"
    )


def test_erfc_bf16_unflushed_zero_and_subnormal_inputs(device):
    """-0 and subnormal inputs (kept by TT-NN's copy into DEST) store torch's result within 1 ULP."""
    words = torch.cat([torch.arange(0x80), torch.arange(0x8000, 0x8080)]).to(torch.int16)
    x = words.view(torch.bfloat16).repeat(4).reshape(1, 1, 32, 32)
    tt_x = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tiny = torch.finfo(torch.bfloat16).tiny

    def stored(t):
        t = t.to(torch.bfloat16).to(torch.float64).flatten()
        t = torch.where(torch.isnan(t), torch.full_like(t, float("inf")), t)
        return torch.where(t.abs() < tiny, torch.zeros_like(t), t)

    actual = stored(ttnn.to_torch(ttnn.erfc(tt_x)))
    x64 = x.flatten().to(torch.float64)
    reference = ttnn.get_golden_function(ttnn.erfc)(x64).to(torch.float64).flatten()
    rounded = stored(reference)
    spacing = 2.0 ** (torch.floor(torch.log2(reference.abs().clamp(min=tiny))) - 7)
    close = torch.isfinite(reference) & (actual != 0) & ((actual - reference).abs() < spacing)
    ok = (actual == rounded) | close
    bits = words.repeat(4).to(torch.int32) & 0xFFFF
    assert ok.all(), (
        f"{(~ok).sum().item()} of {ok.numel()} inputs off torch and stock; first input 0x{bits[~ok][0].item():04x} "
        f"stored {actual[~ok][0].item()!r}, torch {reference[~ok][0].item()!r}"
    )


def test_erfc_bf16_bfloat8_b(device):
    """bfloat8_b in and out: PCC 0.999 against torch, and each output within one bfloat8_b step of the BF16 call's
    packed to bfloat8_b (two in a block holding an infinity or NaN)."""
    torch.manual_seed(0)
    x = torch.empty(1, 1, 64, 64).uniform_(-5.0, 5.0).to(torch.bfloat16)
    tt_x = ttnn.from_torch(x, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device)
    stored = ttnn.to_torch(tt_x).to(torch.float64).flatten()
    output = ttnn.erfc(tt_x)
    assert output.dtype == ttnn.bfloat8_b
    actual = ttnn.to_torch(output).to(torch.float64).flatten()
    golden = ttnn.get_golden_function(ttnn.erfc)(stored).to(torch.float64)
    pcc = torch.corrcoef(torch.stack([actual, golden]))[0, 1].item()
    assert pcc >= 0.999, f"PCC {pcc:.6f} against torch"
    # Every finite BF16 exponent and +-0, one exponent per 16-value face row.
    words = torch.arange(1 << 16, dtype=torch.int32)
    exponent = (words >> 7) & 0xFF
    zeros = torch.tensor([0x0000] * 256 + [0x8000] * 256, dtype=torch.int32)
    words = torch.cat([words[(exponent > 0) & (exponent < 0xFF)], zeros])
    x = words.to(torch.int16).view(torch.bfloat16).reshape(1, 1, 256, 256)
    tt_x = ttnn.from_torch(x, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device)
    stored = ttnn.to_torch(tt_x).to(torch.bfloat16)
    tt_stored = ttnn.from_torch(stored, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    actual = ttnn.to_torch(ttnn.erfc(tt_x)).to(torch.float32).flatten()
    expected = ttnn.to_torch(ttnn.typecast(ttnn.erfc(tt_stored), ttnn.bfloat8_b)).to(torch.float32).flatten()
    # One step of a Bfp8_b block: 7 bits under the largest of its 16 values' exponent.
    finite = torch.isfinite(actual) & torch.isfinite(expected)
    magnitude = torch.where(finite, torch.maximum(actual.abs(), expected.abs()), torch.zeros_like(actual))
    block = magnitude.reshape(-1, 16).amax(dim=1, keepdim=True).expand(-1, 16).reshape(-1)
    step = torch.exp2(torch.floor(torch.log2(block.clamp_min(torch.finfo(torch.float32).tiny))) - 6)
    # The pack and the typecast set a block holding an infinity or NaN apart: two steps there.
    special = (~finite).reshape(-1, 16).any(dim=1, keepdim=True).expand(-1, 16).reshape(-1)
    step = torch.where(special, 2 * step, step)
    # Infinity and NaN keep their class; a finite value, zero included, may round one step apart.
    same = (
        (actual == expected)
        | (torch.isnan(actual) & torch.isnan(expected))
        | (finite & ((actual - expected).abs() <= step))
    )
    assert same.all(), (
        f"{int((~same).sum())} bfloat8_b outputs are more than one bfloat8_b step from the BF16 call's on the same "
        "stored input; first at "
        f"{stored.flatten()[~same][0].item()!r}: {actual[~same][0].item()!r}, BF16 call {expected[~same][0].item()!r}"
    )


def test_erfc_bf16_nonfinite_inputs(device):
    """+-Inf and +-NaN store torch's result (NaN as an infinity) or STOCK's {input bits: word} for the board."""
    board = "blackhole" if ttnn.device.is_blackhole(device) else "wormhole_b0"
    STOCK = {"blackhole": {0x7FC0: 0x2C4F}, "wormhole_b0": {0x7FC0: 0x2C4E}}
    bits = torch.tensor([0x7F80, 0xFF80, 0x7FC0, 0xFFC0], dtype=torch.int32).repeat(256)
    x = bits.to(torch.int16).view(torch.bfloat16).reshape(1, 1, 32, 32)
    tt_x = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    def stored(t):
        return t.to(torch.bfloat16).to(torch.float64).flatten()

    actual = stored(ttnn.to_torch(ttnn.erfc(tt_x)))
    reference = ttnn.get_golden_function(ttnn.erfc)(x.flatten().to(torch.float64)).to(torch.float64)
    expected = stored(reference)
    for input_bits, word in STOCK.get(board, {}).items():
        expected[bits == input_bits] = stored(
            torch.tensor([word], dtype=torch.int32).to(torch.int16).view(torch.bfloat16)
        )
    spacing = 2.0 ** (torch.floor(torch.log2(expected.abs().clamp(min=torch.finfo(torch.bfloat16).tiny))) - 7)
    ok = (actual == expected) | (torch.isfinite(expected) & ((actual - expected).abs() < spacing))
    # A NaN result reaches L1 as a NaN or as an infinity of either sign.
    ok |= torch.isnan(expected) & (torch.isnan(actual) | torch.isinf(actual))
    assert (
        ok.all()
    ), f"input 0x{bits[~ok][0].item():04x} stored {actual[~ok][0].item()!r}, expected {expected[~ok][0].item()!r}"


def test_erfc_bf16_exhaustive_inputs(device):
    """All 65,536 BF16 inputs: torch's result as stored, within STEPS BF16 steps, or STOCK's word per board.

    SKIP lists inputs the LLK sweep judges instead. MODEL: per board, the kernel's offset in BF16 steps from torch's word.
    """
    board = "blackhole" if ttnn.device.is_blackhole(device) else "wormhole_b0"
    STOCK = {"blackhole": {0x2C4F: [(0x7F81, 0x7FFF)]}, "wormhole_b0": {0x2C4E: [(0x7F81, 0x7FFF)]}}
    STEPS = 1
    SKIP = [(0x4114, 0x4116)]
    MODEL = {
        "blackhole": {
            -1: (
                "3A0D-3AE2 3B85-3BAA 3BFA-3C0D 3C38-3C46 3C72-3C7F 3C96-3C9B 3CB3-3CB8 3CD0-3CD4 3CED-3CF1 3D05-3D06 "
                "3D14 3D22-3D23 3D31 3D3F 3F16 3F1D 3F2B 3F41 3F45 3F53 "
                "3F57 3F5C 3F72 3F77 3F7B 3F82 3F8C 3F8F-3F93 3F96 3F98 "
                "3F9C-3F9E 3FB2 3FB6 3FBE 3FC3-3FC4 3FC7 3FCD 3FD0-3FD2 3FD7 3FDB "
                "3FE9 3FF5 4101 4107 410E 4112 BD87 BDA4 BDB2 BDC0 "
                "BDCF BDDD BDEB-BDEC BDFA BE04 BE13 BE1A BE21 BE29 BE30 "
                "BE37 BE3F BE46 BE4D BE55 BE5C BE63 BE6B BE72 BE7A "
                "BE8C BE90 BEAB BEAF BEB3 BEB7 BED8 BEF2"
            ),
            1: (
                "3D92 3D99 3DA0 3DA7 3DB6 3DBD 3DC4 3DCB 3DD2-3DD3 3DDA "
                "3DE1 3DE8 3DEF 3DF6-3DF7 3DFD-3DFE 3E06 3E0A 3E0D 3E11 3E15 "
                "3E18 3E1C 3E23 3E27 3E2A 3E2E 3E32 3E35 3E39 3E3D "
                "3E40 3E44 3E48 3E4B 3E4F 3E53 3E56 3E5A 3E5E 3E62 "
                "3E65 3E69 3E6D 3E71 3E74 3E78 3E7C 3E80 3E8B 3E8D "
                "3E8F 3E91 3E93 3EAA 3EAC 3EAE 3EB0 3EB2 3EB4 3EB6 "
                "3EB8 3EBA 3ED5 3ED7 3EE6 3EF7 4019 4028 4036 403D-4040 "
                "404E-4050 4052 4054 4059-405A 405C 405F-4060 4064 4068 406A-406C 406E "
                "4070-4071 4073 4076 407D 4080 4084 4088-4089 408C-408D 409C-409D 40A0 "
                "40A4 40A6 40AB 40AD 40B3 40B7 40BA 40BC 40BE 40C0 "
                "40D0 BB15-BB62 BC1A-BC2A BC88-BC8D BCC2-BCC6 BCFC-BCFF BD1B-BD1C BD38 BF2A BF59 "
                "BF75 BF84"
            ),
        },
        "wormhole_b0": {
            -1: (
                "39EF-3AE2 3B82-3BAA 3BF7-3C0D 3C36-3C46 3C70-3C7F 3C95-3C9B 3CB3-3CB8 3CD0-3CD4 3CED-3CF1 3D05-3D06 "
                "3D14 3D22-3D23 3D31 3D3F 3F16 3F1D 3F2B 3F41 3F45 3F49 "
                "3F4E 3F53 3F57 3F5C 3F72 3F77 3F7B 3F82 3F8C 3F8F-3F93 "
                "3F96 3F98 3F9C-3F9E 3FB2 3FB6 3FBD-3FBE 3FC3-3FC5 3FC7 3FCD 3FD0-3FD2 "
                "3FD7 3FDB 3FE9 3FF5 4101 4107 410E 4112 BD87 BDA4 "
                "BDB2 BDC0 BDCF BDDD BDEB-BDEC BDFA BE04 BE13 BE1A BE21 "
                "BE29 BE30 BE37 BE46 BE4D BE55 BE5C BE63 BE6B BE72 "
                "BE7A BE8C BE90 BEAB BEAF BEB3 BEB7 BED8 BEF2"
            ),
            1: (
                "3D99 3DA0 3DA7 3DB6 3DBD 3DC4 3DCB 3DD2-3DD3 3DDA 3DE1 "
                "3DE8 3DEF 3DF6-3DF7 3DFD-3DFE 3E06 3E0A 3E0D 3E11 3E15 3E18 "
                "3E1C 3E23 3E27 3E2A 3E2E 3E32 3E35 3E39 3E3D 3E40 "
                "3E44 3E48 3E4B 3E4F 3E53 3E56 3E5A 3E5E 3E62 3E65 "
                "3E69 3E6D 3E74 3E78 3E7C 3E80 3E8B 3E8D 3E8F 3E91 "
                "3E93 3EAA 3EAC 3EAE 3EB0 3EB2 3EB4 3EB6 3EB8 3EBA "
                "3ED5 3ED7 3EE6 3EF7 4036 403D-403F 404E-4050 4052 4059-405A 405C "
                "405F-4060 4064 4068 406A-406C 406E 4070-4071 4073 407D 4080 4084 "
                "4088-4089 408C 409C-409D 40A0 40A4 40A6 40AB 40AD 40B3 40B7 "
                "40BA 40BC 40BE 40C0 40D0 BB0F-BB62 BC19-BC2A BC87-BC8D BCC1-BCC6 BCFB-BCFF "
                "BD1B-BD1C BD38 BF2A BF59 BF75 BF84"
            ),
        },
    }
    bits = torch.arange(1 << 16, dtype=torch.int32)
    x = bits.to(torch.int16).view(torch.bfloat16).reshape(1, 1, 256, 256)
    tt_x = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tiny = torch.finfo(torch.bfloat16).tiny

    def stored(t):
        t = t.to(torch.bfloat16).to(torch.float64).flatten()
        return torch.where(t.abs() < tiny, torch.zeros_like(t), t)

    def ordered(t):
        # BF16 words in value order, so that adjacent finite values differ by one step.
        words = t.to(torch.bfloat16).view(torch.int16).to(torch.int32)
        return torch.where(words < 0, -(words & 0x7FFF), words)

    actual = stored(ttnn.to_torch(ttnn.erfc(tt_x)))
    x = x.flatten().to(torch.float64)
    reference = (torch.special.erfc(x)).to(torch.float64).flatten()
    expected = stored(reference)
    finite = torch.isfinite(expected) & torch.isfinite(actual)
    ok = (actual == expected) | (finite & ((ordered(actual) - ordered(expected)).abs() <= STEPS))
    # A NaN result reaches L1 as a NaN or as an infinity of either sign.
    ok |= torch.isnan(reference) & (torch.isnan(actual) | torch.isinf(actual))
    # The SFPU flushes a result below the smallest normal, or within 2^-12 above it, to 0.
    ok |= (actual == 0) & (reference.abs() < tiny * (1 + 2.0**-12))
    listed = torch.zeros_like(ok)
    for word, ranges in STOCK.get(board, {}).items():
        value = stored(torch.tensor([word], dtype=torch.int32).to(torch.int16).view(torch.bfloat16))
        for first, last in ranges:
            lanes = (bits >= first) & (bits <= last)
            listed |= lanes
            ok |= lanes & ((actual == value) | (torch.isnan(actual) & torch.isnan(value)))
    for first, last in SKIP:
        listed |= (bits >= first) & (bits <= last)
        ok |= (bits >= first) & (bits <= last)
    shift = torch.zeros_like(bits)
    for offset, spans in MODEL.get(board, {}).items():
        for span in spans.split():
            first, _, last = span.partition("-")
            shift[(bits >= int(first, 16)) & (bits <= int(last or first, 16))] = offset
    moved = ordered(expected) + shift
    word = torch.where(moved < 0, (-moved) | 0x8000, moved)
    held = stored(word.to(torch.int16).view(torch.bfloat16))
    normal = ((bits & 0x7F80) != 0) & ((bits & 0x7F80) != 0x7F80)
    ok = torch.where(normal & ~listed & torch.isfinite(expected), actual == held, ok)
    assert ok.all(), (
        f"{(~ok).sum().item()} of {ok.numel()} inputs off torch and stock; first input 0x{bits[~ok][0].item():04x} "
        f"stored {actual[~ok][0].item()!r}, torch {reference[~ok][0].item()!r}"
    )
