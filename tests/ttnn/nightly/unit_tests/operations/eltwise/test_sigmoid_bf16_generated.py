# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Generated BF16 sigmoid cases, beside the unary category suites."""

import torch
import ttnn


def test_sigmoid_bf16_unflushed_zero_and_subnormal_inputs(device):
    """-0 and subnormal inputs (kept by TT-NN's copy into DEST) store torch's result within 1 ULP."""
    words = torch.cat([torch.arange(0x80), torch.arange(0x8000, 0x8080)]).to(torch.int16)
    x = words.view(torch.bfloat16).repeat(4).reshape(1, 1, 32, 32)
    tt_x = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tiny = torch.finfo(torch.bfloat16).tiny

    def stored(t):
        t = t.to(torch.bfloat16).to(torch.float64).flatten()
        t = torch.where(torch.isnan(t), torch.full_like(t, float("inf")), t)
        return torch.where(t.abs() < tiny, torch.zeros_like(t), t)

    actual = stored(ttnn.to_torch(ttnn.sigmoid(tt_x)))
    x64 = x.flatten().to(torch.float64)
    reference = ttnn.get_golden_function(ttnn.sigmoid)(x64).to(torch.float64).flatten()
    rounded = stored(reference)
    spacing = 2.0 ** (torch.floor(torch.log2(reference.abs().clamp(min=tiny))) - 7)
    close = torch.isfinite(reference) & (actual != 0) & ((actual - reference).abs() < spacing)
    ok = (actual == rounded) | close
    bits = words.repeat(4).to(torch.int32) & 0xFFFF
    assert ok.all(), (
        f"{(~ok).sum().item()} of {ok.numel()} inputs off torch and stock; first input 0x{bits[~ok][0].item():04x} "
        f"stored {actual[~ok][0].item()!r}, torch {reference[~ok][0].item()!r}"
    )


def test_sigmoid_bf16_bfloat8_b(device):
    """bfloat8_b in and out: each output within one bfloat8_b step of the BF16 call's packed to bfloat8_b (two in a
    block holding an infinity or NaN). The exhaustive case holds the BF16 call to torch on every input."""
    # Every finite BF16 exponent and +-0, one exponent per 16-value face row.
    words = torch.arange(1 << 16, dtype=torch.int32)
    exponent = (words >> 7) & 0xFF
    zeros = torch.tensor([0x0000] * 256 + [0x8000] * 256, dtype=torch.int32)
    words = torch.cat([words[(exponent > 0) & (exponent < 0xFF)], zeros])
    x = words.to(torch.int16).view(torch.bfloat16).reshape(1, 1, 256, 256)
    tt_x = ttnn.from_torch(x, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device)
    stored = ttnn.to_torch(tt_x).to(torch.bfloat16)
    tt_stored = ttnn.from_torch(stored, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    output = ttnn.sigmoid(tt_x)
    assert output.dtype == ttnn.bfloat8_b
    actual = ttnn.to_torch(output).to(torch.float32).flatten()
    expected = ttnn.to_torch(ttnn.typecast(ttnn.sigmoid(tt_stored), ttnn.bfloat8_b)).to(torch.float32).flatten()
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


def test_sigmoid_bf16_nonfinite_inputs(device):
    """+-Inf and +-NaN store torch's result (NaN as an infinity) or STOCK's {input bits: word} for the board."""
    board = "blackhole" if ttnn.device.is_blackhole(device) else "wormhole_b0"
    STOCK = {"blackhole": {0x7FC0: 0x0000, 0xFFC0: 0x0000}, "wormhole_b0": {0x7FC0: 0x3F80, 0xFFC0: 0x0000}}
    bits = torch.tensor([0x7F80, 0xFF80, 0x7FC0, 0xFFC0], dtype=torch.int32).repeat(256)
    x = bits.to(torch.int16).view(torch.bfloat16).reshape(1, 1, 32, 32)
    tt_x = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    def stored(t):
        return t.to(torch.bfloat16).to(torch.float64).flatten()

    actual = stored(ttnn.to_torch(ttnn.sigmoid(tt_x)))
    reference = ttnn.get_golden_function(ttnn.sigmoid)(x.flatten().to(torch.float64)).to(torch.float64)
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


def test_sigmoid_bf16_exhaustive_inputs(device):
    """All 65,536 BF16 inputs: torch's result as stored, within STEPS BF16 steps, or STOCK's word per board.

    SKIP lists inputs the LLK sweep judges instead. MODEL: per board, the kernel's offset in BF16 steps from torch's word.
    """
    board = "blackhole" if ttnn.device.is_blackhole(device) else "wormhole_b0"
    STOCK = {
        "blackhole": {0x0000: [(0x7F81, 0x7FFF), (0xFF81, 0xFFFF)]},
        "wormhole_b0": {0x0000: [(0xFF81, 0xFFFF)], 0x3F80: [(0x7F81, 0x7FFF)]},
    }
    STEPS = 1
    SKIP = [(0xC2AF, 0xC2B7)]
    MODEL = {
        "blackhole": {
            -1: (
                "3DB1 3DD1-3DD2 3DF1-3DF2 3E09 3E19 3E29-3E2A 3E39-3E3A 3E49-3E4A 3E59-3E5A 3E6A "
                "3E7A 3E85 3E8D 3EB6 3F31 3F51 3F64 3F69 3F6E 3F73 "
                "3F78 3F81 3FD6 4005 BB17-BB80 BC2D-BC40 BC9A-BCA0 BCDD-BCE0 BD10 BEBC "
                "BEC0 BEC4 BEC8 BED5 BED9 BEDD BEE1 BEE5 BEEE BEF2 "
                "BEF6 BEFF BF08 BF0A BF15 BF17 BF20 BF34 BF93 BF95 "
                "BF97 BF9C BFA1 BFA4 BFB4 BFE1-BFE6 BFF4-BFF5 BFFB BFFE C006-C007 "
                "C01C C01F C024-C025 C04E C051 C057-C058 C05E C078 C07A C07C "
                "C07F C082 C091 C096 C098 C0AA C0B2 C0C3 C0C8 C0D9 "
                "C0DB C0EF C0F4 C10E C11A C12E C139 C144 C150 C153 "
                "C15B C174 C184 C194 C19F C1A5 C1B7 C1C1 C1D1 C1DD "
                "C1E2 C1E9 C1F4 C207 C20D C20F C213 C21A C220 C228 "
                "C233 C23E C241 C247 C24C C255 C25D C265 C269 C27E "
                "C282 C289 C298 C2AA"
            ),
            1: (
                "3BD1-3C00 3CB9-3CC0 3D1F-3D20 3ED7 3EF0 3F09 3F16 3F35 3F96 3FEE "
                "4000 4034 BD51 BD71-BD72 BD89 BD99-BD9A BDA9-BDAA BDB9-BDBA BDC9-BDCB BDD9-BDDB "
                "BDE9-BDEB BDF9-BDFB BE05 BE0D BE15-BE16 BE1D-BE1E BE25-BE26 BE2D-BE2E BE35-BE36 BE3D-BE3E "
                "BE45-BE46 BE4D-BE4E BE55-BE56 BE5D-BE5E BE65-BE66 BE6E BE76 BE7E BE83 BE87 "
                "BE8B BE8F BF29 BF49 BF50 BF57 BF5E BF63 BF6F BF74 "
                "BF79 BF7E BFB1 BFBD BFC2-BFC3 BFC8-BFC9 BFD0-BFD1 C002 C009-C00A C010 "
                "C03A-C03B C041 C05D C063 C066 C069-C06A C088-C089 C08D-C08E C09E-C0A0 C0B5-C0B6 "
                "C0BA C0CE C0E1-C0E3 C0E5 C108-C109 C110 C112 C11C C12B C133-C134 "
                "C13F C14B C154 C160 C16B C181 C185-C187 C197 C1A8 C1AD "
                "C1BE C1C3-C1C4 C1D5 C1D9 C1E5 C1FA C20B C20E C211 C219 "
                "C22D C240 C243 C24B C253 C256 C261 C272 C27D C280 "
                "C287-C288 C28F C29E"
            ),
        },
        "wormhole_b0": {
            -1: (
                "3DB1 3DD1-3DD2 3DF1-3DF2 3E09 3E19 3E29-3E2A 3E39-3E3A 3E49-3E4A 3E59-3E5A 3E6A "
                "3E7A-3E7B 3E85 3E8D 3EAE 3EB6 3F31 3F51 3F64 3F69 3F6E "
                "3F73 3F78 3F7D 3F81 3FD6 4005 4089 4094 40C8-40C9 BB0A-BB80 "
                "BC2A-BC40 BC99-BCA0 BCDC-BCE0 BD0F-BD10 BEBC BEC0 BEC4 BEC8 BED5 BED9 "
                "BEDD BEE1 BEE5 BEEE BEF2 BEF6 BEFA BEFF BF08 BF0A "
                "BF15 BF17 BF20 BF34 BF93 BF95 BF97 BF9C BFA1 BFA4 "
                "BFB4 BFE0-BFE7 BFF4-BFF5 BFFB BFFE C006-C007 C01C C01F C024-C025 C04E "
                "C051 C056-C058 C05E C078 C07A C07C C07F C082 C091 C096 "
                "C098 C0AA C0B2 C0C3 C0C8 C0D9 C0DB C0EF C0F4 C10E "
                "C11A C12E C139 C144 C150 C153 C15B C174 C184 C194 "
                "C19F-C1A0 C1A5 C1B7 C1C1 C1D1 C1DD C1E2 C1E9 C1F4 C207 "
                "C20D C20F C213 C21A C220 C228 C233 C23E C241 C247 "
                "C24C C255 C25D C265 C269 C27E C282 C289 C298 C2AA"
            ),
            1: (
                "3BD5-3C00 3CB9-3CC0 3D1F-3D20 3ED7 3EF0 3F09 3F16 3F35 3FEE 4000 "
                "4034 BD51 BD71-BD72 BD89 BD99-BD9A BDA9-BDAA BDB9-BDBA BDC9-BDCB BDD9-BDDB BDE9-BDEB "
                "BDF9-BDFB BE05 BE0D BE15-BE16 BE1D-BE1E BE25-BE26 BE2D-BE2E BE35-BE36 BE3D-BE3E BE45-BE46 "
                "BE4D-BE4E BE55-BE56 BE5D-BE5E BE65-BE66 BE6E BE76 BE7E BE83 BE87 BE8B "
                "BE8F BF29 BF49 BF50 BF57 BF5E BF63 BF6F BF74 BF79 "
                "BF7E BFB1 BFBD BFC2 BFC8-BFC9 BFD0-BFD1 C002 C009-C00A C010 C03A-C03B "
                "C041 C05D C063 C066 C069-C06A C088-C089 C08D-C08E C09E-C0A0 C0B5-C0B6 C0CE "
                "C0E1-C0E3 C0E5 C108-C109 C110 C112 C11C C12B C133-C134 C13F C14B "
                "C154 C160 C16B C181 C185-C187 C197 C1A8 C1BE C1C3-C1C4 C1D5 "
                "C1D9 C1E5 C1FA C20B C20E C211 C219 C22D C240 C243 "
                "C24B C253 C256 C261 C272 C27D C280 C287-C288 C28F C29E"
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

    actual = stored(ttnn.to_torch(ttnn.sigmoid(tt_x)))
    x = x.flatten().to(torch.float64)
    reference = (torch.sigmoid(x)).to(torch.float64).flatten()
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
