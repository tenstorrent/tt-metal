# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Generated BF16 softplus cases, beside the unary category suites."""

import torch
import pytest
import ttnn


def test_softplus_bf16_unflushed_zero_and_subnormal_inputs(device):
    """-0 and subnormal inputs (kept by TT-NN's copy into DEST) store torch's result within 1 ULP."""
    board = "blackhole" if ttnn.device.is_blackhole(device) else "wormhole_b0"
    if board not in ("blackhole", "wormhole_b0"):
        pytest.skip(f"ttnn.softplus keeps its stock kernel on {board}")
    words = torch.cat([torch.arange(0x80), torch.arange(0x8000, 0x8080)]).to(torch.int16)
    x = words.view(torch.bfloat16).repeat(4).reshape(1, 1, 32, 32)
    tt_x = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tiny = torch.finfo(torch.bfloat16).tiny

    def stored(t):
        t = t.to(torch.bfloat16).to(torch.float64).flatten()
        t = torch.where(torch.isnan(t), torch.full_like(t, float("inf")), t)
        return torch.where(t.abs() < tiny, torch.zeros_like(t), t)

    actual = stored(ttnn.to_torch(ttnn.softplus(tt_x)))
    x64 = x.flatten().to(torch.float64)
    reference = ttnn.get_golden_function(ttnn.softplus)(x64).to(torch.float64).flatten()
    rounded = stored(reference)
    spacing = 2.0 ** (torch.floor(torch.log2(reference.abs().clamp(min=tiny))) - 7)
    close = torch.isfinite(reference) & (actual != 0) & ((actual - reference).abs() < spacing)
    ok = (actual == rounded) | close
    bits = words.repeat(4).to(torch.int32) & 0xFFFF
    assert ok.all(), (
        f"{(~ok).sum().item()} of {ok.numel()} inputs off torch and stock; first input 0x{bits[~ok][0].item():04x} "
        f"stored {actual[~ok][0].item()!r}, torch {reference[~ok][0].item()!r}"
    )


def test_softplus_bf16_bfloat8_b(device):
    """bfloat8_b in and out: PCC 0.999 against torch, and each output within one bfloat8_b step of the BF16 call's
    packed to bfloat8_b (two in a block holding an infinity or NaN)."""
    board = "blackhole" if ttnn.device.is_blackhole(device) else "wormhole_b0"
    if board not in ("blackhole", "wormhole_b0"):
        pytest.skip(f"ttnn.softplus keeps its stock kernel on {board}")
    torch.manual_seed(0)
    x = torch.empty(1, 1, 64, 64).uniform_(-10.0, 10.0).to(torch.bfloat16)
    tt_x = ttnn.from_torch(x, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device)
    stored = ttnn.to_torch(tt_x).to(torch.float64).flatten()
    output = ttnn.softplus(tt_x)
    assert output.dtype == ttnn.bfloat8_b
    actual = ttnn.to_torch(output).to(torch.float64).flatten()
    golden = ttnn.get_golden_function(ttnn.softplus)(stored).to(torch.float64)
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
    actual = ttnn.to_torch(ttnn.softplus(tt_x)).to(torch.float32).flatten()
    expected = ttnn.to_torch(ttnn.typecast(ttnn.softplus(tt_stored), ttnn.bfloat8_b)).to(torch.float32).flatten()
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


def test_softplus_bf16_nonfinite_inputs(device):
    """+-Inf and +-NaN store torch's result (NaN as an infinity) or STOCK's {input bits: word} for the board."""
    board = "blackhole" if ttnn.device.is_blackhole(device) else "wormhole_b0"
    if board not in ("blackhole", "wormhole_b0"):
        pytest.skip(f"ttnn.softplus keeps its stock kernel on {board}")
    STOCK = {"blackhole": {0xFFC0: 0x0000}, "wormhole_b0": {0xFFC0: 0x0000}}
    bits = torch.tensor([0x7F80, 0xFF80, 0x7FC0, 0xFFC0], dtype=torch.int32).repeat(256)
    x = bits.to(torch.int16).view(torch.bfloat16).reshape(1, 1, 32, 32)
    tt_x = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    def stored(t):
        return t.to(torch.bfloat16).to(torch.float64).flatten()

    actual = stored(ttnn.to_torch(ttnn.softplus(tt_x)))
    reference = ttnn.get_golden_function(ttnn.softplus)(x.flatten().to(torch.float64)).to(torch.float64)
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


def test_softplus_bf16_exhaustive_inputs(device):
    """All 65,536 BF16 inputs: torch's result as stored, within STEPS BF16 steps, or STOCK's word per board.

    SKIP lists inputs the LLK sweep judges instead. MODEL: per board, the kernel's offset in BF16 steps from torch's word.
    """
    board = "blackhole" if ttnn.device.is_blackhole(device) else "wormhole_b0"
    if board not in ("blackhole", "wormhole_b0"):
        pytest.skip(f"ttnn.softplus keeps its stock kernel on {board}")
    STOCK = {"blackhole": {0x0000: [(0xFF81, 0xFFFF)]}, "wormhole_b0": {0x0000: [(0xFF81, 0xFFFF)]}}
    STEPS = 1
    SKIP = [(0xC2AF, 0xC2B7)]
    MODEL = {
        "blackhole": {
            -1: (
                "39DF-3B2C 3C07-3C24 3C83-3C8E 3CC3-3CCB 3D01-3D03 3D21 3D40 3EE0 3EEA 3EF4 "
                "3EF7 3EFA 3F02 3F0A 3F1F 3F22 3F25 3F3D 3F40 3F96 "
                "3FA8 BBAF-BBF2 BC60-BC7A BCB4-BCBD BCF8-BCFE BD1E-BD1F BD40 BEDB BEE1 BEE6 "
                "BEE9 BEEB BEEE BEF3 BEF6 BEF8 BEFB BF00 BF04 BF08 "
                "BF0B-BF0C BF0F-BF10 BF13 BF16-BF17 BF1A BF1D BF33 BF36 BF39 BF44 "
                "BF9B-BF9D BFA3 BFB2 BFEB C026 C033 C054 C07B C080 C097 "
                "C0A8 C0C3 C0C6 C0C8 C0D9 C0DB C0DF C0EF C0F4 C10E "
                "C11A C139 C13D C144 C146 C150 C153 C15B C174 C184 "
                "C194 C19F-C1A0 C1A5 C1B7 C1C1 C1D1 C1DD C1E2 C1E9 C1F4 "
                "C207 C20D C213 C22C C247 C24C C255 C25D C265 C269 "
                "C27E C282 C285 C289 C2AA C2AD"
            ),
            1: (
                "0080-3631 3D7D 3D8E 3D9D 3DAC-3DAD 3DBB-3DBC 3DCA-3DCB 3DD9-3DDB 3DE8-3DEA 3DF6-3DF9 "
                "3E03-3E04 3E0A-3E0B 3E12-3E13 3E19-3E1A 3E21-3E22 3E28-3E29 3E2F-3E30 3E37-3E38 3E3E-3E3F 3E45-3E46 "
                "3E4D-3E4E 3E54-3E55 3E5B-3E5C 3E62-3E63 3E6A 3E71-3E72 3E78-3E79 3E7F-3E80 3E83 3E87 "
                "3E8E 3E95 3E9C 3EA3 3EAA 3F31 3F5A 8080-B631 BD82 BD92-BD93 "
                "BDA3-BDA4 BDB4-BDB5 BDC4-BDC6 BDD5-BDD7 BDE6-BDE9 BDF7-BDFA BE04-BE05 BE0D-BE0E BE15-BE17 BE1E-BE1F "
                "BE27-BE28 BE30-BE31 BE38-BE3A BE41-BE43 BE4A-BE4B BE53-BE54 BE5C-BE5D BE65-BE66 BE6E-BE6F BE77-BE78 "
                "BE80 BE85 BE89 BE8E BE93 BE97 BE9C BEAA BF51 BF56 "
                "BF5B BF60 BF65 BF67 BF6C BF75 BF7E BFAF BFB1 BFC0 "
                "BFC2 BFC4 BFC7 BFC9 BFCC BFCF BFD2 BFD6 BFDA BFDE "
                "C00A C00E-C015 C019 C01E C039 C03D-C03E C044-C045 C049 C05D C066 "
                "C068-C069 C06D-C070 C08B C0A1 C0A3-C0A5 C0B8 C0C7 C0CA C0CD C0E2 "
                "C0E5 C0FC C0FF C108-C109 C110 C112 C115 C120 C12B C133-C134 "
                "C13F-C140 C14B C160 C16B C178 C181 C185-C187 C18D C1A8 C1BE "
                "C1C4 C1D5 C1D9 C1E5 C1FA C209 C20B C20E C211 C219 "
                "C21F C22D C240 C243 C24B C253 C256 C261-C262 C26A C272 "
                "C27D C280 C283 C287 C28F C295 C29E C2A7"
            ),
        },
        "wormhole_b0": {
            -1: (
                "39DF-3B2C 3C07-3C24 3C83-3C8E 3CC3-3CCB 3D01-3D03 3D21 3D40 3EE0 3EEA 3EF4 "
                "3EF7 3EFA 3F02 3F0A 3F1F 3F22 3F25 3F3D 3F40 3F96 "
                "3FA8 BBAF-BBF2 BC60-BC7A BCB4-BCBD BCF8-BCFE BD1E-BD1F BD40 BEDB BEE1 BEE6 "
                "BEE9 BEEB BEEE BEF3 BEF6 BEF8 BEFB BF00 BF04 BF08 "
                "BF0B-BF0C BF0F-BF10 BF13 BF16-BF17 BF1A BF1D BF33 BF36 BF39 BF44 "
                "BF9B-BF9D BFA3 BFB2 BFEB C026 C033 C054 C07B C080 C097 "
                "C0A8 C0C3 C0C6 C0C8 C0D9 C0DB C0DF C0EF C0F4 C10E "
                "C11A C139 C144 C146 C150 C153 C15B C174 C184 C194 "
                "C19F-C1A0 C1A5 C1B7 C1C1 C1D1 C1DD C1E2 C1E9 C1F4 C207 "
                "C20D C213 C22C C247 C24C C255 C25D C265 C269 C27E "
                "C282 C285 C289 C2AA C2AD"
            ),
            1: (
                "0080-3631 3D7D 3D8E 3D9D 3DAC-3DAD 3DBB-3DBC 3DCA-3DCB 3DD9-3DDB 3DE8-3DEA 3DF6-3DF9 "
                "3E03-3E04 3E0A-3E0B 3E12-3E13 3E19-3E1A 3E21-3E22 3E28-3E29 3E2F-3E30 3E37-3E38 3E3E-3E3F 3E45-3E46 "
                "3E4D-3E4E 3E54-3E55 3E5B-3E5C 3E62-3E63 3E6A 3E71-3E72 3E78-3E79 3E7F-3E80 3E83 3E87 "
                "3E8E 3E95 3E9C 3EA3 3EAA 3F31 3F5A 8080-B631 BD82 BD92-BD93 "
                "BDA3-BDA4 BDB4-BDB5 BDC4-BDC6 BDD5-BDD7 BDE6-BDE9 BDF7-BDFA BE04-BE05 BE0D-BE0E BE15-BE17 BE1E-BE1F "
                "BE27-BE28 BE30-BE31 BE38-BE3A BE41-BE43 BE4A-BE4B BE53-BE54 BE5C-BE5D BE65-BE66 BE6E-BE6F BE77-BE78 "
                "BE80 BE85 BE89 BE8E BE93 BE97 BE9C BEAA BF51 BF56 "
                "BF5B BF60 BF65 BF67 BF6C BF75 BF7E BFAF BFB1 BFC0 "
                "BFC2 BFC4 BFC7 BFC9 BFCC BFCF BFD2 BFD6 BFDA BFDE "
                "C00A C00E-C015 C019 C01E C039 C03D-C03E C044-C045 C049 C05D C066 "
                "C068-C069 C06D-C070 C08B C0A1 C0A3-C0A5 C0B8 C0C7 C0CA C0CD C0E2 "
                "C0E5 C0FC C0FF C108-C109 C110 C112 C115 C120 C12B C133-C134 "
                "C13F-C140 C14B C160 C16B C178 C181 C185-C187 C18D C1A8 C1BE "
                "C1C4 C1D5 C1D9 C1E5 C1FA C209 C20B C20E C211 C219 "
                "C21F C22D C240 C243 C24B C253 C256 C261-C262 C26A C272 "
                "C27D C280 C283 C287 C28F C295 C29E C2A7"
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

    actual = stored(ttnn.to_torch(ttnn.softplus(tt_x)))
    x = x.flatten().to(torch.float64)
    reference = (torch.nn.functional.softplus(x)).to(torch.float64).flatten()
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
