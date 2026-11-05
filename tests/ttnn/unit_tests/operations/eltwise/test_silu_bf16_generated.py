# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Generated BF16 silu cases, beside the unary category suites."""

import torch
import pytest
import ttnn
from tests.ttnn.unit_tests.operations.eltwise.eltwise_test_utils import (
    generate_bfloat16_bits,
    to_tt_tensor,
    SMALLEST_NORMAL_BF16,
)


@pytest.mark.parametrize("generated_first", [True, False])
def test_silu_with_sqrt_in_sfpu_chain(device, generated_first):
    """silu chained with stock sqrt, both orders: each step stores what its op stores standalone."""
    generated = (ttnn.UnaryWithParam(ttnn.UnaryOpType.SILU), lambda tensor: ttnn.silu(tensor))
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


def test_silu_bf16_unflushed_zero_and_subnormal_inputs(device):
    """-0 and subnormal inputs (kept by TT-NN's copy into DEST) store +0."""
    words = torch.cat([torch.arange(0x80), torch.arange(0x8000, 0x8080)]).to(torch.int16)
    x = words.view(torch.bfloat16).repeat(4).reshape(1, 1, 32, 32)
    tt_x = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    raw = ttnn.to_torch(ttnn.silu(tt_x)).reshape(-1).view(torch.int16).to(torch.int32) & 0xFFFF
    assert torch.all(raw == 0x0000)


def test_silu_bf16_bfloat8_b(device):
    """bfloat8_b in and out: PCC 0.999 against torch, and each output within one bfloat8_b step of the BF16 call's
    packed to bfloat8_b (two in a block holding an infinity or NaN)."""
    torch.manual_seed(0)
    x = torch.empty(1, 1, 64, 64).uniform_(-10.0, 10.0).to(torch.bfloat16)
    tt_x = ttnn.from_torch(x, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device)
    stored = ttnn.to_torch(tt_x).to(torch.float64).flatten()
    output = ttnn.silu(tt_x)
    assert output.dtype == ttnn.bfloat8_b
    actual = ttnn.to_torch(output).to(torch.float64).flatten()
    golden = ttnn.get_golden_function(ttnn.silu)(stored).to(torch.float64)
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
    actual = ttnn.to_torch(ttnn.silu(tt_x)).to(torch.float32).flatten()
    expected = ttnn.to_torch(ttnn.typecast(ttnn.silu(tt_stored), ttnn.bfloat8_b)).to(torch.float32).flatten()
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


def test_silu_bf16_nonfinite_inputs(device):
    """+-Inf and +-NaN store torch's result (NaN as an infinity) or STOCK's {input bits: word} for the board."""
    board = "blackhole" if ttnn.device.is_blackhole(device) else "wormhole_b0"
    STOCK = {"wormhole_b0": {0xFF80: 0xFF80, 0xFFC0: 0xFF80}}
    bits = torch.tensor([0x7F80, 0xFF80, 0x7FC0, 0xFFC0], dtype=torch.int32).repeat(256)
    x = bits.to(torch.int16).view(torch.bfloat16).reshape(1, 1, 32, 32)
    tt_x = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    def stored(t):
        return t.to(torch.bfloat16).to(torch.float64).flatten()

    actual = stored(ttnn.to_torch(ttnn.silu(tt_x)))
    reference = ttnn.get_golden_function(ttnn.silu)(x.flatten().to(torch.float64)).to(torch.float64)
    expected = stored(reference)
    # TT-NN's golden departs from torch here; torch's own words, as {input bits: word}.
    for input_bits, word in {0x7F80: 0x7F80, 0x7FC0: 0x7FC0, 0xFF80: 0x7FC0, 0xFFC0: 0x7FC0}.items():
        expected[bits == input_bits] = stored(
            torch.tensor([word], dtype=torch.int32).to(torch.int16).view(torch.bfloat16)
        )
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


def test_silu_bf16_exhaustive_inputs(device):
    """All 65,536 BF16 inputs: torch's result as stored, within STEPS BF16 steps, or STOCK's word per board.

    SKIP lists inputs the LLK sweep judges instead. MODEL: per board, the kernel's offset in BF16 steps from torch's word.
    """
    board = "blackhole" if ttnn.device.is_blackhole(device) else "wormhole_b0"
    STOCK = {"wormhole_b0": {0xFF80: [(0xFF81, 0xFFFF)]}}
    STEPS = 1
    SKIP = [(0x0003, 0x00FE), (0x8004, 0x80FE), (0xC2B8, 0xC2C0), (0xFF80, 0xFF80)]
    MODEL = {
        "blackhole": {
            -1: (
                "3D78 3D88 3DB0 3DBB 3DC6 3DD0 3DDA 3DE3 3DEC 3E0C "
                "3E13 3E1A 3E27 3E2D 3E33 3E3E 3E43 3E49 3E4E 3E53 "
                "3E5C 3E61 3E6C 3E79 3E8E 3E95 3F2F 3F54 3F5C 3F65 "
                "3F6F 3F7B 3FB1 BD78 BD84 BD88 BDB0 BDBB BDC6 BDD0 "
                "BDDA BDE3 BDEC BDF4-BDF5 BDFD BE06 BE0C BE13 BE1A BE27 "
                "BE2D BE33 BE3E BE43 BE49 BE4E BE53 BE5C BE61 BE6A "
                "BE6E BE77 BE7B BE7F BE8B BE8D BF31 BF45 BF4A BF52 "
                "BF56 BF5A BF5E BF63 BF72 BF78 BF7E BF87 BFBE BFC4 "
                "BFC9 BFCE BFD2 BFD6 C00D C00F C014 C019 C03C C041 "
                "C063 C06B-C070 C08B C08E-C08F C0A2-C0A4 C0B5-C0B8 C0CB C0DD C0E1 C0F3 "
                "C107 C109 C113 C115 C12A C133-C134 C13E C149 C14B C156 "
                "C160-C161 C16C-C16D C185 C192 C19E C1B3 C1BF C1C9 C1CD C1DA "
                "C1E0 C1E5 C1F6 C206 C20B C21E C229 C22D C24E C256 "
                "C259 C25F C262 C275 C27A C280 C292 C296 C299 C2A1 "
                "C2B6"
            ),
            1: (
                "3B51-3B7F 3B9E-3BB5 3C00 3C54-3C5D 3C7F 3C99-3C9C 3CC7-3CCA 3CED-3CEF 3CFF 3D0E-3D0F "
                "3D29 3D40 3ED2 3EED 3EF8 3F0E 3F10 3F19 3F35 3F8F "
                "4000 4081 40BE BB51-BB80 BB9E-BBB5 BC54-BC5D BC99-BC9C BCC7-BCCA BCED-BCEF BD0E-BD0F "
                "BD29 BD40 BED2 BED7 BEEC BEF7 BEF9 BF15-BF16 BF1D BF33 "
                "BF3A BF8C BF93 BFA0-BFA8 BFE7 BFEA BFED BFF1 BFF4 BFF7-BFF8 "
                "BFFB BFFE C021 C026 C028 C04E C056 C058 C05E C079 "
                "C083 C095 C0A8 C0C1-C0C3 C0D9 C0DC C0EB C0EF C0F4 C102 "
                "C10E C11A C125 C130 C139 C13D C14F C151 C159 C169 "
                "C174 C17C C18F C1AA C1B7 C1BB C1C6 C1D1-C1D2 C1D7 C1EE "
                "C1F4 C1F7 C202 C20A C21B C223 C237 C244 C247 C255 "
                "C258 C273 C281 C286 C289 C29F C2A2 C2AD C2B4"
            ),
        },
        "wormhole_b0": {
            -1: (
                "3D78 3D88 3DB0 3DBB 3DC6 3DD0 3DDA 3DE3 3DEC 3E0C "
                "3E13 3E1A 3E27 3E2D 3E33 3E3E 3E43 3E4E 3E53 3E5C "
                "3E61 3E6C 3E79 3E8E 3E95 3F2F 3F54 3F5C 3F65 3F6F "
                "3F7B 3FB1 BD78 BD84 BD88 BDB0 BDBB BDC6 BDD0 BDDA "
                "BDE3 BDEC BDF4-BDF5 BDFD BE06 BE0C BE13 BE1A BE27 BE2D "
                "BE33 BE3E BE43 BE4E BE53 BE5C BE61 BE6A BE6E BE77 "
                "BE7B BE7F BE8B BE8D BF31 BF45 BF4A BF52 BF56 BF5A "
                "BF5E BF63 BF72 BF78 BF7E BF87 BFBE BFC4 BFC9 BFCE "
                "BFD2 BFD6 C00D C00F C014 C019 C03C C041 C063 C06B-C070 "
                "C08E-C08F C0A2-C0A3 C0B5-C0B8 C0CB C0DD C0E1 C0F3 C107 C109 C113 "
                "C115 C12A C133-C134 C13E C149 C14B C156 C160-C161 C16C-C16D C185 "
                "C192 C19E C1B3 C1C9 C1CD C1DA C1E0 C1E5 C1F6 C20B "
                "C21E C229 C22D C24E C256 C259 C25F C27A C280 C292 "
                "C296 C299 C2A1 C2B6"
            ),
            1: (
                "3B4D-3B7F 3B9C-3BB5 3C00 3C53-3C5D 3C7F 3C99-3C9C 3CC7-3CCA 3CED-3CEF 3CFF 3D0E-3D0F "
                "3D29 3D40 3ED2 3EED 3EF8 3F0E 3F10 3F19 3F35 3F8F "
                "4000 4081 40BE BB4D-BB80 BB9C-BBB5 BC53-BC5D BC99-BC9C BCC7-BCCA BCED-BCEF BD0E-BD0F "
                "BD29 BD40 BED2 BED7 BEEC BEF7 BEF9 BEFB BF15-BF16 BF1D "
                "BF33 BF3A BF8C BF93 BFA0-BFA8 BFE7 BFEA BFED BFF1 BFF4 "
                "BFF7-BFF8 BFFB BFFE C021 C026 C028 C04E C056 C058 C05E "
                "C079 C083 C095 C0A8 C0C1-C0C3 C0D9 C0DC C0EB C0EF C0F4 "
                "C102 C10E C11A C125 C12E C130 C139 C13D C14F C151 "
                "C159 C169 C174 C17C C18F C1A6 C1AA C1B0 C1B7 C1BB "
                "C1C6 C1D1-C1D2 C1D7 C1DE C1EE C1F4 C1F7 C202 C20A C21B "
                "C221 C223 C237 C244 C247 C255 C258 C25E C273 C281 "
                "C286 C289 C29F C2A2 C2AD C2B4"
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

    actual = stored(ttnn.to_torch(ttnn.silu(tt_x)))
    x = x.flatten().to(torch.float64)
    reference = (torch.nn.functional.silu(x)).to(torch.float64).flatten()
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
