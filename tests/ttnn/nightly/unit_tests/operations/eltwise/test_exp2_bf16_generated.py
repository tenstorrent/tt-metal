# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Generated BF16 exp2 cases, beside the unary category suites."""

import torch
import pytest
import ttnn
from tests.ttnn.unit_tests.operations.eltwise.eltwise_test_utils import (
    generate_bfloat16_bits,
    to_tt_tensor,
    SMALLEST_NORMAL_BF16,
)


@pytest.mark.parametrize("generated_first", [True, False])
def test_exp2_with_sqrt_in_sfpu_chain(device, generated_first):
    """exp2 chained with stock sqrt, both orders: each step stores what its op stores standalone."""
    generated = (ttnn.UnaryWithParam(ttnn.UnaryOpType.EXP2), lambda tensor: ttnn.exp2(tensor))
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


def test_exp2_bf16_unflushed_zero_and_subnormal_inputs(device):
    """-0 and subnormal inputs (kept by TT-NN's copy into DEST) store torch's result within 1 ULP."""
    words = torch.cat([torch.arange(0x80), torch.arange(0x8000, 0x8080)]).to(torch.int16)
    x = words.view(torch.bfloat16).repeat(4).reshape(1, 1, 32, 32)
    tt_x = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tiny = torch.finfo(torch.bfloat16).tiny

    def stored(t):
        t = t.to(torch.bfloat16).to(torch.float64).flatten()
        t = torch.where(torch.isnan(t), torch.full_like(t, float("inf")), t)
        return torch.where(t.abs() < tiny, torch.zeros_like(t), t)

    actual = stored(ttnn.to_torch(ttnn.exp2(tt_x)))
    x64 = x.flatten().to(torch.float64)
    reference = ttnn.get_golden_function(ttnn.exp2)(x64).to(torch.float64).flatten()
    rounded = stored(reference)
    spacing = 2.0 ** (torch.floor(torch.log2(reference.abs().clamp(min=tiny))) - 7)
    close = torch.isfinite(reference) & (actual != 0) & ((actual - reference).abs() < spacing)
    ok = (actual == rounded) | close
    bits = words.repeat(4).to(torch.int32) & 0xFFFF
    assert ok.all(), (
        f"{(~ok).sum().item()} of {ok.numel()} inputs off torch and stock; first input 0x{bits[~ok][0].item():04x} "
        f"stored {actual[~ok][0].item()!r}, torch {reference[~ok][0].item()!r}"
    )


def test_exp2_bf16_bfloat8_b(device):
    """bfloat8_b in and out: PCC 0.999 against torch, and each output within one bfloat8_b step of the BF16 call's
    packed to bfloat8_b (two in a block holding an infinity or NaN)."""
    torch.manual_seed(0)
    x = torch.empty(1, 1, 64, 64).uniform_(-7.0, 7.0).to(torch.bfloat16)
    tt_x = ttnn.from_torch(x, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device)
    stored = ttnn.to_torch(tt_x).to(torch.float64).flatten()
    output = ttnn.exp2(tt_x)
    assert output.dtype == ttnn.bfloat8_b
    actual = ttnn.to_torch(output).to(torch.float64).flatten()
    golden = ttnn.get_golden_function(ttnn.exp2)(stored).to(torch.float64)
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
    actual = ttnn.to_torch(ttnn.exp2(tt_x)).to(torch.float32).flatten()
    expected = ttnn.to_torch(ttnn.typecast(ttnn.exp2(tt_stored), ttnn.bfloat8_b)).to(torch.float32).flatten()
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


def test_exp2_bf16_nonfinite_inputs(device):
    """+-Inf and +-NaN store torch's result (NaN as an infinity) or STOCK's {input bits: word} for the board."""
    board = "blackhole" if ttnn.device.is_blackhole(device) else "wormhole_b0"
    STOCK = {"wormhole_b0": {0xFFC0: 0x0000}}
    bits = torch.tensor([0x7F80, 0xFF80, 0x7FC0, 0xFFC0], dtype=torch.int32).repeat(256)
    x = bits.to(torch.int16).view(torch.bfloat16).reshape(1, 1, 32, 32)
    tt_x = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    def stored(t):
        return t.to(torch.bfloat16).to(torch.float64).flatten()

    actual = stored(ttnn.to_torch(ttnn.exp2(tt_x)))
    reference = ttnn.get_golden_function(ttnn.exp2)(x.flatten().to(torch.float64)).to(torch.float64)
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


def test_exp2_bf16_exhaustive_inputs(device):
    """All 65,536 BF16 inputs: torch's result as stored, within STEPS BF16 steps, or STOCK's word per board.

    SKIP lists inputs the LLK sweep judges instead. MODEL: per board, the kernel's offset in BF16 steps from torch's word.
    """
    board = "blackhole" if ttnn.device.is_blackhole(device) else "wormhole_b0"
    STOCK = {"wormhole_b0": {0x0000: [(0xFF81, 0xFFFF)]}}
    STEPS = 1
    SKIP = [(0xC2FD, 0xC304)]
    MODEL = {
        -1: (
            "3D79 3D93 3DA9 3DBE-3DC0 3DD4-3DD6 3DE9-3DEC 3DFF-3E00 3E0A-3E0B 3E15-3E16 3E1F-3E20 "
            "3E29-3E2B 3E34-3E35 3E3E-3E3F 3E48-3E49 3E52-3E54 3E5C-3E5E 3E66-3E67 3E70-3E71 3E7A-3E7B 3E82 "
            "3E87 3E8B-3E8C 3E90 3E95 3E9A 3E9E 3EA3 3EA8 3EAC 3EB1 "
            "3EB5 3EBA 3EBE 3EC7 3ED4 3F78 3F7B 3F7E 3F8C 3F90 "
            "3F94 3F99 3F9E 3FA3-3FA4 3FAA-3FAB 3FB5 3FFC 3FFF 4006 4008 "
            "400A 400F 4012 4015 403E 4046 4048 404A 404F 4052 "
            "4055 407E 4083-4085 4089 409F 40A3-40A5 40A9 40BF 40C3-40C5 40C9 "
            "40DF 40E3-40E5 40E9 40FF 4102 4112 4122 4132 4142 4152 "
            "4162 4172 4181 4189 4191 4199 41A1 41A9 41B1 41B9 "
            "41C1 41C9 41D1 41D9 41E1 41E9 41F1 41F9 B9B3-BB38 BBCC-BC0A "
            "BC47-BC67 BC94-BCA2 BCC5-BCD1 BCF5-BD00 BD13-BD17 BD2C-BD2F BD44-BD47 BD5D-BD5F BD76-BD77 BF16 "
            "BF21 BF23 BF2A BF2C BF31 BF33 BF38 BF3A BF3F BF44 "
            "BF49 BF4B BF4E BF53 BF58 BF60 BF68 BF81 BF84 BFCB "
            "BFD5-BFD6 BFDC-BFDD BFE2 BFE7 BFEC BFF0 BFF4 C002 C02B C02E "
            "C031 C036 C038 C03A C042 C06B C06E C071 C076 C078 "
            "C07A C081 C097 C09B-C09D C0A1 C0B7 C0BB-C0BD C0C1 C0D7 C0DB-C0DD "
            "C0E1 C0F7 C0FB-C0FD C10E C11E C12E C13E C14E C15E C16E "
            "C17E C187 C18F C197 C19F C1A7 C1AF C1B7 C1BF C1C7 "
            "C1CF C1D7 C1DF C1E7 C1EF C1F7 C1FF"
        ),
        1: (
            "3B5A-3BB8 3C78-3C89 3CDB-3CE4 3D1D-3D1F 3D4C 3F03 3F05 3F07 3F09 3F0B "
            "3F1C 3F1E 3F20 3F22 3F27 3F29 3F2B 3F2D 3F32 3F34 "
            "3F39 3F3B 3F40 3F42 3F45 3F47 3F4A 3F4C 3F4F 3F51 "
            "3F54 3F56 3F59 3F5E 3F61 3F82 3FCE-3FD1 3FD9-3FDA 3FE0-3FE1 3FE5-3FE6 "
            "3FEA-3FEB 3FEF 4001 4027-4028 402D 4030 4033 4035 4041 4067-4068 "
            "406D 4070 4073 4075 4094 4098 40B4 40B8 40D4 40D8 "
            "40F4 40F8 410A 410C 411A 411C 412A 412C 413A 413C "
            "414A 414C 415A 415C 416A 416C 417A 417C 4185-4186 418D-418E "
            "4195-4196 419D-419E 41A5-41A6 41AD-41AE 41B5-41B6 41BD-41BE 41C5-41C6 41CD-41CE 41D5-41D6 41DD-41DE "
            "41E5-41E6 41ED-41EE 41F5-41F6 41FD-41FE 4203 4207 420B 420F 4213 4217 "
            "421B 421F 4223 4227 422B 422F 4233 4237 423B 423F "
            "4243 4247 424B 424F 4253 4257 425B 425F 4263 4267 "
            "426B 426F 4273 4277 427B 427F BDB9 BDC5 BDD2 BDDE-BDDF "
            "BDEB BDF7-BDF8 BE02 BE08-BE09 BE0F BE15-BE16 BE1C BE22-BE23 BE28-BE29 BE2F-BE30 "
            "BE35-BE36 BE3C-BE3D BE42-BE44 BE49-BE4A BE50-BE51 BE56-BE58 BE5D-BE5E BE64-BE65 BE6A-BE6C BE71-BE73 "
            "BE78-BE7A BE7F-BE80 BE83 BE87 BE8A BE8E BE91 BE95 BE98 BE9C "
            "BE9F BEA3 BEA6-BEA7 BEAA BEAE BEB1-BEB2 BEB5 BEB9 BEBC-BEBD BEC0 "
            "BEC4 BEC8 BECB BECF BED3 BED7 BEDB BEEA BEEE BEF2 "
            "BEF6 BEFA BF79 BF7C BF7F BF91 BF95-BF96 BF9A-BF9B BF9F-BFA0 BFA6-BFA7 "
            "BFAF-BFB2 BFFE C00B C00D C010 C013 C018-C019 C03F C04B C04D "
            "C050 C053 C058-C059 C07F C088 C08C C0A8 C0AC C0C8 C0CC "
            "C0E8 C0EC C104 C106 C114 C116 C124 C126 C134 C136 "
            "C144 C146 C154 C156 C164 C166 C174 C176 C182-C183 C18A-C18B "
            "C192-C193 C19A-C19B C1A2-C1A3 C1AA-C1AB C1B2-C1B3 C1BA-C1BB C1C2-C1C3 C1CA-C1CB C1D2-C1D3 C1DA-C1DB "
            "C1E2-C1E3 C1EA-C1EB C1F2-C1F3 C1FA-C1FB C201 C205 C209 C20D C211 C215 "
            "C219 C21D C221 C225 C229 C22D C231 C235 C239 C23D "
            "C241 C245 C249 C24D C251 C255 C259 C25D C261 C265 "
            "C269 C26D C271 C275 C279 C27D"
        ),
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

    actual = stored(ttnn.to_torch(ttnn.exp2(tt_x)))
    x = x.flatten().to(torch.float64)
    reference = (torch.exp2(x)).to(torch.float64).flatten()
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
    for offset, spans in MODEL.items():
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
