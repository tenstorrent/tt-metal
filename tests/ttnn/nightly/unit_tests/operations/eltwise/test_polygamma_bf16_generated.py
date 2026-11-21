# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Generated BF16 polygamma cases, beside the unary category suites."""

import torch
import ttnn


def test_polygamma_bf16_unflushed_zero_and_subnormal_inputs(device):
    """-0 and subnormal inputs (kept by TT-NN's copy into DEST) store torch's result within 1 ULP."""
    words = torch.cat([torch.arange(0x80), torch.arange(0x8000, 0x8080)]).to(torch.int16)
    x = words.view(torch.bfloat16).repeat(4).reshape(1, 1, 32, 32)
    tt_x = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tiny = torch.finfo(torch.bfloat16).tiny

    def stored(t):
        t = t.to(torch.bfloat16).to(torch.float64).flatten()
        t = torch.where(torch.isnan(t), torch.full_like(t, float("inf")), t)
        return torch.where(t.abs() < tiny, torch.zeros_like(t), t)

    actual = stored(ttnn.to_torch(ttnn.polygamma(tt_x, 1)))
    x64 = x.flatten().to(torch.float64)
    reference = ttnn.get_golden_function(ttnn.polygamma)(x64, 1).to(torch.float64).flatten()
    rounded = stored(reference)
    spacing = 2.0 ** (torch.floor(torch.log2(reference.abs().clamp(min=tiny))) - 7)
    close = torch.isfinite(reference) & (actual != 0) & ((actual - reference).abs() < spacing)
    ok = (actual == rounded) | close
    bits = words.repeat(4).to(torch.int32) & 0xFFFF
    assert ok.all(), (
        f"{(~ok).sum().item()} of {ok.numel()} inputs off torch and stock; first input 0x{bits[~ok][0].item():04x} "
        f"stored {actual[~ok][0].item()!r}, torch {reference[~ok][0].item()!r}"
    )


def test_polygamma_bf16_bfloat8_b(device):
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
    output = ttnn.polygamma(tt_x, 1)
    assert output.dtype == ttnn.bfloat8_b
    actual = ttnn.to_torch(output).to(torch.float32).flatten()
    expected = ttnn.to_torch(ttnn.typecast(ttnn.polygamma(tt_stored, 1), ttnn.bfloat8_b)).to(torch.float32).flatten()
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


def test_polygamma_bf16_nonfinite_inputs(device):
    """+-Inf and +-NaN store torch's result (NaN as an infinity) or STOCK's {input bits: word} for the board."""
    board = "blackhole" if ttnn.device.is_blackhole(device) else "wormhole_b0"
    STOCK = {"blackhole": {0x7FC0: 0x0000, 0xFFC0: 0x0000}}
    bits = torch.tensor([0x7F80, 0xFF80, 0x7FC0, 0xFFC0], dtype=torch.int32).repeat(256)
    x = bits.to(torch.int16).view(torch.bfloat16).reshape(1, 1, 32, 32)
    tt_x = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    def stored(t):
        return t.to(torch.bfloat16).to(torch.float64).flatten()

    actual = stored(ttnn.to_torch(ttnn.polygamma(tt_x, 1)))
    reference = ttnn.get_golden_function(ttnn.polygamma)(x.flatten().to(torch.float64), 1).to(torch.float64)
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


def test_polygamma_bf16_exhaustive_inputs(device):
    """All 65,536 BF16 inputs: torch's result as stored, within STEPS BF16 steps, or STOCK's word per board.

    SKIP lists inputs the LLK sweep judges instead. MODEL: per board, the kernel's offset in BF16 steps from torch's word. EXACT: poles where torch's float64 result is finite.
    """
    board = "blackhole" if ttnn.device.is_blackhole(device) else "wormhole_b0"
    STOCK = {"blackhole": {0x0000: [(0x7F81, 0x7FFF), (0xFF81, 0xFFFF)]}}
    STEPS = 1
    SKIP = [(0x7E81, 0x7F7F)]
    EXACT = {
        0x7F80: [
            (0xBF80, 0xBF80),
            (0xC000, 0xC000),
            (0xC040, 0xC040),
            (0xC080, 0xC080),
            (0xC0A0, 0xC0A0),
            (0xC0C0, 0xC0C0),
            (0xC0E0, 0xC0E0),
            (0xC100, 0xC100),
            (0xC110, 0xC110),
            (0xC120, 0xC120),
            (0xC130, 0xC130),
            (0xC140, 0xC140),
            (0xC150, 0xC150),
            (0xC160, 0xC160),
            (0xC170, 0xC170),
            (0xC180, 0xC180),
            (0xC188, 0xC188),
            (0xC190, 0xC190),
            (0xC198, 0xC198),
            (0xC1A0, 0xC1A0),
            (0xC1A8, 0xC1A8),
            (0xC1B0, 0xC1B0),
            (0xC1B8, 0xC1B8),
            (0xC1C0, 0xC1C0),
            (0xC1C8, 0xC1C8),
            (0xC1D0, 0xC1D0),
            (0xC1D8, 0xC1D8),
            (0xC1E0, 0xC1E0),
            (0xC1E8, 0xC1E8),
            (0xC1F0, 0xC1F0),
            (0xC1F8, 0xC1F8),
            (0xC200, 0xC200),
            (0xC204, 0xC204),
            (0xC208, 0xC208),
            (0xC20C, 0xC20C),
            (0xC210, 0xC210),
            (0xC214, 0xC214),
            (0xC218, 0xC218),
            (0xC21C, 0xC21C),
            (0xC220, 0xC220),
            (0xC224, 0xC224),
            (0xC228, 0xC228),
            (0xC22C, 0xC22C),
            (0xC230, 0xC230),
            (0xC234, 0xC234),
            (0xC238, 0xC238),
            (0xC23C, 0xC23C),
            (0xC240, 0xC240),
            (0xC244, 0xC244),
            (0xC248, 0xC248),
            (0xC24C, 0xC24C),
            (0xC250, 0xC250),
            (0xC254, 0xC254),
            (0xC258, 0xC258),
            (0xC25C, 0xC25C),
            (0xC260, 0xC260),
            (0xC264, 0xC264),
            (0xC268, 0xC268),
            (0xC26C, 0xC26C),
            (0xC270, 0xC270),
            (0xC274, 0xC274),
            (0xC278, 0xC278),
            (0xC27C, 0xC27C),
            (0xC280, 0xC280),
            (0xC282, 0xC282),
            (0xC284, 0xC284),
            (0xC286, 0xC286),
            (0xC288, 0xC288),
            (0xC28A, 0xC28A),
            (0xC28C, 0xC28C),
            (0xC28E, 0xC28E),
            (0xC290, 0xC290),
            (0xC292, 0xC292),
            (0xC294, 0xC294),
            (0xC296, 0xC296),
            (0xC298, 0xC298),
            (0xC29A, 0xC29A),
            (0xC29C, 0xC29C),
            (0xC29E, 0xC29E),
            (0xC2A0, 0xC2A0),
            (0xC2A2, 0xC2A2),
            (0xC2A4, 0xC2A4),
            (0xC2A6, 0xC2A6),
            (0xC2A8, 0xC2A8),
            (0xC2AA, 0xC2AA),
            (0xC2AC, 0xC2AC),
            (0xC2AE, 0xC2AE),
            (0xC2B0, 0xC2B0),
            (0xC2B2, 0xC2B2),
            (0xC2B4, 0xC2B4),
            (0xC2B6, 0xC2B6),
            (0xC2B8, 0xC2B8),
            (0xC2BA, 0xC2BA),
            (0xC2BC, 0xC2BC),
            (0xC2BE, 0xC2BE),
            (0xC2C0, 0xC2C0),
            (0xC2C2, 0xC2C2),
            (0xC2C4, 0xC2C4),
            (0xC2C6, 0xC2C6),
            (0xC2C8, 0xC2C8),
            (0xC2CA, 0xC2CA),
            (0xC2CC, 0xC2CC),
            (0xC2CE, 0xC2CE),
            (0xC2D0, 0xC2D0),
            (0xC2D2, 0xC2D2),
            (0xC2D4, 0xC2D4),
            (0xC2D6, 0xC2D6),
            (0xC2D8, 0xC2D8),
            (0xC2DA, 0xC2DA),
            (0xC2DC, 0xC2DC),
            (0xC2DE, 0xC2DE),
            (0xC2E0, 0xC2E0),
            (0xC2E2, 0xC2E2),
            (0xC2E4, 0xC2E4),
            (0xC2E6, 0xC2E6),
            (0xC2E8, 0xC2E8),
            (0xC2EA, 0xC2EA),
            (0xC2EC, 0xC2EC),
            (0xC2EE, 0xC2EE),
            (0xC2F0, 0xC2F0),
            (0xC2F2, 0xC2F2),
            (0xC2F4, 0xC2F4),
            (0xC2F6, 0xC2F6),
            (0xC2F8, 0xC2F8),
            (0xC2FA, 0xC2FA),
            (0xC2FC, 0xC2FC),
            (0xC2FE, 0xC2FE),
            (0xC300, 0xFF7F),
        ]
    }
    MODEL = {
        "blackhole": {
            -1: (
                "3B77 3D05 3D0F 3FAF 3FB6 3FD6 4001 4019 4023 402F "
                "4031 403B 405A 4116 4300 BB77 BDD4 BDE6 BE10 BE19 "
                "BE40 BE51 BE55 BE58 BE64 BE67 BE6A BE6F BE75-BE76 BE82 "
                "BE86 BE90 BE92 BE94 BE97-BE98 BEA0-BEA1 BEA7 BEAB BEB5 BF04 "
                "BF21 BF2A-BF2C BF30 BF32-BF33 BF35-BF39 BF3C BF4C BF5F BF99 BF9D-BF9F "
                "BFA4 BFA7 BFA9 BFAC BFC2 BFDA BFDC BFE1-BFE3 C00D C020 "
                "C02B C031 C053 C06C C090 C097 C0A8 C0AA C0B8 C0BA "
                "C0C6 C0D0 C105 C10B C138 C148 C158 C19A C19E C1A2 "
                "C1A6 C1AA C1AE C1B2 C1B6 C1BA C1BE C1C2 C1C6 C1CA "
                "C1CE C1D2 C1D6 C1DA C1DE C1E2 C1E6 C1EA C1EE C1F2 "
                "C1FB C1FD C21A C21E C222 C226 C22A C22E C232 C236 "
                "C23A C23E C242 C246 C24A C24E C252 C256 C25A C25E "
                "C262 C266 C26A C26E C272 C276 C27A C27E C281 C283 "
                "C285 C287 C289 C28B C28D C28F C291 C293 C295 C297 "
                "C299 C29B C29D C29F C2A1 C2A3 C2A5 C2A7 C2A9 C2AB "
                "C2AD C2AF C2B1 C2B3 C2B5 C2B7 C2B9 C2BB C2BD C2BF "
                "C2C1 C2C3 C2C5 C2C7 C2C9 C2CB C2CD C2CF C2D1 C2D3 "
                "C2D5 C2D7 C2D9 C2DB C2DD C2DF C2E1 C2E3 C2E5 C2E7 "
                "C2E9 C2EB C2ED C2EF C2F1 C2F3 C2F5 C2F7 C2F9 C2FB "
                "C2FD C2FF"
            ),
            1: (
                "3DC3 3DD4 3E39 3E41 3E53 3E57 3E86 3E93 3ECB 3F08 "
                "BEDB BEEE BEF4 BF07 BF0B BF0E BF1A BFBA BFBC BFCB "
                "C01B C05D-C05E C062-C063 C0AE C0ED C0EF C0F1 C0F3 C107 C109"
            ),
        },
        "wormhole_b0": {
            -1: (
                "1FDA 1FE9 205A 2069 20DA 20E9 215A 2169 21DA 21E9 "
                "225A 2269 22DA 22E9 235A 2369 23DA 23E9 245A 2469 "
                "24DA 24E9 255A 2569 25DA 25E9 265A 2669 26DA 26E9 "
                "275A 2769 27DA 27E9 285A 2869 28DA 28E9 295A 2969 "
                "29DA 29E9 2A5A 2A69 2ADA 2AE9 2B5A 2B69 2BDA 2BE9 "
                "2C5A 2C69 2CDA 2CE9 2D5A 2D69 2DDA 2DE9 2E5A 2E69 "
                "2EDA 2EE9 2F5A 2F69 2FDA 2FE9 305A 3069 30DA 30E9 "
                "315A 3169 31DA 31E9 325A 3269 32DA 32E9 335A 3369 "
                "33DA 33E9 345A 3469 34DA 34E9 355A 3569 35DA 35E9 "
                "365A 3669 36DA 36E9 375A 3769 37DA 37E9 385A 3869 "
                "38DA 38E9 395A 3969 39DA 39E9 3A5A 3A69 3ADA 3AE9 "
                "3B5A 3B69 3BB7 3BDA 3C14 3C57 3C83 3C96 3CB0 3CDC "
                "3D0F 3D2A 3D5D 3DDF 3E5A 3F1D 3F68 3F80 3FAF 3FB6 "
                "3FD6 3FDA 4001 4019 4023 402F 4031 403B 405A 40DD "
                "4116 4223 422E 42AD 4300 4381 43A7 43E3 448D 44E6 "
                "45FF 467F 46FF 477F 47AC 47FF 482C 487F 48AC 48FF "
                "492C 497F 49AC 49FF 4A2C 4A7F 4AAC 4AFF 4B2C 4B7F "
                "4BAC 4BFF 4C2C 4C7F 4CAC 4CFF 4D2C 4D7F 4DAC 4DFF "
                "4E2C 4E7F 4EAC 4EFF 4F2C 4F7F 4FAC 4FFF 502C 507F "
                "50AC 50FF 512C 517F 51AC 51FF 522C 527F 52AC 52FF "
                "532C 537F 53AC 53FF 542C 547F 54AC 54FF 552C 557F "
                "55AC 55FF 562C 567F 56AC 56FF 572C 577F 57AC 57FF "
                "582C 587F 58AC 58FF 592C 597F 59AC 59FF 5A2C 5A7F "
                "5AAC 5AFF 5B2C 5B7F 5BAC 5BFF 5C2C 5C7F 5CAC 5CFF "
                "5D2C 5D7F 5DAC 5DFF 5E2C 5E7F 5EAC 5EFF 5F2C 5F7F "
                "5FAC 5FFF 602C 607F 60AC 60FF 612C 617F 61AC 61FF "
                "622C 627F 62AC 62FF 632C 637F 63AC 63FF 642C 647F "
                "64AC 64FF 652C 657F 65AC 65FF 662C 667F 66AC 66FF "
                "672C 677F 67AC 67FF 682C 687F 68AC 68FF 692C 697F "
                "69AC 69FF 6A2C 6A7F 6AAC 6AFF 6B2C 6B7F 6BAC 6BFF "
                "6C2C 6C7F 6CAC 6CFF 6D2C 6D7F 6DAC 6DFF 6E2C 6E7F "
                "6EAC 6EFF 6F2C 6F7F 6FAC 6FFF 702C 707F 70AC 70FF "
                "712C 717F 71AC 71FF 722C 727F 72AC 72FF 732C 737F "
                "73AC 73FF 742C 747F 74AC 74FF 752C 757F 75AC 75FF "
                "762C 767F 76AC 76FF 772C 777F 77AC 77FF 782C 787F "
                "78AC 78FF 792C 797F 79AC 79FF 7A2C 7A7F 7AAC 7AFF "
                "7B2C 7B7F 7BAC 7BFF 7C2C 7C7F 7CAC 7CFF 7D2C 7D7F "
                "7DAC 7DFF 7E2C 7E7F 9FDA 9FE9 A05A A069 A0DA A0E9 "
                "A15A A169 A1DA A1E9 A25A A269 A2DA A2E9 A35A A369 "
                "A3DA A3E9 A45A A469 A4DA A4E9 A55A A569 A5DA A5E9 "
                "A65A A669 A6DA A6E9 A75A A769 A7DA A7E9 A85A A869 "
                "A8DA A8E9 A95A A969 A9DA A9E9 AA5A AA69 AADA AAE9 "
                "AB5A AB69 ABDA ABE9 AC5A AC69 ACDA ACE9 AD5A AD69 "
                "ADDA ADE9 AE5A AE69 AEDA AEE9 AF5A AF69 AFDA AFE9 "
                "B05A B069 B0DA B0E9 B15A B169 B1DA B1E9 B25A B269 "
                "B2DA B2E9 B35A B369 B3DA B3E9 B45A B469 B4DA B4E9 "
                "B55A B569 B5DA B5E9 B65A B669 B6DA B6E9 B75A B769 "
                "B7DA B7E9 B85A B869 B8DA B8E9 B95A B969 B9DA B9E9 "
                "BA5A BA69 BADA BAE9 BB5A BB69 BBB7 BBDA BC14 BC57 "
                "BC83 BC96 BCB0 BCDC BD6C BDB6 BDCE BDD4 BDE6 BE10 "
                "BE19 BE40 BE51 BE55 BE58 BE64 BE67 BE6A BE6F BE75-BE76 "
                "BE82 BE86 BE90 BE92 BE94 BE97-BE98 BEA0-BEA1 BEA7 BEAB BEB5 "
                "BF04 BF21 BF29-BF2C BF30 BF32-BF33 BF35-BF39 BF3C BF4A BF4C BF5F "
                "BF64 BF72 BF79 BF85 BF87 BF99 BF9C-BF9F BFA4 BFA7 BFA9 "
                "BFAC BFC2 BFD3 BFDA BFDC BFE1-BFE3 C00D C020 C02B C031 "
                "C03C C044 C053 C06C C090 C097 C0A8 C0AA C0B8 C0BA "
                "C0D0 C0DC C0E4 C105 C10B C138 C148 C158 C19A C19E "
                "C1A2 C1A6 C1AA C1AE C1B2 C1B6 C1BA C1BE C1C2 C1C6 "
                "C1CA C1CE C1D2 C1D6 C1DA C1DE C1E2 C1E6 C1EA C1EE "
                "C1F2 C1F6 C1FA-C1FB C1FE C201 C203 C205 C21A C21E C222 "
                "C226 C22A C22E C232 C236 C23A C23E C242 C246 C24A "
                "C24E C252 C256 C25A C25E C262 C266 C26A C26E C272 "
                "C276 C27A C27E C281 C283 C285 C287 C289 C28B C28D "
                "C28F C291 C293 C295 C297 C299 C29B C29D C29F C2A1 "
                "C2A3 C2A5 C2A7 C2A9 C2AB C2AD C2AF C2B1 C2B3 C2B5 "
                "C2B7 C2B9 C2BB C2BD C2BF C2C1 C2C3 C2C5 C2C7 C2C9 "
                "C2CB C2CD C2CF C2D1 C2D3 C2D5 C2D7 C2D9 C2DB C2DD "
                "C2DF C2E1 C2E3 C2E5 C2E7 C2E9 C2EB C2ED C2EF C2F1 "
                "C2F3 C2F5 C2F7 C2F9 C2FB C2FD C2FF"
            ),
            1: (
                "3DC3 3E39 3E41 3E86 3E93 3ECB 3F08 BEEE BEF4 BF07 "
                "BF0B BF0E BF1A BFBA BFBC BFCB C01B C05D-C05E C062-C063 C0AE "
                "C0ED C0EF C0F1 C0F3 C107 C109"
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

    actual = stored(ttnn.to_torch(ttnn.polygamma(tt_x, 1)))
    x = x.flatten().to(torch.float64)
    reference = (torch.special.polygamma(1, x)).to(torch.float64).flatten()
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
    for word, ranges in EXACT.items():
        value = stored(torch.tensor([word], dtype=torch.int32).to(torch.int16).view(torch.bfloat16))
        for first, last in ranges:
            lanes = (bits >= first) & (bits <= last)
            ok = torch.where(lanes, actual == value, ok)
    assert ok.all(), (
        f"{(~ok).sum().item()} of {ok.numel()} inputs off torch and stock; first input 0x{bits[~ok][0].item():04x} "
        f"stored {actual[~ok][0].item()!r}, torch {reference[~ok][0].item()!r}"
    )
