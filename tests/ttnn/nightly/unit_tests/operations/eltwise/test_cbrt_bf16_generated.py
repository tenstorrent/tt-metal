# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Generated BF16 cbrt cases, beside the unary category suites."""

import torch
import pytest
import ttnn
from tests.ttnn.unit_tests.operations.eltwise.eltwise_test_utils import (
    generate_bfloat16_bits,
    to_tt_tensor,
    SMALLEST_NORMAL_BF16,
)


@pytest.mark.parametrize("generated_first", [True, False])
def test_cbrt_with_log_in_sfpu_chain(device, generated_first):
    """cbrt chained with stock log, both orders: each step stores what its op stores standalone."""
    generated = (ttnn.UnaryWithParam(ttnn.UnaryOpType.CBRT), lambda tensor: ttnn.cbrt(tensor))
    stock = (ttnn.UnaryWithParam(ttnn.UnaryOpType.LOG), lambda tensor: ttnn.log(tensor))
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


def test_cbrt_bf16_unflushed_zero_and_subnormal_inputs(device):
    """-0 and subnormal inputs (kept by TT-NN's copy into DEST) store +0."""
    board = "blackhole" if ttnn.device.is_blackhole(device) else "wormhole_b0"
    if board not in ("blackhole",):
        pytest.skip(f"ttnn.cbrt keeps its stock kernel on {board}")
    words = torch.cat([torch.arange(0x80), torch.arange(0x8000, 0x8080)]).to(torch.int16)
    x = words.view(torch.bfloat16).repeat(4).reshape(1, 1, 32, 32)
    tt_x = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    raw = ttnn.to_torch(ttnn.cbrt(tt_x)).reshape(-1).view(torch.int16).to(torch.int32) & 0xFFFF
    assert torch.all(raw == 0x0000)


def test_cbrt_bf16_bfloat8_b(device):
    """bfloat8_b in and out: PCC 0.999 against torch, and each output within one bfloat8_b step of the BF16 call's
    packed to bfloat8_b (two in a block holding an infinity or NaN)."""
    board = "blackhole" if ttnn.device.is_blackhole(device) else "wormhole_b0"
    if board not in ("blackhole",):
        pytest.skip(f"ttnn.cbrt keeps its stock kernel on {board}")
    torch.manual_seed(0)
    x = torch.empty(1, 1, 64, 64).uniform_(-100.0, 100.0).to(torch.bfloat16)
    tt_x = ttnn.from_torch(x, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device)
    stored = ttnn.to_torch(tt_x).to(torch.float64).flatten()
    output = ttnn.cbrt(tt_x)
    assert output.dtype == ttnn.bfloat8_b
    actual = ttnn.to_torch(output).to(torch.float64).flatten()
    golden = ttnn.get_golden_function(ttnn.cbrt)(stored).to(torch.float64)
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
    actual = ttnn.to_torch(ttnn.cbrt(tt_x)).to(torch.float32).flatten()
    expected = ttnn.to_torch(ttnn.typecast(ttnn.cbrt(tt_stored), ttnn.bfloat8_b)).to(torch.float32).flatten()
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


def test_cbrt_bf16_exhaustive_inputs(device):
    """All 65,536 BF16 inputs: torch's result as stored, within STEPS BF16 steps.

    SKIP lists inputs the LLK sweep judges instead. MODEL: per board, the kernel's offset in BF16 steps from torch's word.
    """
    board = "blackhole" if ttnn.device.is_blackhole(device) else "wormhole_b0"
    if board not in ("blackhole",):
        pytest.skip(f"ttnn.cbrt keeps its stock kernel on {board}")
    STEPS = 1
    SKIP = [(0x0001, 0x007F), (0x8001, 0x807F)]
    MODEL = {
        -1: (
            "0112 0153 019D 0292 02D3 031D 0412 0453 049D 0592 "
            "05D3 061D 0712 0753 079D 0892 08D3 091D 0A12 0A53 "
            "0A9D 0B92 0BD3 0C1D 0D12 0D53 0D9D 0E92 0ED3 0F1D "
            "1012 1053 109D 1192 11D3 121D 1312 1353 139D 1492 "
            "14D3 151D 1612 1653 169D 1792 17D3 181D 1912 1953 "
            "199D 1A92 1AD3 1B1D 1C12 1C53 1C9D 1D92 1DD3 1E1D "
            "1F12 1F53 1F9D 2092 20D3 211D 2212 2253 229D 2392 "
            "23D3 241D 2512 2553 259D 2692 26D3 271D 2812 2853 "
            "289D 2992 29D3 2A1D 2B12 2B53 2B9D 2C92 2CD3 2D1D "
            "2E12 2E53 2E9D 2F92 2FD3 301D 3112 3153 319D 3292 "
            "32D3 331D 3412 3453 349D 3592 35D3 361D 3712 3753 "
            "379D 3892 38D3 391D 3A12 3A53 3A9D 3B92 3BD3 3C1D "
            "3D12 3D53 3D9D 3E92 3ED3 3F1D 4012 4053 409D 4192 "
            "41D3 421D 4312 4353 439D 4492 44D3 451D 4612 4653 "
            "469D 4792 47D3 481D 4912 4953 499D 4A92 4AD3 4B1D "
            "4C12 4C53 4C9D 4D92 4DD3 4E1D 4F12 4F53 4F9D 5092 "
            "50D3 511D 5212 5253 529D 5392 53D3 541D 5512 5553 "
            "559D 5692 56D3 571D 5812 5853 589D 5992 59D3 5A1D "
            "5B12 5B53 5B9D 5C92 5CD3 5D1D 5E12 5E53 5E9D 5F92 "
            "5FD3 601D 6112 6153 619D 6292 62D3 631D 6412 6453 "
            "649D 6592 65D3 661D 6712 6753 679D 6892 68D3 691D "
            "6A12 6A53 6A9D 6B92 6BD3 6C1D 6D12 6D53 6D9D 6E92 "
            "6ED3 6F1D 7012 7053 709D 7192 71D3 721D 7312 7353 "
            "739D 7492 74D3 751D 7612 7653 769D 7792 77D3 781D "
            "7912 7953 799D 7A92 7AD3 7B1D 7C12 7C53 7C9D 7D92 "
            "7DD3 7E1D 7F12 7F53 80BC 81CF 823C 834F 83BC 84CF "
            "853C 864F 86BC 87CF 883C 894F 89BC 8ACF 8B3C 8C4F "
            "8CBC 8DCF 8E3C 8F4F 8FBC 90CF 913C 924F 92BC 93CF "
            "943C 954F 95BC 96CF 973C 984F 98BC 99CF 9A3C 9B4F "
            "9BBC 9CCF 9D3C 9E4F 9EBC 9FCF A03C A14F A1BC A2CF "
            "A33C A44F A4BC A5CF A63C A74F A7BC A8CF A93C AA4F "
            "AABC ABCF AC3C AD4F ADBC AECF AF3C B04F B0BC B1CF "
            "B23C B34F B3BC B4CF B53C B64F B6BC B7CF B83C B94F "
            "B9BC BACF BB3C BC4F BCBC BDCF BE3C BF4F BFBC C0CF "
            "C13C C24F C2BC C3CF C43C C54F C5BC C6CF C73C C84F "
            "C8BC C9CF CA3C CB4F CBBC CCCF CD3C CE4F CEBC CFCF "
            "D03C D14F D1BC D2CF D33C D44F D4BC D5CF D63C D74F "
            "D7BC D8CF D93C DA4F DABC DBCF DC3C DD4F DDBC DECF "
            "DF3C E04F E0BC E1CF E23C E34F E3BC E4CF E53C E64F "
            "E6BC E7CF E83C E94F E9BC EACF EB3C EC4F ECBC EDCF "
            "EE3C EF4F EFBC F0CF F13C F24F F2BC F3CF F43C F54F "
            "F5BC F6CF F73C F84F F8BC F9CF FA3C FB4F FBBC FCCF "
            "FD3C FE4F FEBC"
        ),
        1: (
            "00BC 01CF 023C 034F 03BC 04CF 053C 064F 06BC 07CF "
            "083C 094F 09BC 0ACF 0B3C 0C4F 0CBC 0DCF 0E3C 0F4F "
            "0FBC 10CF 113C 124F 12BC 13CF 143C 154F 15BC 16CF "
            "173C 184F 18BC 19CF 1A3C 1B4F 1BBC 1CCF 1D3C 1E4F "
            "1EBC 1FCF 203C 214F 21BC 22CF 233C 244F 24BC 25CF "
            "263C 274F 27BC 28CF 293C 2A4F 2ABC 2BCF 2C3C 2D4F "
            "2DBC 2ECF 2F3C 304F 30BC 31CF 323C 334F 33BC 34CF "
            "353C 364F 36BC 37CF 383C 394F 39BC 3ACF 3B3C 3C4F "
            "3CBC 3DCF 3E3C 3F4F 3FBC 40CF 413C 424F 42BC 43CF "
            "443C 454F 45BC 46CF 473C 484F 48BC 49CF 4A3C 4B4F "
            "4BBC 4CCF 4D3C 4E4F 4EBC 4FCF 503C 514F 51BC 52CF "
            "533C 544F 54BC 55CF 563C 574F 57BC 58CF 593C 5A4F "
            "5ABC 5BCF 5C3C 5D4F 5DBC 5ECF 5F3C 604F 60BC 61CF "
            "623C 634F 63BC 64CF 653C 664F 66BC 67CF 683C 694F "
            "69BC 6ACF 6B3C 6C4F 6CBC 6DCF 6E3C 6F4F 6FBC 70CF "
            "713C 724F 72BC 73CF 743C 754F 75BC 76CF 773C 784F "
            "78BC 79CF 7A3C 7B4F 7BBC 7CCF 7D3C 7E4F 7EBC 8112 "
            "8153 819D 8292 82D3 831D 8412 8453 849D 8592 85D3 "
            "861D 8712 8753 879D 8892 88D3 891D 8A12 8A53 8A9D "
            "8B92 8BD3 8C1D 8D12 8D53 8D9D 8E92 8ED3 8F1D 9012 "
            "9053 909D 9192 91D3 921D 9312 9353 939D 9492 94D3 "
            "951D 9612 9653 969D 9792 97D3 981D 9912 9953 999D "
            "9A92 9AD3 9B1D 9C12 9C53 9C9D 9D92 9DD3 9E1D 9F12 "
            "9F53 9F9D A092 A0D3 A11D A212 A253 A29D A392 A3D3 "
            "A41D A512 A553 A59D A692 A6D3 A71D A812 A853 A89D "
            "A992 A9D3 AA1D AB12 AB53 AB9D AC92 ACD3 AD1D AE12 "
            "AE53 AE9D AF92 AFD3 B01D B112 B153 B19D B292 B2D3 "
            "B31D B412 B453 B49D B592 B5D3 B61D B712 B753 B79D "
            "B892 B8D3 B91D BA12 BA53 BA9D BB92 BBD3 BC1D BD12 "
            "BD53 BD9D BE92 BED3 BF1D C012 C053 C09D C192 C1D3 "
            "C21D C312 C353 C39D C492 C4D3 C51D C612 C653 C69D "
            "C792 C7D3 C81D C912 C953 C99D CA92 CAD3 CB1D CC12 "
            "CC53 CC9D CD92 CDD3 CE1D CF12 CF53 CF9D D092 D0D3 "
            "D11D D212 D253 D29D D392 D3D3 D41D D512 D553 D59D "
            "D692 D6D3 D71D D812 D853 D89D D992 D9D3 DA1D DB12 "
            "DB53 DB9D DC92 DCD3 DD1D DE12 DE53 DE9D DF92 DFD3 "
            "E01D E112 E153 E19D E292 E2D3 E31D E412 E453 E49D "
            "E592 E5D3 E61D E712 E753 E79D E892 E8D3 E91D EA12 "
            "EA53 EA9D EB92 EBD3 EC1D ED12 ED53 ED9D EE92 EED3 "
            "EF1D F012 F053 F09D F192 F1D3 F21D F312 F353 F39D "
            "F492 F4D3 F51D F612 F653 F69D F792 F7D3 F81D F912 "
            "F953 F99D FA92 FAD3 FB1D FC12 FC53 FC9D FD92 FDD3 "
            "FE1D FF12 FF53"
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

    actual = stored(ttnn.to_torch(ttnn.cbrt(tt_x)))
    x = x.flatten().to(torch.float64)
    reference = (torch.sign(x) * torch.abs(x) ** (1.0 / 3.0)).to(torch.float64).flatten()
    expected = stored(reference)
    finite = torch.isfinite(expected) & torch.isfinite(actual)
    ok = (actual == expected) | (finite & ((ordered(actual) - ordered(expected)).abs() <= STEPS))
    # A NaN result reaches L1 as a NaN or as an infinity of either sign.
    ok |= torch.isnan(reference) & (torch.isnan(actual) | torch.isinf(actual))
    # The SFPU flushes a result below the smallest normal, or within 2^-12 above it, to 0.
    ok |= (actual == 0) & (reference.abs() < tiny * (1 + 2.0**-12))
    listed = torch.zeros_like(ok)
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
