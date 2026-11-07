# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Generated BF16 rsqrt cases, beside the unary category suites."""

import torch
import ttnn


def test_rsqrt_bf16_unflushed_zero_and_subnormal_inputs(device):
    """-0 and subnormal inputs (kept by TT-NN's copy into DEST) store torch's result within 1 ULP.

    STOCK: per board, {word: [(first, last input bits)]} where the kernel stores stock's word."""
    board = "blackhole" if ttnn.device.is_blackhole(device) else "wormhole_b0"
    STOCK = {
        0x5FB3: [(0x007F, 0x007F)],
        0x5FB4: [(0x007D, 0x007E)],
        0x5FB5: [(0x007B, 0x007C)],
        0x5FB6: [(0x0078, 0x007A)],
        0x5FB7: [(0x0076, 0x0077)],
        0x5FB8: [(0x0074, 0x0075)],
        0x5FB9: [(0x0071, 0x0073)],
        0x5FBA: [(0x006F, 0x0070)],
        0x5FBB: [(0x006D, 0x006E)],
        0x5FBC: [(0x006A, 0x006C)],
        0x5FBD: [(0x0068, 0x0069)],
        0x5FBE: [(0x0066, 0x0067)],
        0x5FBF: [(0x0063, 0x0065)],
        0x5FC0: [(0x0061, 0x0062)],
        0x5FC1: [(0x005E, 0x0060)],
        0x5FC2: [(0x005C, 0x005D)],
        0x5FC3: [(0x005A, 0x005B)],
        0x5FC4: [(0x0057, 0x0059)],
        0x5FC5: [(0x0055, 0x0056)],
        0x5FC6: [(0x0053, 0x0054)],
        0x5FC7: [(0x0050, 0x0052)],
        0x5FC8: [(0x004E, 0x004F)],
        0x5FC9: [(0x004C, 0x004D)],
        0x5FCA: [(0x0049, 0x004B)],
        0x5FCB: [(0x0047, 0x0048)],
        0x5FCC: [(0x0045, 0x0046)],
        0x5FCD: [(0x0042, 0x0044)],
        0x5FCE: [(0x0040, 0x0041)],
        0x5FCF: [(0x003E, 0x003F)],
        0x5FD0: [(0x003B, 0x003D)],
        0x5FD1: [(0x0039, 0x003A)],
        0x5FD2: [(0x0037, 0x0038)],
        0x5FD3: [(0x0034, 0x0036)],
        0x5FD4: [(0x0032, 0x0033)],
        0x5FD5: [(0x0030, 0x0031)],
        0x5FD6: [(0x002D, 0x002F)],
        0x5FD7: [(0x002B, 0x002C)],
        0x5FD8: [(0x0029, 0x002A)],
        0x5FD9: [(0x0026, 0x0028)],
        0x5FDA: [(0x0024, 0x0025)],
        0x5FDB: [(0x0022, 0x0023)],
        0x5FDC: [(0x0021, 0x0021)],
        0x5FDD: [(0x0020, 0x0020)],
        0x5FDE: [(0x001F, 0x001F)],
        0x5FDF: [(0x001E, 0x001E)],
        0x5FE0: [(0x001C, 0x001D)],
        0x5FE1: [(0x001B, 0x001B)],
        0x5FE2: [(0x001A, 0x001A)],
        0x5FE3: [(0x0019, 0x0019)],
        0x5FE4: [(0x0018, 0x0018)],
        0x5FE5: [(0x0017, 0x0017)],
        0x5FE6: [(0x0015, 0x0016)],
        0x5FE7: [(0x0014, 0x0014)],
        0x5FE8: [(0x0013, 0x0013)],
        0x5FE9: [(0x0012, 0x0012)],
        0x5FEA: [(0x0011, 0x0011)],
        0x5FEB: [(0x000F, 0x0010)],
        0x5FEC: [(0x000E, 0x000E)],
        0x5FED: [(0x000D, 0x000D)],
        0x5FEE: [(0x000C, 0x000C)],
        0x5FEF: [(0x000B, 0x000B)],
        0x5FF0: [(0x000A, 0x000A)],
        0x5FF1: [(0x0008, 0x0008)],
        0x5FF2: [(0x0007, 0x0007)],
        0x5FF3: [(0x0006, 0x0006)],
        0x5FF4: [(0x0005, 0x0005)],
        0x5FF5: [(0x0004, 0x0004)],
        0x5FF6: [(0x0003, 0x0003)],
        0x5FF7: [(0x0001, 0x0002)],
    }
    words = torch.cat([torch.arange(0x80), torch.arange(0x8000, 0x8080)]).to(torch.int16)
    x = words.view(torch.bfloat16).repeat(4).reshape(1, 1, 32, 32)
    tt_x = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tiny = torch.finfo(torch.bfloat16).tiny

    def stored(t):
        t = t.to(torch.bfloat16).to(torch.float64).flatten()
        t = torch.where(torch.isnan(t), torch.full_like(t, float("inf")), t)
        return torch.where(t.abs() < tiny, torch.zeros_like(t), t)

    actual = stored(ttnn.to_torch(ttnn.rsqrt(tt_x)))
    x64 = x.flatten().to(torch.float64)
    reference = ttnn.get_golden_function(ttnn.rsqrt)(x64).to(torch.float64).flatten()
    rounded = stored(reference)
    spacing = 2.0 ** (torch.floor(torch.log2(reference.abs().clamp(min=tiny))) - 7)
    close = torch.isfinite(reference) & (actual != 0) & ((actual - reference).abs() < spacing)
    ok = (actual == rounded) | close
    bits = words.repeat(4).to(torch.int32) & 0xFFFF
    for word, ranges in STOCK.items():
        value = stored(torch.tensor([word], dtype=torch.int32).to(torch.int16).view(torch.bfloat16))
        for first, last in ranges:
            ok |= (bits >= first) & (bits <= last) & (actual == value)
    assert ok.all(), (
        f"{(~ok).sum().item()} of {ok.numel()} inputs off torch and stock; first input 0x{bits[~ok][0].item():04x} "
        f"stored {actual[~ok][0].item()!r}, torch {reference[~ok][0].item()!r}"
    )


def test_rsqrt_bf16_bfloat8_b(device):
    """bfloat8_b in and out: PCC 0.999 against torch, and each output within one bfloat8_b step of the BF16 call's
    packed to bfloat8_b (two in a block holding an infinity or NaN)."""
    torch.manual_seed(0)
    x = torch.empty(1, 1, 64, 64).uniform_(1.0, 100.0).to(torch.bfloat16)
    tt_x = ttnn.from_torch(x, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device)
    stored = ttnn.to_torch(tt_x).to(torch.float64).flatten()
    output = ttnn.rsqrt(tt_x)
    assert output.dtype == ttnn.bfloat8_b
    actual = ttnn.to_torch(output).to(torch.float64).flatten()
    golden = ttnn.get_golden_function(ttnn.rsqrt)(stored).to(torch.float64)
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
    actual = ttnn.to_torch(ttnn.rsqrt(tt_x)).to(torch.float32).flatten()
    expected = ttnn.to_torch(ttnn.typecast(ttnn.rsqrt(tt_stored), ttnn.bfloat8_b)).to(torch.float32).flatten()
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


def test_rsqrt_bf16_exhaustive_inputs(device):
    """All 65,536 BF16 inputs: torch's result as stored, within STEPS BF16 steps.

    SKIP lists inputs the LLK sweep judges instead.
    """
    STEPS = 0
    SKIP = [(0x0001, 0x007F), (0x8000, 0x8000)]
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

    actual = stored(ttnn.to_torch(ttnn.rsqrt(tt_x)))
    x = x.flatten().to(torch.float64)
    reference = (torch.rsqrt(x)).to(torch.float64).flatten()
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
    normal = ((bits & 0x7F80) != 0) & ((bits & 0x7F80) != 0x7F80)
    ok = torch.where(normal & ~listed & torch.isfinite(expected), actual == expected, ok)
    assert ok.all(), (
        f"{(~ok).sum().item()} of {ok.numel()} inputs off torch and stock; first input 0x{bits[~ok][0].item():04x} "
        f"stored {actual[~ok][0].item()!r}, torch {reference[~ok][0].item()!r}"
    )
