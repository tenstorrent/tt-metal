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
    """bfloat8_b in and out: PCC 0.999 against torch, and each output within one bfloat8_b step of the BF16 call's
    packed to bfloat8_b (two in a block holding an infinity or NaN)."""
    torch.manual_seed(0)
    x = torch.empty(1, 1, 64, 64).uniform_(0.125, 10.0).to(torch.bfloat16)
    tt_x = ttnn.from_torch(x, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device)
    stored = ttnn.to_torch(tt_x).to(torch.float64).flatten()
    output = ttnn.polygamma(tt_x, 1)
    assert output.dtype == ttnn.bfloat8_b
    actual = ttnn.to_torch(output).to(torch.float64).flatten()
    golden = ttnn.get_golden_function(ttnn.polygamma)(stored, 1).to(torch.float64)
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
    actual = ttnn.to_torch(ttnn.polygamma(tt_x, 1)).to(torch.float32).flatten()
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

    SKIP lists inputs the LLK sweep judges instead. EXACT: poles where torch's float64 result is finite.
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
    reference = (ttnn.get_golden_function(ttnn.polygamma)(x, 1)).to(torch.float64).flatten()
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
    for word, ranges in EXACT.items():
        value = stored(torch.tensor([word], dtype=torch.int32).to(torch.int16).view(torch.bfloat16))
        for first, last in ranges:
            lanes = (bits >= first) & (bits <= last)
            ok = torch.where(lanes, actual == value, ok)
    assert ok.all(), (
        f"{(~ok).sum().item()} of {ok.numel()} inputs off torch and stock; first input 0x{bits[~ok][0].item():04x} "
        f"stored {actual[~ok][0].item()!r}, torch {reference[~ok][0].item()!r}"
    )
