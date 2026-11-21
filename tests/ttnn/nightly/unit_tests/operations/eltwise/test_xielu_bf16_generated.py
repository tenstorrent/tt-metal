# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Generated BF16 xielu cases, beside the unary category suites."""

import torch
import ttnn


def test_xielu_bf16_unflushed_zero_and_subnormal_inputs(device):
    """-0 and subnormal inputs (kept by TT-NN's copy into DEST) store torch's result within 1 ULP."""
    words = torch.cat([torch.arange(0x80), torch.arange(0x8000, 0x8080)]).to(torch.int16)
    x = words.view(torch.bfloat16).repeat(4).reshape(1, 1, 32, 32)
    tt_x = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tiny = torch.finfo(torch.bfloat16).tiny

    def stored(t):
        t = t.to(torch.bfloat16).to(torch.float64).flatten()
        t = torch.where(torch.isnan(t), torch.full_like(t, float("inf")), t)
        return torch.where(t.abs() < tiny, torch.zeros_like(t), t)

    actual = stored(ttnn.to_torch(ttnn.xielu(tt_x)))
    x64 = x.flatten().to(torch.float64)
    reference = ttnn.get_golden_function(ttnn.xielu)(x64).to(torch.float64).flatten()
    rounded = stored(reference)
    spacing = 2.0 ** (torch.floor(torch.log2(reference.abs().clamp(min=tiny))) - 7)
    close = torch.isfinite(reference) & (actual != 0) & ((actual - reference).abs() < spacing)
    ok = (actual == rounded) | close
    bits = words.repeat(4).to(torch.int32) & 0xFFFF
    assert ok.all(), (
        f"{(~ok).sum().item()} of {ok.numel()} inputs off torch and stock; first input 0x{bits[~ok][0].item():04x} "
        f"stored {actual[~ok][0].item()!r}, torch {reference[~ok][0].item()!r}"
    )


def test_xielu_bf16_bfloat8_b(device):
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
    output = ttnn.xielu(tt_x)
    assert output.dtype == ttnn.bfloat8_b
    actual = ttnn.to_torch(output).to(torch.float32).flatten()
    expected = ttnn.to_torch(ttnn.typecast(ttnn.xielu(tt_stored), ttnn.bfloat8_b)).to(torch.float32).flatten()
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


def test_xielu_bf16_exhaustive_inputs(device):
    """All 65,536 BF16 inputs: torch's result as stored, within STEPS BF16 steps.

    SKIP lists inputs the LLK sweep judges instead.
    """
    STEPS = 0
    SKIP = [
        (0x0003, 0x00FF),
        (0x3C20, 0x3C20),
        (0x3DF0, 0x3DF0),
        (0x3EC8, 0x3EC8),
        (0x3F8C, 0x3F8C),
        (0x4166, 0x4166),
        (0x4220, 0x4220),
        (0x4C34, 0x4C34),
        (0x4CB4, 0x4CB4),
        (0x4D34, 0x4D34),
        (0x4DB4, 0x4DB4),
        (0x4E34, 0x4E34),
        (0x4EB4, 0x4EB4),
        (0x4F34, 0x4F34),
        (0x4FB4, 0x4FB4),
        (0x5034, 0x5034),
        (0x50B4, 0x50B4),
        (0x5134, 0x5134),
        (0x51B4, 0x51B4),
        (0x5234, 0x5234),
        (0x52B4, 0x52B4),
        (0x5334, 0x5334),
        (0x53B4, 0x53B4),
        (0x5434, 0x5434),
        (0x54B4, 0x54B4),
        (0x5534, 0x5534),
        (0x55B4, 0x55B4),
        (0x5634, 0x5634),
        (0x56B4, 0x56B4),
        (0x5734, 0x5734),
        (0x57B4, 0x57B4),
        (0x5834, 0x5834),
        (0x58B4, 0x58B4),
        (0x5934, 0x5934),
        (0x59B4, 0x59B4),
        (0x5A34, 0x5A34),
        (0x5AB4, 0x5AB4),
        (0x5B34, 0x5B34),
        (0x5BB4, 0x5BB4),
        (0x5C34, 0x5C34),
        (0x5CB4, 0x5CB4),
        (0x5D34, 0x5D34),
        (0x5DB4, 0x5DB4),
        (0x5E34, 0x5E34),
        (0x5EB4, 0x5EB4),
        (0x5F34, 0x5F34),
        (0xC55D, 0xC55D),
        (0xC567, 0xC567),
        (0xC571, 0xC571),
        (0xC57B, 0xC57B),
    ]
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

    actual = stored(ttnn.to_torch(ttnn.xielu(tt_x)))
    x = x.flatten().to(torch.float64)
    reference = (
        (
            torch.where(
                x <= -9.999999974752427e-07,
                0.800000011920929 * (torch.exp(x) - 1) - 0.30000001192092896 * x,
                torch.where(
                    x < 1.401298464324817e-45,
                    0.800000011920929 * (0.9999990000005026 - 1) - 0.30000001192092896 * x,
                    0.800000011920929 * x**2 + 0.5 * x,
                ),
            )
        )
        .to(torch.float64)
        .flatten()
    )
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
