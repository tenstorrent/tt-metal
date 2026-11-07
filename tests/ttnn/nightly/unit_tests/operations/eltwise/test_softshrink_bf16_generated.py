# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Generated BF16 softshrink cases, beside the unary category suites."""

import torch
import pytest
import ttnn


def test_softshrink_bf16_unflushed_zero_and_subnormal_inputs(device):
    """-0 and subnormal inputs (kept by TT-NN's copy into DEST) store +0."""
    board = "blackhole" if ttnn.device.is_blackhole(device) else "wormhole_b0"
    if board not in ("blackhole",):
        pytest.skip(f"ttnn.softshrink keeps its stock kernel on {board}")
    words = torch.cat([torch.arange(0x80), torch.arange(0x8000, 0x8080)]).to(torch.int16)
    x = words.view(torch.bfloat16).repeat(4).reshape(1, 1, 32, 32)
    tt_x = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    raw = ttnn.to_torch(ttnn.softshrink(tt_x, lambd=0.5)).reshape(-1).view(torch.int16).to(torch.int32) & 0xFFFF
    assert torch.all(raw == 0x0000)


def test_softshrink_bf16_bfloat8_b(device):
    """bfloat8_b in and out: PCC 0.999 against torch, and each output within one bfloat8_b step of the BF16 call's
    packed to bfloat8_b (two in a block holding an infinity or NaN)."""
    board = "blackhole" if ttnn.device.is_blackhole(device) else "wormhole_b0"
    if board not in ("blackhole",):
        pytest.skip(f"ttnn.softshrink keeps its stock kernel on {board}")
    torch.manual_seed(0)
    x = torch.empty(1, 1, 64, 64).uniform_(-11.0, 11.0).to(torch.bfloat16)
    tt_x = ttnn.from_torch(x, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device)
    stored = ttnn.to_torch(tt_x).to(torch.float64).flatten()
    output = ttnn.softshrink(tt_x, lambd=0.5)
    assert output.dtype == ttnn.bfloat8_b
    actual = ttnn.to_torch(output).to(torch.float64).flatten()
    golden = ttnn.get_golden_function(ttnn.softshrink)(stored, lambd=0.5).to(torch.float64)
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
    actual = ttnn.to_torch(ttnn.softshrink(tt_x, lambd=0.5)).to(torch.float32).flatten()
    expected = (
        ttnn.to_torch(ttnn.typecast(ttnn.softshrink(tt_stored, lambd=0.5), ttnn.bfloat8_b)).to(torch.float32).flatten()
    )
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


def test_softshrink_bf16_exhaustive_inputs(device):
    """All 65,536 BF16 inputs: torch's result as stored, within STEPS BF16 steps.

    MODEL: per board, the kernel's offset in BF16 steps from torch's word.
    """
    board = "blackhole" if ttnn.device.is_blackhole(device) else "wormhole_b0"
    if board not in ("blackhole",):
        pytest.skip(f"ttnn.softshrink keeps its stock kernel on {board}")
    STEPS = 1
    MODEL = {
        -1: (
            "4302 4304 4306 4308 430A 430C 430E 4310 4312 4314 "
            "4316 4318 431A 431C 431E 4320 4322 4324 4326 4328 "
            "432A 432C 432E 4330 4332 4334 4336 4338 433A 433C "
            "433E 4340 4342 4344 4346 4348 434A 434C 434E 4350 "
            "4352 4354 4356 4358 435A 435C 435E 4360 4362 4364 "
            "4366 4368 436A 436C 436E 4370 4372 4374 4376 4378 "
            "437A 437C 437E 4380-4B00"
        ),
        1: (
            "C302 C304 C306 C308 C30A C30C C30E C310 C312 C314 "
            "C316 C318 C31A C31C C31E C320 C322 C324 C326 C328 "
            "C32A C32C C32E C330 C332 C334 C336 C338 C33A C33C "
            "C33E C340 C342 C344 C346 C348 C34A C34C C34E C350 "
            "C352 C354 C356 C358 C35A C35C C35E C360 C362 C364 "
            "C366 C368 C36A C36C C36E C370 C372 C374 C376 C378 "
            "C37A C37C C37E C380-CB00"
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

    actual = stored(ttnn.to_torch(ttnn.softshrink(tt_x, lambd=0.5)))
    x = x.flatten().to(torch.float64)
    reference = (torch.nn.functional.softshrink(x)).to(torch.float64).flatten()
    expected = stored(reference)
    finite = torch.isfinite(expected) & torch.isfinite(actual)
    ok = (actual == expected) | (finite & ((ordered(actual) - ordered(expected)).abs() <= STEPS))
    # A NaN result reaches L1 as a NaN or as an infinity of either sign.
    ok |= torch.isnan(reference) & (torch.isnan(actual) | torch.isinf(actual))
    # The SFPU flushes a result below the smallest normal, or within 2^-12 above it, to 0.
    ok |= (actual == 0) & (reference.abs() < tiny * (1 + 2.0**-12))
    shift = torch.zeros_like(bits)
    for offset, spans in MODEL.items():
        for span in spans.split():
            first, _, last = span.partition("-")
            shift[(bits >= int(first, 16)) & (bits <= int(last or first, 16))] = offset
    moved = ordered(expected) + shift
    word = torch.where(moved < 0, (-moved) | 0x8000, moved)
    held = stored(word.to(torch.int16).view(torch.bfloat16))
    normal = ((bits & 0x7F80) != 0) & ((bits & 0x7F80) != 0x7F80)
    ok = torch.where(normal & torch.isfinite(expected), actual == held, ok)
    assert ok.all(), (
        f"{(~ok).sum().item()} of {ok.numel()} inputs off torch and stock; first input 0x{bits[~ok][0].item():04x} "
        f"stored {actual[~ok][0].item()!r}, torch {reference[~ok][0].item()!r}"
    )
