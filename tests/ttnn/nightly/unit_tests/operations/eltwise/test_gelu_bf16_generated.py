# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Generated BF16 gelu cases, beside the unary category suites."""

import torch
import pytest
import ttnn
from tests.ttnn.unit_tests.operations.eltwise.eltwise_test_utils import (
    generate_bfloat16_bits,
    to_tt_tensor,
    SMALLEST_NORMAL_BF16,
)


@pytest.mark.parametrize("generated_first", [True, False])
def test_gelu_with_sqrt_in_sfpu_chain(device, generated_first):
    """gelu chained with stock sqrt, both orders: each step stores what its op stores standalone."""
    generated = (ttnn.UnaryWithParam(ttnn.UnaryOpType.GELU, 0.0), lambda tensor: ttnn.gelu(tensor))
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


def test_gelu_bf16_unflushed_zero_and_subnormal_inputs(device):
    """-0 and subnormal inputs (kept by TT-NN's copy into DEST) store +0."""
    words = torch.cat([torch.arange(0x80), torch.arange(0x8000, 0x8080)]).to(torch.int16)
    x = words.view(torch.bfloat16).repeat(4).reshape(1, 1, 32, 32)
    tt_x = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    raw = ttnn.to_torch(ttnn.gelu(tt_x)).reshape(-1).view(torch.int16).to(torch.int32) & 0xFFFF
    assert torch.all(raw == 0x0000)


def test_gelu_bf16_bfloat8_b(device):
    """bfloat8_b in and out: PCC 0.999 against torch, and each output within one bfloat8_b step of the BF16 call's
    packed to bfloat8_b (two in a block holding an infinity or NaN)."""
    torch.manual_seed(0)
    x = torch.empty(1, 1, 64, 64).uniform_(-10.0, 10.0).to(torch.bfloat16)
    tt_x = ttnn.from_torch(x, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device)
    stored = ttnn.to_torch(tt_x).to(torch.float64).flatten()
    output = ttnn.gelu(tt_x)
    assert output.dtype == ttnn.bfloat8_b
    actual = ttnn.to_torch(output).to(torch.float64).flatten()
    golden = ttnn.get_golden_function(ttnn.gelu)(stored).to(torch.float64)
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
    actual = ttnn.to_torch(ttnn.gelu(tt_x)).to(torch.float32).flatten()
    expected = ttnn.to_torch(ttnn.typecast(ttnn.gelu(tt_stored), ttnn.bfloat8_b)).to(torch.float32).flatten()
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


def test_gelu_bf16_nonfinite_inputs(device):
    """+-Inf and +-NaN store torch's result (NaN as an infinity) or STOCK's {input bits: word} for the board."""
    board = "blackhole" if ttnn.device.is_blackhole(device) else "wormhole_b0"
    STOCK = {}
    bits = torch.tensor([0x7F80, 0xFF80, 0x7FC0, 0xFFC0], dtype=torch.int32).repeat(256)
    x = bits.to(torch.int16).view(torch.bfloat16).reshape(1, 1, 32, 32)
    tt_x = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    def stored(t):
        return t.to(torch.bfloat16).to(torch.float64).flatten()

    actual = stored(ttnn.to_torch(ttnn.gelu(tt_x)))
    reference = ttnn.get_golden_function(ttnn.gelu)(x.flatten().to(torch.float64)).to(torch.float64)
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


def test_gelu_bf16_exhaustive_inputs(device):
    """All 65,536 BF16 inputs: torch's result as stored, within STEPS BF16 steps.

    SKIP lists inputs the LLK sweep judges instead. MODEL: per board, the kernel's offset in BF16 steps from torch's word.
    """
    board = "blackhole" if ttnn.device.is_blackhole(device) else "wormhole_b0"
    STEPS = 1
    SKIP = [(0x0003, 0x00FE), (0x8004, 0x80FE), (0xC0F9, 0xC101), (0xC103, 0xC158)]
    MODEL = {
        -1: (
            "3B4B 3C63 3CBE 3CD7 3CEE 3D37 3D51 3D5D 3D8B 3D94 "
            "3DAC 3DC8 3DCE 3DE0 3DE6 3DF9 3DFE 3E04 3E12 3E32 "
            "3E39 3E40 3E4A 3E4D 3E6C 3E6F 3E7F 3E8D 3E96 3EB6 "
            "3EB8 3EBF 3EC1 3EC7 3ECC 3ED1 3EDC 3EE2 3EE8 3EEE "
            "3EF5 3EFC 3F06 3F0A 3F0F 3F14 3F1A-3F1B 3F22-3F23 3F35-3F3C 3F45-3F4C "
            "3F65-3F66 3F75 3F81 BB4B BC63 BCBE BCD7 BCEE BD37 BD51 "
            "BD5D BD8B BD94 BDAC BDC8 BDCE BDE0 BDE6 BDF1 BDF6 "
            "BE0A BE12 BE32 BE39 BE40 BE4A BE4D BE65 BE68 BE6B "
            "BE73 BE76 BE7B BE80 BE84-BE85 BE8A BE91-BE92 BE9B-BEA0 BEAC-BEAE BEB6 "
            "BEB8 BEBF BEC1 BEC6 BECB BED0 BED5 BED8 BEDB BEE0 "
            "BEE3 BEE6 BEE9 BEED BEF0 BEF3 BEF6-BEF7 BEFA BEFE BF01 "
            "BF07 BF09 BF0B BF0E BF10 BF13 BF16 BF19 BF1C-BF1D BF20-BF21 "
            "BF24-BF25 BF29-BF2B BF30-BF33 BF4E-BF51 BF57-BF59 BF5D-BF5F BF63-BF64 BF67-BF68 BF6C BF70 "
            "BF73 BF77 BF7A BF7D BF80 BF83 BF87 BF8C BF93 C003-C005 "
            "C010 C013 C016 C038 C03A C03C-C03E C049-C04B C04F C058 C05B "
            "C062 C0A6 C0B3 C0B6 C0B8 C0BC C0C0 C0C7-C0C8 C0CC-C0CD C0D1 "
            "C0D3 C0D7-C0D8 C0DB-C0DD C0E1-C0E2 C0E4 C0E7 C0E9 C0ED C0EF C0F1-C0F2 "
            "C0F5 C102"
        ),
        1: (
            "BFB6-BFBA BFD3-BFD6 BFDB-BFDC BFE0 BFE4 BFEA-BFEB BFEE-BFEF C022-C023 C025-C027 C043 "
            "C064 C06B C075 C078 C07A-C07B C081 C086 C088 C08C C091 "
            "C095 C097 C09A C09F C0CB"
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

    actual = stored(ttnn.to_torch(ttnn.gelu(tt_x)))
    x = x.flatten().to(torch.float64)
    reference = (torch.nn.functional.gelu(x)).to(torch.float64).flatten()
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
