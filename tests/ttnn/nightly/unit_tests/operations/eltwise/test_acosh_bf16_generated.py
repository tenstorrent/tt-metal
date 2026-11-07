# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Generated BF16 acosh cases, beside the unary category suites."""

import torch
import pytest
import ttnn
from tests.ttnn.unit_tests.operations.eltwise.eltwise_test_utils import (
    generate_bfloat16_bits,
    to_tt_tensor,
    SMALLEST_NORMAL_BF16,
)


@pytest.mark.parametrize("generated_first", [True, False])
def test_acosh_with_sqrt_in_sfpu_chain(device, generated_first):
    """acosh chained with stock sqrt, both orders: each step stores what its op stores standalone."""
    generated = (ttnn.UnaryWithParam(ttnn.UnaryOpType.ACOSH), lambda tensor: ttnn.acosh(tensor))
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


def test_acosh_bf16_unflushed_zero_and_subnormal_inputs(device):
    """-0 and subnormal inputs (kept by TT-NN's copy into DEST) store torch's result within 1 ULP."""
    words = torch.cat([torch.arange(0x80), torch.arange(0x8000, 0x8080)]).to(torch.int16)
    x = words.view(torch.bfloat16).repeat(4).reshape(1, 1, 32, 32)
    tt_x = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tiny = torch.finfo(torch.bfloat16).tiny

    def stored(t):
        t = t.to(torch.bfloat16).to(torch.float64).flatten()
        t = torch.where(torch.isnan(t), torch.full_like(t, float("inf")), t)
        return torch.where(t.abs() < tiny, torch.zeros_like(t), t)

    actual = stored(ttnn.to_torch(ttnn.acosh(tt_x)))
    x64 = x.flatten().to(torch.float64)
    reference = ttnn.get_golden_function(ttnn.acosh)(x64).to(torch.float64).flatten()
    rounded = stored(reference)
    spacing = 2.0 ** (torch.floor(torch.log2(reference.abs().clamp(min=tiny))) - 7)
    close = torch.isfinite(reference) & (actual != 0) & ((actual - reference).abs() < spacing)
    ok = (actual == rounded) | close
    bits = words.repeat(4).to(torch.int32) & 0xFFFF
    assert ok.all(), (
        f"{(~ok).sum().item()} of {ok.numel()} inputs off torch and stock; first input 0x{bits[~ok][0].item():04x} "
        f"stored {actual[~ok][0].item()!r}, torch {reference[~ok][0].item()!r}"
    )


def test_acosh_bf16_bfloat8_b(device):
    """bfloat8_b in and out: PCC 0.999 against torch, and each output within one bfloat8_b step of the BF16 call's
    packed to bfloat8_b (two in a block holding an infinity or NaN)."""
    torch.manual_seed(0)
    x = torch.empty(1, 1, 64, 64).uniform_(1.01, 100.0).to(torch.bfloat16)
    tt_x = ttnn.from_torch(x, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device)
    stored = ttnn.to_torch(tt_x).to(torch.float64).flatten()
    output = ttnn.acosh(tt_x)
    assert output.dtype == ttnn.bfloat8_b
    actual = ttnn.to_torch(output).to(torch.float64).flatten()
    golden = ttnn.get_golden_function(ttnn.acosh)(stored).to(torch.float64)
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
    actual = ttnn.to_torch(ttnn.acosh(tt_x)).to(torch.float32).flatten()
    expected = ttnn.to_torch(ttnn.typecast(ttnn.acosh(tt_stored), ttnn.bfloat8_b)).to(torch.float32).flatten()
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


def test_acosh_bf16_exhaustive_inputs(device):
    """All 65,536 BF16 inputs: torch's result as stored, within STEPS BF16 steps.

    MODEL: per board, the kernel's offset in BF16 steps from torch's word.
    """
    board = "blackhole" if ttnn.device.is_blackhole(device) else "wormhole_b0"
    STEPS = 1
    MODEL = {
        -1: (
            "3F82 3F8D 3F8F 3F91 3F99-3F9A 3FF1 4000 4009-4015 401E-4020 4025-4027 "
            "402B-402C 4030-4031 4034-4035 4038-4039 403C 4040 4043 4046 404A 4052 "
            "4055 4058 405D 4062 4070 414A 4157 415E 4165 4169 "
            "416C 4170 4174 4178 417C 4180 4182 4184 4186 4188 "
            "418A 418F 4191 4193 4198 419A 419D 419F 41A2 41A4 "
            "41A7 41A9-41AA 41AC 41AF 41B1-41B2 41B4 41B7 41BA 41BD 41C0 "
            "41C3 41C6 41C9 41CC 41CF-41D0 41D3 41D6 41D9-41DA 41DE-41DF 41E5-41E6 "
            "41ED 41F4 41FC 4202 4206 4213 42B6 42BC 42C2-42C3 42C8 "
            "430D 4388 4403 4407 448F 450C 4583 4602 4709 4780 "
            "4788 4810 4906 4985 4B07 4D04 4E8A 4F01 5087 5284 "
            "5481 5607 590B 5993 5B08 5D05 5F02 640B 6608 6805 "
            "6A02 6F82 7710 7885 7995"
        ),
        1: (
            "3F84 3FA0 3FA8 3FAB 3FAF-3FB2 3FB8-3FBC 3FC0-3FC1 3FC4-3FC6 3FCC 408B "
            "408D 4096 4098 409D 40A4 40A9 40AC 40B1 40B4 40B7 "
            "40BF 40C2 40C5 40C8 40CF 40D2 40D5 40D8 40DC 40DF "
            "40E3 40E6 40EA 40ED-40EE 40F1 40F5 40F9 40FD 410D 410F "
            "4116 411D 4122 412A 4135 4231 423C 4242 4248 424E-424F "
            "4255 425B-425C 4262-4263 4269-426A 4271-4272 4278-4279 4280 4284-4285 4289 428D "
            "4292 4296 429B 42A0 42CD-42CE 42D4 42DA-42DB 42E1-42E2 42E8-42E9 42EF-42F0 "
            "42F7-42F8 42FF 432F 433A 4340 4346 434C 4352-4353 4359-435A 4360-4361 "
            "4367-4368 436E-436F 4375-4377 437D-437E 43AE 43B9 43BF 43C5 43CB 43D1-43D2 "
            "43D8 43DE-43DF 43E5-43E6 43ED-43EE 43F4-43F5 43FC-43FD 442D 4438 443E 4444 "
            "444A 4450-4451 4457 445D-445E 4464-4465 446B-446C 4473-4474 447A-447C 44A7 44AC "
            "44B7 44C0 44CC 44D9 44E6-44E7 44F5-44F6 4533 453F 454B 4557-4558 "
            "4565-4566 4574-4575 45B2 45BE 45CA 45D6-45D7 45E4-45E5 45F2-45F4 4631 4648-4649 "
            "4655-4656 4663 4671-4672 46B0 46BB 46C7-46C8 46D4 46E1-46E2 46F0-46F1 46FF "
            "472F 473A 4746 4753 4760-4761 476E-476F 477E-477F 479A 47AE 47B9 "
            "47C5 47D2 47DF-47E0 47ED-47EE 47FC-47FD 4823 4838 4844 4850-4851 485E "
            "486C-486D 487B-487C 48A2 48B7 48C3 48CF-48D0 48DC-48DD 48EA-48EB 48F9-48FB 4921 "
            "4936 4942 494E-494F 495B-495C 4969-496A 4978-4979 49B5 49C1 49CD 49DA-49DB "
            "49E8-49E9 49F7-49F8 4A34 4A40 4A4C 4A59 4A67 4A75-4A76 4AB9 4AD1-4AD2 "
            "4AEC-4AED 4B2D 4B44 4B5D-4B5E 4B7A-4B7B 4BB7 4BCF 4BEA-4BEB 4C2B 4C41 "
            "4C5B 4C77-4C79 4CB5 4CCC-4CCD 4CE7-4CE8 4D29 4D3F 4D58-4D59 4D75-4D76 4D9E "
            "4DB3 4DCA-4DCB 4DE5-4DE6 4E27 4E3D 4E56 4E72-4E73 4EB1 4EC8 4EE2-4EE3 "
            "4F3B 4F53-4F54 4F6F-4F70 4FAF 4FC6 4FE0 4FFD-4FFE 5039 5051-5052 506D-506E "
            "50AD 50C4 50DD-50DE 50FA-50FB 5137 514F 516A-516B 51AB 51C1-51C2 51DB "
            "51F7-51F9 5235 524D 5267-5268 52A9 52BF 52D8-52D9 52F5-52F6 531E 5333 "
            "534A-534B 5365-5366 53BD 53D6 53F2-53F3 5431 5448 5462-5463 54BB 54D3-54D4 "
            "54EF-54F0 552F 5546 5560 557D-557E 55B9 55D1-55D2 55ED-55EE 5650 56AC "
            "56DC-56DD 5736 5769-576A 57C0-57C1 57F6-57F7 581F 584B-584C 58A8 58D7-58D8 5932 "
            "5964 59BC 59F1-59F2 5A47 5A7F 5AD2-5AD3 5B2E 5B5F 5BB8 5BEB-5BEC "
            "5C43 5C79-5C7A 5CCE 5D2A 5D5A 5DB4 5DE6-5DE7 5E3E 5E74-5E75 5EC9-5ECA "
            "5F55 5FB0 5FE1-5FE2 603A 606E-606F 60C5 60FC-60FD 6150-6151 61AC 61DC-61DD "
            "6236 6269-626A 62C1 62F6-62F8 634C 63D7-63D8 6432 6464-6465 64BC 64F1-64F2 "
            "6547 657F 65D3 662E 665F 66B8 66EC-66ED 6743 6779-677A 67A1 "
            "67CE 685A 68B4 68E6-68E7 693F 6974-6975 69C9-69CA 6A55-6A56 6AB0 6AE1-6AE2 "
            "6B3A 6B6E-6B6F 6BC5 6BFC-6BFD 6C23 6C50-6C51 6CAC 6CDC-6CDD 6D4E 6E66-6E67 "
            "6EBF 6FD5-6FD6 7030 70EE-70EF 7145 71A3 725C-725D 72B6 7377-7378 73CC "
            "74E4-74E5 753D 75FF 7653 776C-776D 77C3 78DA-78DB 7934 79F4-79F5 7A4A "
            "7B62 7BBB 7C7C-7C7E 7CD1 7DE9-7DEA 7E41 7F58"
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

    actual = stored(ttnn.to_torch(ttnn.acosh(tt_x)))
    x = x.flatten().to(torch.float64)
    reference = (torch.acosh(x)).to(torch.float64).flatten()
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
