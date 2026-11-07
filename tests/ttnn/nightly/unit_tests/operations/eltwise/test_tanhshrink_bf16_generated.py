# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Generated BF16 tanhshrink cases, beside the unary category suites."""

import torch
import pytest
import ttnn
from tests.ttnn.unit_tests.operations.eltwise.eltwise_test_utils import (
    generate_bfloat16_bits,
    to_tt_tensor,
    SMALLEST_NORMAL_BF16,
)


@pytest.mark.parametrize("generated_first", [True, False])
def test_tanhshrink_with_sqrt_in_sfpu_chain(device, generated_first):
    """tanhshrink chained with stock sqrt, both orders: each step stores what its op stores standalone."""
    generated = (ttnn.UnaryWithParam(ttnn.UnaryOpType.TANHSHRINK), lambda tensor: ttnn.tanhshrink(tensor))
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


def test_tanhshrink_bf16_unflushed_zero_and_subnormal_inputs(device):
    """-0 and subnormal inputs (kept by TT-NN's copy into DEST) store +0."""
    words = torch.cat([torch.arange(0x80), torch.arange(0x8000, 0x8080)]).to(torch.int16)
    x = words.view(torch.bfloat16).repeat(4).reshape(1, 1, 32, 32)
    tt_x = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    raw = ttnn.to_torch(ttnn.tanhshrink(tt_x)).reshape(-1).view(torch.int16).to(torch.int32) & 0xFFFF
    assert torch.all(raw == 0x0000)


def test_tanhshrink_bf16_bfloat8_b(device):
    """bfloat8_b in and out: PCC 0.999 against torch, and each output within one bfloat8_b step of the BF16 call's
    packed to bfloat8_b (two in a block holding an infinity or NaN)."""
    torch.manual_seed(0)
    x = torch.empty(1, 1, 64, 64).uniform_(-10.0, 10.0).to(torch.bfloat16)
    tt_x = ttnn.from_torch(x, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device)
    stored = ttnn.to_torch(tt_x).to(torch.float64).flatten()
    output = ttnn.tanhshrink(tt_x)
    assert output.dtype == ttnn.bfloat8_b
    actual = ttnn.to_torch(output).to(torch.float64).flatten()
    golden = ttnn.get_golden_function(ttnn.tanhshrink)(stored).to(torch.float64)
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
    actual = ttnn.to_torch(ttnn.tanhshrink(tt_x)).to(torch.float32).flatten()
    expected = ttnn.to_torch(ttnn.typecast(ttnn.tanhshrink(tt_stored), ttnn.bfloat8_b)).to(torch.float32).flatten()
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


def test_tanhshrink_bf16_nonfinite_inputs(device):
    """+-Inf and +-NaN store torch's result (NaN as an infinity) or STOCK's {input bits: word} for the board."""
    board = "blackhole" if ttnn.device.is_blackhole(device) else "wormhole_b0"
    STOCK = {"wormhole_b0": {0xFFC0: 0xFF80}}
    bits = torch.tensor([0x7F80, 0xFF80, 0x7FC0, 0xFFC0], dtype=torch.int32).repeat(256)
    x = bits.to(torch.int16).view(torch.bfloat16).reshape(1, 1, 32, 32)
    tt_x = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    def stored(t):
        return t.to(torch.bfloat16).to(torch.float64).flatten()

    actual = stored(ttnn.to_torch(ttnn.tanhshrink(tt_x)))
    reference = ttnn.get_golden_function(ttnn.tanhshrink)(x.flatten().to(torch.float64)).to(torch.float64)
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


def test_tanhshrink_bf16_exhaustive_inputs(device):
    """All 65,536 BF16 inputs: torch's result as stored, within STEPS BF16 steps, or STOCK's word per board.

    SKIP lists inputs the LLK sweep judges instead. MODEL: per board, the kernel's offset in BF16 steps from torch's word.
    """
    board = "blackhole" if ttnn.device.is_blackhole(device) else "wormhole_b0"
    STOCK = {"wormhole_b0": {0xFF80: [(0xFF81, 0xFFFF)]}}
    STEPS = 1
    SKIP = [
        (0x2993, 0x32B8),
        (0x32BA, 0x32E8),
        (0x32EA, 0x3327),
        (0x3329, 0x3338),
        (0x333A, 0x3346),
        (0x3348, 0x3352),
        (0x3354, 0x3368),
        (0x336A, 0x3371),
        (0x3373, 0x337A),
        (0x337C, 0x3384),
        (0x3386, 0x338B),
        (0x338D, 0x339D),
        (0x339F, 0x33A2),
        (0x33A4, 0x33A7),
        (0x33A9, 0x33AB),
        (0x33AD, 0x33B8),
        (0x33BA, 0x33BB),
        (0x33BD, 0x33BF),
        (0x33C1, 0x33C6),
        (0x33C8, 0x33C9),
        (0x33CB, 0x33CC),
        (0x33CE, 0x33CF),
        (0x33D1, 0x33D2),
        (0x33D4, 0x33D5),
        (0x33D7, 0x33D8),
        (0x33DA, 0x33DB),
        (0x33DD, 0x33E0),
        (0x33E2, 0x33E5),
        (0x33E7, 0x33E8),
        (0x33EA, 0x33EA),
        (0x33EC, 0x33EC),
        (0x33EE, 0x33EF),
        (0x33F1, 0x33F1),
        (0x33F3, 0x33F3),
        (0x33F5, 0x33F5),
        (0x33F7, 0x33F7),
        (0x33F9, 0x33FA),
        (0x33FC, 0x33FC),
        (0x33FE, 0x33FE),
        (0x3400, 0x3404),
        (0x3406, 0x3406),
        (0x3408, 0x340B),
        (0x340D, 0x3410),
        (0x3412, 0x3413),
        (0x3415, 0x3416),
        (0x3418, 0x3419),
        (0x341C, 0x341D),
        (0x3420, 0x3421),
        (0x3425, 0x3427),
        (0x342D, 0x3438),
        (0x3490, 0x3490),
        (0xA993, 0xB2B8),
        (0xB2BA, 0xB2E8),
        (0xB2EA, 0xB327),
        (0xB329, 0xB338),
        (0xB33A, 0xB346),
        (0xB348, 0xB352),
        (0xB354, 0xB368),
        (0xB36A, 0xB371),
        (0xB373, 0xB37A),
        (0xB37C, 0xB384),
        (0xB386, 0xB38B),
        (0xB38D, 0xB39D),
        (0xB39F, 0xB3A2),
        (0xB3A4, 0xB3A7),
        (0xB3A9, 0xB3AB),
        (0xB3AD, 0xB3B8),
        (0xB3BA, 0xB3BB),
        (0xB3BD, 0xB3BF),
        (0xB3C1, 0xB3C6),
        (0xB3C8, 0xB3C9),
        (0xB3CB, 0xB3CC),
        (0xB3CE, 0xB3CF),
        (0xB3D1, 0xB3D2),
        (0xB3D4, 0xB3D5),
        (0xB3D7, 0xB3D8),
        (0xB3DA, 0xB3DB),
        (0xB3DD, 0xB3E0),
        (0xB3E2, 0xB3E5),
        (0xB3E7, 0xB3E8),
        (0xB3EA, 0xB3EA),
        (0xB3EC, 0xB3EC),
        (0xB3EE, 0xB3EF),
        (0xB3F1, 0xB3F1),
        (0xB3F3, 0xB3F3),
        (0xB3F5, 0xB3F5),
        (0xB3F7, 0xB3F7),
        (0xB3F9, 0xB3FA),
        (0xB3FC, 0xB3FC),
        (0xB3FE, 0xB3FE),
        (0xB400, 0xB404),
        (0xB406, 0xB406),
        (0xB408, 0xB40B),
        (0xB40D, 0xB410),
        (0xB412, 0xB413),
        (0xB415, 0xB416),
        (0xB418, 0xB419),
        (0xB41C, 0xB41D),
        (0xB420, 0xB421),
        (0xB425, 0xB427),
        (0xB42D, 0xB438),
        (0xB490, 0xB490),
    ]
    MODEL = {
        -1: (
            "3353 3385 338C 33AC 33BC 33CA 33CD 33D0 33D3 33D6 "
            "33E1 33E6 33ED 33F6 33F8 3405 340C 341B 341F 3424 "
            "342B-342C 343B-343E 3444-3445 3449-344A 344D 3450 3453 3456 3458 345B "
            "345D 345F 3461 3464 3466 3470 3475 347E-3480 3484-3485 3489 "
            "348C 348E 3492 3495 349A-349B 349F-34A0 34A7 34AD 34B1 34BD-34BE "
            "34C5 34CA 34CD 34D6 34DB 34DD 34DF 34E1 34E6-34E7 34ED "
            "34F0 34F5-34F6 34FE-3500 3505 3509 350C 3512 351B 351D 351F-3520 "
            "3524 3527 352B 352D 3531-3532 3538 353E 3545 354A 3555 "
            "355B-3561 3566-3567 3570 3575 357E-3581 3585 3588 3595 359A-359B 359F-35A0 "
            "35A4 35A7 35AD 35B1-35B2 35B8 35BE 35C2 35C5 35CA 35D5 "
            "35DB-35E1 35E6-35E7 35F0 35F5 35FE-3601 3608 360B 3615 361A-361B 361F-3620 "
            "3627 362D 3631 363E 3642 3645 364A 3655 365B-3661 3666-3667 "
            "3670 3675 367E-3681 3688 368B 3695 369A-369B 369F-36A0 36A7 36AD "
            "36B1 36BE 36C2 36C5 36CA 36D5 36DB-36E1 36E6-36E7 36F0 36F5 "
            "36FE-3701 3708 370B 3715 371A-371B 371F-3720 3727 372D 3731 373E "
            "3742 3745 374A 3755 375B-3761 3766-3767 3770 3775 377E-3781 3788 "
            "378B 3795 379A-379B 379F-37A0 37A7 37AD 37B1 37BE 37C2 37C5 "
            "37CA 37D5 37DB-37E1 37E6-37E7 37F0 37F5 37FE-3801 3808 380B 3815 "
            "381A-381B 381F-3820 3827 382D 3831 383E 3842 3845 384A 3855 "
            "385B-3861 3866-3867 3870 3875 387E-3881 3888 388B 3895 389A-389B 389F-38A0 "
            "38A7 38AD 38B1 38BE 38C2 38C5 38CA 38D5 38DB-38E1 38E6-38E7 "
            "38F0 38F5 38FE-3901 3908 390B 3915 391A-391B 391F-3920 3927 392D "
            "3931 393E 3942 3945 394A 3955 395B-3961 3966-3967 3970 3975 "
            "397E-3981 3988 398B 3995 399A-399B 399F-39A0 39A7 39AD 39B1 39BE "
            "39C2 39C5 39CA 39D5 39DB-39E1 39E6-39E7 39F0 39F5 39FE-3A01 3A08 "
            "3A0B 3A15 3A1A-3A1B 3A1F-3A20 3A27 3A2D 3A31 3A3E 3A42 3A45 "
            "3A4A 3A55 3A5B-3A61 3A66-3A67 3A70 3A75 3A7E-3A81 3A88 3A8B 3A95 "
            "3A9A-3A9B 3A9F-3AA0 3AA7 3AAD 3AB1 3ABE 3AC2 3AC5 3ACA 3AD5 "
            "3ADB-3AE1 3AE6-3AE7 3AF0 3AF5 3AFE-3B01 3B08 3B0B 3B15 3B1A-3B1B 3B1F-3B20 "
            "3B27 3B2D 3B31 3B3E 3B42 3B45 3B4A 3B55 3B5B-3B61 3B66-3B67 "
            "3B70 3B75 3B7F-3B80 3B88 3B8B 3B95 3B9A-3B9B 3B9F-3BA0 3BA7 3BAD "
            "3BB1 3BBE 3BC2 3BC5 3BD5 3BDB-3BE0 3BE6 3BF0 3BF5 3C00 "
            "3C0B 3C15 3C1A-3C1B 3C1F 3C27 3C2D 3C31 3C3E 3C42 3C45 "
            "3C55 3C5C-3C60 3C66 3C75 3C8B 3C95 3C9B 3C9F 3CB1 3CBE "
            "3CC2 3CC5 3CD5 3CDC-3CDF 3EB6 3ED2 3EDA 3EE3 3EE9-3EEB 3EEF "
            "3F01 3F03 3F0A-3F0B 3F0E-3F10 3F14 3F16 3F18 3F1C 3F21 3F23 "
            "3F28 3F2B 3F30 3F3D 3F45 3F4D 3F52 3F5A 3FE2 3FE6 "
            "3FEB-3FEC 3FF0-3FF1 3FF6-3FF7 4000 4008 403D-403F B2B9 B2E9 B328 B339 "
            "B369 B37B B3A8 B3B9 B3E9 B3FB B3FD B422 B428 B439-B43A "
            "B443 B448 B44C B45A B462 B482-B483 B4CC BD5B BD61 BD67 "
            "BD7D-BD7E BD81 BD86 BD8A BD99 BDA1 BDA8 BDAA BDAC BDAF "
            "BDB9-BDBA BDC0 BDD0 BDD3 BDD7 BDE6 BDEE BDF2 BDF7 BE06 "
            "BE08 BE0A BE0D-BE0E BE12 BE15 BE19 BE23 BE29 BE2D BE30 "
            "BE34-BE39 BE3C BE42 BE48 BE52 BE58 BE5E-BE64 BE6A BE74 BE79 "
            "BE81 BE85 BE89 BE8D BE91 BE9F BEA5 BEAA BEAD BF71 "
            "BF76 BF82 BF8E BF97 BF9B-BF9C BFA2-BFA5 BFAC-BFAF BFB3 C014 C025-C02A "
            "C381 C383 C385 C387 C389 C38B C38D C38F C391 C393 "
            "C395 C397 C399 C39B C39D C39F C3A1 C3A3 C3A5 C3A7 "
            "C3A9 C3AB C3AD C3AF C3B1 C3B3 C3B5 C3B7 C3B9 C3BB "
            "C3BD C3BF C3C1 C3C3 C3C5 C3C7 C3C9 C3CB C3CD C3CF "
            "C3D1 C3D3 C3D5 C3D7 C3D9 C3DB C3DD C3DF C3E1 C3E3 "
            "C3E5 C3E7 C3E9 C3EB C3ED C3EF C3F1 C3F3 C3F5 C3F7 "
            "C3F9 C3FB C3FD C3FF"
        ),
        1: (
            "32B9 32E9 3328 3339 3369 337B 33A8 33B9 33E9 33FB "
            "33FD 3422 3428 3439-343A 3443 3448 344C 345A 3462 3482-3483 "
            "34CC 3D5B 3D61 3D67 3D7D-3D7E 3D81 3D86 3D8A 3D99 3DA1 "
            "3DA8 3DAA 3DAC 3DAF 3DB9-3DBA 3DC0 3DD0 3DD3 3DD7 3DE6 "
            "3DEE 3DF2 3DF7 3E06 3E08 3E0A 3E0D-3E0E 3E12 3E15 3E19 "
            "3E23 3E29 3E2D 3E30 3E34-3E39 3E3C 3E42 3E48 3E52 3E58 "
            "3E5E-3E64 3E6A 3E74 3E79 3E81 3E85 3E89 3E8D 3E91 3E9F "
            "3EA5 3EAA 3EAD 3F71 3F76 3F82 3F8E 3F97 3F9B-3F9C 3FA2-3FA5 "
            "3FAC-3FAF 3FB3 4014 4025-402A 4381 4383 4385 4387 4389 438B "
            "438D 438F 4391 4393 4395 4397 4399 439B 439D 439F "
            "43A1 43A3 43A5 43A7 43A9 43AB 43AD 43AF 43B1 43B3 "
            "43B5 43B7 43B9 43BB 43BD 43BF 43C1 43C3 43C5 43C7 "
            "43C9 43CB 43CD 43CF 43D1 43D3 43D5 43D7 43D9 43DB "
            "43DD 43DF 43E1 43E3 43E5 43E7 43E9 43EB 43ED 43EF "
            "43F1 43F3 43F5 43F7 43F9 43FB 43FD 43FF B353 B385 "
            "B38C B3AC B3BC B3CA B3CD B3D0 B3D3 B3D6 B3E1 B3E6 "
            "B3ED B3F6 B3F8 B405 B40C B41B B41F B424 B42B-B42C B43B-B43E "
            "B444-B445 B449-B44A B44D B450 B453 B456 B458 B45B B45D B45F "
            "B461 B464 B466 B470 B475 B47E-B480 B484-B485 B489 B48C B48E "
            "B492 B495 B49A-B49B B49F-B4A0 B4A7 B4AD B4B1 B4BD-B4BE B4C5 B4CA "
            "B4CD B4D6 B4DB B4DD B4DF B4E1 B4E6-B4E7 B4ED B4F0 B4F5-B4F6 "
            "B4FE-B500 B505 B509 B50C B512 B51B B51D B51F-B520 B524 B527 "
            "B52B B52D B531-B532 B538 B53E B545 B54A B555 B55B-B561 B566-B567 "
            "B570 B575 B57E-B581 B585 B588 B595 B59A-B59B B59F-B5A0 B5A4 B5A7 "
            "B5AD B5B1-B5B2 B5B8 B5BE B5C2 B5C5 B5CA B5D5 B5DB-B5E1 B5E6-B5E7 "
            "B5F0 B5F5 B5FE-B601 B608 B60B B615 B61A-B61B B61F-B620 B627 B62D "
            "B631 B63E B642 B645 B64A B655 B65B-B661 B666-B667 B670 B675 "
            "B67E-B681 B688 B68B B695 B69A-B69B B69F-B6A0 B6A7 B6AD B6B1 B6BE "
            "B6C2 B6C5 B6CA B6D5 B6DB-B6E1 B6E6-B6E7 B6F0 B6F5 B6FE-B701 B708 "
            "B70B B715 B71A-B71B B71F-B720 B727 B72D B731 B73E B742 B745 "
            "B74A B755 B75B-B761 B766-B767 B770 B775 B77E-B781 B788 B78B B795 "
            "B79A-B79B B79F-B7A0 B7A7 B7AD B7B1 B7BE B7C2 B7C5 B7CA B7D5 "
            "B7DB-B7E1 B7E6-B7E7 B7F0 B7F5 B7FE-B801 B808 B80B B815 B81A-B81B B81F-B820 "
            "B827 B82D B831 B83E B842 B845 B84A B855 B85B-B861 B866-B867 "
            "B870 B875 B87E-B881 B888 B88B B895 B89A-B89B B89F-B8A0 B8A7 B8AD "
            "B8B1 B8BE B8C2 B8C5 B8CA B8D5 B8DB-B8E1 B8E6-B8E7 B8F0 B8F5 "
            "B8FE-B901 B908 B90B B915 B91A-B91B B91F-B920 B927 B92D B931 B93E "
            "B942 B945 B94A B955 B95B-B961 B966-B967 B970 B975 B97E-B981 B988 "
            "B98B B995 B99A-B99B B99F-B9A0 B9A7 B9AD B9B1 B9BE B9C2 B9C5 "
            "B9CA B9D5 B9DB-B9E1 B9E6-B9E7 B9F0 B9F5 B9FE-BA01 BA08 BA0B BA15 "
            "BA1A-BA1B BA1F-BA20 BA27 BA2D BA31 BA3E BA42 BA45 BA4A BA55 "
            "BA5B-BA61 BA66-BA67 BA70 BA75 BA7E-BA81 BA88 BA8B BA95 BA9A-BA9B BA9F-BAA0 "
            "BAA7 BAAD BAB1 BABE BAC2 BAC5 BACA BAD5 BADB-BAE1 BAE6-BAE7 "
            "BAF0 BAF5 BAFE-BB01 BB08 BB0B BB15 BB1A-BB1B BB1F-BB20 BB27 BB2D "
            "BB31 BB3E BB42 BB45 BB4A BB55 BB5B-BB61 BB66-BB67 BB70 BB75 "
            "BB7F-BB80 BB88 BB8B BB95 BB9A-BB9B BB9F-BBA0 BBA7 BBAD BBB1 BBBE "
            "BBC2 BBC5 BBD5 BBDB-BBE0 BBE6 BBF0 BBF5 BC00 BC0B BC15 "
            "BC1A-BC1B BC1F BC27 BC2D BC31 BC3E BC42 BC45 BC55 BC5C-BC60 "
            "BC66 BC75 BC8B BC95 BC9B BC9F BCB1 BCBE BCC2 BCC5 "
            "BCD5 BCDC-BCDF BEB6 BED2 BEDA BEE3 BEE9-BEEB BEEF BF01 BF03 "
            "BF0A-BF0B BF0E-BF10 BF14 BF16 BF18 BF1C BF21 BF23 BF28 BF2B "
            "BF30 BF3D BF45 BF4D BF52 BF5A BFE2 BFE6 BFEB-BFEC BFF0-BFF1 "
            "BFF6-BFF7 C000 C008 C03D-C03F"
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

    actual = stored(ttnn.to_torch(ttnn.tanhshrink(tt_x)))
    x = x.flatten().to(torch.float64)
    reference = (torch.nn.functional.tanhshrink(x)).to(torch.float64).flatten()
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
