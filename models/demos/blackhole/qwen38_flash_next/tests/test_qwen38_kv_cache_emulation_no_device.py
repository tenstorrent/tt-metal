# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""The KV cache emulation without a model: the e4m3 conversion against a scalar port of tt-metal's ``float8.cpp``
(nearest-even, ties, saturation, the flush below 2^-6, signed zero, NaN) and against torch's e4m3fn where the two
agree; the bfp8_b block rounding; the scaled quantiser's scales and codes; and the injection: the rounding QSA over
small synthetic weights is bitwise the oracle at the identity and holds e4m3-representable K/V otherwise."""

from __future__ import annotations

import math
import struct

import pytest
import torch

from models.demos.blackhole.qwen38_flash_next.tools import qwen38_kv_cache_emulation as kv
from models.demos.blackhole.qwen38_flash_next.tools import qwen38_reference_corpus as corpus
from models.demos.blackhole.qwen38_flash_next.tt.qsa import Qwen38QSA, Qwen38QSADimensions, Qwen38QSAWeights


def e4m3_bits_reference(value: float) -> int:
    """A scalar port of ``fp32_to_fp8_e4m3_bits`` (tt_metal/impl/data_format/float8.cpp)."""
    if math.isnan(value):
        return 0x7F
    u = struct.unpack("<I", struct.pack("<f", value))[0]
    sign = (u >> 31) & 1
    if math.isinf(value):
        return (sign << 7) | 0x7E
    exponent = (u >> 23) & 0xFF
    mantissa = u & 0x7FFFFF
    if exponent == 0:
        return sign << 7
    fp8_exponent = exponent - 127 + 7
    if fp8_exponent > 15:
        return (sign << 7) | 0x7E
    if fp8_exponent <= 0:
        return sign << 7
    fp8_mantissa = (mantissa >> 20) & 0x7
    round_bit = (mantissa >> 19) & 1
    sticky = mantissa & 0x7FFFF
    if round_bit and (sticky or (fp8_mantissa & 1)):
        fp8_mantissa += 1
        if fp8_mantissa >= 8:
            fp8_mantissa = 0
            fp8_exponent += 1
            if fp8_exponent > 15:
                return (sign << 7) | 0x7E
    if fp8_exponent == 15 and fp8_mantissa == 7:
        return (sign << 7) | 0x7E
    return (sign << 7) | (fp8_exponent << 3) | fp8_mantissa


EDGE_VALUES = [
    0.0,
    -0.0,
    1.0,
    -1.0,
    1.0625,  # tie between 1.0 (M=0) and 1.125 (M=1): to even -> 1.0
    1.1875,  # tie between 1.125 (M=1) and 1.25 (M=2): to even -> 1.25
    1.0626,
    1.1874,
    1.9375,  # 1 + 7/8, exact
    1.96875,  # tie between 1.875 and 2.0: to even -> 2.0 (carry into the exponent)
    448.0,
    -448.0,
    447.9,
    456.0,  # the tie between 448 and the NaN slot: saturates to 448
    463.9,
    464.0,
    1.0e6,
    float("inf"),
    -float("inf"),
    2.0**-6,  # the smallest normal
    -(2.0**-6),
    2.0**-6 * 1.875,
    2.0**-6 * 0.999,  # below the smallest normal: flushed, not rounded up
    2.0**-7,
    2.0**-7 * 1.9999,  # would round to 2^-6 if subnormals rounded; the flush decides first
    -(2.0**-9),
    1.0e-30,
    1.0e-45,  # an fp32 subnormal
    -1.0e-45,
]


def test_e4m3_codes_match_the_float8_cpp_port_on_edges_and_random_values():
    generator = torch.Generator().manual_seed(11)
    random = torch.cat(
        [
            torch.randn(4096, generator=generator) * 4.0,
            torch.randn(4096, generator=generator).to(torch.bfloat16).float(),  # bf16 inputs, the cache's case
            torch.rand(1024, generator=generator) * 1000.0 - 500.0,
            torch.exp2(torch.rand(1024, generator=generator) * 20.0 - 12.0),
        ]
    )
    values = torch.cat([torch.tensor(EDGE_VALUES, dtype=torch.float32), random, torch.tensor([float("nan")])])
    codes = kv.fp8_e4m3_codes(values)
    expected = torch.tensor([e4m3_bits_reference(float(v)) for v in values.tolist()], dtype=torch.uint8)
    mismatches = (codes != expected).nonzero().flatten().tolist()
    assert not mismatches, (
        f"{len(mismatches)} codes differ from the float8.cpp port, first at {mismatches[:5]}: "
        f"values {values[mismatches[:5]].tolist()} got {codes[mismatches[:5]].tolist()} "
        f"expected {expected[mismatches[:5]].tolist()}"
    )
    print(f"e4m3 codes: {len(values)} values, 0 mismatches vs the float8.cpp port (expected 0)")


@pytest.mark.parametrize(
    "value, code, decoded",
    [
        (0.0, 0x00, 0.0),
        (-0.0, 0x80, -0.0),
        (1.0, 0x38, 1.0),
        (1.0625, 0x38, 1.0),
        (1.1875, 0x3A, 1.25),
        (1.96875, 0x40, 2.0),
        (448.0, 0x7E, 448.0),
        (456.0, 0x7E, 448.0),
        (1.0e6, 0x7E, 448.0),
        (-1.0e6, 0xFE, -448.0),
        (float("inf"), 0x7E, 448.0),
        (2.0**-6, 0x08, 2.0**-6),
        (2.0**-6 * 0.999, 0x00, 0.0),
        (-(2.0**-7) * 1.9999, 0x80, -0.0),
        (2.0**-6 * 1.875, 0x0F, 2.0**-6 * 1.875),
    ],
)
def test_e4m3_named_vectors(value, code, decoded):
    got = kv.fp8_e4m3_codes(torch.tensor([value], dtype=torch.float32))
    back = kv.fp8_e4m3_values(got)
    print(f"e4m3({value}) = 0x{int(got[0]):02X} -> {float(back[0])} (expected 0x{code:02X} -> {decoded})")
    assert int(got[0]) == code
    assert float(back[0]) == decoded
    assert math.copysign(1.0, float(back[0])) == math.copysign(1.0, decoded)


def test_e4m3_nan_and_negative_zero_decode():
    codes = torch.tensor([0x7F, 0xFF, 0x80, 0x00, 0x01, 0x87], dtype=torch.uint8)
    values = kv.fp8_e4m3_values(codes)
    assert torch.isnan(values[0]) and torch.isnan(values[1])
    assert float(values[2]) == 0.0 and math.copysign(1.0, float(values[2])) == -1.0
    assert float(values[3]) == 0.0 and math.copysign(1.0, float(values[3])) == 1.0
    assert float(values[4]) == 0.0 and float(values[5]) == 0.0, "subnormal codes decode to the signed zero"
    assert int(kv.fp8_e4m3_codes(torch.tensor([float("nan")]))[0]) == 0x7F
    print("NaN codes decode to NaN, subnormal codes to signed zero (expected)")


def test_e4m3_agrees_with_torch_e4m3fn_on_the_normal_range():
    generator = torch.Generator().manual_seed(3)
    values = torch.randn(65536, generator=generator).to(torch.bfloat16).float() * 8.0
    inside = (values.abs() >= kv.E4M3_MIN_NORMAL) & (values.abs() < 448.0)
    ours = kv.fp8_e4m3_round(values[inside])
    theirs = values[inside].to(torch.float8_e4m3fn).float()
    differing = int((ours != theirs).sum())
    print(f"vs torch.float8_e4m3fn on {int(inside.sum())} normal-range values: {differing} differ (expected 0)")
    assert differing == 0


def test_e4m3_round_trip_is_idempotent_and_exact_in_bf16():
    generator = torch.Generator().manual_seed(5)
    values = torch.randn(8192, generator=generator) * 30.0
    once = kv.fp8_e4m3_round(values)
    assert torch.equal(kv.fp8_e4m3_round(once), once)
    assert torch.equal(once.to(torch.bfloat16).float(), once), "every e4m3 value is exact in bf16"
    step = (once - values).abs().max()
    print(f"max |e4m3(x) - x| over |x| < ~150: {float(step):.4f} (expected <= 8.0, half an e4m3 step at 128..256)")
    assert float(step) <= 8.0


def test_round_mantissa_ten_bits():
    # ties to even (1 + 2^-11 sits between 1.0 and 1 + 2^-10), above a tie rounds up, a tie at the top carries into
    # the exponent (2 - 2^-11 -> 2.0)
    x = torch.tensor(
        [1.0 + 2.0**-11, 1.0 + 3 * 2.0**-11, 1.0 + 2.0**-10 + 2.0**-11 + 2.0**-12, 1.99951171875],
        dtype=torch.float32,
    )
    got = kv.round_mantissa(x, 10)
    expected = torch.tensor([1.0, 1.0 + 2 * 2.0**-10, 1.0 + 2 * 2.0**-10, 2.0], dtype=torch.float32)
    print(f"round_mantissa(10): {got.tolist()} (expected {expected.tolist()})")
    assert torch.equal(got, expected)
    special = torch.tensor([float("nan"), float("inf"), -float("inf"), 0.0, -0.0])
    back = kv.round_mantissa(special, 10)
    assert torch.isnan(back[0]) and torch.equal(back[1:], special[1:])
    bf16 = torch.randn(256, generator=torch.Generator().manual_seed(9)).to(torch.bfloat16).float()
    assert torch.equal(
        kv.round_mantissa(bf16, 10), bf16
    ), "bf16 inputs (7 mantissa bits) are unchanged by the 10-bit step"


def test_bfp8_block_rounding(expect_error):
    # one 16-block: the largest element sets the shared exponent; a value within 8x of it is exact when it has <= 4
    # significant bits; a value at 1/16 of the maximum with 4 significant bits loses its lowest bit.
    block = torch.zeros(16)
    block[0] = 1.875  # 1.111b, shared exponent 0: q = 1.875 * 64 = 120
    block[1] = 1.0 / 8 * 1.875  # 4 significant bits at exponent -3: q = 15, exact
    block[2] = 1.0 / 16 * 1.875  # exponent -4: q = 7.5 -> 8 (rne) or 7 (truncate)
    block[3] = 1.0 / 128  # exponent -7: q = 0.5 -> 0 (rne, tie to even) or 0 (truncate)
    block[4] = 1.0 / 128 * 1.5  # q = 0.75 -> 1 (rne) or 0 (truncate)
    block[5] = -1.0 / 16 * 1.875
    rne = kv.bfp8_b_round(block, rounding="rne")
    truncate = kv.bfp8_b_round(block, rounding="truncate")
    print(f"bfp8 rne: {rne[:6].tolist()}  truncate: {truncate[:6].tolist()}")
    assert rne[0] == 1.875 and truncate[0] == 1.875
    assert rne[1] == block[1] and truncate[1] == block[1]
    assert rne[2] == 8 / 64 and truncate[2] == 7 / 64
    assert rne[3] == 0.0 and truncate[3] == 0.0
    assert rne[4] == 1 / 64 and truncate[4] == 0.0
    assert rne[5] == -8 / 64 and truncate[5] == -7 / 64
    assert torch.equal(kv.bfp8_b_round(torch.zeros(32)), torch.zeros(32))
    # e4m3 values within 8x of their block maximum are exact
    generator = torch.Generator().manual_seed(2)
    values = kv.fp8_e4m3_round(
        torch.exp2(torch.rand(64, 16, generator=generator) * 3.0) * torch.randn(64, 16, generator=generator).sign()
    )
    assert torch.equal(kv.bfp8_b_round(values), values)
    # the clamp at 127: 1.9921875 (q would be 127.5 -> 128) stays at 127/64
    assert float(kv.bfp8_b_round(torch.tensor([1.9921875] + [0.0] * 15))[0]) == 127 / 64
    with expect_error(ValueError):
        kv.bfp8_b_round(torch.zeros(15))
    with expect_error(ValueError):
        kv.bfp8_b_round(torch.zeros(16), rounding="nearest")


def test_scaled_fp8_scales_and_codes(expect_error):
    generator = torch.Generator().manual_seed(4)
    x = torch.randn(3, 256, generator=generator).to(torch.bfloat16).float()
    x[1, :128] = 0.0  # a zero block: the clamp at 1e-4 sets its scale
    x[2, 128] = 300.0  # a block dominated by one large value
    codes, scales = kv.scaled_fp8_quantize(x, power_of_two_scale=False)
    assert codes.shape == x.shape and codes.dtype == torch.uint8
    assert scales.shape == (3, 2) and scales.dtype == torch.float32
    amax = x.reshape(3, 2, 128).abs().amax(dim=-1)
    expected = amax.clamp(min=1e-4) * torch.tensor(1.0 / 448.0, dtype=torch.float32)
    print(f"scales {scales.tolist()} (expected {expected.tolist()})")
    assert torch.equal(scales, expected)
    assert float(scales[1, 0]) == float(
        torch.tensor(1e-4, dtype=torch.float32) * torch.tensor(1.0 / 448.0, dtype=torch.float32)
    )
    # the block maximum quantises to 448 (0x7E with its sign); the reconstruction error stays within one e4m3 step
    flat_codes = codes.reshape(3, 2, 128)
    for row in range(3):
        for block in range(2):
            if float(amax[row, block]) == 0.0:
                assert int(flat_codes[row, block].max()) == 0
                continue
            position = int(x.reshape(3, 2, 128)[row, block].abs().argmax())
            assert int(flat_codes[row, block, position]) & 0x7F == 0x7E, (row, block, position)
    back = kv.scaled_fp8_dequantize(codes, scales, subnormals=False)
    error = (back - x).abs().reshape(3, 2, 128).amax(dim=-1)
    bound = scales * 32.0  # half an e4m3 step at 256..448 is 16 code units; the top step is 32 * scale
    print(f"scaled reconstruction max error per block {error.tolist()} vs bound {bound.tolist()}")
    assert torch.all(error <= bound)
    # power-of-two scales: exact powers are kept, others go up to the next power
    _, pow2 = kv.scaled_fp8_quantize(x, power_of_two_scale=True)
    ratio = pow2 / expected
    assert torch.all((ratio >= 1.0) & (ratio < 2.0)), ratio
    assert torch.all(torch.log2(pow2) == torch.round(torch.log2(pow2)))
    exact = torch.zeros(1, 128)
    exact[0, 0] = 448.0 * 2.0**-3  # amax * (1/448) = 2^-3 exactly (the fp32 product of 56 and fp32(1/448) is 0.125)
    _, kept = kv.scaled_fp8_quantize(exact, power_of_two_scale=True)
    print(f"power-of-two scale of an exact power: {float(kept[0, 0])} (expected 0.125)")
    assert float(kept[0, 0]) == 0.125
    with expect_error(ValueError):
        kv.scaled_fp8_quantize(torch.zeros(100))


@pytest.mark.parametrize(
    "value, code, decoded",
    [
        (1.0, 0x37, 0.9375),  # exactly on a code: falls one code toward zero (1.875 * 2^-1)
        (56.0, 0x65, 52.0),
        (448.0, 0x7D, 416.0),
        (60.0, 0x66, 56.0),  # 1.875 * 32, exact -> 1.75 * 32
        (1.3, 0x3A, 1.25),  # 1.3 = 1.0100b: truncated to 1.25 (nearest would be 1.25 too)
        (1.24, 0x39, 1.125),  # truncated to 1.125 (nearest would be 1.25)
        (-1.24, 0xB9, -1.125),
        (2.0**-7 * 1.1875, 0x01, 2.0**-9),  # in [2^-7, 2^-6): truncated mantissa bits under E=0, not the value
        (0.00928, 0x01, 2.0**-9),
        (0.0155, 0x07, 7 * 2.0**-9),
        (2.0**-7, 0x00, 0.0),  # exactly on the binade's edge: the fall takes it below 2^-7, a signed zero
        (0.007, 0x00, 0.0),
        (-0.007, 0x80, -0.0),
        (0.0, 0x00, 0.0),
        (-0.0, 0x80, -0.0),
        (500.0, 0x7E, 448.0),
        (-500.0, 0xFE, -448.0),
    ],
)
def test_e4m3_device_codes_named_vectors(value, code, decoded):
    got = kv.fp8_e4m3_codes_device(torch.tensor([value], dtype=torch.float32))
    back = kv.fp8_e4m3_values(got, subnormals=True)
    print(f"device e4m3({value}) = 0x{int(got[0]):02X} -> {float(back[0])} (expected 0x{code:02X} -> {decoded})")
    assert int(got[0]) == code
    assert float(back[0]) == decoded
    assert math.copysign(1.0, float(back[0])) == math.copysign(1.0, decoded)


def test_e4m3_device_codes_truncate_and_stay_below_the_value():
    generator = torch.Generator().manual_seed(8)
    values = torch.randn(65536, generator=generator).to(torch.bfloat16).float() * 40.0
    codes = kv.fp8_e4m3_codes_device(values)
    back = kv.fp8_e4m3_values(codes, subnormals=True)
    normal = values.abs() >= 2.0**-6
    assert torch.all(back[normal].abs() < values[normal].abs()), "every normal-range magnitude is written below itself"
    assert torch.all(back[normal].abs() >= values[normal].abs() * 0.875 - 1e-6), "by less than one e4m3 step"
    nearest = kv.fp8_e4m3_round(values)
    differing = float((codes != kv.fp8_e4m3_codes(values))[normal].float().mean())
    print(f"device codes differ from nearest-even on {differing:.3f} of normal-range values (device half: 0.522)")
    assert 0.4 < differing < 0.65
    assert torch.all(back[normal].abs() <= nearest[normal].abs()), "truncation never exceeds nearest-even"


def test_scaled_fp8_device_scales_and_codes():
    x = torch.zeros(4, 256)
    x[0, 0] = 448.0  # raw = 1.0 exactly: kept
    x[1, 0] = 300.0  # raw = 0.6696: next power of two = 1.0
    x[2, 0] = 500.0  # raw = 1.116: 2.0
    x[3, 0] = 56.0  # raw = 0.125 exactly: kept
    x[:, 128] = 1.0  # second block: raw = 0.00223: 2^-8 = 0.0039
    codes, scales = kv.scaled_fp8_quantize_device(x)
    expected = torch.tensor([[1.0, 2.0**-8], [1.0, 2.0**-8], [2.0, 2.0**-8], [0.125, 2.0**-8]])
    print(f"device scales {scales.tolist()} (expected {expected.tolist()})")
    assert torch.equal(scales, expected)
    assert torch.all(torch.log2(scales) == torch.round(torch.log2(scales)))
    back = kv.scaled_fp8_dequantize(codes, scales)
    # 448 / 1 -> 416; 300 / 1 = 1.171875 * 256 truncates to 1.125 * 256 = 288; 500 / 2 = 250 truncates to 240 -> 480;
    # 56 / 0.125 = 448 -> 416 -> 52
    print(f"block maxima {x[:, 0].tolist()} -> {back[:, 0].tolist()} (expected 416, 288, 480, 52)")
    assert back[:, 0].tolist() == [416.0, 288.0, 480.0, 52.0]
    assert torch.all(back[:, 128] == 240 * 2.0**-8), "1.0 / 2^-8 = 256, exactly on a code, falls to 240"
    zero = torch.zeros(1, 128)
    _, tiny = kv.scaled_fp8_quantize_device(zero)
    clamp_raw = torch.tensor(1e-4, dtype=torch.float32) * torch.tensor(1.0 / 448.0, dtype=torch.float32)
    assert float(tiny[0, 0]) == float(torch.exp2(torch.ceil(torch.log2(clamp_raw))))
    assert torch.equal(
        kv.scaled_fp8_scales(x, power_of_two=False)[:, 0], x[:, 0].clamp(min=1e-4) * torch.tensor(1.0 / 448.0)
    )


def test_kv_rounding_formats_and_descriptions(expect_error):
    generator = torch.Generator().manual_seed(6)
    key = torch.randn(1, 2, 5, 256, generator=generator).to(torch.bfloat16)
    value = torch.randn(1, 2, 5, 256, generator=generator).to(torch.bfloat16)
    identity = kv.KVCacheRounding("bf16")
    assert identity.identity and identity.key(key) is key and identity.value(value) is value
    for name in kv.KV_FORMATS[1:]:
        rounding = kv.KVCacheRounding(name)
        k, v = rounding.key(key), rounding.value(value)
        assert k.dtype == torch.bfloat16 and v.dtype == torch.bfloat16 and k.shape == key.shape
        assert not torch.equal(v, value), name
        assert not torch.equal(k, key), name
        relative = float(((v.float() - value.float()).norm() / value.float().norm()))
        bias = float((v.float().abs() - value.float().abs()).mean() / value.float().abs().mean())
        print(f"{rounding.describe()}: relative V error {relative:.4f}, magnitude bias {bias:+.4f}")
        assert 0.005 < relative < 0.1, (name, relative)
        assert name in rounding.describe()
        if name == "scaled_fp8":
            assert bias < -0.02, "the truncating packer biases every magnitude low"
        else:
            assert abs(bias) < 0.01, (name, bias)
    device = kv.KVCacheRounding("scaled_fp8").value(value).float()
    hypothetical = kv.KVCacheRounding("scaled_fp8_rne").value(value).float()
    assert not torch.equal(device, hypothetical)
    assert float((device - value.float()).norm()) > float((hypothetical - value.float()).norm())
    storage = kv.KVCacheRounding("fp8_e4m3_storage").value(value).float()
    assert torch.equal(kv.fp8_e4m3_round(storage), storage), "the storage format holds e4m3 values exactly"
    tilized = kv.KVCacheRounding("fp8_e4m3").value(value).float()
    assert torch.equal(kv.bfp8_b_round(tilized), tilized)
    with expect_error(ValueError):
        kv.KVCacheRounding("fp8")
    with expect_error(ValueError):
        kv.KVCacheRounding("fp8_e4m3", bfp8_rounding="stochastic")


def _synthetic_weights(generator: torch.Generator) -> Qwen38QSAWeights:
    dims = Qwen38QSADimensions(
        hidden_size=32,
        query_heads=4,
        kv_heads=2,
        head_dim=32,
        rope_dim=8,
        index_query_heads=4,
        index_kv_heads=1,
        index_head_dim=16,
        token_budget=4,
        compress_ratio=2,
    )
    scaled = lambda *shape, factor=0.2: (torch.randn(*shape, generator=generator) * factor).to(
        torch.bfloat16
    )  # noqa: E731
    return Qwen38QSAWeights(
        layer_idx=3,
        dimensions=dims,
        rms_norm_eps=1e-6,
        qg=scaled(2 * dims.query_width, dims.hidden_size),
        k=scaled(dims.kv_width, dims.hidden_size),
        v=scaled(dims.kv_width, dims.hidden_size, factor=1.0),
        out=scaled(dims.hidden_size, dims.query_width),
        q_norm=scaled(dims.head_dim, factor=0.1),
        k_norm=scaled(dims.head_dim, factor=0.1),
        index_qk=scaled((dims.index_query_heads + dims.index_kv_heads) * dims.index_head_dim, dims.hidden_size),
        index_q_norm=scaled(dims.index_head_dim, factor=0.1),
        index_k_norm=scaled(dims.index_head_dim, factor=0.1),
    )


def _run(attention: Qwen38QSA, hidden: torch.Tensor, chunks: tuple[int, ...]):
    """Prefill ``hidden`` in ``chunks`` through ``attention`` the way the model does (causal float mask, full rope)."""
    batch, total, _ = hidden.shape
    rope_dim = attention.weights.dimensions.rope_dim
    positions = torch.arange(total).float()
    frequencies = torch.exp2(-torch.arange(0, rope_dim, 2).float() / rope_dim)
    angles = positions[:, None] * frequencies[None, :]
    cos = torch.cat([angles.cos(), angles.cos()], dim=-1).to(torch.bfloat16).expand(batch, -1, -1)
    sin = torch.cat([angles.sin(), angles.sin()], dim=-1).to(torch.bfloat16).expand(batch, -1, -1)
    outputs, state, start = [], None, 0
    for chunk in chunks:
        end = start + chunk
        query_positions = torch.arange(start, end).view(chunk, 1)
        key_positions = torch.arange(end).view(1, end)
        mask = torch.where(key_positions <= query_positions, 0.0, torch.finfo(torch.float32).min).view(1, 1, chunk, end)
        output, state, _ = attention.forward(
            hidden[:, start:end], (cos[:, :end], sin[:, :end]), mask.expand(batch, -1, -1, -1), state=state
        )
        outputs.append(output)
        start = end
    return torch.cat(outputs, dim=1), state


def test_rounding_qsa_is_bitwise_the_oracle_at_the_identity():
    generator = torch.Generator().manual_seed(21)
    weights = _synthetic_weights(generator)
    hidden = (torch.randn(1, 13, 32, generator=generator) * 2.0).to(torch.bfloat16)
    oracle = Qwen38QSA(weights)
    rounded = kv.Qwen38QSAKVRounded(weights, kv.KVCacheRounding("bf16"))
    for chunks in ((13,), (5, 8), (1,) * 13):
        expected, expected_state = _run(oracle, hidden, chunks)
        got, got_state = _run(rounded, hidden, chunks)
        assert torch.equal(got, expected), chunks
        assert torch.equal(got_state.keys, expected_state.keys) and torch.equal(got_state.values, expected_state.values)
        assert torch.equal(got_state.raw_index_keys, expected_state.raw_index_keys)
    print("identity rounding: output and state bitwise the oracle over 3 chunkings (expected)")


def test_rounding_qsa_holds_rounded_kv_and_captures():
    generator = torch.Generator().manual_seed(22)
    weights = _synthetic_weights(generator)
    hidden = (torch.randn(1, 9, 32, generator=generator) * 2.0).to(torch.bfloat16)
    exact, exact_state = _run(Qwen38QSA(weights), hidden, (4, 5))
    captured = []
    rounding = kv.KVCacheRounding("fp8_e4m3_storage")
    attention = kv.Qwen38QSAKVRounded(
        weights, rounding, capture=lambda index, tensors: captured.append((index, tensors))
    )
    got, state = _run(attention, hidden, (4, 5))
    assert torch.equal(state.keys, rounding.key(exact_state.keys)), "the cache holds the rounded K of every position"
    assert torch.equal(state.values, rounding.value(exact_state.values))
    assert torch.equal(state.raw_index_keys, exact_state.raw_index_keys), "the index cache is untouched"
    assert not torch.equal(got, exact)
    assert [index for index, _ in captured] == [3, 3]
    first, second = captured[0][1], captured[1][1]
    assert first["previous_length"] == 0 and second["previous_length"] == 4
    assert torch.equal(first["exact_key"], exact_state.keys[:, :, :4]) and torch.equal(
        second["exact_value"], exact_state.values[:, :, 4:]
    )
    assert torch.equal(second["current_key"], state.keys[:, :, 4:])
    assert first["combined_mask"].shape == (1, 1, 4, 4) and second["combined_mask"].shape == (1, 1, 5, 9)
    assert first["query"].shape == (1, 4, 4, 32)
    relative = float((got.float() - exact.float()).norm() / exact.float().norm())
    print(f"fp8_e4m3_storage on synthetic weights: relative output change {relative:.4f} (expected > 0)")


def test_install_kv_rounding_wraps_only_qsa_layers_and_composes():
    class _Layer:
        def __init__(self, attention):
            self.attention = attention

    weights = _synthetic_weights(torch.Generator().manual_seed(23))

    class _Oracle:
        def __init__(self):
            self.created = {0: _Layer(object()), 3: _Layer(Qwen38QSA(weights))}
            self.calls = []

        def layer(self, index):
            self.calls.append(index)
            return self.created[index]

    oracle = _Oracle()
    rounding = kv.KVCacheRounding("fp8_e4m3")
    kv.install_kv_rounding(oracle, rounding)
    assert type(oracle.layer(0).attention) is object
    wrapped = oracle.layer(3).attention
    assert isinstance(wrapped, kv.Qwen38QSAKVRounded) and wrapped.weights is weights and wrapped.rounding is rounding
    assert oracle.layer(3).attention is wrapped, "a second call keeps the same instance"
    assert oracle.calls == [0, 3, 3]
    print("install_kv_rounding: GDN layer untouched, QSA layer wrapped once (expected)")


def test_reference_corpus_oracle_takes_kv(expect_error):
    assert corpus.KV_CHOICES == kv.KV_FORMATS, "the CLI choices pin the emulation's formats"
    parser = corpus._parser()
    args = parser.parse_args(["oracle", "--checkpoint", "c", "--out", "o", "--kv", "scaled_fp8"])
    assert args.kv == "scaled_fp8"
    assert parser.parse_args(["oracle", "--checkpoint", "c", "--out", "o"]).kv == "bf16"
    with expect_error(SystemExit):
        parser.parse_args(["oracle", "--checkpoint", "c", "--out", "o", "--kv", "fp8"])
    with expect_error(SystemExit):
        parser.parse_args(["hf", "--checkpoint", "c", "--out", "o", "--kv", "bf16"])
    print("oracle --kv parses with the emulation's formats; hf has no --kv (expected)")
