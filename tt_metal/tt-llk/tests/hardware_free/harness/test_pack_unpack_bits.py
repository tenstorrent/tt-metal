# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Bit-level contract of the L1 packers and unpackers.

The device reads and writes these bytes, so the encoding is fixed to the bit: which
NaN pattern a NaN packs to, which way a value between two float16 rounds, how a
sign-magnitude integer is laid out. A float cast that is equal for every ordinary
value can still differ on exactly those, so they are pinned here by value.
"""

import hashlib
import struct

import pytest
import torch
from helpers import pack, unpack


def _fp32(patterns):
    """float32 tensor with the given bit patterns."""
    return torch.tensor(
        [p - (1 << 32) if p >> 31 else p for p in patterns], dtype=torch.int32
    ).view(torch.float32)


def _fp16(patterns):
    """float16 tensor with the given bit patterns."""
    return torch.tensor(
        [p - (1 << 16) if p >> 15 else p for p in patterns], dtype=torch.int16
    ).view(torch.float16)


def _double_bits(values):
    return [struct.pack("<d", value).hex() for value in values]


# quiet +/-, signalling +/- with the smallest payload, +/- with every payload bit set, and two
# payloads that differ in the bits a 16-bit format keeps.
FP32_NANS = [
    0x7FC00000,
    0xFFC00000,
    0x7F800001,
    0xFF800001,
    0x7FFFFFFF,
    0xFFFFFFFF,
    0x7F802000,
    0xFFABCDEF,
]


def test_bfloat16_nan_is_the_quiet_nan_with_the_sign_of_the_input():
    assert pack.pack_bfp16(_fp32(FP32_NANS)).hex() == "c07fc0ff" * 4


def test_float16_nan_keeps_the_top_payload_bits():
    # The lowest payload bit is set where the kept bits are all zero (0x7F800001 -> 0x7C01).
    assert pack.pack_fp16(_fp32(FP32_NANS)).hex() == "007e00fe017c01fcff7fffff017c5efd"


def test_float16_nan_widens_to_float32_with_its_payload():
    assert (
        pack.pack_fp32(_fp16([0x7C01, 0xFC01, 0x7E00, 0x3C00])).hex()
        == "0020807f002080ff0000c07f0000803f"
    )


FINITE = [
    1.0,
    -2.5,
    65504.0,  # the largest finite float16
    65519.996,  # just under the float16 overflow threshold
    65520.0,  # the threshold: rounds to infinity
    -70000.0,
    1e-8,
    0.0,
    -0.0,
    float("inf"),
    1.0009765625 + 2**-12,  # a float16 step and a half above 1.0
]


def test_finite_float32_rounds_to_nearest_even():
    values = torch.tensor(FINITE, dtype=torch.float32)
    assert (
        pack.pack_bfp16(values).hex() == "803f20c080478047804789c72c3200000080807f803f"
    )
    assert (
        pack.pack_fp16(values).hex() == "003c00c1ff7bff7b007c00fc000000000080007c013c"
    )


def test_float64_rounds_to_float16_in_one_step():
    # Each of these is a float16 tie once rounded to float32, so a narrowing through
    # float32 goes to the even neighbour instead of the nearest one.
    values = torch.tensor(
        [
            1 + 2**-11 + 2**-40,  # just above the tie between 1.0 and the next float16
            1 + 2**-11,  # exactly the tie: to even
            1 + 3 * 2**-11 - 2**-40,  # just below the next tie
            65519.999999,  # below the overflow threshold, at it in float32
            65520.0,
            2**-25 * (1 + 2**-30),  # just above half the smallest subnormal
            2**-25,  # exactly half of it: to even, which is zero
            -(1 + 2**-11 + 2**-40),
        ],
        dtype=torch.float64,
    )
    assert pack.pack_fp16(values).hex() == "013c003c013cff7b007c0100000001bc"


INTEGERS = torch.tensor(
    [0, 1, -1, 127, -127, -128, 300, -300, 32767, -32768, 2**31 - 1, -(2**31), 70000],
    dtype=torch.int64,
)


@pytest.mark.parametrize(
    "packer, sign_magnitude, twos_complement",
    [
        (
            pack.pack_int32,
            "0000000001000000010000807f0000007f000080800000802c0100002c010080"
            "ff7f000000800080ffffff7fffffffff70110100",
            "0000000001000000ffffffff7f00000081ffffff80ffffff2c010000d4feffff"
            "ff7f00000080ffffffffff7f0000008070110100",
        ),
        (
            pack.pack_int16,
            "0000010001807f007f8080802c012c81ff7fffffff7fffffff7f",
            "00000100ffff7f0081ff80ff2c01d4feff7f0080ffff00007011",
        ),
        (pack.pack_int8, "0001817fffff7fff7fff7fff7f", "0001ff7f81802cd4ff00ff0070"),
    ],
    ids=["int32", "int16", "int8"],
)
def test_signed_integers(packer, sign_magnitude, twos_complement):
    # Sign-magnitude clips to the symmetric range first; two's complement keeps the low bits.
    assert packer(INTEGERS).hex() == sign_magnitude
    assert packer(INTEGERS, twos_complement=True).hex() == twos_complement


def test_unsigned_integers():
    assert (
        pack.pack_uint32(torch.tensor([0, 1, 2**32 - 1, 2**31])).hex()
        == "0000000001000000ffffffff00000080"
    )
    assert (
        pack.pack_uint16(torch.tensor([0, 1, 65535, 32768], dtype=torch.int32)).hex()
        == "00000100ffff0080"
    )
    assert (
        pack.pack_uint8(torch.tensor([0, 1, 255, 128], dtype=torch.uint8)).hex()
        == "0001ff80"
    )


def test_four_bit_integers():
    signed = torch.tensor([0, 1, -1, 7, -7, -8, 9, -20], dtype=torch.int8)
    unsigned = torch.tensor([0, 1, 15, 16, 7, 200, 3, 2], dtype=torch.uint8)
    assert pack.pack_int4(signed).hex() == "1079fff7"
    assert pack.pack_uint4(unsigned).hex() == "10fff723"
    # Two rows of 16 datums per face: every face is padded to a 16-byte L1 boundary.
    faces = torch.arange(-8, 8, dtype=torch.int8).repeat(4)
    assert pack.pack_int4(faces, face_r_dim=2).hex() == "ffdebc9a10325476" * 4
    with pytest.raises(  # allow-pytest.raises: no expect_error fixture in LLK suite
        ValueError, match="even datum count"
    ):
        pack.pack_int4(signed[:3])


def test_tensor_bytes_follows_element_order():
    values = torch.arange(6, dtype=torch.int16).reshape(2, 3)
    assert pack.tensor_bytes(values) == struct.pack("<6h", *range(6))
    assert pack.tensor_bytes(values.t()) == struct.pack("<6h", 0, 3, 1, 4, 2, 5)
    assert pack.tensor_bytes(torch.tensor(1.5)) == struct.pack("<f", 1.5)
    assert pack.tensor_bytes(torch.empty(0)) == b""


def test_unpacked_floats_keep_nan_payloads():
    # float16: both signalling NaNs keep their quiet bit clear.
    assert _double_bits(
        unpack.unpack_fp16(bytes.fromhex("017c01fc007e003c0080ff7f"))
    ) == [
        "000000000004f07f",
        "000000000004f0ff",
        "000000000000f87f",
        "000000000000f03f",
        "0000000000000080",
        "0000000000fcff7f",
    ]
    assert _double_bits(
        unpack.unpack_bfp16(bytes.fromhex("817f81ffc07f803f0080ffff"))
    ) == [
        "000000000020f87f",
        "000000000020f8ff",
        "000000000000f87f",
        "000000000000f03f",
        "0000000000000080",
        "0000000000e0ffff",
    ]


WORDS = bytes.fromhex("00000080ffffffff0100008005000000")


def test_unpacked_integers():
    assert unpack.unpack_int32(WORDS) == [0, -2147483647, -1, 5]
    assert unpack.unpack_int32(WORDS, twos_complement=True) == [
        -2147483648,
        -1,
        -2147483647,
        5,
    ]
    assert unpack.unpack_uint32(WORDS) == [2147483648, 4294967295, 2147483649, 5]
    assert unpack.unpack_int16(WORDS) == [0, 0, -32767, -32767, 1, 0, 5, 0]
    assert unpack.unpack_int16(WORDS, twos_complement=True) == [
        0,
        -32768,
        -1,
        -1,
        1,
        -32768,
        5,
        0,
    ]
    assert unpack.unpack_uint16(WORDS) == [0, 32768, 65535, 65535, 1, 32768, 5, 0]
    assert unpack.unpack_int8(WORDS[:8]) == [0, 0, 0, 0, -127, -127, -127, -127]
    assert unpack.unpack_int8(WORDS[:8], twos_complement=True) == [
        0,
        0,
        0,
        -128,
        -1,
        -1,
        -1,
        -1,
    ]
    assert unpack.unpack_uint8(WORDS[:8]) == [0, 0, 0, 128, 255, 255, 255, 255]
    assert all(type(v) is int for v in unpack.unpack_uint32(WORDS))


def test_unpackers_accept_lists_and_empty_buffers():
    assert unpack.unpack_int32(list(WORDS)) == unpack.unpack_int32(WORDS)
    assert unpack.unpack_fp32(b"") == []
    assert unpack.unpack_uint8([]) == []
    with pytest.raises(  # allow-pytest.raises: no expect_error fixture in LLK suite
        ValueError
    ):
        unpack.unpack_fp32(b"\x01\x02\x03")


def _digest(data) -> str:
    return hashlib.sha256(bytes(data)).hexdigest()[:16]


@pytest.mark.parametrize(
    "unpacker, magnitude_bits, digest, sampled",
    [
        (
            unpack.unpack_bfp8_b,
            7,
            "25b7998cb8bdcce9",
            [11, 48, 182, -32768, -16394, 16406, 32620, 32640, 32640],
        ),
        (
            unpack.unpack_bfp4_b,
            3,
            "7740270ea1cab4f2",
            [-32720, 0, 96, 96, 16192, -16448, 32576, 32512, 32512],
        ),
        (
            unpack.unpack_bfp2_b,
            1,
            "55ae9da6d5ebc6c3",
            [-32704, -32768, -32640, -32640, -16512, -16384, -128, 32640, -32768],
        ),
    ],
    ids=["bfp8", "bfp4", "bfp2"],
)
def test_block_float_unpack_across_the_exponent_range(
    unpacker, magnitude_bits, digest, sampled
):
    # Shared exponents from the smallest to the largest: the scale 2 ** (exponent - 127) is
    # outside the float32 range at both ends, and the result must still be exact.
    exponents = [0, 1, 126, 127, 128, 133, 254, 255] + [127] * 8
    mantissa_bytes = 256 * (magnitude_bits + 1) // 8
    block = bytes(exponents) + bytes((i * 37 + 11) % 256 for i in range(mantissa_bytes))
    values = unpacker(block, num_faces=1, face_r_dim=16)
    assert values.dtype == torch.bfloat16 and values.shape == (256,)
    bits = values.view(torch.int16)
    assert bits[[0, 1, 16, 17, 48, 64, 112, 113, 127]].tolist() == sampled
    assert _digest(pack.tensor_bytes(values)) == digest


@pytest.mark.parametrize(
    "packer, length, digest, dense_length, dense_digest",
    [
        (pack.pack_bfp8_b, 1088, "747f17de96539dfc", 3264, "c1ff05eeaff0e039"),
        (pack.pack_bfp4_b, 576, "9a41494d9a68ecd6", 1728, "a30f939b37c421c3"),
        (pack.pack_bfp2_b, 320, "9a3cde4377fc75f1", 960, "230e09662187b82e"),
    ],
    ids=["bfp8", "bfp4", "bfp2"],
)
def test_block_float_pack(packer, length, digest, dense_length, dense_digest):
    tile = torch.tensor(
        [
            ((i * 7919) % 2001 - 1000) / 128.0 * (2.0 ** ((i // 16) % 5 - 2))
            for i in range(1024)
        ],
        dtype=torch.float32,
    )
    packed = packer(tile)
    assert len(packed) == length and _digest(packed) == digest
    dense = pack.pack_dense_tiles(torch.cat([tile, -tile, tile * 3]), 3, packer, 4, 16)
    assert len(dense) == dense_length and _digest(dense) == dense_digest
