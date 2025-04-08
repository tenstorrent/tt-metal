# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

# unpack.py

import struct
import torch
from .utils import reverse_endian_chunk
from .format_config import DataFormat


def int_to_bytes_list(n):
    return [(n >> (24 - i * 8)) & 0xFF for i in range(4)]


def unpack_fp16(packed_list, unpack_src, pack_dst):
    def bytes_to_float16(byte_list):
        bytes_data = bytes(byte_list[:2])
        unpacked_value = struct.unpack(">e", bytes_data)[0]
        return unpacked_value

    limited_packed_list = packed_list[:2048]
    result = [
        bytes_to_float16(limited_packed_list[i : i + 2])
        for i in range(0, len(limited_packed_list), 2)
    ]

    # Patch Up! Fixes incorrect reading of numbers in L1:
    # When input format i.e `unpack_src` is BFP8_b but the result is packed into a different format then consecutive pairs of numbers are inverted in L1.
    # Instead of being placed as (a,b,c,d,e,f,...) in L1, they are placed as (b,a,d,c,f,e,...).
    # This caused the test to fail as the results were correctly computed but read incorrectly.
    # The loop reinverts the numbers back to their correct positions in order to read them properly and pass the test as expected.
    if unpack_src == DataFormat.Bfp8_b and pack_dst != unpack_src:
        for i in range(0, len(result), 2):
            result[i], result[i + 1] = result[i + 1], result[i]
    return result


def unpack_bfp16(packed_list, unpack_src, pack_dst):
    def bytes_to_bfloat16(byte_list):
        bytes_data = bytes(byte_list[:2] + [0, 0])  # Ensure we include padding
        unpacked_value = struct.unpack(">f", bytes_data)[0]
        return unpacked_value

    limited_packed_list = packed_list[:2048]
    result = [
        bytes_to_bfloat16(limited_packed_list[i : i + 2])
        for i in range(0, len(limited_packed_list), 2)
    ]

    if unpack_src == pack_dst:
        for i in range(0, len(result), 2):
            result[i], result[i + 1] = result[i + 1], result[i]

    # Patch Up! Fixes incorrect reading of numbers in L1:
    # When input format i.e `unpack_src` is BFP8_b but the result is packed into a different format then consecutive pairs of numbers are inverted in L1.
    # Instead of being placed as (a,b,c,d,e,f,...) in L1, they are placed as (b,a,d,c,f,e,...).
    # This caused the test to fail as the results were correctly computed but read incorrectly.
    # The loop reinverts the numbers back to their correct positions in order to read them properly and pass the test as expected.
    if unpack_src == DataFormat.Bfp8_b and pack_dst != unpack_src:
        for i in range(0, len(result), 2):
            result[i], result[i + 1] = result[i + 1], result[i]
    return result


def unpack_float32(packed_list):
    def bytes_to_float32(byte_list):
        bytes_data = bytes(byte_list)
        unpacked_value = struct.unpack(">f", bytes_data)[0]
        return unpacked_value

    return [
        bytes_to_float32(packed_list[i : i + 4]) for i in range(0, len(packed_list), 4)
    ]


def unpack_int32(packed_list):
    def bytes_to_int32(byte_list):
        bytes_data = bytes(byte_list)
        unpacked_value = struct.unpack(">I", bytes_data)[0]
        return unpacked_value

    return [
        bytes_to_int32(packed_list[i : i + 4]) for i in range(0, len(packed_list), 4)
    ]


def bfp8_to_float_block(exponent, bfp8_mantissas, unpacked_bfp8):
    # Bug fix and improvement:
    # 1. Caching: If the (exponent, mantissa) pair is already processed, the precomputed value is reused.
    # 2. Sign and Fractional Calculation: The sign bit is extracted, and the fractional part is calculated by iterating
    #    over the mantissa bits, adding `1 / (2 ** i)` for each '1' bit.
    # 3. Exponent Scaling: The final value is scaled by `2^exponent` and adjusted by the sign bit.
    # 4. Efficient Storage: The computed value is stored in `unpacked_bfp8` for future use.

    bfloat16_values = []
    exponent = exponent - 127

    for mantissa in bfp8_mantissas:
        if (exponent, mantissa) in unpacked_bfp8:
            bfloat16_values.append(unpacked_bfp8[(exponent, mantissa)])
            continue

        sign_mantissa = str(format(mantissa, "08b"))
        # Extract the sign bit (most significant bit)
        sign = int(sign_mantissa[0], 2)
        # Get the remaining bits which represent the fractional part of the mantissa
        mantissa_value = sign_mantissa[1:]
        # Changed computation of mantissa to fix , accumulate fractional value
        fract_value = 0.0
        for i in range(len(mantissa_value)):
            # If the bit is '1', add the corresponding fractional value to fract_value
            if mantissa_value[i] == "1":
                fract_value += 1 / (2 ** (i))

        bfloat16_values.append(((-1.0) ** sign) * (2**exponent) * (fract_value))

        unpacked_bfp8[(exponent, mantissa)] = (
            ((-1.0) ** sign) * (2**exponent) * (fract_value)
        )

    return bfloat16_values


def unpack_bfp8_b(bfp8_block, unpack_src, pack_dst, sfpu=False):

    if not sfpu:
        exponents = bfp8_block[:64]
        mantissas = bfp8_block[64:]
    else:
        exponents = bfp8_block[:16]
        mantissas = bfp8_block[16:272]
    reversed_exponents = reverse_endian_chunk(exponents)

    unpacked_bfp8 = {}

    bfloat16_values = []
    for i in range(len(reversed_exponents)):
        exponent = reversed_exponents[i]
        bfp8_mantissas = mantissas[i * 16 : (i + 1) * 16]
        reversed_sign_mantissa = reverse_endian_chunk(bfp8_mantissas)
        block_bfloat16_values = bfp8_to_float_block(
            exponent, reversed_sign_mantissa, unpacked_bfp8
        )
        bfloat16_values.extend(block_bfloat16_values)

    # Patch Up! Fixes incorrect reading of numbers in L1:
    # When input source i.e `unpack_src` is not BFP8_B, but it is packed as BFP8_B then consecutive pairs of numbers are inverted and placed in L1.
    # Instead of being placed as (a,b,c,d,e,f,...) in L1, they are placed as (b,a,d,c,f,e,...).
    # This caused the test to fail as the results were correctly computed but read from L1 incorrectly.
    # The loop reinverts the numbers back to their correct positions in order to read them properly and pass the test as expected.
    if unpack_src != pack_dst:
        for i in range(0, len(bfloat16_values), 2):
            bfloat16_values[i], bfloat16_values[i + 1] = (
                bfloat16_values[i + 1],
                bfloat16_values[i],
            )

    return torch.tensor(bfloat16_values, dtype=torch.bfloat16)
