# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

# unpack.py

import struct

import torch


def unpack_fp16(packed_list, unpack_src, pack_dst):
    return [val[0] for val in struct.iter_unpack("<e", bytes(packed_list))]


def unpack_bfp16(packed_list, unpack_src, pack_dst):
    # Step 1: Promote each 2-byte bfloat16 to 4-byte float32
    # Place bfloat16 in high 2 bytes (little-endian)
    padded_bytes = bytearray()
    for i in range(0, len(packed_list), 2):
        hi, lo = packed_list[i], packed_list[i + 1]
        padded_bytes.extend([0x00, 0x00, hi, lo])  # float32 = [LSB, ..., MSB]

    # Use iter_unpack with "<f" to read float32
    return [val[0] for val in struct.iter_unpack("<f", padded_bytes)]


def unpack_fp32(packed_list):
    return [val[0] for val in struct.iter_unpack("<f", bytes(packed_list))]


def unpack_int32(packed_list):
    return [val[0] for val in struct.iter_unpack("<i", bytes(packed_list))]


def unpack_uint32(packed_list):
    return [val[0] for val in struct.iter_unpack("<I", bytes(packed_list))]


def unpack_uint16(packed_list):
    return [val[0] for val in struct.iter_unpack("<H", bytes(packed_list))]


def unpack_int8(packed_list):
    return [val[0] for val in struct.iter_unpack("<b", bytes(packed_list))]


def unpack_uint8(packed_list):
    return [val[0] for val in struct.iter_unpack("<B", bytes(packed_list))]


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

    unpacked_bfp8 = {}

    bfloat16_values = []
    for i in range(len(exponents)):
        exponent = exponents[i]
        bfp8_mantissas = mantissas[i * 16 : (i + 1) * 16]
        block_bfloat16_values = bfp8_to_float_block(
            exponent, bytes(bfp8_mantissas), unpacked_bfp8
        )
        bfloat16_values.extend(block_bfloat16_values)

    return torch.tensor(bfloat16_values, dtype=torch.bfloat16)
