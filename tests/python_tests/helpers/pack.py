# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

# pack.py

import struct


def flatten_list(sublists):
    return [item for sublist in sublists for item in sublist]


def int_to_bytes_list(n):
    return [(n >> (8 * i)) & 0xFF for i in range(3, -1, -1)]


def fp32_to_bytes(number):
    number_unpacked = struct.unpack("!I", struct.pack("!f", number))[0]
    return int_to_bytes_list(number_unpacked)


def pack_bfp16(torch_tensor):
    def bfloat16_to_bytes(number):
        number_unpacked = struct.unpack("!I", struct.pack("!f", number))[0]
        res_masked = number_unpacked & 0xFFFF0000
        return int_to_bytes_list(res_masked)

    packed_bytes = []
    for i in range(0, len(torch_tensor), 2):
        half1 = bfloat16_to_bytes(torch_tensor[i])
        half2 = bfloat16_to_bytes(torch_tensor[i + 1])
        packed_bytes.extend([half1[0:2][::-1], half2[0:2][::-1]])
    return flatten_list(packed_bytes)


def pack_fp16(torch_tensor):
    def float16_to_bytes(value):
        packed_bytes = struct.pack("<e", value)
        return list(packed_bytes)

    packed_bytes = []
    for i in range(0, len(torch_tensor), 2):
        half1 = float16_to_bytes(torch_tensor[i])
        half2 = float16_to_bytes(torch_tensor[i + 1])
        packed_bytes.extend([half1[0:2], half2[0:2]])
    return flatten_list(packed_bytes)


def pack_fp32(torch_tensor):
    def fp32_to_bytes(number):
        return list(struct.pack("<f", number))

    packed_bytes = [None] * len(torch_tensor)
    for i in range(len(torch_tensor)):
        packed_bytes[i] = fp32_to_bytes(torch_tensor[i])
    return flatten_list(packed_bytes)


def pack_int32(torch_tensor):
    def int32_to_bytes(number):
        return list(struct.pack("<I", number))

    packed_bytes = [None] * len(torch_tensor)
    for i in range(len(torch_tensor)):
        packed_bytes[i] = int32_to_bytes(torch_tensor[i])
    return flatten_list(packed_bytes)


def float_to_bfp8_block(block):
    def bfloat16_to_binary(value):
        float_value = struct.unpack("!I", struct.pack("!f", value))[0]
        bfloat16_value = (float_value & 0xFFFF0000) >> 16
        return f"{(bfloat16_value >> 8) & 0xFF:08b}{bfloat16_value & 0xFF:08b}"

    exponents = []
    mantissas = []
    signs = []
    max_exponent = -float("inf")

    for value in block:
        binary_str = bfloat16_to_binary(value)
        sign = binary_str[0]
        signs.append(int(sign, 2))
        exponent = int(binary_str[1:9], 2)
        mantissa = binary_str[9:-1]  # remove last
        mantissa = "1" + mantissa  ## add 1
        exponents.append(exponent)
        mantissas.append(mantissa)
        max_exponent = max(max_exponent, exponent)

    shared_exponent = max_exponent

    mantissas_explicit = [int(mantissa, 2) for mantissa in mantissas]

    bfp8_mantissas = []
    for i in range(len(block)):
        exponent_delta = shared_exponent - exponents[i]
        mantissa = mantissas_explicit[i] >> exponent_delta
        mantissa = (signs[i] << 7) | mantissa
        bfp8_mantissas.append(mantissa)

    return shared_exponent, bfp8_mantissas


def pack_bfp8_b(tensor, block_size=16):
    flattened_tensor = tensor.flatten()
    num_blocks = len(flattened_tensor) // block_size

    exponents = []
    mantissas = []

    for i in range(num_blocks):
        block = flattened_tensor[i * block_size : (i + 1) * block_size]
        shared_exponent, bfp8_mantissas = float_to_bfp8_block(block)
        exponents.append(shared_exponent)
        mantissas.extend(bfp8_mantissas)

    return exponents + mantissas
