# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

# pack.py

import torch
import struct


def flatten_list(sublists):
    return [item for sublist in sublists for item in sublist]


def int_to_bytes_list(n):
    binary_str = bin(n)[2:].zfill(32)
    return [int(binary_str[i : i + 8], 2) for i in range(0, 32, 8)]


def float16_to_bytes(value):
    float16_value = torch.tensor(value, dtype=torch.float16)
    packed_bytes = struct.pack(">e", float16_value.item())
    return list(packed_bytes) + [0] * (4 - len(packed_bytes))


def bfloat16_to_bytes(number):
    number_unpacked = struct.unpack("!I", struct.pack("!f", number))[0]
    res_masked = number_unpacked & 0xFFFF0000
    return int_to_bytes_list(res_masked)


def fp32_to_bytes(number):
    number_unpacked = struct.unpack("!I", struct.pack("!f", number))[0]
    return int_to_bytes_list(number_unpacked)


def int32_to_bytes(number):
    number = int(number)
    number_unpacked = struct.unpack("!I", struct.pack("!I", number))[0]
    return int_to_bytes_list(number_unpacked)


def bfloat16_to_binary(value):
    float_value = value.to(torch.float32).item()
    bfloat16_bytes = bfloat16_to_bytes(float_value)
    return f"{bfloat16_bytes[0]:08b}{bfloat16_bytes[1]:08b}"


def pack_bfp16(torch_tensor):
    packed_bytes = []
    for i in range(0, len(torch_tensor), 2):
        half1 = bfloat16_to_bytes(torch_tensor[i])
        half2 = bfloat16_to_bytes(torch_tensor[i + 1])
        packed_bytes.extend([half1[0:2][::-1], half2[0:2][::-1]][::-1])
    return flatten_list(packed_bytes)


def pack_fp16(torch_tensor):
    packed_bytes = []
    for i in range(0, len(torch_tensor), 2):
        half1 = float16_to_bytes(torch_tensor[i])
        half2 = float16_to_bytes(torch_tensor[i + 1])
        packed_bytes.extend([half1[0:2][::-1], half2[0:2][::-1]][::-1])
    return flatten_list(packed_bytes)


def pack_fp32(torch_tensor):
    packed_bytes = []
    for i in range(0, len(torch_tensor)):
        packed_bytes.append(fp32_to_bytes(torch_tensor[i])[::-1])
    return flatten_list(packed_bytes)


def pack_int32(torch_tensor):
    packed_bytes = []
    for i in range(0, len(torch_tensor)):
        packed_bytes.append(int32_to_bytes(torch_tensor[i])[::-1])
    return flatten_list(packed_bytes)


def float_to_bfp8_block(block):
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
    blocks = [
        flattened_tensor[i * block_size : (i + 1) * block_size]
        for i in range(num_blocks)
    ]

    exponents = []
    mantissas = []

    for block in blocks:
        shared_exponent, bfp8_mantissas = float_to_bfp8_block(block)
        exponents.append(shared_exponent)
        mantissas.extend(bfp8_mantissas)

    bfp8_result = exponents + mantissas

    return bfp8_result
