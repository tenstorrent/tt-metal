# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

# pack.py

import array
import struct


def pack_bfp16(torch_tensor):
    return array.array(
        "B",
        b"".join(
            struct.pack("<f", value.item())[2:] for value in torch_tensor.view(-1)
        ),
    ).tolist()


def pack_fp16(torch_tensor):
    return array.array(
        "B",
        b"".join(struct.pack("<e", value.item()) for value in torch_tensor.view(-1)),
    ).tolist()


def pack_fp32(torch_tensor):
    return array.array(
        "B",
        b"".join(struct.pack("<f", value.item()) for value in torch_tensor.view(-1)),
    ).tolist()


def pack_int32(torch_tensor):
    return array.array(
        "B", b"".join(struct.pack("<i", value) for value in torch_tensor)
    ).tolist()


def pack_uint32(torch_tensor):
    return array.array(
        "B", b"".join(struct.pack("<I", value) for value in torch_tensor)
    ).tolist()


def pack_uint16(torch_tensor):
    return array.array(
        "B", b"".join(struct.pack("<H", value) for value in torch_tensor)
    ).tolist()


def pack_int8(torch_tensor):
    return array.array(
        "B", b"".join(struct.pack("<b", value) for value in torch_tensor)
    ).tolist()


def pack_uint8(torch_tensor):
    return array.array(
        "B", b"".join(struct.pack("<B", value) for value in torch_tensor)
    ).tolist()


def float_to_bfp8_block(block):
    def bfloat16_to_binary(value):
        float_value = struct.unpack("<I", struct.pack("<f", value))[0]
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
