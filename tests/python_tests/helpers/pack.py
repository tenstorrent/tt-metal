# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import struct

import ml_dtypes
import numpy as np
import torch


def pack_bfp16(torch_tensor):
    fp32_array = torch_tensor.cpu().to(torch.float32).numpy()
    bfp16_array = fp32_array.astype(ml_dtypes.bfloat16)
    return bfp16_array.tobytes()


def pack_fp16(torch_tensor):
    return torch_tensor.cpu().numpy().astype(np.float16).tobytes()


def pack_fp32(torch_tensor):
    return torch_tensor.cpu().numpy().astype(np.float32).tobytes()


def pack_int32(torch_tensor):
    return torch_tensor.cpu().numpy().astype(np.int32).tobytes()


def pack_uint32(torch_tensor):
    return torch_tensor.cpu().numpy().astype(np.uint32).tobytes()


def pack_uint16(torch_tensor):
    return torch_tensor.cpu().numpy().astype(np.uint16).tobytes()


def pack_int8(torch_tensor):
    return torch_tensor.cpu().numpy().astype(np.int8).tobytes()


def pack_uint8(torch_tensor):
    return torch_tensor.cpu().numpy().astype(np.uint8).tobytes()


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


def pack_bfp8_b(tensor, block_size=16, num_faces=4):
    """Pack tensor into BFP8_b format.

    BFP8_b uses 16-element blocks, each with a shared exponent and 8-bit mantissas.
    Only the first (256 * num_faces) elements are packed.

    Args:
        tensor: Input tensor (typically 1024 elements for full tile)
        block_size: Elements per block (always 16 for BFP8_b)
        num_faces: Number of faces to pack (1, 2, or 4)

    Returns:
        List of packed bytes: [exponents...] + [mantissas...]
    """
    flattened_tensor = tensor.flatten()

    # Only pack the first (256 * num_faces) elements
    elements_to_pack = 256 * num_faces
    assert (
        len(flattened_tensor) >= elements_to_pack
    ), f"Tensor has {len(flattened_tensor)} elements, but need at least {elements_to_pack} for {num_faces} face(s)"
    flattened_tensor = flattened_tensor[:elements_to_pack]

    num_blocks = len(flattened_tensor) // block_size

    exponents = []
    mantissas = []

    for i in range(num_blocks):
        block = flattened_tensor[i * block_size : (i + 1) * block_size]
        shared_exponent, bfp8_mantissas = float_to_bfp8_block(block)
        exponents.append(shared_exponent)
        mantissas.extend(bfp8_mantissas)

    return exponents + mantissas
