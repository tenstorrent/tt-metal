# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import struct

import ml_dtypes
import numpy as np
import torch

from .format_config import (
    MXFP8_BLOCK_SIZE,
    MXFP8_E4M3_MAX_NORMAL,
    MXFP8_E5M2_MAX_NORMAL,
)


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


# ============================================================================
# MX (Microscaling) Format Support - OCP Specification
# ============================================================================


def _pack_mxfp8(tensor, fp8_dtype, element_max_normal, num_faces=4):
    """
    Internal helper to pack MXFP8 formats with FULLY SEPARATED layout.

    Layout (similar to BFP8_b): [all_scales][all_elements]
    - BFP8_b: [64 exponents][1024 mantissas]
    - MXFP8:  [32 scales][1024 elements]

    Uses ml_dtypes for FP8 element conversion and E8M0 scale encoding.

    Args:
        tensor: Input tensor (typically 1024 elements for full tile)
        fp8_dtype: ml_dtypes dtype (float8_e5m2 or float8_e4m3fn)
        element_max_normal: Maximum normal value for element format
        num_faces: Number of faces to pack (1, 2, or 4). Defaults to 4.

    Returns:
        List of packed bytes: [all scales][all elements]
    """
    # Convert to numpy and prepare data
    fp32_array = tensor.cpu().to(torch.float32).numpy().flatten()

    elements_per_face = 256
    elements_to_pack = elements_per_face * num_faces
    assert (
        len(fp32_array) >= elements_to_pack
    ), f"Tensor has {len(fp32_array)} elements, need {elements_to_pack} for {num_faces} face(s)"

    fp32_array = fp32_array[:elements_to_pack]

    # Reshape into blocks: (num_blocks, 32)
    num_blocks = len(fp32_array) // MXFP8_BLOCK_SIZE
    blocks = fp32_array[: num_blocks * MXFP8_BLOCK_SIZE].reshape(
        num_blocks, MXFP8_BLOCK_SIZE
    )

    # Vectorized scale encoding - calculate all scales at once
    max_abs_values = np.max(np.abs(blocks), axis=1)

    # Handle special cases: zero, nan, inf
    scale_ratio = max_abs_values / element_max_normal
    exponents = np.ceil(
        np.log2(scale_ratio, where=(scale_ratio > 0), out=np.zeros_like(scale_ratio))
    )

    # Apply special case handling
    exponents = np.where(
        (max_abs_values == 0) | np.isnan(max_abs_values),
        0,  # Neutral scale (2^0 = 1) for zero/nan
        np.where(np.isinf(max_abs_values), 127, exponents),  # Max scale for inf
    )

    # Clamp to E8M0 range [-127, 127] and add bias
    scales_e8m0_array = np.clip(exponents, -127, 127).astype(np.int32) + 127
    scales_e8m0 = scales_e8m0_array.astype(np.uint8).tolist()

    # Vectorized scale decoding for applying to blocks
    scale_factors = np.where(
        scales_e8m0_array == 255,
        np.nan,
        np.exp2(scales_e8m0_array.astype(np.float32) - 127.0),
    )

    # Scale blocks and convert to FP8
    scaled_blocks = blocks / scale_factors[:, np.newaxis]
    fp8_blocks = scaled_blocks.astype(fp8_dtype)

    # FULLY SEPARATED layout: all scales first, then all elements
    # Convert FP8 blocks to list of bytes (integers 0-255)
    fp8_bytes = list(fp8_blocks.tobytes())
    return scales_e8m0 + fp8_bytes


def pack_mxfp8r(tensor, num_faces=4):
    """
    Pack tensor into MXFP8R format (MXFP8 E5M2 variant).

    MXFP8 uses 32-element blocks per OCP MX spec, each with:
    - 1 shared E8M0 scale (8 bits)
    - 32 × float8_e5m2 elements (8 bits each)

    Element format E5M2:
    - 1 sign bit, 5 exponent bits (bias=15), 2 mantissa bits
    - Max normal: ±57,344
    - Has Inf and NaN support

    Args:
        tensor: Input tensor (typically 1024 elements for full tile)
        num_faces: Number of faces to pack (1, 2, or 4). Defaults to 4.

    Returns:
        List of packed bytes in FULLY SEPARATED layout: [all_scales][all_elements]
        Layout: [32 scales (1 per block)][1024 FP8 elements]
    """
    return _pack_mxfp8(tensor, ml_dtypes.float8_e5m2, MXFP8_E5M2_MAX_NORMAL, num_faces)


def pack_mxfp8p(tensor, num_faces=4):
    """
    Pack tensor into MXFP8P format (MXFP8 E4M3 variant).

    MXFP8 uses 32-element blocks per OCP MX spec, each with:
    - 1 shared E8M0 scale (8 bits)
    - 32 × float8_e4m3fn elements (8 bits each)

    Element format E4M3:
    - 1 sign bit, 4 exponent bits (bias=7), 3 mantissa bits
    - Max normal: ±448
    - No Inf support, NaN represented by 0bS1111111

    Args:
        tensor: Input tensor (typically 1024 elements for full tile)
        num_faces: Number of faces to pack (1, 2, or 4). Defaults to 4.

    Returns:
        List of packed bytes in FULLY SEPARATED layout: [all_scales][all_elements]
        Layout: [32 scales (1 per block)][1024 FP8 elements]
    """
    return _pack_mxfp8(
        tensor, ml_dtypes.float8_e4m3fn, MXFP8_E4M3_MAX_NORMAL, num_faces
    )
