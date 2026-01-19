# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

# unpack.py

import ml_dtypes
import numpy as np
import torch
from helpers.format_config import MXFP8_BLOCK_SIZE, DataFormat

from .llk_params import format_dict, format_tile_sizes


def unpack_fp16(packed_list):
    return np.frombuffer(bytes(packed_list), dtype=np.float16).tolist()


def unpack_bfp16(packed_list):
    return (
        np.frombuffer(bytes(packed_list), dtype=ml_dtypes.bfloat16)
        .astype(np.float32)
        .tolist()
    )


def unpack_fp32(packed_list):
    return np.frombuffer(bytes(packed_list), dtype=np.float32).tolist()


def unpack_int32(packed_list):
    return np.frombuffer(bytes(packed_list), dtype=np.int32).tolist()


def unpack_uint32(packed_list):
    return np.frombuffer(bytes(packed_list), dtype=np.uint32).tolist()


def unpack_uint16(packed_list):
    return np.frombuffer(bytes(packed_list), dtype=np.uint16).tolist()


def unpack_int8(packed_list):
    return np.frombuffer(bytes(packed_list), dtype=np.int8).tolist()


def unpack_uint8(packed_list):
    return np.frombuffer(bytes(packed_list), dtype=np.uint8).tolist()


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


def unpack_bfp8_b(bfp8_block, sfpu=False, num_faces=4):

    exponents_per_face = 16
    if not sfpu:
        exponents = bfp8_block[: exponents_per_face * num_faces]
        mantissas = bfp8_block[exponents_per_face * num_faces :]
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


# ============================================================================
# MX (Microscaling) Format Support - OCP Specification
# ============================================================================


def _unpack_mxfp8(packed_bytes, fp8_dtype, num_faces=4):
    """
    Unpack MXFP8 format with layout: [all_scales][all_elements]

    Args:
        packed_bytes: List of bytes in [all scales][all elements] format
        fp8_dtype: ml_dtypes dtype (float8_e5m2 or float8_e4m3fn)
        num_faces: Number of faces (1, 2, or 4). Defaults to 4.

    Returns:
        torch.Tensor of bfloat16 values
    """
    num_scales = num_faces * 8
    num_blocks = num_faces * 8

    scales_e8m0 = packed_bytes[:num_scales]
    elements_bytes = packed_bytes[num_scales:]

    # Convert all elements to FP8 array using ml_dtypes
    fp8_array = np.frombuffer(bytes(elements_bytes), dtype=fp8_dtype)

    # Reshape into blocks: (num_blocks, 32)
    fp8_blocks = fp8_array[: num_blocks * MXFP8_BLOCK_SIZE].reshape(
        num_blocks, MXFP8_BLOCK_SIZE
    )

    # Vectorized scale decoding - decode all E8M0 scales at once
    scales_array = np.frombuffer(bytes(scales_e8m0), dtype=np.uint8)
    # Handle NaN case (255) and compute 2^(exponent) where exponent = value - 127
    scale_factors = np.where(
        scales_array == 255, np.nan, np.exp2(scales_array.astype(np.float32) - 127.0)
    )
    # Replace NaN and zero scales with 0
    scale_factors = np.where(
        np.isnan(scale_factors) | (scale_factors == 0), 0, scale_factors
    )

    # Scale blocks back to float32
    scaled_blocks = fp8_blocks.astype(np.float32) * scale_factors[:, np.newaxis]

    # Flatten and convert to bfloat16 tensor
    return torch.tensor(scaled_blocks.flatten(), dtype=torch.bfloat16)


def unpack_mxfp8r(packed_bytes, num_faces=4):
    """
    Unpack MXFP8R format (E5M2 variant) to bfloat16 tensor.

    Args:
        packed_bytes: Packed MX data in FULLY SEPARATED layout [all_scales][all_elements]
        num_faces: Number of faces to unpack (1, 2, or 4). Defaults to 4.

    Returns:
        torch.Tensor of bfloat16 values
    """
    return _unpack_mxfp8(packed_bytes, ml_dtypes.float8_e5m2, num_faces)


def unpack_mxfp8p(packed_bytes, num_faces=4):
    """
    Unpack MXFP8P format (E4M3 variant) to bfloat16 tensor.

    Args:
        packed_bytes: Packed MX data in FULLY SEPARATED layout [all_scales][all_elements]
        num_faces: Number of faces to unpack (1, 2, or 4). Defaults to 4.

    Returns:
        torch.Tensor of bfloat16 values
    """
    return _unpack_mxfp8(packed_bytes, ml_dtypes.float8_e4m3fn, num_faces)


_UNPACKERS = {
    DataFormat.Float16: unpack_fp16,
    DataFormat.Float16_b: unpack_bfp16,
    DataFormat.Float32: unpack_fp32,
    DataFormat.Int32: unpack_int32,
    DataFormat.UInt32: unpack_uint32,
    DataFormat.UInt16: unpack_uint16,
    DataFormat.Int8: unpack_int8,
    DataFormat.UInt8: unpack_uint8,
}


def unpack_res_tiles(
    packed_list,
    output_format: DataFormat,
    tile_count: int = 1,
    sfpu: bool = False,
    num_faces: int = 4,
    face_r_dim: int = 16,
):
    output_dtype = format_dict[output_format]

    # Calculate tile size and determine elements per tile needed
    tile_size = format_tile_sizes[output_format]  # Full tile size in bytes

    elements_per_tile_needed = output_format.num_bytes_per_tile(
        num_faces * face_r_dim * 16
    )

    total_elements_needed = tile_count * elements_per_tile_needed
    if total_elements_needed > len(packed_list):
        raise IndexError("Buffer access out of bounds")

    if output_format == DataFormat.Bfp8_b:
        unpack_func = unpack_bfp16 if sfpu else unpack_bfp8_b
    elif output_format == DataFormat.MxFp8R:
        unpack_func = unpack_mxfp8r
    elif output_format == DataFormat.MxFp8P:
        unpack_func = unpack_mxfp8p
    else:
        unpack_func = _UNPACKERS[output_format]

    unpacked_data = []

    # Write only values from the selected faces into unpacked_tile
    for tile in range(tile_count):
        # Both paths use byte-based indexing since tile_size and elements_per_tile_needed are in bytes
        start_idx = tile * tile_size
        end_idx = start_idx + elements_per_tile_needed
        tile_data = packed_list[start_idx:end_idx]

        if unpack_func == unpack_bfp8_b:
            unpacked_tile = unpack_func(tile_data, sfpu=sfpu, num_faces=num_faces)
        elif unpack_func in [unpack_mxfp8r, unpack_mxfp8p]:
            unpacked_tile = unpack_func(tile_data, num_faces=num_faces)
        else:
            unpacked_tile = unpack_func(tile_data)

        unpacked_data.extend(unpacked_tile)

    return torch.tensor(unpacked_data, dtype=output_dtype)
