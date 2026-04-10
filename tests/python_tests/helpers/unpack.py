# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

# unpack.py

import ml_dtypes
import numpy as np
import torch
from helpers.format_config import (
    MXFP8_BLOCK_SIZE,
    MXFP8_SRCS_SLICE_32B_PACKED_BYTE_LEN,
    MXFP8_SRCS_SLICE_PACKED_BYTE_LEN,
    DataFormat,
)

from .llk_params import format_dict, format_tile_sizes
from .tile_constants import (
    FACE_C_DIM,
    MAX_FACE_R_DIM,
    MAX_NUM_FACES,
    MIN_BFP_EXPONENTS,
    SRCS_SLICE_32B_ROW_DIM,
    SRCS_SLICE_ROW_DIM,
)


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
    # INT32 uses sign-magnitude format in hardware (not two's complement)
    # Format: bit 31 = sign, bits 30:0 = magnitude
    uint32_array = np.frombuffer(bytes(packed_list), dtype=np.uint32)
    sign = (uint32_array & 0x80000000).astype(bool)
    magnitude = (uint32_array & 0x7FFFFFFF).astype(np.int64)
    return np.where(sign, -magnitude, magnitude).astype(np.int32).tolist()


def unpack_uint32(packed_list):
    return np.frombuffer(bytes(packed_list), dtype=np.uint32).tolist()


def unpack_int16(packed_list):
    # INT16 uses sign-magnitude format in hardware (not two's complement)
    # Format: bit 15 = sign, bits 14:0 = magnitude
    uint16_array = np.frombuffer(bytes(packed_list), dtype=np.uint16)
    sign = (uint16_array & 0x8000).astype(bool)
    magnitude = (uint16_array & 0x7FFF).astype(np.int16)
    return np.where(sign, -magnitude, magnitude).tolist()


def unpack_uint16(packed_list):
    return np.frombuffer(bytes(packed_list), dtype=np.uint16).tolist()


def unpack_fp8_e4m3(packed_list):
    return (
        np.frombuffer(bytes(packed_list), dtype=ml_dtypes.float8_e4m3fn)
        .astype(np.float32)
        .tolist()
    )


def unpack_int8(packed_list):
    # INT8 uses sign-magnitude format in hardware (not two's complement)
    # Format: bit 7 = sign, bits 6:0 = magnitude
    uint8_array = np.frombuffer(bytes(packed_list), dtype=np.uint8)
    sign = (uint8_array & 0x80).astype(bool)
    magnitude = (uint8_array & 0x7F).astype(np.int8)
    return np.where(sign, -magnitude, magnitude).tolist()


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


def unpack_bfp8_b(bfp8_block, sfpu=False, num_faces=4, face_r_dim=16):
    # Each BFP8 block is 16 elements with 1 shared exponent
    # Elements per face = face_r_dim * 16, so blocks per face = face_r_dim
    actual_exponents = face_r_dim * num_faces

    # Hardware requires minimum exponents for both unpacker and packer
    exponents_in_packed = max(actual_exponents, MIN_BFP_EXPONENTS)

    if not sfpu:
        # Read all exponents (including any padding)
        all_exponents = bfp8_block[:exponents_in_packed]
        mantissas = bfp8_block[exponents_in_packed:]
        # Only use the actual exponents (not padding)
        exponents = all_exponents[:actual_exponents]
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


def bfp4_to_float_block(exponent, bfp4_mantissas, unpacked_bfp4):
    bfloat16_values = []
    exp_adj = exponent - 127
    scale = 2.0 ** (exp_adj - 2)

    for mantissa in bfp4_mantissas:
        mag = mantissa & 0x7
        if mag == 0:
            bfloat16_values.append(0.0)
            unpacked_bfp4[(exp_adj, mantissa)] = 0.0
            continue

        key = (exp_adj, mantissa)
        cached = unpacked_bfp4.get(key)
        if cached is not None:
            bfloat16_values.append(cached)
            continue

        sign = -1.0 if mantissa & 0x8 else 1.0
        value = sign * mag * scale
        bfloat16_values.append(value)
        unpacked_bfp4[key] = value

    return bfloat16_values


def unpack_bfp4_b(bfp4_block, sfpu=False, num_faces=4, face_r_dim=16):
    actual_exponents = face_r_dim * num_faces
    exponents_in_packed = max(actual_exponents, MIN_BFP_EXPONENTS)

    if not sfpu:
        exponents = bfp4_block[:actual_exponents]
        packed_mantissas = bfp4_block[exponents_in_packed:]
    else:
        exponents = bfp4_block[:16]
        packed_mantissas = bfp4_block[16 : 16 + actual_exponents * 8]

    # Expand packed bytes into 4-bit datums using NumPy vectorized ops
    # Hardware BFP4_b convention: low nibble = first element, high nibble = second
    packed = np.frombuffer(packed_mantissas, dtype=np.uint8)
    low_nibbles = packed & 0x0F
    high_nibbles = (packed >> 4) & 0x0F
    mantissas = np.empty(len(packed) * 2, dtype=np.uint8)
    mantissas[0::2] = low_nibbles
    mantissas[1::2] = high_nibbles

    unpacked_bfp4 = {}

    bfloat16_values = []
    for i, exponent in enumerate(exponents):
        block_mantissas = mantissas[i * 16 : (i + 1) * 16]
        block_bfloat16_values = bfp4_to_float_block(
            exponent, block_mantissas, unpacked_bfp4
        )
        bfloat16_values.extend(block_bfloat16_values)

    return torch.tensor(bfloat16_values, dtype=torch.bfloat16)


# ============================================================================
# MX (Microscaling) Format Support - OCP Specification
# ============================================================================


def _align16(n: int) -> int:
    """Round *n* up to the next 16-byte boundary."""
    return (n + 15) & ~15


def _unpack_mxfp8(packed_bytes, fp8_dtype, num_faces=4, face_r_dim=MAX_FACE_R_DIM):
    """
    Unpack MXFP8 format with layout: [scales padded to 16B][elements padded to 16B].

    One E8M0 scale byte per 32-element block.

    Args:
        packed_bytes: List of bytes in [all scales][all elements] format
        fp8_dtype: ml_dtypes dtype (float8_e5m2 or float8_e4m3fn)
        num_faces: Number of faces (1, 2, or 4). Defaults to 4.
        face_r_dim: Rows per face (1, 2, 4, 8, or 16). Defaults to 16.

    Returns:
        torch.Tensor of bfloat16 values
    """
    num_elements = face_r_dim * FACE_C_DIM * num_faces
    num_scales = num_elements // MXFP8_BLOCK_SIZE

    scale_section_len = _align16(num_scales)

    scales_e8m0 = packed_bytes[:num_scales]
    elements_bytes = packed_bytes[scale_section_len : scale_section_len + num_elements]

    # Convert elements bytes to FP8 blocks and reshape to (num_scales, 32)
    fp8_blocks = np.frombuffer(bytes(elements_bytes), dtype=fp8_dtype).reshape(
        num_scales, MXFP8_BLOCK_SIZE
    )

    # Vectorized scale decoding - decode all E8M0 scales at once
    scales_array = np.frombuffer(bytes(scales_e8m0), dtype=np.uint8)
    # Handle NaN case (255) and compute 2^(exponent) where exponent = value - 127
    scale_factors = np.where(
        scales_array == 255, 0.0, np.exp2(scales_array.astype(np.float32) - 127.0)
    )

    # Scale blocks back to float32
    scaled_blocks = fp8_blocks.astype(np.float32) * scale_factors[:, np.newaxis]

    # Flatten and convert to bfloat16 tensor
    return torch.tensor(scaled_blocks.flatten(), dtype=torch.bfloat16)


def _unpack_mxfp8_srcs(packed_bytes, fp8_dtype, dest_acc: bool = False):
    """Unpack sequential SrcS slices for MX formats.

    Slice geometry depends on *dest_acc*:
      - 16-bit (dest_acc=False): 8×16 = 128 elements/slice, 144 bytes
      - 32-bit (dest_acc=True):  4×16 =  64 elements/slice,  80 bytes
    """
    if dest_acc:
        slice_len = MXFP8_SRCS_SLICE_32B_PACKED_BYTE_LEN
        slice_row_dim = SRCS_SLICE_32B_ROW_DIM
    else:
        slice_len = MXFP8_SRCS_SLICE_PACKED_BYTE_LEN
        slice_row_dim = SRCS_SLICE_ROW_DIM

    num_bytes = len(packed_bytes)
    if num_bytes % slice_len != 0:
        raise ValueError(
            f"Invalid packed_bytes length for use_srcs=True: got {num_bytes} bytes, "
            f"expected a multiple of {slice_len} bytes per SrcS slice."
        )

    out = []
    for i in range(0, num_bytes, slice_len):
        out.append(
            _unpack_mxfp8(
                packed_bytes[i : i + slice_len],
                fp8_dtype,
                num_faces=1,
                face_r_dim=slice_row_dim,
            )
        )
    return torch.cat(out)


def unpack_mxfp8r(
    packed_bytes,
    num_faces=4,
    face_r_dim=MAX_FACE_R_DIM,
    use_srcs: bool = False,
    dest_acc: bool = False,
):
    """
    Unpack MXFP8R format (E5M2 variant) to bfloat16 tensor.

    Args:
        packed_bytes: Packed MX data in FULLY SEPARATED layout [all_scales][all_elements]
        num_faces: Number of faces to unpack (1, 2, or 4). Defaults to 4.
        face_r_dim: Rows per face (1, 2, 4, 8, or 16). Defaults to 16.
        use_srcs: If True, unpack sequential SrcS slices instead of a
            single flat tile.  Supports sub-tile sizes (any multiple of one slice).
        dest_acc: If True (with use_srcs), use 32-bit SrcS slice geometry
            (4×16, 80 bytes/slice) instead of 16-bit (8×16, 144 bytes/slice).

    Returns:
        torch.Tensor of bfloat16 values
    """
    if use_srcs:
        return _unpack_mxfp8_srcs(packed_bytes, ml_dtypes.float8_e5m2, dest_acc)
    return _unpack_mxfp8(packed_bytes, ml_dtypes.float8_e5m2, num_faces, face_r_dim)


def unpack_mxfp8p(
    packed_bytes,
    num_faces=4,
    face_r_dim=MAX_FACE_R_DIM,
    use_srcs: bool = False,
    dest_acc: bool = False,
):
    """
    Unpack MXFP8P format (E4M3 variant) to bfloat16 tensor.

    Args:
        packed_bytes: Packed MX data in FULLY SEPARATED layout [all_scales][all_elements]
        num_faces: Number of faces to unpack (1, 2, or 4). Defaults to 4.
        face_r_dim: Rows per face (1, 2, 4, 8, or 16). Defaults to 16.
        use_srcs: If True, unpack sequential SrcS slices instead of a
            single flat tile.  Supports sub-tile sizes (any multiple of one slice).
        dest_acc: If True (with use_srcs), use 32-bit SrcS slice geometry
            (4×16, 80 bytes/slice) instead of 16-bit (8×16, 144 bytes/slice).

    Returns:
        torch.Tensor of bfloat16 values
    """
    if use_srcs:
        return _unpack_mxfp8_srcs(packed_bytes, ml_dtypes.float8_e4m3fn, dest_acc)
    return _unpack_mxfp8(packed_bytes, ml_dtypes.float8_e4m3fn, num_faces, face_r_dim)


_UNPACKERS = {
    DataFormat.Float16: unpack_fp16,
    DataFormat.Float16_b: unpack_bfp16,
    DataFormat.Float32: unpack_fp32,
    DataFormat.Int32: unpack_int32,
    DataFormat.UInt32: unpack_uint32,
    DataFormat.Int16: unpack_int16,
    DataFormat.UInt16: unpack_uint16,
    DataFormat.Fp8_e4m3: unpack_fp8_e4m3,
    DataFormat.Int8: unpack_int8,
    DataFormat.UInt8: unpack_uint8,
}


def unpack_res_tiles(
    packed_list,
    output_format: DataFormat,
    tile_count: int = 1,
    sfpu: bool = False,
    num_faces: int = MAX_NUM_FACES,
    face_r_dim: int = MAX_FACE_R_DIM,
    tile_stride_bytes: int = None,
    use_srcs: bool = False,
    dest_acc: bool = False,
):
    output_dtype = format_dict[output_format]

    # Stride between tiles in L1 (in bytes):
    # - Default (None): use full 32x32 tile size for backward compatibility
    # - Explicit: use provided stride (for non-32x32 tile dimensions)
    if tile_stride_bytes is None:
        tile_stride_bytes = format_tile_sizes[output_format]
        # Backward-compatible: extract only the faces we need from each full tile
        elements_per_tile_needed = output_format.num_bytes_per_tile(
            num_faces * face_r_dim * FACE_C_DIM
        )
    else:
        # Dense tile path: extract the entire tile.
        # This is necessary because calculate_tile_size_bytes accounts for
        # hardware constraints (e.g. MIN_BFP_EXPONENTS padding for BFP formats)
        # that num_bytes_per_tile does not.
        elements_per_tile_needed = tile_stride_bytes

    total_elements_needed = tile_count * elements_per_tile_needed
    if total_elements_needed > len(packed_list):
        raise IndexError("Buffer access out of bounds")

    if output_format == DataFormat.Bfp8_b:
        unpack_func = unpack_bfp16 if sfpu else unpack_bfp8_b
    elif output_format == DataFormat.Bfp4_b:
        unpack_func = unpack_bfp16 if sfpu else unpack_bfp4_b
    elif output_format == DataFormat.MxFp8R:
        unpack_func = unpack_mxfp8r
    elif output_format == DataFormat.MxFp8P:
        unpack_func = unpack_mxfp8p
    else:
        unpack_func = _UNPACKERS[output_format]

    unpacked_data = []

    # Stride at tile_stride_bytes (L1 layout), but only extract needed bytes per tile
    for tile in range(tile_count):
        start_idx = tile * tile_stride_bytes
        end_idx = start_idx + elements_per_tile_needed
        tile_data = packed_list[start_idx:end_idx]

        if unpack_func in (unpack_bfp8_b, unpack_bfp4_b):
            unpacked_tile = unpack_func(
                tile_data, sfpu=sfpu, num_faces=num_faces, face_r_dim=face_r_dim
            )
        elif unpack_func in [unpack_mxfp8r, unpack_mxfp8p]:
            unpacked_tile = unpack_func(
                tile_data,
                num_faces=num_faces,
                face_r_dim=face_r_dim,
                use_srcs=use_srcs,
                dest_acc=dest_acc,
            )
        else:
            unpacked_tile = unpack_func(tile_data)

        unpacked_data.extend(unpacked_tile)

    return torch.tensor(unpacked_data, dtype=output_dtype)
