# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

# unpack.py

import ml_dtypes
import numpy as np
import torch
from helpers.format_config import (
    MX_FORMAT_BLOCK_SIZE,
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


def unpack_int32(packed_list, twos_complement=False):
    if twos_complement:
        return np.frombuffer(bytes(packed_list), dtype=np.int32).tolist()
    # INT32 uses sign-magnitude format in hardware (not two's complement)
    # Format: bit 31 = sign, bits 30:0 = magnitude
    uint32_array = np.frombuffer(bytes(packed_list), dtype=np.uint32)
    sign = (uint32_array & 0x80000000).astype(bool)
    magnitude = (uint32_array & 0x7FFFFFFF).astype(np.int64)
    return np.where(sign, -magnitude, magnitude).astype(np.int32).tolist()


def unpack_uint32(packed_list):
    return np.frombuffer(bytes(packed_list), dtype=np.uint32).tolist()


def unpack_int16(packed_list, twos_complement=False):
    if twos_complement:
        return np.frombuffer(bytes(packed_list), dtype=np.int16).tolist()
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


def unpack_int8(packed_list, twos_complement=False):
    if twos_complement:
        return np.frombuffer(bytes(packed_list), dtype=np.int8).tolist()
    # INT8 uses sign-magnitude format in hardware (not two's complement)
    # Format: bit 7 = sign, bits 6:0 = magnitude
    uint8_array = np.frombuffer(bytes(packed_list), dtype=np.uint8)
    sign = (uint8_array & 0x80).astype(bool)
    magnitude = (uint8_array & 0x7F).astype(np.int8)
    return np.where(sign, -magnitude, magnitude).tolist()


def unpack_uint8(packed_list):
    return np.frombuffer(bytes(packed_list), dtype=np.uint8).tolist()


# ============================================================================
# BFP (Block Floating-Point) Unpacking
#
# BFP8_b / BFP4_b / BFP2_b share the same structure: 16-element blocks with a
# shared 8-bit exponent and per-element sign-magnitude datums. They differ only
# in datum width (8/4/2 bits → 7/3/1 magnitude bits) and how datums are packed
# into bytes. The helpers below capture that shared structure; see the matching
# pack-side helpers in pack.py (float_to_bfp{8,4,2}_block / pack_bfp{8,4,2}_b).
# ============================================================================


def _expand_bfp_datums(packed_mantissas, bits_per_datum):
    """Expand packed mantissa bytes into one uint8 per datum (LSB datum first).

    For 8-bit datums each byte is a datum. For narrower datums a byte holds
    multiple datums, lowest-order bits first (matching the hardware packing in
    pack_bfp4_b / pack_bfp2_b).
    """
    packed = np.frombuffer(bytes(packed_mantissas), dtype=np.uint8)
    if bits_per_datum == 8:
        return packed
    datums_per_byte = 8 // bits_per_datum
    mask = (1 << bits_per_datum) - 1
    datums = np.empty(len(packed) * datums_per_byte, dtype=np.uint8)
    for k in range(datums_per_byte):
        datums[k::datums_per_byte] = (packed >> (bits_per_datum * k)) & mask
    return datums


def _bfp_to_float_block(exponent, datums, magnitude_bits, cache):
    """Decode one 16-element BFP block to a list of float values.

    Each datum is sign-magnitude with ``magnitude_bits`` magnitude bits and the
    sign in the next bit up. The dequantized magnitude is
    ``mag * 2^(exp - 127 - (magnitude_bits - 1))`` so that the format's leading
    magnitude bit carries weight ``2^(exp - 127)``. A zero magnitude yields a
    signed zero (the sign is preserved via the signed product), matching the
    hardware unpacker / blockfloat_common.cpp reference.
    """
    exp_adj = exponent - 127
    scale = 2.0 ** (exp_adj - (magnitude_bits - 1))
    sign_mask = 1 << magnitude_bits
    mag_mask = sign_mask - 1

    values = []
    for datum in datums:
        key = (exp_adj, int(datum))
        cached = cache.get(key)
        if cached is not None:
            values.append(cached)
            continue
        sign = -1.0 if datum & sign_mask else 1.0
        value = sign * (datum & mag_mask) * scale
        values.append(value)
        cache[key] = value
    return values


def _unpack_bfp_b(bfp_block, magnitude_bits, sfpu=False, num_faces=4, face_r_dim=16):
    """Unpack a BFP8_b / BFP4_b / BFP2_b tile to a bfloat16 tensor.

    ``magnitude_bits`` selects the format: 7 (BFP8), 3 (BFP4), or 1 (BFP2).
    Hardware stores at least MIN_BFP_EXPONENTS exponents (zero-padded), so the
    mantissa section always starts at that aligned offset.
    """
    bits_per_datum = magnitude_bits + 1
    actual_exponents = face_r_dim * num_faces
    exponents_in_packed = max(actual_exponents, MIN_BFP_EXPONENTS)

    if not sfpu:
        exponents = bfp_block[:actual_exponents]
        packed_mantissas = bfp_block[exponents_in_packed:]
    else:
        # SFPU layout: 16 exponents followed by 16 blocks of 16 datums.
        exponents = bfp_block[:16]
        sfpu_mantissa_bytes = 16 * 16 * bits_per_datum // 8
        packed_mantissas = bfp_block[16 : 16 + sfpu_mantissa_bytes]

    datums = _expand_bfp_datums(packed_mantissas, bits_per_datum)

    cache = {}
    bfloat16_values = []
    for i, exponent in enumerate(exponents):
        block_datums = datums[i * 16 : (i + 1) * 16]
        bfloat16_values.extend(
            _bfp_to_float_block(exponent, block_datums, magnitude_bits, cache)
        )

    return torch.tensor(bfloat16_values, dtype=torch.bfloat16)


def unpack_bfp8_b(bfp8_block, sfpu=False, num_faces=4, face_r_dim=16):
    return _unpack_bfp_b(
        bfp8_block,
        magnitude_bits=7,
        sfpu=sfpu,
        num_faces=num_faces,
        face_r_dim=face_r_dim,
    )


def unpack_bfp4_b(bfp4_block, sfpu=False, num_faces=4, face_r_dim=16):
    return _unpack_bfp_b(
        bfp4_block,
        magnitude_bits=3,
        sfpu=sfpu,
        num_faces=num_faces,
        face_r_dim=face_r_dim,
    )


def unpack_bfp2_b(bfp2_block, sfpu=False, num_faces=4, face_r_dim=16):
    return _unpack_bfp_b(
        bfp2_block,
        magnitude_bits=1,
        sfpu=sfpu,
        num_faces=num_faces,
        face_r_dim=face_r_dim,
    )


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
    num_scales = num_elements // MX_FORMAT_BLOCK_SIZE

    scale_section_len = _align16(num_scales)

    scales_e8m0 = packed_bytes[:num_scales]
    elements_bytes = packed_bytes[scale_section_len : scale_section_len + num_elements]

    # Convert elements bytes to FP8 blocks and reshape to (num_scales, 32)
    fp8_blocks = np.frombuffer(bytes(elements_bytes), dtype=fp8_dtype).reshape(
        num_scales, MX_FORMAT_BLOCK_SIZE
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


def unpack_mxfp4(
    packed_bytes,
    num_faces=4,
    face_r_dim=MAX_FACE_R_DIM,
    use_srcs: bool = False,
    dest_acc: bool = False,
):
    """
    Unpack MXFP4 format (E2M1 variant) to bfloat16 tensor.
    Function is implemented based on the OCP MX specification and Tensix hardware documentation.

    MXFP4 uses 32-element blocks per OCP MX spec, each with:
      - 1 shared E8M0 scale (8 bits)
      - 32 × float4_e2m1fn elements (4 bits each, packed 2 per byte)

    Layout: [all_scales][all_packed_elements]
      - [32 scales (1 per block)][512 bytes (2 FP4 elements per byte)]

    Per Tensix hardware documentation:
      - Block exp = 0xFF (255): NaN block, all elements become NaN
      - Block exp = 0x00 (0): neutral-ish scale for zeros

    Args:
        packed_bytes: Packed MX data in FULLY SEPARATED layout [all_scales][all_elements]
        num_faces: Number of faces to unpack (1, 2, or 4). Defaults to 4.
        face_r_dim: Rows per face (1, 2, 4, 8, or 16). Defaults to 16.
        use_srcs: If True, unpack sequential SrcS slices (not yet implemented for MxFp4).
        dest_acc: If True (with use_srcs), use 32-bit SrcS slice geometry (not yet implemented for MxFp4).

    Returns:
        torch.Tensor of bfloat16 values
    """
    if use_srcs:
        # SrcS mode not yet implemented for MxFp4
        raise NotImplementedError("use_srcs mode is not yet supported for MxFp4")

    block_size = MX_FORMAT_BLOCK_SIZE
    num_elements = face_r_dim * FACE_C_DIM * num_faces
    num_blocks = num_elements // block_size

    if num_elements % block_size != 0:
        raise ValueError(
            "Invalid MXFP4 tile geometry: num_elements must be a multiple of "
            f"{block_size}, got {num_elements}."
        )

    # Expected bytes = 1 scale byte per block (16B-aligned) + packed FP4 elements.
    scale_section_len = _align16(num_blocks)
    element_bytes_len = num_blocks * (block_size // 2)
    expected_len = scale_section_len + element_bytes_len
    if len(packed_bytes) != expected_len:
        raise ValueError(
            "Invalid packed_bytes length for MXFP4: got "
            f"{len(packed_bytes)} bytes, expected {expected_len} bytes."
        )

    scales_u8 = np.frombuffer(bytes(packed_bytes[:num_blocks]), dtype=np.uint8)
    packed_u8 = np.frombuffer(
        bytes(packed_bytes[scale_section_len : scale_section_len + element_bytes_len]),
        dtype=np.uint8,
    )

    # Each byte packs 2 FP4 values: low nibble then high nibble.
    nibbles_u8 = np.empty(packed_u8.size * 2, dtype=np.uint8)
    nibbles_u8[0::2] = packed_u8 & 0x0F
    nibbles_u8[1::2] = packed_u8 >> 4

    fp4_f32 = (
        nibbles_u8.view(ml_dtypes.float4_e2m1fn)[: num_blocks * block_size]
        .reshape(num_blocks, block_size)
        .astype(np.float32)
    )

    block_exp_unbiased = scales_u8.astype(np.int32) - 127  # E8M0 bias=127
    scaled_blocks = fp4_f32 * np.exp2(block_exp_unbiased.astype(np.float32))[:, None]

    # Extract 2-bit exponent field from E2M1 format
    unit_exp_field = (
        ((nibbles_u8 >> 1) & 0x3)
        .astype(np.int32)[: num_blocks * block_size]
        .reshape(num_blocks, block_size)
    )

    # E2M1 unbiased exponent calculation (bias=1):
    # - Normal values (exp_field != 0): unbiased = exp_field - 1
    # - Subnormal values (exp_field == 0): unbiased = 0 (fixed at 1-bias)
    unit_exp_unbiased = np.where(unit_exp_field == 0, 0, unit_exp_field - 1)
    combined_unbiased = block_exp_unbiased[:, None] + unit_exp_unbiased

    nan_blocks = scales_u8 == 0xFF
    overflow_mask = (combined_unbiased >= 128) & ~nan_blocks[:, None]
    underflow_mask = (combined_unbiased < -127) & ~nan_blocks[:, None]

    if np.any(nan_blocks):
        scaled_blocks[nan_blocks] = np.nan

    scaled_blocks[overflow_mask] = np.where(
        scaled_blocks[overflow_mask] >= 0.0, np.inf, -np.inf
    )
    scaled_blocks[underflow_mask] = 0.0

    return torch.tensor(scaled_blocks.ravel(), dtype=torch.bfloat16)


def _mxint_decode_blocks(scales_e8m0, int_blocks, elem_scale_divisor: float):
    """
    Shared unpack core for MxInt formats. Given E8M0 scale bytes and a
    (num_blocks, 32) int8 array of per-element values, return the decoded
    bfloat16 tensor. `elem_scale_divisor` is the format's implicit scale
    denominator (64 for MxInt8's 2^-6, 4 for MxInt4's 2^-2, 1 for MxInt2's
    2^0). NaN scale (0xFF) zeros the block, matching MxFp unpack behavior.
    """
    scales_array = np.frombuffer(bytes(scales_e8m0), dtype=np.uint8)
    scale_factors = np.where(
        scales_array == 255, 0.0, np.exp2(scales_array.astype(np.float32) - 127.0)
    )
    decoded = int_blocks.astype(np.float32) * (
        scale_factors[:, np.newaxis] / elem_scale_divisor
    )
    return torch.tensor(decoded.flatten(), dtype=torch.bfloat16)


def unpack_mxint8(
    packed_bytes,
    num_faces=4,
    face_r_dim=MAX_FACE_R_DIM,
    use_srcs: bool = False,
    dest_acc: bool = False,
):
    """
    Unpack MxInt8 format (signed S1.6 with E8M0 block scale) to bfloat16 tensor.

    Layout: [scales padded to 16B][signed-int8 elements padded to 16B], one E8M0
    scale per 32-element block. Decoded value = (int8 / 64) × 2^(scale_e8m0 − 127).
    """
    if use_srcs:
        raise NotImplementedError("use_srcs=True not yet implemented for unpack_mxint8")

    num_elements = face_r_dim * FACE_C_DIM * num_faces
    num_scales = num_elements // MX_FORMAT_BLOCK_SIZE
    scale_section_len = _align16(num_scales)

    if len(packed_bytes) < scale_section_len + num_elements:
        raise ValueError(
            "Invalid packed_bytes length for MxInt8: got "
            f"{len(packed_bytes)} bytes, expected at least "
            f"{scale_section_len + num_elements} bytes."
        )

    scales_e8m0 = packed_bytes[:num_scales]
    elements_bytes = packed_bytes[scale_section_len : scale_section_len + num_elements]
    int8_blocks = np.frombuffer(bytes(elements_bytes), dtype=np.int8).reshape(
        num_scales, MX_FORMAT_BLOCK_SIZE
    )
    return _mxint_decode_blocks(scales_e8m0, int8_blocks, elem_scale_divisor=64.0)


def unpack_mxint4(
    packed_bytes,
    num_faces=4,
    face_r_dim=MAX_FACE_R_DIM,
    use_srcs: bool = False,
    dest_acc: bool = False,
):
    """
    Unpack MxInt4 format (signed S1.2 with E8M0 block scale) to bfloat16 tensor.

    Layout: [scales padded to 16B][packed nibbles padded to 16B], one E8M0
    scale per 32-element block, 2 elements per byte (low nibble = even index).
    Decoded value = (int4 / 4) × 2^(scale_e8m0 − 127).
    """
    if use_srcs:
        raise NotImplementedError("use_srcs=True not yet implemented for unpack_mxint4")

    num_elements = face_r_dim * FACE_C_DIM * num_faces
    num_scales = num_elements // MX_FORMAT_BLOCK_SIZE
    scale_section_len = _align16(num_scales)
    element_bytes_len = num_elements // 2  # 2 elements per byte

    if len(packed_bytes) < scale_section_len + element_bytes_len:
        raise ValueError(
            "Invalid packed_bytes length for MxInt4: got "
            f"{len(packed_bytes)} bytes, expected at least "
            f"{scale_section_len + element_bytes_len} bytes."
        )

    scales_e8m0 = packed_bytes[:num_scales]
    elements_bytes = packed_bytes[
        scale_section_len : scale_section_len + element_bytes_len
    ]

    # Unpack 2 nibbles per byte: low = even index, high = odd index.
    packed_u8 = np.frombuffer(bytes(elements_bytes), dtype=np.uint8)
    nibbles_u8 = np.empty(packed_u8.size * 2, dtype=np.uint8)
    nibbles_u8[0::2] = packed_u8 & 0x0F
    nibbles_u8[1::2] = packed_u8 >> 4
    # Sign-extend each 4-bit value to int8: nibbles ≥ 8 are negative in 2's comp.
    int4_as_int8 = np.where(
        nibbles_u8 >= 8,
        nibbles_u8.astype(np.int16) - 16,
        nibbles_u8.astype(np.int16),
    ).astype(np.int8)
    int4_blocks = int4_as_int8.reshape(num_scales, MX_FORMAT_BLOCK_SIZE)
    return _mxint_decode_blocks(scales_e8m0, int4_blocks, elem_scale_divisor=4.0)


def unpack_mxint2(
    packed_bytes,
    num_faces=4,
    face_r_dim=MAX_FACE_R_DIM,
    use_srcs: bool = False,
    dest_acc: bool = False,
):
    """
    Unpack MxInt2 format (signed S1.0 with E8M0 block scale) to bfloat16 tensor.

    Layout: [scales padded to 16B][packed crumbs padded to 16B], one E8M0
    scale per 32-element block, 4 elements per byte (crumb layout: bits[1:0]
    = even-most index, then [3:2], [5:4], [7:6]). Decoded value = int2 × 2^(scale_e8m0 − 127).
    """
    if use_srcs:
        raise NotImplementedError("use_srcs=True not yet implemented for unpack_mxint2")

    num_elements = face_r_dim * FACE_C_DIM * num_faces
    num_scales = num_elements // MX_FORMAT_BLOCK_SIZE
    scale_section_len = _align16(num_scales)
    element_bytes_len = num_elements // 4  # 4 elements per byte

    if len(packed_bytes) < scale_section_len + element_bytes_len:
        raise ValueError(
            "Invalid packed_bytes length for MxInt2: got "
            f"{len(packed_bytes)} bytes, expected at least "
            f"{scale_section_len + element_bytes_len} bytes."
        )

    scales_e8m0 = packed_bytes[:num_scales]
    elements_bytes = packed_bytes[
        scale_section_len : scale_section_len + element_bytes_len
    ]

    # Unpack 4 crumbs per byte: bits[1:0], [3:2], [5:4], [7:6].
    packed_u8 = np.frombuffer(bytes(elements_bytes), dtype=np.uint8)
    crumbs_u8 = np.empty(packed_u8.size * 4, dtype=np.uint8)
    crumbs_u8[0::4] = packed_u8 & 0x03
    crumbs_u8[1::4] = (packed_u8 >> 2) & 0x03
    crumbs_u8[2::4] = (packed_u8 >> 4) & 0x03
    crumbs_u8[3::4] = (packed_u8 >> 6) & 0x03
    # Sign-extend each 2-bit value to int8: crumbs ≥ 2 are negative in 2's comp.
    int2_as_int8 = np.where(
        crumbs_u8 >= 2,
        crumbs_u8.astype(np.int16) - 4,
        crumbs_u8.astype(np.int16),
    ).astype(np.int8)
    int2_blocks = int2_as_int8.reshape(num_scales, MX_FORMAT_BLOCK_SIZE)
    return _mxint_decode_blocks(scales_e8m0, int2_blocks, elem_scale_divisor=1.0)


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
    twos_complement: bool = False,
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
    elif output_format == DataFormat.Bfp2_b:
        unpack_func = unpack_bfp16 if sfpu else unpack_bfp2_b
    elif output_format == DataFormat.MxFp8R:
        unpack_func = unpack_mxfp8r
    elif output_format == DataFormat.MxFp8P:
        unpack_func = unpack_mxfp8p
    elif output_format == DataFormat.MxFp4:
        unpack_func = unpack_mxfp4
    elif output_format == DataFormat.MxInt8:
        unpack_func = unpack_mxint8
    elif output_format == DataFormat.MxInt4:
        unpack_func = unpack_mxint4
    elif output_format == DataFormat.MxInt2:
        unpack_func = unpack_mxint2
    else:
        unpack_func = _UNPACKERS[output_format]

    unpacked_data = []

    # Stride at tile_stride_bytes (L1 layout), but only extract needed bytes per tile
    for tile in range(tile_count):
        start_idx = tile * tile_stride_bytes
        end_idx = start_idx + elements_per_tile_needed
        tile_data = packed_list[start_idx:end_idx]

        if unpack_func in (unpack_bfp8_b, unpack_bfp4_b, unpack_bfp2_b):
            unpacked_tile = unpack_func(
                tile_data, sfpu=sfpu, num_faces=num_faces, face_r_dim=face_r_dim
            )
        elif unpack_func in [
            unpack_mxfp8r,
            unpack_mxfp8p,
            unpack_mxfp4,
            unpack_mxint8,
            unpack_mxint4,
            unpack_mxint2,
        ]:
            unpacked_tile = unpack_func(
                tile_data,
                num_faces=num_faces,
                face_r_dim=face_r_dim,
                use_srcs=use_srcs,
                dest_acc=dest_acc,
            )
        elif twos_complement and unpack_func in (
            unpack_int32,
            unpack_int16,
            unpack_int8,
        ):
            unpacked_tile = unpack_func(tile_data, twos_complement=True)
        else:
            unpacked_tile = unpack_func(tile_data)

        unpacked_data.extend(unpacked_tile)

    return torch.tensor(unpacked_data, dtype=output_dtype)
