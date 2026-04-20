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
    l1_align,
)
from .tile_constants import (
    FACE_C_DIM,
    MAX_TILE_ELEMENTS,
    MIN_BFP_EXPONENTS,
    SRCS_SLICE_ELEMENT_COUNT,
    SRCS_SLICE_ROW_DIM,
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
    # INT32 uses sign-magnitude format in hardware (not two's complement)
    # Format: bit 31 = sign, bits 30:0 = magnitude
    # Sign-magnitude INT32 cannot represent -2147483648, so clip to [min+1, max]
    iinfo = torch.iinfo(torch.int32)
    array = torch_tensor.cpu().numpy()
    clipped = np.clip(array, iinfo.min + 1, iinfo.max).astype(np.int32)
    sign = clipped.view(np.uint32) & 0x80000000
    magnitude = np.abs(clipped).astype(np.uint32)
    return (sign | magnitude).tobytes()


def pack_uint32(torch_tensor):
    return torch_tensor.cpu().numpy().astype(np.uint32).tobytes()


def pack_int16(torch_tensor):
    # INT16 uses sign-magnitude format in hardware (not two's complement)
    # Format: bit 15 = sign, bits 14:0 = magnitude
    # Sign-magnitude INT16 cannot represent -32768, so clip to [min+1, max]
    iinfo = torch.iinfo(torch.int16)
    array = torch_tensor.cpu().numpy()
    clipped = np.clip(array, iinfo.min + 1, iinfo.max).astype(np.int16)
    sign = clipped.view(np.uint16) & 0x8000
    magnitude = np.abs(clipped).astype(np.uint16)
    return (sign | magnitude).tobytes()


def pack_uint16(torch_tensor):
    return torch_tensor.cpu().numpy().astype(np.uint16).tobytes()


def pack_fp8_e4m3(torch_tensor):
    fp32_array = torch_tensor.cpu().to(torch.float32).numpy()
    return fp32_array.astype(ml_dtypes.float8_e4m3fn).tobytes()


def pack_int8(torch_tensor):
    # INT8 uses sign-magnitude format in hardware (not two's complement)
    # Format: bit 7 = sign, bits 6:0 = magnitude
    # Sign-magnitude INT8 cannot represent -128, so clip to [min+1, max]
    iinfo = torch.iinfo(torch.int8)
    array = torch_tensor.cpu().numpy()
    clipped = np.clip(array, iinfo.min + 1, iinfo.max).astype(np.int8)
    sign = clipped.view(np.uint8) & 0x80
    magnitude = np.abs(clipped).astype(np.uint8)
    return (sign | magnitude).tobytes()


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
        if exponent_delta > 0:
            # Round-to-nearest, ties away from zero (per ISA spec for BFP8 packing)
            guard_bit = (mantissas_explicit[i] >> (exponent_delta - 1)) & 1
            mantissa = (mantissas_explicit[i] >> exponent_delta) + guard_bit
        else:
            mantissa = mantissas_explicit[i]
        mantissa = mantissa & 0x7F
        mantissa = (signs[i] << 7) | mantissa
        bfp8_mantissas.append(mantissa)

    return shared_exponent, bfp8_mantissas


def pack_bfp8_b(tensor, block_size=16, num_faces=4, face_r_dim=16):
    """Pack tensor into BFP8_b format.

    BFP8_b uses 16-element blocks, each with a shared exponent and 8-bit mantissas.
    Only the first (elements_per_face * num_faces) elements are packed.

    Hardware requires minimum 16 exponents total. If fewer blocks exist, pad with zeros.

    Args:
        tensor: Input tensor (typically 1024 elements for full tile)
        block_size: Elements per block (always 16 for BFP8_b)
        num_faces: Number of faces to pack (1, 2, or 4)
        face_r_dim: Number of rows per face (1, 2, 4, 8, or 16)

    Returns:
        List of packed bytes: [exponents...] + [mantissas...]
    """
    flattened_tensor = tensor.flatten()

    # Calculate elements per face based on face_r_dim
    elements_per_face = face_r_dim * FACE_C_DIM
    elements_to_pack = elements_per_face * num_faces
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

    # Hardware requires minimum exponents - pad if needed
    if len(exponents) < MIN_BFP_EXPONENTS:
        padding_count = MIN_BFP_EXPONENTS - len(exponents)
        exponents.extend([0] * padding_count)

    return exponents + mantissas


def truncate_bfp8_to_bfp4(bfp8_mantissas):
    """Truncate BFP8 mantissas to BFP4 format.

    BFP8 mantissas are 8 bits: 1 sign bit + 7 magnitude bits.
    BFP4 mantissas are 4 bits: 1 sign bit + 3 magnitude bits.

    Truncation: keep the top 3 magnitude bits, drop the last 4 bits.
    This matches hardware behavior which uses simple truncation.

    Args:
        bfp8_mantissas: List of 8-bit BFP8 mantissa values

    Returns:
        List of 4-bit BFP4 mantissa values (as integers 0-15)
    """
    bfp4_mantissas = []
    for bfp8 in bfp8_mantissas:
        sign = (bfp8 >> 7) & 0x1
        # Extract 7-bit magnitude and truncate to top 3 bits
        magnitude = (bfp8 >> 4) & 0x7
        bfp4 = (sign << 3) | magnitude
        bfp4_mantissas.append(bfp4)
    return bfp4_mantissas


def float_to_bfp4_block(block):
    """Pack a 16-element block to BFP4_b format.

    Per hardware spec (WormholeB0 Packers/FormatConversion.md):
      Step 1: Convert to BFP8 using float_to_bfp8_block (round-to-nearest,
              one shared 8-bit exponent per 16 datums).
              BFP8 has 1 sign bit + 7 magnitude bits.
      Step 2: Truncate BFP8 → BFP4 (keep top 3 of the 7 magnitude bits).
    """
    # Step 1: Convert to BFP8 first
    shared_exponent, bfp8_mantissas = float_to_bfp8_block(block)

    # Step 2: Truncate BFP8 mantissas to BFP4
    bfp4_mantissas = truncate_bfp8_to_bfp4(bfp8_mantissas)

    return shared_exponent, bfp4_mantissas


def pack_bfp4_b(tensor, block_size=16, num_faces=4, face_r_dim=16):
    """Pack tensor into BFP4_b format.

    BFP4_b uses 16-element blocks, each with a shared exponent and 4-bit mantissas.
    Two mantissa datums are packed per byte (high nibble = even element, low nibble = odd).

    Args:
        tensor: Input tensor (typically 1024 elements for full tile)
        block_size: Elements per block (always 16 for BFP4_b)
        num_faces: Number of faces to pack (1, 2, or 4)
        face_r_dim: Number of rows per face (1, 2, 4, 8, or 16)

    Returns:
        List of packed bytes: [exponents...] + [packed_mantissas...]
    """
    flattened_tensor = tensor.flatten()

    elements_per_face = face_r_dim * FACE_C_DIM
    elements_to_pack = elements_per_face * num_faces
    assert (
        len(flattened_tensor) >= elements_to_pack
    ), f"Tensor has {len(flattened_tensor)} elements, but need at least {elements_to_pack} for {num_faces} face(s)"
    flattened_tensor = flattened_tensor[:elements_to_pack]

    num_blocks = len(flattened_tensor) // block_size

    exponents = []
    all_mantissas = []

    for i in range(num_blocks):
        block = flattened_tensor[i * block_size : (i + 1) * block_size]
        shared_exponent, bfp4_mantissas = float_to_bfp4_block(block)
        exponents.append(shared_exponent)
        all_mantissas.extend(bfp4_mantissas)

    if len(exponents) < MIN_BFP_EXPONENTS:
        padding_count = MIN_BFP_EXPONENTS - len(exponents)
        exponents.extend([0] * padding_count)

    packed_mantissas = []
    for i in range(0, len(all_mantissas), 2):
        low = all_mantissas[i]
        high = all_mantissas[i + 1] if (i + 1) < len(all_mantissas) else 0
        packed_mantissas.append((high << 4) | low)

    result = exponents + packed_mantissas
    return result


# ============================================================================
# MX (Microscaling) Format Support - OCP Specification
# ============================================================================


def _pad_to_l1_alignment(data: list[int]) -> list[int]:
    """Pad a byte list to the next L1-aligned (16-byte) boundary."""
    aligned_len = l1_align(len(data))
    pad = aligned_len - len(data)
    return data if pad == 0 else data + [0] * pad


def _pack_mxfp8(tensor, fp8_dtype, element_max_normal, num_faces=4, face_r_dim=16):
    """
    Internal helper to pack MXFP8 formats with FULLY SEPARATED layout.

    Layout (similar to BFP8_b): [all_scales][all_elements], each section 16-byte aligned.
    Padding bytes (zeros) are appended after scales and after FP8 payload as needed.
    - Full tile: 32 scales (32 B, aligned) + 1024 FP8 (aligned) → 1056 B.
    - SrcS slice (8×16): 4 scales + pad to 16 B + 128 FP8 (aligned) → 144 B per slice.

    Element count must be a multiple of MXFP8_BLOCK_SIZE (32).

    Uses ml_dtypes for FP8 element conversion and E8M0 scale encoding.

    Args:
        tensor: Input tensor (first face_r_dim * FACE_C_DIM * num_faces elements used)
        fp8_dtype: ml_dtypes dtype (float8_e5m2 or float8_e4m3fn)
        element_max_normal: Maximum normal value for element format
        num_faces: Number of faces to pack (1, 2, or 4). Defaults to 4.
        face_r_dim: Number of rows per face (1, 2, 4, 8, or 16). Defaults to 16.

    Returns:
        List of packed bytes: [all scales][all elements]
    """
    # Convert to numpy and prepare data
    fp32_array = tensor.cpu().to(torch.float32).numpy().flatten()

    # Calculate elements per face based on face_r_dim
    elements_per_face = face_r_dim * FACE_C_DIM
    elements_to_pack = elements_per_face * num_faces
    assert (
        len(fp32_array) >= elements_to_pack
    ), f"Tensor has {len(fp32_array)} elements, need {elements_to_pack} for {num_faces} face(s)"
    assert elements_to_pack % MXFP8_BLOCK_SIZE == 0, (
        f"Element count ({elements_to_pack}) must be a multiple of "
        f"MXFP8_BLOCK_SIZE ({MXFP8_BLOCK_SIZE})"
    )

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

    # FULLY SEPARATED layout: all scales first, then all elements (both 16B-aligned)
    # Convert FP8 blocks to list of bytes (integers 0-255)
    fp8_bytes = list(fp8_blocks.tobytes())
    return _pad_to_l1_alignment(scales_e8m0) + _pad_to_l1_alignment(fp8_bytes)


def _pack_mxfp8_srcs(tensor, fp8_dtype, element_max_normal, dest_acc: bool = False):
    """Pack a tensor into per-slice SrcS blocks for MX formats.

    Splits the tensor into SrcS slices and packs each independently as
    [scales][elements].  Slice geometry depends on *dest_acc*:
      - 16-bit (dest_acc=False): 8×16 = 128 elements/slice, 144 bytes
      - 32-bit (dest_acc=True):  4×16 =  64 elements/slice,  80 bytes
    """
    if dest_acc:
        slice_elem_count = SRCS_SLICE_32B_ELEMENT_COUNT
        slice_row_dim = SRCS_SLICE_32B_ROW_DIM
    else:
        slice_elem_count = SRCS_SLICE_ELEMENT_COUNT
        slice_row_dim = SRCS_SLICE_ROW_DIM

    flat = tensor.flatten()
    num_elements = flat.numel()
    out: list[int] = []
    for i in range(0, num_elements, slice_elem_count):
        out.extend(
            _pack_mxfp8(
                flat[i : i + slice_elem_count],
                fp8_dtype,
                element_max_normal,
                num_faces=1,
                face_r_dim=slice_row_dim,
            )
        )
    return out


def pack_mxfp8r(
    tensor, num_faces=4, face_r_dim=16, use_srcs: bool = False, dest_acc: bool = False
):
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
        tensor: Input tensor (at most one tile worth of elements).
        num_faces: Number of faces to pack (1, 2, or 4). Defaults to 4.
        face_r_dim: Number of rows per face (1, 2, 4, 8, or 16). Defaults to 16.
        use_srcs: If True, split into SrcS slices (per-slice blocks in L1).
        dest_acc: If True (with use_srcs), use 32-bit SrcS slice geometry
            (4×16, 80 bytes/slice) instead of 16-bit (8×16, 144 bytes/slice).

    Returns:
        List of packed bytes in FULLY SEPARATED layout: [all_scales][all_elements]
        Scale count = (face_r_dim * 16 * num_faces) // 32 (one per OCP 32-datum block).
    """
    assert tensor.numel() <= MAX_TILE_ELEMENTS, (
        f"pack_mxfp8r handles at most one tile ({MAX_TILE_ELEMENTS} elements), "
        f"got {tensor.numel()}"
    )
    if use_srcs:
        return _pack_mxfp8_srcs(
            tensor, ml_dtypes.float8_e5m2, MXFP8_E5M2_MAX_NORMAL, dest_acc
        )
    return _pack_mxfp8(
        tensor, ml_dtypes.float8_e5m2, MXFP8_E5M2_MAX_NORMAL, num_faces, face_r_dim
    )


def pack_mxfp8p(
    tensor, num_faces=4, face_r_dim=16, use_srcs: bool = False, dest_acc: bool = False
):
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
        tensor: Input tensor (at most one tile worth of elements).
        num_faces: Number of faces to pack (1, 2, or 4). Defaults to 4.
        face_r_dim: Number of rows per face (1, 2, 4, 8, or 16). Defaults to 16.
        use_srcs: If True, split into SrcS slices (per-slice blocks in L1).
        dest_acc: If True (with use_srcs), use 32-bit SrcS slice geometry
            (4×16, 80 bytes/slice) instead of 16-bit (8×16, 144 bytes/slice).

    Returns:
        List of packed bytes in FULLY SEPARATED layout: [all_scales][all_elements]
        Scale count = (face_r_dim * 16 * num_faces) // 32 (one per OCP 32-datum block).
    """
    assert tensor.numel() <= MAX_TILE_ELEMENTS, (
        f"pack_mxfp8p handles at most one tile ({MAX_TILE_ELEMENTS} elements), "
        f"got {tensor.numel()}"
    )
    if use_srcs:
        return _pack_mxfp8_srcs(
            tensor, ml_dtypes.float8_e4m3fn, MXFP8_E4M3_MAX_NORMAL, dest_acc
        )
    return _pack_mxfp8(
        tensor, ml_dtypes.float8_e4m3fn, MXFP8_E4M3_MAX_NORMAL, num_faces, face_r_dim
    )
