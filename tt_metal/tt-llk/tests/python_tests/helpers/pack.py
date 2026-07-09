# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import struct

import ml_dtypes
import numpy as np
import torch

from .format_config import (
    MX_FORMAT_BLOCK_SIZE,
    MX_FP_SPECS,
    MX_INT_SPECS,
    DataFormat,
    l1_align,
)
from .tile_constants import (
    FACE_C_DIM,
    MAX_TILE_ELEMENTS,
    MIN_BFP_EXPONENTS,
    SRCS_SLICE_32B_ELEMENT_COUNT,
    SRCS_SLICE_32B_ROW_DIM,
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


def pack_int32(torch_tensor, twos_complement=False):
    if twos_complement:
        return torch_tensor.cpu().numpy().astype(np.int32).tobytes()
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


def pack_int16(torch_tensor, twos_complement=False):
    if twos_complement:
        return torch_tensor.cpu().numpy().astype(np.int16).tobytes()
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


def pack_int8(torch_tensor, twos_complement=False):
    if twos_complement:
        return torch_tensor.cpu().numpy().astype(np.int8).tobytes()
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


# ============================================================================
# BFP (Block Floating-Point) Format Helpers
# ============================================================================


def _bfp_prepare_blocks(tensor, block_size, num_faces, face_r_dim):
    """Flatten, validate, and trim the input tensor for BFP packing.

    Args:
        tensor: Input tensor.
        block_size: Elements per BFP block (16 for all current BFP formats).
        num_faces: Number of tile faces to pack.
        face_r_dim: Rows per face.

    Returns:
        Flattened tensor trimmed to (face_r_dim * FACE_C_DIM * num_faces) elements.
    """
    flattened_tensor = tensor.flatten()
    elements_per_face = face_r_dim * FACE_C_DIM
    elements_to_pack = elements_per_face * num_faces
    assert (
        len(flattened_tensor) >= elements_to_pack
    ), f"Tensor has {len(flattened_tensor)} elements, but need at least {elements_to_pack} for {num_faces} face(s)"
    return flattened_tensor[:elements_to_pack]


def _bfp_collect_blocks(flattened_tensor, block_size, float_to_block_fn):
    """Iterate over BFP blocks, collecting shared exponents and mantissas.

    Args:
        flattened_tensor: Pre-processed tensor (output of _bfp_prepare_blocks).
        block_size: Elements per block (16 for all current BFP formats).
        float_to_block_fn: Callable(block) -> (shared_exponent, mantissas).

    Returns:
        (exponents, all_mantissas) where exponents is padded to at least MIN_BFP_EXPONENTS.
    """
    num_blocks = len(flattened_tensor) // block_size
    exponents = []
    all_mantissas = []
    for i in range(num_blocks):
        block = flattened_tensor[i * block_size : (i + 1) * block_size]
        shared_exponent, bfp_mantissas = float_to_block_fn(block)
        exponents.append(shared_exponent)
        all_mantissas.extend(bfp_mantissas)
    if len(exponents) < MIN_BFP_EXPONENTS:
        exponents.extend([0] * (MIN_BFP_EXPONENTS - len(exponents)))
    return exponents, all_mantissas


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
        # Flush negative-zero to +0: when the magnitude rounds/shifts to 0, drop
        # the sign bit. The tt-metal host quantizer (convert_u32_to_bfp) does the
        # same, and the hardware unpacker decodes a sign-only mantissa (0x80) to
        # -inf, so no valid producer should ever emit it.
        mantissa = ((signs[i] << 7) | mantissa) if mantissa != 0 else 0
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
    flattened_tensor = _bfp_prepare_blocks(tensor, block_size, num_faces, face_r_dim)
    exponents, mantissas = _bfp_collect_blocks(
        flattened_tensor, block_size, float_to_bfp8_block
    )
    return exponents + mantissas


def truncate_bfp8(bfp8_mantissas, magnitude_bits):
    """Truncate BFP8 mantissas to a narrower BFP format.

    BFP8 mantissas are 8 bits: 1 sign bit + 7 magnitude bits.
    The output keeps the sign bit and the top ``magnitude_bits`` of the
    7 magnitude bits, dropping the bottom (7 - magnitude_bits) bits.
    This matches hardware behavior which uses simple truncation when
    narrowing BFP8 to BFP4 or BFP2 (per WormholeB0 Packers/FormatConversion.md).

    Args:
        bfp8_mantissas: List of 8-bit BFP8 mantissa values.
        magnitude_bits: Number of magnitude bits to keep in the output
            (3 for BFP4, 1 for BFP2).

    Returns:
        List of (magnitude_bits + 1)-bit truncated mantissas.
    """
    shift = 7 - magnitude_bits
    mag_mask = (1 << magnitude_bits) - 1
    out = []
    for bfp8 in bfp8_mantissas:
        magnitude = (bfp8 >> shift) & mag_mask
        sign = (bfp8 >> 7) & 0x1
        # Flush negative-zero to +0: truncation can drop a small BFP8 magnitude to
        # 0 while leaving the sign bit set. Emit +0 so the hardware unpacker never
        # sees a sign-only mantissa (which it decodes to -inf), matching the host
        # quantizer's behavior.
        out.append(((sign << magnitude_bits) | magnitude) if magnitude != 0 else 0)
    return out


def float_to_bfp4_block(block):
    """Pack a 16-element block to BFP4_b format.

    Per hardware spec (WormholeB0 Packers/FormatConversion.md):
      Step 1: Convert to BFP8 using float_to_bfp8_block (round-to-nearest,
              one shared 8-bit exponent per 16 datums).
              BFP8 has 1 sign bit + 7 magnitude bits.
      Step 2: Truncate BFP8 → BFP4 (keep top 3 of the 7 magnitude bits).
    """
    shared_exponent, bfp8_mantissas = float_to_bfp8_block(block)
    bfp4_mantissas = truncate_bfp8(bfp8_mantissas, magnitude_bits=3)
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
    flattened_tensor = _bfp_prepare_blocks(tensor, block_size, num_faces, face_r_dim)
    exponents, all_mantissas = _bfp_collect_blocks(
        flattened_tensor, block_size, float_to_bfp4_block
    )

    packed_mantissas = []
    for i in range(0, len(all_mantissas), 2):
        low = all_mantissas[i]
        high = all_mantissas[i + 1] if (i + 1) < len(all_mantissas) else 0
        packed_mantissas.append((high << 4) | low)

    return exponents + packed_mantissas


def float_to_bfp2_block(block):
    """Pack a 16-element block to BFP2_b format.

    Per hardware spec (WormholeB0 Packers/FormatConversion.md):
      Step 1: Convert to BFP8 using float_to_bfp8_block (round-to-nearest,
              one shared 8-bit exponent per 16 datums).
              BFP8 has 1 sign bit + 7 magnitude bits.
      Step 2: Truncate BFP8 → BFP2 (keep top 1 of the 7 magnitude bits).
    """
    shared_exponent, bfp8_mantissas = float_to_bfp8_block(block)
    bfp2_mantissas = truncate_bfp8(bfp8_mantissas, magnitude_bits=1)
    return shared_exponent, bfp2_mantissas


def pack_bfp2_b(tensor, block_size=16, num_faces=4, face_r_dim=16):
    """Pack tensor into BFP2_b format.

    BFP2_b uses 16-element blocks, each with a shared exponent and 2-bit mantissas
    (1 sign bit + 1 magnitude bit per element). Four mantissa datums are packed
    per byte (bits[1:0] = first element, bits[3:2] = second, bits[5:4] = third,
    bits[7:6] = fourth).

    Args:
        tensor: Input tensor (typically 1024 elements for full tile)
        block_size: Elements per block (always 16 for BFP2_b)
        num_faces: Number of faces to pack (1, 2, or 4)
        face_r_dim: Number of rows per face (1, 2, 4, 8, or 16)

    Returns:
        List of packed bytes: [exponents...] + [packed_mantissas...]
    """
    flattened_tensor = _bfp_prepare_blocks(tensor, block_size, num_faces, face_r_dim)
    exponents, all_mantissas = _bfp_collect_blocks(
        flattened_tensor, block_size, float_to_bfp2_block
    )

    packed_mantissas = []
    for i in range(0, len(all_mantissas), 4):
        e0 = all_mantissas[i] if i < len(all_mantissas) else 0
        e1 = all_mantissas[i + 1] if (i + 1) < len(all_mantissas) else 0
        e2 = all_mantissas[i + 2] if (i + 2) < len(all_mantissas) else 0
        e3 = all_mantissas[i + 3] if (i + 3) < len(all_mantissas) else 0
        packed_mantissas.append((e3 << 6) | (e2 << 4) | (e1 << 2) | e0)

    return exponents + packed_mantissas


# ============================================================================
# MX (Microscaling) Format Support - OCP Specification
# ============================================================================


def _pad_to_l1_alignment(data: list[int]) -> list[int]:
    """Pad a byte list to the next L1-aligned (16-byte) boundary."""
    aligned_len = l1_align(len(data))
    pad = aligned_len - len(data)
    return data if pad == 0 else data + [0] * pad


def _prepare_mx_blocks(
    tensor, *, num_faces, face_r_dim, data_format: DataFormat
) -> np.ndarray:
    """Flatten a tensor to fp32 and reshape into (num_blocks, MX_FORMAT_BLOCK_SIZE).

    Shared front-end for the MX-float pack paths (MxFp4/MxFp6/MxFp8). Validates
    that the requested geometry spans a whole number of 32-element OCP blocks and
    that the tensor holds enough elements, then returns the block-reshaped fp32
    view (NaN/Inf preserved for the block-scale / element quantizers).
    """
    fp32_array = tensor.cpu().to(torch.float32).numpy().flatten()
    elements_to_pack = face_r_dim * FACE_C_DIM * num_faces
    if len(fp32_array) < elements_to_pack:
        raise ValueError(
            f"{data_format}: tensor has {len(fp32_array)} elements, need "
            f"{elements_to_pack} for {num_faces} face(s)"
        )
    if elements_to_pack % MX_FORMAT_BLOCK_SIZE != 0:
        raise ValueError(
            f"{data_format} requires a block-aligned geometry: "
            f"elements_to_pack={elements_to_pack} is not a multiple of "
            f"MX_FORMAT_BLOCK_SIZE={MX_FORMAT_BLOCK_SIZE} "
            f"(face_r_dim={face_r_dim}, num_faces={num_faces})"
        )

    fp32_array = fp32_array[:elements_to_pack]
    num_blocks = elements_to_pack // MX_FORMAT_BLOCK_SIZE
    return fp32_array.reshape(num_blocks, MX_FORMAT_BLOCK_SIZE)


def _pack_mxfp8_srcs(tensor, *, data_format, exp_rnd_en=False, dest_acc: bool = False):
    """Pack a tensor into per-slice SrcS blocks for MXFP8 (mirrors _pack_mxfp6_srcs)."""
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
                data_format=data_format,
                num_faces=1,
                face_r_dim=slice_row_dim,
                exp_rnd_en=exp_rnd_en,
            )
        )
    return out


def _pack_mxfp8(tensor, *, data_format, num_faces=4, face_r_dim=16, exp_rnd_en=False):
    """Pack a full MXFP8R/MXFP8P tile with FULLY SEPARATED layout: [scales][elements].

    MXFP8 uses 32-element OCP blocks, each with one shared E8M0 scale and 32
    eight-bit elements (1 byte/element in L1). ``data_format`` selects the
    element format: ``DataFormat.MxFp8R`` (E5M2) or ``DataFormat.MxFp8P``
    (E4M3). Shares the block-scale and element-quantization model with
    MXFP6/MXFP4 (floor block exp with optional round-to-inf, RNE elements,
    saturate on overflow). E5M2/E4M3 additionally represent NaN (-> NaN);
    overflow resolves to the format max-normal (the OCP saturation default,
    i.e. FMT_CTRL_FP8_OVF_EN=0).

    Layout: 32 scales (32 B, aligned) + 1024 elements (aligned) -> 1056 B for a
    full tile. Element count must be a multiple of MX_FORMAT_BLOCK_SIZE (32).
    (The SrcS path, ``_pack_mxfp8_srcs``, slices the tile and delegates here.)
    """
    spec = MX_FP_SPECS[data_format]
    blocks_raw = _prepare_mx_blocks(
        tensor, num_faces=num_faces, face_r_dim=face_r_dim, data_format=data_format
    )

    scales_e8m0, scaled_blocks = _mxfp_block_scales(
        blocks_raw,
        elem_exp_max_unbiased=spec.exp_max_unbiased,
        exp_rnd_en=exp_rnd_en,
    )

    # E5M2/E4M3 element codes are a full byte each, stored directly in L1.
    elem_codes = _quantize_to_mx_fp_element_codes(
        scaled_blocks, **spec.element_quantizer_kwargs()
    )

    return _pad_to_l1_alignment(scales_e8m0) + _pad_to_l1_alignment(elem_codes.tolist())


def pack_mxfp8r(
    tensor,
    num_faces=4,
    face_r_dim=16,
    use_srcs: bool = False,
    dest_acc: bool = False,
    exp_rnd_en: bool = False,
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
            tensor,
            data_format=DataFormat.MxFp8R,
            exp_rnd_en=exp_rnd_en,
            dest_acc=dest_acc,
        )
    return _pack_mxfp8(
        tensor,
        data_format=DataFormat.MxFp8R,
        num_faces=num_faces,
        face_r_dim=face_r_dim,
        exp_rnd_en=exp_rnd_en,
    )


def pack_mxfp8p(
    tensor,
    num_faces=4,
    face_r_dim=16,
    use_srcs: bool = False,
    dest_acc: bool = False,
    exp_rnd_en: bool = False,
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
            tensor,
            data_format=DataFormat.MxFp8P,
            exp_rnd_en=exp_rnd_en,
            dest_acc=dest_acc,
        )
    return _pack_mxfp8(
        tensor,
        data_format=DataFormat.MxFp8P,
        num_faces=num_faces,
        face_r_dim=face_r_dim,
        exp_rnd_en=exp_rnd_en,
    )


def pack_mxfp4(
    tensor,
    num_faces=4,
    face_r_dim=16,
    use_srcs: bool = False,
    dest_acc: bool = False,
    exp_rnd_en: bool = False,
):
    """
    Pack tensor into MXFP4 format (E2M1 variant).
    Function is implemented based on the OCP MX specification and Tensix hardware documentation.

    MXFP4 uses 32-element blocks per OCP MX spec, each with:
    - 1 shared E8M0 scale (8 bits)
    - 32 × float4_e2m1fn elements (4 bits each = 16 bytes total)

    Element format E2M1:
    - 1 sign bit, 2 exponent bits (bias=1), 1 mantissa bit
    - Max normal: ±6.0
    - Min normal: ±1.0
    - Max/Min subnormal: ±0.5
    - No Inf or NaN support

    Per OCP MX spec Section 5.3.3 and Tensix hardware documentation:
    - Saturate on overflow, round to zero on underflow
    - NaN → Zero (per hardware spec)
    - Inf → Saturation with block_exp=0xFE (per hardware spec)

    Args:
        tensor: Input tensor (typically 1024 elements for full tile)
        num_faces: Number of faces to pack (1, 2, or 4). Defaults to 4.
        face_r_dim: Number of rows per face (1, 2, 4, 8, or 16). Defaults to 16.
        use_srcs: If True, split into SrcS slices (per-slice blocks in L1).
        dest_acc: If True (with use_srcs), use 32-bit SrcS slice geometry
            (4×16, 40 bytes/slice) instead of 16-bit (8×16, 72 bytes/slice).
        exp_rnd_en: If True, increment non-zero, non-special E8M0 scales to
            model FMT_CTRL_MX_BLOCK_EXP_RND_TO_INF behavior (default: disabled).

    Returns:
        List of packed bytes in FULLY SEPARATED layout: [all_scales][all_elements]
    """
    # For now, use_srcs is not implemented for MxFp4
    # If needed in the future, implement similar to _pack_mxfp8_srcs
    if use_srcs:
        raise NotImplementedError("use_srcs=True not yet implemented for pack_mxfp4")

    # Convert to numpy and reshape into (num_blocks, 32) blocks.
    blocks_raw = _prepare_mx_blocks(
        tensor, num_faces=num_faces, face_r_dim=face_r_dim, data_format=DataFormat.MxFp4
    )

    spec = MX_FP_SPECS[DataFormat.MxFp4]

    # Shared block scale (E8M0); elem_exp_max_unbiased is 2 for E2M1.
    scales_e8m0, scaled_blocks = _mxfp_block_scales(
        blocks_raw, elem_exp_max_unbiased=spec.exp_max_unbiased, exp_rnd_en=exp_rnd_en
    )

    # Convert to FP4 (E2M1) element codes.
    fp4_nibbles = _quantize_to_mx_fp_element_codes(
        scaled_blocks, **spec.element_quantizer_kwargs()
    )

    # Pack FP4 elements: 2 per byte (low nibble = element 0)
    packed_bytes = ((fp4_nibbles[1::2] & 0x0F) << 4) | (fp4_nibbles[0::2] & 0x0F)

    # FULLY SEPARATED layout: [scales padded to 16B][packed elements padded to 16B]
    return _pad_to_l1_alignment(scales_e8m0) + _pad_to_l1_alignment(
        packed_bytes.tolist()
    )


def _mxfp_block_scales(
    blocks_raw: np.ndarray, *, elem_exp_max_unbiased: int, exp_rnd_en: bool
) -> tuple[list[int], np.ndarray]:
    """Compute E8M0 block scales and the scaled (per-block) values for an MX-FP format.

    Shared by every MX floating-point pack path (MxFp4, MxFp6R/P). The shared
    exponent follows the MX block-scale model:
      shared_exp     = floor(log2(amax))               (over finite values)
      shared_exp_adj = max(shared_exp - elem_exp_max_unbiased, -127)
      E8M0           = shared_exp_adj + 127
    `elem_exp_max_unbiased` is the element format's max unbiased exponent
    (2 for E2M1/E2M3, 4 for E3M2). Special block scales mirror the HW spec:
    all-NaN block -> 0xFF; a block of only {Inf, NaN, 0} containing an Inf -> 0xFE.

    Args:
        blocks_raw: (num_blocks, MX_FORMAT_BLOCK_SIZE) float array (NaN/Inf intact).
        elem_exp_max_unbiased: element format max unbiased exponent.
        exp_rnd_en: model FMT_CTRL_MX_BLOCK_EXP_RND_TO_INF (increment non-special scales).

    Returns:
        (scales_e8m0 list[int], scaled_blocks) where scaled_blocks is each block
        divided by its decoded scale factor. NaN/Inf inputs are preserved (the
        element quantizer applies the per-format NaN/Inf rules).
    """
    # Max abs over finite values only (NaN/Inf ignored for scale selection).
    finite_blocks = np.where(np.isfinite(blocks_raw), blocks_raw, 0.0)
    max_abs_values = np.max(np.abs(finite_blocks), axis=1)

    # np.where evaluates both branches eagerly, so log2(0) is still computed for
    # all-zero blocks even though the result is discarded by the mask. Silence
    # the "divide by zero" RuntimeWarning since the mask handles it correctly.
    with np.errstate(divide="ignore"):
        max_abs_exp = np.where(
            max_abs_values == 0, 0, np.floor(np.log2(max_abs_values))
        )
    shared_exp_adj = np.where(
        (max_abs_exp - elem_exp_max_unbiased) >= -127,
        max_abs_exp - elem_exp_max_unbiased,
        -127,
    )
    scales_e8m0_array = shared_exp_adj.astype(np.int32) + 127

    # Special cases (per the HW spec):
    # - All NaN -> 0xFF (NaN block)
    # - Block contains only {Inf, NaN, 0} and has at least one Inf -> 0xFE
    all_nan_blocks = np.all(np.isnan(blocks_raw), axis=1)
    inf_or_zero_or_nan = np.isinf(blocks_raw) | np.isnan(blocks_raw) | (blocks_raw == 0)
    all_inf_or_zero = np.all(inf_or_zero_or_nan, axis=1)
    has_inf = np.any(np.isinf(blocks_raw), axis=1)

    scales_e8m0_array = np.where(all_nan_blocks, 255, scales_e8m0_array)
    scales_e8m0_array = np.where(all_inf_or_zero & has_inf, 254, scales_e8m0_array)

    if exp_rnd_en:
        # Match mx_block_exp_rnd_to_inf: increment only non-zero, non-special exps.
        can_inc = (
            (scales_e8m0_array != 0)
            & (scales_e8m0_array != 254)
            & (scales_e8m0_array != 255)
        )
        scales_e8m0_array = np.where(can_inc, scales_e8m0_array + 1, scales_e8m0_array)

    scales_e8m0 = scales_e8m0_array.astype(np.uint8).tolist()

    scale_factors = np.where(
        scales_e8m0_array == 255,
        np.nan,
        np.exp2(scales_e8m0_array.astype(np.float32) - 127.0),
    )
    scaled_blocks = blocks_raw / scale_factors[:, np.newaxis]
    return scales_e8m0, scaled_blocks


def _quantize_to_mx_fp_element_codes(
    scaled_blocks: np.ndarray,
    *,
    exp_bits: int,
    man_bits: int,
    exp_bias: int,
    exp_max_unbiased: int,
    exp_min_unbiased: int,
    man_max: int = None,
    nan_code: int = None,
) -> np.ndarray:
    """Quantize scaled values to packed MX-FP element codes (sign|exp|man).

    Models the hardware element quantization (round-ties-to-even, saturate on
    overflow, IEEE-style subnormals, Inf -> Saturation) for any SxEyMz element
    with ``exp_bits`` exponent and ``man_bits`` mantissa bits (no hidden bit):
    E2M1 (MxFp4), E3M2 (MxFp6R), E2M3 (MxFp6P), E5M2 (MxFp8R), E4M3 (MxFp8P).

    ``man_max`` is the largest mantissa of a *normal* element at the max
    exponent; it defaults to the full field ``(1 << man_bits) - 1`` but is one
    less for E4M3, where ``{exp=all-ones, man=all-ones}`` is reserved for NaN.

    ``nan_code`` is the element code a NaN input maps to (E5M2/E4M3 represent
    NaN). When ``None`` (E2M1/E3M2/E2M3, which have no NaN), NaN maps to +Zero.

    Returns a ``uint8`` array of element codes, each ``1 + exp_bits + man_bits``
    bits wide with the sign in the MSB. The caller owns the L1 bit layout
    (FP4 packs 2 codes/byte; FP6 stores ``code << 2`` per byte; FP8 is 1
    byte/element).
    """
    sign_shift = exp_bits + man_bits
    man_mask = (1 << man_bits) - 1
    if man_max is None:
        man_max = man_mask
    # Saturated element code = max-normal element (max biased exp, max-normal mantissa).
    sat_pos = ((exp_max_unbiased + exp_bias) << man_bits) | man_max
    sat_neg = sat_pos | (1 << sign_shift)
    shift_out = 23 - man_bits  # fp32 23-bit mantissa -> man_bits

    flat = scaled_blocks.astype(np.float32).ravel()
    ui32 = flat.view(np.uint32)
    sign = (ui32 >> 31) & 0x1
    exp_biased = (ui32 >> 23) & 0xFF
    mant = ui32 & 0x7FFFFF

    out = np.zeros_like(mant, dtype=np.uint8)

    is_nan = (exp_biased == 0xFF) & (mant != 0)
    is_inf = (exp_biased == 0xFF) & (mant == 0)
    is_zero = (exp_biased == 0) & (mant == 0)
    finite_nonzero = ~(is_nan | is_inf | is_zero)

    if np.any(is_inf):
        sat_vals = np.where(sign == 0, sat_pos, sat_neg).astype(np.uint8)
        out[is_inf] = sat_vals[is_inf]
    if np.any(is_zero):
        out[is_zero] = sign[is_zero].astype(np.uint8) << sign_shift
    # NaN -> format NaN code (E5M2/E4M3) or, for formats without NaN, +Zero
    # (out is already zero-initialised, so the None case needs no action).
    if nan_code is not None and np.any(is_nan):
        out[is_nan] = np.uint8(nan_code)

    if np.any(finite_nonzero):
        exp_unbiased = exp_biased.astype(np.int32) - 127

        def _round_ties_to_even(
            input_mantissa: np.ndarray,
        ) -> tuple[np.ndarray, np.ndarray]:
            rounded_bits = input_mantissa & ((1 << shift_out) - 1)
            rounded_msb = (rounded_bits >> (shift_out - 1)) & 0x1
            rounded_lsbs = rounded_bits & ((1 << (shift_out - 1)) - 1)
            mantissa_lsb = (input_mantissa >> shift_out) & 0x1
            round_inc = (
                (rounded_msb == 1) & ((rounded_lsbs != 0) | (mantissa_lsb == 1))
            ).astype(np.uint32)
            new_mantissa = (input_mantissa >> shift_out) + round_inc
            mant_round = new_mantissa & man_mask
            expo_inc = new_mantissa >> man_bits
            return mant_round.astype(np.int32), expo_inc.astype(np.int32)

        mant_round, expo_inc = _round_ties_to_even(mant)
        elem_exp_unbiased = exp_unbiased + expo_inc

        subnormal = elem_exp_unbiased < exp_min_unbiased
        if np.any(subnormal):
            mant_with_hb = mant | (1 << 23)
            shift = np.abs(exp_min_unbiased - exp_unbiased).astype(np.uint32)
            shift = np.minimum(shift, np.uint32(24))
            mant_exp_adjusted = mant_with_hb >> shift
            mant_round_sub, expo_inc_sub = _round_ties_to_even(mant_exp_adjusted)
            # Subnormal encoding: biased exp 0 unless rounding carried back to normal.
            elem_exp_unbiased_sub = -exp_bias + expo_inc_sub
            mant_round[subnormal] = mant_round_sub[subnormal]
            elem_exp_unbiased[subnormal] = elem_exp_unbiased_sub[subnormal]

        sat_mask = (elem_exp_unbiased > exp_max_unbiased) | (
            (elem_exp_unbiased == exp_max_unbiased) & (mant_round > man_max)
        )
        sat_mask &= finite_nonzero
        if np.any(sat_mask):
            sat_vals = np.where(sign == 0, sat_pos, sat_neg).astype(np.uint8)
            out[sat_mask] = sat_vals[sat_mask]

        normal_mask = finite_nonzero & ~sat_mask
        if np.any(normal_mask):
            elem_exp_biased = elem_exp_unbiased + exp_bias
            elem_bits = (
                (elem_exp_biased << man_bits) | (mant_round & man_mask)
            ).astype(np.uint8)
            elem_bits |= sign.astype(np.uint8) << sign_shift
            out[normal_mask] = elem_bits[normal_mask]

    return out


def _pack_mxfp6(tensor, *, data_format, num_faces=4, face_r_dim=16, exp_rnd_en=False):
    """Pack MXFP6R/MXFP6P with FULLY SEPARATED layout: [scales][elements].

    MXFP6 uses 32-element OCP blocks, each with one shared E8M0 scale and 32
    six-bit elements. Each element occupies one 8-bit L1 container with the 6-bit
    code in the upper bits and the low 2 bits zero (byte = code << 2), so the L1
    geometry matches MXFP8 (1 byte/element):
      - Full tile: 32 scales (32 B, aligned) + 1024 elements (aligned) -> 1056 B.

    ``data_format`` selects the element format: ``DataFormat.MxFp6R`` (E3M2) or
    ``DataFormat.MxFp6P`` (E2M3).
    """
    spec = MX_FP_SPECS[data_format]
    blocks_raw = _prepare_mx_blocks(
        tensor, num_faces=num_faces, face_r_dim=face_r_dim, data_format=data_format
    )

    scales_e8m0, scaled_blocks = _mxfp_block_scales(
        blocks_raw,
        elem_exp_max_unbiased=spec.exp_max_unbiased,
        exp_rnd_en=exp_rnd_en,
    )

    elem_codes = _quantize_to_mx_fp_element_codes(
        scaled_blocks, **spec.element_quantizer_kwargs()
    )

    # Each 6-bit element is stored in its own byte, shifted into the upper bits.
    elem_bytes = (elem_codes & 0x3F) << 2

    return _pad_to_l1_alignment(scales_e8m0) + _pad_to_l1_alignment(elem_bytes.tolist())


def _pack_mxfp6_srcs(tensor, *, data_format, exp_rnd_en=False, dest_acc: bool = False):
    """Pack a tensor into per-slice SrcS blocks for MXFP6 (mirrors _pack_mxfp8_srcs)."""
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
            _pack_mxfp6(
                flat[i : i + slice_elem_count],
                data_format=data_format,
                num_faces=1,
                face_r_dim=slice_row_dim,
                exp_rnd_en=exp_rnd_en,
            )
        )
    return out


def pack_mxfp6r(
    tensor,
    num_faces=4,
    face_r_dim=16,
    use_srcs: bool = False,
    dest_acc: bool = False,
    exp_rnd_en: bool = False,
):
    """
    Pack tensor into MXFP6R format (E3M2 variant).

    Element format E3M2:
    - 1 sign bit, 3 exponent bits (bias=3), 2 mantissa bits
    - Max normal: ±28.0, Min normal: ±0.25
    - Max/Min subnormal: ±0.1875 / ±0.0625
    - No Inf or NaN support
    - Stored in an 8-bit L1 container: {1b sign, 3b exp, 2b man, 2'b0}

    Args mirror pack_mxfp8r / pack_mxfp4. Returns the packed byte list in
    FULLY SEPARATED layout: [all_scales][all_elements].
    """
    assert tensor.numel() <= MAX_TILE_ELEMENTS, (
        f"pack_mxfp6r handles at most one tile ({MAX_TILE_ELEMENTS} elements), "
        f"got {tensor.numel()}"
    )
    if use_srcs:
        return _pack_mxfp6_srcs(
            tensor,
            data_format=DataFormat.MxFp6R,
            exp_rnd_en=exp_rnd_en,
            dest_acc=dest_acc,
        )
    return _pack_mxfp6(
        tensor,
        data_format=DataFormat.MxFp6R,
        num_faces=num_faces,
        face_r_dim=face_r_dim,
        exp_rnd_en=exp_rnd_en,
    )


def pack_mxfp6p(
    tensor,
    num_faces=4,
    face_r_dim=16,
    use_srcs: bool = False,
    dest_acc: bool = False,
    exp_rnd_en: bool = False,
):
    """
    Pack tensor into MXFP6P format (E2M3 variant).

    Element format E2M3:
    - 1 sign bit, 2 exponent bits (bias=1), 3 mantissa bits
    - Max normal: ±7.5, Min normal: ±1.0
    - Max/Min subnormal: ±0.875 / ±0.125
    - No Inf or NaN support
    - Stored in an 8-bit L1 container: {1b sign, 2b exp, 3b man, 2'b0}

    Args mirror pack_mxfp8p / pack_mxfp4. Returns the packed byte list in
    FULLY SEPARATED layout: [all_scales][all_elements].
    """
    assert tensor.numel() <= MAX_TILE_ELEMENTS, (
        f"pack_mxfp6p handles at most one tile ({MAX_TILE_ELEMENTS} elements), "
        f"got {tensor.numel()}"
    )
    if use_srcs:
        return _pack_mxfp6_srcs(
            tensor,
            data_format=DataFormat.MxFp6P,
            exp_rnd_en=exp_rnd_en,
            dest_acc=dest_acc,
        )
    return _pack_mxfp6(
        tensor,
        data_format=DataFormat.MxFp6P,
        num_faces=num_faces,
        face_r_dim=face_r_dim,
        exp_rnd_en=exp_rnd_en,
    )


def _mxint_block_scale_and_quantize(
    tensor,
    num_faces,
    face_r_dim,
    *,
    data_format: DataFormat,
):
    """
    Shared block-scale + symmetric quantization for MxInt formats.

    Computes the E8M0 shared exponent per block, then quantizes each scaled
    element to a signed int8 (the smaller-element formats clip to a narrower
    range via `elem_max` and reuse the int8 storage as a sign-extended carrier
    until the caller packs them at the actual bit width).

    ``data_format`` selects the MxInt element parameters from ``MX_INT_SPECS``:
      elem_scale: integer factor in `round(scaled * elem_scale)`, the format's
                  implicit 2^-k scale (64 for MxInt8's 2^-6; 4 for MxInt4's
                  2^-2; 1 for MxInt2's 2^0).
      elem_max:   symmetric clamp magnitude (127 for MxInt8; 7 for MxInt4;
                  1 for MxInt2).

    Returns: (scales_e8m0 as list[int], int_values as np.int8 array, shape (num_blocks, 32)).
    """
    spec = MX_INT_SPECS[data_format]
    elem_scale = spec.elem_scale
    elem_max = spec.elem_max
    fp32_array = tensor.cpu().to(torch.float32).numpy().flatten()
    elements_per_face = face_r_dim * FACE_C_DIM
    elements_to_pack = elements_per_face * num_faces
    assert len(fp32_array) >= elements_to_pack, (
        f"Tensor has {len(fp32_array)} elements, "
        f"need {elements_to_pack} for {num_faces} face(s)"
    )
    if elements_to_pack % MX_FORMAT_BLOCK_SIZE != 0:
        raise ValueError(
            f"{data_format} requires a block-aligned geometry: "
            f"elements_to_pack={elements_to_pack} is not a multiple of "
            f"MX_FORMAT_BLOCK_SIZE={MX_FORMAT_BLOCK_SIZE} "
            f"(face_r_dim={face_r_dim}, num_faces={num_faces})"
        )

    fp32_array = fp32_array[:elements_to_pack]
    num_blocks = len(fp32_array) // MX_FORMAT_BLOCK_SIZE
    blocks_raw = fp32_array.reshape(num_blocks, MX_FORMAT_BLOCK_SIZE)

    # Element-level NaN -> 0 (no NaN representation in MxInt).
    blocks = np.where(np.isnan(blocks_raw), 0.0, blocks_raw)

    # Block scale: shared_exp = floor(log2(amax)) over finite values. MxInt
    # post-scaling values land in [1, 2), so elem_exp_max_unbiased = 0.
    finite_blocks = np.where(np.isfinite(blocks_raw), blocks_raw, 0.0)
    max_abs_values = np.max(np.abs(finite_blocks), axis=1)
    # np.where evaluates both branches eagerly; silence log2(0) warnings for all-zero blocks.
    with np.errstate(divide="ignore"):
        max_abs_exp = np.where(
            max_abs_values == 0, 0, np.floor(np.log2(max_abs_values))
        )
    shared_exp_adj = np.where(max_abs_exp >= -127, max_abs_exp, -127)
    scales_e8m0_array = shared_exp_adj.astype(np.int32) + 127

    # Special-case block scales (mirror MxFp encoding).
    all_nan_blocks = np.all(np.isnan(blocks_raw), axis=1)
    inf_or_zero_or_nan = np.isinf(blocks_raw) | np.isnan(blocks_raw) | (blocks_raw == 0)
    all_inf_or_zero = np.all(inf_or_zero_or_nan, axis=1)
    has_inf = np.any(np.isinf(blocks_raw), axis=1)
    scales_e8m0_array = np.where(all_nan_blocks, 255, scales_e8m0_array)
    scales_e8m0_array = np.where(all_inf_or_zero & has_inf, 254, scales_e8m0_array)

    scales_e8m0 = scales_e8m0_array.astype(np.uint8).tolist()

    # Decode scale factors for applying to blocks (NaN scale -> NaN -> 0 below).
    scale_factors = np.where(
        scales_e8m0_array == 255,
        np.nan,
        np.exp2(scales_e8m0_array.astype(np.float32) - 127.0),
    )

    # Scale blocks; saturate Inf to ±2.0 and replace NaN with 0 so that int
    # conversion below can't overflow.
    scaled_blocks = blocks / scale_factors[:, np.newaxis]
    scaled_blocks = np.nan_to_num(scaled_blocks, nan=0.0, posinf=2.0, neginf=-2.0)

    # Quantize: int_val = round(scaled * elem_scale), symmetric clamp.
    int_values = np.rint(scaled_blocks * float(elem_scale)).astype(np.int32)
    int_values = np.clip(int_values, -elem_max, elem_max).astype(np.int8)
    return scales_e8m0, int_values


def pack_mxint8(
    tensor,
    num_faces=4,
    face_r_dim=16,
    use_srcs: bool = False,
    dest_acc: bool = False,
):
    """
    Pack tensor into MxInt8 format (signed S1.6 elements with E8M0 block scale).

    MxInt8 uses 32-element blocks per OCP MX spec, each with:
    - 1 shared E8M0 scale (8 bits)
    - 32 signed-int8 elements (8 bits each), interpreted as 2's complement with
      an implicit 2^-6 scale. Range: ±127/64 ≈ ±1.984 (symmetric — the −2
      encoding 0x80 is left unused per OCP "preserve symmetry" guidance).

    Per OCP MX spec Section 5.3.4 and Tensix hardware documentation:
    - NaN element → 0 (MxInt has no NaN representation)
    - Inf element → saturating clamp (symmetric mode, no edge-mask -0)
    - Out-of-range after rounding → saturate to ±127

    Block-scale special cases match MxFp encoding:
    - All-NaN block → scale = 0xFF (unpacker yields zero block)
    - Block with any Inf (else all-zero/Inf) → scale = 0xFE (max scale)
    """
    if use_srcs:
        raise NotImplementedError("use_srcs=True not yet implemented for pack_mxint8")

    scales_e8m0, int_values = _mxint_block_scale_and_quantize(
        tensor,
        num_faces,
        face_r_dim,
        data_format=DataFormat.MxInt8,
    )

    # Layout: [scales padded to 16B][int8 elements padded to 16B].
    return _pad_to_l1_alignment(scales_e8m0) + _pad_to_l1_alignment(
        list(int_values.tobytes())
    )


def pack_mxint4(
    tensor,
    num_faces=4,
    face_r_dim=16,
    use_srcs: bool = False,
    dest_acc: bool = False,
):
    """
    Pack tensor into MxInt4 format (signed S1.2 elements with E8M0 block scale).

    MxInt4 uses 32-element blocks, each with:
    - 1 shared E8M0 scale (8 bits)
    - 32 signed-int4 elements (4 bits each, 2 packed per byte), 2's complement
      with an implicit 2^-2 scale. Range: ±7/4 = ±1.75 (symmetric — the -8
      encoding 0b1000 is left unused).

    Element-pair layout per byte: low nibble = element at even index,
    high nibble = element at odd index (matches MxFp4 convention).

    Special-case handling (NaN→0 element, Inf→saturate, all-NaN/Inf block scale)
    mirrors pack_mxint8.
    """
    if use_srcs:
        raise NotImplementedError("use_srcs=True not yet implemented for pack_mxint4")

    scales_e8m0, int_values = _mxint_block_scale_and_quantize(
        tensor,
        num_faces,
        face_r_dim,
        data_format=DataFormat.MxInt4,
    )

    # Pack 2 nibbles per byte: low nibble = even index, high nibble = odd index.
    # int_values is signed int8 in [-7, +7]; mask to 4-bit 2's complement.
    nibbles = int_values.flatten().astype(np.uint8) & 0x0F
    packed_bytes = ((nibbles[1::2] & 0x0F) << 4) | (nibbles[0::2] & 0x0F)

    # Layout: [scales padded to 16B][packed nibbles padded to 16B].
    return _pad_to_l1_alignment(scales_e8m0) + _pad_to_l1_alignment(
        packed_bytes.tolist()
    )


def pack_mxint2(
    tensor,
    num_faces=4,
    face_r_dim=16,
    use_srcs: bool = False,
    dest_acc: bool = False,
):
    """
    Pack tensor into MxInt2 format (signed S1.0 elements with E8M0 block scale).

    MxInt2 uses 32-element blocks, each with:
    - 1 shared E8M0 scale (8 bits)
    - 32 signed-int2 elements (2 bits each, 4 packed per byte), 2's complement
      with an implicit 2^0 scale. Range: ±1 (symmetric — the -2 encoding 0b10
      is left unused). Only three element values are representable: -1, 0, +1.

    Element-quad layout per byte (low bits to high): bits[1:0] = element at
    index i, bits[3:2] = i+1, bits[5:4] = i+2, bits[7:6] = i+3. This mirrors
    MxInt4's even-index-in-low convention extended to four elements.

    Special-case handling (NaN→0 element, Inf→saturate, all-NaN/Inf block scale)
    mirrors pack_mxint8.
    """
    if use_srcs:
        raise NotImplementedError("use_srcs=True not yet implemented for pack_mxint2")

    scales_e8m0, int_values = _mxint_block_scale_and_quantize(
        tensor,
        num_faces,
        face_r_dim,
        data_format=DataFormat.MxInt2,
    )

    # Pack 4 crumbs per byte: bits[1:0]=i, [3:2]=i+1, [5:4]=i+2, [7:6]=i+3.
    # int_values is signed int8 in [-1, +1]; mask to 2-bit 2's complement.
    crumbs = int_values.flatten().astype(np.uint8) & 0x03
    packed_bytes = (
        (crumbs[0::4] & 0x03)
        | ((crumbs[1::4] & 0x03) << 2)
        | ((crumbs[2::4] & 0x03) << 4)
        | ((crumbs[3::4] & 0x03) << 6)
    )

    # Layout: [scales padded to 16B][packed crumbs padded to 16B].
    return _pad_to_l1_alignment(scales_e8m0) + _pad_to_l1_alignment(
        packed_bytes.tolist()
    )
