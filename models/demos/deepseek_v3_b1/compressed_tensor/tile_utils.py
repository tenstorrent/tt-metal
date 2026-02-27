# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from functools import lru_cache

import numpy as np
import torch

import ttnn

from .metrics import metric_value, pearson_corr

COMPRESSED_FORMATS = ["bf16", "bfp8", "bfp4", "bfp2"]
COMPRESSED_BYTES_PER_ELEM = {
    "bf16": 2.0,
    "bfp8": 1.088,
    "bfp4": 0.50097,
    "bfp2": 0.25097,
}


def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


@lru_cache(maxsize=None)
def _ttnn_bfp_decode_table(mant_bits: int) -> tuple[np.ndarray, np.ndarray]:
    """Build lookup tables for BFP mantissa normalization (matches ttnn HW decode)."""
    mask = (1 << mant_bits) - 1
    shift_cnt = np.zeros(mask + 1, dtype=np.uint32)
    man_shifted = np.zeros(mask + 1, dtype=np.uint32)
    for man in range(1, mask + 1):
        msb_pos = int(np.floor(np.log2(man)))
        shift = (mant_bits - 1) - msb_pos
        shift_cnt[man] = shift
        man_shifted[man] = (man << (shift + 1)) & mask
    return shift_cnt, man_shifted


def quantize_dequantize_bfp(x: np.ndarray, mant_bits: int) -> np.ndarray:
    """Quantize-dequantize via BFP format with exact ttnn-matching bit manipulation.

    Supports any mant_bits: 7 (bfp8), 3 (bfp4), 1 (bfp2).
    Shared exponent is computed per face row (16 elements) within 32x32 tiles.

    Ported from github.com/johanna-rock/quantization_analysis.
    """
    x = np.asarray(x, dtype=np.float32)
    if x.size == 0:
        return x.astype(np.float32)

    orig_shape = x.shape
    if x.ndim == 0:
        batch, height, width = 1, 1, 1
        x = x.reshape(batch, height, width)
    elif x.ndim == 1:
        batch, height, width = 1, 1, x.shape[0]
        x = x.reshape(batch, height, width)
    else:
        height, width = x.shape[-2], x.shape[-1]
        batch = int(np.prod(x.shape[:-2])) if x.ndim > 2 else 1
        x = x.reshape(batch, height, width)

    tile_h = 32
    tile_w = 32
    pad_h = _ceil_div(height, tile_h) * tile_h
    pad_w = _ceil_div(width, tile_w) * tile_w

    x_pad = np.zeros((batch, pad_h, pad_w), dtype=np.float32)
    x_pad[:, :height, :width] = x

    if pad_h == 0 or pad_w == 0:
        return np.zeros(orig_shape, dtype=np.float32)

    tiles_h = pad_h // tile_h
    tiles_w = pad_w // tile_w

    # Reshape into faces: (batch, tiles_h, 2, 16, tiles_w, 2, 16)
    # Shared exponent is per face row (last axis of 16 elements)
    x_faces = x_pad.reshape(batch, tiles_h, 2, 16, tiles_w, 2, 16)
    u32 = x_faces.view(np.uint32)

    exp = (u32 >> 23) & 0xFF
    shared_exp = exp.max(axis=-1, keepdims=True)

    mantissa = u32 & 0x007FFFFF
    sign = (u32 >> 31) & 0x1
    zero_or_denorm = exp == 0

    # Restore implicit leading 1
    mantissa = (1 << 23) | mantissa
    exp_diff = shared_exp.astype(np.uint32) - exp.astype(np.uint32)
    while np.any(exp_diff > 31):
        mask_big = exp_diff > 31
        mantissa = np.where(mask_big, mantissa >> 31, mantissa)
        exp_diff = np.where(mask_big, exp_diff - 31, exp_diff)
    mantissa = mantissa >> exp_diff

    # Round-to-nearest-even
    shift = 24 - mant_bits
    round_mask = (1 << shift) - 1
    tie_value = 1 << (shift - 1)
    round_value = mantissa & round_mask
    mantissa = mantissa >> shift
    guard_bit = mantissa & 0x1
    round_up = (round_value > tie_value) | ((round_value == tie_value) & (guard_bit == 1))
    mantissa = mantissa + round_up.astype(np.uint32)
    mantissa = np.minimum(mantissa, (1 << mant_bits) - 1).astype(np.uint32)

    sign = np.where(mantissa == 0, 0, sign)
    code = (sign << mant_bits) | mantissa
    code = np.where(zero_or_denorm, 0, code).astype(np.uint32)

    # Decode: normalize mantissa (matches ttnn HW unpack)
    mask = (1 << mant_bits) - 1
    man = code & mask
    sign = code >> mant_bits
    shift_cnt_table, man_shifted_table = _ttnn_bfp_decode_table(mant_bits)
    shift_cnt = shift_cnt_table[man]
    man_shifted = man_shifted_table[man]

    exp_out = shared_exp.astype(np.uint32) - shift_cnt.astype(np.uint32)
    exp_out = np.where(man == 0, 0, exp_out).astype(np.uint32)

    mant_shift = 23 - mant_bits
    u32_out = (sign << 31) | (exp_out << 23) | (man_shifted << mant_shift)
    y_pad = u32_out.view(np.float32).reshape(x_pad.shape)

    y = y_pad[:, :height, :width]
    if orig_shape == ():
        return np.array(y[0, 0, 0], dtype=np.float32)
    return y.reshape(orig_shape)


TTNN_DTYPE_MAP = {
    "bf16": ttnn.bfloat16,
    "bfp8": ttnn.bfloat8_b,
    "bfp4": ttnn.bfloat4_b,
}

BFP_MANT_BITS = {"bfp8": 7, "bfp4": 3, "bfp2": 1}


def ttnn_quantize_fn(x: torch.Tensor, fmt: str) -> torch.Tensor:
    """Quantize-dequantize round trip. Uses ttnn for bfp8/bfp4, numpy emulation for bfp2."""
    if fmt in BFP_MANT_BITS and fmt not in TTNN_DTYPE_MAP:
        xn = x.detach().float().cpu().numpy()
        q = quantize_dequantize_bfp(xn, mant_bits=BFP_MANT_BITS[fmt])
        return torch.from_numpy(q).to(device=x.device)
    tt_dtype = TTNN_DTYPE_MAP[fmt]
    tt_tensor = ttnn.from_torch(x, dtype=tt_dtype, layout=ttnn.TILE_LAYOUT)
    return ttnn.to_torch(tt_tensor).to(dtype=torch.float32)


def bfp_tile_packed_size(mant_bits: int, tile_hw: int = 32) -> int:
    """Return the packed byte size of a single BFP tile.

    Layout: [exponent bytes] [packed mantissa+sign data bytes]
    - Exponents: 1 byte per face row. 4 faces × 16 rows = 64 bytes.
    - Data: (1 + mant_bits) bits per element, packed into bytes.
      bfp8 (mant_bits=7): 8 bits × 1024 = 1024 bytes → total 1088
      bfp4 (mant_bits=3): 4 bits × 1024 =  512 bytes → total  576
      bfp2 (mant_bits=1): 2 bits × 1024 =  256 bytes → total  320
    """
    num_elements = tile_hw * tile_hw  # 1024
    num_faces = 4
    face_h = tile_hw // 2  # 16
    exp_bytes = num_faces * face_h  # 64
    bits_per_element = 1 + mant_bits  # sign + mantissa
    data_bytes = (num_elements * bits_per_element) // 8
    return exp_bytes + data_bytes


# C++ pack/unpack bindings (SIMD-optimized)
_bfp = ttnn._ttnn.bfp_utils

_PACK_FN = {
    7: _bfp.pack_bfp8,
    3: _bfp.pack_bfp4,
}

_UNPACK_FN = {
    7: _bfp.unpack_bfp8,
    3: _bfp.unpack_bfp4,
}


def pack_bfp_tile(tile: np.ndarray, mant_bits: int) -> np.ndarray:
    """Pack a single 32×32 float32 tile into raw BFP bytes via C++ SIMD pack.

    Returns uint8 view of the packed uint32 data.
    """
    assert tile.shape == (32, 32), f"Expected (32, 32), got {tile.shape}"
    pack_fn = _PACK_FN[mant_bits]
    # C++ expects tile-ordered flat float32; tile is already 32×32
    packed_u32 = np.asarray(pack_fn(tile.astype(np.float32).ravel()))
    return packed_u32.view(np.uint8)


def unpack_bfp_tile(packed: np.ndarray, mant_bits: int) -> np.ndarray:
    """Unpack raw BFP bytes back to a 32×32 float32 tile via C++ SIMD unpack."""
    unpack_fn = _UNPACK_FN[mant_bits]
    packed_u32 = packed.view(np.uint32)
    unpacked = np.asarray(unpack_fn(packed_u32))
    return unpacked.reshape(32, 32)


def pack_compressed_tiles(
    data: torch.Tensor, assignment: np.ndarray, tile_hw: int = 32
) -> tuple[torch.Tensor, list[int]]:
    """Pack a 2D float32 tensor into a flat uint8 tensor with per-tile BFP formats.

    Args:
        data: Float32 torch tensor of shape (H, W), must be tile-aligned.
        assignment: (tiles_h, tiles_w) int array of format indices into COMPRESSED_FORMATS.
        tile_hw: Tile dimension (default 32).

    Returns:
        (flat_buffer, tile_mant_bits):
          - flat_buffer: uint8 torch tensor of concatenated packed tiles.
          - tile_mant_bits: list of mant_bits per tile in row-major order.
    """
    data_np = data.detach().float().cpu().numpy()
    tiles_h, tiles_w = assignment.shape
    chunks = []
    tile_mant_bits = []
    for tr in range(tiles_h):
        for tc in range(tiles_w):
            tile = data_np[tr * tile_hw : (tr + 1) * tile_hw, tc * tile_hw : (tc + 1) * tile_hw]
            fmt_name = COMPRESSED_FORMATS[assignment[tr, tc]]
            mant_bits = BFP_MANT_BITS[fmt_name]
            tile_mant_bits.append(mant_bits)
            chunks.append(pack_bfp_tile(tile, mant_bits))
    flat_np = np.concatenate(chunks)
    return torch.from_numpy(flat_np.copy()), tile_mant_bits


def unpack_compressed_tiles(
    flat_buffer: torch.Tensor, tile_mant_bits: list[int], tiles_h: int, tiles_w: int, tile_hw: int = 32
) -> torch.Tensor:
    """Unpack a flat uint8 tensor back into a 2D float32 tensor.

    Args:
        flat_buffer: uint8 torch tensor from pack_compressed_tiles.
        tile_mant_bits: list of mant_bits per tile in row-major order.
        tiles_h: Number of tile rows.
        tiles_w: Number of tile columns.
        tile_hw: Tile dimension (default 32).

    Returns:
        Float32 torch tensor of shape (tiles_h * tile_hw, tiles_w * tile_hw).
    """
    flat_np = flat_buffer.numpy().astype(np.uint8)
    out = np.zeros((tiles_h * tile_hw, tiles_w * tile_hw), dtype=np.float32)
    offset = 0
    idx = 0
    for tr in range(tiles_h):
        for tc in range(tiles_w):
            mant_bits = tile_mant_bits[idx]
            size = bfp_tile_packed_size(mant_bits)
            tile = unpack_bfp_tile(flat_np[offset : offset + size], mant_bits)
            out[tr * tile_hw : (tr + 1) * tile_hw, tc * tile_hw : (tc + 1) * tile_hw] = tile
            offset += size
            idx += 1
    return torch.from_numpy(out)


def compressed_total_bytes(counts: dict[str, int], tile_hw: int = 32) -> float:
    total = 0.0
    elems_per_tile = float(tile_hw * tile_hw)
    for fmt, count in counts.items():
        total += float(count) * elems_per_tile * COMPRESSED_BYTES_PER_ELEM.get(fmt, 0.0)
    return total


def tile_metrics(ref_tiles: np.ndarray, q_tiles: np.ndarray, metric: str) -> np.ndarray:
    if metric == "pcc":
        scores = []
        for i in range(ref_tiles.shape[0]):
            scores.append(pearson_corr(ref_tiles[i], q_tiles[i]))
        return np.asarray(scores, dtype=np.float32)
    diff = np.abs(ref_tiles - q_tiles)
    if metric == "mae":
        return diff.reshape(diff.shape[0], -1).mean(axis=1)
    if metric == "atol":
        return diff.reshape(diff.shape[0], -1).max(axis=1)
    raise ValueError(f"Unsupported metric: {metric}")


def reshape_to_2d_with_padding(xf: np.ndarray) -> tuple[np.ndarray, tuple, tuple]:
    xf = np.asarray(xf, dtype=np.float32)
    if xf.ndim == 0:
        data2d = xf.reshape(1, 1)
        shape_info = ("scalar", xf.shape)
    elif xf.ndim == 1:
        n = xf.shape[0]
        h = int(np.ceil(n / 32.0))
        w = 32
        data2d = np.zeros((h, w), dtype=np.float32)
        data2d.reshape(-1)[:n] = xf.reshape(-1)
        shape_info = ("vector", n)
    else:
        w = xf.shape[-1]
        h = int(np.prod(xf.shape[:-1]))
        data2d = xf.reshape(h, w)
        shape_info = ("nd", xf.shape)

    h, w = data2d.shape
    h_pad = int(np.ceil(h / 32.0)) * 32
    w_pad = int(np.ceil(w / 32.0)) * 32
    padded = np.zeros((h_pad, w_pad), dtype=np.float32)
    padded[:h, :w] = data2d
    pad_info = (h, w, h_pad, w_pad)
    return padded, shape_info, pad_info


def reconstruct_from_tiles(tiles: np.ndarray, shape_info: tuple, pad_info: tuple, tile_hw: int = 32) -> np.ndarray:
    h, w, h_pad, w_pad = pad_info
    tiles_h = h_pad // tile_hw
    tiles_w = w_pad // tile_hw
    padded = tiles.reshape(tiles_h, tiles_w, tile_hw, tile_hw).transpose(0, 2, 1, 3).reshape(h_pad, w_pad)
    data2d = padded[:h, :w]
    if shape_info[0] == "scalar":
        return np.array(data2d[0, 0], dtype=np.float32)
    if shape_info[0] == "vector":
        n = shape_info[1]
        return data2d.reshape(-1)[:n].astype(np.float32)
    if shape_info[0] == "nd":
        orig_shape = shape_info[1]
        return data2d.reshape(orig_shape).astype(np.float32)
    raise ValueError("Invalid shape_info")


def global_metric(xf: np.ndarray, tiles: np.ndarray, shape_info: tuple, pad_info: tuple, metric: str) -> float:
    y = reconstruct_from_tiles(tiles, shape_info, pad_info)
    return metric_value(xf, y, metric)
