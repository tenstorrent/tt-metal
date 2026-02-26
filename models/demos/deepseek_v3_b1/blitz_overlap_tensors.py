from dataclasses import dataclass

import numpy as np
import torch

import ttnn


@dataclass
class OverlappedShardSpec:
    """Describes one sub-tensor within a fused WIDTH_SHARDED raw-byte buffer.

    Multiple ``OverlappedShardSpec`` instances share a single device
    allocation whose per-core shard is zero-padded to a common maximum
    byte size.  Each spec carries the core range, logical tensor shape,
    dtype, tile dimensions, and byte offset needed to address its
    portion of the shard.

    Tile byte sizes for BFP formats are computed from ``tile_h`` and
    ``tile_w`` rather than stored as constants, so non-standard tile
    shapes (e.g. 1x32, 16x32) are handled automatically.

    Shape tuples follow (height, width) convention.
    """

    core_range_set: ttnn.CoreRangeSet
    raw_tensor_shape: tuple[int, int]
    dtype: ttnn.DataType
    # byte offset within a core shard
    byte_offset: int = 0

    tile_h: int = 32
    tile_w: int = 32
    bfp8_tile_bytes: int = 1088
    bfp4_tile_bytes: int = 576

    def _tile_bytes(self) -> int:
        num_elements = self.tile_h * self.tile_w
        if self.dtype in (ttnn.DataType.BFLOAT8_B, ttnn.DataType.BFLOAT4_B):
            num_blocks = num_elements // 16
            exponent_bytes = (num_blocks + 3) // 4 * 4
            mantissa_bytes = num_elements if self.dtype == ttnn.DataType.BFLOAT8_B else num_elements // 2
            return exponent_bytes + mantissa_bytes
        else:
            return num_elements * self.dtype.size()

    @property
    def tiles_per_shard(self) -> int:
        shard_w = self.raw_tensor_shape[1] // self.core_range_set.num_cores()
        return (self.raw_tensor_shape[0] // self.tile_h) * (shard_w // self.tile_w)

    @property
    def shard_bytes(self) -> int:
        return self.tiles_per_shard * self._tile_bytes()

    @property
    def total_bytes(self) -> int:
        return self.shard_bytes * self.core_range_set.num_cores()


def max_shard_bytes(shard_specs: list[OverlappedShardSpec]) -> int:
    return max(spec.shard_bytes for spec in shard_specs)


def tilize_and_pack_bfp8(data_2d: torch.Tensor, tile_h: int = 32, tile_w: int = 32) -> bytes:
    """Tilize a 2-D tensor and pack as BFP8_b raw bytes.

    Produces the exact byte layout the hardware expects:
    ``[16 exponent uint32 words][256 mantissa uint32 words]`` per tile,
    with tiles in row-major order across the tensor.

    Matches the C++ ``pack_as_bfp_tiles<Bfp8_b>`` with
    ``row_major_input=false, is_exp_a=false``:

    * Shared exponent = max float32 exponent in each 16-element block.
    * Per-element mantissa = 24-bit explicit mantissa (hidden-1 + 23
        fractional bits), right-shifted by the exponent delta *and* by
        17 (``24 - 7``) to yield 7 bits, with round-to-nearest-even.
    * Packed byte = ``sign(1) | mantissa(7)``, zeroed when mantissa
        rounds to 0.

    Tile layout: 4 faces (16x16) in order
    face0 (rows 0-15, cols 0-15), face1 (rows 0-15, cols 16-31),
    face2 (rows 16-31, cols 0-15), face3 (rows 16-31, cols 16-31).
    Each face row of 16 elements forms one BFP8 block.
    """
    H, W = data_2d.shape
    face_h, face_w = tile_h // 2, tile_w // 2
    tr, tc = H // tile_h, W // tile_w
    num_tiles = tr * tc

    data_np = data_2d.contiguous().float().numpy()

    # Reshape into tile grid -> (tr, tc, tile_h, tile_w)
    tiles = data_np.reshape(tr, tile_h, tc, tile_w).transpose(0, 2, 1, 3)
    tiles = tiles.reshape(num_tiles, tile_h, tile_w)

    # Extract 4 faces per tile -> face-ordered (N, 1024)
    face_ordered = np.concatenate(
        [
            tiles[:, :face_h, :face_w].reshape(num_tiles, -1),
            tiles[:, :face_h, face_w:].reshape(num_tiles, -1),
            tiles[:, face_h:, :face_w].reshape(num_tiles, -1),
            tiles[:, face_h:, face_w:].reshape(num_tiles, -1),
        ],
        axis=1,
    )

    # Reshape into BFP8 blocks: (N, 64 blocks, 16 elements)
    blocks = face_ordered.reshape(num_tiles, 64, 16)

    # --- float32 field extraction (vectorised) ---
    float_bits = blocks.view(np.uint32)
    signs = ((float_bits >> 31) & 1).astype(np.uint8)
    exponents = ((float_bits >> 23) & 0xFF).astype(np.int32)
    mantissa23 = (float_bits & 0x007F_FFFF).astype(np.int32)

    # 24-bit explicit mantissa with hidden 1; zero for denormals
    explicit_mant = np.where(exponents == 0, np.int32(0), np.int32(1 << 23) | mantissa23)

    # Shared exponent = max float32 exponent in each 16-element block
    shared_exp = np.max(exponents, axis=2)  # (N, 64)

    # Shift mantissa by exponent delta, then by 17 to get 7-bit result
    delta = shared_exp[:, :, np.newaxis] - exponents
    shifted = explicit_mant >> np.minimum(delta, 31)

    MANT_SHIFT = 17  # 24 - 7
    ROUND_MASK = (1 << MANT_SHIFT) - 1
    TIE = np.int32(1 << (MANT_SHIFT - 1))

    round_value = shifted & ROUND_MASK
    mantissa7 = (shifted >> MANT_SHIFT).astype(np.int32)
    guard_bit = mantissa7 & 1
    round_up = (round_value > TIE) | ((round_value == TIE) & (guard_bit == 1))
    mantissa7 = np.where(round_up, np.minimum(mantissa7 + 1, 127), mantissa7)

    # Zero sign when mantissa is zero
    signs = np.where(mantissa7 == 0, np.uint8(0), signs)
    packed_mant = ((signs.astype(np.int32) << 7) | (mantissa7 & 0x7F)).astype(np.uint8)

    # Assemble per-tile bytes: [exp words][mant words]
    exp_bytes = shared_exp.astype(np.uint8)  # (N, 64)
    exp_words = exp_bytes.view(np.uint32).reshape(num_tiles, 16)

    mant_words = packed_mant.reshape(num_tiles, 1024).view(np.uint32).reshape(num_tiles, 256)

    tile_words = np.concatenate([exp_words, mant_words], axis=1)  # (N, 272)
    return tile_words.tobytes()


def tilize_and_pack_bfloat16(data_2d: torch.Tensor, tile_h: int = 32, tile_w: int = 32) -> bytes:
    """Tilize a 2-D tensor and pack as bfloat16 (Float16_b) raw bytes.

    Each tile is 2048 bytes: 1024 elements x 2 bytes, stored in
    face order (face0, face1, face2, face3), row-major within each
    face.  bfloat16 is the top 16 bits of IEEE-754 float32.
    """
    H, W = data_2d.shape
    face_h, face_w = tile_h // 2, tile_w // 2
    tr, tc = H // tile_h, W // tile_w
    num_tiles = tr * tc

    data_np = data_2d.contiguous().float().numpy()

    tiles = data_np.reshape(tr, tile_h, tc, tile_w).transpose(0, 2, 1, 3)
    tiles = tiles.reshape(num_tiles, tile_h, tile_w)

    face_ordered = np.concatenate(
        [
            tiles[:, :face_h, :face_w].reshape(num_tiles, -1),
            tiles[:, :face_h, face_w:].reshape(num_tiles, -1),
            tiles[:, face_h:, :face_w].reshape(num_tiles, -1),
            tiles[:, face_h:, face_w:].reshape(num_tiles, -1),
        ],
        axis=1,
    )  # (N, 1024)

    # bfloat16 = top 16 bits of float32
    float_bits = face_ordered.view(np.uint32)
    bf16_bits = (float_bits >> 16).astype(np.uint16)
    return bf16_bits.tobytes()


def pack_bfloat16_1x32(data: torch.Tensor) -> bytes:
    """Pack a 1-row tensor as raw bfloat16 bytes with 1×32 tile layout.

    For 1×32 tiles there is no face reordering; elements are stored
    sequentially in tile-width chunks.  bfloat16 is the top 16 bits
    of IEEE-754 float32.
    """
    flat = data.contiguous().float().reshape(-1).numpy()
    float_bits = flat.view(np.uint32)
    bf16_bits = (float_bits >> 16).astype(np.uint16)
    return bf16_bits.tobytes()


def stitch_width_sharded(
    tensor1: torch.Tensor,
    tensor2: torch.Tensor,
    num_cores: int,
    tile_h: int = 32,
    tile_w: int = 32,
) -> tuple[torch.Tensor, tuple[int, int]]:
    """Stitch two width-sharded tensors into one fused tensor.

    For every core the two shards are concatenated vertically.  When
    the shard widths differ, the wider shard is tile-reshaped to match
    the narrower one (preserving tile ordering) so the concatenation
    is well-defined.

    Args:
        tensor1: First weight tensor (H1, W1).
        tensor2: Second weight tensor (H2, W2).
        num_cores: Total cores in the width-sharded grid.
        tile_h: Tile height (default 32).
        tile_w: Tile width (default 32).

    Returns:
        (fused_tensor, shard_shape) ready for WIDTH_SHARDED
        placement on num_cores cores.
    """
    H1, W1 = tensor1.shape
    H2, W2 = tensor2.shape

    shard_w1 = W1 // num_cores
    shard_w2 = W2 // num_cores

    # Use the narrower shard width as target; tile-reshape the wider.
    if shard_w1 <= shard_w2:
        target_w = shard_w1
        narrow, wide = tensor1, tensor2
        narrow_h, wide_h = H1, H2
        narrow_sw, wide_sw = shard_w1, shard_w2
    else:
        target_w = shard_w2
        narrow, wide = tensor2, tensor1
        narrow_h, wide_h = H2, H1
        narrow_sw, wide_sw = shard_w2, shard_w1

    # Height of each wide shard after tile-reshape to target_w
    reshaped_h = wide_h * wide_sw // target_w

    fused_shard_h = narrow_h + reshaped_h
    fused = torch.zeros(fused_shard_h, target_w * num_cores, dtype=tensor1.dtype)

    for core_idx in range(num_cores):
        col_start = core_idx * target_w
        col_end = col_start + target_w

        # Narrow shard: already at target width, just copy.
        n_start = core_idx * narrow_sw
        n_end = n_start + narrow_sw
        fused[:narrow_h, col_start:col_end] = narrow[:, n_start:n_end]

        # Wide shard: tile-reshape from (wide_h, wide_sw) to
        # (reshaped_h, target_w), then copy.
        w_start = core_idx * wide_sw
        w_end = w_start + wide_sw
        w_shard = wide[:, w_start:w_end]
        w_reshaped = tile_reshape(
            w_shard,
            src_shape=(wide_h, wide_sw),
            dst_shape=(reshaped_h, target_w),
            tile_h=tile_h,
            tile_w=tile_w,
        )
        fused[narrow_h:, col_start:col_end] = w_reshaped

    shard_shape = (fused_shard_h, target_w)
    return fused, shard_shape


def tile_reshape(
    tensor: torch.Tensor,
    src_shape: tuple[int, int],
    dst_shape: tuple[int, int],
    tile_h: int = 32,
    tile_w: int = 32,
) -> torch.Tensor:
    """Reshape a 2-D tensor while preserving row-major tile ordering.

    Data is stored as a grid of (tile_h x tile_w) tiles in row-major
    order.  A naive torch.reshape changes which values land in each
    tile.  This helper keeps every tile's contents unchanged by:

    1. Splitting into the source tile grid.
    2. Flattening to a 1-D tile sequence (row-major).
    3. Re-gridding into the destination tile dimensions.

    Total tile count must be identical for source and destination.
    """
    src_h, src_w = src_shape
    dst_h, dst_w = dst_shape
    src_tr, src_tc = src_h // tile_h, src_w // tile_w
    dst_tr, dst_tc = dst_h // tile_h, dst_w // tile_w
    assert src_tr * src_tc == dst_tr * dst_tc, f"Tile count mismatch: {src_tr * src_tc} vs {dst_tr * dst_tc}"
    # (H, W) -> (tile_rows, tile_h, tile_cols, tile_w)
    #         -> (tile_rows, tile_cols, tile_h, tile_w)
    tiles = tensor.reshape(src_tr, tile_h, src_tc, tile_w).permute(0, 2, 1, 3)
    # Flatten to 1-D tile sequence, re-grid to destination layout
    tiles = tiles.reshape(-1, tile_h, tile_w).reshape(dst_tr, dst_tc, tile_h, tile_w)
    # (dst_tr, dst_tc, tile_h, tile_w) -> (dst_H, dst_W)
    return tiles.permute(0, 2, 1, 3).reshape(dst_h, dst_w)
