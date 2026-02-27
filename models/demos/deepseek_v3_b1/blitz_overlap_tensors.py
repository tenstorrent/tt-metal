from dataclasses import dataclass

import numpy as np
import torch

import ttnn

_DTYPE_ELEMENT_BYTES = {
    ttnn.bfloat16: 2,
    ttnn.float32: 4,
    ttnn.uint16: 2,
    ttnn.uint32: 4,
    ttnn.int32: 4,
}


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

    tp: int = 1

    def _tile_bytes(self) -> int:
        num_elements = self.tile_h * self.tile_w
        if self.dtype in (ttnn.bfloat8_b, ttnn.bfloat4_b):
            num_blocks = num_elements // 16
            exponent_bytes = (num_blocks + 3) // 4 * 4
            mantissa_bytes = num_elements if self.dtype == ttnn.bfloat8_b else num_elements // 2
            return exponent_bytes + mantissa_bytes
        return num_elements * _DTYPE_ELEMENT_BYTES[self.dtype]

    @property
    def tiles_per_shard(self) -> int:
        shard_w = self.raw_tensor_shape[1] // self.core_range_set.num_cores()
        return (self.per_device_height // self.tile_h) * (shard_w // self.tile_w)

    @property
    def shard_bytes(self) -> int:
        return self.tiles_per_shard * self._tile_bytes()

    @property
    def total_bytes(self) -> int:
        return self.shard_bytes * self.core_range_set.num_cores()

    @property
    def per_device_height(self) -> int:
        return self.raw_tensor_shape[0] // self.tp


def max_shard_bytes(shard_specs: list[list[OverlappedShardSpec]]) -> int:
    return max(sum(spec.shard_bytes for spec in lane) for lane in shard_specs)


def tilize_and_pack(data_2d: torch.Tensor, spec: OverlappedShardSpec) -> bytes:
    match spec.dtype:
        case ttnn.bfloat8_b:
            return tilize_and_pack_bfp8(data_2d, spec.tile_h, spec.tile_w)
        case ttnn.bfloat4_b:
            return tilize_and_pack_bfp4(data_2d, spec.tile_h, spec.tile_w)
        case ttnn.bfloat16:
            if spec.tile_h == 1 and spec.tile_w == 32:
                return pack_bfloat16_1x32(data_2d)
            return tilize_and_pack_bfloat16(data_2d, spec.tile_h, spec.tile_w)
        case _:
            raise ValueError(f"Unsupported dtype: {spec.dtype}")


@dataclass
class OverlappedTensor:
    """A logical view of a sub-tensor within a fused (overlapped) device buffer.

    The fused tensor is a raw byte container whose own tensor properties
    (dtype, layout, shard spec) are generally meaningless for the individual
    sub-tensors.  This class carries the intended per-sub-tensor properties
    alongside a shared reference to the underlying fused buffer.
    """

    fused_tensor: ttnn.Tensor
    tensor_shape: tuple[int, int]
    shard_shape: tuple[int, int]
    core_range_set: ttnn.CoreRangeSet
    dtype: ttnn.DataType
    tile_shape: tuple[int, int]
    byte_offset: int = 0
    total_size: int = 0

    def get_tile(self) -> ttnn.Tile:
        return ttnn.Tile(self.tile_shape)


def overlap_tensors(
    tensors: list[list[tuple[torch.Tensor, OverlappedShardSpec]]],
    device: ttnn.Device,
    move_to_device: bool = True,
) -> list[OverlappedTensor]:
    """Overlap a list of tensors into a single fused WIDTH_SHARDED tensor.

    Args:
        tensors: A list of "lanes".  Each lane is a list of
            ``(torch.Tensor, OverlappedShardSpec)`` tuples that share
            the same core range set and are packed back-to-back within
            each core's shard.  Lanes must occupy disjoint core ranges.
        device: The mesh device to place the fused tensor on.
        move_to_device: If True (default), place the result on device.

    Returns:
        A flat list of ``OverlappedTensor`` views, one per
        ``(tensor, spec)`` pair across all lanes, in input order.
    """

    for lane in tensors:
        assert len(lane) > 0, "Lane must contain at least one tensor"
        for tensor, spec in lane:
            assert (
                tuple(tensor.shape) == spec.raw_tensor_shape
            ), f"Tensor shape {tuple(tensor.shape)} does not match spec shape {spec.raw_tensor_shape}"
            assert spec.core_range_set == lane[0][1].core_range_set, (
                f"Core range set {spec.core_range_set} does not match {lane[0][1].core_range_set}, "
                "all core range sets must be the same within a lane"
            )
            assert spec.tp > 0 and (spec.tp & (spec.tp - 1)) == 0, "TP must be a positive power of 2"

    def _core_set(crs: ttnn.CoreRangeSet) -> set[tuple[int, int]]:
        cores = set()
        for cr in crs.ranges():
            for x in range(cr.start.x, cr.end.x + 1):
                for y in range(cr.start.y, cr.end.y + 1):
                    cores.add((x, y))
        return cores

    for i, lane_a in enumerate(tensors):
        cores_a = _core_set(lane_a[0][1].core_range_set)
        for lane_b in tensors[i + 1 :]:
            cores_b = _core_set(lane_b[0][1].core_range_set)
            assert not cores_a & cores_b, "Lanes must have separate core range sets"

    needed_shard_bytes = max_shard_bytes([[spec for _, spec in lane] for lane in tensors])
    assert needed_shard_bytes % 4 == 0, "shard bytes must be UINT32-aligned"
    uint32_per_shard = needed_shard_bytes // 4

    max_tp = max(spec.tp for lane in tensors for _, spec in lane)
    total_cores = sum(lane[0][1].core_range_set.num_cores() for lane in tensors)

    byte_offsets: dict[int, int] = {}

    per_tp_raw = []
    for tp_idx in range(max_tp):
        tp_packed = bytearray()
        for lane in tensors:
            num_cores = lane[0][1].core_range_set.num_cores()
            for core_idx in range(num_cores):
                shard_data = bytearray()
                for tensor, spec in lane:
                    slice_idx = tp_idx * spec.tp // max_tp
                    per_dev_h = spec.per_device_height
                    device_slice = tensor[slice_idx * per_dev_h : (slice_idx + 1) * per_dev_h, :]
                    shard_w = spec.raw_tensor_shape[1] // num_cores
                    shard_col = device_slice[:, core_idx * shard_w : (core_idx + 1) * shard_w].contiguous()
                    shard_raw = tilize_and_pack(shard_col, spec)
                    assert len(shard_raw) == spec.shard_bytes
                    byte_offsets[id(spec)] = len(shard_data)
                    shard_data.extend(shard_raw)

                if len(shard_data) < needed_shard_bytes:
                    shard_data.extend(b"\x00" * (needed_shard_bytes - len(shard_data)))
                tp_packed.extend(shard_data)

        per_tp_raw.append(torch.frombuffer(bytes(tp_packed), dtype=torch.int32).clone())

    if max_tp == 1:
        combined = per_tp_raw[0].reshape(1, uint32_per_shard * total_cores)
    else:
        combined = torch.cat([t.reshape(1, -1) for t in per_tp_raw], dim=1)

    combined_crs_ranges = [cr for lane in tensors for cr in lane[0][1].core_range_set.ranges()]
    combined_crs = ttnn.CoreRangeSet(combined_crs_ranges)
    shard_spec = ttnn.ShardSpec(
        combined_crs,
        (1, uint32_per_shard),
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        shard_spec,
    )

    if max_tp == 1:
        mesh_mapper = ttnn.ReplicateTensorToMesh(device)
    else:
        mesh_shape = (device.shape[0], device.shape[1])
        mesh_mapper = ttnn.ShardTensor2dMesh(device, mesh_shape=mesh_shape, dims=(None, 1))
    device_for_torch = device if move_to_device else None

    fused = ttnn.from_torch(
        combined,
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device_for_torch,
        memory_config=mem_config,
        mesh_mapper=mesh_mapper,
    )

    result = []
    for lane in tensors:
        num_cores = lane[0][1].core_range_set.num_cores()
        for tensor, spec in lane:
            shard_w = spec.raw_tensor_shape[1] // num_cores
            result.append(
                OverlappedTensor(
                    fused_tensor=fused,
                    tensor_shape=(spec.per_device_height, spec.raw_tensor_shape[1]),
                    shard_shape=(spec.per_device_height, shard_w),
                    core_range_set=spec.core_range_set,
                    dtype=spec.dtype,
                    tile_shape=(spec.tile_h, spec.tile_w),
                    byte_offset=byte_offsets[id(spec)],
                    total_size=spec.shard_bytes,
                )
            )

    return result


def tilize_and_pack_bfp8(data_2d: torch.Tensor, tile_h: int = 32, tile_w: int = 32) -> bytes:
    """Tilize a 2-D tensor and pack as BFP8_b raw bytes.

    Delegates to the C++ ``pack_as_bfp8_tiles`` via nanobind.
    """
    H, W = data_2d.shape
    data_np = data_2d.contiguous().float().numpy()
    return ttnn._ttnn.core.tilize_and_pack_bfp8_b(data_np, H, W, tile_h, tile_w)


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

    float_bits = face_ordered.view(np.uint32)
    bf16_bits = (float_bits >> 16).astype(np.uint16)
    return bf16_bits.tobytes()


def tilize_and_pack_bfp4(data_2d: torch.Tensor, tile_h: int = 32, tile_w: int = 32) -> bytes:
    """Tilize a 2-D tensor and pack as BFP4_b raw bytes.

    Delegates to the C++ ``pack_as_bfp4_tiles`` via nanobind.
    """
    H, W = data_2d.shape
    data_np = data_2d.contiguous().float().numpy()
    return ttnn._ttnn.core.tilize_and_pack_bfp4_b(data_np, H, W, tile_h, tile_w)


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
