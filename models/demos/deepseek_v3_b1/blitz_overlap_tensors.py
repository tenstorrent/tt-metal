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
    """Describes one sub-tensor within a fused raw-byte buffer.

    Multiple ``OverlappedShardSpec`` instances share a single L1
    buffer whose per-core shard is zero-padded to a common maximum
    byte size.  Each spec carries the core range, logical tensor shape,
    dtype, and tile dimensions needed to pack its portion of the shard.

    Tile byte sizes for BFP formats are computed from ``tile_h`` and
    ``tile_w`` rather than stored as constants, so non-standard tile
    shapes (e.g. 1x32, 16x32) are handled automatically.

    Shape tuples follow (height, width) convention.

    ``sharding`` controls how cores partition the per-device tensor:

    - ``WIDTH_SHARDED``: each core gets a column slice (full height,
      partial width).  ``num_cores`` divides the width.
    - ``HEIGHT_SHARDED``: each core gets a row slice (partial height,
      full width).  ``num_cores`` divides the height.

    When data is preprocessed (shuffled / block-sharded) before
    overlapping, ``raw_tensor_shape`` reflects the physical layout
    while ``logical_tensor_shape`` preserves the original per-device
    shape for downstream consumers.
    """

    core_range_set: ttnn.CoreRangeSet
    raw_tensor_shape: tuple[int, int]
    dtype: ttnn.DataType
    sharding: ttnn.TensorMemoryLayout = ttnn.TensorMemoryLayout.WIDTH_SHARDED

    tile_h: int = 32
    tile_w: int = 32

    tp_dim: tuple[int | None, int | None] = (None, None)

    logical_tensor_shape: tuple[int, int] | None = None

    def _tile_bytes(self) -> int:
        num_elements = self.tile_h * self.tile_w
        if self.dtype in (ttnn.bfloat8_b, ttnn.bfloat4_b):
            num_blocks = num_elements // 16
            exponent_bytes = (num_blocks + 3) // 4 * 4
            mantissa_bytes = num_elements if self.dtype == ttnn.bfloat8_b else num_elements // 2
            return exponent_bytes + mantissa_bytes
        return num_elements * _DTYPE_ELEMENT_BYTES[self.dtype]

    def _dim_tp(self, tensor_dim: int, mesh_shape: tuple[int, int]) -> int:
        """TP factor for a single tensor dimension (0=height, 1=width)."""
        result = 1
        for mesh_dim, d in enumerate(self.tp_dim):
            if d == tensor_dim:
                result *= mesh_shape[mesh_dim]
        return result

    def _dim_slice_idx(self, tensor_dim: int, row: int, col: int, mesh_shape: tuple[int, int]) -> int:
        """Slice index for a single tensor dimension at mesh coordinate (row, col)."""
        idx = 0
        stride = 1
        mesh_coord = (row, col)
        for mesh_dim in reversed(range(2)):
            if self.tp_dim[mesh_dim] == tensor_dim:
                idx += mesh_coord[mesh_dim] * stride
                stride *= mesh_shape[mesh_dim]
        return idx

    def tp(self, mesh_shape: tuple[int, int]) -> int:
        """Total TP factor (product of all non-None mesh dimensions)."""
        return self._dim_tp(0, mesh_shape) * self._dim_tp(1, mesh_shape)

    def per_device_height(self, mesh_shape: tuple[int, int]) -> int:
        return self.raw_tensor_shape[0] // self._dim_tp(0, mesh_shape)

    def per_device_width(self, mesh_shape: tuple[int, int]) -> int:
        return self.raw_tensor_shape[1] // self._dim_tp(1, mesh_shape)

    def shard_shape(self, mesh_shape: tuple[int, int]) -> tuple[int, int]:
        pdh = self.per_device_height(mesh_shape)
        pdw = self.per_device_width(mesh_shape)
        num_cores = self.core_range_set.num_cores()
        if self.sharding == ttnn.TensorMemoryLayout.WIDTH_SHARDED:
            return (pdh, pdw // num_cores)
        else:
            return (pdh // num_cores, pdw)

    def tiles_per_shard(self, mesh_shape: tuple[int, int]) -> int:
        sh, sw = self.shard_shape(mesh_shape)
        return (sh // self.tile_h) * (sw // self.tile_w)

    def shard_bytes(self, mesh_shape: tuple[int, int]) -> int:
        return self.tiles_per_shard(mesh_shape) * self._tile_bytes()


def max_shard_bytes(shard_specs: list[list[OverlappedShardSpec]], mesh_shape: tuple[int, int]) -> int:
    return max(sum(spec.shard_bytes(mesh_shape) for spec in lane) for lane in shard_specs)


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
    """Overlap a list of tensors into a single fused tensor.

    The fused tensor is always stored as WIDTH_SHARDED on the device
    (one flat shard per core).  Individual sub-tensors within the fused
    buffer can be either WIDTH_SHARDED or HEIGHT_SHARDED — the
    ``sharding`` field on each ``OverlappedShardSpec`` controls how the
    per-device tensor is sliced across cores before tilization.

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
            assert len(spec.tp_dim) == 2 and all(
                d is None or d in (0, 1) for d in spec.tp_dim
            ), "tp_dim must be a 2-tuple of None, 0, or 1"
            assert spec.sharding in (
                ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ), f"sharding must be WIDTH_SHARDED or HEIGHT_SHARDED, got {spec.sharding}"

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

    mesh_shape = (device.shape[0], device.shape[1])
    mesh_rows, mesh_cols = mesh_shape
    num_devices = mesh_rows * mesh_cols

    needed_shard_bytes = max_shard_bytes([[spec for _, spec in lane] for lane in tensors], mesh_shape)
    assert needed_shard_bytes % 4 == 0, "shard bytes must be UINT32-aligned"
    uint32_per_shard = needed_shard_bytes // 4

    total_cores = sum(lane[0][1].core_range_set.num_cores() for lane in tensors)

    byte_offsets: dict[int, int] = {}

    per_device_raw: list[list[torch.Tensor]] = [[] for _ in range(mesh_rows)]
    for row in range(mesh_rows):
        for col in range(mesh_cols):
            dev_packed = bytearray()
            for lane in tensors:
                num_cores = lane[0][1].core_range_set.num_cores()
                for core_idx in range(num_cores):
                    shard_data = bytearray()
                    for tensor, spec in lane:
                        h_idx = spec._dim_slice_idx(0, row, col, mesh_shape)
                        w_idx = spec._dim_slice_idx(1, row, col, mesh_shape)
                        per_dev_h = spec.per_device_height(mesh_shape)
                        per_dev_w = spec.per_device_width(mesh_shape)
                        device_slice = tensor[
                            h_idx * per_dev_h : (h_idx + 1) * per_dev_h,
                            w_idx * per_dev_w : (w_idx + 1) * per_dev_w,
                        ]
                        if spec.sharding == ttnn.TensorMemoryLayout.WIDTH_SHARDED:
                            shard_w = per_dev_w // num_cores
                            core_slice = device_slice[:, core_idx * shard_w : (core_idx + 1) * shard_w]
                        else:
                            shard_h = per_dev_h // num_cores
                            core_slice = device_slice[core_idx * shard_h : (core_idx + 1) * shard_h, :]
                        shard_raw = tilize_and_pack(core_slice.contiguous(), spec)
                        assert len(shard_raw) == spec.shard_bytes(mesh_shape)
                        byte_offsets[id(spec)] = len(shard_data)
                        shard_data.extend(shard_raw)

                    if len(shard_data) < needed_shard_bytes:
                        shard_data.extend(b"\x00" * (needed_shard_bytes - len(shard_data)))
                    dev_packed.extend(shard_data)

            per_device_raw[row].append(torch.frombuffer(bytes(dev_packed), dtype=torch.int32).clone())

    shard_elems = uint32_per_shard * total_cores
    if num_devices == 1:
        combined = per_device_raw[0][0].reshape(1, shard_elems)
    else:
        row_tensors = [torch.cat([t.reshape(1, -1) for t in row_list], dim=1) for row_list in per_device_raw]
        combined = torch.cat(row_tensors, dim=0)

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

    if num_devices == 1:
        mesh_mapper = ttnn.ReplicateTensorToMesh(device)
    else:
        mesh_mapper = ttnn.ShardTensor2dMesh(device, mesh_shape=(mesh_rows, mesh_cols), dims=(0, 1))
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
        for tensor, spec in lane:
            ts = spec.logical_tensor_shape or (
                spec.per_device_height(mesh_shape),
                spec.per_device_width(mesh_shape),
            )
            result.append(
                OverlappedTensor(
                    fused_tensor=fused,
                    tensor_shape=ts,
                    shard_shape=spec.shard_shape(mesh_shape),
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
