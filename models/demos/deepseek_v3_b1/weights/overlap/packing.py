# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Overlap packing — fuse multiple tensors into a single L1 buffer with per-core shards."""

from dataclasses import dataclass
from functools import reduce

import numpy as np
import torch

import ttnn
from models.demos.deepseek_v3_b1.weights.overlap.spec import OverlappedTensorSpec, max_shard_bytes


def tilize_and_pack(data_2d: torch.Tensor, spec: OverlappedTensorSpec) -> bytes:
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
class OverlapEntry:
    """A named tensor paired with its overlap spec for use in ``overlap_tensors``."""

    name: str
    tensor: torch.Tensor
    spec: OverlappedTensorSpec


OverlappedTensor = ttnn.OverlappedTensor


def overlap_tensors(
    tensors: list[list[OverlapEntry]],
    device: ttnn.Device,
    move_to_device: bool = True,
) -> dict[str, OverlappedTensor]:
    """Overlap a list of tensors into a single fused tensor.

    On tenstorrent devices, all L1 allocations are lockstep. This means that
    if tensor is used only on subset of cores, the memory for it is still
    allocated on all cores. To avoid wasting memory, this function allows to
    overlap multiple tensors into a single fused tensor.
    When the fused tensor is sharded over cores, each core gets a
    shard of the fused tensor, corresponding to the desired original tensor.

    The fused tensor is always stored as WIDTH_SHARDED on the device
    (one flat shard per core).  Individual sub-tensors within the fused
    buffer can be either WIDTH_SHARDED or HEIGHT_SHARDED — the
    ``sharding`` field on each ``OverlappedTensorSpec`` controls how the
    per-device tensor is sliced across cores before tilization.

    Entries within a lane may have different ``core_range_set`` values.
    Each entry's data is written only to cores in its own core range;
    cores not covered by an entry have zeros at that entry's offset.
    Byte offsets are uniform (the same on every core in the lane),
    computed as the cumulative sum of ``shard_bytes`` across all entries
    in the lane.

    Args:
        tensors: A list of "lanes".  Each lane is a list of
            ``OverlapEntry`` instances packed back-to-back within each
            core's shard.  Entries within a lane may have different
            core ranges; the lane's core set is the union of all its
            entries' core ranges.  Lanes must occupy disjoint core
            ranges (across the union of their entries).
        device: The mesh device to place the fused tensor on.
        move_to_device: If True (default), place the result on device.

    Returns:
        A dict of ``OverlappedTensor`` views, keyed by tensor name.
    """

    def _core_list(crs: ttnn.CoreRangeSet) -> list[tuple[int, int]]:
        """Ordered list of cores from a CoreRangeSet."""
        cores = []
        for cr in crs.ranges():
            for y in range(cr.start.y, cr.end.y + 1):
                for x in range(cr.start.x, cr.end.x + 1):
                    cores.append((x, y))
        return cores

    for lane in tensors:
        assert len(lane) > 0, "Lane must contain at least one tensor"
        for entry in lane:
            assert (
                tuple(entry.tensor.shape) == entry.spec.raw_tensor_shape
            ), f"Tensor shape {tuple(entry.tensor.shape)} does not match spec shape {entry.spec.raw_tensor_shape}"
            assert len(entry.spec.tp_dim) == 2 and all(
                d is None or d in (0, 1) for d in entry.spec.tp_dim
            ), "tp_dim must be a 2-tuple of None, 0, or 1"
            assert entry.spec.sharding in (
                ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ), f"sharding must be WIDTH_SHARDED or HEIGHT_SHARDED, got {entry.spec.sharding}"

    # Build merged core list per lane (union of all entries' core ranges).
    lane_cores: list[list[tuple[int, int]]] = []
    for lane in tensors:
        merged = reduce(lambda a, b: a.merge(b), (e.spec.core_range_set for e in lane)).merge_ranges()
        lane_cores.append(_core_list(merged))

    for i in range(len(lane_cores)):
        for j in range(i + 1, len(lane_cores)):
            assert not set(lane_cores[i]) & set(lane_cores[j]), "Lanes must have separate core range sets"

    mesh_shape = (device.shape[0], device.shape[1])
    mesh_rows, mesh_cols = mesh_shape
    num_devices = mesh_rows * mesh_cols

    needed_shard_bytes = max_shard_bytes([[entry.spec for entry in lane] for lane in tensors], mesh_shape)
    assert needed_shard_bytes % 4 == 0, "shard bytes must be UINT32-aligned"
    uint32_per_shard = needed_shard_bytes // 4

    # Compute uniform byte offsets: cumulative sum of shard_bytes across
    # all entries in each lane, regardless of per-entry core ranges.
    byte_offsets: dict[tuple[int, int], int] = {}
    for lane_idx, lane in enumerate(tensors):
        running = 0
        for spec_idx, entry in enumerate(lane):
            byte_offsets[(lane_idx, spec_idx)] = running
            running += entry.spec.shard_bytes(mesh_shape)

    # For each entry, build a set of its cores and a mapping from
    # core coord -> core_idx (position within the entry's own core range).
    entry_core_indices: dict[tuple[int, int, int], dict[tuple[int, int], int]] = {}
    entry_core_sets: dict[tuple[int, int], set[tuple[int, int]]] = {}
    for lane_idx, lane in enumerate(tensors):
        for spec_idx, entry in enumerate(lane):
            cores = _core_list(entry.spec.core_range_set)
            entry_core_indices[(lane_idx, spec_idx)] = {c: i for i, c in enumerate(cores)}
            entry_core_sets[(lane_idx, spec_idx)] = set(cores)

    total_cores = sum(len(cores) for cores in lane_cores)

    per_device_raw: list[list[torch.Tensor]] = [[] for _ in range(mesh_rows)]
    for row in range(mesh_rows):
        for col in range(mesh_cols):
            dev_packed = bytearray()
            for lane_idx, lane in enumerate(tensors):
                for core in lane_cores[lane_idx]:
                    shard_data = bytearray(needed_shard_bytes)
                    for spec_idx, entry in enumerate(lane):
                        key = (lane_idx, spec_idx)
                        if core not in entry_core_sets[key]:
                            continue
                        core_idx = entry_core_indices[key][core]
                        num_cores = entry.spec.core_range_set.num_cores()
                        h_idx = entry.spec._dim_slice_idx(0, row, col, mesh_shape)
                        w_idx = entry.spec._dim_slice_idx(1, row, col, mesh_shape)
                        per_dev_h = entry.spec.per_device_height(mesh_shape)
                        per_dev_w = entry.spec.per_device_width(mesh_shape)
                        device_slice = entry.tensor[
                            h_idx * per_dev_h : (h_idx + 1) * per_dev_h,
                            w_idx * per_dev_w : (w_idx + 1) * per_dev_w,
                        ]
                        if entry.spec.sharding == ttnn.TensorMemoryLayout.WIDTH_SHARDED:
                            shard_w = per_dev_w // num_cores
                            core_slice = device_slice[:, core_idx * shard_w : (core_idx + 1) * shard_w]
                        else:
                            shard_h = per_dev_h // num_cores
                            core_slice = device_slice[core_idx * shard_h : (core_idx + 1) * shard_h, :]
                        shard_raw = tilize_and_pack(core_slice.contiguous(), entry.spec)
                        assert len(shard_raw) == entry.spec.shard_bytes(mesh_shape)
                        offset = byte_offsets[key]
                        shard_data[offset : offset + len(shard_raw)] = shard_raw

                    dev_packed.extend(shard_data)

            per_device_raw[row].append(torch.frombuffer(bytes(dev_packed), dtype=torch.int32).clone())

    shard_elems = uint32_per_shard * total_cores
    if num_devices == 1:
        combined = per_device_raw[0][0].reshape(1, shard_elems)
    else:
        row_tensors = [torch.cat([t.reshape(1, -1) for t in row_list], dim=1) for row_list in per_device_raw]
        combined = torch.cat(row_tensors, dim=0)

    combined_crs = reduce(lambda a, b: a + b, lane_cores)
    combined_crs = ttnn.CoreRangeSet(
        [ttnn.CoreRange(ttnn.CoreCoord(x, y), ttnn.CoreCoord(x, y)) for x, y in combined_crs]
    ).merge_ranges()
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

    result = {}
    for lane_idx, lane in enumerate(tensors):
        for spec_idx, entry in enumerate(lane):
            ts = entry.spec.logical_tensor_shape or (
                entry.spec.per_device_height(mesh_shape),
                entry.spec.per_device_width(mesh_shape),
            )
            result[entry.name] = OverlappedTensor(
                fused_tensor=fused,
                tensor_shape=ts,
                shard_shape=entry.spec.shard_shape(mesh_shape),
                core_range_set=entry.spec.core_range_set,
                dtype=entry.spec.dtype,
                tile_shape=(entry.spec.tile_h, entry.spec.tile_w),
                byte_offset=byte_offsets[(lane_idx, spec_idx)],
                total_size=entry.spec.shard_bytes(mesh_shape),
            )

    return result


def tilize_and_pack_bfp8(data_2d: torch.Tensor, tile_h: int = 32, tile_w: int = 32) -> bytes:
    """Tilize a 2-D tensor and pack as BFP8_b raw bytes."""
    H, W = data_2d.shape
    data_np = data_2d.contiguous().float().numpy()
    return ttnn._ttnn.core.tilize_and_pack_bfp8_b(data_np, H, W, tile_h, tile_w)


def tilize_and_pack_bfloat16(data_2d: torch.Tensor, tile_h: int = 32, tile_w: int = 32) -> bytes:
    """Tilize a 2-D tensor and pack as bfloat16 (Float16_b) raw bytes."""
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
    )

    float_bits = face_ordered.view(np.uint32)
    bf16_bits = (float_bits >> 16).astype(np.uint16)
    return bf16_bits.tobytes()


def tilize_and_pack_bfp4(data_2d: torch.Tensor, tile_h: int = 32, tile_w: int = 32) -> bytes:
    """Tilize a 2-D tensor and pack as BFP4_b raw bytes."""
    H, W = data_2d.shape
    data_np = data_2d.contiguous().float().numpy()
    return ttnn._ttnn.core.tilize_and_pack_bfp4_b(data_np, H, W, tile_h, tile_w)


def pack_bfloat16_1x32(data: torch.Tensor) -> bytes:
    """Pack a 1-row tensor as raw bfloat16 bytes with 1x32 tile layout."""
    flat = data.contiguous().float().reshape(-1).numpy()
    float_bits = flat.view(np.uint32)
    bf16_bits = (float_bits >> 16).astype(np.uint16)
    return bf16_bits.tobytes()
