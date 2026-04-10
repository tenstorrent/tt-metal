# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Overlap packing — fuse multiple tensors into a single L1 buffer with per-core shards."""

from dataclasses import dataclass

import numpy as np
import torch

import ttnn
from models.demos.deepseek_v3_b1.weights.overlap.spec import OverlappedTensorSpec, _core_list, _greedy_place


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
    entries: list[OverlapEntry],
    device: ttnn.Device,
    move_to_device: bool = True,
) -> dict[str, OverlappedTensor]:
    """Overlap a list of tensors into a single fused tensor.

    On Tenstorrent devices, all L1 allocations are lockstep.  If a
    tensor is used only on a subset of cores, memory for it is still
    reserved on every core.  This function packs multiple tensors into
    a single fused buffer to avoid that waste.

    The fused buffer is always ``WIDTH_SHARDED`` on the device (one
    flat shard per core).  Individual sub-tensors can be either
    ``WIDTH_SHARDED`` or ``HEIGHT_SHARDED`` — controlled by each
    ``OverlappedTensorSpec.sharding``.

    Placement is greedy: entries are ordered by ``overlap_priority``
    ascending (lower first); unset (``None``) comes after all explicit
    values; ties use larger per-core shard bytes first.  Each entry gets
    the earliest byte offset that does not overlap earlier entries on any
    shared core; disjoint core sets may reuse the same offset range.

    Args:
        entries: A flat list of ``OverlapEntry`` items.
        device: The mesh device to place the fused tensor on.
        move_to_device: If True (default), place the result on device.

    Returns:
        A dict of ``OverlappedTensor`` views, keyed by tensor name.
    """
    for entry in entries:
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

    mesh_shape = (device.shape[0], device.shape[1])
    mesh_rows, mesh_cols = mesh_shape
    num_devices = mesh_rows * mesh_cols

    specs = [e.spec for e in entries]
    offsets = _greedy_place(specs, mesh_shape)
    needed = max(off + s.shard_bytes(mesh_shape) for off, s in zip(offsets, specs))
    assert needed % 4 == 0, "shard bytes must be UINT32-aligned"
    uint32_per_shard = needed // 4

    entry_cores = [_core_list(e.spec.core_range_set) for e in entries]
    entry_core_sets = [set(c) for c in entry_cores]
    entry_core_idx = [{c: i for i, c in enumerate(cores)} for cores in entry_cores]

    all_cores_deduped: list[tuple[int, int]] = list(dict.fromkeys(c for cores in entry_cores for c in cores))
    total_cores = len(all_cores_deduped)

    per_device_raw: list[list[torch.Tensor]] = [[] for _ in range(mesh_rows)]
    for row in range(mesh_rows):
        for col in range(mesh_cols):
            dev_packed = bytearray()
            for core in all_cores_deduped:
                shard_data = bytearray(needed)
                for eidx, entry in enumerate(entries):
                    if core not in entry_core_sets[eidx]:
                        continue
                    cidx = entry_core_idx[eidx][core]
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
                        core_slice = device_slice[:, cidx * shard_w : (cidx + 1) * shard_w]
                    else:
                        shard_h = per_dev_h // num_cores
                        core_slice = device_slice[cidx * shard_h : (cidx + 1) * shard_h, :]
                    shard_raw = tilize_and_pack(core_slice.contiguous(), entry.spec)
                    assert len(shard_raw) == entry.spec.shard_bytes(mesh_shape)
                    off = offsets[eidx]
                    shard_data[off : off + len(shard_raw)] = shard_raw

                dev_packed.extend(shard_data)

            per_device_raw[row].append(torch.frombuffer(bytes(dev_packed), dtype=torch.int32).clone())

    shard_elems = uint32_per_shard * total_cores
    if num_devices == 1:
        combined = per_device_raw[0][0].reshape(1, shard_elems)
    else:
        row_tensors = [torch.cat([t.reshape(1, -1) for t in row_list], dim=1) for row_list in per_device_raw]
        combined = torch.cat(row_tensors, dim=0)

    combined_crs = ttnn.CoreRangeSet(
        [ttnn.CoreRange(ttnn.CoreCoord(x, y), ttnn.CoreCoord(x, y)) for x, y in all_cores_deduped]
    )
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
    for eidx, entry in enumerate(entries):
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
            byte_offset=offsets[eidx],
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
