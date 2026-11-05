# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""A numpy model of a checkpoint for the device-free planner tests.

``FakeCheckpoint.save`` runs the save planner over global device-layout arrays and keeps the resulting files in
memory; ``FakeCheckpoint.load`` runs the load planner for a destination world and executes its reads and fills with
numpy. The oracle for every case is plain array slicing through the transforms, written independently of the planner.
"""

from __future__ import annotations

import numpy as np

from ttml.distributed_checkpointing._boxes import Box
from ttml.distributed_checkpointing._index import FileRecord, Index, data_file_name
from ttml.distributed_checkpointing._planner import TensorInfo, create_load_plan, create_save_plan
from ttml.distributed_checkpointing._topology import Label, Placement, dist_coords, local_shape as label_local_shape

R = Placement.replicate
S = Placement.shard


def world(placements, dist_shape, mesh_shape=None) -> Label:
    """A label whose distribution coordinates are the row-major walk of ``mesh_shape`` (``dist_shape`` by default)."""
    return Label(tuple(placements), tuple(dist_shape), tuple(dist_coords(mesh_shape or dist_shape)))


def arange(shape) -> np.ndarray:
    return np.arange(int(np.prod(shape)), dtype=np.float32).reshape(shape)


def device_global(canonical: np.ndarray, transform, global_shape, pad_value: float = 0.0) -> np.ndarray:
    """The device-layout global array holding ``canonical`` under ``transform``; padding rows get ``pad_value``."""
    out = np.full(global_shape, pad_value, dtype=canonical.dtype)
    for can, dev in transform.from_canonical(Box.whole(canonical.shape)):
        out[dev.slices()] = canonical[can.slices()]
    return out


def canonical_of(device: np.ndarray, transform, canonical_shape) -> np.ndarray:
    out = np.zeros(canonical_shape, dtype=device.dtype)
    for dev, can in transform.to_canonical(Box.whole(device.shape)):
        out[can.slices()] = device[dev.slices()]
    return out


def info(key, label, global_shape, transform, dtype="F32", role="param") -> TensorInfo:
    return TensorInfo(key, role, dtype, "TILE", label, label_local_shape(label, global_shape), transform, (key,))


class FakeCheckpoint:
    def __init__(self) -> None:
        self.index: Index | None = None
        self.files: dict[int, bytearray] = {}
        self.save_plan = None

    def save(self, tensors: dict[str, tuple[np.ndarray, Label, object]]) -> Index:
        """``tensors[key] = (device-layout global array, label, transform)``."""
        infos = [info(key, label, arr.shape, transform) for key, (arr, label, transform) in tensors.items()]
        plan = create_save_plan(infos, lambda coord: data_file_name(0, 0, coord))
        self.save_plan = plan
        self.files = {}
        records = []
        for fplan in plan.files:
            buf = bytearray(fplan.data_bytes)
            for item in fplan.items:
                arr = tensors[item.key][0]
                chunk = np.ascontiguousarray(arr[item.chunk.box.slices()])
                assert chunk.nbytes == item.nbytes
                buf[item.data_offset : item.data_offset + item.nbytes] = chunk.tobytes()
            self.files[fplan.index] = buf
            records.append(FileRecord(fplan.path, len(fplan.header) + fplan.data_bytes, 0, 0, fplan.coord))
        self.index = Index(checkpoint_id="fake", world={"group_size": 1}, files=records, entries=plan.entries())
        return self.index

    def load(self, dests: dict[str, tuple[Label, object, tuple]], strict=True):
        """``dests[key] = (label, transform, global shape)``; returns the plan and ``{key: {group: shard array}}``."""
        infos = [info(key, label, global_shape, transform) for key, (label, transform, global_shape) in dests.items()]
        plan = create_load_plan(self.index, infos, strict=strict)
        shards: dict[str, dict[tuple, np.ndarray]] = {}
        for key, kplan in plan.keys.items():
            out = {group: np.full(box.sizes, np.nan, dtype=np.float32) for group, box in kplan.boxes.items()}
            for read in kplan.reads:
                record = kplan.entry.chunks[read.chunk]
                raw = self.files[record.file][record.data_offset : record.data_offset + record.nbytes]
                chunk = np.frombuffer(bytes(raw), dtype=np.float32).reshape(record.box.sizes)
                out[read.group][read.dst.slices()] = chunk[read.src.slices()]
            for fill in kplan.fills:
                assert fill.fill == "zeros"
                out[fill.group][fill.dst.slices()] = 0.0
            for group, arr in out.items():
                assert not np.isnan(arr).any(), f"{key}: shard {group} not fully written"
            shards[key] = out
        return plan, shards


def expected_shards(expected_global: np.ndarray, label: Label) -> dict[tuple, np.ndarray]:
    """What every replica group of ``label`` should hold of ``expected_global``."""
    from ttml.distributed_checkpointing._topology import chunk_boxes, replica_groups

    local = label_local_shape(label, expected_global.shape)
    return {
        group: expected_global[chunks[0].box.slices()]
        for group, chunks in replica_groups(chunk_boxes(label, local)).items()
    }


def assert_shards_equal(actual: dict[tuple, np.ndarray], expected: dict[tuple, np.ndarray]) -> None:
    assert actual.keys() == expected.keys(), (sorted(actual), sorted(expected))
    for group in expected:
        np.testing.assert_array_equal(actual[group], expected[group], err_msg=f"group {group}")
