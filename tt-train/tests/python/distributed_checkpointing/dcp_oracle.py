# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""A numpy model of a checkpoint for the device-free planner tests.

``FakeCheckpoint.save`` runs the save planner over global device-layout arrays and keeps the resulting files in
memory; ``FakeCheckpoint.load`` runs the load planner for a destination world and executes its reads and fills with
numpy. The oracle for every case is plain array slicing, written independently of the planner: through the transforms
for a tensor's layout, and with ``np.split`` per mesh coordinate (``device_shards``) for what each device holds, both
for the bytes a save writes and for what a load must give each device. The planner's own label mapping
(``chunk_boxes``) is never used to cut either side, so a mapping that gave a device the wrong rows fails the round
trips instead of agreeing with itself.
"""

from __future__ import annotations

import json
import os
import random
import sys
import zlib
from itertools import product

import numpy as np

import ttml.init
from ttml.distributed_checkpointing._boxes import Box
from ttml.distributed_checkpointing._errors import TopologyError
from ttml.distributed_checkpointing._index import HEADER_NAME, FileRecord, Index, data_file_name
from ttml.distributed_checkpointing._planner import TensorInfo, create_load_plan, create_save_plan
from ttml.distributed_checkpointing._topology import Label, Placement, crc31, dist_coords

R = Placement.replicate
S = Placement.shard


def one_process_owner(key: str, group) -> object:
    """A single-process owner for the planner tests and ``FakeCheckpoint``: a hash of the key picks one of the group's
    chunks, so the writes of replicated keys spread over the devices. (A real save elects by bytes over every key.)"""
    ordered = sorted(group, key=lambda c: c.mesh_coord)
    return ordered[crc31(key) % len(ordered)]


def label_local_shape(label: Label, global_shape) -> tuple[int, ...]:
    """The per-device shape ``label`` gives a global array of ``global_shape``."""
    shape = []
    for d, n in enumerate(global_shape):
        count = label.shard_count(d)
        if n % count:
            raise TopologyError(f"dim {d} of extent {n} does not split evenly into {count} shards")
        shape.append(n // count)
    return tuple(shape)


def world(placements, dist_shape, mesh_shape=None) -> Label:
    """A label whose distribution coordinates are the row-major walk of ``mesh_shape`` (``dist_shape`` by default)."""
    return Label(tuple(placements), tuple(dist_shape), tuple(dist_coords(mesh_shape or dist_shape)))


def per_shard_fused(canonical: np.ndarray, sizes, shards: int, dim: int) -> np.ndarray:
    """The device-layout array a model with ``shards`` TP shards holds for a weight fused from ``sizes`` segments:
    shard ``j`` holds piece ``j`` of every segment, in segment order, the way ColumnParallelLinear lays it out. Built
    by slicing, independently of ``fused_segments``."""
    starts = np.cumsum((0,) + tuple(sizes[:-1]))
    pieces = []
    for j in range(shards):
        for g, size in enumerate(sizes):
            n = size // shards
            lo = starts[g] + j * n
            pieces.append(np.take(canonical, range(lo, lo + n), axis=dim))
    return np.concatenate(pieces, axis=dim)


def arange(shape) -> np.ndarray:
    return np.arange(int(np.prod(shape)), dtype=np.float32).reshape(shape)


def device_global(canonical: np.ndarray, transform, global_shape, pad_value: float = 0.0) -> np.ndarray:
    """The device-layout global array holding ``canonical`` under ``transform``; padding rows get ``pad_value``."""
    out = np.full(global_shape, pad_value, dtype=canonical.dtype)
    for can, dev in transform.from_canonical(Box.whole(canonical.shape)):
        out[dev.slices()] = canonical[can.slices()]
    return out


def canonical_of(device: np.ndarray, transform, canonical_shape) -> np.ndarray:
    """The canonical array ``device`` holds under ``transform``; an element no piece writes stays NaN, so a transform
    that misses one cannot pass a comparison."""
    out = np.full(canonical_shape, np.nan, dtype=device.dtype)
    for dev, can in transform.to_canonical(Box.whole(device.shape)):
        out[can.slices()] = device[dev.slices()]
    return out


def tp_split_of(label: Label, rank: int) -> tuple[int, ...]:
    """The TP split a save records for a rank-``rank`` tensor under ``label`` when none of its axes is FSDP's."""
    return tuple(label.shard_count(d) for d in range(rank))


def info(key, label, global_shape, transform, dtype="F32", role="param") -> TensorInfo:
    local_shape = label_local_shape(label, global_shape)
    return TensorInfo(
        key, role, dtype, "TILE", label, local_shape, transform, (key,), tp_split=tp_split_of(label, len(global_shape))
    )


def _devices(label: Label):
    """``(distribution coordinate, mesh coordinate)`` of every device of ``label``: the distribution shape walked in
    row-major order, paired with the label's mesh coordinates in turn."""
    return zip(product(*(range(n) for n in label.dist_shape)), label.mesh_coords)


def device_shards(global_array: np.ndarray, label: Label) -> dict[tuple, np.ndarray]:
    """What each device of ``label`` holds of ``global_array``, by mesh coordinate, cut with ``np.split`` as a mesh
    mapper lays a tensor out: every axis that shards a dim splits it evenly, and axes sharding the same dim nest in
    axis order, the first outermost."""
    out = {}
    for coord, mesh_coord in _devices(label):
        piece = global_array
        for axis, placement in enumerate(label.placements):
            if placement.kind == "shard":
                piece = np.split(piece, label.dist_shape[axis], axis=placement.dim)[coord[axis]]
        out[mesh_coord] = piece
    return out


class FakeCheckpoint:
    def __init__(self) -> None:
        self.index: Index | None = None
        self.files: dict[int, bytearray] = {}

    def save(self, tensors: dict[str, tuple[np.ndarray, Label, object]]) -> Index:
        """``tensors[key] = (device-layout global array, label, transform)``. Each chunk's bytes are what its device
        holds (``device_shards``), as a real save writes each device's own tensor."""
        infos = [info(key, label, arr.shape, transform) for key, (arr, label, transform) in tensors.items()]
        plan = create_save_plan(infos, lambda coord: data_file_name(0, 0, coord), one_process_owner)
        held = {key: device_shards(arr, label) for key, (arr, label, _) in tensors.items()}
        self.files = {}
        records = []
        crcs = {}
        for fplan in plan.files:
            buf = bytearray(fplan.data_bytes)
            for item in fplan.items:
                chunk = np.ascontiguousarray(held[item.key][item.chunk.mesh_coord])
                assert chunk.shape == item.chunk.box.sizes and chunk.nbytes == item.nbytes
                buf[item.data_offset : item.data_offset + item.nbytes] = chunk.tobytes()
                crcs[(item.key, fplan.index)] = zlib.crc32(chunk.tobytes())
            self.files[fplan.index] = buf
            records.append(FileRecord(fplan.path, len(fplan.header) + fplan.data_bytes, 0, 0, fplan.coord))
        self.index = Index(checkpoint_id="fake", world={"group_size": 1}, files=records, entries=plan.entries(crcs))
        return self.index

    def load(self, dests: dict[str, tuple[Label, object, tuple]]):
        """``dests[key] = (label, transform, global shape)``; returns the plan and ``{key: {group: shard array}}``."""
        infos = [info(key, label, global_shape, transform) for key, (label, transform, global_shape) in dests.items()]
        plan = create_load_plan(self.index, infos)
        shards: dict[str, dict[tuple, np.ndarray]] = {}
        for key, kplan in plan.keys.items():
            out = {group: np.full(box.sizes, np.nan, dtype=np.float32) for group, box in kplan.boxes.items()}
            for read in kplan.reads:
                record = kplan.entry.chunks[read.chunk]
                raw = self.files[record.file][record.data_offset : record.data_offset + record.nbytes]
                chunk = np.frombuffer(bytes(raw), dtype=np.float32).reshape(record.box.sizes)
                out[read.group][read.dst.slices()] = chunk[read.src.slices()]
            for fill in kplan.fills:
                out[fill.group][fill.dst.slices()] = 0.0
            for group, arr in out.items():
                assert not np.isnan(arr).any(), f"{key}: shard {group} not fully written"
            shards[key] = out
        return plan, shards


def expected_shards(expected_global: np.ndarray, label: Label) -> dict[tuple, np.ndarray]:
    """What every replica group of ``label`` should hold of ``expected_global``: ``device_shards``, keyed the way the
    planner keys a group (a device's distribution coordinates along the sharding axes). Replicas must agree."""
    held = device_shards(expected_global, label)
    out: dict[tuple, np.ndarray] = {}
    for coord, mesh_coord in _devices(label):
        group = tuple(coord[a] for a, p in enumerate(label.placements) if p.kind == "shard")
        if group in out:
            np.testing.assert_array_equal(out[group], held[mesh_coord], err_msg=f"replicas of group {group}")
        else:
            out[group] = held[mesh_coord]
    return out


def assert_shards_equal(actual: dict[tuple, np.ndarray], expected: dict[tuple, np.ndarray]) -> None:
    assert actual.keys() == expected.keys(), (sorted(actual), sorted(expected))
    for group in expected:
        np.testing.assert_array_equal(actual[group], expected[group], err_msg=f"group {group}")


def advance_generators() -> None:
    """Draw from every host generator once, so a state captured next is mid-stream: python's cached Gaussian, numpy's
    cached Gaussian and position and the half-used 32-bit word of ttml.init's PCG64 are then set, all of which a state
    captured right after seeding leaves at their defaults, where a decoder that dropped them would still pass."""
    random.gauss(0.0, 1.0)
    np.random.standard_normal()
    ttml.init._rng.integers(0, 2**32, dtype=np.uint32)
    torch = sys.modules.get("torch")
    if torch is not None:
        torch.randn(1)


def host_draws() -> tuple:
    """The next draws from python's, numpy's and torch's generators, and ttml.init's state. The Gaussians come first:
    each returns its generator's cached Gaussian (python's gauss_next, numpy's cached Gaussian, torch's cached normal),
    which uniform draws never read, so a restore that dropped one shows here."""
    torch = sys.modules.get("torch")
    return (
        random.gauss(0.0, 1.0),
        float(np.random.standard_normal()),
        torch.randn(3).tolist() if torch is not None else None,
        random.random(),
        float(np.random.random()),
        torch.rand(3).tolist() if torch is not None else None,
        ttml.init.get_rng_state(),  # what the next layer's initial weights are drawn from
    )


class Loader:
    """A stateful dataloader stand-in for the tests and the driver: its only state is how many batches it handed out."""

    def __init__(self) -> None:
        self.position = 0

    def advance(self) -> None:
        self.position += 1

    def get_state_dict(self) -> dict:
        return {"position": self.position}

    def set_state_dict(self, state: dict) -> None:
        self.position = int(state["position"])


def set_ttml_token(own: dict, index: int, value: str) -> None:
    """Replace token ``index`` of the ttml generator's text in one rank's RNG entry (the last token is its position)."""
    tokens = own["ttml"].split()
    tokens[index] = value
    own["ttml"] = " ".join(tokens)


def shorten_hex(text: str, nbytes: int) -> str:
    return text[: -2 * nbytes]


# Damage to one rank's RNG entry, by test id; every case leaves valid JSON.
# The hex or text of a generator's state, the wrong length for it.
WRONG_LENGTH_RNG = {
    "numpy_623_words": lambda own: own["numpy"].update(state=shorten_hex(own["numpy"]["state"], 4)),
    "python_624_words": lambda own: own["python"].update(state=shorten_hex(own["python"]["state"], 4)),
    "torch_one_byte_short": lambda own: own.update(torch=shorten_hex(own["torch"], 1)),
    "ttml_600_tokens": lambda own: own.update(ttml=" ".join(own["ttml"].split()[:600])),
}
# The right length, but what a setter would misread: the ttml text the C++ stream stops partway through (non-ASCII
# digits, a word past 64 bits) or reads into a state no mt19937 has (a word past 32 bits, a position past 624), a cached
# Gaussian Python's setstate keeps but the next random.gauss cannot use, a numpy position past its 624 words (625: the
# first one numpy's set_state takes and our check must refuse), and a ttml.init state for another bit generator than
# the PCG64 it was captured from, with a state past 128 bits or a negative increment (which numpy would wrap silently),
# or a cached half-output past 32 bits (an OverflowError from numpy).
MISREAD_RNG = {
    "ttml_superscript_digit": lambda own: set_ttml_token(own, 5, "²"),
    "ttml_arabic_indic_digit": lambda own: set_ttml_token(own, 5, "٣"),
    "ttml_word_past_32_bits": lambda own: set_ttml_token(own, 5, str(2**32)),
    "ttml_word_past_64_bits": lambda own: set_ttml_token(own, 5, str(2**64)),
    "ttml_position_past_624": lambda own: set_ttml_token(own, -1, "625"),
    "python_gauss_next_text": lambda own: own["python"].update(gauss_next="corrupt"),
    "python_gauss_next_list": lambda own: own["python"].update(gauss_next=[0.5]),
    "python_gauss_next_bool": lambda own: own["python"].update(gauss_next=True),
    "numpy_position_625": lambda own: own["numpy"].update(pos=625),
    "numpy_negative_position": lambda own: own["numpy"].update(pos=-1),
    "ttml_init_not_pcg64": lambda own: own["ttml_init"].update(bit_generator="MT19937"),
    "ttml_init_state_past_128_bits": lambda own: own["ttml_init"].update(state=format(2**200, "x")),
    "ttml_init_negative_increment": lambda own: own["ttml_init"].update(inc="-1"),
    "ttml_init_uinteger_past_32_bits": lambda own: own["ttml_init"].update(uinteger=2**32),
}


def damage_stored_rng(ckpt_dir: str, damage, world_rank: int = 0) -> None:
    """Apply ``damage`` to world rank ``world_rank``'s RNG entry in the header.json of the checkpoint ``ckpt_dir``."""
    path = os.path.join(ckpt_dir, HEADER_NAME)
    with open(path) as f:
        header = json.load(f)
    damage(header["train_state"]["rng"]["ranks"][str(world_rank)])
    with open(path, "w") as f:
        json.dump(header, f)
